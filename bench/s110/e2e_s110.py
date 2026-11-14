"""S1-10 gate: real-weight end-to-end determinism of the KV CPU offload against offload off, WITH PREFIX REUSE (offline LLM, TP8 + EP8, one process).
Run through the host runner under the GPU lock: env SRC=<tree> HOST_DIR=<host> GPU=0,..,7 VLLM_ENABLE_V1_MULTIPROCESSING=0 [E2E_*] runsrc_topo.sh e2e_s110.py OUT.json
(the master of the gate, master_det.sh, does it). Successor of the S1-9 e2e_l11.py: the same 85-prompt set (9 .. 23,000 tokens, 64 short ones for wide verify batches,
several chunked), now with PREFIX CACHING ON in both arms and three passes per sampling mode:
  C  cold      the GPU prefix cache and the connector tier are emptied first (reset_prefix_cache(reset_connector=True)): every prompt is computed
  H  reuse     the same prompts again: GPU prefix-cache hits (both arms)
  X  tier hit  the GPU pool only is reset (reset_prefix_cache()) and the prompts are sent again: with the tier on every prompt is a CONNECTOR hit (CPU -> GPU
               loads), with offload off a recompute
per request num_cached_tokens is recorded (the hit length): the comparer only equates passes with the same hit length (greedy outputs depend on the cached length
and the chunk shapes; a recompute is a different computation from a hit). Sampling modes: greedy, seeded t1k64 (temperature 1, top_k 64, per-request seed), mixed
(request i % 4: seeded t1k64, seeded top_p 0.95, seeded frequency_penalty 0.2, greedy); plus `lp`: the 8 mid-length prompts with logprobs=5 (value-level comparison;
no prompt_logprobs: a chunked prompt_logprobs request needs more memory than the exact-bytes tier leaves). Token ids (and the top-5 logprob values of `lp`) are dumped.
env: E2E_ARM=off|on  E2E_KV_BYTES  E2E_PER_RANK_BYTES (tier, arm on)  E2E_SPEC=1|0 (RadixArk DFlash2 K=7 / none)  E2E_MNS (128)  E2E_MNBT (8199)  E2E_CAP (256)
     E2E_GEN_TOKENS (128)  E2E_MAXLEN (32768)  E2E_NUMA_BIND=1|0 (arm on, default 1)  E2E_DRAFTER  E2E_GPU_UTIL (0.95)
Output files: OUT.<mode>_<C|H|X>.json (and OUT.lp_<C|H|X>.json) with {"meta":..., "requests":[{"name","prompt_len","token_ids","logprobs","num_cached_tokens"}]}.
"""
import argparse
import json
import os
import random
import time

MODEL = "unsloth/Muse-Spark-Open-816B-A42B-NVFP4"
PASSAGE = (
    "The history of the lighthouse is a history of human attempts to make the sea less dangerous. The earliest beacons "
    "were simple fires lit on hilltops by fishermen who wanted their neighbours to find the harbour after dark. As trade "
    "grew, the fires moved to towers, and the towers grew taller, until the Pharos of Alexandria rose above the "
    "Mediterranean and became one of the wonders of the ancient world. For many centuries afterwards the technology "
    "hardly changed: a coal or wood fire burned in an iron basket, and a keeper carried fuel up a spiral stair through "
    "the night. The invention of the Argand oil lamp in the late eighteenth century made the flame brighter and steadier, "
    "but it was Augustin Fresnel who transformed the lighthouse in 1822 with a lens built from concentric rings of glass "
    "prisms. The Fresnel lens collected almost all of the light of a small lamp and bent it into a narrow horizontal "
    "beam, which could be seen from more than twenty miles away. Rotating the lens on a clockwork carriage turned the "
    "beam into a regular flash, and every lighthouse could be given its own signature of long and short flashes, so that "
    "a navigator holding a stopwatch could tell which light he was looking at. Engineers soon discovered that the "
    "hardest problems were not optical at all. A tower built on a rock in open water must survive waves that strike with "
    "the force of several tons per square metre, and the builders had to cut each stone to interlock with its "
    "neighbours, like the teeth of a puzzle, so that the whole structure behaved as one piece. Smeaton's tower at "
    "Eddystone, finished in 1759, took its shape from the trunk of an oak tree, wide at the base and curving inwards as "
    "it rose, and it stood for more than a century. Later towers were assembled on land, taken apart, numbered and "
    "reassembled on the reef during the few calm weeks of summer. Today most lighthouses are automatic, powered by "
    "solar panels and monitored from distant offices, and satellite navigation has taken over much of their work. Yet "
    "mariners still value them as a reliable check that needs no electronics, and many communities have kept their "
    "lights burning as landmarks, museums and symbols of the long partnership between people and the sea."
)
# (prompt length in tokens, number of requests); the request order below interleaves them (the set of the S1-9 e2e scripts).
LENGTHS = [
    (9, 2), (90, 2), (600, 2), (2500, 2), (4100, 2), (4500, 2), (6000, 1), (8191, 1), (8193, 1), (9100, 2),
    (12500, 2), (17000, 1), (23000, 1), (90, 64),
]
GEN_TOKENS = int(os.environ.get("E2E_GEN_TOKENS", "128"))
LP_PROMPTS = 8   # `lp` pass: the first prompts with 600 .. 9100 tokens


def lp_dicts(lps):
    out = []
    for d in lps or []:
        out.append(None if d is None else {str(t): [v.logprob, v.rank] for t, v in d.items()})
    return out


def dump(outs, tag, plen):
    rows = []
    for k, o in enumerate(outs):
        c = o.outputs[0]
        rows.append({"name": f"{tag}_{k}", "prompt_len": plen[k], "token_ids": list(c.token_ids), "logprobs": lp_dicts(c.logprobs),
                     "num_cached_tokens": o.num_cached_tokens})
    return rows


def build_prompts(tok):
    words = PASSAGE.split()
    base = tok.encode(" ".join(words) + " ", add_special_tokens=False)
    rng = random.Random(1234)
    reqs = []
    for n, count in LENGTHS:
        for _ in range(count):
            off = rng.randrange(len(base))
            ids = []
            while len(ids) < n:
                ids += base[off:] + base[:off]
                off = rng.randrange(len(base))
            reqs.append(ids[:n])
    order = list(range(len(reqs)))
    rng.shuffle(order)
    return [reqs[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    arm = os.environ.get("E2E_ARM", "off")
    t0 = time.time()
    kw = {}
    if os.environ.get("E2E_KV_BYTES"):
        kw["kv_cache_memory_bytes"] = int(os.environ["E2E_KV_BYTES"])
    if os.environ.get("E2E_SPEC", "1") == "1":
        drafter = os.environ.get("E2E_DRAFTER", "/raid/.cache/huggingface/hub/models--RadixArk--Muse-Spark-Open-DFlash2/snapshots/46c90f412549c472941a68382d91a99e5e6372a0")
        kw["speculative_config"] = {"method": "dflash", "model": drafter, "num_speculative_tokens": 7, "rejection_sample_method": "block", "draft_sample_method": "probabilistic"}
    if arm == "on":
        per_rank = int(os.environ["E2E_PER_RANK_BYTES"])
        kw["kv_transfer_config"] = KVTransferConfig(kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both",
                                                    kv_connector_extra_config={"cpu_bytes_to_use": per_rank * 8, "cpu_bytes_to_use_per_rank": per_rank})
        if os.environ.get("E2E_NUMA_BIND", "1") == "1":
            kw["numa_bind"] = True
    llm = LLM(model=MODEL, quantization="modelopt_fp4", kv_cache_dtype="fp8", max_model_len=int(os.environ.get("E2E_MAXLEN", "32768")), tensor_parallel_size=8,
              enable_expert_parallel=True, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
              gpu_memory_utilization=float(os.environ.get("E2E_GPU_UTIL", "0.95")), max_num_seqs=int(os.environ.get("E2E_MNS", "128")), seed=0,
              max_num_batched_tokens=int(os.environ.get("E2E_MNBT", "8199")), enable_prefix_caching=True,
              compilation_config={"max_cudagraph_capture_size": int(os.environ.get("E2E_CAP", "256"))}, **kw)
    print(f"E2E_S110 arm={arm} load_time_s={time.time() - t0:.1f} mns={os.environ.get('E2E_MNS', '128')} cap={os.environ.get('E2E_CAP', '256')} "
          f"spec={os.environ.get('E2E_SPEC', '1')} kv_bytes={os.environ.get('E2E_KV_BYTES', '')}", flush=True)

    prompts = build_prompts(llm.get_tokenizer())
    plen = [len(p) for p in prompts]
    n = len(prompts)
    print(f"E2E_S110 {n} prompts, lengths {plen}", flush=True)

    def sp(**kwargs):
        return SamplingParams(max_tokens=GEN_TOKENS, ignore_eos=True, **kwargs)

    greedy = [sp(temperature=0.0) for _ in range(n)]
    t1k64 = [sp(temperature=1.0, top_k=64, top_p=1.0, seed=1000 + i) for i in range(n)]
    mixed = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            mixed.append(sp(temperature=1.0, top_k=64, top_p=1.0, seed=4000 + i))
        elif kind == 1:
            mixed.append(sp(temperature=1.0, top_k=64, top_p=0.95, seed=4000 + i))
        elif kind == 2:
            mixed.append(sp(temperature=1.0, top_k=64, frequency_penalty=0.2, seed=4000 + i))
        else:
            mixed.append(sp(temperature=0.0))
    inputs = [{"prompt_token_ids": p} for p in prompts]
    lp_idx = [i for i in range(n) if 600 <= plen[i] <= 9100][:LP_PROMPTS]
    lp_inputs = [inputs[i] for i in lp_idx]
    lp_sps = [sp(temperature=0.0, logprobs=5) for _ in lp_idx]
    lp_plen = [plen[i] for i in lp_idx]
    base = args.out[:-5] if args.out.endswith(".json") else args.out
    meta = {"arm": arm, "env": {k: v for k, v in os.environ.items() if k.startswith(("E2E_", "NCCL_", "MUSE_", "VLLM_MUSE"))}, "layers": 62}

    def gpu_reset(tag):
        """reset the GPU prefix cache only (the tier is kept); a reset fails while blocks are still held by in-flight offload transfers: step the engine with a 1-token request and retry"""
        for i in range(40):
            if llm.reset_prefix_cache():
                return i
            llm.generate([{"prompt_token_ids": prompts[0][:16]}], SamplingParams(max_tokens=1), use_tqdm=False)
            time.sleep(0.5)
        raise RuntimeError(f"GPU prefix cache reset kept failing before {tag}")

    def full_reset(tag):
        for i in range(40):
            if llm.reset_prefix_cache(reset_connector=True):
                return i
            llm.generate([{"prompt_token_ids": prompts[0][:16]}], SamplingParams(max_tokens=1), use_tqdm=False)
            time.sleep(0.5)
        raise RuntimeError(f"full prefix cache reset kept failing before {tag}")

    def run(tag, ins, sps, pl):
        t1 = time.time()
        outs = llm.generate(ins, sps, use_tqdm=False)
        cached = sum(o.num_cached_tokens or 0 for o in outs)
        print(f"E2E_S110 pass {tag} took {time.time() - t1:.1f} s cached_tokens={cached} of {sum(pl)}", flush=True)
        with open(f"{base}.{tag}.json", "w") as f:
            json.dump({"meta": meta, "requests": dump(outs, "req", pl)}, f)

    for mode, ins, sps, pl in (("greedy", inputs, greedy, plen), ("t1k64", inputs, t1k64, plen), ("mixed", inputs, mixed, plen), ("lp", lp_inputs, lp_sps, lp_plen)):
        r = full_reset(mode + "_C")
        print(f"E2E_S110 full reset before {mode}_C (retries {r})", flush=True)
        run(mode + "_C", ins, sps, pl)
        run(mode + "_H", ins, sps, pl)
        r = gpu_reset(mode + "_X")
        print(f"E2E_S110 GPU-only reset before {mode}_X (retries {r})", flush=True)
        run(mode + "_X", ins, sps, pl)
    print("E2E_S110 DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
