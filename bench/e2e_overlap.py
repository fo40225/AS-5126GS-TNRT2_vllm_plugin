"""Real-weight end-to-end dump of the FULL Muse Spark model (TP8 + EP8) for the prefill-overlap A/B.

The microbatched prefill (VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP=1) splits every eager batch of at least 4096 tokens in two,
so the prompts here are long: several of them need more than one 8192-token chunk, a few are short (they share batches
with the long ones, and with the decodes of earlier requests, which makes mixed batches), and the longest ends in a chunk
of a different size than the others. Prefix caching is off. Greedy decoding; the dump holds the token ids, the top-5
logprobs of every generated token and the prompt logprobs of the first requests, in the format of nccl/e2e_bitwise.py, so
nccl/compare.py works on it. The same request set is generated twice in one process (pass 2 goes to OUT.r1.json) to show
run-to-run determinism.

usage: runsrc_topo.sh e2e_overlap.py OUT.json   (VLLM_ENABLE_V1_MULTIPROCESSING=0 for bitwise A/B runs)
"""
import argparse
import json
import math
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

# (prompt length in tokens, number of requests); the request order below interleaves them.
LENGTHS = [
    (9, 2), (90, 2), (600, 2), (2500, 2), (4100, 2), (4500, 2), (6000, 1), (8191, 1), (8193, 1), (9100, 2),
    (12500, 2), (17000, 1), (23000, 1),
]
PROMPT_LOGPROB_REQUESTS = 6  # the first requests also dump their prompt logprobs (bitwise comparison of the prefill)
GEN_TOKENS = 24


def lp_dicts(lps):
    out = []
    for d in lps or []:
        out.append(None if d is None else {str(t): [v.logprob, v.rank] for t, v in d.items()})
    return out


def dump(outs, tag, plen):
    rows = []
    for k, o in enumerate(outs):
        c = o.outputs[0]
        rows.append({"name": f"{tag}_{k}", "prompt_len": plen[k], "token_ids": list(c.token_ids),
                     "logprobs": lp_dicts(c.logprobs),
                     "prompt_logprobs": lp_dicts(o.prompt_logprobs) if o.prompt_logprobs else None,
                     "text": c.text})
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

    t0 = time.time()
    llm = LLM(model=MODEL, quantization="modelopt_fp4", kv_cache_dtype="fp8", max_model_len=32768, tensor_parallel_size=8,
              enable_expert_parallel=True, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
              gpu_memory_utilization=float(os.environ.get("E2E_GPU_UTIL", "0.95")), max_num_seqs=128, seed=0,
              max_num_batched_tokens=8192, enable_prefix_caching=False)
    print(f"E2E_OVERLAP load_time_s={time.time() - t0:.1f}", flush=True)

    prompts = build_prompts(llm.get_tokenizer())
    plen = [len(p) for p in prompts]
    print(f"E2E_OVERLAP {len(prompts)} prompts, lengths {plen}", flush=True)
    sps = [SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS, min_tokens=GEN_TOKENS, ignore_eos=True, logprobs=5,
                          prompt_logprobs=1 if i < PROMPT_LOGPROB_REQUESTS else None) for i in range(len(prompts))]
    inputs = [{"prompt_token_ids": p} for p in prompts]

    for tag, path in (("req", args.out), ("reqR1", args.out.replace(".json", ".r1.json"))):
        t1 = time.time()
        outs = llm.generate(inputs, sps, use_tqdm=False)
        print(f"E2E_OVERLAP pass {tag} took {time.time() - t1:.1f} s", flush=True)
        rows = dump(outs, "req", plen)
        meta = {"env": {k: v for k, v in os.environ.items()
                        if k.startswith(("NCCL_", "MUSE_TOPO", "VLLM_MUSE"))}, "layers": 62}
        with open(path, "w") as f:
            json.dump({"meta": meta, "requests": rows}, f)
        if tag == "req":
            for k, o in enumerate(outs[:PROMPT_LOGPROB_REQUESTS]):
                vals = [-d[t].logprob for t, d in zip(o.prompt_token_ids, o.prompt_logprobs) if d is not None]
                print(f"NLL req_{k} tokens={len(vals)} nll_mean={sum(vals) / len(vals):.6f} "
                      f"ppl={math.exp(sum(vals) / len(vals)):.4f}", flush=True)
    print("E2E_OVERLAP DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
