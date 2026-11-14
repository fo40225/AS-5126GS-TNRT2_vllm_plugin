"""Real-weight end-to-end dump of the FULL Muse Spark model (TP8 + EP8, 62 layers, NVFP4 checkpoint) for the topo-vs-NCCL
comparison: greedy continuations of natural-language prompts (single batch of 8, batch of 32 for a bigger decode graph) and
the teacher-forced negative log-likelihood of two passages. Same dump format as nccl/e2e_bitwise.py, so nccl/compare.py
works on it; the NLL lines are printed at the end of the run.

usage: runsrc_topo.sh e2e_real.py OUT.json            (VLLM_ENABLE_V1_MULTIPROCESSING=0 for bitwise A/B runs;
                                                        MUSE_TOPO_ENABLE=1 / 0 selects the collectives)
"""
import argparse
import json
import math
import os
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

SHORT = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain the theory of relativity in one sentence.",
    "1, 2, 3, 4, 5,",
    "Once upon a time in a distant land",
    "The quick brown fox jumps over",
    "import numpy as np\n",
    "Q: What is 17 times 23? A:",
]


def lp_dicts(lps):
    out = []
    for d in lps or []:
        out.append(None if d is None else {str(t): [v.logprob, v.rank] for t, v in d.items()})
    return out


def dump(outs, tag):
    rows = []
    for k, o in enumerate(outs):
        c = o.outputs[0]
        rows.append({"name": f"{tag}_{k}", "prompt_len": len(o.prompt_token_ids), "token_ids": list(c.token_ids),
                     "logprobs": lp_dicts(c.logprobs),
                     "prompt_logprobs": lp_dicts(o.prompt_logprobs) if o.prompt_logprobs else None,
                     "text": c.text})
    return rows


def nll(o):
    """Mean negative log-likelihood of the prompt tokens after the first (teacher forced)."""
    vals = []
    for tok, d in zip(o.prompt_token_ids, o.prompt_logprobs):
        if d is not None:
            vals.append(-d[tok].logprob)
    return sum(vals) / len(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=MODEL, quantization="modelopt_fp4", kv_cache_dtype="fp8", max_model_len=8192, tensor_parallel_size=8,
              enable_expert_parallel=True, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
              # 0.9 leaves no room for KV blocks next to the 69 GiB of weights per GPU ("No available memory for the
              # cache blocks"); 0.95 + max_num_seqs 128 is what the benchmark server runs with
              gpu_memory_utilization=float(os.environ.get("E2E_GPU_UTIL", "0.95")), max_num_seqs=128, seed=0,
              enable_prefix_caching=False)
    print(f"E2E_REAL load_time_s={time.time() - t0:.1f}", flush=True)

    def sp(n, **kw):
        return SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True, logprobs=5, **kw)

    rows = []
    rows += dump(llm.generate(SHORT, sp(48), use_tqdm=False), "short")
    words = PASSAGE.split()
    prompts = [" ".join(words[:60]), " ".join(words[:120]), PASSAGE]
    outs = llm.generate(prompts, [sp(32, prompt_logprobs=1) for _ in prompts], use_tqdm=False)
    rows += dump(outs, "passage")
    for k, o in enumerate(outs):
        m, n = nll(o)
        print(f"NLL passage_{k} tokens={n} nll_mean={m:.6f} ppl={math.exp(m):.4f}", flush=True)
    batch = [f"Question {i}: what is {3 + i} times {7 + 2 * i}? Answer:" for i in range(32)]
    rows += dump(llm.generate(batch, sp(48), use_tqdm=False), "batch32")
    rows += dump(llm.generate(SHORT, sp(48), use_tqdm=False), "shortR1")
    meta = {"env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "MUSE_TOPO"))}, "layers": 62}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "requests": rows}, f)
    print("E2E_REAL DONE", args.out, flush=True)
    for r in rows[:3] + rows[8:9]:
        print("SAMPLE", r["name"], repr(r["text"][:100]), flush=True)


if __name__ == "__main__":
    main()
