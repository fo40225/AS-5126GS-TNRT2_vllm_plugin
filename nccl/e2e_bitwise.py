"""End-to-end bitwise output dump for reduced-layer Muse Spark (dummy weights).

Usage: e2e_bitwise.py OUT.json [--eager]
Dumps generated token ids, top-5 logprobs and (for the 3000-token prompt) prompt logprobs.
"""
import argparse
import glob
import json
import os
import time

from vllm import LLM, SamplingParams

MODEL = "unsloth/Muse-Spark-Open-816B-A42B-NVFP4"
LAYERS = 6


def lp_dicts(lps):
    # Per position: {token_id: [logprob, rank]}; Python float repr round-trips exactly.
    out = []
    for d in lps or []:
        if d is None:
            out.append(None)
        else:
            out.append({str(t): [v.logprob, v.rank] for t, v in d.items()})
    return out


def ids_for(n, mult, off):
    return [(i * mult) % 150000 + off for i in range(n)]


def dump(outs, tag):
    rows = []
    for k, o in enumerate(outs):
        c = o.outputs[0]
        rows.append({
            "name": f"{tag}_{k}",
            "prompt_len": len(o.prompt_token_ids),
            "token_ids": list(c.token_ids),
            "logprobs": lp_dicts(c.logprobs),
            "prompt_logprobs": lp_dicts(o.prompt_logprobs) if o.prompt_logprobs else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--repeat", type=int, default=0, help="extra in-process repeats of the short batch")
    args = ap.parse_args()

    snap = glob.glob(
        "/raid/.cache/huggingface/hub/models--unsloth--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/*/config.json"
    )[0]
    text = json.load(open(snap))["text_config"]
    n = LAYERS
    text_over = {
        "num_hidden_layers": n,
        "layer_types": text["layer_types"][:n],
        "mlp_layer_types": text["mlp_layer_types"][:n],
        "layer_rope_theta": text["layer_rope_theta"][:n],
    }

    t0 = time.time()
    llm = LLM(
        model=MODEL,
        load_format="dummy",
        quantization="modelopt_fp4",
        kv_cache_dtype="fp8",
        max_model_len=8192,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        moe_backend="cutlass",
        enforce_eager=args.eager,
        hf_overrides={"text_config": text_over},
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
        gpu_memory_utilization=0.5,
        seed=0,
    )
    print(f"E2E load_time_s={time.time() - t0:.1f}", flush=True)

    def sp(max_tokens, **kw):
        return SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens,
                              ignore_eos=True, logprobs=5, **kw)

    rows = []

    # 1) 8 short prompts x 64 tokens in one batch.
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Explain the theory of relativity in one sentence.",
        "1, 2, 3, 4, 5,",
        "Once upon a time in a distant land",
        "The quick brown fox jumps over",
        "import numpy as np\n",
        "Q: What is 17 times 23? A:",
    ]
    rows += dump(llm.generate(prompts, sp(64), use_tqdm=False), "short")

    for k in range(args.repeat):
        rows += dump(llm.generate(prompts, sp(64), use_tqdm=False), f"shortR{k + 1}")

    # 2) Three prompts of different lengths x 32 tokens; prompt logprobs only for the 3000-token one.
    lens = [17, 300, 3000]
    reqs = [{"prompt_token_ids": ids_for(L, 7919, 1000)} for L in lens]
    sps = [sp(32), sp(32), sp(32, prompt_logprobs=1)]
    rows += dump(llm.generate(reqs, sps, use_tqdm=False), "mixed")

    # 3) Single 6000-token prompt x 16 tokens (chunked prefill). Different ids avoid prefix-cache sharing.
    reqs = [{"prompt_token_ids": ids_for(6000, 104729, 2000)}]
    rows += dump(llm.generate(reqs, sp(16), use_tqdm=False), "long")

    meta = {
        "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        "eager": args.eager,
        "layers": n,
    }
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "requests": rows}, f)
    print("E2E DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
