"""Teacher-forced end-to-end logprob dump (reduced-layer Muse Spark, dummy weights) to quantify the numeric difference of
the muse_topo collectives against the NCCL baseline without the tie-flipping of greedy decoding.

Prompt logprobs are computed on the prompt tokens, so two runs see the same token ids at every position; prompts of at
most 512 tokens are prefilled in one chunk, i.e. every all-reduce / all-gather of the forward takes the topo path when
MUSE_TOPO_ENABLE=1 (and NCCL when 0). A batch of short prompts exercises the small kernels, the long ones the big ones.

usage: runsrc_topo.sh e2e_prompt_lp.py OUT.json [--layers N]      (run once with MUSE_TOPO_ENABLE=1, once with 0)
       e2e_prompt_lp.py --compare REF.json OTHER.json              (CPU only; statistics of |d logprob|)
"""
import argparse
import glob
import json
import math
import os
import sys

MODEL = "unsloth/Muse-Spark-Open-816B-A42B-NVFP4"
LENS = [16, 64, 128, 256, 384, 512, 512, 512]


def ids_for(n, mult, off):
    return [(i * mult) % 150000 + off for i in range(n)]


def dump(path, layers):
    from vllm import LLM, SamplingParams

    snap = glob.glob(
        "/raid/.cache/huggingface/hub/models--unsloth--Muse-Spark-Open-816B-A42B-NVFP4/snapshots/*/config.json")[0]
    text = json.load(open(snap))["text_config"]
    n = layers
    over = {"num_hidden_layers": n, "layer_types": text["layer_types"][:n], "mlp_layer_types": text["mlp_layer_types"][:n],
            "layer_rope_theta": text["layer_rope_theta"][:n]}
    llm = LLM(model=MODEL, load_format="dummy", quantization="modelopt_fp4", kv_cache_dtype="fp8", max_model_len=8192,
              tensor_parallel_size=8, enable_expert_parallel=True, moe_backend="cutlass", hf_overrides={"text_config": over},
              limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0}, gpu_memory_utilization=0.5, seed=0,
              enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5, logprobs=5)
    rows = []
    # one request per call (a single prefill chunk each) and one batched call (several prompts per chunk)
    for k, L in enumerate(LENS):
        o = llm.generate([{"prompt_token_ids": ids_for(L, 7919 + 2 * k, 1000 + 37 * k)}], sp, use_tqdm=False)[0]
        rows.append(rowdict(f"single{k}_{L}", o))
    reqs = [{"prompt_token_ids": ids_for(L, 104729 + 2 * k, 3000 + 53 * k)} for k, L in enumerate([20, 60, 100, 140])]
    for k, o in enumerate(llm.generate(reqs, sp, use_tqdm=False)):
        rows.append(rowdict(f"batch{k}_{len(o.prompt_token_ids)}", o))
    # sustained decode at a large batch (CUDA graph sizes up to 128): the same 128 requests twice, ids must repeat
    reqs = [{"prompt_token_ids": ids_for(24 + k % 7, 7907 + 2 * k, 5000 + 11 * k)} for k in range(128)]
    sp2 = SamplingParams(temperature=0.0, max_tokens=24, min_tokens=24, ignore_eos=True)
    ids1 = [list(o.outputs[0].token_ids) for o in llm.generate(reqs, sp2, use_tqdm=False)]
    ids2 = [list(o.outputs[0].token_ids) for o in llm.generate(reqs, sp2, use_tqdm=False)]
    print(f"DECODE128 repeat identical: {ids1 == ids2} ({sum(a == b for a, b in zip(ids1, ids2))}/128 requests)", flush=True)
    env = {k: v for k, v in os.environ.items() if k.startswith(("MUSE_TOPO", "NCCL_"))}
    json.dump({"layers": n, "decode128": ids1, "decode128_repeat_identical": ids1 == ids2, "env": env,
               "requests": rows}, open(path, "w"))
    print("E2E_LP DONE", path, flush=True)


def rowdict(name, o):
    pl = []
    for tok, d in zip(o.prompt_token_ids, o.prompt_logprobs or []):
        pl.append(None if d is None else {str(t): v.logprob for t, v in d.items()})
    ol = [{str(t): v.logprob for t, v in d.items()} for d in (o.outputs[0].logprobs or [])]
    return {"name": name, "plen": len(o.prompt_token_ids), "prompt_logprobs": pl, "first_token": list(o.outputs[0].token_ids),
            "out_logprobs": ol}


def compare(a_path, b_path):
    a = {r["name"]: r for r in json.load(open(a_path))["requests"]}
    b = {r["name"]: r for r in json.load(open(b_path))["requests"]}
    allv, same_tok, n_tok = [], 0, 0
    print(f"{'request':14s} {'positions':>9s} {'values':>7s} {'bitwise':>8s} {'mean|d|':>10s} {'max|d|':>10s} "
          f"{'argmax same':>12s}")
    for name, ra in a.items():
        rb = b[name]
        vals, ident = [], 0
        for da, db in zip(ra["prompt_logprobs"], rb["prompt_logprobs"]):
            if da is None:
                continue
            for t in da.keys() & db.keys():
                vals.append(abs(da[t] - db[t]))
                ident += da[t] == db[t]
        am = sum(1 for da, db in zip(ra["prompt_logprobs"], rb["prompt_logprobs"])
                 if da is not None and max(da, key=da.get) == max(db, key=db.get))
        npos = sum(1 for da in ra["prompt_logprobs"] if da is not None)
        n_tok += npos
        same_tok += am
        allv += vals
        mean = sum(vals) / max(len(vals), 1)
        print(f"{name:14s} {npos:9d} {len(vals):7d} {ident / max(len(vals), 1):8.3f} {mean:10.3e} {max(vals, default=0):10.3e} "
              f"{am:5d}/{npos:<5d}")
    da, db = json.load(open(a_path)).get("decode128"), json.load(open(b_path)).get("decode128")
    if da and db:
        print(f"decode128: {sum(x == y for x, y in zip(da, db))}/128 requests give identical 24 greedy tokens "
              f"(first-token identical: {sum(x[0] == y[0] for x, y in zip(da, db))}/128)")
    allv.sort()
    q = lambda p: allv[min(len(allv) - 1, int(p * len(allv)))] if allv else math.nan  # noqa: E731
    print(f"TOTAL values={len(allv)} mean|d|={sum(allv) / max(len(allv), 1):.3e} median={q(0.5):.3e} p99={q(0.99):.3e} "
          f"max={allv[-1] if allv else 0:.3e}; top-1 token (of the top-5 set) identical at {same_tok}/{n_tok} positions")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=6)
    args = ap.parse_args()
    dump(args.out, args.layers)
