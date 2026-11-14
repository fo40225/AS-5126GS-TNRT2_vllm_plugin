#!/usr/bin/env python3
"""S1-10 gate: forced-preemption smoke of the KV CPU tier (stdlib only; run with python3 -I) against a RUNNING server whose GPU pool is far too small
for the load (launch it with a small --num-gpu-blocks-override and a short --max-model-len; the master does).

Why: the connector's preemption and re-admission path (SimpleCPUOffloadManager: on a preempted step the scheduler sets need_flush and the worker drains every
in-flight load and store; the eager-store cursors of the preempted request are reset) is not exercised by any workload whose GPU pool holds the traffic (the
S1-10 study and stress runs report 0 preemptions). This script overcommits the pool so that running requests ARE preempted and re-admitted, with the tier on.

Phases (all prompts are random token ids, greedy, logprobs=1):
  storm1  N distinct prompts of --prompt-len tokens sent at once (--max-tokens output each): the pool holds only a few of them, running requests are
          preempted when they cannot get blocks; every request must finish with exactly --max-tokens tokens and status 200;
  storm2  the same N prompts again at once (tier / GPU hits and preemptions mixed);
  drain   wait until the connector's pending stores are 0 (a leak of in-flight stores would never drain);
  check   for --check prompts of the storm: reset the GPU prefix cache (the tier is kept), send the prompt (a tier hit), send it again (a GPU hit of the
          same length): when the first send was a FULL tier hit (>= 98 % of the prompt from the external cache) the pair must be bitwise equal
          (tokens and logprob values): a block that a preemption flush corrupted or stored stale would show here.
Verdict line REUSE_PREEMPT: PASS needs health 200 afterwards, 0 failed requests, every request complete, preemptions > 0 (proof that the path ran:
without them the result is INCONCLUSIVE, not a pass), pending stores drained, >= --min-pairs comparable pairs, all equal.
usage: preempt_smoke.py --port 8100 --model M --out FILE.json [--n 32] [--prompt-len 2500] [--max-tokens 2000] [--check 8] [--min-pairs 4]
(AMENDMENT 1 of ADOPTION_RULE_FINAL, after the first smoke boot: 24 prompts of 12,000 tokens with 300 output tokens never preempted (admission-limited: the scheduler admits ~2 such prompts into the pool and the
decode growth is small) and three outputs ended early on an EOS token; now 32 prompts of 2,500 tokens with 600 output tokens and ignore_eos, so that the running set outgrows the pool during decode.
AMENDMENT 2 (second inconclusive boot: 600 output tokens still did not preempt, the free blocks left by the admission cover the decode growth): 2,000 output tokens and a pool of 14,000 blocks (spec, --max-model-len 8192)
or 4,000 blocks (ns256).)
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reuse_driver as rd  # noqa: E402


def storm(port, model, prompts, max_tokens, label):
    m0 = rd.metrics(port)
    recs = [None] * len(prompts)

    def one(i):
        r = rd.send(port, model, prompts[i], max_tokens=max_tokens, timeout=3600, ignore_eos=True)
        recs[i] = {"i": i, "status": r["status"], "ntok": len(r["toks"]), "ttft_s": r["ttft_s"], "total_s": round(r["total_s"], 2),
                   "error": r.get("error")}

    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.time()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.time() - t0
    d = rd.delta(m0, rd.metrics(port))
    pre = sum(v for k, v in d.items() if k.startswith("vllm:num_preemptions_total"))
    ext = sum(v for k, v in d.items() if k.startswith("vllm:external_prefix_cache_hits_total"))
    bad = [r for r in recs if r["status"] != 200 or r["ntok"] != max_tokens]
    s = {"label": label, "n": len(recs), "wall_s": round(wall, 1), "preemptions": pre, "external_hit_tokens": ext, "bad": len(bad),
         "bad_rows": bad[:5], "delta_offload": {k: v for k, v in d.items() if "offload" in k}}
    print("REUSE_PREEMPT_STORM %s" % json.dumps(s, sort_keys=True), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=2500)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--check", type=int, default=8)
    ap.add_argument("--min-pairs", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=4_400_000)
    a = ap.parse_args()
    prompts = [rd.gen_tokens(a.seed_base + i, a.prompt_len) for i in range(a.n)]
    info = {"args": vars(a)}
    rd.send(a.port, a.model, rd.gen_tokens(a.seed_base - 1, 512), max_tokens=4)
    s1 = storm(a.port, a.model, prompts, a.max_tokens, "storm1")
    s2 = storm(a.port, a.model, prompts, a.max_tokens, "storm2")
    drain = rd.wait_stores(a.port, quiet_s=8.0, max_s=900.0)
    m = rd.metrics(a.port)
    pending = sum(v for k, v in m.items() if rd.PENDING.search(k))
    print("REUSE_PREEMPT_DRAIN wait_s=%s pending_after=%s" % (drain, pending), flush=True)
    run = rd.Run(argparse.Namespace(port=a.port, model=a.model))
    pairs = []
    for i in range(min(a.check, a.n)):
        rd.wait_stores(a.port, quiet_s=4.0, max_s=300.0)
        tries = rd.reset_gpu_cache(a.port)
        r1 = run.timed_single(prompts[i], "pc%d_tier" % i, pas="check")
        h = sum(v for k, v in r1["metric_delta"].items() if k.startswith("vllm:external_prefix_cache_hits_total"))
        r2 = run.timed_single(prompts[i], "pc%d_gpu" % i, pas="check")
        full = h >= 0.98 * a.prompt_len
        eq = bool(full and r1["toks"] == r2["toks"] and r1["vals"] == r2["vals"] and r1["status"] == 200 and r2["status"] == 200)
        pairs.append({"i": i, "reset_tries": tries, "external_hit_tokens": h, "full_tier_hit": full, "equal": eq,
                      "ttft_tier": r1["ttft_s"], "ttft_gpu": r2["ttft_s"], "ntok": [len(r1["toks"]), len(r2["toks"])]})
    comparable = [p for p in pairs if p["full_tier_hit"]]
    equal = [p for p in comparable if p["equal"]]
    healthy = False
    try:
        c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        c.request("GET", "/health")
        healthy = c.getresponse().status == 200
        c.close()
    except Exception:  # noqa: BLE001
        healthy = False
    preempts = s1["preemptions"] + s2["preemptions"]
    failed = s1["bad"] + s2["bad"] + sum(1 for r in run.recs if r["status"] != 200)
    reasons = []
    if not healthy:
        reasons.append("health != 200")
    if failed:
        reasons.append("%d failed / incomplete requests" % failed)
    if pending:
        reasons.append("pending stores did not drain (%s)" % pending)
    if len(comparable) < a.min_pairs:
        reasons.append("only %d comparable tier-hit pairs (< %d)" % (len(comparable), a.min_pairs))
    if len(equal) != len(comparable):
        reasons.append("%d of %d comparable pairs differ" % (len(comparable) - len(equal), len(comparable)))
    if preempts <= 0:
        verdict = "INCONCLUSIVE" if not reasons else "FAIL"
        reasons.append("no preemption happened (the path did not run)")
    else:
        verdict = "PASS" if not reasons else "FAIL"
    info.update({"storm1": s1, "storm2": s2, "drain": {"wait_s": drain, "pending_after": pending}, "pairs": pairs, "preemptions": preempts,
                 "comparable": len(comparable), "equal": len(equal), "healthy": healthy, "failed": failed, "verdict": verdict, "reasons": reasons})
    json.dump(info, open(a.out, "w"), indent=1)
    print("REUSE_PREEMPT: %s preemptions=%s comparable_pairs=%d equal=%d failed=%d healthy=%s reasons=%s" % (
        verdict, preempts, len(comparable), len(equal), failed, healthy, "; ".join(reasons) or "-"), flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
