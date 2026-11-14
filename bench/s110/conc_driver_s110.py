# S1-10: copy of /raid/vllm-runtime/work/s1-9/L11/g/conc_driver_l7g.py (S1-9 L7 driver: --cycle gives every request its own cache_salt); imports acceptance.py from the harness directory.
"""S1-9 L7: the L5b driver plus the per-request acceptance histograms in the output JSON (key `requests`: k, n_out, hist of every ok request), for the request-level
resampling of the acceptance-length guard.
S1-9 L5b: the L3b driver plus the mode `toppg` (top_p 0.95 at temperature 1.5 on every request: OUT of the level-4 class, the full route) and the overrides --temperature / --top-k for
the modes toppall / toppg (the live both-routes audit sweeps the class corners).
S1-9 L3b: the L3 driver with the modes `toppall` (top_p 0.95 on every request), `minp` (min_p 0.05 on every request) and `mixpen`
(the default mode with frequency_penalty 0.1 on request idx % --top-p-every == 0: a request that needs the logits).
S1-9 L3 copy of the harness conc_driver.py (not edited in place), with four additions:
  * modes `default` (no sampling parameter at all: temperature, top_k and top_p come from the checkpoint's generation_config,
    which vLLM applies to every request that sets nothing) and `mixtp` (the default mode, but request idx % --top-p-every == 0
    carries top_p 0.95: one such request sends every verify step it is part of down the full path);
  * --cycle: prompts are cycled when --num-prompts exceeds the set size (steady state at c128), every use with its own
    cache_salt (--salt-prefix + index), so a repeat is no prefix-cache hit;
  * --top-p-every N / --top-p P for the mixtp mode.
Everything else (streaming chat completions, TPOT definition, result keys, CONC_ROW line) is the original.
Original docstring:

Fixed-concurrency throughput driver on REAL prompts (stdlib only; run with python3 -I).

`vllm bench serve --dataset-name custom` needs pandas, which the dev venv does not have (vllm[bench] is not installed and
nothing may be installed), and random-token prompts give an acceptance length near 1, so rows that must show the
speculative gain are measured with this driver instead: closed loop, `--concurrency` requests in flight, streaming chat
completions, one request per prompt (no repeats, so the prefix cache stays out of the numbers).

usage: conc_driver.py --set gsm80,mtbench --concurrency 16 --num-prompts 64 --max-tokens 128 --mode greedy
                      --out row.json [--port 8100] [--model REPO] [--effort xhigh] [--seed N]
Prompts of several sets are interleaved round-robin; the number of prompts is capped at what the sets contain.
The result JSON uses the keys of `vllm bench serve --save-result` that summarize_bench.py reads (completed, failed,
output_throughput, total_token_throughput, mean/median/p99 ttft and tpot, ...). TPOT = (e2e - ttft) / (out - 1) per request.
Per-request metrics.speculative_decoding (final usage chunk) is summed into spec_hist / spec_decode_acceptance_length.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/raid/vllm-runtime/work/sd/harness")
from acceptance import MODES as _MODES, add_hist, al_from_hist, load_set  # noqa: E402

MODES = dict(_MODES)
MODES["default"] = {}
MODES["mixtp"] = {}
MODES["toppall"] = {}
MODES["minp"] = {}
MODES["mixpen"] = {}
MODES["toppg"] = {}


def one_request(it, a, params, url):
    body = {"model": a.model, "messages": [{"role": "user", "content": it["turns"][0]}], "max_tokens": a.max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": a.effort, "current_date": a.date}, **params}
    if a.seed is not None:
        body["seed"] = a.seed + it["idx"]
    if a.cycle:
        body["cache_salt"] = "%s%d" % (a.salt_prefix, it["k"])
    if a.mode == "mixtp" and it["k"] % a.top_p_every == 0:
        body["top_p"] = a.top_p
    if a.mode in ("toppall", "toppg"):
        body["top_p"] = a.top_p
        if a.mode == "toppg":
            body["temperature"] = 1.5
        if a.temperature is not None:
            body["temperature"] = a.temperature
        if a.top_k is not None:
            body["top_k"] = a.top_k
    if a.mode == "minp":
        body["min_p"] = 0.05
    if a.mode == "mixpen" and it["k"] % a.top_p_every == 0:
        body["frequency_penalty"] = 0.1
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    usage, spec = None, None
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                d = json.loads(payload)
                if d.get("choices"):
                    delta = d["choices"][0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                if d.get("usage"):
                    usage = d["usage"]
                    spec = (d.get("metrics") or {}).get("speculative_decoding")
        if usage is None or t_first is None:
            return {"ok": False, "error": "no tokens or no usage chunk"}
        n_out = usage.get("completion_tokens", 0)
        return {"ok": True, "k": it["k"], "ttft": t_first - t0, "e2e": t_last - t0,
                "tpot": (t_last - t_first) / (n_out - 1) if n_out > 1 else None, "n_out": n_out,
                "n_in": usage.get("prompt_tokens", 0), "spec_hist": (spec or {}).get("acceptance_histogram")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="gsm80,mtbench")
    ap.add_argument("--mode", default="greedy", choices=sorted(MODES))
    ap.add_argument("--count", action="store_true", help="print the number of prompts in --set and exit")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--num-prompts", type=int)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="unsloth/Muse-Spark-Open-816B-A42B-NVFP4")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--date", default="2026-10-24")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--cycle", action="store_true")
    ap.add_argument("--salt-prefix", default="l3x")
    ap.add_argument("--top-p-every", type=int, default=16)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    a = ap.parse_args()

    pools = [load_set(s) for s in a.set.split(",")]
    if a.count:
        print(sum(len(p) for p in pools))
        return
    if not (a.concurrency and a.num_prompts and a.out):
        ap.error("--concurrency, --num-prompts and --out are required")
    items = []
    for k in range(max(len(p) for p in pools)):
        for p in pools:
            if k < len(p):
                items.append(p[k])
    if a.cycle:
        items = [dict(items[k % len(items)], k=k) for k in range(a.num_prompts)]
    else:
        items = [dict(it, k=k) for k, it in enumerate(items[:a.num_prompts])]
    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    params = dict(MODES[a.mode])
    results, lock = [], threading.Lock()
    t0 = time.perf_counter()

    def work(it):
        res = one_request(it, a, params, url)
        with lock:
            results.append(res)

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(work, items))
    dur = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    n_out = sum(r["n_out"] for r in ok)
    n_in = sum(r["n_in"] for r in ok)
    ttft = [r["ttft"] * 1000 for r in ok]
    tpot = [r["tpot"] * 1000 for r in ok if r["tpot"] is not None]
    hist = None
    for r in ok:
        if r.get("spec_hist"):
            hist = add_hist(hist, r["spec_hist"])
    res = {"completed": len(ok), "failed": len(results) - len(ok), "duration": dur, "total_input_tokens": n_in,
           "total_output_tokens": n_out, "request_throughput": len(ok) / dur, "output_throughput": n_out / dur,
           "total_token_throughput": (n_in + n_out) / dur, "max_concurrency": a.concurrency, "mode": a.mode,
           "sets": a.set, "max_tokens": a.max_tokens,
           "mean_ttft_ms": statistics.mean(ttft) if ttft else None, "median_ttft_ms": pct(ttft, 50), "p99_ttft_ms": pct(ttft, 99),
           "mean_tpot_ms": statistics.mean(tpot) if tpot else None, "median_tpot_ms": pct(tpot, 50), "p99_tpot_ms": pct(tpot, 99),
           "mean_e2el_ms": statistics.mean(r["e2e"] * 1000 for r in ok) if ok else None,
           "spec_hist": hist, "spec_decode_acceptance_length": al_from_hist(hist) if hist else None,
           "errors": [r["error"] for r in results if not r["ok"]][:5],
           "requests": sorted(({"k": r["k"], "n_out": r["n_out"], "hist": r.get("spec_hist")} for r in ok), key=lambda x: x["k"])}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("CONC_ROW c=%d n=%d completed=%d failed=%d out_tps=%.1f ttft_ms=%s tpot_ms=%s al_req=%s" % (
        a.concurrency, len(items), res["completed"], res["failed"], res["output_throughput"],
        "NA" if not ttft else f"{res['mean_ttft_ms']:.1f}", "NA" if not tpot else f"{res['mean_tpot_ms']:.2f}",
        "NA" if not hist else f"{res['spec_decode_acceptance_length']:.4f}"))
    sys.exit(1 if res["failed"] else 0)


if __name__ == "__main__":
    main()
