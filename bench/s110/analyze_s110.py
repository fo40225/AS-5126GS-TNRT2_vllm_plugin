#!/usr/bin/env python3
"""S1-10 result summary of one or more boots (stdlib only; run with python3 -I).

usage: analyze_s110.py [--results /raid/vllm-runtime/musetest/results] [--json OUT] tag [tag ...]
Per tag: status (READY load_s), GPU idle used MiB, KV pool tokens, connector facts from the serve log, per-node peaks, the decode
rows (mean output tok/s and TPOT per concurrency over the repetitions), the reuse passes (TTFT p50/p99, tok/s, hit counters)
and the single-request rows (A/B long prompts, smoke).
"""
import glob
import json
import os
import re
import sys


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def summarize(root, tag):
    d = os.path.join(root, tag)
    r = {"tag": tag}
    st = read(d + "/status.txt")
    m = re.findall(r"(READY|LOAD_FAIL|LOAD_TIMEOUT) load_s=(\d+)", st)
    r["state"] = m[-1][0] if m else "NOT_STARTED"
    r["load_s"] = int(m[-1][1]) if m else None
    r["done"] = "DONE unsloth" in st
    r["watchdog"] = "WATCHDOG" in st
    g = sorted(set(re.findall(r"^\d+, (\d+) MiB", read(d + "/gpu_mem_ready.txt"), re.M)))
    r["gpu_idle_used_mib"] = g
    log = read(d + "/serve.log")
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    r["pool_tokens"] = int(m.group(1).replace(",", "")) if m else None
    r["conn_lines"] = sorted(set(re.sub(r"^.*?\] ", "", l)[:230] for l in log.splitlines()
                                 if re.search(r"SimpleCPUOffload(Scheduler|Worker)|Created mmap|Unlinked mmap|Allocating %d|OffloadingConnector|CPUOffloadingSpec|host_register failed|layout", l)
                                 and "pid=" in l))[:14]
    r["node_peaks"] = read(d + "/node_peaks.txt").strip().splitlines()
    dec = {}
    for f in sorted(glob.glob(d + "/real_r*_c*.json")):
        mm = re.search(r"real_r(\d+)_c(\d+)\.json$", f)
        if not mm or ".spec" in f:
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        c = int(mm.group(2))
        dec.setdefault(c, []).append((j.get("output_throughput"), (j.get("tpot_tok_ms") or {}).get("mean") or j.get("mean_tpot_ms"),
                                      j.get("median_ttft_ms"), j.get("completed"), j.get("failed")))
    r["decode"] = {c: {"out_tps": [round(x[0], 1) for x in v], "tpot_ms": [round(x[1], 2) if x[1] else None for x in v],
                       "ttft_p50_ms": [round(x[2], 1) if x[2] else None for x in v], "failed": sum(x[4] or 0 for x in v)}
                   for c, v in sorted(dec.items())}
    for name in ("cycle", "chat", "smoke"):
        f = "%s/reuse_%s.json" % (d, name)
        if os.path.exists(f):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            r["reuse_" + name] = {"passes": {k: {kk: v.get(kk) for kk in ("n", "ok", "ttft_p50", "ttft_p90", "ttft_p99", "wall_s", "tok_per_s")}
                                             for k, v in j.get("passes", {}).items()},
                                  "hits": {k: {mk.replace("vllm:", ""): mv for mk, mv in v.get("metric_delta", {}).items()
                                               if "external_prefix" in mk or "prefix_cache" in mk or "kv_offload" in mk}
                                           for k, v in j.get("passes", {}).items()},
                                  "singles": [{"label": s["label"], "len": s["len"], "ttft_s": s["ttft_s"], "sha": s["text_sha"], "lp": s.get("lp_sha"),
                                               "ext_hits": {mk.replace("vllm:", ""): mv for mk, mv in s.get("metric_delta", {}).items()
                                                            if "external_prefix_cache_hits" in mk or "load_blocks" in mk}}
                                              for s in j.get("sends", []) if s.get("pass") == "single"],
                                  "info": {k: j.get(k) for k in ("cycle", "chat")}}
    return r


def main():
    a = sys.argv[1:]
    root = "/raid/vllm-runtime/musetest/results"
    out = None
    if "--results" in a:
        i = a.index("--results"); root = a[i + 1]; del a[i:i + 2]
    if "--json" in a:
        i = a.index("--json"); out = a[i + 1]; del a[i:i + 2]
    res = [summarize(root, t) for t in a]
    for r in res:
        print("== %(tag)s state=%(state)s load_s=%(load_s)s done=%(done)s watchdog=%(watchdog)s pool=%(pool_tokens)s gpu_idle_used=%(gpu_idle_used_mib)s" % r)
        for l in r["conn_lines"]:
            print("   LOG", l)
        for l in r["node_peaks"]:
            print("   NODE", l)
        for c, v in r["decode"].items():
            print("   DEC c%-3d out_tps=%s tpot=%s ttft_p50=%s failed=%d" % (c, v["out_tps"], v["tpot_ms"], v["ttft_p50_ms"], v["failed"]))
        for k in ("reuse_cycle", "reuse_chat", "reuse_smoke"):
            if k in r:
                print("   %s info=%s" % (k, json.dumps(r[k]["info"])))
                for p, v in r[k]["passes"].items():
                    print("   PASS %-10s %s" % (p, json.dumps(v)))
                for s in r[k]["singles"]:
                    print("   ONE  %-22s len=%-6d ttft=%-7s sha=%s lp=%s ext=%s" % (s["label"], s["len"], s["ttft_s"], s["sha"], s["lp"], json.dumps(s["ext_hits"])))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
