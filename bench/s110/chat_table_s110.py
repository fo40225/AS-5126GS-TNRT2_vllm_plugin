#!/usr/bin/env python3
"""Per-turn table of the chat scenario of several boots (S1-10; stdlib only).
usage: chat_table_s110.py TAG [TAG ...] [--results DIR]     prints turn, n, TTFT p50 / p99 (s), wall (s), prompt tok/s, hit tokens (local, external)"""
import json
import sys

root = "/raid/vllm-runtime/musetest/results"
a = sys.argv[1:]
if "--results" in a:
    i = a.index("--results"); root = a[i + 1]; del a[i:i + 2]
for t in a:
    try:
        j = json.load(open("%s/%s/reuse_chat.json" % (root, t)))
    except Exception as e:
        print("%s: no chat json (%r)" % (t, e))
        continue
    print("== %s  %s" % (t, json.dumps(j.get("chat"))))
    tot_w = tot_t = 0.0
    for k, v in j["passes"].items():
        md = v.get("metric_delta", {})
        loc = sum(x for m, x in md.items() if m.startswith("vllm:prefix_cache_hits"))
        ext = sum(x for m, x in md.items() if m.startswith("vllm:external_prefix_cache_hits"))
        print("  %-8s n=%3d ttft p50 %6.3f p99 %7.3f wall %7.1f s  %9.0f tok/s  hits local %9.0f ext %9.0f of %9d" % (
            k, v["n"], v["ttft_p50"], v["ttft_p99"], v["wall_s"], v["tok_per_s"], loc, ext, v["prompt_tokens"]))
        tot_w += v["wall_s"]; tot_t += v["prompt_tokens"]
    print("  total wall %.1f s for %d prompt tokens = %.0f tok/s" % (tot_w, tot_t, tot_t / tot_w))
