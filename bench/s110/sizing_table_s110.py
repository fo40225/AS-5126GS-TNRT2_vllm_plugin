#!/usr/bin/env python3
"""Sizing table of the tier boots (S1-10 step 3; stdlib only): per boot the tier, READY seconds, the avail_safe (MemFree + file cache, no slab, no shmem) of the four GPU
nodes at READY and the minimum over the whole run (node sampler, 2 s), the reclaim / major-fault counter deltas of /proc/vmstat between the first sample and the end, and the smoke TTFT.
usage: sizing_table_s110.py TAG [TAG ...] [--results DIR]"""
import json
import re
import sys

root = "/raid/vllm-runtime/musetest/results"
a = sys.argv[1:]
if "--results" in a:
    i = a.index("--results"); root = a[i + 1]; del a[i:i + 2]
for t in a:
    d = "%s/%s" % (root, t)
    st = open(d + "/status.txt", errors="replace").read() if True else ""
    m = re.findall(r"READY load_s=(\d+)", st)
    ready = int(m[-1]) if m else None
    cmd = open(d + "/serve_cmd.txt", errors="replace").read() if True else ""
    mt = re.search(r'cpu_bytes_to_use_per_rank\\?":(\d+)', cmd)
    tier = int(mt.group(1)) if mt else 0
    mn = {}
    first = last = None
    for line in open(d + "/node_mem_ts.txt"):
        p = line.split()
        if len(p) < 7:
            continue
        n = p[1]
        if n not in ("node0", "node3", "node4", "node7"):
            continue
        kv = dict(x.split("=") for x in p[2:])
        av = int(kv["free"]) + int(kv["file"]) - int(kv["shmem"])
        mn[n] = min(mn.get(n, 1 << 40), av)
    vs = [l.split() for l in open(d + "/vmstat_ts.txt")] if True else []
    def vals(l):
        return {x.split("=")[0]: int(x.split("=")[1]) for x in l[1:] if "=" in x}
    delta = {}
    if len(vs) > 2:
        v0, v1 = vals(vs[0]), vals(vs[-1])
        delta = {k: v1[k] - v0[k] for k in v0 if k in v1 and k in ("allocstall_normal", "allocstall_movable", "pgmajfault", "pgscan_direct", "pgsteal_direct", "numa_pages_migrated")}
    ready_av = {}
    try:
        for l in open(d + "/numa_ready.txt"):
            mm = re.match(r"(node[0347]) .*avail_safe=(\d+)", l)
            if mm:
                ready_av[mm.group(1)] = int(mm.group(2))
    except OSError:
        pass
    smoke = ""
    try:
        j = json.load(open(d + "/reuse_smoke.json"))
        smoke = " ".join("%s=%.3fs" % (s["label"].replace("smoke_", ""), s["ttft_s"]) for s in j["sends"] if "after_reset1" in s["label"] or "cold" in s["label"])
    except Exception:
        pass
    print("%s tier=%.3f GiB (%d B) ready_s=%s avail_safe@ready MiB %s min over run %s" % (t, tier / 2**30, tier, ready, ready_av, mn))
    print("    vmstat delta %s | smoke %s" % (delta, smoke))
