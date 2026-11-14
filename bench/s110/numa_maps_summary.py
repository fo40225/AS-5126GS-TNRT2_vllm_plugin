#!/usr/bin/env python3
"""Per-node resident page counts of /proc/<pid>/numa_maps, grouped by mapping category (stdlib only).

usage: numa_maps_summary.py <pid> [--json]
Categories: shm_offload (the unlinked /dev/shm/vllm_offload_* KV tier), shm_other, nvidia_dev, safetensors, lib_file,
other_file, anon, heap, stack. Values are MiB per NUMA node (pages x kernelpagesize). Also prints the memory policies
seen (e.g. bind:3, default) with the MiB they cover.
"""
import json
import re
import sys


def classify(path, tokens):
    if path:
        if "vllm_offload" in path:
            return "shm_offload"
        if path.startswith("/dev/shm") or "/dev/shm/" in path:
            return "shm_other"
        if path.startswith("/dev/nvidia") or path.startswith("/dev/dri"):
            return "nvidia_dev"
        if path.endswith(".safetensors"):
            return "safetensors"
        if ".so" in path:
            return "lib_file"
        return "other_file"
    if "heap" in tokens:
        return "heap"
    if "stack" in tokens:
        return "stack"
    return "anon"


def summarize(pid):
    cats = {}
    pols = {}
    with open("/proc/%s/numa_maps" % pid) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            policy = parts[1]
            path = None
            kv = {}
            flags = set()
            for t in parts[2:]:
                if t.startswith("file="):
                    path = t[5:]
                elif "=" in t:
                    k, v = t.split("=", 1)
                    kv[k] = v
                else:
                    flags.add(t)
            nodes = {k: int(v) for k, v in kv.items() if re.fullmatch(r"N\d+", k)}
            if not nodes:
                continue
            psz = int(kv.get("kernelpagesize_kB", "4"))
            cat = classify(path, flags)
            d = cats.setdefault(cat, {})
            for k, pages in nodes.items():
                d[k] = d.get(k, 0.0) + pages * psz / 1024.0
            tot = sum(nodes.values()) * psz / 1024.0
            pols[policy] = pols.get(policy, 0.0) + tot
    return {"pid": int(pid), "mib_by_category_node": {c: {k: round(v, 1) for k, v in sorted(d.items())}
                                                         for c, d in sorted(cats.items())},
            "mib_by_policy": {k: round(v, 1) for k, v in sorted(pols.items())}}


def main():
    pid = sys.argv[1]
    r = summarize(pid)
    if "--json" in sys.argv:
        print(json.dumps(r))
        return
    print("pid %s" % pid)
    for c, d in r["mib_by_category_node"].items():
        tot = sum(d.values())
        if tot < 1:
            continue
        print("  %-12s total %9.1f MiB  %s" % (c, tot, " ".join("%s=%.0f" % (k, v) for k, v in d.items() if v >= 1)))
    print("  policies: " + " ".join("%s=%.0f" % (k, v) for k, v in r["mib_by_policy"].items() if v >= 1))


if __name__ == "__main__":
    main()
