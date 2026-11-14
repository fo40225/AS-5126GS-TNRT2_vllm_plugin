#!/usr/bin/env python3
"""Diff of the live (active_allocated) blocks of two CUDA allocator snapshots (S1-10 probe; stdlib only).
usage: snap_diff.py A.pickle B.pickle [--min-mib 4] [--depth 5]
A block is keyed by the innermost frames of its allocation stack (vllm/ frames preferred); bytes are summed per key.
Prints the keys whose total differs by at least --min-mib, and the totals of both snapshots.
"""
import pickle
import sys


def live(path, depth):
    snap = pickle.load(open(path, "rb"))
    agg = {}
    total = 0
    for seg in snap["segments"]:
        for b in seg["blocks"]:
            if b["state"] != "active_allocated":
                continue
            total += b["size"]
            frames = (b.get("frames") or [])
            keep = [f for f in frames if "/vllm/" in f.get("filename", "") or "/src_" in f.get("filename", "")]
            keep = keep[:depth] if keep else frames[:depth]
            key = " < ".join("%s:%d %s" % (f["filename"].split("/vllm/")[-1][-60:], f["line"], f["name"]) for f in keep) or "(no frames)"
            agg[key] = agg.get(key, 0) + b["size"]
    return agg, total


def main():
    a, b = sys.argv[1], sys.argv[2]
    mn = 4.0
    depth = 5
    if "--min-mib" in sys.argv:
        mn = float(sys.argv[sys.argv.index("--min-mib") + 1])
    if "--depth" in sys.argv:
        depth = int(sys.argv[sys.argv.index("--depth") + 1])
    ga, ta = live(a, depth)
    gb, tb = live(b, depth)
    print("live total A %.1f MiB  B %.1f MiB  diff %+.1f MiB" % (ta / 2**20, tb / 2**20, (tb - ta) / 2**20))
    rows = []
    for k in set(ga) | set(gb):
        d = gb.get(k, 0) - ga.get(k, 0)
        if abs(d) >= mn * 2**20:
            rows.append((d, ga.get(k, 0), gb.get(k, 0), k))
    for d, x, y, k in sorted(rows, reverse=True):
        print("%+9.1f MiB  (A %8.1f  B %8.1f)  %s" % (d / 2**20, x / 2**20, y / 2**20, k))


if __name__ == "__main__":
    main()
