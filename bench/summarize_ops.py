"""Summarize the one-time per-op log of the muse_topo vLLM plugin (MUSE_TOPO_LOG_OPS=1|2) from a vLLM log.

usage: summarize_ops.py LOG [--paths] [--ranks N]
Groups the ``muse_topo op[rank r]: ...`` lines by (op, dim, shape, dtype, path) and prints how many ranks logged each key,
whether it was seen eagerly and/or while capturing a CUDA graph, and the largest number of distinct streams of one
rank. ``--paths`` hides the topo lines and prints only the keys that fell back to NCCL (with the reason). Exit code 1
if a key was logged by fewer ranks than --ranks (default 8) or if the same (op, dim, shape, dtype) took different paths
on different ranks.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"muse_topo op\[rank (\d+)\]: (\w+)(?:\(dim=(-?\d+)\))? shape=\(([^)]*)\) (\w+) stream=(0x[0-9a-f]+) "
                  r"capturing=(True|False) -> (\w+)(?: \((.*)\))?$")


def main():
    path = sys.argv[1]
    only_fallback = "--paths" in sys.argv
    nranks = int(sys.argv[sys.argv.index("--ranks") + 1]) if "--ranks" in sys.argv else 8
    keys = defaultdict(lambda: {"ranks": set(), "cap": set(), "streams": set(), "reason": ""})
    for ln in open(path, errors="replace"):
        m = LINE.search(ln.rstrip())
        if not m:
            continue
        rank, op, dim, shape, dtype, stream, cap, kind, reason = m.groups()
        shape = tuple(int(v) for v in shape.replace(" ", "").split(",") if v)
        k = (op, dim, shape, dtype, kind)
        e = keys[k]
        e["ranks"].add(int(rank))
        e["cap"].add(cap)
        e["streams"].add((int(rank), stream))
        e["reason"] = reason or e["reason"]
    bad = 0
    by_sig = defaultdict(set)
    for (op, dim, shape, dtype, kind), e in keys.items():
        by_sig[(op, dim, shape, dtype)].add(kind)
    for sig, kinds in by_sig.items():
        if len(kinds) > 1:
            bad += 1
            print("INCONSISTENT paths across ranks/calls:", sig, sorted(kinds))
    print(f"{'op':14s} {'dim':>3s} {'shape':16s} {'dtype':9s} {'path':5s} {'ranks':>5s} {'capture':>8s} {'streams':>7s}")
    for (op, dim, shape, dtype, kind), e in sorted(keys.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][4])):
        if only_fallback and kind == "topo":
            continue
        if len(e["ranks"]) < nranks:
            bad += 1
        cap = "+".join(sorted("cap" if c == "True" else "eager" for c in e["cap"]))
        print(f"{op:14s} {dim or '-':>3s} {str(shape):16s} {dtype:9s} {kind:5s} {len(e['ranks']):>5d} {cap:>8s} "
              f"{max(sum(1 for r, _ in e['streams'] if r == q) for q in e['ranks']):>7d} {e['reason']}")
    topo = sum(1 for k in keys if k[4] == "topo")
    print(f"keys: {len(keys)} (topo {topo}, fallback {len(keys) - topo}); problems: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
