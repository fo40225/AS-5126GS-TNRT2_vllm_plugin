"""Turn bench_h1a.py sweep results into the dispatch tables of muse_topo/collectives.py.
usage: python3 make_table.py results.jsonl [more.jsonl]
Prints _AR_TABLE and _AG_TABLES literals: (largest nbytes covered, algorithm, grid, block, unroll). The winning
configuration of every measured size is applied to the bucket that ends at that size."""
import collections
import json
import sys

rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p) if l.strip()]

# ---- all_reduce: best (min b2b) configuration per message size, sentinel algorithms preferred unless flag is 3% faster
by_bytes = collections.defaultdict(list)
for r in rows:
    if r.get("op") == "all_reduce" and "topo_algo" in r:
        by_bytes[r["bytes"]].append(r)
print("_AR_TABLE = [")
sizes = sorted(by_bytes)
for i, nb in enumerate(sizes):
    cand = sorted(by_bytes[nb], key=lambda r: r["topo_b2b_us"])
    best = cand[0]
    sll = [r for r in cand if r["topo_algo"] in (1, 2)]
    if sll and best["topo_algo"] == 3 and sll[0]["topo_b2b_us"] <= 1.03 * best["topo_b2b_us"]:
        best = sll[0]
    c = best["topo_cfg"]
    limit = "1 << 62" if i == len(sizes) - 1 else str(nb)
    print(f"    ({limit}, {best['topo_algo']}, {c['grid']}, {c['block']}, {c['unroll']}),  # {nb} B: {best['topo_b2b_us']:.1f} us (NCCL {best['nccl_b2b_us']:.1f})")
print("]")

# ---- all_gather per layout
print("_AG_TABLES = {")
for layout, name in ((0, "dim-1"), (1, "dim0")):
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("op") == "all_gather" and "topo_algo" in r and r["layout"] == name:
            g[r["shard_bytes"]].append(r)
    print(f"    {layout}: [  # {name}")
    sizes = sorted(g)
    for i, nb in enumerate(sizes):
        best = min(g[nb], key=lambda r: r["topo_b2b_us"])
        c = best["topo_cfg"]
        limit = "1 << 62" if i == len(sizes) - 1 else str(nb)
        print(f"        ({limit}, {best['topo_algo']}, {c['grid']}, {c['block']}, {c['unroll']}),  # {nb} B: {best['topo_b2b_us']:.1f} us (NCCL {best['nccl_b2b_us']:.1f})")
    print("    ],")
print("}")
