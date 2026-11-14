"""Turn bench_h1b.py results into the dispatch-table literals of muse_topo/collectives.py.
usage: python3 make_table_h1b.py results.jsonl [more.jsonl ...]
For each operation and measured size the fastest EXACT variant (sweep candidates and default variants; the bf16-exchange
variants are never selected) becomes the row that ends at that size; the last row covers everything above.
Also prints the all_reduce crossover (smallest size at which a large variant beats the H1a algorithm 2) and, for
all_gather, the smallest shard at which the large engine beats the H1a flag path."""
import collections
import json
import sys

rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p) if l.strip()]

DEFAULT_AR = dict(mode=1, chunk_bytes=1 << 20, rs_grid=8, ag_grid=24, block=512, unroll=1, cut_rs=0, cut_ag=0)
AR_VARIANTS = {
    "large_fused": dict(),
    "large_2launch": dict(mode=0, ag_grid=32),
    "large_fused_cut": dict(cut_rs=1, cut_ag=1),
    "large_fused_cutrs": dict(cut_rs=1),
}
DEFAULT_RS = dict(chunk_bytes=1 << 20, grid=8, block=256, unroll=1, cut=0)
RS_VARIANTS = {"large": dict(), "large_cut": dict(cut=1)}
DEFAULT_AG = dict(chunk_bytes=1 << 20, grid=32, block=256, cut=0)
AG_VARIANTS = {"large_default": dict(), "large_cut": dict(cut=1)}


def exact_candidates(rs, base, variants):
    out = []
    for r in rs:
        v = r.get("variant")
        if "topo_b2b_us" not in r or "bx" in (v or ""):
            continue
        if v == "sweep_best":
            out.append((r["topo_b2b_us"], dict(base, **r["cfg"]), v))
        elif v in variants:
            if r.get("bitwise_contract", r.get("bitwise", True)) is False:
                continue
            out.append((r["topo_b2b_us"], dict(base, **variants[v]), v))
    return sorted(out, key=lambda c: c[0])


def table(op, key, base, variants, name, fields, scale):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("op") == op:
            by[(r[key], r.get("layout"))].append(r)
    sizes = sorted({k[0] for k in by})
    print(f"{name} = [")
    for i, s in enumerate(sizes):
        rs = [r for (m, lay), v in by.items() if m == s for r in v]
        cands = exact_candidates(rs, base, variants)
        if not cands:
            continue
        us, cfg, v = cands[0]
        nccl = next((r["nccl_b2b_us"] for r in rs if "nccl_b2b_us" in r), float("nan"))
        limit = "1 << 62" if s == sizes[-1] else str(s * scale)
        print(f"    ({limit}, " + ", ".join(str(cfg[f]) for f in fields) + f"),  # {s} MiB: {us:.1f} us ({v}), NCCL {nccl:.1f} us")
    print("]")


table("all_reduce", "mib", DEFAULT_AR, AR_VARIANTS, "_AR_LARGE_TABLE",
      ["mode", "chunk_bytes", "rs_grid", "ag_grid", "block", "unroll", "cut_rs", "cut_ag"], 1 << 20)
table("reduce_scatter", "mib", DEFAULT_RS, RS_VARIANTS, "_RS_TABLE", ["chunk_bytes", "grid", "block", "unroll", "cut"], 1 << 20)
# all_gather: the size key is the OUTPUT MiB; the table is indexed by shard bytes (= output / 8)
table("all_gather", "out_mib", DEFAULT_AG, AG_VARIANTS, "_AGL_TABLE", ["chunk_bytes", "grid", "block", "cut"], (1 << 20) // 8)

# crossover all_reduce: large variants vs H1a algorithm 2 (and the dispatch of the small path)
by = collections.defaultdict(dict)
for r in rows:
    if r.get("op") == "all_reduce" and "topo_b2b_us" in r:
        by[r["mib"]].setdefault(r["variant"], r["topo_b2b_us"])
        if r["variant"] == "sweep_best":
            by[r["mib"]]["sweep_best"] = min(by[r["mib"]]["sweep_best"], r["topo_b2b_us"])
print("\n# all_reduce crossover (us): H1a algo2 vs best large variant")
for mib in sorted(by):
    d = by[mib]
    large = {k: v for k, v in d.items() if k.startswith("large") and "bx" not in k or k == "sweep_best"}
    if "h1a_algo2" in d and large:
        k = min(large, key=large.get)
        print(f"{mib} MiB: h1a_algo2 {d['h1a_algo2']:.1f}  best large {k} {large[k]:.1f}  -> {'LARGE' if large[k] < d['h1a_algo2'] else 'h1a'}")
