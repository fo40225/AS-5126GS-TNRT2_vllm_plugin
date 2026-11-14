"""Turn bench_medium.py rows into the dispatch table of the bf16-exchange all-reduce and a markdown summary.

usage: make_table_medium.py <rows.jsonl>... --json <tuning.json> [--md <table.md>] [--margin 0.02]
Selection (all numbers are the full-protocol retimes of bench_medium.py, score = mean of b2b and spacer time):
  * per measured size the best (grid, block, unroll) of the ar2 bf16-exchange rows (within 1.5 % of the best, the one
    with the fewest threads wins: fewer co-resident spinning CTAs are safer next to other kernels),
  * bx_min = smallest size from which the bf16 exchange beats the best exact ar2 by more than --margin at every
    measured size above it (no bf16 exchange below it),
  * large_min_bx = smallest size from which the large algorithm (bf16 exchange) beats the best ar2 BX by more than
    --margin at every measured size above it (default: the end of the measured range).
The JSON file is read by TopoComm through MUSE_TOPO_AR_TUNING_FILE; the printed Python literal goes into collectives.py.
"""
import argparse
import json
from collections import defaultdict


def score(r):
    return 0.5 * (r["b2b_us"] + r["spacer_us"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--max-bytes", type=int, default=8 << 20)
    args = ap.parse_args()
    by = defaultdict(lambda: defaultdict(list))
    nccl = {}
    for path in args.rows:
        for line in open(path):
            r = json.loads(line)
            if r.get("op") != "all_reduce" or "variant" not in r or "error" in r:
                continue
            by[r["bytes"]][r["variant"]].append(r)
            nccl[r["bytes"]] = (r["nccl_b2b_us"], r["nccl_spacer_us"])
    sizes = sorted(by)
    best = {}
    for b in sizes:
        v = by[b]
        e = {}
        for name in ("ar2", "ar2bx", "largebx"):
            rows = [r for r in v.get(name, []) if r.get("bitwise_ok", True)]
            if not rows:
                continue
            m = min(score(r) for r in rows)
            near = [r for r in rows if score(r) <= m * 1.015]
            if name != "largebx":
                near.sort(key=lambda r: (r["cfg"]["grid"] * r["cfg"]["block"], score(r)))
            else:
                near.sort(key=score)
            e[name] = near[0]
        best[b] = e

    def wins(b, a_name, b_name):
        e = best[b]
        return a_name in e and b_name in e and score(e[a_name]) < (1.0 - args.margin) * score(e[b_name])

    bx_min = None
    for i, b in enumerate(sizes):
        if all(wins(s, "ar2bx", "ar2") for s in sizes[i:] if "ar2" in best[s] and "ar2bx" in best[s]):
            bx_min = b
            break
    large_min = None
    for i, b in enumerate(sizes):
        if "largebx" in best[b] and all(wins(s, "largebx", "ar2bx") for s in sizes[i:] if "largebx" in best[s]):
            large_min = b
            break
    table = []
    for b in sizes:
        r = best[b].get("ar2bx")
        if r is None:
            continue
        c = r["cfg"]
        table.append([b, 2, c["grid"], c["block"], c["unroll"]])
    if table:
        table.append([1 << 62] + table[-1][1:])
    tuning = {"bx_table": table, "bx_min": bx_min if bx_min is not None else max(sizes) + 1,
              "large_min_bx": large_min if large_min is not None else args.max_bytes + 1}
    json.dump(tuning, open(args.json, "w"), indent=1)
    print("bx_min", tuning["bx_min"], "large_min_bx", tuning["large_min_bx"])
    print("_AR_BX_TABLE = [")
    for row in table:
        print(f"    ({row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}),")
    print("]")
    lines = ["| KiB | shape | NCCL b2b | NCCL spacer | exact ar2 b2b / spacer (cfg) | ar2 BX b2b / spacer (cfg) | large BX b2b / spacer | BX vs exact (score) |",
             "|---|---|---|---|---|---|---|---|"]
    for b in sizes:
        e = best[b]

        def cell(name):
            r = e.get(name)
            if r is None:
                return "-"
            c = None if name == "largebx" else r.get("cfg")
            cs = "" if c is None else " (" + "/".join(str(c[k]) for k in ("grid", "block", "unroll") if k in c) + ")"
            return f"{r['b2b_us']:.1f} / {r['spacer_us']:.1f}{cs}"

        sp = "-"
        if "ar2" in e and "ar2bx" in e:
            sp = f"{score(e['ar2']) / score(e['ar2bx']):.2f}x"
        shape = by[b][next(iter(by[b]))][0]["shape"]
        lines.append(f"| {b // 1024} | {shape} | {nccl[b][0]:.1f} | {nccl[b][1]:.1f} | {cell('ar2')} | {cell('ar2bx')} | {cell('largebx')} | {sp} |")
    md = "\n".join(lines)
    print(md)
    if args.md:
        open(args.md, "w").write(md + "\n")


if __name__ == "__main__":
    main()
