"""ms/step estimate of a table change (S1-9 L9 A1): sum over the collectives of a band of (calls per step) x (dispatch time delta).

usage: ms_step_a1.py <sizes.json> <before.jsonl> <after.jsonl> [--tag-before X --tag-after Y]
sizes.json: list of {band, site, op (AR / RS / AG0 / AG-1), shape, bytes, calls} (docs/a1_table_retune.md, generated from the model code
and the round-5 op log). before / after: bench_a1_tables.py --mode dispatch rows. A band's delta is reported per protocol (warm, spacer,
cold = model-like cold, coldw = spike cold); the microbench has no EP straggler skew and no eager CPU cost, so the numbers are the
GPU-time change of the collective kernels on the critical path, an upper bound of the step gain (a collective that waits for a slow
rank in the model does not shorten that wait).
"""
import json
import statistics
import sys


def target_name(op, shape):
    r, c = shape
    if op == "AR":
        return f"ar_T{r}x{c}"
    if op == "RS":
        return f"rs_T{r}x{c}"
    if op == "AG-1":
        return f"agc_T{r}x{c}"
    # AG0: payload dtype is not in the shape; the model code gives it: 8192 / 4096 bf16, 2304 u8, 16 i32
    suffix = {2304: "u8", 16: "i32"}.get(c, "")
    return f"ag0_R{r}x{c}{suffix}"


def load(path, tag):
    d = {}
    for line in open(path):
        r = json.loads(line)
        if tag is None or r.get("tag") == tag:
            d.setdefault(r["target"], []).append(r)
    out = {}
    for t, rs in d.items():
        out[t] = {p: statistics.median(statistics.median(x[p]["max"] for x in r["rounds"]) for r in rs) for p in ("warm", "spacer", "cold", "coldw")}
    return out


def main():
    args = sys.argv[1:]
    tb = ta = None
    for flag in ("--tag-before", "--tag-after"):
        if flag in args:
            i = args.index(flag)
            v = args[i + 1]
            del args[i:i + 2]
            if flag == "--tag-before":
                tb = v
            else:
                ta = v
    sizes = json.load(open(args[0]))
    b, a = load(args[1], tb), load(args[2], ta)
    protos = ("warm", "spacer", "cold", "coldw")
    bands = {}
    missing = set()
    for r in sizes:
        t = target_name(r["op"], r["shape"])
        if t not in b or t not in a:
            missing.add(t)
            continue
        e = bands.setdefault(r["band"], {p: 0.0 for p in protos} | {"before": {p: 0.0 for p in protos}, "rows": []})
        for p in protos:
            e[p] += r["calls"] * (a[t][p] - b[t][p]) / 1000.0
            e["before"][p] += r["calls"] * b[t][p] / 1000.0
        d = {p: a[t][p] - b[t][p] for p in protos}
        if any(abs(v) > 0.05 for v in d.values()):
            e["rows"].append((r["site"], t, r["calls"], d))
    print("band | ms/step delta (negative = faster): warm / spacer / cold / coldw | collective ms/step before (cold)")
    for band, e in bands.items():
        print(f"{band:24s} " + " / ".join(f"{e[p]:+.3f}" for p in protos) + f" | {e['before']['cold']:.2f}")
        for site, t, calls, d in e["rows"]:
            print(f"    {site:36s} {t:20s} x{calls:<4d} " + " / ".join(f"{d[p]:+6.1f}" for p in protos) + " us per call")
    if missing:
        print("not measured:", sorted(missing))


if __name__ == "__main__":
    main()
