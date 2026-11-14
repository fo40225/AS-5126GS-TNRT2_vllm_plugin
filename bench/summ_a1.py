"""Summaries of bench_a1_tables.py output (S1-9 L9 A1).

usage: summ_a1.py sweep <jsonl> [...]            adoption verdict per (target, finalist) under the pre-registered rule
       summ_a1.py cmp <before.jsonl> <after.jsonl> [--tag-before X --tag-after Y]   per-size dispatch timing and digest comparison

Rule (sweep): a finalist replaces the baseline dispatch of a size only if
  * baseline and finalist are bitwise equal to the contract emulation on every rank and all four data kinds,
  * the model-like COLD protocol (read flush + produced input) improves in the median of 3 rounds by more than
    max(1.5 us, 3 % of the baseline, 2 x the baseline's own round spread), in every round, on both islands,
  * SPACER is not slower by more than max(1 us, 1.5 %), WARM not by more than max(1 us, 2 %), the memset COLD (spike protocol,
    "coldw") not by more than max(1.5 us, 3 %).
"""
import json
import statistics
import sys


def rounds_stat(rounds, proto, f="max"):
    v = [r[proto][f] for r in rounds]
    return statistics.median(v), max(v) - min(v), v


def verdict(base, cand):
    out = {}
    ok = True
    why = []
    bc, spread, bv = rounds_stat(base["rounds"], "cold")
    cc, _, cv = rounds_stat(cand["rounds"], "cold")
    thr = max(1.5, 0.03 * bc, 2 * spread)
    gains = [b - c for b, c in zip(bv, cv)]
    isl_ok = all(rounds_stat(base["rounds"], "cold", f)[0] - rounds_stat(cand["rounds"], "cold", f)[0] > 0 for f in ("isl0", "isl1"))
    out["cold"] = (round(bc, 1), round(cc, 1), round(bc - cc, 1), round(thr, 1))
    if not (bc - cc > thr and all(g > 0 for g in gains) and isl_ok):
        ok = False
        why.append(f"cold gain {bc - cc:.1f} us (thr {thr:.1f}, rounds {[round(g, 1) for g in gains]}, islands {'ok' if isl_ok else 'BAD'})")
    for proto, rel, ab in (("spacer", 0.015, 1.0), ("warm", 0.02, 1.0), ("coldw", 0.03, 1.5)):
        bm, _, _ = rounds_stat(base["rounds"], proto)
        cm, _, _ = rounds_stat(cand["rounds"], proto)
        out[proto] = (round(bm, 1), round(cm, 1), round(bm - cm, 1))
        if cm > bm + max(ab, rel * bm):
            ok = False
            why.append(f"{proto} slower by {cm - bm:.1f} us")
    if not cand.get("bitwise_all_kinds", False):
        ok = False
        why.append("NOT bitwise")
    return ok, out, why


def sweep(paths):
    rows = [json.loads(l) for p in paths for l in open(p)]
    byt = {}
    for r in rows:
        if "variant" in r:
            byt.setdefault(r["target"], []).append(r)
    heads = {r["target"]: r for r in rows if "variant" not in r}
    for t, rs in byt.items():
        base = next(r for r in rs if r["variant"] == "baseline")
        h = heads.get(t, {})
        print(f"== {t} ({base['bytes'] / 1024:.0f} KiB) baseline_ok={h.get('baseline_ok')} screened={h.get('screened')} invalid={h.get('invalid')}")
        bc = rounds_stat(base["rounds"], "cold")
        print(f"   baseline cold {bc[0]:.1f} (spread {bc[1]:.1f}) spacer {rounds_stat(base['rounds'], 'spacer')[0]:.1f} warm {rounds_stat(base['rounds'], 'warm')[0]:.1f} "
              f"coldw {rounds_stat(base['rounds'], 'coldw')[0]:.1f}")
        best = None
        for c in rs:
            if c["variant"] == "baseline":
                continue
            ok, m, why = verdict(base, c)
            print(f"   {'ADOPT' if ok else 'no   '} {c['variant']:44s} cold {m['cold'][1]:.1f} ({-m['cold'][2]:+.1f}) spacer {m['spacer'][1]:.1f} ({-m['spacer'][2]:+.1f}) "
                  f"warm {m['warm'][1]:.1f} ({-m['warm'][2]:+.1f}) coldw {m['coldw'][1]:.1f} ({-m['coldw'][2]:+.1f})" + ("" if ok else "  [" + "; ".join(why) + "]"))
            if ok and (best is None or m["cold"][2] > best[1]["cold"][2]):
                best = (c, m)
        if best:
            print(f"   -> best {best[0]['variant']} cfg={json.dumps(best[0]['cfg'])}")


def med(r, proto):
    return statistics.median(x[proto]["max"] for x in r["rounds"])


def cmp(before, after, tb=None, ta=None):
    def load(p, tag):
        d = {}
        for l in open(p):
            r = json.loads(l)
            if tag is None or r.get("tag") == tag:
                d.setdefault(r["target"], []).append(r)
        return d

    b, a = load(before, tb), load(after, ta)
    print(f"{'target':20s} {'KiB':>8s} | {'warm':>8s} {'spacer':>8s} {'cold':>8s} {'coldw':>8s} (before) | delta after-before warm spacer cold coldw | digests")
    for t in b:
        if t not in a:
            continue
        rb, ra = b[t], a[t]
        mb = {p: statistics.median(med(r, p) for r in rb) for p in ("warm", "spacer", "cold", "coldw")}
        ma = {p: statistics.median(med(r, p) for r in ra) for p in ("warm", "spacer", "cold", "coldw")}
        same = all(rb[0]["digests"][k] == ra[0]["digests"][k] for k in rb[0]["digests"])
        oks = all(all(r["bitwise_contract"].values()) for r in rb + ra)
        print(f"{t:20s} {rb[0]['bytes'] / 1024:8.0f} | " + " ".join(f"{mb[p]:8.1f}" for p in mb) + " | " + " ".join(f"{ma[p] - mb[p]:+8.1f}" for p in mb)
              + f" | digests {'IDENTICAL' if same else 'DIFFER'} contract {'ok' if oks else 'FAIL'}")


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep(sys.argv[2:])
    else:
        args = sys.argv[2:]
        tb = ta = None
        if "--tag-before" in args:
            i = args.index("--tag-before")
            tb = args[i + 1]
            del args[i:i + 2]
        if "--tag-after" in args:
            i = args.index("--tag-after")
            ta = args[i + 1]
            del args[i:i + 2]
        cmp(args[0], args[1], tb, ta)
