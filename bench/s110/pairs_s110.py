#!/usr/bin/env python3
"""G5 pair checks of the reuse driver output (S1-10; stdlib only; run with python3 -I).

usage: pairs_s110.py <results_dir>/<tag> [...]      (reads reuse_cycle.json / reuse_smoke.json / reuse_chat.json)
Pairs with the same prompt and (by design) the same hit length:
  smoke : smoke_gpuhit2 vs smoke_after_reset1 vs smoke_after_reset2           (GPU hit vs connector hit, repeated connector hit)
  cycle : ab_gpuhit_<L> vs ab_readmit_<L>   (same prompt; the re-admission is a connector hit with a connector, a recompute without)
          doc<i>_p2 vs doc<i>_p3 for every document                          (two reuse passes of the same prompt)
For every pair: text equal?, logprobs equal?, first divergent token index, max |dlogprob| over the common prefix of equal tokens.
"""
import json
import os
import sys


def cmp(a, b):
    ta, tb = a.get("toks") or [], b.get("toks") or []
    va, vb = a.get("vals") or [], b.get("vals") or []
    n = min(len(ta), len(tb))
    div = next((i for i in range(n) if ta[i] != tb[i]), None)
    m = n if div is None else div
    md = max((abs(va[i] - vb[i]) for i in range(m) if va[i] is not None and vb[i] is not None), default=0.0)
    # the logprob sha hashes the per-chunk strings of the stream (chunking varies); the arrays decide
    arr_eq = ta == tb and va == vb
    return {"text_eq": a.get("text_sha") == b.get("text_sha"), "lp_eq": arr_eq,
            "first_div": div, "max_dlp": md}


def main():
    for d in sys.argv[1:]:
        print("== %s" % os.path.basename(d.rstrip("/")))
        for name in ("smoke", "cycle", "chat"):
            f = "%s/reuse_%s.json" % (d, name)
            if not os.path.exists(f):
                continue
            sends = {}
            for s in json.load(open(f)).get("sends", []):
                sends[s["label"]] = s
            pairs = []
            if name == "smoke":
                ls = sorted({k.split("_")[1] for k in sends if k.startswith("smoke_") and k.endswith("_cold")}, key=lambda x: int(x) if x.isdigit() else 0)
                if not ls:  # old driver labels
                    pairs = [("smoke_gpuhit2", "smoke_after_reset1"), ("smoke_after_reset1", "smoke_after_reset2")]
                for L in ls:
                    t = "smoke_%s_" % L
                    pairs += [(t + "gpuhit2", t + "after_reset1"), (t + "after_reset1", t + "after_reset2")]
            elif name == "cycle":
                for L in sorted({int(k.rsplit("_", 1)[1]) for k in sends if k.startswith("ab_gpuhit_")}):
                    pairs.append(("ab_gpuhit_%d" % L, "ab_readmit_%d" % L))
                idx = sorted({int(k[3:].split("_")[0]) for k in sends if k.startswith("doc") and k.endswith("_p2")})
                pairs += [("doc%d_p2" % i, "doc%d_p3" % i) for i in idx]
            for a, b in pairs:
                if a in sends and b in sends and not a.startswith("doc"):
                    print("  %-20s vs %-20s %s" % (a, b, json.dumps(cmp(sends[a], sends[b]))))
            docs = [(a, b) for a, b in pairs if a.startswith("doc") and a in sends and b in sends]
            if docs:
                r = [cmp(sends[a], sends[b]) for a, b in docs]
                print("  cycle p2 vs p3 over %d documents: text_eq %d, lp_eq %d, first_div none %d, max |dlogprob| %.3g" % (
                    len(r), sum(x["text_eq"] for x in r), sum(x["lp_eq"] for x in r), sum(x["first_div"] is None for x in r),
                    max(x["max_dlp"] for x in r)))


if __name__ == "__main__":
    main()
