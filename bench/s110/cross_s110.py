#!/usr/bin/env python3
"""Cross-boot comparison of the smoke requests of two boots (S1-10; stdlib only): for every smoke label present in both, text / logprob equality.
usage: cross_s110.py <results>/<tagA> <results>/<tagB> [label-regex]     e.g. OFF_B vs SIM_B: gpuhit2 (A) against gpuhit2 and after_reset1 (B)"""
import json
import re
import sys

a, b = sys.argv[1], sys.argv[2]
rx = re.compile(sys.argv[3] if len(sys.argv) > 3 else r".*")
ja = {s["label"]: s for s in json.load(open(a + "/reuse_smoke.json"))["sends"]}
jb = {s["label"]: s for s in json.load(open(b + "/reuse_smoke.json"))["sends"]}


def cmp(x, y):
    n = min(len(x["toks"]), len(y["toks"]))
    div = next((i for i in range(n) if x["toks"][i] != y["toks"][i]), None)
    m = n if div is None else div
    md = max((abs(x["vals"][i] - y["vals"][i]) for i in range(m) if x["vals"][i] is not None and y["vals"][i] is not None), default=0.0)
    return x["text_sha"] == y["text_sha"], (x["toks"] == y["toks"] and x["vals"] == y["vals"]), div, md


for la in sorted(ja):
    if not rx.search(la):
        continue
    for lb in sorted(jb):
        if lb != la and not (la.endswith("gpuhit2") and lb.endswith("after_reset1") and la.split("_")[1] == lb.split("_")[1]):
            continue
        t, l, d, m = cmp(ja[la], jb[lb])
        print("%-26s (A) vs %-26s (B): text_eq=%s lp_eq=%s first_div=%s max|dlp|=%.3g" % (la, lb, t, l, d, m))
