"""S1-10 gate: determinism comparison of the four e2e_s110.py processes (stdlib only): python3 -I cmp_e2e_s110.py <dir> <ONa> <ONb> <OFFa> <OFFb> [<min coverage of equal hit lengths, default 0.90>]
Per process <dir>/<name>.<mode>_<C|H|X>.json (modes greedy, t1k64, mixed, lp). Token ids are compared exactly (the `lp` pass also the top-5 logprob values and ranks);
a pair of passes is compared per REQUEST, only where both requests had the same num_cached_tokens (the hit length): greedy outputs depend on the cached length and the
chunk shapes, so a hit and a recompute are different computations. Gates (pre-registered in ADOPTION_RULE_FINAL.txt):
  D1 run-to-run: ONa == ONb and OFFa == OFFb, every pass, every request (same process type: identical hit lengths too, which is checked as well)
  D2 cold: ONa C == OFFa C (and ONb C == OFFb C)
  D3 GPU hit: ONa H == OFFa H
  D4 tier hit: ONa X == ONa H and ONa X == OFFa H  (with the tier on, X is a connector hit; the pairs with equal hit lengths must cover >= the minimum coverage)
  D5 recompute: OFFa X == OFFa C (a recompute after the GPU reset is the cold computation again) and ONa C == OFFa X
Prints one line per comparison and E2E_GATE=PASS|FAIL (FAIL on any equal-hit-length difference or a coverage below the minimum; MISSING pass files are FAIL)."""
import json
import sys

W, ONA, ONB, OFFA, OFFB = sys.argv[1:6]
MINCOV = float(sys.argv[6]) if len(sys.argv) > 6 else 0.90
MODES = ("greedy", "t1k64", "mixed", "lp")
ok = True


def load(arm, tag):
    return json.load(open("%s/%s.%s.json" % (W, arm, tag)))["requests"]


def cmp_pair(a, ta, b, tb, label, need_same_hit=True, strict_hits=False):
    """compare request by request; returns (compared, total, differing)"""
    global ok
    try:
        x, y = load(a, ta), load(b, tb)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print("GATE %s: MISSING %s FAIL" % (label, e))
        ok = False
        return
    n = len(x)
    same_hit = [i for i in range(min(n, len(y))) if x[i].get("num_cached_tokens") == y[i].get("num_cached_tokens")]
    use = same_hit if need_same_hit else list(range(min(n, len(y))))
    diff = [i for i in use if x[i]["token_ids"] != y[i]["token_ids"] or x[i].get("logprobs") != y[i].get("logprobs")]
    cov = len(use) / n if n else 0.0
    good = (not diff) and n == len(y) and n > 0 and cov >= (1.0 if strict_hits else MINCOV)
    ok = ok and good
    print("GATE %s: %d/%d requests compared (equal hit length%s), %d differ%s, hits ON-side sum %d vs %d %s" % (
        label, len(use), n, ", all" if len(use) == n else "", len(diff), (" first at request %d (prompt %d tokens)" % (diff[0], x[diff[0]]["prompt_len"])) if diff else "",
        sum(r.get("num_cached_tokens") or 0 for r in x), sum(r.get("num_cached_tokens") or 0 for r in y), "PASS" if good else "FAIL"))


for m in MODES:
    for t in "CHX":
        tag = "%s_%s" % (m, t)
        cmp_pair(ONA, tag, ONB, tag, "D1 %s vs %s %s" % (ONA, ONB, tag), strict_hits=True)
        cmp_pair(OFFA, tag, OFFB, tag, "D1 %s vs %s %s" % (OFFA, OFFB, tag), strict_hits=True)
    cmp_pair(ONA, m + "_C", OFFA, m + "_C", "D2 cold %s vs %s %s_C" % (ONA, OFFA, m))
    cmp_pair(ONB, m + "_C", OFFB, m + "_C", "D2 cold %s vs %s %s_C" % (ONB, OFFB, m))
    cmp_pair(ONA, m + "_H", OFFA, m + "_H", "D3 GPU hit %s vs %s %s_H" % (ONA, OFFA, m))
    cmp_pair(ONA, m + "_X", ONA, m + "_H", "D4 tier hit vs GPU hit inside %s %s" % (ONA, m))
    cmp_pair(ONA, m + "_X", OFFA, m + "_H", "D4 tier hit %s vs GPU hit %s %s" % (ONA, OFFA, m))
    cmp_pair(OFFA, m + "_X", OFFA, m + "_C", "D5 recompute vs cold inside %s %s" % (OFFA, m))
    cmp_pair(ONA, m + "_C", OFFA, m + "_X", "D5 cold %s vs recompute %s %s" % (ONA, OFFA, m))
print("E2E_GATE=%s" % ("PASS" if ok else "FAIL"))
