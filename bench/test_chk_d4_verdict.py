"""Unit test of the D4' check tools (s110/chk_driver.py, s110/chk_worker.py): the pure compare / negative-control / verdict logic on synthetic digests.

usage: python3 bench/test_chk_d4_verdict.py       (no GPU, no vLLM; the torch digest test runs only where torch is importable)
Cases: a clean run is PASS; one corrupted block is reported exactly (H-vs-X leg for a tier corruption, X-vs-tier leg for a GPU corruption); a run without a
valid negative control is ARTIFACT (also when bytes differ); a real difference with a valid control is FAIL; any recorded error, missing record, uncovered KV group
or void block is ARTIFACT, never PASS; a single-stream token / logprob / accept-count difference is FAIL.
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "s110"))
os.environ.pop("CHK", None)
import chk_driver as C  # noqa: E402
import chk_worker as W  # noqa: E402

fails = 0


def check(name, ok, info=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  " + str(info)))
    fails += 0 if ok else 1


NG, TP, NREG = 4, 8, 5


def dg(tag):
    return b"".join(bytes([hash((tag, q)) & 0xFF] * 8) for q in range(NREG))


def pairs_of(n, with_h=True):
    return [(f"{k:04x}" * 4, k % NG, k % 3, 100 + k, 200 + k, 1 if with_h else 0) for k in range(n)]


def rank_results(pairs, hx=(), xc=()):
    """What the workers would return: pair k digests equal everywhere except the listed (rank, k) legs."""
    out = []
    for r in range(TP):
        dx = [dg(("x", k)) for k in range(len(pairs))]
        dc = [dg(("x", k)) for k in range(len(pairs))]
        hd = {p[3] - 100 + 1000: dg(("x", k)) for k, p in enumerate(pairs)}
        for (rr, k) in hx:
            if rr == r:
                hd[pairs[k][3] - 100 + 1000] = dg(("other", k))
        for (rr, k) in xc:
            if rr == r:
                dc[k] = dg(("other", k))
        out.append(W.compare_digests(r, NREG, dx, dc, [[p[3] - 100 + 1000] if p[5] else [] for p in pairs], hd))
    return out


def kv_of(pairs, hx=(), xc=(), void=0):
    return C.aggregate_compare(C.merge_rank_results([(0, rank_results(pairs, hx, xc))]), pairs, TP, void)


pairs = pairs_of(40)
clean = kv_of(pairs)
check("clean compare: no mismatch, all ranks", not clean["xc_bad"] and not clean["hx_bad"] and clean["ranks"] == list(range(TP)) and clean["n_with_h"] == 40, clean)
one = kv_of(pairs, hx=[(3, 7)])
check("one corrupted H leg is reported exactly (rank 3, pair 7)", one["hx_bad"] == [[3, pairs[7][0], pairs[7][1], pairs[7][2]]] and not one["xc_bad"], one["hx_bad"])
two = kv_of(pairs, xc=[(6, 9)])
check("one corrupted tier leg is reported exactly (rank 6, pair 9)", two["xc_bad"] == [[6, pairs[9][0], pairs[9][1], pairs[9][2]]] and not two["hx_bad"], two["xc_bad"])
nh = kv_of(pairs_of(10, with_h=False))
check("pairs without a pass-H copy are counted, not compared", nh["n_with_h"] == 0 and not nh["hx_bad"])
a, bad = C.eval_kv(clean, NG, [])
check("eval_kv clean: no artifact reasons", a == [] and bad == {"xc": 0, "hx": 0}, a)
a, bad = C.eval_kv(kv_of(pairs_of(2)), NG, [])
check("eval_kv: uncovered KV groups are an artifact", any("no restored block" in x for x in a), a)
a, _ = C.eval_kv(kv_of(pairs, void=3), NG, [])
check("eval_kv: void blocks are an artifact", any("unverifiable" in x for x in a), a)
a, _ = C.eval_kv(clean, NG, [5])
check("eval_kv: a request with cached tokens but no tier load is an artifact", any("no captured tier load" in x for x in a), a)

# merge of chunks keeps global indices
half = C.merge_rank_results([(0, rank_results(pairs[:20], hx=[(1, 3)])), (20, rank_results(pairs[20:], hx=[(1, 3)]))])
check("chunk merge: global indices", half[1]["hx_bad"] == [3, 23] and half[1]["n"] == 40, half[1])


def nc_of(tier_t=(3, 7), gpu_t=(6, 12), ok=True):
    """A negative control as the driver records it."""
    base = kv_of(pairs)
    t, g = pairs[tier_t[1]], pairs[gpu_t[1]]
    c1 = kv_of(pairs, hx=[tier_t])
    c2 = kv_of(pairs, hx=[tier_t, gpu_t], xc=[gpu_t])
    return {"baseline": base, "after_tier_corrupt": c1, "after_gpu_corrupt": c2, "tier_digest_changed": True, "tier_undo_ok": True, "target_in_loads": True,
            "tier_target": {"rank": tier_t[0], "key": t[0]}, "gpu_target": {"rank": gpu_t[0], "key": g[0]}}


ok, why = C.eval_nc(nc_of())
check("negative control: exact reports are valid", ok, why)
nc = nc_of()
nc["after_tier_corrupt"] = kv_of(pairs)
ok, why = C.eval_nc(nc)
check("negative control: a compare that reports nothing is invalid", not ok, why)
nc = nc_of()
nc["after_tier_corrupt"] = kv_of(pairs, hx=[(3, 7), (4, 8)])
ok, why = C.eval_nc(nc)
check("negative control: an extra reported block is invalid", not ok, why)
nc = nc_of()
nc["after_gpu_corrupt"] = kv_of(pairs, hx=[(3, 7), (6, 12)])
ok, why = C.eval_nc(nc)
check("negative control: the X-vs-tier leg must report the GPU corruption", not ok, why)
for flag in ("tier_digest_changed", "tier_undo_ok", "target_in_loads"):
    nc = nc_of()
    nc[flag] = False
    check(f"negative control: {flag} false is invalid", not C.eval_nc(nc)[0])
check("negative control: absent is invalid", not C.eval_nc(None)[0])


def b_rec(i, kind, **over):
    sp = {"drafted": 70, "accepted": 20, "calls": 10}
    r = {"req": i, "kind": kind, "plen": 100, "lp_requested": False, "tokens_equal": True, "first_diff": None, "lp_equal": True, "lp_max_abs": None,
         "cached": {"C": 0, "H": 64, "X": 64}, "spec": {"C": sp, "H": sp, "X": sp}, "spec_equal": True}
    r.update(over)
    return r


def full_D(spec=True):
    return {"profile": "spec" if spec else "ns256", "spec": spec, "n_groups": NG, "n_det": C.N_DET, "a": clean, "a_uncovered_requests": [], "nc": nc_of(),
            "b": [b_rec(i, k) for k in C.B_KINDS for i in range(C.N_DET)], "errors": {}, "tracer_errors": []}


v = C.final_verdict(full_D())
check("full clean run: PASS", v["verdict"] == "PASS" and not v["reasons_fail"] and not v["reasons_artifact"], v)
D = full_D()
D["nc"] = None
check("no negative control: ARTIFACT", C.final_verdict(D)["verdict"] == "ARTIFACT")
D = full_D()
D["a"] = kv_of(pairs, hx=[(2, 5)])
v = C.final_verdict(D)
check("bytes differ with a valid control: FAIL", v["verdict"] == "FAIL" and "connector bug" in v["reasons_fail"][0], v)
D = full_D()
D["a"] = kv_of(pairs, hx=[(2, 5)])
D["nc"] = None
check("bytes differ without a valid control: ARTIFACT", C.final_verdict(D)["verdict"] == "ARTIFACT")
D = full_D()
D["errors"] = {"d4_b": "Traceback"}
check("a recorded probe error: ARTIFACT", C.final_verdict(D)["verdict"] == "ARTIFACT")
D = full_D()
D["b"] = D["b"][:-1]
check("a missing single-stream record: ARTIFACT", C.final_verdict(D)["verdict"] == "ARTIFACT")
D = full_D()
D["b"][5] = {"req": 5, "kind": "greedy", "error": True}
check("a failed single-stream record: ARTIFACT", C.final_verdict(D)["verdict"] == "ARTIFACT")
D = full_D()
D["b"][5]["tokens_equal"] = False
D["b"][5]["first_diff"] = 17
v = C.final_verdict(D)
check("a token difference H vs X: FAIL", v["verdict"] == "FAIL" and "request 5 greedy" in v["reasons_fail"][0], v)
D = full_D()
D["b"][90]["spec_equal"] = False
check("an accept-count difference (spec): FAIL", C.final_verdict(D)["verdict"] == "FAIL")
D = full_D(spec=False)
D["b"][90]["spec_equal"] = False
for r in D["b"]:
    r["spec"] = {"C": None, "H": None, "X": None}
check("accept counts are ignored on the non-spec profile", C.final_verdict(D)["verdict"] == "PASS")
D = full_D()
for r in D["b"]:
    r["spec"] = {"C": None, "H": None, "X": None}
check("spec profile without any counted draft: ARTIFACT", C.final_verdict(D)["verdict"] == "ARTIFACT")
D = full_D()
D["b"][3]["lp_requested"] = True
D["b"][3]["lp_equal"] = False
check("a logprob difference where requested: FAIL", C.final_verdict(D)["verdict"] == "FAIL")
D = full_D()
D["b"][3]["cached"]["X"] = 32
v = C.final_verdict(D)
check("a hit-length difference with equal tokens is a warning only", v["verdict"] == "PASS" and v["n_warnings"] == 1, v)
D = full_D()
for r in D["b"]:
    r["cached"]["X"] = 0
check("no tier hit at all in (b): ARTIFACT", C.final_verdict(D)["verdict"] == "ARTIFACT")
check("an empty run is ARTIFACT, never PASS", C.final_verdict({"profile": "spec", "spec": True, "n_groups": NG})["verdict"] == "ARTIFACT")

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    # digests() on CPU tensors: a one-byte change moves exactly one region digest of exactly one block
    names = ["a", "b"]
    ts = {n: torch.randint(0, 255, (16, 64), dtype=torch.uint8) for n in names}
    d0 = W.digests(ts, names, list(range(16)))
    ts["b"][5, 17] ^= 0x5A
    d1 = W.digests(ts, names, list(range(16)))
    diff = [(k, q) for k in range(16) for q in range(2) if d0[k][8 * q:8 * q + 8] != d1[k][8 * q:8 * q + 8]]
    check("digests: a one-byte change moves exactly block 5, region b", diff == [(5, 1)], diff)
    ts2 = {n: torch.randint(-128, 127, (16, 64), dtype=torch.int8) for n in names}
    check("digests: int8 regions (the KV region dtype) are digested as bytes", len(W.digests(ts2, names, [0, 3])) == 2)
else:
    print("skip digests test (no torch)")

print("ALL CASES PASSED" if not fails else f"{fails} CASE(S) FAILED")
sys.exit(1 if fails else 0)
