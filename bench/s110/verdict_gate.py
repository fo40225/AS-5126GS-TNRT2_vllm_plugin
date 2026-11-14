#!/usr/bin/env python3
"""S1-10 FINAL GATE verdict: implements /tmp/muse_wf/s1-10/ADOPTION_RULE_FINAL.txt (copy: bench/s110/rules/ADOPTION_RULE_FINAL.txt). stdlib only: python3 -I verdict_gate.py ...

usage: verdict_gate.py [--root /raid/vllm-runtime/musetest/results] [--prefix s110g] [--groups 1,2,3] [--profiles spec,ns256]
                       [--det spec=FILE,ns256=FILE] [--preempt DIR] [--fulltier DIR] [--stress spec=DIR[,DIR],ns256=DIR,ns128=DIR] [--json OUT] [--selftest]
Boot directories: <root>/<prefix>_<profile>_<ARM>_<g> (ARM = CTL | OFF | ON), written by master_gate.sh / job_gate.sh.
Every threshold below is a constant of this file (the rule's text and this file are frozen together: md5 in PROGRESS.md before the first gate boot).
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import tempfile

# ---- thresholds of the rule -------------------------------------------------------------------------------------------------------------------------
P_TTFT_P50_MAX = 0.70        # primary: reuse passes TTFT p50 ratio ON/OFF, every group and pooled
P_TOKS_MIN = 1.30            # reuse passes prompt tok/s ratio ON/OFF
P_TTFT_P99_MAX = 1.00        # reuse passes TTFT p99 ratio ON/OFF
P_READMIT_MAX = 0.70         # 118k re-admission TTFT ratio ON/OFF
P_HIT_MIN = 0.90             # external prefix cache hit tokens / queried tokens over the reuse passes
G1_STEADY_GMEAN_MIN = 0.990  # STEADY step ratio OFF/ON (or CTL/ON), geometric mean over the cells
G1_STEADY_CELL_MIN = 0.985
G1_DEC_GMEAN_MIN = 0.990     # DEC output tok/s ratio ON/OFF (or ON/CTL)
G1_DEC_CELL_MIN = 0.975
G2_COLD_TOKS_MIN = 0.98      # cold pass 1 prompt tok/s ratio ON/OFF pooled
G2_COLD_TOKS_GROUP_MIN = 0.96
G2_COLD_TTFT_MAX = 1.02      # ab_cold 64k / 118k TTFT ratio ON/OFF pooled
G3_READY_PLUS_S = 180.0      # ON READY seconds <= median OFF READY + this
G3_MIN_AVAIL_MIB = 4096.0    # min over the run of MemFree + file cache of every GPU node
G4_GPU_DELTA_MIB = 96.0      # GPU used at READY: ON - OFF
STEADY_CELLS = [1, 16, 64, 128]
DEC_CELLS = [1, 16, 64, 128]
GPU_NODES = [0, 3, 4, 7]
PRIMARY_PASSES = {"spec": 2, "ns256": 2}   # first pass that counts as a reuse pass (pass 1 is the cold fill); all later passes are pooled
STRESS_FLOOR = {"spec": 1256.0, "ns256": 500.0, "ns128": 500.0}


def jload(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(math.ceil(p / 100.0 * len(s))) - 1))]


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class Boot:
    def __init__(self, root, tag):
        self.dir = os.path.join(root, tag)
        self.tag = tag
        self.exists = os.path.isdir(self.dir)

    def f(self, name):
        return os.path.join(self.dir, name)

    def done(self):
        try:
            return "DONE unsloth" in open(self.f("status.txt"), errors="replace").read()
        except OSError:
            return False

    def ready_s(self):
        try:
            for line in open(self.f("status.txt"), errors="replace"):
                m = re.search(r"READY load_s=(\d+)", line)
                if m:
                    return float(m.group(1))
        except OSError:
            pass
        return None

    def steady(self, c):
        d = jload(self.f("steady/steady_s_c%d.json" % c))
        if not d or not d.get("values"):
            return None
        return mean(d["values"]), d.get("failed", 0)

    def dec(self, c):
        vals, bad = [], 0
        for p in sorted(glob.glob(self.f("real_r*_c%d.json" % c))):
            d = jload(p)
            if d:
                vals.append(d["output_throughput"])
                bad += d.get("failed", 0)
        return (mean(vals), bad) if vals else None

    def reuse(self):
        return jload(self.f("reuse_cycle.json"))

    def gpu_used(self):
        try:
            v = [float(re.search(r",\s*(\d+)", l).group(1)) for l in open(self.f("gpu_mem_ready.txt")) if l.strip()]
            return max(v) if v else None
        except (OSError, AttributeError):
            return None

    def oom_lines(self):
        try:
            return sum(1 for l in open(self.f("serve.log"), errors="replace") if re.search(r"OutOfMemoryError|CUDA out of memory", l))
        except OSError:
            return None

    def watchdog_trips(self):
        try:
            return sum(1 for l in open(self.f("status.txt"), errors="replace") if "WATCHDOG" in l)
        except OSError:
            return None

    def min_avail(self):
        """min over the run of MemFree + file cache (MiB) of every GPU node, from the 2 s sampler"""
        best = None
        try:
            for l in open(self.f("node_mem_ts.txt")):
                m = re.match(r"\S+ node(\d) free=(\d+) file=(\d+)", l)
                if m and int(m.group(1)) in GPU_NODES:
                    a = int(m.group(2)) + int(m.group(3))
                    best = a if best is None else min(best, a)
        except OSError:
            return None
        return best

    def preemptions(self):
        tot = 0.0
        seen = False
        for p in glob.glob(self.f("metrics_after_*.txt")):
            for l in open(p):
                if l.startswith("vllm:num_preemptions_total"):
                    tot = max(tot, float(l.rsplit(" ", 1)[1])); seen = True
        return tot if seen else None


def reuse_stats(boot, first_pass):
    d = boot.reuse()
    if not d:
        return None
    ttft, toks_n, wall, hit, q = [], 0.0, 0.0, 0.0, 0.0
    for name, ps in d.get("passes", {}).items():
        m = re.match(r"cycle_p(\d+)$", name)
        if not m or int(m.group(1)) < first_pass:
            continue
        toks_n += ps["prompt_tokens"]; wall += ps["wall_s"]
        md = ps.get("metric_delta", {})
        hit += md.get("vllm:external_prefix_cache_hits_total", 0.0)
        q += md.get("vllm:external_prefix_cache_queries_total", 0.0)
    for s in d.get("sends", []):
        m = re.match(r"cycle_p(\d+)$", str(s.get("pass")))
        if m and int(m.group(1)) >= first_pass and s.get("ttft_s") is not None and s.get("status") == 200:
            ttft.append(s["ttft_s"])
    sends = {s["label"]: s for s in d.get("sends", [])}
    p1 = d.get("passes", {}).get("cycle_p1", {})
    failed = sum(1 for s in d.get("sends", []) if s.get("status") != 200)
    return {"p50": pct(ttft, 50), "p99": pct(ttft, 99), "toks": toks_n / wall if wall else None, "hit": (hit / q) if q else None,
            "readmit118": (sends.get("ab_readmit_117964") or {}).get("ttft_s"), "readmit64": (sends.get("ab_readmit_65536") or {}).get("ttft_s"),
            "cold_toks": p1.get("tok_per_s"), "cold64": (sends.get("ab_cold_65536") or {}).get("ttft_s"),
            "cold118": (sends.get("ab_cold_117964") or {}).get("ttft_s"), "failed": failed, "n": len(ttft)}


def ratio(a, b):
    return (a / b) if (a is not None and b not in (None, 0)) else None


class Report:
    def __init__(self):
        self.lines = []
        self.results = {}

    def crit(self, key, ok, text):
        self.results[key] = bool(ok)
        self.lines.append("%-6s %s  %s" % ("PASS" if ok else "FAIL", key, text))

    def info(self, text):
        self.lines.append("       " + text)


def fmt(x, n=3):
    return "n/a" if x is None else ("%.*f" % (n, x))


def judge_profile(root, prefix, profile, groups, rep):
    boots = {arm: {g: Boot(root, "%s_%s_%s_%d" % (prefix, profile, arm, g)) for g in groups} for arm in ("CTL", "OFF", "ON")}
    k = profile
    first = PRIMARY_PASSES[profile]
    # ---- boots present and finished
    for arm in ("CTL", "OFF", "ON"):
        miss = [g for g in groups if not boots[arm][g].exists or not boots[arm][g].done()]
        rep.crit("%s.boots.%s" % (k, arm), not miss, "%s: %d/%d boots finished%s" % (arm, len(groups) - len(miss), len(groups), (" (missing groups %s)" % miss) if miss else ""))
    # ---- PRIMARY (reuse), ON vs OFF per group
    rs = {arm: {g: reuse_stats(boots[arm][g], first) for g in groups} for arm in ("OFF", "ON")}
    rows = []
    allok = True
    for g in groups:
        on, off = rs["ON"][g], rs["OFF"][g]
        if not on or not off:
            allok = False
            rep.info("group %d: reuse data missing (ON %s, OFF %s)" % (g, bool(on), bool(off)))
            continue
        r = {"p50": ratio(on["p50"], off["p50"]), "p99": ratio(on["p99"], off["p99"]), "toks": ratio(on["toks"], off["toks"]),
             "readmit": ratio(on["readmit118"], off["readmit118"]), "hit": on["hit"]}
        ok = (r["p50"] is not None and r["p50"] <= P_TTFT_P50_MAX and r["toks"] is not None and r["toks"] >= P_TOKS_MIN and r["p99"] is not None and r["p99"] <= P_TTFT_P99_MAX
              and r["readmit"] is not None and r["readmit"] <= P_READMIT_MAX and r["hit"] is not None and r["hit"] >= P_HIT_MIN)
        allok = allok and ok
        rows.append(r)
        rep.info("group %d: TTFT p50 ON %s / OFF %s s = x%s | p99 ON %s / OFF %s s = x%s | prompt tok/s ON %s / OFF %s = x%s | 118k readmit ON %s / OFF %s s = x%s | HIT %s | %s" % (
            g, fmt(on["p50"]), fmt(off["p50"]), fmt(r["p50"]), fmt(on["p99"]), fmt(off["p99"]), fmt(r["p99"]), fmt(on["toks"], 0), fmt(off["toks"], 0), fmt(r["toks"], 2),
            fmt(on["readmit118"]), fmt(off["readmit118"]), fmt(r["readmit"]), fmt(on["hit"]), "ok" if ok else "FAILS"))
    rep.crit("%s.primary" % k, allok and len(rows) == len(groups),
             "reuse passes ON vs OFF in every group: TTFT p50 <= %.2f, prompt tok/s >= %.2f, TTFT p99 <= %.2f, 118k re-admission <= %.2f, HIT >= %.2f" % (
                 P_TTFT_P50_MAX, P_TOKS_MIN, P_TTFT_P99_MAX, P_READMIT_MAX, P_HIT_MIN))
    if rows:
        rep.info("mean over groups: p50 x%s p99 x%s tok/s x%s readmit x%s; 64k re-admission (reported) %s" % (
            fmt(mean([r["p50"] for r in rows])), fmt(mean([r["p99"] for r in rows])), fmt(mean([r["toks"] for r in rows]), 2), fmt(mean([r["readmit"] for r in rows])),
            ", ".join("ON %s / OFF %s s" % (fmt((rs["ON"][g] or {}).get("readmit64")), fmt((rs["OFF"][g] or {}).get("readmit64"))) for g in groups)))
    # ---- G1: STEADY and DEC, ON vs OFF and ON vs CTL
    for base in ("OFF", "CTL"):
        sr = {}
        for c in STEADY_CELLS:
            per = []
            for g in groups:
                a, b = boots[base][g].steady(c), boots["ON"][g].steady(c)
                per.append(ratio(a[0], b[0]) if a and b else None)        # base step / ON step: >1 = ON faster
            sr[c] = per
        cell_means = {c: mean(v) for c, v in sr.items()}
        gm = gmean(list(cell_means.values()))
        complete = all(None not in v for v in sr.values())
        ok = complete and gm is not None and gm >= G1_STEADY_GMEAN_MIN and all(cell_means[c] >= G1_STEADY_CELL_MIN for c in STEADY_CELLS)
        rep.crit("%s.G1.steady.vs%s" % (k, base), ok, "STEADY step ratio %s/ON gmean %s (>= %.3f), cells %s (each >= %.3f), per-group spread %s" % (
            base, fmt(gm, 4), G1_STEADY_GMEAN_MIN, " ".join("c%d %s" % (c, fmt(cell_means[c], 4)) for c in STEADY_CELLS), G1_STEADY_CELL_MIN,
            " ".join("c%d sd %s" % (c, fmt(statistics.pstdev([x for x in sr[c] if x is not None]) if len([x for x in sr[c] if x is not None]) > 1 else None, 4)) for c in STEADY_CELLS)))
        dr = {}
        for c in DEC_CELLS:
            per = []
            for g in groups:
                a, b = boots[base][g].dec(c), boots["ON"][g].dec(c)
                per.append(ratio(b[0], a[0]) if a and b else None)         # ON tok/s / base tok/s
            dr[c] = per
        dm = {c: mean(v) for c, v in dr.items()}
        dg = gmean(list(dm.values()))
        dcomplete = all(None not in v for v in dr.values())
        dok = dcomplete and dg is not None and dg >= G1_DEC_GMEAN_MIN and all(dm[c] >= G1_DEC_CELL_MIN for c in DEC_CELLS)
        rep.crit("%s.G1.decode.vs%s" % (k, base), dok, "DEC real greedy tok/s ON/%s gmean %s (>= %.3f), cells %s (each >= %.3f)" % (
            base, fmt(dg, 4), G1_DEC_GMEAN_MIN, " ".join("c%d %s" % (c, fmt(dm[c], 4)) for c in DEC_CELLS), G1_DEC_CELL_MIN))
    # ---- G2: cold prefill with eager stores
    cr = [(ratio((rs["ON"][g] or {}).get("cold_toks"), (rs["OFF"][g] or {}).get("cold_toks")),
           ratio((rs["ON"][g] or {}).get("cold64"), (rs["OFF"][g] or {}).get("cold64")), ratio((rs["ON"][g] or {}).get("cold118"), (rs["OFF"][g] or {}).get("cold118"))) for g in groups]
    ct, c64, c118 = mean([x[0] for x in cr]), mean([x[1] for x in cr]), mean([x[2] for x in cr])
    ok = (None not in [x[0] for x in cr] and ct is not None and ct >= G2_COLD_TOKS_MIN and all(x[0] >= G2_COLD_TOKS_GROUP_MIN for x in cr)
          and c64 is not None and c64 <= G2_COLD_TTFT_MAX and c118 is not None and c118 <= G2_COLD_TTFT_MAX)
    rep.crit("%s.G2.cold" % k, ok, "cold prefill with eager stores: pass-1 prompt tok/s ON/OFF %s (>= %.2f pooled, each group >= %.2f; groups %s), 64k cold TTFT x%s, 118k x%s (<= %.2f)" % (
        fmt(ct, 4), G2_COLD_TOKS_MIN, G2_COLD_TOKS_GROUP_MIN, " ".join(fmt(x[0], 4) for x in cr), fmt(c64, 4), fmt(c118, 4), G2_COLD_TTFT_MAX))
    # ---- G3: host memory / boot
    off_ready = [boots["OFF"][g].ready_s() for g in groups if boots["OFF"][g].ready_s() is not None]
    lim = (statistics.median(off_ready) + G3_READY_PLUS_S) if off_ready else None
    on_ready = [boots["ON"][g].ready_s() for g in groups]
    all_boots = [b for arm in boots.values() for b in arm.values()]
    ooms = [b.oom_lines() for b in all_boots if b.exists]
    wds = [b.watchdog_trips() for b in all_boots if b.exists]
    avail = [boots["ON"][g].min_avail() for g in groups]
    ok = (lim is not None and None not in on_ready and all(r <= lim for r in on_ready) and all((o or 0) == 0 for o in ooms) and all((w or 0) == 0 for w in wds)
          and None not in avail and all(a >= G3_MIN_AVAIL_MIB for a in avail))
    rep.crit("%s.G3.host" % k, ok, "READY s ON %s vs OFF median %s (+%.0f s limit %s); OOM lines %s; watchdog trips %s; min MemFree+file cache of the GPU nodes in the ON runs %s MiB (>= %.0f); READY CTL %s" % (
        on_ready, fmt(statistics.median(off_ready) if off_ready else None, 0), G3_READY_PLUS_S, fmt(lim, 0), sum(o or 0 for o in ooms), sum(w or 0 for w in wds), avail, G3_MIN_AVAIL_MIB,
        [boots["CTL"][g].ready_s() for g in groups]))
    # ---- G4: GPU memory at READY
    gu = [(boots["ON"][g].gpu_used(), boots["OFF"][g].gpu_used()) for g in groups]
    deltas = [(a - b) if (a is not None and b is not None) else None for a, b in gu]
    ok = None not in deltas and all(d <= G4_GPU_DELTA_MIB for d in deltas)
    rep.crit("%s.G4.gpu" % k, ok, "GPU used at READY ON - OFF MiB per group %s (<= %.0f); the S1-7 stress of the ON bytes decides S_live (criterion %s.X)" % (deltas, G4_GPU_DELTA_MIB, k))
    # ---- G5: correctness
    failed = sum((rs["ON"][g] or {}).get("failed", 0) + (rs["OFF"][g] or {}).get("failed", 0) for g in groups)
    bad_dec = 0
    for g in groups:
        for arm in ("ON", "OFF", "CTL"):
            for c in DEC_CELLS:
                d = boots[arm][g].dec(c)
                bad_dec += d[1] if d else 0
    steady_missing = sum(1 for g in groups for arm in ("ON", "OFF", "CTL") for c in STEADY_CELLS if boots[arm][g].steady(c) is None)
    pre = [boots["ON"][g].preemptions() for g in groups]
    ok = failed == 0 and bad_dec == 0 and steady_missing == 0 and all((p or 0) == 0 for p in pre)
    rep.crit("%s.G5.correct" % k, ok, "failed reuse requests %d, failed DEC requests %d, missing STEADY cells %d, preemptions in the ON boots %s (all must be 0)" % (failed, bad_dec, steady_missing, pre))


def read_stress(dirs, floor):
    out = []
    for d in dirs:
        s = jload(os.path.join(d, "x", "summary.json"))
        if not s:
            out.append((os.path.basename(d), None, False))
            continue
        sl = s.get("slack_live_mib")
        clean = (not s.get("fails")) and s.get("health_after") == "200" and sl is not None and not s.get("probe_ooms") and not s.get("stress_phase_failures") \
            and s.get("preempt_log_lines", 0) == 0
        out.append((os.path.basename(d), sl, bool(clean and sl >= floor), s.get("kv_bytes")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/raid/vllm-runtime/musetest/results")
    ap.add_argument("--prefix", default="s110g")
    ap.add_argument("--groups", default="1,2,3")
    ap.add_argument("--profiles", default="spec,ns256")
    ap.add_argument("--det", default="", help="profile=FILE,... each FILE holds PASS / FAIL / ARTIFACT")
    ap.add_argument("--preempt", default="", help="result dir of the preemption smoke boot")
    ap.add_argument("--fulltier", default="", help="result dir of the full-tier boot")
    ap.add_argument("--stress", default="", help="profile=DIR[+DIR],... the final stress boots of the ON bytes")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    groups = [int(x) for x in a.groups.split(",") if x]
    rep = Report()
    for prof in [x for x in a.profiles.split(",") if x]:
        rep.lines.append("==== profile %s" % prof)
        judge_profile(a.root, a.prefix, prof, groups, rep)
    for item in [x for x in a.det.split(",") if x]:
        prof, f = item.split("=", 1)
        try:
            v = open(f).read().strip()
        except OSError:
            v = "MISSING"
        rep.crit("%s.D.determinism" % prof, v == "PASS", "e2e bitwise vs offload off with prefix reuse: %s (%s)" % (v, f))
    if a.preempt:
        s = jload(os.path.join(a.preempt, "preempt_smoke.json")) or {}
        rep.crit("spec.S.preemption", s.get("verdict") == "PASS", "forced-preemption smoke: %s, preemptions %s, comparable pairs %s equal %s, failed %s, reasons %s" % (
            s.get("verdict"), s.get("preemptions"), s.get("comparable"), s.get("equal"), s.get("failed"), s.get("reasons")))
    if a.fulltier:
        s = jload(os.path.join(a.fulltier, "reuse_fulltier.json")) or {}
        tc = s.get("tiercheck") or {}
        bad = sum(1 for x in s.get("sends", []) if x.get("status") != 200)
        ok = bool(tc) and bad == 0 and tc.get("comparable_pairs", 0) >= 6 and tc.get("equal_pairs") == tc.get("comparable_pairs")
        ps = s.get("passes", {})
        rep.crit("spec.T.fulltier", ok, "tier-full regime (%s docs over the tier): failed requests %d, tiercheck comparable %s equal %s; hit fraction of passes 2,3: %s; TTFT p50 %s" % (
            (s.get("cycle") or {}).get("docs"), bad, tc.get("comparable_pairs"), tc.get("equal_pairs"),
            [round((ps[n]["metric_delta"].get("vllm:external_prefix_cache_hits_total", 0) / max(1.0, ps[n]["metric_delta"].get("vllm:external_prefix_cache_queries_total", 1))), 3) for n in sorted(ps) if n != "cycle_p1"],
            [ps[n].get("ttft_p50") for n in sorted(ps)]))
    for item in [x for x in a.stress.split(",") if x]:
        prof, dirs = item.split("=", 1)
        res = read_stress(dirs.split("+"), STRESS_FLOOR[prof])
        rep.crit("%s.X.stress" % prof, bool(res) and all(r[2] for r in res), "S1-7 stress at the ON bytes (floor %.0f MiB): %s" % (STRESS_FLOOR[prof], res))
    print("\n".join(rep.lines))
    r = rep.results
    spec_keys = [k for k in r if k.startswith("spec.")]
    ns_keys = [k for k in r if k.startswith("ns256.")]
    adopt_spec = bool(spec_keys) and all(r[k] for k in spec_keys)
    adopt_ns = bool(ns_keys) and all(r[k] for k in ns_keys)
    print("VERDICT ADOPT_SPEC=%s ADOPT_NS256=%s failed=%s" % ("yes" if adopt_spec else "no", "yes" if adopt_ns else "no", [k for k in r if not r[k]]))
    if a.json:
        json.dump({"results": r, "lines": rep.lines, "adopt_spec": adopt_spec, "adopt_ns256": adopt_ns}, open(a.json, "w"), indent=1)
    return 0


def selftest():
    """Build fake boot directories (a passing ON, then a slow ON) and check that the verdict flips as it must."""
    root = tempfile.mkdtemp(prefix="vg_")

    def mk(arm, g, step_scale=1.0, tps_scale=1.0, reuse_good=True, hit=0.99):
        d = os.path.join(root, "s110g_spec_%s_%d" % (arm, g)); os.makedirs(d + "/steady", exist_ok=True)
        open(d + "/status.txt", "w").write("2026-01-01 00:00:00 READY load_s=%d\nDONE unsloth\n" % (100 if arm == "ON" else 90))
        open(d + "/gpu_mem_ready.txt", "w").write("".join("%d, %d MiB\n" % (i, 94141 + (48 if arm == "ON" else 0)) for i in range(8)))
        open(d + "/serve.log", "w").write("ok\n")
        open(d + "/node_mem_ts.txt", "w").write("".join("00:00:00 node%d free=%d file=%d anon=1 shmem=1\n" % (n, 5000, 20000) for n in GPU_NODES))
        for c, base in zip(STEADY_CELLS, (17.3, 55.0, 104.0, 155.0)):
            json.dump({"values": [base * step_scale] * 2, "failed": 0}, open(d + "/steady/steady_s_c%d.json" % c, "w"))
        for rep_ in (1, 2):
            for c, base in zip(DEC_CELLS, (228.0, 760.0, 1250.0, 1500.0)):
                json.dump({"output_throughput": base * tps_scale, "failed": 0}, open(d + "/real_r%d_c%d.json" % (rep_, c), "w"))
        on = arm == "ON"
        sends = []
        for p in (1, 2, 3):
            for i in range(10):
                t = (0.25 if (on and p > 1 and reuse_good) else 5.5) + 0.01 * i
                sends.append({"label": "doc%d_p%d" % (i, p), "pass": "cycle_p%d" % p, "ttft_s": t, "status": 200})
        for L, tc, tr in ((65536, 6.3, 0.29 if on else 6.3), (117964, 12.4, 0.47 if on else 12.4)):
            sends.append({"label": "ab_cold_%d" % L, "pass": "single", "ttft_s": tc, "status": 200})
            sends.append({"label": "ab_readmit_%d" % L, "pass": "single", "ttft_s": tr, "status": 200})
        passes = {}
        for p in (1, 2, 3):
            fast = on and p > 1 and reuse_good
            passes["cycle_p%d" % p] = {"prompt_tokens": 2_600_000, "wall_s": 12.0 if fast else 241.0, "tok_per_s": 2_600_000 / (12.0 if fast else 241.0),
                                       "metric_delta": {"vllm:external_prefix_cache_hits_total": 2_600_000 * hit if fast else 0.0, "vllm:external_prefix_cache_queries_total": 2_600_000.0}}
        json.dump({"passes": passes, "sends": sends}, open(d + "/reuse_cycle.json", "w"))
        open(d + "/metrics_after_reuse.txt", "w").write("vllm:num_preemptions_total{engine=\"0\"} 0.0\n")

    for g in (1, 2, 3):
        mk("CTL", g); mk("OFF", g); mk("ON", g)
    rep = Report(); judge_profile(root, "s110g", "spec", [1, 2, 3], rep)
    good = all(rep.results[k] for k in rep.results)
    print("\n".join(rep.lines)); print("SELFTEST good case all pass: %s" % good)
    ok = good
    for g in (1, 2, 3):
        mk("ON", g, step_scale=1.03)
    rep2 = Report(); judge_profile(root, "s110g", "spec", [1, 2, 3], rep2)
    ok = ok and (not rep2.results["spec.G1.steady.vsOFF"]) and rep2.results["spec.primary"]
    print("SELFTEST slow-steady case flips G1.steady only: %s" % ((not rep2.results["spec.G1.steady.vsOFF"]) and rep2.results["spec.primary"]))
    for g in (1, 2, 3):
        mk("ON", g, reuse_good=False)
    rep3 = Report(); judge_profile(root, "s110g", "spec", [1, 2, 3], rep3)
    ok = ok and not rep3.results["spec.primary"]
    print("SELFTEST no-benefit case fails the primary: %s" % (not rep3.results["spec.primary"]))
    for g in (1, 2, 3):
        mk("ON", g, hit=0.5)
    rep4 = Report(); judge_profile(root, "s110g", "spec", [1, 2, 3], rep4)
    ok = ok and not rep4.results["spec.primary"]
    print("SELFTEST low hit fraction fails the primary: %s" % (not rep4.results["spec.primary"]))
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
