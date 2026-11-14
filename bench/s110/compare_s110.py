#!/usr/bin/env python3
"""S1-10 adoption-rule metrics of connector arms against the OFF arm of the same bind policy (stdlib only; python3 -I).

usage: compare_s110.py --off TAG[,TAG2] [--off-dec TAG[,TAG2]] --arm TAG[,TAG2] [--arm ...] [--results DIR] [--json OUT]
A comma list pools boots of one arm (retries / repeats). Metrics (ADOPTION_RULE.txt):
  DEC   geometric mean over the decode cells of tok/s(arm) / tok/s(off) (cells = real_r*_c<c>, mean of the reps and boots of an arm), the minimum cell ratio
  REU   passes 2+ pooled (per-request TTFT p50 / p99 over every request of the passes, prompt tok/s = sum tokens / sum wall), ratios arm / off
  AB    TTFT of ab_readmit_* (re-admission of the 64k / 118k prompts) arm / off
  HIT   (local + external) prefix-cache hit tokens / queried tokens over passes 2+ ; external part alone
  BOOT  READY seconds, GPU idle used MiB, per-node peak anon+shmem
"""
import glob
import json
import math
import os
import re
import statistics
import sys


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(math.ceil(p / 100.0 * len(s))) - 1))]


def arm_data(root, tags, parts=("dec", "reu")):
    dec = {}
    reu_t, reu_tok, reu_wall = [], 0, 0.0
    ab = {}
    hit = {"local": 0.0, "ext": 0.0, "queried": 0.0, "prompt": 0.0}
    boot, gpu, peaks = [], [], {}
    for tag in tags:
        d = os.path.join(root, tag)
        st = open(d + "/status.txt", errors="replace").read() if os.path.exists(d + "/status.txt") else ""
        m = re.findall(r"READY load_s=(\d+)", st)
        if m:
            boot.append(int(m[-1]))
        g = re.findall(r"^\d+, (\d+) MiB", open(d + "/gpu_mem_ready.txt").read(), re.M) if os.path.exists(d + "/gpu_mem_ready.txt") else []
        gpu += [int(x) for x in g]
        for line in (open(d + "/node_peaks.txt").read().splitlines() if os.path.exists(d + "/node_peaks.txt") else []):
            mm = re.match(r"(node\d) peak_anon_plus_shmem_mib=(\d+)", line)
            if mm and mm.group(1) in ("node0", "node3", "node4", "node7"):
                peaks[mm.group(1)] = max(peaks.get(mm.group(1), 0), int(mm.group(2)))
        for f in (glob.glob(d + "/real_r*_c*.json") if "dec" in parts else []):
            mm = re.search(r"real_r\d+_c(\d+)\.json$", f)
            j = jl(f)
            if mm and j and j.get("failed", 1) == 0:
                dec.setdefault(int(mm.group(1)), []).append(j["output_throughput"])
        j = jl(d + "/reuse_cycle.json") if "reu" in parts else None
        if j:
            for s in j.get("sends", []):
                if re.match(r"doc\d+_p([2-9])$", s["label"]) and s["status"] == 200:
                    reu_t.append(s["ttft_s"])
                if s["label"].startswith("ab_readmit_") and s["status"] == 200:
                    ab.setdefault(s["label"], []).append(s["ttft_s"])
            for k, v in j.get("passes", {}).items():
                mm = re.match(r"cycle_p(\d+)$", k)
                if mm and int(mm.group(1)) >= 2:
                    reu_tok += v["prompt_tokens"]
                    reu_wall += v["wall_s"]
                    md = v.get("metric_delta", {})
                    for mk, mv in md.items():
                        if mk.startswith("vllm:prefix_cache_hits_total"):
                            hit["local"] += mv
                        elif mk.startswith("vllm:external_prefix_cache_hits_total"):
                            hit["ext"] += mv
                        elif mk.startswith("vllm:prefix_cache_queries_total"):
                            hit["queried"] += mv
                    hit["prompt"] += v["prompt_tokens"]
    return {"dec": {c: statistics.mean(v) for c, v in dec.items()}, "reu_p50": pct(reu_t, 50), "reu_p99": pct(reu_t, 99), "reu_n": len(reu_t),
            "reu_tps": (reu_tok / reu_wall) if reu_wall else None, "ab": {k: statistics.mean(v) for k, v in ab.items()}, "hit": hit,
            "boot_s": boot, "gpu_idle_used": sorted(set(gpu)), "peaks": peaks}


def main():
    a = sys.argv[1:]
    root = "/raid/vllm-runtime/musetest/results"
    out = None
    off = None
    off_dec = None
    arms = []
    i = 0
    while i < len(a):
        if a[i] == "--results":
            root = a[i + 1]; i += 2
        elif a[i] == "--json":
            out = a[i + 1]; i += 2
        elif a[i] == "--off":
            off = a[i + 1].split(","); i += 2
        elif a[i] == "--off-dec":
            off_dec = a[i + 1].split(","); i += 2
        elif a[i] == "--arm":
            arms.append(a[i + 1].split(",")); i += 2
        else:
            raise SystemExit("bad arg %s" % a[i])
    o = arm_data(root, off, ("reu",) if off_dec else ("dec", "reu"))
    if off_dec:  # the decode baseline of the OFF arm comes from other boots (same protocol as the arms)
        od = arm_data(root, off_dec, ("dec",))
        o["dec"] = od["dec"]; o["boot_s"] = od["boot_s"] or o["boot_s"]; o["gpu_idle_used"] = od["gpu_idle_used"] or o["gpu_idle_used"]
    res = {"off": {"tags": off, **o}, "arms": []}
    print("OFF %s (dec from %s) boot=%s gpu_idle=%s DEC=%s REU p50=%s p99=%s tps=%s n=%s AB=%s" % (
        ",".join(off), ",".join(off_dec or off), o["boot_s"], o["gpu_idle_used"], {c: round(v, 1) for c, v in sorted(o["dec"].items())}, o["reu_p50"], o["reu_p99"],
        o["reu_tps"] and round(o["reu_tps"], 1), o["reu_n"], {k: round(v, 3) for k, v in o["ab"].items()}))
    for tags in arms:
        x = arm_data(root, tags)
        cells = sorted(set(x["dec"]) & set(o["dec"]))
        ratios = {c: x["dec"][c] / o["dec"][c] for c in cells}
        gm = math.exp(sum(math.log(r) for r in ratios.values()) / len(ratios)) if ratios else None
        r = {"tags": tags, "dec_ratio": {c: round(v, 4) for c, v in ratios.items()}, "dec_gmean": gm and round(gm, 4),
             "dec_min": min(ratios.values()) if ratios else None,
             "reu_p50_ratio": x["reu_p50"] / o["reu_p50"] if x["reu_p50"] and o["reu_p50"] else None,
             "reu_p99_ratio": x["reu_p99"] / o["reu_p99"] if x["reu_p99"] and o["reu_p99"] else None,
             "reu_tps_ratio": x["reu_tps"] / o["reu_tps"] if x["reu_tps"] and o["reu_tps"] else None,
             "ab_ratio": {k: x["ab"][k] / o["ab"][k] for k in x["ab"] if k in o["ab"]},
             "hit_total": (x["hit"]["local"] + x["hit"]["ext"]) / x["hit"]["queried"] if x["hit"]["queried"] else None,
             "hit_ext_over_prompt": x["hit"]["ext"] / x["hit"]["prompt"] if x["hit"]["prompt"] else None,
             "boot_s": x["boot_s"], "gpu_idle_used": x["gpu_idle_used"], "peaks": x["peaks"], "reu_p50": x["reu_p50"], "reu_p99": x["reu_p99"],
             "reu_tps": x["reu_tps"], "ab": x["ab"], "dec": x["dec"]}
        res["arms"].append(r)
        print("ARM %s: DEC gmean=%s min=%s cells=%s" % (",".join(tags), r["dec_gmean"], r["dec_min"] and round(r["dec_min"], 4), r["dec_ratio"]))
        print("     REU p50=%s (x%s) p99=%s (x%s) tps=%s (x%s) n=%s AB ratio=%s HIT total=%s ext/prompt=%s" % (
            x["reu_p50"], r["reu_p50_ratio"] and round(r["reu_p50_ratio"], 3), x["reu_p99"], r["reu_p99_ratio"] and round(r["reu_p99_ratio"], 3),
            x["reu_tps"] and round(x["reu_tps"], 1), r["reu_tps_ratio"] and round(r["reu_tps_ratio"], 3), x["reu_n"],
            {k: round(v, 3) for k, v in r["ab_ratio"].items()}, r["hit_total"] and round(r["hit_total"], 3),
            r["hit_ext_over_prompt"] and round(r["hit_ext_over_prompt"], 3)))
        print("     BOOT ready_s=%s gpu_idle_used=%s (off %s) node peaks=%s" % (x["boot_s"], x["gpu_idle_used"], o["gpu_idle_used"], x["peaks"]))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
