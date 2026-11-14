"""Markdown tables from bench_h1b.py result files (jsonl) for docs/h1b_latency.md.
usage: python3 summarize_h1b.py results_ar.jsonl results_rs.jsonl results_ag.jsonl [--all]
Per operation and size: NCCL, the dispatched topo path (what the plugin calls), speedup, and for all_reduce the H1a table
path (decode/medium algorithms) and the best single large variant; the opt-in bf16-exchange numbers (class
deterministic_different) are listed in their own column. --all adds a table of every measured variant."""
import collections
import json
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
show_all = "--all" in sys.argv
rows = []
for p in args:
    rows += [json.loads(l) for l in open(p) if l.strip()]


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def cfg_str(c):
    return "-" if not c else ",".join(f"{k}={v}" for k, v in c.items() if k not in ("pdl", "ramp") or v)


def size_str(m):
    return f"{m:g}"


def table(op, keyname, title, small_variant=None):
    sel = [r for r in rows if r.get("op") == op]
    if not sel:
        return
    print(f"\n### {title}\n")
    by = collections.defaultdict(list)
    for r in sel:
        by[(r.get(keyname), r.get("layout"))].append(r)
    hdr = ["size (MiB)"] + (["layout"] if op == "all_gather" else []) + [
        "NCCL b2b us", "NCCL spacer us", "topo b2b us", "topo spacer us", "speedup b2b", "speedup spacer"]
    if small_variant:
        hdr.append(f"{small_variant} us")
    hdr += ["bf16 exchange us (opt-in)", "bitwise ok"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for (mib, layout), rs in sorted(by.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))):
        if op == "all_gather" and layout is None:
            continue
        nccl = next((r["nccl_b2b_us"] for r in rs if "nccl_b2b_us" in r), None)
        nsp = next((r.get("nccl_spacer_us") for r in rs if "nccl_spacer_us" in r), None)
        disp = next((r for r in rs if r.get("variant") == "dispatch"), None)
        if disp is None and op == "reduce_scatter":  # the public reduce_scatter path is the 'large' variant with table config
            disp = next((r for r in rs if r.get("variant") == "large_default"), None)
        small = next((r for r in rs if r.get("variant") == small_variant), None) if small_variant else None
        bx = [r for r in rs if "bx" in r.get("variant", "") and "topo_b2b_us" in r]
        bxb = min(bx, key=lambda r: r["topo_b2b_us"])["topo_b2b_us"] if bx else None
        ok = None if disp is None else disp.get("bitwise_contract", disp.get("bitwise"))
        t = disp["topo_b2b_us"] if disp else None
        ts = disp.get("topo_spacer_us") if disp else None
        line = [size_str(mib)] + ([layout] if op == "all_gather" else []) + [
            fmt(nccl), fmt(nsp), fmt(t), fmt(ts), (fmt(nccl / t, 2) + "x") if (nccl and t) else "-",
            (fmt(nsp / ts, 2) + "x") if (nsp and ts) else "-"]
        if small_variant:
            line.append(fmt(small["topo_b2b_us"]) if small else "-")
        line += [fmt(bxb), str(ok)]
        print("| " + " | ".join(line) + " |")
    if show_all:
        print("\nAll variants:\n")
        print("| size | layout | variant | b2b us | spacer us | cfg | bitwise |")
        print("|---|---|---|---|---|---|---|")
        for (mib, layout), rs in sorted(by.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))):
            for r in rs:
                if "variant" in r and "topo_b2b_us" in r:
                    print(f"| {size_str(mib)} | {layout or '-'} | {r['variant']} | {fmt(r['topo_b2b_us'])} | {fmt(r.get('topo_spacer_us'))} | "
                          f"{cfg_str(r.get('cfg'))} | {r.get('bitwise_contract', r.get('bitwise', '-'))} |")


table("all_reduce", "mib", "all_reduce bf16 [T, 8192] (CUDA graph, us per call; T = 64 x MiB)", small_variant="h1a_table")
table("reduce_scatter", "mib", "reduce_scatter bf16 [T, 8192] -> [T/8, 8192] (CUDA graph, us per call)")
table("all_gather", "out_mib", "all_gather bf16, size = OUTPUT MiB (CUDA graph, us per call)")
