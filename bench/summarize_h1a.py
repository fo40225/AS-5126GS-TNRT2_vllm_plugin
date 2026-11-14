"""Render the jsonl produced by bench_h1a.py as markdown tables. usage: python3 summarize_h1a.py <results.jsonl> [more.jsonl ...]"""
import collections
import json
import sys

rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p) if l.strip()]
null = [r for r in rows if r.get("op") == "null_kernel"]
if null:
    print(f"null kernel back-to-back in a graph: {null[-1]['b2b_us']:.2f} us; 50 us sleep kernel measures {null[-1]['spacer50_us']:.2f} us\n")

ar = collections.defaultdict(list)
for r in rows:
    if r.get("op") == "all_reduce" and "topo_algo" in r:
        ar[tuple(r["shape"])].append(r)
print("## all_reduce bf16 (us per call, CUDA graph; b2b = back-to-back, spacer = exposed time behind a 50 us kernel)\n")
print("| shape | bytes | NCCL b2b | NCCL spacer | best topo algo (grid/block/unroll) | topo b2b | topo spacer | speedup (spacer) | bitwise contract |")
print("|---|---|---|---|---|---|---|---|---|")
for shape in sorted(ar, key=lambda s: (-s[1], s[0])):
    best = min(ar[shape], key=lambda r: r["topo_b2b_us"])
    c = best["topo_cfg"]
    others = " ; ".join(f"alg{r['topo_algo']} {r['topo_b2b_us']:.1f}" for r in sorted(ar[shape], key=lambda r: r["topo_algo"]))
    print(f"| {shape} | {best['bytes']} | {best['nccl_b2b_us']:.1f} | {best['nccl_spacer_us']:.1f} | alg{best['topo_algo']} "
          f"({c['grid']}/{c['block']}/{c['unroll']}) [{others}] | {best['topo_b2b_us']:.2f} | {best['topo_spacer_us']:.2f} | "
          f"{best['nccl_spacer_us'] / max(best['topo_spacer_us'], 1e-9):.1f}x | {best.get('bitwise_contract')} |")
ag = collections.defaultdict(list)
for r in rows:
    if r.get("op") == "all_gather" and "topo_algo" in r:
        ag[(tuple(r["shape"]), r["layout"])].append(r)
print("\n## all_gather bf16 ([T,w] -> [T,8w] for dim-1, [8T,w] for dim0)\n")
print("| shard shape | layout | shard bytes | NCCL b2b | NCCL spacer | best topo algo (grid/block/unroll) | topo b2b | topo spacer | speedup (spacer) |")
print("|---|---|---|---|---|---|---|---|---|")
for key in sorted(ag, key=lambda k: (k[0][1], k[0][0], k[1])):
    best = min(ag[key], key=lambda r: r["topo_b2b_us"])
    c = best["topo_cfg"]
    others = " ; ".join(f"alg{r['topo_algo']} {r['topo_b2b_us']:.1f}" for r in sorted(ag[key], key=lambda r: r["topo_algo"]))
    print(f"| {key[0]} | {key[1]} | {best['shard_bytes']} | {best['nccl_b2b_us']:.1f} | {best['nccl_spacer_us']:.1f} | "
          f"alg{best['topo_algo']} ({c['grid']}/{c['block']}/{c['unroll']}) [{others}] | {best['topo_b2b_us']:.2f} | "
          f"{best['topo_spacer_us']:.2f} | {best['nccl_spacer_us'] / max(best['topo_spacer_us'], 1e-9):.1f}x |")
