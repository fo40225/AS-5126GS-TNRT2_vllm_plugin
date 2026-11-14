"""Latency of the large-message collectives of muse_topo vs NCCL (H1b).

usage: py_topo.sh bench/bench_h1b.py --out <jsonl> [--what ar,rs,ag] [--mibs 1,2,4,8,16,32,64,128] [--sweep]
Protocol (same as bench_h1a.py): CUDA graph of N identical calls (N = 20 below 8 MiB, 6 from 8 MiB), replayed; per-call
time = replay time / N, median over trials, max over ranks. "b2b" = back-to-back, "spacer" = (50 us sleep kernel, call)
pairs minus the sleep-only time. Input randn bf16 [T, 8192]; all_gather sizes are the OUTPUT bytes ([T/8,8192] shards
-> [T,8192], i.e. the unpooler/pooler gathers of a prefill chunk are 128 / 64 MiB).
Baseline: vLLM's PyNcclCommunicator (all_reduce, reduce_scatter, all_gather), NCCL_P2P_LEVEL=SYS.
--sweep additionally times configuration candidates (chunk bytes, blocks, threads, unroll, mode); every candidate is
checked bitwise against the dispatched path before it is timed.
"""
import argparse
import itertools
import sys
import traceback

import torch
import torch.multiprocessing as mp

from bench_h1a import timed, spacer_time, pair_time
from common import *  # noqa: F401,F403


def ncalls(nbytes):
    return 20 if nbytes < 8 * MiB else 6


def fmt(r):
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("shape",)}


def agree(ok, cpu_group):
    return all(gather_all(bool(ok), cpu_group))


def descend(evaluate, start, space, rounds=2):
    """Coordinate descent: vary one parameter at a time over its candidates, keep improvements (>1%), repeat."""
    best_cfg, best = dict(start), evaluate(start)
    seen = {tuple(sorted(start.items())): best}
    for _ in range(rounds):
        improved = False
        for key, values in space.items():
            for v in values:
                cfg = dict(best_cfg, **{key: v})
                tk = tuple(sorted(cfg.items()))
                if tk in seen:
                    continue
                seen[tk] = evaluate(cfg)
                if seen[tk] < 0.99 * best:
                    best_cfg, best, improved = cfg, seen[tk], True
        if not improved:
            break
    return best_cfg, best, sorted(((us, dict(k)) for k, us in seen.items()), key=lambda t: t[0])


AR_SPACE = dict(mode=(0, 1), chunk_bytes=(1 << 18, 1 << 19, 1 << 20), rs_grid=(1, 2, 3, 4, 6, 8), ag_grid=(2, 4, 6, 8, 12, 16, 32),
                block=(256, 512), unroll=(1, 2), cut_rs=(0, 1), cut_ag=(0, 1, 2), ramp=(0, 1, 2), pdl=(0, 1))
RS_SPACE = dict(chunk_bytes=(1 << 18, 1 << 19, 1 << 20), grid=(1, 2, 3, 4, 6, 8, 12), block=(128, 256, 512), unroll=(1, 2), cut=(0, 1))
AG_SPACE = dict(chunk_bytes=(1 << 18, 1 << 19, 1 << 20), grid=(2, 4, 6, 8, 12, 16, 32), block=(256, 512, 1024), cut=(0, 1))


def ar_candidates(args, nbytes):
    if not args.sweep:
        return []
    out = []
    for ch, rg, ag, blk, un, (crs, cag) in itertools.product((1 << 20,), (2, 4, 8, 16), (2, 4, 8, 24, 64), (256, 512), (1, 2),
                                                             ((0, 0), (1, 0), (0, 1), (1, 1))):
        out.append(dict(mode=1, chunk_bytes=ch, rs_grid=rg, ag_grid=ag, block=blk, unroll=un, cut_rs=crs, cut_ag=cag))
    for ch, rg, ag, crs, cag in itertools.product((1 << 19,), (4, 8), (24,), (0, 1), (0, 1)):
        out.append(dict(mode=1, chunk_bytes=ch, rs_grid=rg, ag_grid=ag, block=512, unroll=1, cut_rs=crs, cut_ag=cag))
    for rg, crs, cag in itertools.product((4, 8), (0, 1), (0, 1)):
        out.append(dict(mode=0, chunk_bytes=1 << 20, rs_grid=rg, ag_grid=32, block=512, unroll=1, cut_rs=crs, cut_ag=cag))
    return out


def rs_candidates(args, nbytes):
    if not args.sweep:
        return []
    return [dict(chunk_bytes=ch, grid=g, block=blk, unroll=un, cut=cut)
            for ch, g, blk, un, cut in itertools.product((1 << 19, 1 << 20), (2, 4, 8, 16, 32), (256, 512), (1, 2), (0, 1))]


def ag_candidates(args, nbytes):
    if not args.sweep:
        return []
    return [dict(chunk_bytes=ch, grid=g, block=blk, cut=cut)
            for ch, g, blk, cut in itertools.product((1 << 19, 1 << 20), (2, 4, 8, 16, 32, 64), (256, 512, 1024), (0, 1))]


def bench_ar(rank, comm, nccl, cpu_group, dev, args, mib, spacer_cycles, s_only):
    T = int(mib * MiB) // (8192 * 2)
    shape = (T, 8192)
    nbytes = T * 8192 * 2
    xs = [make_input(r, shape, 55, dev, torch.bfloat16, "randn") for r in range(WORLD)]
    x = xs[rank]
    ref = contract_ref(xs, comm.info.members)
    del xs
    out = torch.empty_like(x)
    n = ncalls(nbytes)
    row = {"op": "all_reduce", "mib": mib, "shape": list(shape), "bytes": nbytes}
    nout = torch.empty_like(x)
    row["nccl_b2b_us"] = timed(lambda: nccl.all_reduce(x, nout), n, cpu_group)
    if nbytes <= 16 * MiB:
        row["nccl_spacer_us"] = pair_time(lambda: nccl.all_reduce(x, nout), n, spacer_cycles, s_only, cpu_group)
    rows = []
    # dispatched path (what the plugin calls)
    out.zero_()
    comm.all_reduce(x, out=out)
    torch.cuda.synchronize()
    ok = agree(bits_equal(out, ref), cpu_group)
    call = lambda: comm.all_reduce(x, out=out)  # noqa: E731
    r = dict(row, variant="dispatch", bitwise_contract=ok, topo_b2b_us=timed(call, n, cpu_group))
    if nbytes <= 16 * MiB:
        r["topo_spacer_us"] = pair_time(call, n, spacer_cycles, s_only, cpu_group)
    rows.append(r)
    if nbytes <= comm.max_bytes:  # the H1a table path (decode/medium algorithms), for the crossover
        from muse_topo.collectives import _AR_TABLE, _pick, MAX_GRID

        algo, grid, block, unroll = _pick(_AR_TABLE, nbytes)[1:]
        a2 = lambda: comm._all_reduce(x, algo, block, unroll, MAX_GRID, grid or None, out)  # noqa: E731
        rows.append(dict(row, variant="h1a_table", topo_b2b_us=timed(a2, n, cpu_group)))
    if nbytes > comm.max_bytes:  # the baseline without the new kernels: H1a algorithm 2 on 8 MiB pieces
        piece = comm.max_bytes // 2  # elements per piece (bf16)
        xv, ov = x.view(-1), out.view(-1)

        def loop():
            for i in range(0, xv.numel(), piece):
                comm._all_reduce(xv[i:i + piece], 2, 256, 2, grid=4, out=ov[i:i + piece])

        out.zero_()
        loop()
        torch.cuda.synchronize()
        ok = agree(bits_equal(out, ref), cpu_group)
        rows.append(dict(row, variant="h1a_loop_8mib", bitwise_contract=ok, topo_b2b_us=timed(loop, n, cpu_group)))
    # large algorithm, default configuration, both modes and the bf16 exchange
    for name, kw in (("large_fused", dict(mode=1)), ("large_2launch", dict(mode=0)),
                     ("large_fused_cut", dict(mode=1, cut_rs=1, cut_ag=1)),
                     ("large_fused_cutrs", dict(mode=1, cut_rs=1)),
                     ("large_fused_merged", dict(mode=1, rs_grid=4, ag_grid=8, unroll=2, cut_rs=1, cut_ag=2)),
                     ("large_fused_tuned", dict(mode=1, rs_grid=4, ag_grid=8, unroll=2, cut_rs=0, cut_ag=1)),
                     ("large_fused_tuned_bx", dict(mode=1, rs_grid=4, ag_grid=8, unroll=2, cut_rs=0, cut_ag=1, bx=True)),
                     ("large_fused_tuned_ramp1", dict(mode=1, rs_grid=4, ag_grid=8, unroll=2, cut_rs=0, cut_ag=1, ramp=1)),
                     ("large_fused_tuned_ramp2", dict(mode=1, rs_grid=4, ag_grid=8, unroll=2, cut_rs=0, cut_ag=1, ramp=2)),
                     ("large_fused_tuned_pdl", dict(mode=1, rs_grid=4, ag_grid=8, unroll=2, cut_rs=0, cut_ag=1, pdl=1)),
                     ("large_fused_tuned_ramp2_pdl", dict(mode=1, rs_grid=4, ag_grid=8, unroll=2, cut_rs=0, cut_ag=1, ramp=2, pdl=1)),
                     ("large_fused_bx", dict(mode=1, bx=True)), ("large_fused_bx_cut", dict(mode=1, bx=True, cut_rs=1, cut_ag=1))):
        out.zero_()
        comm._all_reduce_large(x, out, **kw)
        torch.cuda.synchronize()
        ok = agree(bits_equal(out, ref), cpu_group) if "bx" not in name else None
        call = lambda kw=kw: comm._all_reduce_large(x, out, **kw)  # noqa: E731
        r = dict(row, variant=name, bitwise_contract=ok, topo_b2b_us=timed(call, n, cpu_group))
        if nbytes <= 16 * MiB and "bx" not in name:
            r["topo_spacer_us"] = pair_time(call, n, spacer_cycles, s_only, cpu_group)
        rows.append(r)
    cands = []
    if args.search:
        def evaluate(kw):
            kw = dict(kw)
            out.zero_()
            comm._all_reduce_large(x, out, **kw)
            torch.cuda.synchronize()
            if not agree(bits_equal(out, ref), cpu_group):
                log(rank, "INVALID ar config", kw, shape)
                return float("inf")
            return timed(lambda: comm._all_reduce_large(x, out, **kw), min(n, 6), cpu_group, quick=True)

        start = dict(mode=1, chunk_bytes=1 << 20, rs_grid=4, ag_grid=8, block=512, unroll=2, cut_rs=0, cut_ag=1, ramp=0, pdl=0)
        space = {k: tuple(v for v in vs if k != "chunk_bytes" or v * 8 <= max(nbytes, 1 << 18) * 2) for k, vs in AR_SPACE.items()}
        _, _, ranked = descend(evaluate, start, space)
        cands = [(us, kw) for us, kw in ranked if us < float("inf")]
    for kw in ([] if args.search else ar_candidates(args, nbytes)):
        out.zero_()
        comm._all_reduce_large(x, out, **kw)
        torch.cuda.synchronize()
        if not agree(bits_equal(out, ref), cpu_group):
            log(rank, "INVALID ar config", kw, shape)
            continue
        us = timed(lambda kw=kw: comm._all_reduce_large(x, out, **kw), min(n, 4), cpu_group, quick=True)
        cands.append((us, kw))
    cands.sort(key=lambda c: c[0])
    for us, kw in cands[:3]:
        rows.append(dict(row, variant="sweep_best", cfg=kw, topo_b2b_us=us))
    return rows


def bench_rs(rank, comm, nccl, cpu_group, dev, args, mib, spacer_cycles, s_only):
    T = max(8, int(mib * MiB) // (8192 * 2) // 8 * 8)
    shape = (T, 8192)
    nbytes = T * 8192 * 2
    xs = [make_input(r, shape, 56, dev, torch.bfloat16, "randn") for r in range(WORLD)]
    x = xs[rank]
    ref = contract_ref(xs, comm.info.members)[rank * (T // WORLD):(rank + 1) * (T // WORLD)]
    del xs
    out = torch.empty((T // WORLD, 8192), dtype=torch.bfloat16, device=dev)
    n = ncalls(nbytes)
    row = {"op": "reduce_scatter", "mib": mib, "shape": list(shape), "bytes": nbytes}
    nout = torch.empty_like(out)
    row["nccl_b2b_us"] = timed(lambda: nccl.reduce_scatter(nout, x), n, cpu_group)
    if nbytes <= 16 * MiB:
        row["nccl_spacer_us"] = pair_time(lambda: nccl.reduce_scatter(nout, x), n, spacer_cycles, s_only, cpu_group)
    rows = []
    if comm.supports("reduce_scatter", x):  # the public path (dispatch table), what the plugin calls
        out.zero_()
        comm.reduce_scatter(x, out=out)
        torch.cuda.synchronize()
        ok = agree(bits_equal(out, ref), cpu_group)
        call = lambda: comm.reduce_scatter(x, out=out)  # noqa: E731
        r = dict(row, variant="dispatch", bitwise_contract=ok, topo_b2b_us=timed(call, n, cpu_group))
        if nbytes <= 16 * MiB:
            r["topo_spacer_us"] = pair_time(call, n, spacer_cycles, s_only, cpu_group)
        rows.append(r)
    for name, kw in (("large", {}), ("large_cut", dict(cut=1)), ("large_bx", dict(bx=True)), ("large_bx_cut", dict(bx=True, cut=1))):
        out.zero_()
        comm._reduce_scatter_large(x, out, **kw)
        torch.cuda.synchronize()
        ok = agree(bits_equal(out, ref), cpu_group) if "bx" not in name else None
        call = lambda kw=kw: comm._reduce_scatter_large(x, out, **kw)  # noqa: E731
        r = dict(row, variant=name, bitwise_contract=ok, topo_b2b_us=timed(call, n, cpu_group))
        if nbytes <= 16 * MiB and "bx" not in name:
            r["topo_spacer_us"] = pair_time(call, n, spacer_cycles, s_only, cpu_group)
        rows.append(r)
    cands = []
    if args.search:
        def evaluate(kw):
            kw = dict(kw)
            out.zero_()
            comm._reduce_scatter_large(x, out, **kw)
            torch.cuda.synchronize()
            if not agree(bits_equal(out, ref), cpu_group):
                log(rank, "INVALID rs config", kw, shape)
                return float("inf")
            return timed(lambda: comm._reduce_scatter_large(x, out, **kw), min(n, 6), cpu_group, quick=True)

        start = dict(chunk_bytes=1 << 20, grid=4, block=256, unroll=1, cut=0)
        space = {k: tuple(v for v in vs if k != "chunk_bytes" or v * 8 <= max(nbytes, 1 << 18) * 2) for k, vs in RS_SPACE.items()}
        _, _, ranked = descend(evaluate, start, space)
        cands = [(us, kw) for us, kw in ranked if us < float("inf")]
    for kw in ([] if args.search else rs_candidates(args, nbytes)):
        out.zero_()
        comm._reduce_scatter_large(x, out, **kw)
        torch.cuda.synchronize()
        if not agree(bits_equal(out, ref), cpu_group):
            log(rank, "INVALID rs config", kw, shape)
            continue
        us = timed(lambda kw=kw: comm._reduce_scatter_large(x, out, **kw), min(n, 4), cpu_group, quick=True)
        cands.append((us, kw))
    cands.sort(key=lambda c: c[0])
    for us, kw in cands[:3]:
        rows.append(dict(row, variant="sweep_best", cfg=kw, topo_b2b_us=us))
    return rows


def bench_ag(rank, comm, nccl, cpu_group, dev, args, mib, w, spacer_cycles, s_only):
    """Output = mib MiB bf16: [8T, w] rank-major (dim 0) or [T, 8w]; shard [T, w], T = mib MiB / (8 * w * 2)."""
    T = max(1, int(mib * MiB) // (8 * w * 2))
    nbytes = T * w * 2  # shard bytes
    xs = [make_input(r, (T, w), 77, dev, torch.bfloat16) for r in range(WORLD)]
    x = xs[rank]
    n = ncalls(nbytes * 8)
    row = {"op": "all_gather", "out_mib": mib, "shape": [T, w], "shard_bytes": nbytes}
    nout = torch.empty((WORLD * T, w), dtype=torch.bfloat16, device=dev)
    row["nccl_b2b_us"] = timed(lambda: nccl.all_gather(nout, x), n, cpu_group)
    if mib <= 16:
        row["nccl_spacer_us"] = pair_time(lambda: nccl.all_gather(nout, x), n, spacer_cycles, s_only, cpu_group)
    rows = []
    for dim, layout, lname in ((0, 1, "dim0"), (-1, 0, "dim-1")):
        ref = torch.cat(xs, dim)
        out = torch.empty_like(ref)
        if not comm.supports("all_gather", x, dim):
            rows.append(dict(row, layout=lname, error="unsupported"))
            continue
        out.zero_()
        comm.all_gather(x, dim, out=out)
        torch.cuda.synchronize()
        ok = agree(bits_equal(out, ref), cpu_group)
        call = lambda: comm.all_gather(x, dim, out=out)  # noqa: E731
        r = dict(row, layout=lname, variant="dispatch", bitwise=ok, topo_b2b_us=timed(call, n, cpu_group))
        if mib <= 16:
            r["topo_spacer_us"] = pair_time(call, n, spacer_cycles, s_only, cpu_group)
        rows.append(r)
        if nbytes > 16 and nbytes % 16 == 0 and (layout == 1 or (w * 2) % 16 == 0):  # large engine explicitly (also below the dispatch threshold, for the crossover)
            for vname, kw in (("large_default", {}), ("large_cut", dict(cut=1))):
                out.zero_()
                comm._all_gather_large(x, layout, out, **kw)
                torch.cuda.synchronize()
                ok = agree(bits_equal(out, ref), cpu_group)
                call = lambda kw=kw: comm._all_gather_large(x, layout, out, **kw)  # noqa: E731
                rows.append(dict(row, layout=lname, variant=vname, bitwise=ok, topo_b2b_us=timed(call, n, cpu_group)))
        cands = []
        if layout == 1 and args.search:
            def evaluate(kw):
                kw = dict(kw)
                out.zero_()
                comm._all_gather_large(x, layout, out, **kw)
                torch.cuda.synchronize()
                if not agree(bits_equal(out, ref), cpu_group):
                    log(rank, "INVALID ag config", kw, (T, w))
                    return float("inf")
                return timed(lambda: comm._all_gather_large(x, layout, out, **kw), min(n, 6), cpu_group, quick=True)

            start = dict(chunk_bytes=1 << 19, grid=8, block=1024, cut=1)
            space = {k: tuple(v for v in vs if k != "chunk_bytes" or v <= max(nbytes, 1 << 18) * 2) for k, vs in AG_SPACE.items()}
            _, _, ranked = descend(evaluate, start, space)
            cands = [(us, kw) for us, kw in ranked if us < float("inf")]
            for us, kw in cands[:3]:
                rows.append(dict(row, layout=lname, variant="sweep_best", cfg=kw, topo_b2b_us=us))
        elif layout == 1:  # sweep on one layout (the engine is the same)
            for kw in ag_candidates(args, nbytes):
                out.zero_()
                comm._all_gather_large(x, layout, out, **kw)
                torch.cuda.synchronize()
                if not agree(bits_equal(out, ref), cpu_group):
                    log(rank, "INVALID ag config", kw, (T, w))
                    continue
                us = timed(lambda kw=kw: comm._all_gather_large(x, layout, out, **kw), min(n, 4), cpu_group, quick=True)
                cands.append((us, kw))
            cands.sort(key=lambda c: c[0])
            for us, kw in cands[:3]:
                rows.append(dict(row, layout=lname, variant="sweep_best", cfg=kw, topo_b2b_us=us))
        del ref, out
    return rows


def worker(rank, args):
    try:
        dev, cpu_group, nccl = init_rank(rank, args.port, nccl=True)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD)
        log(rank, "TopoComm up", f"workspace {comm.workspace_bytes / 2**20:.1f} MiB chunk {comm.large_chunk_bytes}")
        spacer_cycles = spin_cycles_for_us(args.spacer_us)
        s_only = spacer_time(spacer_cycles, cpu_group)
        mibs = [float(m) if "." in m else int(m) for m in args.mibs.split(",")]
        for what in args.what.split(","):
            for mib in mibs:
                if what == "ar":
                    rows = bench_ar(rank, comm, nccl, cpu_group, dev, args, mib, spacer_cycles, s_only)
                elif what == "rs":
                    rows = bench_rs(rank, comm, nccl, cpu_group, dev, args, mib, spacer_cycles, s_only)
                else:
                    rows = []
                    for w in (1024, 512):
                        rows += bench_ag(rank, comm, nccl, cpu_group, dev, args, mib, w, spacer_cycles, s_only)
                append_jsonl(args.out, rows, rank)
                for r in rows:
                    log(rank, what.upper(), mib, "MiB", fmt(r))
                torch.cuda.empty_cache()
        dist.barrier(cpu_group)
        log(rank, "ALL DONE")
        comm.close()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--what", default="ar,rs,ag")
    ap.add_argument("--mibs", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--search", action="store_true", help="coordinate-descent search of the configuration instead of the grid sweep")
    ap.add_argument("--spacer-us", type=float, default=50.0)
    ap.add_argument("--port", type=int, default=29723)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
