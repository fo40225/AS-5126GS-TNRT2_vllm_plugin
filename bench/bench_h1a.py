"""Latency sweep of the muse_topo decode-size collectives vs NCCL (vLLM's PyNcclCommunicator, NCCL_P2P_LEVEL=SYS).

usage: py_topo.sh bench/bench_h1a.py --out <jsonl> [--what ar,ag] [--sweep] [--spacer-us 50] [--quick]
Protocol: CUDA graph of N identical calls (N=100, 20 for >= 1 MiB, 5 for >= 8 MiB), replayed; per call time = replay
time / N, median over trials, max over ranks. "b2b" = back-to-back, "spacer" = (sleep kernel of --spacer-us, call)
pairs minus the measured sleep-only time (the exposed latency a compute-separated collective adds).
The topo configuration (algorithm, grid, block, unroll) is swept per shape; every candidate is checked bitwise against
the arithmetic contract (all_reduce) or torch.cat (all_gather) before it is timed.
"""
import argparse
import itertools
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

GRIDS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 188]


def ncalls_for(nbytes):
    return 100 if nbytes < MiB else (20 if nbytes < 8 * MiB else 6)


def timed(call, ncalls, cpu_group, spacer_cycles=0, quick=False):
    g = build_graph(call, ncalls, spacer_cycles)
    per = time_graph(g, ncalls, cpu_group, target_s=0.01 if quick else 0.05, trials=3 if quick else 5)
    del g
    return med_over_ranks(per, cpu_group)


def spacer_time(spacer_cycles, cpu_group, n=100):
    g = build_graph(None_call, n, spacer_cycles)
    per = time_graph(g, n, cpu_group, target_s=0.03, trials=3)
    del g
    return med_over_ranks(per, cpu_group)


def None_call():
    pass


def pair_time(call, ncalls, spacer_cycles, spacer_us, cpu_group):
    return timed(call, ncalls, cpu_group, spacer_cycles) - spacer_us


def sweep_configs(items_for_algo, algos, blocks, unrolls, flag_algos=()):
    for algo in algos:
        items = items_for_algo(algo)
        if algo in flag_algos:  # flag-protocol kernels: grid = number of chunks (<= 128), block up to 1024, no unroll
            for block in (128, 256, 512, 1024):
                for grid in sorted({g for g in GRIDS if g <= min(128, max(1, items))}):
                    yield (algo, grid, block, 1)
            continue
        for block, unroll in itertools.product(blocks, unrolls):
            gmax = max(1, -(-items // (block * unroll)))
            grids = sorted({g for g in GRIDS if g <= gmax} | {min(gmax, 188)})
            for grid in grids:
                yield (algo, grid, block, unroll)


def bench_ar(rank, comm, nccl, cpu_group, dev, args, shape, dtype, spacer_cycles, spacer_us):
    nbytes = int(torch.empty(shape, dtype=dtype, device="meta").numel() * torch.empty((), dtype=dtype).element_size())
    xs = [make_input(r, shape, 55, dev, dtype, args.kind) for r in range(WORLD)]
    x = xs[rank]
    out = torch.empty_like(x)
    ref = contract_ref(xs, comm.info.members)
    n = ncalls_for(nbytes)
    row = {"op": "all_reduce", "shape": list(shape), "dtype": str(dtype).split(".")[-1], "bytes": nbytes}
    # --- NCCL baseline
    nout = torch.empty_like(x)
    row["nccl_b2b_us"] = timed(lambda: nccl.all_reduce(x, nout), n, cpu_group)
    row["nccl_spacer_us"] = pair_time(lambda: nccl.all_reduce(x, nout), n, spacer_cycles, spacer_us, cpu_group)
    # --- topo sweep
    results = []
    blocks = (128, 256, 512) if args.quick else (32, 64, 128, 256, 512)
    unrolls = (1, 2) if args.quick else (1, 2, 4)
    algos = [a for a in (1, 2, 3) if (a == 1 and nbytes <= comm.ar1_max_bytes) or (a in (2, 3) and nbytes % 64 == 0)]
    if args.algos:
        algos = [a for a in algos if a in args.algos]
    items = lambda a: nbytes // 16 if a == 1 else nbytes // 64
    cfgs = list(sweep_configs(items, algos, blocks, unrolls, flag_algos=(3,))) if args.sweep else []
    if not args.sweep:
        for algo, blk, un, gm, in [(1, 128, 1, 128), (2, 256, 1, 128), (3, 256, 1, 128)]:
            if algo in algos:
                cfgs.append((algo, max(1, min(gm, -(-items(algo) // (blk * un)))), blk, un))
    for cfg in cfgs:
        algo, grid, block, unroll = cfg
        out.zero_()
        comm._c.all_reduce(x, out, algo, grid, block, unroll)
        torch.cuda.synchronize()
        ok = bits_equal(out, ref)
        oks = all(gather_all(ok, cpu_group))
        if not oks:
            if rank == 0:
                print("INVALID config", cfg, shape, flush=True)
            continue
        us = timed(lambda: comm._c.all_reduce(x, out, algo, grid, block, unroll), n if not args.sweep else min(n, 20),
                   cpu_group, quick=args.sweep)
        results.append((us, cfg))
    if not results:
        return [dict(row, error="no valid config")]
    rows = []
    results.sort()
    best_per_algo = {}
    for us, cfg in results:
        best_per_algo.setdefault(cfg[0], (us, cfg))
    for algo, (us, cfg) in sorted(best_per_algo.items()):
        _, grid, block, unroll = cfg
        call = lambda: comm._c.all_reduce(x, out, algo, grid, block, unroll)
        b2b = timed(call, n, cpu_group)
        sp = pair_time(call, n, spacer_cycles, spacer_us, cpu_group)
        out.zero_()
        call()
        torch.cuda.synchronize()
        rows.append(dict(row, topo_algo=algo, topo_cfg=dict(grid=grid, block=block, unroll=unroll), topo_b2b_us=b2b,
                         topo_spacer_us=sp, bitwise_contract=bits_equal(out, ref)))
    return rows


def bench_ar_public(rank, comm, nccl, cpu_group, dev, args, shape, dtype, spacer_cycles, spacer_us):
    """The dispatch-table path (what the plugin calls): comm.all_reduce(x, out=out), verified against the contract."""
    nbytes = int(torch.empty(shape, dtype=dtype, device="meta").numel() * torch.empty((), dtype=dtype).element_size())
    xs = [make_input(r, shape, 55, dev, dtype, args.kind) for r in range(WORLD)]
    x = xs[rank]
    out = torch.empty_like(x)
    ref = contract_ref(xs, comm.info.members)
    n = ncalls_for(nbytes)
    row = {"op": "all_reduce", "shape": list(shape), "dtype": str(dtype).split(".")[-1], "bytes": nbytes}
    nout = torch.empty_like(x)
    row["nccl_b2b_us"] = timed(lambda: nccl.all_reduce(x, nout), n, cpu_group)
    row["nccl_spacer_us"] = pair_time(lambda: nccl.all_reduce(x, nout), n, spacer_cycles, spacer_us, cpu_group)
    assert comm.supports("all_reduce", x)
    out.zero_()
    comm.all_reduce(x, out=out)
    torch.cuda.synchronize()
    ok = all(gather_all(bits_equal(out, ref), cpu_group))
    call = lambda: comm.all_reduce(x, out=out)
    b2b = timed(call, n, cpu_group)
    sp = pair_time(call, n, spacer_cycles, spacer_us, cpu_group)
    from muse_topo.collectives import _AR_TABLE, _pick
    algo, grid, block, unroll = _pick(_AR_TABLE, nbytes)[1:]
    return [dict(row, topo_algo=algo, topo_cfg=dict(grid=grid, block=block, unroll=unroll), topo_b2b_us=b2b,
                 topo_spacer_us=sp, bitwise_contract=ok)]


def bench_ag_public(rank, comm, nccl, cpu_group, dev, args, T, w, dtype, spacer_cycles, spacer_us):
    xs = [make_input(r, (T, w), 77, dev, dtype) for r in range(WORLD)]
    x = xs[rank]
    esz = torch.empty((), dtype=dtype).element_size()
    nbytes = T * w * esz
    n = ncalls_for(nbytes * 8)
    row = {"op": "all_gather", "shape": [T, w], "out_shape": [T, 8 * w], "dtype": str(dtype).split(".")[-1], "shard_bytes": nbytes}
    nout = torch.empty((WORLD * T, w), dtype=dtype, device=dev)
    row["nccl_b2b_us"] = timed(lambda: nccl.all_gather(nout, x), n, cpu_group)
    row["nccl_spacer_us"] = pair_time(lambda: nccl.all_gather(nout, x), n, spacer_cycles, spacer_us, cpu_group)
    rows = []
    for dim, name in ((-1, "dim-1"), (0, "dim0")):
        out = torch.empty((T, WORLD * w) if dim == -1 else (WORLD * T, w), dtype=dtype, device=dev)
        ref = torch.cat(xs, dim)
        if not comm.supports("all_gather", x, dim):
            rows.append(dict(row, layout=name, error="unsupported"))
            continue
        out.zero_()
        comm.all_gather(x, dim, out=out)
        torch.cuda.synchronize()
        ok = all(gather_all(bits_equal(out, ref), cpu_group))
        call = lambda: comm.all_gather(x, dim, out=out)
        b2b = timed(call, n, cpu_group)
        sp = pair_time(call, n, spacer_cycles, spacer_us, cpu_group)
        from muse_topo.collectives import _AG_TABLES, _pick
        plan = comm._ag_plan(nbytes, 0 if dim == -1 else 1, w * esz)
        rows.append(dict(row, layout=name, topo_algo=plan[1], topo_cfg=dict(grid=plan[2], block=plan[3], unroll=plan[4]),
                         topo_b2b_us=b2b, topo_spacer_us=sp, bitwise=ok))
    return rows


def bench_ag(rank, comm, nccl, cpu_group, dev, args, T, w, dtype, spacer_cycles, spacer_us):
    xs = [make_input(r, (T, w), 77, dev, dtype) for r in range(WORLD)]
    x = xs[rank]
    esz = torch.empty((), dtype=dtype).element_size()
    nbytes = T * w * esz
    n = ncalls_for(nbytes * 8)
    row = {"op": "all_gather", "shape": [T, w], "out_shape": [T, 8 * w], "dtype": str(dtype).split(".")[-1], "shard_bytes": nbytes}
    nout = torch.empty((WORLD * T, w), dtype=dtype, device=dev)
    row["nccl_b2b_us"] = timed(lambda: nccl.all_gather(nout, x), n, cpu_group)
    row["nccl_spacer_us"] = pair_time(lambda: nccl.all_gather(nout, x), n, spacer_cycles, spacer_us, cpu_group)
    rows = []
    for layout, name in ((0, "dim-1"), (1, "dim0")):
        out = torch.empty((T, WORLD * w) if layout == 0 else (WORLD * T, w), dtype=dtype, device=dev)
        ref = torch.cat(xs, -1 if layout == 0 else 0)
        blocks = (128, 256, 512) if args.quick else (32, 64, 128, 256, 512)
        unrolls = (1, 2) if args.quick else (1, 2, 4)
        ag_algos = [1] if nbytes <= comm.ag_ll_max_bytes else []
        if nbytes % 16 == 0 and (layout == 1 or (w * esz) % 16 == 0) and nbytes <= comm.ag_flag_max_bytes:
            ag_algos.append(2)
        cfgs = list(sweep_configs(lambda a: nbytes // 4 if a == 1 else nbytes // 16, ag_algos, blocks, unrolls,
                                  flag_algos=(2,))) if args.sweep else [
            (a, max(1, min(128, -(-(nbytes // (4 if a == 1 else 16)) // 128))), 128 if a == 1 else 256, 1) for a in ag_algos]
        res = []
        for cfg in cfgs:
            algo, grid, block, unroll = cfg
            out.zero_()
            comm._c.all_gather(x, out, layout, algo, grid, block, unroll)
            torch.cuda.synchronize()
            if not all(gather_all(bits_equal(out, ref), cpu_group)):
                if rank == 0:
                    print("INVALID ag config", cfg, (T, w), flush=True)
                continue
            us = timed(lambda: comm._c.all_gather(x, out, layout, algo, grid, block, unroll), min(n, 20) if args.sweep else n,
                       cpu_group, quick=args.sweep)
            res.append((us, cfg))
        if not res:
            rows.append(dict(row, layout=name, error="no valid config"))
            continue
        res.sort()
        best_per_algo = {}
        for us, cfg in res:
            best_per_algo.setdefault(cfg[0], (us, cfg))
        for algo, (us, cfg) in sorted(best_per_algo.items()):
            _, grid, block, unroll = cfg
            call = lambda: comm._c.all_gather(x, out, layout, algo, grid, block, unroll)
            b2b = timed(call, n, cpu_group)
            sp = pair_time(call, n, spacer_cycles, spacer_us, cpu_group)
            rows.append(dict(row, layout=name, topo_algo=algo, topo_cfg=dict(grid=grid, block=block, unroll=unroll),
                             topo_b2b_us=b2b, topo_spacer_us=sp))
    return rows


def worker(rank, args):
    try:
        dev, cpu_group, nccl = init_rank(rank, args.port, nccl=True)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD)
        log(rank, "TopoComm up", f"workspace {comm.workspace_bytes / 2**20:.1f} MiB")
        spacer_cycles = spin_cycles_for_us(args.spacer_us)
        s_only = spacer_time(spacer_cycles, cpu_group)
        log(rank, f"spacer request {args.spacer_us} us -> {s_only:.2f} us per sleep kernel in a graph")
        null_us = spacer_time(1, cpu_group)
        log(rank, f"null kernel (sleep 1 cycle) back-to-back in a graph: {null_us:.2f} us per kernel")
        append_jsonl(args.out, [{"op": "null_kernel", "b2b_us": null_us, "spacer50_us": s_only}], rank)
        if "ar" in args.what:
            Ts = [int(t) for t in args.ts.split(",")]
            for hid in [int(h) for h in args.hids.split(",")]:
                for T in Ts:
                    rows = (bench_ar_public if args.public else bench_ar)(rank, comm, nccl, cpu_group, dev, args, (T, hid), torch.bfloat16, spacer_cycles, s_only)
                    append_jsonl(args.out, rows, rank)
                    for r in rows:
                        log(rank, "AR", r["shape"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                                                    if k in ("topo_algo", "topo_cfg", "nccl_b2b_us", "topo_b2b_us", "nccl_spacer_us", "topo_spacer_us", "error")})
        if "ag" in args.what:
            Ts = [int(t) for t in args.ag_ts.split(",")]
            for w in (512, 1024):
                for T in Ts:
                    rows = (bench_ag_public if args.public else bench_ag)(rank, comm, nccl, cpu_group, dev, args, T, w, torch.bfloat16, spacer_cycles, s_only)
                    append_jsonl(args.out, rows, rank)
                    for r in rows:
                        log(rank, "AG", r["shape"], r.get("layout"), {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                                                                     if k in ("topo_algo", "topo_cfg", "nccl_b2b_us", "topo_b2b_us", "nccl_spacer_us", "topo_spacer_us", "error")})
            if args.logits:
                rows = (bench_ag_public if args.public else bench_ag)(rank, comm, nccl, cpu_group, dev, args, 1, 25256, torch.bfloat16, spacer_cycles, s_only)
                append_jsonl(args.out, rows, rank)
                for r in rows:
                    log(rank, "AG logits", r.get("layout"), {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                                                            if k in ("topo_algo", "topo_cfg", "nccl_b2b_us", "topo_b2b_us", "nccl_spacer_us", "topo_spacer_us", "error")})
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
    ap.add_argument("--what", default="ar,ag")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--public", action="store_true", help="measure the dispatch-table path comm.all_reduce/all_gather")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--spacer-us", type=float, default=50.0)
    ap.add_argument("--ts", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--hids", default="8192,4096")
    ap.add_argument("--ag-ts", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--logits", action="store_true", default=True)
    ap.add_argument("--kind", default="randn")
    ap.add_argument("--algos", type=lambda s: [int(a) for a in s.split(",")], default=None)
    ap.add_argument("--port", type=int, default=29713)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
