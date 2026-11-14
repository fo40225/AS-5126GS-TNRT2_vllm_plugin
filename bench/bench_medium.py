"""Medium-size all_reduce sweep (bf16, 64 KiB .. 8 MiB): exact algorithm 2 vs its bf16-exchange variant vs the large algorithm.

usage: py_topo.sh bench/bench_medium.py --out <jsonl> [--kib 64,96,128,...] [--quick] [--no-sweep]
For every message size the script
  * times NCCL (vLLM's PyNcclCommunicator, NCCL_P2P_LEVEL=SYS) back-to-back,
  * sweeps (grid, block, unroll) of algorithm 2 with the exact fp32 exchange (checked against the arithmetic contract)
    and with the bf16 exchange (checked bitwise against the torch emulation: both island partials rounded to bf16),
  * times the large algorithm with the bf16 exchange (default configuration of the table) where it applies,
  * times the dispatch path comm.all_reduce(x, out=out) with the medium bf16 exchange off and on.
Protocol: CUDA graph of N identical calls (see bench_h1a.py), "b2b" = back-to-back, "spacer" = (sleep kernel of
--spacer-us, call) pairs minus the sleep-only time; per call time = replay time / N, median over trials, max over ranks.
The final rows (best configuration per variant) are re-timed with the full protocol; candidates use a shorter one.
"""
import argparse
import itertools
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403
from bench_h1a import GRIDS, None_call, pair_time, spacer_time, timed


def shape_for(nbytes):
    rows = nbytes // (8192 * 2)
    if rows * 8192 * 2 == nbytes:
        return (rows, 8192)
    return (nbytes // (4096 * 2), 4096)


def contract_ref_bx(xs, members):
    """Torch emulation of the bf16 exchange: each island partial (fp32 chain, ascending ranks) is rounded to bf16, the
    two rounded partials are added in fp32 and rounded once."""
    dt = xs[0].dtype
    ps = []
    for isl in members:
        acc = xs[isl[0]].float()
        for r in isl[1:]:
            acc = acc + xs[r].float()
        ps.append(acc.to(dt).float())
    return canon_words((ps[0] + ps[1]).to(dt))


def ncalls_for(nbytes):
    return 100 if nbytes < MiB else (20 if nbytes < 8 * MiB else 6)


def configs(nbytes, blocks, unrolls):
    items = nbytes // 64
    # the tuned tables never used 64-thread blocks from 1 MiB on or unroll 4 from 2 MiB on: keep the sweep short there
    blocks = [b for b in blocks if not (nbytes >= MiB and b == 64)]
    unrolls = [u for u in unrolls if not (nbytes >= 2 * MiB and u == 4)]
    for block, unroll in itertools.product(blocks, unrolls):
        gmax = max(1, -(-items // (block * unroll)))
        grids = sorted({g for g in GRIDS if g <= min(gmax, 128)} | {min(gmax, 128)})
        for grid in grids:
            yield (grid, block, unroll)


def bench_size(rank, comm, nccl, cpu_group, dev, args, nbytes, spacer_cycles, spacer_us):
    shape = shape_for(nbytes)
    dtype = torch.bfloat16
    xs = [make_input(r, shape, 55, dev, dtype, args.kind) for r in range(WORLD)]
    x = xs[rank]
    out = torch.empty_like(x)
    ref = contract_ref(xs, comm.info.members)
    ref_bx = contract_ref_bx(xs, comm.info.members)
    n = ncalls_for(nbytes)
    row = {"op": "all_reduce", "shape": list(shape), "bytes": nbytes}
    nout = torch.empty_like(x)
    row["nccl_b2b_us"] = timed(lambda: nccl.all_reduce(x, nout), n, cpu_group)
    row["nccl_spacer_us"] = pair_time(lambda: nccl.all_reduce(x, nout), n, spacer_cycles, spacer_us, cpu_group)
    rows = []
    blocks = (128, 256, 512) if args.quick else (64, 128, 256, 512)
    unrolls = (1, 2) if args.quick else (1, 2, 4)
    for name, bx, expect in (("ar2", 0, ref), ("ar2bx", 1, ref_bx)):
        if not args.sweep:
            cfgs = [(8, 256, 1)]
        else:
            cfgs = list(configs(nbytes, blocks, unrolls))
        res = []
        for cfg in cfgs:
            grid, block, unroll = cfg
            out.zero_()
            comm._c.all_reduce(x, out, 2, grid, block, unroll, bx)
            torch.cuda.synchronize()
            if not all(gather_all(bits_equal(out, expect), cpu_group)):
                if rank == 0:
                    print("INVALID config", name, cfg, shape, flush=True)
                continue
            us = timed(lambda: comm._c.all_reduce(x, out, 2, grid, block, unroll, bx), min(n, 20), cpu_group,
                       quick=args.sweep)
            res.append((us, cfg))
        if not res:
            rows.append(dict(row, variant=name, error="no valid config"))
            continue
        res.sort()
        for us, cfg in res[:args.top]:
            grid, block, unroll = cfg
            call = lambda: comm._c.all_reduce(x, out, 2, grid, block, unroll, bx)
            b2b = timed(call, n, cpu_group)
            sp = pair_time(call, n, spacer_cycles, spacer_us, cpu_group)
            rows.append(dict(row, variant=name, cfg=dict(grid=grid, block=block, unroll=unroll), b2b_us=b2b,
                             spacer_us=sp, sweep_us=us))
    # large algorithm, bf16 exchange, table configuration
    from muse_topo.collectives import _AR_LARGE_TABLE, _pick
    if comm._large_ok(nbytes) and args.large:
        _, mode, ch, rg, ag, blk, un, crs, cag = _pick(_AR_LARGE_TABLE, nbytes)
        for mode_ in (mode, 1 - mode):
            out.zero_()
            comm._all_reduce_large(x, out, mode_, ch, rg, ag, blk, un, True, crs, cag)
            torch.cuda.synchronize()
            ok = all(gather_all(bits_equal(out, ref_bx), cpu_group))
            call = lambda: comm._all_reduce_large(x, out, mode_, ch, rg, ag, blk, un, True, crs, cag)
            b2b = timed(call, n, cpu_group)
            sp = pair_time(call, n, spacer_cycles, spacer_us, cpu_group)
            rows.append(dict(row, variant="largebx", cfg=dict(mode=mode_, chunk=ch, rs=rg, ag=ag, block=blk, unroll=un),
                             b2b_us=b2b, spacer_us=sp, bitwise_ok=ok))
    # dispatch path
    old = (comm.medium_bx, comm.bf16_exchange)
    for label, mbx, bx in (("dispatch_exact", False, False), ("dispatch_largebx_only", False, True),
                           ("dispatch_mediumbx", True, True)):
        comm.medium_bx, comm.bf16_exchange = mbx, bx
        out.zero_()
        comm.all_reduce(x, out=out)
        torch.cuda.synchronize()
        got = out.clone()
        call = lambda: comm.all_reduce(x, out=out)
        b2b = timed(call, n, cpu_group)
        sp = pair_time(call, n, spacer_cycles, spacer_us, cpu_group)
        expect = ref_bx if comm._medium_bx(x, nbytes) or (bx and nbytes >= comm.ar_large_min) else ref
        rows.append(dict(row, variant=label, b2b_us=b2b, spacer_us=sp, bitwise_ok=all(gather_all(bits_equal(got, expect), cpu_group))))
    comm.medium_bx, comm.bf16_exchange = old
    return rows


def worker(rank, args):
    try:
        dev, cpu_group, nccl = init_rank(rank, args.port, nccl=True)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD, bf16_exchange=True)
        log(rank, "TopoComm up", f"workspace {comm.workspace_bytes / 2**20:.1f} MiB")
        spacer_cycles = spin_cycles_for_us(args.spacer_us)
        s_only = spacer_time(spacer_cycles, cpu_group)
        log(rank, f"spacer request {args.spacer_us} us -> {s_only:.2f} us per sleep kernel in a graph")
        for kib in [int(k) for k in args.kib.split(",")]:
            rows = bench_size(rank, comm, nccl, cpu_group, dev, args, kib * 1024, spacer_cycles, s_only)
            append_jsonl(args.out, rows, rank)
            for r in rows:
                log(rank, f"{kib:5d} KiB", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                                            if k in ("variant", "cfg", "nccl_b2b_us", "b2b_us", "spacer_us", "bitwise_ok", "error")})
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
    ap.add_argument("--kib", default="64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192")
    ap.add_argument("--sweep", action="store_true", default=True)
    ap.add_argument("--no-sweep", dest="sweep", action="store_false")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--large", action="store_true", default=True)
    ap.add_argument("--top", type=int, default=2, help="best N configurations per variant are re-timed")
    ap.add_argument("--spacer-us", type=float, default=50.0)
    ap.add_argument("--kind", default="randn")
    ap.add_argument("--port", type=int, default=29714)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
