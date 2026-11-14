"""In-kernel role timeline of the fused large all-reduce (needs MUSE_TOPO_TRACE=1: a separate JIT build).

usage: MUSE_TOPO_TRACE=1 py_topo.sh bench/trace_h1b.py [--mib 128] [--rank-print 0]
One eager all_reduce_large call per configuration; per launch (pipeline step) the first block of every role stamps
%globaltimer at its start and end (per GPU clock; only differences are meaningful). Printed per step: start offset from
the call start, duration of each role (RS workers, AG workers, RS cut block, AG cut block), step end, gap between the end
of the previous step and the start of this one (launch latency + block scheduling).
"""
import argparse
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

CONFIGS = [
    ("default fused", dict(mode=1, chunk_bytes=1 << 20, rs_grid=8, ag_grid=24, block=512, unroll=1, cut_rs=0, cut_ag=0)),
    ("fused cut_rs+cut_ag", dict(mode=1, chunk_bytes=1 << 20, rs_grid=8, ag_grid=24, block=512, unroll=1, cut_rs=1, cut_ag=1)),
    ("fused tuned (rs4 ag8 u2 cut_ag)", dict(mode=1, chunk_bytes=1 << 20, rs_grid=4, ag_grid=8, block=512, unroll=2, cut_rs=0, cut_ag=1)),
    ("fused merged cut (rs4 ag8 u2)", dict(mode=1, chunk_bytes=1 << 20, rs_grid=4, ag_grid=8, block=512, unroll=2, cut_rs=1, cut_ag=2)),
    ("fused merged cut (rs8 ag8 u1)", dict(mode=1, chunk_bytes=1 << 20, rs_grid=8, ag_grid=8, block=512, unroll=1, cut_rs=1, cut_ag=2)),
]


def worker(rank, args):
    try:
        dev, cpu_group, _ = init_rank(rank, args.port)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD)
        comm._c.trace_enable()
        T = args.mib * MiB // (8192 * 2)
        x = make_input(rank, (T, 8192), 3, dev)
        out = torch.empty_like(x)
        for name, kw in CONFIGS:
            for _ in range(3):
                comm._all_reduce_large(x, out, **kw)
            torch.cuda.synchronize()
            dist.barrier(cpu_group)
            comm._c.trace_clear()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            comm._all_reduce_large(x, out, **kw)
            e.record()
            e.synchronize()
            rows = comm._c.trace_rows()
            steps = []
            for r in range(256):
                st = rows[r * 8:(r + 1) * 8]
                if any(st):
                    steps.append(st)
            steps.sort(key=lambda st: min(v for v in st if v))
            if rank == args.rank_print:
                t0 = min(v for st in steps for v in st if v)
                print(f"=== {name}: {args.mib} MiB, event time {s.elapsed_time(e) * 1e3:.0f} us, {len(steps)} launches", flush=True)
                prev_end = None
                for i, st in enumerate(steps):
                    starts = [v for v in st[0::2] if v]
                    ends = [v for v in st[1::2] if v]
                    s0, e0 = min(starts), max(ends)
                    dur = [((st[2 * k + 1] - st[2 * k]) / 1e3 if st[2 * k] and st[2 * k + 1] else float("nan")) for k in range(4)]
                    gap = (s0 - prev_end) / 1e3 if prev_end else float("nan")
                    print(f"step {i:2d} start {(s0 - t0) / 1e3:8.1f} end {(e0 - t0) / 1e3:8.1f} len {(e0 - s0) / 1e3:7.1f} gap {gap:6.1f} | "
                          f"rs {dur[0]:7.1f} ag {dur[1]:7.1f} rs_cut {dur[2]:7.1f} ag_cut {dur[3]:7.1f}", flush=True)
                    prev_end = e0
            dist.barrier(cpu_group)
        comm.close()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=128)
    ap.add_argument("--rank-print", type=int, default=0)
    ap.add_argument("--port", type=int, default=29728)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
