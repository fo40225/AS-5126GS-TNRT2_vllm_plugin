"""Latency of the fused all-reduce + boundary vs the two-step path when the parameters are NOT in L2.

In a decode step the per-layer vectors of the boundary (alpha, beta, the norm weights: ~110 KiB) are read once per step, with ~14 GB of
weight traffic in between, so they come from HBM; bench_boundary.py repeats the same call and keeps them in L2. Here a 256 MiB
copy runs before every call, and the time is taken from the kernel timestamps of the profiler: from the start of the all-reduce
kernel to the end of the boundary kernel (two-step), or the duration of the fused kernel. Median over iterations, max over ranks.

usage (server, GPU lock): GPU=0,1,2,3,4,5,6,7 SRC=<vllm tree> scripts/py_topo.sh bench/bench_boundary_cold.py [--modes 0,12]
(--modes selects the placement of the operand loads of the fused kernels, BndParams::ld_mode in muse_topo/csrc/topo_types.h)
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile

from common import *  # noqa: F401,F403

D = 8192


def worker(rank, args):
    try:
        run(rank, args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)


def kernel_events(prof):
    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    os.remove(path)
    ks = [(e["ts"], e["dur"], e["name"]) for e in ev if e.get("cat") == "kernel"]
    ks.sort()
    return ks


def samples(ks, first, last):
    """Time from the start of a kernel whose name contains `first` to the end of the next one that contains `last`."""
    out, t0 = [], None
    for ts, dur, name in ks:
        if first in name and t0 is None:
            t0 = ts
        if t0 is not None and last in name:
            out.append(ts + dur - t0)
            t0 = None
    return out


def run(rank, args):
    dev, cpu_group, _ = init_rank(rank, args.port, nccl=False)
    from muse_topo import TopoComm
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd, fused_rmsnorm_fwd

    comm = TopoComm(cpu_group, dev, rank, WORLD)
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    # one parameter set per "layer" so that repeated calls of a graph do not hit the same lines
    nl = args.layers
    params = []
    for _ in range(nl):
        alpha = (torch.rand(D, generator=g, device=dev) * 0.9 + 0.1).float()
        beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
        post_w = (torch.randn(D, generator=g, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
        pre_w = (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
        params.append((alpha, beta, post_w, pre_w))
    big_a = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    big_b = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    big_b.normal_()
    rows = []
    for T in args.ts:
        x = make_input(rank, (T, D), 11, dev, torch.bfloat16, "mixed")
        stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()

        def unfused(k):
            alpha, beta, post_w, pre_w = params[k]
            big_a.copy_(big_b)
            y = comm.all_reduce(x)
            fused_norm_residual_norm_fwd(y, stream, post_w, pre_w, 1e-8, 1e-6, alpha=alpha, beta=beta,
                                         post_gain_center=1.0, pre_gain_center=1.0, post_input_scale=True,
                                         pre_input_scale=False, out_dtype=torch.bfloat16)

        def fused(k):
            alpha, beta, post_w, pre_w = params[k]
            big_a.copy_(big_b)
            comm.all_reduce_boundary(x, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)

        shard = make_input(rank, (T, 1024), 12, dev, torch.bfloat16, "mixed")

        def unfused_ag(k):
            alpha, beta, post_w, pre_w = params[k]
            big_a.copy_(big_b)
            y = comm.all_gather(shard, -1)
            fused_norm_residual_norm_fwd(y, stream, post_w, pre_w, 1e-8, 1e-6, alpha=alpha, beta=beta,
                                         post_gain_center=1.0, pre_gain_center=1.0, post_input_scale=True,
                                         pre_input_scale=False, out_dtype=torch.bfloat16)

        def fused_ag(k):
            alpha, beta, post_w, pre_w = params[k]
            big_a.copy_(big_b)
            comm.all_gather_boundary(shard, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)

        shard_p = make_input(rank, (T, 512), 13, dev, torch.bfloat16, "mixed")
        w4 = (torch.randn(4096, generator=g, device=dev) * 0.3).to(torch.bfloat16)

        def unfused_rms(k):
            big_a.copy_(big_b)
            fused_rmsnorm_fwd(comm.all_gather(shard_p, -1), w4, 1e-6, gain_center=1.0, use_kahan_summation=True)

        def fused_rms(k):
            big_a.copy_(big_b)
            comm.all_gather_rmsnorm(shard_p, w4, 1e-6, 1.0)

        row = {"T": T}
        cases = [("unfused", unfused, "ar2_kernel", "_norm_residual_norm")]
        for mode in args.modes:
            cases.append((f"fused_mode{mode}", fused, "ar2_bnd_kernel", "ar2_bnd_kernel", mode))
        if T <= 32:
            cases.append(("ag_unfused", unfused_ag, "ag_ll_kernel", "_norm_residual_norm"))
            for mode in args.modes:
                cases.append((f"ag_fused_mode{mode}", fused_ag, "ag_bnd_kernel", "ag_bnd_kernel", mode))
        cases.append(("rms_unfused", unfused_rms, "ag_ll_kernel", "_rmsnorm_fwd_kernel"))
        for mode in args.modes[:1]:
            cases.append((f"rms_fused_mode{mode}", fused_rms, "ag_bnd_kernel", "ag_bnd_kernel", mode))
        if T > 64:
            cases = cases[:0]
        for case in cases:
            name, fn, first, last = case[:4]
            if len(case) > 4:
                comm._c.set_ld_mode(case[4])
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for k in range(nl):
                    fn(k)
            gr.replay()
            torch.cuda.synchronize()
            dist.barrier(cpu_group)
            vals = []
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(args.replays):
                    gr.replay()
                torch.cuda.synchronize()
            vals = samples(kernel_events(prof), first, last)
            med = statistics.median(vals) if vals else float("nan")
            row[name] = round(max(gather_all(med, cpu_group)), 2)
            row[name + "_n"] = len(vals)
        comm._c.set_ld_mode(1)
        rows.append(row)
        log(rank, json.dumps(row))
    comm.close()


if __name__ == "__main__":
    import torch.distributed as dist  # noqa: F401 (used by run through common's init_rank)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 4, 8])
    ap.add_argument("--modes", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2, 3])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--replays", type=int, default=6)
    ap.add_argument("--port", type=int, default=29761)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
