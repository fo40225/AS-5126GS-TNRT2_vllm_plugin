"""Latency of the collectives fused with the Muse Spark residual boundary / pre_expert_norm vs the unfused two-step path.

Protocol: CUDA graph of N identical iterations replayed back to back; time per iteration = replay time / N, median over trials,
max over ranks (the slowest rank defines the collective). Unfused = TopoComm collective + the Triton kernel of vLLM
(PDL launch as in the decode graphs when VLLM_MUSE_SPARK_SM120_PDL is on), fused = the new kernels.

usage (server, GPU lock): GPU=0,1,2,3,4,5,6,7 SRC=<vllm tree> scripts/py_topo.sh bench/bench_boundary.py [--out file.json]
"""
import argparse
import json
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

D = 8192


def worker(rank, args):
    try:
        run(rank, args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)


def run(rank, args):
    dev, cpu_group, _ = init_rank(rank, args.port, nccl=False)
    from muse_topo import TopoComm
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd, fused_rmsnorm_fwd

    comm = TopoComm(cpu_group, dev, rank, WORLD)
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    alpha = (torch.rand(D, generator=g, device=dev) * 0.9 + 0.1).float()
    beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
    post_w = (torch.randn(D, generator=g, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
    pre_w = (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    w4 = (torch.randn(4096, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    rows = []
    N = args.n
    for T in args.ts:
        x = make_input(rank, (T, D), 11, dev, torch.bfloat16, "mixed")
        stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
        shard_u = make_input(rank, (T, 1024), 12, dev, torch.bfloat16, "mixed")
        shard_p = make_input(rank, (T, 512), 13, dev, torch.bfloat16, "mixed")
        kw = dict(alpha=alpha, beta=beta, post_gain_center=1.0, pre_gain_center=1.0, post_input_scale=True,
                  pre_input_scale=False, out_dtype=torch.bfloat16)

        def triton_boundary(y):
            return fused_norm_residual_norm_fwd(y, stream, post_w, pre_w, 1e-8, 1e-6, **kw)

        cases = {}
        cases["ar"] = lambda: comm.all_reduce(x)
        cases["boundary_triton"] = lambda: triton_boundary(x)
        cases["ar+triton"] = lambda: triton_boundary(comm.all_reduce(x))
        if comm.supports_all_reduce_boundary(x, stream):
            cases["ar_boundary_fused"] = lambda: comm.all_reduce_boundary(x, stream, alpha, beta, pre_w, post_w, 1e-8,
                                                                          1e-6, 1.0, 1.0)
        if T <= 32:
            cases["ag_u"] = lambda: comm.all_gather(shard_u, -1)
            cases["ag_u+triton"] = lambda: triton_boundary(comm.all_gather(shard_u, -1))
            cases["ag_u_boundary_fused"] = lambda: comm.all_gather_boundary(shard_u, stream, alpha, beta, pre_w, post_w,
                                                                           1e-8, 1e-6, 1.0, 1.0)
        cases["ag_p"] = lambda: comm.all_gather(shard_p, -1)
        cases["ag_p+rms_triton"] = lambda: fused_rmsnorm_fwd(comm.all_gather(shard_p, -1), w4, 1e-6, gain_center=1.0,
                                                           use_kahan_summation=True)
        if comm.supports_all_gather_rmsnorm(shard_p):
            cases["ag_p_rms_fused"] = lambda: comm.all_gather_rmsnorm(shard_p, w4, 1e-6, 1.0)
        row = {"T": T}
        for name, call in cases.items():
            gr = build_graph(call, N)
            per = time_graph(gr, N, cpu_group, target_s=0.05, trials=args.trials)
            row[name] = round(med_over_ranks(per, cpu_group), 2)
        rows.append(row)
        log(rank, json.dumps(row))
    if rank == 0 and args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
    comm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--port", type=int, default=29741)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
