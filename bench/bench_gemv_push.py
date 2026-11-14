"""Latency of the sharded GEMV fused with the all-gather vs the two-step path (cuBLAS GEMV, then the fused gather kernel).

Protocol: CUDA graph of N identical iterations replayed back to back (single stream, as in a decode graph), time per iteration =
replay time / N, median over trials, max over ranks. The chains are decode-layer-like, so that the GEMV kernel sees its real
predecessor (an all-reduce kernel) and the next iteration its real successor:
  pooler:    all_reduce_boundary(x8192) -> pooler GEMV + all_gather_rmsnorm      (the chain after the attention-output all-reduce)
  unpooler:  all_reduce(x4096) -> unpooler GEMV + all_gather_boundary            (the chain after the MoE all-reduce)
Variants: two-step = F.linear (cuBLAS) + the fused gather kernels of r2d; gemv = TopoComm.gemv_all_gather_* with the pushes left to the
gather kernel (comm.gemv_push = False); push = every GEMV CTA pushes its columns, last = the last CTA pushes the block, late = push with the weights loaded behind the wait (`_vN`: geometry variant N, `--variants`). `--cold`: a 256 MiB memset before every iteration (the
weights then come from HBM; the memset time is subtracted).

usage (server, GPU lock): GPU=0,1,2,3,4,5,6,7 SRC=<vllm tree> scripts/py_topo.sh bench/bench_gemv_push.py [--ts 1,2,4,8] [--cold]
"""
import argparse
import json
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

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

    comm = TopoComm(cpu_group, dev, rank, WORLD)
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    alpha = (torch.rand(D, generator=g, device=dev) * 0.9 + 0.1).float()
    beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
    post_w = (torch.randn(D, generator=g, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
    pre_w = (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    w4 = (torch.randn(4096, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    wp = (torch.randn(512, 8192, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    wu = (torch.randn(1024, 4096, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows = []
    N = args.n
    for T in args.ts:
        xr = make_input(rank, (T, D), 11, dev, torch.bfloat16, "mixed")
        xu = make_input(rank, (T, 4096), 12, dev, torch.bfloat16, "mixed")
        stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()

        def pooler(variant):
            y, _ = comm.all_reduce_boundary(xr, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
            if variant == "two-step":
                shard = F.linear(y, wp)
                return comm.all_gather_rmsnorm(shard, w4, 1e-6, 1.0)
            return comm.gemv_all_gather_rmsnorm(y, wp, w4, 1e-6, 1.0)

        def unpooler(variant):
            y = comm.all_reduce(xu)
            if variant == "two-step":
                shard = F.linear(y, wu)
                return comm.all_gather_boundary(shard, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
            return comm.gemv_all_gather_boundary(y, wu, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)

        def layer(variant):
            pooled = pooler(variant)
            y = comm.all_reduce(pooled)
            if variant == "two-step":
                shard = F.linear(y, wu)
                return comm.all_gather_boundary(shard, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
            return comm.gemv_all_gather_boundary(y, wu, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)

        cases = {
            "ar_bnd_only": lambda: comm.all_reduce_boundary(xr, stream, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0),
            "ar_only": lambda: comm.all_reduce(xu),
        }
        for name, fn in (("pooler", pooler), ("unpooler", unpooler), ("layer", layer)):
            cases[f"{name}/two-step"] = (lambda fn=fn: fn("two-step"))
            for gv in args.variants:
                for tr in args.triggers:
                    # GEMV kernel without / with the pushes, geometry variant gv, trigger stage tr of the all-reduce in front
                    for mode in ("gemv", "push", "last", "late"):
                        cases[f"{name}/{mode}_v{gv}_t{tr}"] = (lambda fn=fn, mode=mode: fn(mode))
        row = {"T": T, "cold": args.cold}
        base_flush = 0.0
        if args.cold:
            gr = build_graph(lambda: flush.zero_(), N)
            base_flush = med_over_ranks(time_graph(gr, N, cpu_group, target_s=0.05, trials=args.trials), cpu_group)
            row["flush_only"] = round(base_flush, 2)
        for name, call in cases.items():
            variant = name.split("/")[-1]
            comm.gemv_push = {"push": 1, "last": 2, "late": 5}.get(variant.split("_")[0], 0)
            if "_v" in variant:
                comm.set_gemv_variant(int(variant.split("_v")[1].split("_")[0]))
                comm._c.set_trigger(int(variant.split("_t")[-1]))
            if args.cold:
                def one(call=call):
                    flush.zero_()
                    return call()
            else:
                one = call
            one()  # eager warm-up: cuBLAS handles and workspaces must not be created during the capture
            torch.cuda.synchronize()
            gr = build_graph(one, N)
            per = time_graph(gr, N, cpu_group, target_s=0.05, trials=args.trials)
            row[name] = round(med_over_ranks(per, cpu_group) - base_flush, 2)
        comm.gemv_push = 1
        comm.set_gemv_variant(None)
        comm._c.set_trigger(comm.gemv_trigger)
        rows.append(row)
        log(rank, json.dumps(row))
    if rank == 0 and args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    comm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=lambda s: [int(v) for v in s.split(",")], default=[1, 2, 4, 8])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--triggers", type=lambda s: [int(v) for v in s.split(",")], default=[3],
                    help="trigger stages of the all-reduce kernels in front of the GEMV kernel (1 start, 2 last phase begins, 3 ends)")
    ap.add_argument("--variants", type=lambda s: [int(v) for v in s.split(",")], default=[0],
                    help="geometry variants of the GEMV kernels to time (csrc/gemv_push.cu)")
    ap.add_argument("--out", default="")
    ap.add_argument("--port", type=int, default=29771)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
