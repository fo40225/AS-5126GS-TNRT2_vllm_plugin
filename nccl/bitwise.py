#!/usr/bin/env python3
"""Bitwise-identity test of NCCL all-reduce (vLLM PyNcclCommunicator, 8 ranks).

usage: bitwise.py --out RESULT.json [--ref REF.pt | --save-ref REF.pt] [--nccl-so PATH]
NCCL_* env vars are inherited from the caller. The communicator is bootstrapped
from a gloo group exactly like vLLM does. Every shape is reduced eagerly and,
for small shapes, also inside a CUDA graph (as decode does).
"""
import argparse, hashlib, json, os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
SHAPES = [(8192, 8192), (4096, 8192), (8192, 4096), (2048, 8192), (1024, 8192), (512, 8192),
          (256, 8192), (128, 8192), (100, 8192), (64, 8192), (56, 8192), (48, 8192), (40, 8192),
          (33, 8192), (32, 8192), (24, 8192), (16, 8192), (8, 8192), (3, 8192), (1, 8192),
          (4096, 4096), (1024, 4096), (256, 4096), (128, 4096), (64, 4096), (32, 4096), (1, 4096)]
GRAPH_MAX_ELEMS = 2048 * 8192
REPEATS = 3


def make_input(shape, idx, rank):
    g = torch.Generator(device="cuda").manual_seed(1000 * (idx + 1) + rank)
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.bfloat16)


def md5(t):
    return hashlib.md5(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def worker(rank, args, port):
    if args.small:
        rows = list(range(1, 41)) + [48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 256]
        SHAPES[:] = [(r, w) for w in (8192, 4096) for r in rows]
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    comm = PyNcclCommunicator(dist.group.WORLD, device=rank, library_path=args.nccl_so)
    if rank == 0:
        print("nccl_version", comm.nccl_version, flush=True)
    ref = torch.load(args.ref, map_location="cuda") if args.ref else None
    outs, res = {}, {"nccl_version": comm.nccl_version, "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}, "shapes": {}}
    stream = torch.cuda.current_stream()
    for idx, shape in enumerate(SHAPES):
        x = make_input(shape, idx, rank)
        entry = {}
        for mode in ("eager", "graph"):
            if mode == "graph" and shape[0] * shape[1] > GRAPH_MAX_ELEMS:
                continue
            hs = []
            last = None
            for _ in range(REPEATS):
                if mode == "eager":
                    y = comm.all_reduce(x, stream=stream)
                    torch.cuda.synchronize()
                else:
                    # Warm up outside capture, then capture and replay one all-reduce.
                    comm.all_reduce(x, stream=stream)
                    torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        y = comm.all_reduce(x, stream=torch.cuda.current_stream())
                    gr.replay()
                    torch.cuda.synchronize()
                last = y.clone()
                hs.append(md5(last))
            entry[mode] = {"md5": hs[0], "repeat_identical": len(set(hs)) == 1}
            key = f"{shape[0]}x{shape[1]}"
            if mode == "eager":
                outs[key] = last
            if ref is not None and mode == "eager":
                r = ref[key]
                d = (last.float() - r.float()).abs()
                entry[mode]["n_diff_vs_ref"] = int((last.view(torch.int16) != r.view(torch.int16)).sum().item())
                entry[mode]["max_abs_diff_vs_ref"] = float(d.max().item())
                entry[mode]["numel"] = last.numel()
            if mode == "graph" and "eager" in entry:
                entry["graph_eq_eager"] = entry["graph"]["md5"] == entry["eager"]["md5"]
            del y
        res["shapes"][f"{shape[0]}x{shape[1]}"] = entry
        del x
    # Cross-rank agreement and (rank-local) diff counts.
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, res["shapes"])
    if rank == 0:
        for key, ent in res["shapes"].items():
            md5s = {g[key]["eager"]["md5"] for g in gathered}
            ent["all_ranks_identical"] = len(md5s) == 1
            if ref is not None:
                ent["n_diff_per_rank"] = [g[key]["eager"]["n_diff_vs_ref"] for g in gathered]
                ent["max_abs_diff_per_rank"] = [g[key]["eager"]["max_abs_diff_vs_ref"] for g in gathered]
        if args.save_ref:
            torch.save({k: v.cpu() for k, v in outs.items()}, args.save_ref)
        json.dump(res, open(args.out, "w"), indent=1)
        print("wrote", args.out, flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--save-ref")
    ap.add_argument("--nccl-so")
    ap.add_argument("--port", type=int, default=29517)
    ap.add_argument("--small", action="store_true", help="decode-sized sweep: rows 1..40 and CUDA-graph batch sizes, widths 8192 and 4096")
    args = ap.parse_args()
    mp.spawn(worker, args=(args, args.port), nprocs=WORLD, join=True)
