"""8-process bitwise test of TopoComm.all_reduce_boundary (all-reduce fused with the Muse Spark residual boundary).

Reference = ``TopoComm.all_reduce`` followed by vLLM's Triton ``fused_norm_residual_norm_fwd`` (the unfused production path):
the fused kernel must give the same bits for ``out`` (bf16), ``new_stream`` (fp32) and ``out_fp32`` on every rank, eager and in
a CUDA graph, for T = 1..64, both boundary variants (post-norm without weight / weight-first post-norm), mixed with other
collectives (counter and double-buffer alignment). The bf16 exchange is pinned off in the test (it would turn the
unfused reference of T = 4 .. 8 into a bf16-exchange all-reduce, which the always-exact fused kernel does not match).

usage (server, under the GPU lock): GPU=0,1,2,3,4,5,6,7 SRC=<vllm tree> scripts/py_topo.sh bench/test_ar_boundary.py [--quick]
"""
import argparse
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

RESULTS = []
D = 8192


def check(rank, name, ok):
    RESULTS.append((name, bool(ok)))
    if rank == 0 or not ok:
        print(("PASS " if ok else f"FAIL[r{rank}] ") + name, flush=True)


def worker(rank, args):
    try:
        run(rank, args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"[rank {rank}] checks={len(RESULTS)} failures={len(bad)}", flush=True)
    if bad:
        sys.exit(1)


def make_params(dev, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    alpha = (torch.rand(D, generator=g, device=dev) * 0.9 + 0.1).float()
    beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
    post_w = (torch.randn(D, generator=g, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
    pre_w = (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    return alpha, beta, post_w, pre_w


def run(rank, args):
    dev, cpu_group, _ = init_rank(rank, args.port, nccl=False)
    from muse_topo import TopoComm
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd

    # The reference is the exact comm.all_reduce, which the fused kernel always matches. env/host.env turns the bf16
    # exchange on (large and medium all-reduces, T = 4 .. 8 included): pin it off whatever the environment says.
    comm = TopoComm(cpu_group, dev, rank, WORLD, bf16_exchange=False)
    log(rank, "TopoComm up:", comm.info.summary)
    Ts = [1, 2, 3, 4, 5, 8, 9, 16, 24, 33, 48, 64] if not args.quick else [1, 8, 64]
    alpha, beta, post_w, pre_w = make_params(dev, 5)
    cfgs = [(1e-8, 1e-6, 1.0, 1.0), (1e-6, 1e-6, 0.0, 0.0)] if not args.quick else [(1e-8, 1e-6, 1.0, 1.0)]

    def unfused(x, stream, post_scale, fp32, cfg):
        post_eps, pre_eps, post_gain, pre_gain = cfg
        y = comm.all_reduce(x)
        f32 = torch.empty_like(stream) if fp32 else None
        out, new = fused_norm_residual_norm_fwd(
            y, stream, post_w if post_scale else None, pre_w, post_eps, pre_eps, alpha=alpha, beta=beta,
            post_gain_center=post_gain if post_scale else 0.0, pre_gain_center=pre_gain, post_input_scale=post_scale,
            pre_input_scale=False, out_dtype=torch.bfloat16, out_fp32=f32)
        return out, new, f32

    def fused(x, stream, post_scale, fp32, cfg):
        post_eps, pre_eps, post_gain, pre_gain = cfg
        f32 = torch.empty_like(stream) if fp32 else None
        out, new = comm.all_reduce_boundary(x, stream, alpha, beta, pre_w, post_w if post_scale else None, post_eps,
                                            pre_eps, post_gain if post_scale else 0.0, pre_gain, out_fp32=f32)
        return out, new, f32

    for kind in ("mixed", "randn"):
        for T in Ts:
            xs = [make_input(r, (T, D), 300 + T, dev, torch.bfloat16, kind) for r in range(WORLD)]
            g = torch.Generator(device=dev)
            g.manual_seed(900 + T)
            stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
            for cfg in cfgs:
                for post_scale in (False, True):
                    for fp32 in (False, True):
                        assert comm.supports_all_reduce_boundary(xs[rank], stream)
                        ro, rn, rf = unfused(xs[rank], stream, post_scale, fp32, cfg)
                        fo, fn, ff = fused(xs[rank], stream, post_scale, fp32, cfg)
                        torch.cuda.synchronize()
                        ok = bits_equal(fo, ro) and bits_equal(fn, rn) and (not fp32 or bits_equal(ff, rf))
                        same = all_equal_across_ranks(fo, cpu_group) and all_equal_across_ranks(fn, cpu_group)
                        check(rank, f"fused == unfused {kind} T={T} post_scale={post_scale} fp32={fp32} cfg={cfg[:2]}",
                              ok and same)

    # ---- all-gather fused with the boundary (unpooler shards) and with pre_expert_norm (pooler shards) -------------
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_rmsnorm_fwd

    w4 = (torch.randn(4096, generator=torch.Generator(device=dev).manual_seed(77), device=dev) * 0.3).to(torch.bfloat16)
    for kind in ("mixed", "randn"):
        for T in ([1, 2, 3, 4, 8, 13, 16, 24, 32] if not args.quick else [1, 8, 32]):
            shards = [make_input(r, (T, 1024), 400 + T, dev, torch.bfloat16, kind) for r in range(WORLD)]
            g = torch.Generator(device=dev)
            g.manual_seed(950 + T)
            stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
            for cfg in cfgs:
                for post_scale in (False, True):
                    for fp32 in (False, True):
                        assert comm.supports_all_gather_boundary(shards[rank], stream)
                        post_eps, pre_eps, post_gain, pre_gain = cfg
                        y = comm.all_gather(shards[rank], -1)
                        f32 = torch.empty_like(stream) if fp32 else None
                        ro, rn = fused_norm_residual_norm_fwd(
                            y, stream, post_w if post_scale else None, pre_w, post_eps, pre_eps, alpha=alpha, beta=beta,
                            post_gain_center=post_gain if post_scale else 0.0, pre_gain_center=pre_gain,
                            post_input_scale=post_scale, pre_input_scale=False, out_dtype=torch.bfloat16, out_fp32=f32)
                        f32b = torch.empty_like(stream) if fp32 else None
                        fo, fn = comm.all_gather_boundary(shards[rank], stream, alpha, beta, pre_w,
                                                          post_w if post_scale else None, post_eps, pre_eps,
                                                          post_gain if post_scale else 0.0, pre_gain, out_fp32=f32b)
                        torch.cuda.synchronize()
                        ok = bits_equal(fo, ro) and bits_equal(fn, rn) and (not fp32 or bits_equal(f32b, f32))
                        same = all_equal_across_ranks(fo, cpu_group) and all_equal_across_ranks(fn, cpu_group)
                        check(rank, f"ag+boundary == unfused {kind} T={T} post_scale={post_scale} fp32={fp32} cfg={cfg[:2]}",
                              ok and same)
    for kind in ("mixed", "randn"):
        for T in ([1, 2, 3, 4, 8, 13, 16, 24, 32, 33, 48, 64] if not args.quick else [1, 8, 64]):
            shards = [make_input(r, (T, 512), 600 + T, dev, torch.bfloat16, kind) for r in range(WORLD)]
            for eps, gain in ((1e-6, 1.0), (1e-8, 0.0), (1e-6, 0.5)):
                assert comm.supports_all_gather_rmsnorm(shards[rank])
                y = comm.all_gather(shards[rank], -1)
                ref = fused_rmsnorm_fwd(y, w4, eps, gain_center=gain, input_scale=False, use_kahan_summation=True)
                got = comm.all_gather_rmsnorm(shards[rank], w4, eps, gain)
                torch.cuda.synchronize()
                check(rank, f"ag+rmsnorm4096 == unfused {kind} T={T} eps={eps} gain={gain}",
                      bits_equal(got, ref) and all_equal_across_ranks(got, cpu_group))

    # ---- misaligned operands on some ranks only: staged through aligned copies, the gates stay rank-uniform, no hang ----------
    def misaligned(t):
        flat = torch.empty(t.numel() + 8, dtype=t.dtype, device=t.device)
        v = flat[1:1 + t.numel()].view(t.shape)
        assert v.data_ptr() % 16 != 0
        v.copy_(t)
        return v

    for T in (1, 4, 8):
        xs = [make_input(r, (T, D), 800 + T, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
        g = torch.Generator(device=dev)
        g.manual_seed(2900 + T)
        stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
        shard = make_input(rank, (T, 1024), 900 + T, dev, torch.bfloat16, "mixed")
        pshard = make_input(rank, (T, 512), 950 + T, dev, torch.bfloat16, "mixed")
        cfg = cfgs[0]
        ro, rn, rf = unfused(xs[rank], stream, True, True, cfg)
        gy = comm.all_gather(shard, -1)
        go, gn = fused_norm_residual_norm_fwd(gy, stream, post_w, pre_w, cfg[0], cfg[1], alpha=alpha, beta=beta,
                                              post_gain_center=cfg[2], pre_gain_center=cfg[3], post_input_scale=True,
                                              pre_input_scale=False, out_dtype=torch.bfloat16)
        pref = fused_rmsnorm_fwd(comm.all_gather(pshard, -1), w4, 1e-6, gain_center=1.0, use_kahan_summation=True)
        bad = rank % 2 == 1  # odd ranks: every operand misaligned
        m = misaligned if bad else (lambda t: t)
        x_, st_, al_, be_, pw_, qw_ = m(xs[rank]), m(stream), m(alpha), m(beta), m(pre_w), m(post_w)
        f32 = m(torch.empty_like(stream))
        assert comm.supports_all_reduce_boundary(x_, st_) and comm.boundary_params_ok(al_, be_, pw_, qw_, f32)
        fo, fn = comm.all_reduce_boundary(x_, st_, al_, be_, pw_, qw_, cfg[0], cfg[1], cfg[2], cfg[3], out_fp32=f32)
        sh_ = m(shard)
        assert comm.supports_all_gather_boundary(sh_, st_)
        ao, an = comm.all_gather_boundary(sh_, st_, al_, be_, pw_, qw_, cfg[0], cfg[1], cfg[2], cfg[3])
        ps_, w4_ = m(pshard), m(w4)
        assert comm.supports_all_gather_rmsnorm(ps_)
        po = comm.all_gather_rmsnorm(ps_, w4_, 1e-6, 1.0)
        torch.cuda.synchronize()
        ok = (bits_equal(fo, ro) and bits_equal(fn, rn) and bits_equal(f32, rf) and bits_equal(ao, go)
              and bits_equal(an, gn) and bits_equal(po, pref))
        check(rank, f"misaligned operands on odd ranks, T={T}", ok and all_equal_across_ranks(fo, cpu_group))

    # ---- CUDA graph: replays equal the eager result, interleaved with other collectives --------------------------
    for T in ([1, 4, 16, 64] if not args.quick else [1, 64]):
        xs = [make_input(r, (T, D), 700 + T, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
        x = xs[rank].clone()
        g = torch.Generator(device=dev)
        g.manual_seed(1900 + T)
        stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
        cfg = cfgs[0]
        post_scale = True
        ro, rn, _ = unfused(x, stream, post_scale, False, cfg)
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fused(x, stream, post_scale, False, cfg)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            cap = []
            for _ in range(3):
                cap.append(fused(x, stream, post_scale, False, cfg))
                comm.all_reduce(x)
        okall = True
        for rep in range(20):
            gr.replay()
            # eager calls between replays advance the epoch / parity: a different parity of the double buffers each time
            if rep % 3 == 0:
                comm.all_reduce(x)
            if rep % 4 == 0:
                comm.all_gather(x[:, :1024].contiguous(), -1)
            torch.cuda.synchronize()
            for fo, fn, _ in cap:
                okall = okall and bits_equal(fo, ro) and bits_equal(fn, rn)
        check(rank, f"graph replays == unfused, T={T}", okall)

    # ---- random sequences mixing fused and plain collectives -------------------------------------------------------
    import random

    rng = random.Random(1234)  # same sequence on every rank
    okseq = True
    for it in range(args.seq):
        T = rng.choice([1, 2, 6, 17, 40, 64])
        op = rng.choice(["fused", "ar", "ag", "fused"])
        xs = [make_input(r, (T, D), 5000 + it, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
        g = torch.Generator(device=dev)
        g.manual_seed(it)
        stream = torch.randn(T, D, generator=g, device=dev).float()
        if op == "fused":
            ro, rn, _ = unfused(xs[rank], stream, True, False, cfgs[0])
            fo, fn, _ = fused(xs[rank], stream, True, False, cfgs[0])
            torch.cuda.synchronize()
            okseq = okseq and bits_equal(fo, ro) and bits_equal(fn, rn)
        elif op == "ar":
            comm.all_reduce(xs[rank])
        else:
            comm.all_gather(xs[rank][:, :512].contiguous(), -1)
    torch.cuda.synchronize()
    check(rank, f"random sequence of {args.seq} mixed collectives", okseq)
    comm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seq", type=int, default=300)
    ap.add_argument("--port", type=int, default=29731)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
