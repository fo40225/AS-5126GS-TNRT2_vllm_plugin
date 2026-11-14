"""Standalone test of the muse_topo vLLM plugin on the real 8-GPU box (no model): 8 processes build vLLM's TP group
through the plugin entry point (``load_general_plugins`` finds muse_topo-0.1.dist-info on PYTHONPATH), then issue the
collectives the Muse model issues through ``tensor_model_parallel_all_reduce`` / ``tensor_model_parallel_all_gather`` /
``tp.reduce_scatter``: eager, inside ``graph_capture`` + CUDA graph replays, on a side stream, mixed with fallback
(NCCL) calls. Results are compared bitwise against the arithmetic contract (all_reduce, reduce_scatter blocks), the exact
concatenation (all_gather) and NCCL. Since H2b the large-message algorithms are part of the checks: prefill-sized
all_reduce (16-128 MiB), reduce_scatter, all_gather with big shards (pooler / unpooler / logits of a prefill chunk),
reduce_scatter + all_gather == all_reduce, and a CUDA graph that contains large collectives.

usage (server, under the GPU lock): GPU=0,1,2,3,4,5,6,7 SRC=<vllm tree> scripts/runsrc_topo.sh bench/test_plugin.py [--quick]
The checks are the exact fp32 arithmetic contract, so the run pins MUSE_TOPO_LARGE_BF16_EXCHANGE=0 (env/host.env turns the bf16
exchange ON for serving, under which the large-message contract checks fail by design); --keep-bf16-exchange keeps the exported value.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (WORLD, all_equal_across_ranks, bits_equal, contract_ref, gather_all, log,  # noqa: E402
                    make_input)

AR_SHAPES = [(1, 8192), (2, 8192), (4, 8192), (8, 4096), (24, 8192), (64, 8192), (128, 4096), (300, 8192),
             (512, 8192)]
AG_DIM0 = [(1, 512), (8, 512), (64, 512), (256, 512), (1, 1024), (8, 1024), (128, 1024)]  # pooler / unpooler shards
AG_LAST = [(1, 25256), (4, 25256), (1, 1024), (8, 512)]  # logits shard and column concatenation


class Checker:
    def __init__(self, rank):
        self.rank, self.fail, self.n = rank, 0, 0

    def check(self, ok, what):
        self.n += 1
        if not ok:
            self.fail += 1
            print(f"[rank {self.rank}] FAIL {what}", flush=True)
        return ok


def ar_err(y, xs):
    """Largest elementwise |y - exact| / sum_r |x_r| (exact = fp64 sum; 2^-9 bounds a single final bf16 rounding)."""
    exact = sum(t.double() for t in xs)
    l1 = sum(t.double().abs() for t in xs)
    return ((y.double() - exact).abs() / l1.clamp_min(1e-300)).max().item()


def ar_close(y, xs, tol_rel=2 ** -6):
    return ar_err(y, xs) <= tol_rel


def reference_ar(shape, seed, dtype, dev, members, kind="randn"):
    xs = [make_input(r, shape, seed, dev, dtype, kind) for r in range(WORLD)]
    return xs, contract_ref(xs, members)


def worker(rank, port, quick):
    ck = Checker(rank)
    try:
        run(rank, port, quick, ck)
    except BaseException:
        traceback.print_exc()
        ck.fail += 1
    finally:
        print(f"[rank {rank}] checks={ck.n} failures={ck.fail}", flush=True)
    if ck.fail:
        sys.exit(1)


def run(rank, port, quick, ck):
    wd = int(os.environ.get("BENCH_WATCHDOG_S", "0"))
    if wd > 0:
        import faulthandler

        faulthandler.dump_traceback_later(wd, exit=True)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")  # vLLM's logging is not configured here
    logging.getLogger("vllm.muse_topo").setLevel(logging.INFO)

    # 1) plugin discovery exactly as vLLM does it (entry point in muse_topo-0.1.dist-info)
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    ck.check(CudaCommunicator.__dict__.get("_muse_topo_patched") is True, "plugin patched CudaCommunicator")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (ensure_model_parallel_initialized, init_distributed_environment,
                                  tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
    from vllm.distributed.parallel_state import get_tp_group, graph_capture

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=WORLD, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                     local_rank=rank, backend="nccl")
        ensure_model_parallel_initialized(WORLD, 1)
    tp = get_tp_group()
    dc = tp.device_communicator
    comm = dc.topo_comm
    ck.check(comm is not None, "TopoComm created for the TP group")
    # the capability the sequence-parallel stream asks for (False on vLLM's base class, a property of the patched one)
    ck.check(dc.reduce_scatter_all_gather_matches_all_reduce is True, "capability RS + AG == AR advertised")
    saved = dc.topo_comm
    dc.topo_comm = None
    ck.check(dc.reduce_scatter_all_gather_matches_all_reduce is False, "capability off without a TopoComm")
    dc.topo_comm = saved
    # the capability is about the three collectives; the fused boundary op has no part in it
    saved_ops = dc.topo_ops
    dc.topo_ops = frozenset(saved_ops) - {"boundary"}
    ck.check(dc.reduce_scatter_all_gather_matches_all_reduce is True, "capability stays on without the boundary op")
    dc.topo_ops = frozenset(saved_ops) - {"reduce_scatter"}
    ck.check(dc.reduce_scatter_all_gather_matches_all_reduce is False, "capability off without reduce_scatter routed")
    dc.topo_ops = saved_ops
    log(rank, "TP group", tp.unique_name, "topo_comm", type(comm).__name__, "workspace MiB",
        f"{comm.workspace_bytes / 2**20:.1f}", comm.info.summary)
    members = comm.info.members
    cpu = dc.cpu_group
    cur = torch.cuda.current_stream()
    seed = [100]

    def nseed():
        seed[0] += 1
        return seed[0]

    def nccl_ar(x):  # the original vLLM chain (NCCL), by switching the topo path off for one call
        c, dc.topo_comm = dc.topo_comm, None
        try:
            return tensor_model_parallel_all_reduce(x)
        finally:
            dc.topo_comm = c

    def nccl_ag(x, dim):
        c, dc.topo_comm = dc.topo_comm, None
        try:
            return tensor_model_parallel_all_gather(x, dim)
        finally:
            dc.topo_comm = c

    # 2) eager all_reduce: bitwise == contract, identical on all ranks, close to NCCL
    for dtype in (torch.bfloat16, torch.float16):
        for shape in AR_SHAPES if dtype == torch.bfloat16 else AR_SHAPES[:3]:
            xs, ref = reference_ar(shape, nseed(), dtype, dev, members, "mixed")
            y = tensor_model_parallel_all_reduce(xs[rank])
            ck.check(bits_equal(y, ref), f"eager AR {dtype} {shape} == contract")
            ck.check(all_equal_across_ranks(y, cpu), f"eager AR {dtype} {shape} identical on all ranks")
            if dtype == torch.bfloat16:
                e_topo, e_nccl = ar_err(y, xs), ar_err(nccl_ar(xs[rank]), xs)
                ck.check(e_topo <= 2 ** -8, f"AR {shape} topo error {e_topo:.3g} of L1 > 2^-8")
                ck.check(e_nccl <= 2 ** -4, f"AR {shape} NCCL error {e_nccl:.3g} of L1 > 2^-4")
                if rank == 0:
                    print(f"AR {shape} max err/L1: topo {e_topo:.3g} nccl {e_nccl:.3g} (bf16 unit roundoff = {2 ** -8:.3g})",
                          flush=True)
    # 3) eager all_gather: exact, both layouts
    for shape in AG_DIM0:
        xs = [make_input(r, shape, nseed(), dev) for r in range(WORLD)]
        y = tensor_model_parallel_all_gather(xs[rank], 0)
        ck.check(bits_equal(y, torch.cat(xs, 0)), f"eager AG dim0 {shape}")
    for shape in AG_LAST:
        xs = [make_input(r, shape, nseed(), dev) for r in range(WORLD)]
        y = tensor_model_parallel_all_gather(xs[rank], -1)
        ck.check(bits_equal(y, torch.cat(xs, -1)), f"eager AG dim-1 {shape}")
        ck.check(bits_equal(y, nccl_ag(xs[rank], -1)), f"eager AG dim-1 {shape} == NCCL")

    # 4) CUDA graph (vLLM graph_capture: capture stream), then replays on the main stream, mixed with eager calls
    shapes_ar = [(1, 8192), (1, 4096), (8, 8192), (64, 4096)]
    static_ar = [torch.zeros(s, dtype=torch.bfloat16, device=dev) for s in shapes_ar]
    static_ag0 = [torch.zeros(s, dtype=torch.bfloat16, device=dev) for s in ((1, 512), (8, 1024))]
    static_agl = torch.zeros((1, 25256), dtype=torch.bfloat16, device=dev)
    graph = torch.cuda.CUDAGraph()
    for _ in range(2):  # warm-up on the capture stream (eager) like vLLM does
        with graph_capture(device=dev) as gctx:
            with torch.cuda.stream(gctx.stream):
                for s in static_ar:
                    tensor_model_parallel_all_reduce(s)
    with graph_capture(device=dev) as gctx:
        with torch.cuda.graph(graph, stream=gctx.stream):
            outs_ar = [tensor_model_parallel_all_reduce(s) for s in static_ar]
            outs_ag0 = [tensor_model_parallel_all_gather(s, 0) for s in static_ag0]
            outs_agl = tensor_model_parallel_all_gather(static_agl, -1)
            outs_ar2 = [tensor_model_parallel_all_reduce(s) for s in static_ar]  # second pass: counter parity
    for it in range(6 if quick else 40):
        xs_ar = [[make_input(r, s, nseed(), dev, kind="mixed") for r in range(WORLD)] for s in shapes_ar]
        xs_ag0 = [[make_input(r, s, nseed(), dev) for r in range(WORLD)] for s in ((1, 512), (8, 1024))]
        xs_agl = [make_input(r, (1, 25256), nseed(), dev) for r in range(WORLD)]
        if it % 3 == 1:  # an eager collective between replays
            e = make_input(rank, (4, 8192), nseed(), dev)
            tensor_model_parallel_all_reduce(e)
        for s, xs in zip(static_ar, xs_ar):
            s.copy_(xs[rank])
        for s, xs in zip(static_ag0, xs_ag0):
            s.copy_(xs[rank])
        static_agl.copy_(xs_agl[rank])
        graph.replay()
        for o, o2, xs in zip(outs_ar, outs_ar2, xs_ar):
            ref = contract_ref(xs, members)
            ck.check(bits_equal(o, ref) and bits_equal(o2, ref), f"graph AR {tuple(o.shape)} == contract (iter {it})")
        for o, xs in zip(outs_ag0, xs_ag0):
            ck.check(bits_equal(o, torch.cat(xs, 0)), f"graph AG dim0 {tuple(o.shape)} (iter {it})")
        ck.check(bits_equal(outs_agl, torch.cat(xs_agl, -1)), f"graph AG logits (iter {it})")
    torch.cuda.synchronize()

    # 5) fallbacks must give correct NCCL results (and not disturb the topo call sequence)
    xs = [make_input(r, (1, 100), nseed(), dev) for r in range(WORLD)]  # 200 B: not a multiple of 64
    ck.check(ar_close(tensor_model_parallel_all_reduce(xs[rank]), xs), "fallback AR 200 B")
    xs = [make_input(r, (1, (8 << 20) // 2 + 32), nseed(), dev) for r in range(WORLD)]  # 8 MiB + 64 B: no 128 B multiple
    ck.check(ar_close(tensor_model_parallel_all_reduce(xs[rank]), xs), "fallback AR 8 MiB + 64 B")
    del xs
    xi = torch.full((4, 64), rank + 1, dtype=torch.int32, device=dev)
    ck.check(bool((tensor_model_parallel_all_reduce(xi) == sum(range(1, WORLD + 1))).all()), "fallback AR int32")
    ncs = [make_input(r, (8, 2048), nseed(), dev)[:, :1024] for r in range(WORLD)]  # non-contiguous shard
    ck.check(bits_equal(tensor_model_parallel_all_gather(ncs[rank], 0), torch.cat(ncs, 0)), "fallback AG non-contiguous")
    xs = [make_input(r, (4, 3, 64), nseed(), dev) for r in range(WORLD)]
    ck.check(bits_equal(tensor_model_parallel_all_gather(xs[rank], 1), torch.cat(xs, 1)), "fallback AG middle dim")
    xs = [make_input(r, (64, 25256), nseed(), dev) for r in range(WORLD)]  # shard 3.2 MB > 1 MiB: large all-gather
    ck.check(bits_equal(tensor_model_parallel_all_gather(xs[rank], -1), torch.cat(xs, -1)), "large AG logits shard")
    xs = [make_input(r, (1, 8192), nseed(), dev, torch.float32) for r in range(WORLD)]
    ck.check(bits_equal(tensor_model_parallel_all_gather(xs[rank], 0), torch.cat(xs, 0)), "AG fp32 dim0")

    # 5b) rank-local alignment must not change the path: every rank passes a differently aligned view of the same shape
    # (a rank on NCCL while the others run topo kernels would deadlock); results exact, all counted on the topo path
    counts_a = dict(dc.topo_trace.counts)
    abuf = torch.zeros((1 << 20) + 64, dtype=torch.bfloat16, device=dev)
    off = (0, 1, 3, 5, 2, 7, 4, 6)[rank]  # element offsets: only offset 0 is 16-byte aligned
    xs = [make_input(r, (8, 4096), nseed(), dev) for r in range(WORLD)]
    v = abuf[off:off + 8 * 4096].view(8, 4096)
    v.copy_(xs[rank])
    ck.check(bits_equal(tensor_model_parallel_all_reduce(v), contract_ref(xs, members)), "AR misaligned view == contract")
    xs = [make_input(r, (8, 512), nseed(), dev) for r in range(WORLD)]
    v = abuf[off:off + 8 * 512].view(8, 512)
    v.copy_(xs[rank])
    ck.check(bits_equal(tensor_model_parallel_all_gather(v, 0), torch.cat(xs, 0)), "AG dim0 misaligned view")
    ck.check(bits_equal(tensor_model_parallel_all_gather(v, -1), torch.cat(xs, -1)), "AG dim-1 misaligned view")
    xs = [make_input(r, (64, 4096), nseed(), dev) for r in range(WORLD)]
    v = abuf[off:off + 64 * 4096].view(64, 4096)
    v.copy_(xs[rank])
    ck.check(bits_equal(tp.reduce_scatter(v, 0), contract_ref(xs, members)[rank * 8:(rank + 1) * 8]),
             "RS misaligned view == contract block")
    nccl_a = {k: n for k, n in dc.topo_trace.counts.items() if k[6] == "nccl" and n > counts_a.get(k, 0)}
    ck.check(not nccl_a, f"misaligned views took the NCCL path: {list(nccl_a)[:3]}")
    del abuf, v, xs

    # 6) reduce_scatter wrapper: topo path (large reduce-scatter), bitwise == rows block of the contract
    xs = [make_input(r, (64, 8192), nseed(), dev, kind="mixed") for r in range(WORLD)]
    y = tp.reduce_scatter(xs[rank], 0)
    blk = 64 // WORLD
    ref = contract_ref(xs, members)[rank * blk:(rank + 1) * blk]
    topo_rs = comm.supports("reduce_scatter", xs[rank], 0)
    ck.check(topo_rs, "reduce_scatter [64, 8192] is eligible for the topo path")
    ck.check(bits_equal(y, ref), "RS == contract block")
    log(rank, "reduce_scatter routed through topo:", topo_rs)
    # dim != 0 stays on NCCL (correct, close to exact)
    xs = [make_input(r, (16, 8192), nseed(), dev, kind="mixed") for r in range(WORLD)]
    y = tp.reduce_scatter(xs[rank], 1)
    exact = sum(t.double() for t in xs)[:, rank * 1024:(rank + 1) * 1024]
    ck.check(y.shape == exact.shape and (y.double() - exact).abs().max().item() <= exact.abs().max().item() * 2 ** -6,
            "RS dim 1 (NCCL fallback) close to exact")

    # 6b) large messages through the vLLM wrappers (prefill chunks of the Muse model)
    counts0 = dict(dc.topo_trace.counts)  # fallbacks counted so far are the deliberate ones of sections 5 and 6
    large_ar = [((1024, 8192), torch.bfloat16, "randn"),    # 16 MiB
                ((2048, 8192), torch.float16, "randn"),     # 32 MiB
                ((8192, 4096), torch.bfloat16, "mixed"),    # 64 MiB: moe-out of an 8192-token chunk
                ((8192, 8192), torch.bfloat16, "randn"),    # 128 MiB: attn-out of an 8192-token chunk
                ((1234, 8192), torch.bfloat16, "mixed"),    # prefill tail chunk
                ((512, 8192), torch.float32, "randn")]      # 16 MiB fp32
    if quick:
        large_ar = large_ar[:3] + large_ar[4:5]
    for shape, dtype, kind in large_ar:
        xs, ref = reference_ar(shape, nseed(), dtype, dev, members, kind)
        nb = xs[0].numel() * xs[0].element_size()
        ck.check(comm.supports("all_reduce", xs[rank]), f"large AR {dtype} {shape} eligible")
        y = tensor_model_parallel_all_reduce(xs[rank])
        ck.check(bits_equal(y, ref), f"large AR {dtype} {shape} == contract")
        ck.check(all_equal_across_ranks(y, cpu), f"large AR {dtype} {shape} identical on all ranks")
        if nb <= 64 << 20 and dtype == torch.bfloat16 and kind == "randn":
            ck.check(ar_err(y, xs) <= 2 ** -8, f"large AR {shape} error within the single-rounding bound")
        # reduce_scatter of the same data: rows block r of the contract, and RS + AG(dim 0) == AR bitwise
        t_rows = shape[0] - shape[0] % WORLD
        xr = [x[:t_rows].contiguous() for x in xs]
        ref_r = contract_ref(xr, members)
        yr = tp.reduce_scatter(xr[rank], 0)
        b = t_rows // WORLD
        ck.check(bits_equal(yr, ref_r[rank * b:(rank + 1) * b]), f"large RS {dtype} {shape} == contract block")
        full = tensor_model_parallel_all_gather(yr, 0)
        ck.check(bits_equal(full, ref_r), f"RS + AG(dim 0) {dtype} {shape} == AR contract")
        del xs, xr, y, yr
    # the capability itself, on the shapes of the sequence-parallel stream (attention output [T, 8192], expert output
    # [T, 4096], T = a multiple of 8 from the smallest sharded batch up to a full chunk, and a mid size that is not a
    # multiple of 128 rows): all_gather(reduce_scatter(x)) is bit for bit all_reduce(x), without going through the contract
    for shape in ((520, 8192), (1024, 4096), (3000, 8192), (4104, 4096), (8192, 8192), (8192, 4096)):
        xs = [make_input(r, shape, nseed(), dev) for r in range(WORLD)]
        ar = tensor_model_parallel_all_reduce(xs[rank].clone())
        rsag = tensor_model_parallel_all_gather(tp.reduce_scatter(xs[rank], 0), 0)
        ck.check(bits_equal(rsag, ar), f"AG(RS(x)) == AR(x) bitwise {shape}")
        ck.check(all_equal_across_ranks(rsag, cpu), f"AG(RS(x)) {shape} identical on all ranks")
        del xs, ar, rsag
    # big all-gather shards: unpooler / pooler of a prefill chunk (dim 0) and the logits (dim -1)
    for shape, dim in (((8192, 512), 0), ((4096, 1024), 0), ((64, 25256), -1), ((256, 25256), -1), ((2048, 1024), -1)):
        xs = [make_input(r, shape, nseed(), dev) for r in range(WORLD)]
        y = tensor_model_parallel_all_gather(xs[rank], dim)
        ck.check(comm.supports("all_gather", xs[rank], dim % 2), f"large AG {shape} dim {dim} eligible")
        ck.check(bits_equal(y, torch.cat(xs, dim)), f"large AG {shape} dim {dim} exact")
        del xs, y
    # every one of the above must have been counted on the topo path, none on NCCL
    nccl_new = {k: n - counts0.get(k, 0) for k, n in dc.topo_trace.counts.items()
                if k[6] == "nccl" and n > counts0.get(k, 0)}
    ck.check(not nccl_new, f"large collectives took the NCCL path: {list(nccl_new)[:3]}")
    # a CUDA graph with a large all_reduce, reduce_scatter and all_gather (replayed next to eager large calls)
    g_ar = torch.zeros((1024, 8192), dtype=torch.bfloat16, device=dev)
    g_rs = torch.zeros((512, 8192), dtype=torch.bfloat16, device=dev)
    g_ag = torch.zeros((1024, 512), dtype=torch.bfloat16, device=dev)
    graph_l = torch.cuda.CUDAGraph()
    for _ in range(2):
        with graph_capture(device=dev) as gctx:
            with torch.cuda.stream(gctx.stream):
                tensor_model_parallel_all_reduce(g_ar)
    with graph_capture(device=dev) as gctx:
        with torch.cuda.graph(graph_l, stream=gctx.stream):
            o_ar = tensor_model_parallel_all_reduce(g_ar)
            o_rs = tp.reduce_scatter(g_rs, 0)
            o_ag = tensor_model_parallel_all_gather(g_ag, 0)
    for it in range(2 if quick else 5):
        xa = [make_input(r, (1024, 8192), nseed(), dev, kind="mixed") for r in range(WORLD)]
        xb = [make_input(r, (512, 8192), nseed(), dev, kind="mixed") for r in range(WORLD)]
        xc = [make_input(r, (1024, 512), nseed(), dev) for r in range(WORLD)]
        if it % 2:
            e = make_input(rank, (2048, 8192), nseed(), dev)
            tensor_model_parallel_all_reduce(e)
        g_ar.copy_(xa[rank])
        g_rs.copy_(xb[rank])
        g_ag.copy_(xc[rank])
        graph_l.replay()
        ck.check(bits_equal(o_ar, contract_ref(xa, members)), f"graph large AR == contract (iter {it})")
        ck.check(bits_equal(o_rs, contract_ref(xb, members)[rank * 64:(rank + 1) * 64]),
                 f"graph large RS == contract block (iter {it})")
        ck.check(bits_equal(o_ag, torch.cat(xc, 0)), f"graph large AG exact (iter {it})")
    torch.cuda.synchronize()

    # 7) a collective on a side stream (serialized behind the main stream by TopoComm), then back on the main stream
    side = torch.cuda.Stream()
    for _ in range(3):
        xs, ref = reference_ar((8, 8192), nseed(), torch.bfloat16, dev, members)
        a = tensor_model_parallel_all_reduce(xs[rank])
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            b = tensor_model_parallel_all_reduce(xs[rank])
        cur.wait_stream(side)
        c = tensor_model_parallel_all_reduce(xs[rank])
        ck.check(bits_equal(a, ref) and bits_equal(b, ref) and bits_equal(c, ref), "AR across streams == contract")

    # 8) random mixed sequence of topo and NCCL collectives (counter alignment, no hang)
    rng = random.Random(1234)  # same on every rank
    n_ops = 60 if quick else 400
    acc_ok = True
    for i in range(n_ops):
        kind = rng.choice(["ar", "ar", "ag0", "agl", "nccl_ar", "nccl_ag", "ar_big", "rs_big", "ag_big"])
        t = rng.choice([1, 2, 3, 8, 17, 64])
        s = nseed()
        if kind == "ar_big":
            xs, ref = reference_ar((rng.choice([256, 512, 1024, 2048, 3000]), 4096), s, torch.bfloat16, dev, members)
            acc_ok &= bits_equal(tensor_model_parallel_all_reduce(xs[rank]), ref)
        elif kind == "rs_big":
            rows = 8 * rng.choice([8, 40, 128, 512])
            xs = [make_input(r, (rows, 4096), s, dev, kind="mixed") for r in range(WORLD)]
            b = rows // WORLD
            acc_ok &= bits_equal(tp.reduce_scatter(xs[rank], 0), contract_ref(xs, members)[rank * b:(rank + 1) * b])
        elif kind == "ag_big":
            rows = rng.choice([128, 512, 2048])
            xs = [make_input(r, (rows, 1024), s, dev) for r in range(WORLD)]
            dim = rng.choice([0, -1])
            acc_ok &= bits_equal(tensor_model_parallel_all_gather(xs[rank], dim), torch.cat(xs, dim))
        elif kind == "ar":
            xs, ref = reference_ar((t, 4096), s, torch.bfloat16, dev, members)
            acc_ok &= bits_equal(tensor_model_parallel_all_reduce(xs[rank]), ref)
        elif kind == "ag0":
            xs = [make_input(r, (t, 512), s, dev) for r in range(WORLD)]
            acc_ok &= bits_equal(tensor_model_parallel_all_gather(xs[rank], 0), torch.cat(xs, 0))
        elif kind == "agl":
            xs = [make_input(r, (t, 1024), s, dev) for r in range(WORLD)]
            acc_ok &= bits_equal(tensor_model_parallel_all_gather(xs[rank], -1), torch.cat(xs, -1))
        elif kind == "nccl_ar":
            x = make_input(rank, (t, 4096), s, dev)
            acc_ok &= nccl_ar(x).shape == x.shape
        else:
            xs = [make_input(r, (t, 512), s, dev) for r in range(WORLD)]
            acc_ok &= bits_equal(nccl_ag(xs[rank], 0), torch.cat(xs, 0))
    ck.check(acc_ok, f"{n_ops} mixed topo/NCCL ops")
    torch.cuda.synchronize()

    # 9) dispatch-level graph timing (20 chained calls), topo vs the NCCL path of the same wrapper
    def graph_us(fn, reps=20, iters=200):
        g = torch.cuda.CUDAGraph()
        with graph_capture(device=dev) as gx:
            with torch.cuda.graph(g, stream=gx.stream):
                for _ in range(reps):
                    fn()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / (iters * reps) * 1e6

    x = make_input(rank, (1, 8192), 7, dev)
    p = make_input(rank, (1, 512), 8, dev)
    rows = []
    for name, fn in (("AR [1,8192]", lambda: tensor_model_parallel_all_reduce(x)),
                     ("AG [1,512] dim0", lambda: tensor_model_parallel_all_gather(p, 0))):
        us_topo = graph_us(fn)
        c, dc.topo_comm = dc.topo_comm, None
        try:
            us_nccl = graph_us(fn)
        finally:
            dc.topo_comm = c
        rows.append((name, us_topo, us_nccl))
    allrows = gather_all(rows, cpu)
    if rank == 0:
        for i, (name, _, _) in enumerate(rows):
            print(f"PLUGIN-LEVEL graph timing {name}: topo {max(r[i][1] for r in allrows):.2f} us, "
                  f"nccl {max(r[i][2] for r in allrows):.2f} us per call (max over ranks)", flush=True)

    # 10) per-op summary and teardown (destroy dumps the call counters)
    if rank == 0:
        for key, n in sorted(dc.topo_trace.counts.items(), key=lambda kv: str(kv[0])):
            print("  calls", n, key[0], f"dim={key[1]}", key[2], str(key[3]).replace("torch.", ""), f"capturing={key[5]}",
                  key[6], flush=True)
    tp.destroy()
    ck.check(dc.topo_comm is None, "destroy closed the topo communicator")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--port", type=int, default=29731)
    ap.add_argument("--keep-bf16-exchange", action="store_true",
                    help="keep MUSE_TOPO_LARGE_BF16_EXCHANGE as exported (env/host.env turns it ON for serving; the "
                         "checks below are the exact fp32 contract and fail with it, see docs/h2_vllm_plugin.md)")
    args = ap.parse_args()
    if not args.keep_bf16_exchange:
        os.environ["MUSE_TOPO_LARGE_BF16_EXCHANGE"] = "0"
    os.environ.setdefault("MUSE_TOPO_ENABLE", "1")
    os.environ.setdefault("MUSE_TOPO_TIMEOUT_S", "60")
    os.environ.setdefault("MUSE_TOPO_LOG_OPS", "1")
    mp.spawn(worker, args=(args.port, args.quick), nprocs=WORLD, join=True)
    print("TEST_PLUGIN PASSED", flush=True)


if __name__ == "__main__":
    main()
