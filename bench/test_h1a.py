"""Standalone 8-process correctness tests of TopoComm (all-reduce algorithms 1/2, all-gather), no vLLM.

usage: py_topo.sh bench/test_h1a.py [--quick] [--replays N]   (run under the GPU lock)
Checks per op/size/layout: bitwise equality with the arithmetic contract computed in torch, rank-identical bits
(sha1 over all ranks, incl. sign of zero), run-to-run determinism, eager == graph replay, algo 1 == algo 2, mixed
size sequences (counter alignment), N back-to-back graph replays.
"""
import argparse
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hlib import *
from muse_topo import TopoComm

FAILS = []


def check(rank, cond, what):
    ok = all(gather_obj(bool(cond), CPU))
    if not ok:
        FAILS.append(what)
        log(rank, "FAIL", what)
    return ok


def inputs(shape, seed, dev, dtype, kind="randn"):
    return [make_input(r, shape, seed, dev, dtype, kind) for r in range(WORLD)]


def ar_cases(quick):
    cases = []
    ts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if not quick else [1, 8, 64, 512]
    for hid in (8192, 4096):
        for t in ts:
            cases.append((t, hid))
    cases += [(1, 512), (3, 1024), (5, 2048), (7, 8192), (1, 32), (16, 1032)]
    return cases


def test_ar(rank, comm, dev, quick):
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for (t, hid) in ar_cases(quick):
            shape = (t, hid)
            nb = t * hid * torch.empty((), dtype=dtype).element_size()
            if dtype != torch.bfloat16 and not (t in (1, 8, 64) and hid == 8192):
                continue
            for kind in ("randn", "realish", "zeros"):
                if kind != "randn" and not (dtype == torch.bfloat16 and (t, hid) in ((1, 8192), (8, 8192), (64, 4096), (512, 8192))):
                    continue
                xs = inputs(shape, 100 + t, dev, dtype, kind)
                x = xs[rank]
                ref = contract_ref(xs, dtype)
                tag = f"AR {dtype} {shape} {kind}"
                if not comm.supports("all_reduce", x):
                    check(rank, False, tag + " unsupported")
                    continue
                outs = {}
                for name, algo, blk, unr in (("a1", 1, 128, 1), ("a1u2", 1, 256, 2), ("a2", 2, 256, 1), ("a2u4", 2, 128, 4)):
                    if algo == 1 and nb > comm.ar1_max_bytes:
                        continue
                    if algo == 2 and nb % 64:
                        continue
                    y = comm._all_reduce(x, algo, blk, unr)
                    torch.cuda.synchronize()
                    y2 = comm._all_reduce(x, algo, blk, unr)
                    torch.cuda.synchronize()
                    outs[name] = y
                    check(rank, bits_equal(y, ref), f"{tag} {name} != contract")
                    check(rank, bits_equal(y, y2), f"{tag} {name} run-to-run")
                    d = digest(y)
                    check(rank, len(set(gather_obj(d, CPU))) == 1, f"{tag} {name} rank-identical")
                y = comm.all_reduce(x)
                torch.cuda.synchronize()
                check(rank, bits_equal(y, ref), f"{tag} default != contract")
                # eager == graph replay
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    comm.all_reduce(x)
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(g):
                    go = comm.all_reduce(x)
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                check(rank, bits_equal(go, ref), f"{tag} graph != contract")
                del g
    log(rank, "AR tests done")


def ag_cases(quick):
    cases = []
    ts = [1, 2, 4, 8, 16, 32, 64, 128, 256] if not quick else [1, 8, 64]
    for w in (512, 1024):
        for t in ts:
            cases.append((t, w))
    cases += [(1, 25256), (2, 25256), (3, 4), (1, 8), (5, 24)]
    return cases


def test_ag(rank, comm, dev, quick):
    for dtype in (torch.bfloat16, torch.float32, torch.int8):
        for (t, w) in ag_cases(quick):
            if dtype != torch.bfloat16 and (t, w) not in ((1, 1024), (8, 512), (3, 4)):
                continue
            g = torch.Generator(device=dev)
            g.manual_seed(7000 + rank * 31 + t * 7 + w)
            # random bit patterns (includes NaNs, infs, -0): all-gather must be bit exact
            raw = torch.randint(-128, 128, (t, w * torch.empty((), dtype=dtype).element_size()), device=dev, dtype=torch.int8, generator=g)
            x = raw.view(dtype).reshape(t, w).contiguous()
            allx = []
            for r in range(WORLD):
                g2 = torch.Generator(device=dev)
                g2.manual_seed(7000 + r * 31 + t * 7 + w)
                rr = torch.randint(-128, 128, (t, w * torch.empty((), dtype=dtype).element_size()), device=dev, dtype=torch.int8, generator=g2)
                allx.append(rr.view(dtype).reshape(t, w))
            ref_cols = torch.cat(allx, dim=-1)
            ref_rows = torch.cat(allx, dim=0)
            tag = f"AG {dtype} [{t},{w}]"
            for dim, ref in ((-1, ref_cols), (0, ref_rows)):
                if not comm.supports("all_gather", x, dim):
                    check(rank, False, f"{tag} dim={dim} unsupported")
                    continue
                y = comm.all_gather(x, dim)
                torch.cuda.synchronize()
                y2 = comm.all_gather(x, dim)
                torch.cuda.synchronize()
                check(rank, bits_equal(y, ref), f"{tag} dim={dim} != reference")
                check(rank, bits_equal(y, y2), f"{tag} dim={dim} run-to-run")
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    go = comm.all_gather(x, dim)
                gr.replay()
                gr.replay()
                torch.cuda.synchronize()
                check(rank, bits_equal(go, ref), f"{tag} dim={dim} graph != reference")
                del gr
    log(rank, "AG tests done")


def test_mixed(rank, comm, dev, n_eager, replays):
    """Random sequences of different ops/sizes (all ranks use the same sequence): counters must stay aligned."""
    rng = random.Random(1234)
    shapes_ar = [(1, 8192), (2, 8192), (8, 4096), (64, 8192), (3, 1024), (256, 8192), (512, 8192), (1, 512)]
    shapes_ag = [(1, 1024), (8, 512), (64, 1024), (1, 25256), (256, 512)]
    seq = []
    for _ in range(n_eager):
        if rng.random() < 0.6:
            seq.append(("ar", rng.choice(shapes_ar), rng.choice((None, 1, 2))))
        else:
            seq.append(("ag", rng.choice(shapes_ag), rng.choice((-1, 0))))
    cache = {}

    def get(kind, shape):
        key = (kind, shape)
        if key not in cache:
            seed = 300 + hash(key) % 1000 if False else 300 + len(cache)
            xs = inputs(shape, seed, dev, torch.bfloat16)
            if kind == "ar":
                cache[key] = (xs[rank], contract_ref(xs, torch.bfloat16), None)
            else:
                cache[key] = (xs[rank], torch.cat(xs, -1), torch.cat(xs, 0))
        return cache[key]

    def run(op):
        kind, shape, opt = op
        x, ref_a, ref_b = get(kind, shape)
        if kind == "ar":
            nb = x.numel() * 2
            if opt == 1 and nb <= comm.ar1_max_bytes:
                return comm._all_reduce(x, 1, 128, 1), ref_a
            if opt == 2:
                return comm._all_reduce(x, 2, 256, 1), ref_a
            return comm.all_reduce(x), ref_a
        return comm.all_gather(x, opt), (ref_a if opt == -1 else ref_b)

    # eager: check every result after the whole sequence (results kept alive)
    res = [run(op) for op in seq]
    torch.cuda.synchronize()
    bad = sum(0 if bits_equal(y, r) else 1 for y, r in res)
    check(rank, bad == 0, f"mixed eager sequence: {bad}/{len(res)} mismatches")
    del res
    # graph: the same kind of sequence captured once and replayed
    gseq = seq[:48]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [run(op) for op in gseq]
    for i in range(replays):
        g.replay()
        if (i + 1) % max(1, replays // 5) == 0:
            torch.cuda.synchronize()
            bad = sum(0 if bits_equal(y, r) else 1 for y, r in outs)
            check(rank, bad == 0, f"mixed graph replay {i + 1}: {bad} mismatches")
    torch.cuda.synchronize()
    log(rank, f"mixed sequences done ({n_eager} eager calls, {replays} replays of {len(gseq)} calls)")


def test_replays(rank, comm, dev, n_replays):
    """N back-to-back replays of a graph with 10 ARs + 10 AGs (no hang, no corruption)."""
    xs = inputs((1, 8192), 55, dev, torch.bfloat16)
    ref = contract_ref(xs, torch.bfloat16)
    x = xs[rank]
    ag_in = inputs((1, 1024), 56, dev, torch.bfloat16)
    ref_ag = torch.cat(ag_in, -1)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = []
        for _ in range(10):
            outs.append(("ar", comm.all_reduce(x)))
            outs.append(("ag", comm.all_gather(ag_in[rank], -1)))
    chunk = 1000
    done = 0
    while done < n_replays:
        n = min(chunk, n_replays - done)
        for _ in range(n):
            g.replay()
        torch.cuda.synchronize()
        done += n
        bad = sum(0 if bits_equal(o, ref if k == "ar" else ref_ag) else 1 for k, o in outs)
        if not check(rank, bad == 0, f"replays: {bad} corrupted outputs after {done} replays"):
            return
    log(rank, f"{n_replays} graph replays ({n_replays * 20} collectives) ok")


def worker(rank, args):
    global CPU
    dev, CPU = init_rank(rank, args.port, watchdog_s=args.watchdog)
    try:
        comm = TopoComm(CPU, dev, rank, WORLD, max_bytes=8 << 20, check_env=False)
        log(rank, f"TopoComm ready, workspace {comm.workspace_bytes / 2**20:.1f} MiB; {comm.info.summary}")
        test_ar(rank, comm, dev, args.quick)
        test_ag(rank, comm, dev, args.quick)
        test_mixed(rank, comm, dev, 300 if args.quick else 3000, 40 if args.quick else 400)
        test_replays(rank, comm, dev, args.replays)
        torch.cuda.synchronize()
        comm.close()
    except BaseException:
        traceback.print_exc()
        FAILS.append("exception")
    n = len(FAILS)
    tot = sum(gather_obj(n, CPU))
    log(rank, "ALL TESTS PASSED" if tot == 0 else f"{tot} FAILURES: {FAILS[:20]}")
    sys.stdout.flush()
    os._exit(0 if tot == 0 else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replays", type=int, default=10000)
    ap.add_argument("--port", type=int, default=29731)
    ap.add_argument("--watchdog", type=int, default=900)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
