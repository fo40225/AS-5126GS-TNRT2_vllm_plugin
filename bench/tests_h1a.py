"""Standalone 8-process correctness tests of the muse_topo collectives (decode-size algorithms 1-3).

usage: py_topo.sh bench/tests_h1a.py [--soak-replays N] [--quick]
Checks (each prints PASS/FAIL): arithmetic contract bitwise (forced algorithms and default dispatch, all dtypes,
three input kinds), rank-identical bits (sign of zero included), run-to-run determinism, eager == graph replay,
algorithm 1 == algorithm 2, fp64 accuracy vs NCCL, all_gather exactness (both layouts, random bit patterns),
mixed random op sequences (counter alignment), and a soak of back-to-back graph replays.
"""
import argparse
import random
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

RESULTS = []


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


def run(rank, args):
    dev, cpu_group, nccl = init_rank(rank, args.port, nccl=True)
    from muse_topo import TopoComm

    comm = TopoComm(cpu_group, dev, rank, WORLD)
    members = comm.info.members
    log(rank, "TopoComm up:", comm.info.summary, f"workspace {comm.workspace_bytes / 2**20:.1f} MiB")
    Ts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if not args.quick else [1, 8, 64]
    dtypes = (torch.bfloat16, torch.float16, torch.float32) if not args.quick else (torch.bfloat16,)

    # ---- 1. contract: default dispatch, all sizes / hidden sizes / dtypes / kinds --------------------------------
    for dtype in dtypes:
        for hid in (8192, 4096):
            for T in Ts:
                if T * hid * torch.empty((), dtype=dtype).element_size() > comm.max_bytes:
                    continue
                kinds = ("randn", "mixed", "zeros") if (T <= 64 or dtype == torch.bfloat16) else ("randn",)
                for kind in kinds:
                    xs = [make_input(r, (T, hid), 100 + T, dev, dtype, kind) for r in range(WORLD)]
                    ref = contract_ref(xs, members)
                    assert comm.supports("all_reduce", xs[rank])
                    y = comm.all_reduce(xs[rank])
                    torch.cuda.synchronize()
                    ok = bits_equal(y, ref)
                    same = all_equal_across_ranks(y, cpu_group)
                    check(rank, f"ar contract {dtype} [{T},{hid}] {kind}", ok and same)
    # ---- 2. forced algorithms, determinism, eager == graph, algorithms agree bitwise -------------------------------
    for T in (1, 4, 16, 64, 256):
        xs = [make_input(r, (T, 8192), 7, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
        x = xs[rank]
        nb = x.numel() * 2
        y2 = comm._all_reduce(x, 2, 256, 1)
        y3 = comm._all_reduce(x, 3, 256, 1)
        torch.cuda.synchronize()
        check(rank, f"ar algo2 (S-LL) == algo3 (flag) bitwise [{T},8192]", bits_equal(y3, y2))
        if nb <= comm.ar1_max_bytes:
            y1 = comm._all_reduce(x, 1, 128, 1)
            torch.cuda.synchronize()
            check(rank, f"ar algo1 == algo2 bitwise [{T},8192]", bits_equal(y1, y2))
        reps = [comm._all_reduce(x, 2, 256, 1) for _ in range(5)] + [comm._all_reduce(x, 3, 256, 1) for _ in range(5)]
        torch.cuda.synchronize()
        check(rank, f"ar run-to-run [{T},8192]", all(bits_equal(r, y2) for r in reps))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yg = comm.all_reduce(x)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        check(rank, f"ar graph replay == eager [{T},8192]", bits_equal(yg, y2))
    # ---- 3. accuracy vs fp64 and vs NCCL ---------------------------------------------------------------------
    for kind in ("randn", "mixed"):
        shape = (16, 8192)
        xs = [make_input(r, shape, 31, dev, torch.bfloat16, kind) for r in range(WORLD)]
        s64 = exact_sum(xs)
        y = comm.all_reduce(xs[rank])
        yn = nccl.all_reduce(xs[rank].clone())
        torch.cuda.synchronize()
        st, sn = err_stats(y, s64), err_stats(yn, s64)
        log(rank, f"accuracy {kind} {shape}: topo {st} | nccl {sn}")
        check(rank, f"ar correctly rounded 100% vs fp64 ({kind})", st["correctly_rounded_pct"] >= 99.99)
        check(rank, f"ar more accurate than nccl ({kind})", st["mean_ulp"] <= sn["mean_ulp"])
    # ---- 4. NaN / inf behaviour -------------------------------------------------------------------------------
    xs = [make_input(r, (4, 8192), 5, dev) for r in range(WORLD)]
    for r in range(WORLD):
        xs[r][0, 0] = float("inf") if r % 2 == 0 else float("-inf")  # inf - inf = NaN
        xs[r][1, 1] = float("nan")
        xs[r][2, 2] = 65504.0 * 4e33 if False else xs[r][2, 2]
    ref = contract_ref(xs, members)
    y = comm.all_reduce(xs[rank])
    torch.cuda.synchronize()
    check(rank, "ar nan/inf bitwise vs contract", bits_equal(y, ref) and all_equal_across_ranks(y, cpu_group))
    # ---- 5. all_gather -----------------------------------------------------------------------------------------
    for (T, w) in [(1, 512), (1, 1024), (2, 1024), (8, 512), (8, 1024), (64, 1024), (128, 512), (3, 256), (1, 25256)]:
        for dtype in (torch.bfloat16, torch.float32):
            xs = [make_input(r, (T, w), 41, dev, dtype) for r in range(WORLD)]
            if not comm.supports("all_gather", xs[rank], -1):
                log(rank, f"skip ag [{T},{w}] {dtype}: unsupported")
                continue
            y = comm.all_gather(xs[rank], -1)
            y0 = comm.all_gather(xs[rank], 0)
            torch.cuda.synchronize()
            check(rank, f"ag dim=-1 [{T},{w}] {dtype}", bits_equal(y, torch.cat(xs, -1)))
            check(rank, f"ag dim=0 [{T},{w}] {dtype}", bits_equal(y0, torch.cat(xs, 0)))
            nb = xs[rank].numel() * xs[rank].element_size()
            if nb % 16 == 0 and (w * xs[rank].element_size()) % 16 == 0:
                yf = comm._all_gather(xs[rank], 1, 0, 2, 256, 1)
                yf0 = comm._all_gather(xs[rank], 0, 1, 2, 256, 1)
                torch.cuda.synchronize()
                check(rank, f"ag flag dim=-1 [{T},{w}] {dtype}", bits_equal(yf, torch.cat(xs, -1)))
                check(rank, f"ag flag dim=0 [{T},{w}] {dtype}", bits_equal(yf0, torch.cat(xs, 0)))
    # random bit patterns including the all-ones word (the AR sentinel) must survive all_gather
    xs = [torch.randint(-(2**15), 2**15, (4, 1024), device=dev, dtype=torch.int16).view(torch.bfloat16)
          for _ in range(WORLD)]
    gen = torch.Generator(device=dev)
    gen.manual_seed(rank + 1)
    xr = torch.randint(-(2**15), 2**15, (4, 1024), device=dev, dtype=torch.int16, generator=gen)
    xr[:, ::3] = -1
    xr = xr.view(torch.bfloat16)
    allx = [None] * WORLD
    parts = [torch.zeros_like(xr) for _ in range(WORLD)]
    gather_cpu = [None] * WORLD
    dist.all_gather_object(gather_cpu, xr.cpu(), group=cpu_group)
    ref = torch.cat([g.to(dev) for g in gather_cpu], -1)
    y = comm.all_gather(xr, -1)
    torch.cuda.synchronize()
    check(rank, "ag exact for arbitrary bit patterns (incl. 0xFFFF words)", bits_equal(y, ref))
    # ---- 5b. supports() vs the kernels (review): everything supports() accepts must run and be exact -----------------
    # small shapes of several element sizes, dims 0 and last, odd rows included (a dim-0 gather of a narrow odd row used to
    # pass supports() and raise in the LL8 kernel); the five failing shapes of the review are listed explicitly
    n_run = n_fb = 0
    bad = []
    shapes = ([(a,) for a in range(1, 10)] + [(a, b) for a in range(1, 10) for b in range(1, 10)]
              + [(a, b, c) for a in (1, 2) for b in (1, 3) for c in (1, 2, 3, 5, 6)] + [(64, 3), (6, 1), (2, 6)])
    gcpu = torch.Generator().manual_seed(77)
    for dt in (torch.int8, torch.bfloat16, torch.float32):
        es = torch.empty((), dtype=dt).element_size()
        for shape in shapes:
            nb = es
            for dsz in shape:
                nb *= dsz
            for dim in sorted({0, len(shape) - 1}):
                raw = torch.randint(0, 256, (WORLD, nb), generator=gcpu, dtype=torch.uint8).to(dev)
                xs = [raw[r].clone().view(dt).reshape(shape) for r in range(WORLD)]
                if not comm.supports("all_gather", xs[rank], dim):
                    n_fb += 1
                    continue
                n_run += 1
                try:
                    y = comm.all_gather(xs[rank], dim)
                    torch.cuda.synchronize()
                    if not bits_equal(y, torch.cat(xs, dim)):
                        bad.append(("wrong", str(dt), shape, dim))
                except RuntimeError as e:
                    bad.append(("raises", str(dt), shape, dim, str(e)[:40]))
    check(rank, f"supports() vs kernel sweep: {n_run} gathers run, {n_fb} routed to NCCL, none raised / wrong ({bad[:3]})",
          not bad and n_run > 0)
    x3 = torch.zeros((4, 4, 4), dtype=torch.bfloat16, device=dev)
    try:
        comm.all_gather(x3, 1)
        raised = False
    except ValueError:
        raised = True
    check(rank, "all_gather of a middle dim raises ValueError, supports() is False",
          raised and not comm.supports("all_gather", x3, 1))
    # eligibility must not depend on rank-local state: every rank gets a differently aligned view of the same shape
    # (a mix of topo and NCCL ranks would deadlock) and the ops stage misaligned operands through aligned copies
    buf = torch.zeros((1 << 21) + 64, dtype=torch.bfloat16, device=dev)
    off = (0, 1, 3, 5, 2, 7, 4, 6)[rank]  # element offsets 0..7: byte offsets 0 .. 14 (only 0 is 16-byte aligned)
    ok_sup = ok_bits = True
    for (T, hid, kind, opname, dim) in ((4, 8192, "ar", "all_reduce", None), (256, 8192, "ar", "all_reduce", None),
                                        (32, 512, "ag0", "all_gather", 0), (32, 512, "ag1", "all_gather", -1),
                                        (64, 4096, "rs", "reduce_scatter", None)):
        xs = [make_input(r, (T, hid), 55 + T, dev) for r in range(WORLD)]
        v = buf[off:off + T * hid].view(T, hid)
        v.copy_(xs[rank])
        ok_sup &= all(gather_all(comm.supports(opname, v, dim), cpu_group))
        if kind == "ar":
            y, ref = comm.all_reduce(v), contract_ref(xs, members)
            ob = torch.zeros(T * hid + 8, dtype=torch.bfloat16, device=dev)
            o = ob[1:1 + T * hid].view(T, hid)  # misaligned caller-provided output
            r2 = comm.all_reduce(v, out=o)
            torch.cuda.synchronize()
            ok_bits &= bits_equal(o, ref) and r2 is o
        elif kind == "ag0":
            y, ref = comm.all_gather(v, 0), torch.cat(xs, 0)
        elif kind == "ag1":
            y, ref = comm.all_gather(v, -1), torch.cat(xs, -1)
        else:
            y = comm.reduce_scatter(v)
            ref = contract_ref(xs, members)[rank * (T // WORLD):(rank + 1) * (T // WORLD)]
        torch.cuda.synchronize()
        ok_bits &= bits_equal(y, ref)
    check(rank, "misaligned operands (a different alignment on every rank): supports() holds, results exact",
          ok_sup and ok_bits)
    # ---- 6. mixed random sequences (counter / parity alignment) -------------------------------------------------
    rng = random.Random(1234)  # identical sequence on every rank
    nseq = 120 if not args.quick else 40
    ok_all = True
    for i in range(nseq):
        op = rng.choice(["ar", "ar", "ag", "ag0"])
        if op == "ar":
            T = rng.choice([1, 2, 3, 5, 8, 16, 31, 64, 128, 256, 512])
            hid = rng.choice([4096, 8192])
            dt = rng.choice([torch.bfloat16, torch.float16, torch.float32])
            xs = [make_input(r, (T, hid), 900 + i, dev, dt, "mixed") for r in range(WORLD)]
            if not comm.supports("all_reduce", xs[rank]):
                continue
            y = comm.all_reduce(xs[rank])
            ref = contract_ref(xs, members)
        else:
            T = rng.choice([1, 2, 4, 7, 16, 64])
            w = rng.choice([512, 1024, 256])
            xs = [make_input(r, (T, w), 900 + i, dev) for r in range(WORLD)]
            d = -1 if op == "ag" else 0
            if not comm.supports("all_gather", xs[rank], d):
                continue
            y = comm.all_gather(xs[rank], d)
            ref = torch.cat(xs, d)
        if rng.random() < 0.3:
            torch.cuda.synchronize()
        ok_all &= bits_equal(y, ref)
    torch.cuda.synchronize()
    check(rank, f"mixed random op sequence ({nseq} ops)", ok_all)
    # ---- 7. soak: back-to-back graph replays ----------------------------------------------------------------------
    xa = make_input(rank, (1, 8192), 3, dev)
    xb = make_input(rank, (8, 4096), 4, dev)
    xc = make_input(rank, (1, 1024), 5, dev)
    xd = make_input(rank, (4, 8192), 6, dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ya = comm.all_reduce(xa)
        yb = comm.all_reduce(xb)
        yc = comm.all_gather(xc, -1)
        yd = comm.all_reduce(xd)
        ye = comm.all_gather(xc, 0)
    refs = [contract_ref([make_input(r, (1, 8192), 3, dev) for r in range(WORLD)], members),
            contract_ref([make_input(r, (8, 4096), 4, dev) for r in range(WORLD)], members),
            torch.cat([make_input(r, (1, 1024), 5, dev) for r in range(WORLD)], -1),
            contract_ref([make_input(r, (4, 8192), 6, dev) for r in range(WORLD)], members),
            torch.cat([make_input(r, (1, 1024), 5, dev) for r in range(WORLD)], 0)]
    outs = [ya, yb, yc, yd, ye]
    soak_ok = True
    t0 = time.time()
    n = args.soak_replays
    chunk = max(1, n // 10)
    done = 0
    while done < n:
        for _ in range(min(chunk, n - done)):
            g.replay()
        done += min(chunk, n - done)
        torch.cuda.synchronize()
        soak_ok &= all(bits_equal(o, r) for o, r in zip(outs, refs))
    check(rank, f"soak {n} graph replays x 5 collectives ({time.time() - t0:.1f}s)", soak_ok)
    nfail = sum(1 for _, ok in RESULTS if not ok)
    allf = gather_all(nfail, cpu_group)
    log(rank, f"SUMMARY: {len(RESULTS)} checks per rank, failures per rank {allf}")
    log(rank, "RESULT", "PASS" if sum(allf) == 0 else "FAIL")
    torch.cuda.synchronize()
    comm.close()
    sys.stdout.flush()
    os._exit(0 if nfail == 0 else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--soak-replays", type=int, default=100000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--port", type=int, default=29712)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
