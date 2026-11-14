"""Standalone 8-process correctness tests of the large-message collectives of muse_topo (H1b: chunked all-reduce,
reduce-scatter, all-gather), no vLLM model.

usage: py_topo.sh bench/tests_h1b.py [--quick] [--soak-replays N] [--groups a,b,...]
Groups (each check prints PASS/FAIL once, aggregated over the 8 ranks):
  ar      all-reduce large (fused and two-launch modes, bf16/fp16/fp32, randn/mixed/zeros) bitwise vs the arithmetic
          contract at 16/64/128 MiB and odd sizes (partial last chunk), medium sizes vs the H1a algorithm 2
  bits    random bit patterns incl. 0xFFFF words / NaN / inf: large path == H1a algorithm 2 bitwise
  rs      reduce-scatter == rows block of the contract; reduce_scatter + all_gather == all_reduce bitwise
  ag      large all-gather, both layouts, odd chunk counts, arbitrary bit patterns
  det     rank-identical bits, run-to-run, eager == CUDA graph replay, in-place all-reduce, batch invariance
  bx      opt-in bf16 exchange (bf16 inputs only; fp16/fp32 keep the exact exchange): bitwise vs a torch emulation,
          batch invariant inside the large algorithm, deterministic, accuracy
  acc     accuracy vs fp64 (and vs NCCL) at 16 MiB
  seq     random mixed sequences of small/medium/large ops (epoch alignment of the shared call counter)
  soak    back-to-back graph replays of a mixed graph (large + small ops)
"""
import argparse
import random
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

RESULTS = []
MIB = 1 << 20


def check(rank, cpu_group, name, ok):
    oks = gather_all(bool(ok), cpu_group)
    good = all(oks)
    RESULTS.append((name, good))
    if rank == 0:
        print(("PASS " if good else f"FAIL(ranks {[i for i, o in enumerate(oks) if not o]}) ") + name, flush=True)
    return good


def rand_bits(shape, rank, seed, dev, dtype, ones_frac=0.02):
    """Arbitrary bit patterns (NaN, inf, subnormals, +-0 included); a fraction of the 32-bit words is all-ones."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed * 7919 + rank * 104729 + 3)
    esz = torch.empty((), dtype=dtype).element_size()
    nwords = (int(torch.empty(shape, device="meta").numel()) * esz) // 4
    w = torch.randint(-(2**31), 2**31, (nwords,), device=dev, dtype=torch.int32, generator=g)
    m = torch.rand((nwords,), device=dev, generator=g) < ones_frac
    w = torch.where(m, torch.full_like(w, -1), w)
    return w.view(dtype).reshape(shape)


def finite_mixed(shape, rank, seed, dev, dtype):
    return make_input(rank, shape, seed, dev, dtype, "mixed")


def free():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


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
    log(rank, "TopoComm up:", comm.info.summary, f"workspace {comm.workspace_bytes / 2**20:.1f} MiB",
        f"chunk {comm.large_chunk_bytes}")
    groups = set(args.groups.split(","))
    quick = args.quick
    ck = lambda name, ok: check(rank, cpu_group, name, ok)  # noqa: E731

    def mib_shape(mib, hid=8192, dtype=torch.bfloat16):
        esz = torch.empty((), dtype=dtype).element_size()
        return (mib * MIB // (hid * esz), hid)

    # ---- supports() semantics ---------------------------------------------------------------------------------
    if "ar" in groups:
        x = torch.empty(mib_shape(64), dtype=torch.bfloat16, device=dev)
        ck("supports: all_reduce 64 MiB", comm.supports("all_reduce", x))
        ck("supports: reduce_scatter 64 MiB", comm.supports("reduce_scatter", x))
        ck("supports: all_reduce non-contiguous", not comm.supports("all_reduce", x.t()))
        ck("supports: reduce_scatter T % 8 != 0", not comm.supports("reduce_scatter", torch.empty(
            (4099, 8192), dtype=torch.bfloat16, device=dev)))
        ck("supports: all_reduce int32", not comm.supports("all_reduce", torch.empty((8192, 8192), dtype=torch.int32, device=dev)))
        ck("supports: all_gather 16 MiB shard", comm.supports("all_gather", torch.empty((8192, 1024), dtype=torch.bfloat16, device=dev), -1))
        ck("supports: all_reduce odd nbytes", not comm.supports("all_reduce", torch.empty(
            (9 * MIB // 2 + 16,), dtype=torch.bfloat16, device=dev)))
        del x

    # ---- 1. all-reduce large vs the contract ---------------------------------------------------------------------
    if "ar" in groups:
        cases = [(16, torch.bfloat16, "randn"), (16, torch.bfloat16, "mixed"), (16, torch.bfloat16, "zeros"),
                 (16, torch.float16, "mixed"), (16, torch.float32, "mixed"), (64, torch.bfloat16, "mixed"),
                 (128, torch.bfloat16, "randn"), (128, torch.bfloat16, "mixed")]
        if quick:
            cases = [(16, torch.bfloat16, "mixed"), (16, torch.float32, "randn"), (64, torch.bfloat16, "mixed")]
        for mib, dtype, kind in cases:
            shape = mib_shape(mib, 8192, dtype)
            xs = [make_input(r, shape, 200 + mib, dev, dtype, kind) for r in range(WORLD)]
            ref = contract_ref(xs, members)
            x = xs[rank]
            y = comm.all_reduce(x)  # public dispatch
            free()
            ck(f"ar large dispatch {dtype} {mib} MiB {kind} == contract", bits_equal(y, ref))
            for mode in (0, 1):
                y = comm._all_reduce_large(x, None, mode)
                free()
                ck(f"ar large mode{mode} {dtype} {mib} MiB {kind} == contract", bits_equal(y, ref))
            if mib == 16:
                for (ch, rg, ag, blk, un) in [(1 << 18, 4, 16, 256, 1), (1 << 19, 16, 8, 256, 2), (1 << 20, 8, 64, 512, 2)]:
                    y = comm._all_reduce_large(x, None, 1, ch, rg, ag, blk, un)
                    free()
                    ck(f"ar large fused chunk{ch >> 10}K rs{rg} ag{ag} blk{blk} u{un} {dtype} {kind} == contract",
                       bits_equal(y, ref))
                for (rp, pd) in [(1, 0), (2, 0), (2, 1), (0, 1)]:  # ramp chunk schedule (small chunks at both ends) and PDL
                    y = comm._all_reduce_large(x, None, 1, 1 << 18, 4, 8, 512, 2, None, 0, 1, rp, pd)
                    free()
                    ck(f"ar large fused chunk256K ramp{rp} pdl{pd} {dtype} {kind} == contract", bits_equal(y, ref))
                for (crs, cag, mode, un) in [(1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 2), (1, 1, 0, 1), (1, 2, 1, 1), (1, 2, 1, 2), (1, 2, 0, 1)]:
                    y = comm._all_reduce_large(x, None, mode, 1 << 19, 8, 16, 512, un, None, crs, cag)
                    free()
                    ck(f"ar large mode{mode} cut_rs{crs} cut_ag{cag} u{un} {dtype} {kind} == contract", bits_equal(y, ref))
            del xs, ref, x, y
            free()
        # odd sizes: partial last chunk (3 chunks of 1 MiB, 1 MiB, 14 KiB per block), tiny chunks, fp16
        for T, dtype in [(1031, torch.bfloat16), (1031, torch.float16), (777, torch.float32), (200, torch.bfloat16)]:
            xs = [make_input(r, (T, 8192), 300 + T, dev, dtype, "mixed") for r in range(WORLD)]
            ref = contract_ref(xs, members)
            for mode in (0, 1):
                y = comm._all_reduce_large(xs[rank], None, mode)
                free()
                ck(f"ar large mode{mode} {dtype} [{T},8192] == contract", bits_equal(y, ref))
            y = comm._all_reduce_large(xs[rank], None, 1, 1 << 20, 4, 8, 256, 1, None, 1, 1)
            free()
            ck(f"ar large fused cut_rs1 cut_ag1 {dtype} [{T},8192] == contract", bits_equal(y, ref))
            y = comm._all_reduce_large(xs[rank], None, 1, 1 << 20, 4, 8, 512, 2, None, 1, 2)
            free()
            ck(f"ar large fused merged cut {dtype} [{T},8192] == contract", bits_equal(y, ref))
            y = comm._all_reduce_large(xs[rank], None, 1, 1 << 16, 4, 8, 512, 1, None, 0, 1, 2)
            free()
            ck(f"ar large fused chunk64K ramp2 {dtype} [{T},8192] == contract", bits_equal(y, ref))
            del xs, ref
            free()
        # medium sizes: the large algorithm == H1a algorithm 2 (the decisive cross-check)
        for T, hid, dtype in [(64, 8192, torch.bfloat16), (512, 8192, torch.bfloat16), (256, 4096, torch.float16),
                              (128, 8192, torch.float32), (8, 8192, torch.bfloat16), (40, 2048, torch.bfloat16)]:
            xs = [make_input(r, (T, hid), 400 + T, dev, dtype, "mixed") for r in range(WORLD)]
            x = xs[rank]
            nb = x.numel() * x.element_size()
            small = comm._all_reduce(x, 2, 256, 1)
            for mode in (0, 1):
                y = comm._all_reduce_large(x, None, mode)
                torch.cuda.synchronize()
                ck(f"ar large mode{mode} == H1a algo2 {dtype} [{T},{hid}] ({nb >> 10} KiB)", bits_equal(y, small))
            y = comm._all_reduce_large(x, None, 1, 1 << 20, 4, 16, 256, 1, None, 1, 2)
            torch.cuda.synchronize()
            ck(f"ar large fused merged cut == H1a algo2 {dtype} [{T},{hid}] ({nb >> 10} KiB)", bits_equal(y, small))
            del xs
            free()

    # ---- 2. arbitrary bit patterns: large == small algorithm 2 (canonicalization, NaN, inf, 0xFFFF words) -----------
    if "bits" in groups:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for T in ((64, 512) if dtype != torch.float32 else (64, 256)):  # algorithm 2 handles up to 8 MiB
                x = rand_bits((T, 8192), rank, 11 + T, dev, dtype)
                small = comm._all_reduce(x, 2, 256, 1)
                for mode in (0, 1):
                    y = comm._all_reduce_large(x, None, mode)
                    torch.cuda.synchronize()
                    ck(f"ar large mode{mode} == algo2 on random bits {dtype} [{T},8192]", bits_equal(y, small))
                ck(f"ar random bits rank-identical {dtype} [{T},8192]", all_equal_across_ranks(small, cpu_group))
                # reduce-scatter + all-gather on the same bits
                y = comm._all_reduce_large(x, None, 1, 1 << 20, 8, 16, 512, 1, None, 1, 2)
                torch.cuda.synchronize()
                ck(f"ar large fused merged cut == algo2 on random bits {dtype} [{T},8192]", bits_equal(y, small))
                rs = comm._reduce_scatter_large(x)
                ag = comm._all_gather_large(rs, 1)
                torch.cuda.synchronize()
                ck(f"rs+ag == algo2 on random bits {dtype} [{T},8192]", bits_equal(ag, small))
                rs = comm._reduce_scatter_large(x, None, 1 << 20, 8, 256, 1, None, 1)
                ag = comm._all_gather_large(rs, 1, None, 1 << 20, 16, 256, 1)
                torch.cuda.synchronize()
                ck(f"rs+ag with cut blocks == algo2 on random bits {dtype} [{T},8192]", bits_equal(ag, small))
            free()
        # explicit inf / NaN / sign-of-zero rows at large size (torch reference, like tests_h1a test 4)
        xs = [make_input(r, (1024, 8192), 5, dev) for r in range(WORLD)]
        for r in range(WORLD):
            xs[r][0, 0] = float("inf") if r % 2 == 0 else float("-inf")
            xs[r][1, 1] = float("nan")
            xs[r][2, :4] = -0.0
        ref = contract_ref(xs, members)
        y = comm._all_reduce_large(xs[rank], None, 1)
        torch.cuda.synchronize()
        ck("ar large nan/inf/-0 bitwise vs contract", bits_equal(y, ref))
        del xs, ref
        free()

    # ---- 3. reduce-scatter ----------------------------------------------------------------------------------------
    if "rs" in groups:
        for mib, dtype, kind in [(16, torch.bfloat16, "mixed"), (64, torch.bfloat16, "randn"), (16, torch.float32, "mixed"),
                                 (16, torch.float16, "zeros"), (4, torch.bfloat16, "mixed"), (128, torch.bfloat16, "mixed")]:
            if quick and mib > 16:
                continue
            shape = mib_shape(mib, 8192, dtype)
            xs = [make_input(r, shape, 500 + mib, dev, dtype, kind) for r in range(WORLD)]
            ref = contract_ref(xs, members)
            x = xs[rank]
            tb = shape[0] // WORLD
            rs = comm._reduce_scatter_large(x)
            torch.cuda.synchronize()
            ck(f"rs {dtype} {mib} MiB {kind} == contract rows block", rs.shape == (tb, shape[1])
               and bits_equal(rs, ref[rank * tb:(rank + 1) * tb]))
            for (ch, g, blk, un, cut) in [(1 << 18, 4, 256, 1, 0), (1 << 19, 16, 512, 2, 0), (1 << 20, 8, 512, 1, 1),
                                          (1 << 19, 4, 256, 2, 1)]:
                rs2 = comm._reduce_scatter_large(x, None, ch, g, blk, un, None, cut)
                torch.cuda.synchronize()
                ck(f"rs chunk{ch >> 10}K grid{g} blk{blk} u{un} cut{cut} {dtype} {mib} MiB == contract",
                   bits_equal(rs2, ref[rank * tb:(rank + 1) * tb]))
            if comm.supports("reduce_scatter", x):
                rs3 = comm.reduce_scatter(x)
                ck(f"rs public {dtype} {mib} MiB", bits_equal(rs3, rs))
            for layout_dim in (0,):
                ag = comm._all_gather_large(rs, 1)
                torch.cuda.synchronize()
                ck(f"rs + ag(dim0) == all_reduce {dtype} {mib} MiB {kind}", bits_equal(ag, ref))
            del xs, ref
            free()
        # an odd T (partial last chunk) and out= given
        xs = [make_input(r, (1032, 8192), 77, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
        ref = contract_ref(xs, members)
        out = torch.empty((129, 8192), dtype=torch.bfloat16, device=dev)
        comm._reduce_scatter_large(xs[rank], out)
        torch.cuda.synchronize()
        ck("rs [1032,8192] out= partial last chunk", bits_equal(out, ref[rank * 129:(rank + 1) * 129]))
        del xs, ref
        free()

    # ---- 4. all-gather large ------------------------------------------------------------------------------------
    if "ag" in groups:
        shapes = [(2048, 512, torch.bfloat16), (1536, 512, torch.bfloat16), (1031, 512, torch.bfloat16),
                  (8192, 512, torch.bfloat16), (1030, 512, torch.float32), (4096, 1024, torch.bfloat16)]
        if quick:
            shapes = shapes[:3]
        for T, w, dtype in shapes:
            xs = [make_input(r, (T, w), 600 + T, dev, dtype) for r in range(WORLD)]
            x = xs[rank]
            assert comm.supports("all_gather", x, -1) and comm.supports("all_gather", x, 0)
            y1 = comm.all_gather(x, -1)
            y0 = comm.all_gather(x, 0)
            torch.cuda.synchronize()
            ck(f"ag large dim=-1 [{T},{w}] {dtype}", bits_equal(y1, torch.cat(xs, -1)))
            ck(f"ag large dim=0 [{T},{w}] {dtype}", bits_equal(y0, torch.cat(xs, 0)))
            for (ch, g, blk, cut) in [(1 << 18, 16, 512, 0), (1 << 19, 64, 1024, 0), (1 << 20, 16, 512, 1), (1 << 19, 32, 1024, 1)]:
                y2 = comm._all_gather_large(x, 0, None, ch, g, blk, cut)
                y3 = comm._all_gather_large(x, 1, None, ch, g, blk, cut)
                torch.cuda.synchronize()
                ck(f"ag large chunk{ch >> 10}K grid{g} blk{blk} cut{cut} [{T},{w}] {dtype}",
                   bits_equal(y2, y1) and bits_equal(y3, y0))
            del xs, y0, y1
            free()
        # arbitrary bit patterns incl. all-ones words survive (flag protocol is bit exact)
        xr = rand_bits((1031, 512), rank, 4, dev, torch.bfloat16, ones_frac=0.3)
        parts = gather_all(xr.cpu(), cpu_group)
        ref1 = torch.cat([p.to(dev) for p in parts], -1)
        y = comm._all_gather_large(xr, 0)
        torch.cuda.synchronize()
        ck("ag large exact for arbitrary bit patterns (dim=-1)", bits_equal(y, ref1))
        ref0 = torch.cat([p.to(dev) for p in parts], 0)
        y = comm._all_gather_large(xr, 1)
        torch.cuda.synchronize()
        ck("ag large exact for arbitrary bit patterns (dim=0)", bits_equal(y, ref0))
        free()

    # ---- 5. determinism, graph replay, in-place, batch invariance --------------------------------------------------
    if "det" in groups:
        shape = mib_shape(16)
        xs = [make_input(r, shape, 700, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
        x = xs[rank]
        ref = contract_ref(xs, members)
        y0 = comm.all_reduce(x)
        reps = [comm.all_reduce(x) for _ in range(4)] + [comm._all_reduce_large(x, None, 0) for _ in range(3)]
        torch.cuda.synchronize()
        ck("ar large run-to-run (7 repeats)", all(bits_equal(r, y0) for r in reps))
        ck("ar large rank-identical (sha256 over ranks)", all_equal_across_ranks(y0, cpu_group))
        # eager == CUDA graph replay (fused pipeline: many launches in one graph)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yg = comm.all_reduce(x)
            rg = comm.reduce_scatter(x)
            ag_in = rg
            agg = comm.all_gather(ag_in, 0)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        ck("ar large graph replay == eager", bits_equal(yg, ref) and bits_equal(yg, y0))
        tb = shape[0] // WORLD
        ck("rs graph replay == contract rows block", bits_equal(rg, ref[rank * tb:(rank + 1) * tb]))
        ck("rs+ag graph replay == all_reduce", bits_equal(agg, ref))
        # in-place: out is x
        xi = x.clone()
        comm._all_reduce_large(xi, xi, 1)
        torch.cuda.synchronize()
        ck("ar large in-place (out is x)", bits_equal(xi, ref))
        xi = x.clone()
        comm._all_reduce_large(xi, xi, 0)
        torch.cuda.synchronize()
        ck("ar large in-place mode0", bits_equal(xi, ref))
        # batch invariance: the first T/2 rows reduce to the same bits whatever the size of the batch they ride in
        half = shape[0] // 2
        yh = comm._all_reduce_large(x[:half].contiguous(), None, 1)
        torch.cuda.synchronize()
        ck("ar large batch invariant (rows 0..T/2 alone == rows of the full batch)", bits_equal(yh, y0[:half]))
        yq = comm._all_reduce_large(x[: half // 2].contiguous(), None, 0)
        torch.cuda.synchronize()
        ck("ar large batch invariant (T/4 rows)", bits_equal(yq, y0[: half // 2]))
        del xs, ref
        free()

    # ---- 6. opt-in bf16 exchange ----------------------------------------------------------------------------------
    if "bx" in groups:
        for dtype in (torch.bfloat16,):
            shape = mib_shape(16, 8192, dtype)
            xs = [make_input(r, shape, 800, dev, dtype, "mixed") for r in range(WORLD)]
            x = xs[rank]
            ps = []
            for isl in members:
                acc = xs[isl[0]].float()
                for r in isl[1:]:
                    acc = acc + xs[r].float()
                ps.append(acc.to(dtype).float())  # each island partial rounded to bf16
            emu = canon_words((ps[0] + ps[1]).to(dtype))
            yb = comm._all_reduce_large(x, None, 1, bx=True)
            ya = comm._all_reduce_large(x, None, 1, bx=False)
            torch.cuda.synchronize()
            ck(f"bx all_reduce {dtype} == torch emulation (both partials rounded)", bits_equal(yb, emu))
            ck(f"bx all_reduce {dtype} differs from the exact path (not a no-op)", not bits_equal(yb, ya))
            yb2 = comm._all_reduce_large(x, None, 0, bx=True)
            yb3 = comm._all_reduce_large(x, None, 1, 1 << 20, 8, 16, 512, 1, True, 1, 2)
            torch.cuda.synchronize()
            ck(f"bx fused with cut blocks == bx plain {dtype}", bits_equal(yb, yb3))
            rsb = comm._reduce_scatter_large(x, None, 1 << 20, 8, 256, 1, bx=True)
            torch.cuda.synchronize()
            tb = shape[0] // WORLD
            ck(f"bx mode0 == mode1 {dtype}", bits_equal(yb, yb2))
            ck(f"bx reduce_scatter == rows block of bx all_reduce {dtype}", bits_equal(rsb, yb[rank * tb:(rank + 1) * tb]))
            ck(f"bx rank-identical {dtype}", all_equal_across_ranks(yb, cpu_group))
            half = shape[0] // 2
            yh = comm._all_reduce_large(x[:half].contiguous(), None, 1, bx=True)
            torch.cuda.synchronize()
            ck(f"bx batch invariant {dtype}", bits_equal(yh, yb[:half]))
            reps = [comm._all_reduce_large(x, None, 1, bx=True) for _ in range(3)]
            torch.cuda.synchronize()
            ck(f"bx run-to-run {dtype}", all(bits_equal(r, yb) for r in reps))
            if dtype == torch.bfloat16:
                s64 = exact_sum(xs)
                eb, ea = err_stats(yb, s64), err_stats(ya, s64)
                log(rank, f"bx accuracy bf16 16 MiB mixed: bx {eb} | exact path {ea}")
            del xs, ps
            free()
        # fp32 and fp16 ignore the flag (fp32 has nothing to gain, an fp16 partial above 65504 would become inf):
        # same bits as the exact path, for all_reduce and reduce_scatter
        for dtype, kind in ((torch.float32, "mixed"), (torch.float16, "mixed")):
            xs = [make_input(r, (1024, 4096), 801, dev, dtype, kind) for r in range(WORLD)]
            ya = comm._all_reduce_large(xs[rank], None, 1, bx=False)
            yb = comm._all_reduce_large(xs[rank], None, 1, bx=True)
            ra = comm._reduce_scatter_large(xs[rank], None, bx=False)
            rb = comm._reduce_scatter_large(xs[rank], None, bx=True)
            torch.cuda.synchronize()
            ck(f"bx on {dtype} input falls back to the exact exchange (all_reduce, reduce_scatter)",
               bits_equal(ya, yb) and bits_equal(ra, rb))
            del xs
            free()
        # regression (review): fp16 island partials of +-160000 are outside the fp16 range; with a rounded exchange the
        # whole output became NaN, the exact exchange gives 0
        x = torch.full((256, 8192), 40000.0 if rank < 4 else -40000.0, dtype=torch.float16, device=dev)
        for flag in (False, True):
            c2 = comm.bf16_exchange
            comm.bf16_exchange = flag
            y = comm.all_reduce(x)
            torch.cuda.synchronize()
            comm.bf16_exchange = c2
            ck(f"fp16 +-40000 all_reduce is finite and exact (bf16_exchange={flag})",
               bool(torch.isfinite(y).all()) and float(y.float().abs().max()) == 0.0)
        del x
        free()

    # ---- 7. accuracy vs fp64 and NCCL -----------------------------------------------------------------------------
    if "acc" in groups:
        for kind in ("randn", "mixed"):
            shape = mib_shape(16)
            xs = [make_input(r, shape, 31, dev, torch.bfloat16, kind) for r in range(WORLD)]
            s64 = exact_sum(xs)
            y = comm.all_reduce(xs[rank])
            yn = nccl.all_reduce(xs[rank].clone())
            torch.cuda.synchronize()
            st, sn = err_stats(y, s64), err_stats(yn, s64)
            log(rank, f"accuracy {kind} {shape} 16 MiB: topo large {st} | nccl {sn}")
            ck(f"ar large correctly rounded 100% vs fp64 ({kind})", st["correctly_rounded_pct"] >= 99.99)
            ck(f"ar large more accurate than nccl ({kind})", st["mean_ulp"] <= sn["mean_ulp"])
            del xs, s64
            free()

    # ---- 8. mixed random sequences (call-counter alignment) -------------------------------------------------------
    if "seq" in groups:
        rng = random.Random(4321)  # identical on every rank
        nseq = 70 if not quick else 24
        ok_all = True
        for i in range(nseq):
            op = rng.choice(["ar_small", "ar_small", "ag_small", "ar_mid", "ar_large", "ar_large", "rs", "ag_large", "ag0_large"])
            dt = rng.choice([torch.bfloat16, torch.bfloat16, torch.float16, torch.float32])
            if op == "ar_small":
                T = rng.choice([1, 3, 8, 64, 200])
                xs = [make_input(r, (T, 8192), 900 + i, dev, dt, "mixed") for r in range(WORLD)]
                y, ref = comm.all_reduce(xs[rank]), contract_ref(xs, members)
            elif op == "ag_small":
                T, w = rng.choice([1, 8, 64]), rng.choice([512, 1024])
                xs = [make_input(r, (T, w), 900 + i, dev, dt) for r in range(WORLD)]
                y, ref = comm.all_gather(xs[rank], -1), torch.cat(xs, -1)
            elif op == "ar_mid":
                T = rng.choice([512, 700, 1032])
                xs = [make_input(r, (T, 8192), 900 + i, dev, dt, "mixed") for r in range(WORLD)]
                y, ref = comm.all_reduce(xs[rank]), contract_ref(xs, members)
            elif op == "ar_large":
                T = rng.choice([1031, 1500, 2048, 3001])
                xs = [make_input(r, (T, 4096), 900 + i, dev, dt, "mixed") for r in range(WORLD)]
                y, ref = comm._all_reduce_large(xs[rank], None, rng.choice([0, 1])), contract_ref(xs, members)
            elif op == "rs":
                T = rng.choice([1024, 1032, 2048])
                xs = [make_input(r, (T, 4096), 900 + i, dev, dt, "mixed") for r in range(WORLD)]
                ref = contract_ref(xs, members)
                tb = T // WORLD
                y, ref = comm.reduce_scatter(xs[rank]) if comm.supports("reduce_scatter", xs[rank]) else \
                    comm._reduce_scatter_large(xs[rank]), ref[rank * tb:(rank + 1) * tb]
            else:
                T, w = rng.choice([1031, 1536, 2048]), rng.choice([512, 256])
                xs = [make_input(r, (T, w), 900 + i, dev, dt) for r in range(WORLD)]
                d = -1 if op == "ag_large" else 0
                y, ref = comm.all_gather(xs[rank], d), torch.cat(xs, d)
            if rng.random() < 0.2:
                torch.cuda.synchronize()
            ok_all &= bits_equal(y, ref)
            del xs, y, ref
        torch.cuda.synchronize()
        ck(f"mixed random op sequence ({nseq} ops, small/medium/large)", ok_all)
        free()

    # ---- 9. soak: back-to-back graph replays of a mixed graph --------------------------------------------------------
    if "soak" in groups:
        shape_l = (2056, 4096)  # 16.06 MiB bf16: 3 chunks
        xs_l = [make_input(r, shape_l, 31, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
        xm = [make_input(r, (200, 8192), 32, dev) for r in range(WORLD)]
        xg = [make_input(r, (1031, 512), 33, dev) for r in range(WORLD)]
        xa = make_input(rank, (1, 8192), 3, dev)
        xc = make_input(rank, (1, 1024), 5, dev)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o_ar = comm._all_reduce_large(xs_l[rank], None, 1)
            o_a = comm.all_reduce(xa)
            o_rs = comm._reduce_scatter_large(xs_l[rank])
            o_m = comm.all_reduce(xm[rank])
            o_ag = comm.all_gather(xg[rank], 0)
            o_c = comm.all_gather(xc, -1)
            o_ar2 = comm._all_reduce_large(xs_l[rank], None, 0)
        refs = [contract_ref(xs_l, members), contract_ref([make_input(r, (1, 8192), 3, dev) for r in range(WORLD)], members),
                contract_ref(xs_l, members)[rank * 257:(rank + 1) * 257], contract_ref(xm, members), torch.cat(xg, 0),
                torch.cat([make_input(r, (1, 1024), 5, dev) for r in range(WORLD)], -1), contract_ref(xs_l, members)]
        outs = [o_ar, o_a, o_rs, o_m, o_ag, o_c, o_ar2]
        n = args.soak_replays
        chunk = max(1, n // 20)
        done, soak_ok = 0, True
        t0 = time.time()
        while done < n:
            for _ in range(min(chunk, n - done)):
                g.replay()
            done += min(chunk, n - done)
            torch.cuda.synchronize()
            soak_ok &= all(bits_equal(o, r) for o, r in zip(outs, refs))
        ck(f"soak {n} graph replays x 7 collectives incl. large ({time.time() - t0:.0f}s)", soak_ok)

    nfail = sum(1 for _, ok in RESULTS if not ok)
    log(rank, f"SUMMARY: {len(RESULTS)} checks (x{WORLD} ranks), {nfail} failed")
    log(rank, "RESULT", "PASS" if nfail == 0 else "FAIL")
    torch.cuda.synchronize()
    comm.close()
    sys.stdout.flush()
    os._exit(0 if nfail == 0 else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--soak-replays", type=int, default=100000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--groups", default="ar,bits,rs,ag,det,bx,acc,seq,soak")
    ap.add_argument("--port", type=int, default=29722)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
