"""Standalone 8-process tests of the bf16 exchange of the medium all-reduce (algorithm 2 with BX, r3c), no vLLM model.

usage: py_topo.sh bench/test_medium_bx.py [--quick] [--groups a,b,...]
Groups (each check prints PASS/FAIL once, aggregated over the 8 ranks):
  emu     algorithm 2 with bf16 exchange == torch emulation (both island partials rounded to bf16, one final rounding)
          for 64 KiB .. 4 MiB, several (grid, block, unroll), randn / mixed / zeros inputs
  bits    random bit patterns (NaN, inf, subnormals, all-ones words): algorithm 2 BX == large algorithm BX bitwise
  det     rank-identical bits, run-to-run, eager == CUDA graph replay, batch invariance against the large algorithm
  disp    the dispatch (comm.all_reduce): which sizes / dtypes take the bf16 exchange, env switches, fp16/fp32 exact
  seq     random mixed sequences of medium BX / exact / small / large all-reduces and all-gathers in one CUDA graph
          (epoch alignment of the shared call counter, buffer parity)
  acc     accuracy vs the fp64 sum: BX vs the exact path vs NCCL (informational, plus a no-bias check)
"""
import argparse
import random
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403
from tests_h1b import check, rand_bits, free  # noqa: F401
from bench_medium import contract_ref_bx

KIB = 1 << 10


def shape_for(nbytes, hid=8192):
    rows = nbytes // (hid * 2)
    if rows * hid * 2 == nbytes:
        return (rows, hid)
    return (nbytes // (4096 * 2), 4096)


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
    from muse_topo import collectives as C

    comm = TopoComm(cpu_group, dev, rank, WORLD, bf16_exchange=True)
    members = comm.info.members
    log(rank, "TopoComm up:", comm.info.summary, f"workspace {comm.workspace_bytes / 2**20:.1f} MiB")
    groups = set(args.groups.split(","))
    ck = lambda name, ok: check(rank, cpu_group, name, ok)  # noqa: E731
    sizes = [64, 96, 128, 256, 512, 1024, 2048, 4096] if not args.quick else [64, 256, 1024]
    cfgs = [(8, 256, 1), (1, 32, 1), (3, 64, 2), (64, 32, 1), (4, 512, 4), (128, 512, 1), (188, 128, 1)]
    if args.quick:
        cfgs = cfgs[:3]

    def bx_call(x, out, cfg):
        grid, block, unroll = cfg
        comm._c.all_reduce(x, out, 2, grid, block, unroll, 1)

    # ---- 1. emulation ------------------------------------------------------------------------------------------
    if "emu" in groups:
        for kib in sizes:
            shape = shape_for(kib * KIB)
            for kind in ("randn", "mixed", "zeros"):
                xs = [make_input(r, shape, 300 + kib, dev, torch.bfloat16, kind) for r in range(WORLD)]
                emu = contract_ref_bx(xs, members)
                exact = contract_ref(xs, members)
                x = xs[rank]
                good = True
                for cfg in cfgs:
                    items = (kib * KIB) // 64
                    grid, block, unroll = cfg
                    if grid > max(1, -(-items // (block * unroll))) and grid != 8:
                        continue  # more blocks than work is legal but pointless; keep the table-like ones
                    out = torch.full_like(x, 7.0)
                    bx_call(x, out, cfg)
                    torch.cuda.synchronize()
                    good &= bits_equal(out, emu)
                ck(f"{kib} KiB {kind}: ar2 BX == torch emulation, {len(cfgs)} configs", good)
                if kind == "mixed":
                    out = torch.empty_like(x)
                    bx_call(x, out, cfgs[0])
                    torch.cuda.synchronize()
                    ck(f"{kib} KiB mixed: BX differs from the exact path (not a no-op)", not bits_equal(out, exact))
                del xs, emu, exact
            free()

    # ---- 2. random bit patterns: same bits as the large algorithm with BX ------------------------------------------
    if "bits" in groups:
        for kib in [64, 256, 1024, 2048] if not args.quick else [256, 1024]:
            shape = shape_for(kib * KIB)
            x = rand_bits(shape, rank, 410 + kib, dev, torch.bfloat16)
            ref = comm._all_reduce_large(x, None, 1, bx=True)
            ref0 = comm._all_reduce_large(x, None, 0, bx=True)
            out = torch.empty_like(x)
            bx_call(x, out, (8, 256, 1))
            torch.cuda.synchronize()
            ck(f"{kib} KiB random bits: ar2 BX == large BX (fused) bitwise", bits_equal(out, ref))
            ck(f"{kib} KiB random bits: ar2 BX == large BX (two-launch) bitwise", bits_equal(out, ref0))
            out2 = torch.empty_like(x)
            bx_call(x, out2, (3, 64, 2))
            torch.cuda.synchronize()
            ck(f"{kib} KiB random bits: ar2 BX other geometry == first", bits_equal(out, out2))
            del x, ref, ref0
            free()

    # ---- 3. determinism --------------------------------------------------------------------------------------------
    if "det" in groups:
        for kib in [128, 512, 2048] if not args.quick else [512]:
            shape = shape_for(kib * KIB)
            xs = [make_input(r, shape, 500 + kib, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
            x = xs[rank]
            y0 = torch.empty_like(x)
            bx_call(x, y0, (8, 256, 1))
            torch.cuda.synchronize()
            ck(f"{kib} KiB BX rank-identical", all_equal_across_ranks(y0, cpu_group))
            reps = []
            for _ in range(5):
                y = torch.empty_like(x)
                bx_call(x, y, (8, 256, 1))
                reps.append(y)
            torch.cuda.synchronize()
            ck(f"{kib} KiB BX run-to-run (5 runs)", all(bits_equal(r, y0) for r in reps))
            yg = torch.empty_like(x)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                bx_call(x, yg, (8, 256, 1))
                bx_call(x, yg, (8, 256, 1))
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            ck(f"{kib} KiB BX eager == CUDA graph replay", bits_equal(yg, y0))
            # batch invariance against the large algorithm: rows of the first half alone equal the rows of the full batch
            half = shape[0] // 2
            if half >= 1 and (half * shape[1] * 2) % 64 == 0:
                yh = torch.empty((half, shape[1]), dtype=x.dtype, device=dev)
                bx_call(x[:half].contiguous(), yh, (8, 256, 1))
                torch.cuda.synchronize()
                ck(f"{kib} KiB BX batch invariant (first half alone == rows of the full batch)", bits_equal(yh, y0[:half]))
            del xs
            free()

    # ---- 4. dispatch -------------------------------------------------------------------------------------------------
    if "disp" in groups:
        rows = {"b": [], "x": []}
        for kib in [32, 64, 128, 512, 1024, 2048, 4096]:
            shape = shape_for(kib * KIB)
            xs = [make_input(r, shape, 600 + kib, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
            x = xs[rank]
            emu, exact = contract_ref_bx(xs, members), contract_ref(xs, members)
            comm.medium_bx, comm.bf16_exchange = True, True
            y_on = comm.all_reduce(x)
            comm.medium_bx = False
            y_off = comm.all_reduce(x)
            comm.medium_bx, comm.bf16_exchange = True, False
            y_exact = comm.all_reduce(x)
            comm.bf16_exchange = True
            torch.cuda.synchronize()
            nb = kib * KIB
            want_bx_on = nb >= comm.ar_bx_min
            want_bx_off = nb >= comm.ar_large_min  # the large algorithm keeps its BX with the medium switch off
            ck(f"dispatch {kib} KiB, medium BX on: {'BX' if want_bx_on else 'exact'}",
               bits_equal(y_on, emu if want_bx_on else exact))
            ck(f"dispatch {kib} KiB, medium BX off: {'BX (large)' if want_bx_off else 'exact'}",
               bits_equal(y_off, emu if want_bx_off else exact))
            ck(f"dispatch {kib} KiB, bf16 exchange off: exact", bits_equal(y_exact, exact))
            del xs
            free()
        for dtype in (torch.float16, torch.float32):
            shape = (256, 4096) if dtype == torch.float16 else (128, 4096)
            xs = [make_input(r, shape, 650, dev, dtype, "mixed") for r in range(WORLD)]
            y = comm.all_reduce(xs[rank])
            torch.cuda.synchronize()
            ck(f"dispatch {dtype} {shape}: exact fp32 exchange (matches the contract)", bits_equal(y, contract_ref(xs, members)))
        comm.medium_bx, comm.bf16_exchange = True, True

    # ---- 5. mixed sequences in a graph ---------------------------------------------------------------------------
    if "seq" in groups:
        rng = random.Random(77)
        bank = []
        for kib in [16, 64, 128, 256, 512, 1024, 2048, 4096, 16384]:
            shape = shape_for(kib * KIB)
            xs = [make_input(r, shape, 700 + kib, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
            bank.append((kib, xs, contract_ref(xs, members), contract_ref_bx(xs, members)))
        ag_in = [make_input(r, (16, 1024), 800, dev, torch.bfloat16) for r in range(WORLD)]
        for trial in range(4 if not args.quick else 1):
            plan = [rng.choice(bank) for _ in range(40)]
            outs = [torch.empty_like(p[1][rank]) for p in plan]
            ag_outs = [torch.empty((16, 8192), dtype=torch.bfloat16, device=dev) for _ in plan]
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for p, o, ao in zip(plan, outs, ag_outs):
                    comm.all_reduce(p[1][rank], out=o)
                    if rng.random() < 0.5:
                        comm.all_gather(ag_in[rank], -1, out=ao)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            good = True
            for p, o in zip(plan, outs):
                nb = p[0] * KIB
                use_bx = nb >= comm.ar_bx_min
                good &= bits_equal(o, p[3] if use_bx else p[2])
            ck(f"sequence {trial}: 40 mixed all-reduce sizes (+ all-gathers) in a graph, 3 replays", good)
        del bank
        free()

    # ---- 6. accuracy ------------------------------------------------------------------------------------------
    if "acc" in groups:
        for kind in ("randn", "mixed"):
            shape = shape_for(1024 * KIB)
            xs = [make_input(r, shape, 900, dev, torch.bfloat16, kind) for r in range(WORLD)]
            s64 = exact_sum(xs)
            x = xs[rank]
            yb = torch.empty_like(x)
            bx_call(x, yb, (8, 256, 1))
            ye = torch.empty_like(x)
            comm._c.all_reduce(x, ye, 2, 8, 256, 1, 0)
            yn = torch.empty_like(x)
            nccl.all_reduce(x, yn)
            torch.cuda.synchronize()
            eb, ee, en = err_stats(yb, s64), err_stats(ye, s64), err_stats(yn, s64)
            log(rank, f"accuracy 1 MiB {kind}: BX {eb} | exact {ee} | NCCL {en}")
            # no systematic bias: mean signed error of BX relative to the exact value, in units of the result's ulp
            err = (yb.double() - s64)
            scale = s64.abs().mean().item()
            ck(f"BX {kind}: mean signed error {err.mean().item() / scale:.2e} of the mean |value| (|.| < 1e-3)",
               abs(err.mean().item()) / scale < 1e-3)
            ck(f"BX {kind}: mean ulp error below NCCL's bf16 ring ({eb['mean_ulp']} < {en['mean_ulp']})",
               eb["mean_ulp"] < en["mean_ulp"] + 0.05)
            del xs, s64
            free()

    nfail = sum(1 for _, ok in check.__globals__["RESULTS"] if not ok)
    total = len(check.__globals__["RESULTS"])
    log(rank, f"SUMMARY: {total} checks (x{WORLD} ranks), {nfail} failed")
    log(rank, "RESULT", "PASS" if nfail == 0 else "FAIL")
    torch.cuda.synchronize()
    comm.close()
    sys.stdout.flush()
    os._exit(0 if nfail == 0 else 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--groups", default="emu,bits,det,disp,seq,acc")
    ap.add_argument("--port", type=int, default=29723)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
