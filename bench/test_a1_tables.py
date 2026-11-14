"""Standalone tests of the dispatch tables of muse_topo under the production setting (bf16 exchange + medium bf16 exchange), S1-9 L9 A1.

usage: [SRC=<vLLM tree>] py_topo.sh bench/test_a1_tables.py [--quick] [--groups tables,ar,rs,ag,graph]
Groups:
  tables  (CPU) well-formedness of every dispatch table: ascending bounds ending at 1 << 62, geometry limits of the kernels
          (block multiple of 32 up to 512 / 1024, unroll 1 / 2 / 4, grid up to 128), chunk size limits, and the one geometry that
          deadlocks: a reduce-scatter / all-reduce with CUT blocks needs at least 4 workers (the cut block polls 4 packs per thread ahead,
          csrc/large.cu rs_cut_body; a smaller worker grid ends in a spin timeout)
  ar      dispatch all_reduce == torch emulation of the bf16-exchange contract (fp32 chain per island, both island partials rounded to
          bf16, one final rounding) for every size the verify steps / prefill chunks issue and for the bounds of the table rows
          (and the next size above each bound), data kinds randn / mixed / negzeros / zeros; rank-identical; batch invariant (the rows of
          a half message equal the rows of the full one)
  rs      dispatch reduce_scatter == rows block of the emulation, == the rows block of the dispatch all_reduce (128 KiB and up)
  ag      dispatch all_gather (dim 0 and dim -1, bf16 / uint8 / int32 payloads with all-ones words) == concatenation
  graph   the per-layer collective sequences of the verify bands (replicated c16 / c32, sequence-parallel c48 / c64 / c128) in one CUDA
          graph: replays 1 .. 3 == eager results (epoch / parity of the shared call counter across the multi-kernel calls)
"""
import argparse
import os
import random
import sys
import traceback

import torch
import torch.multiprocessing as mp

from bench_medium import contract_ref_bx
from common import *  # noqa: F401,F403
from tests_h1b import check, free

KIB = 1 << 10


def shape_for(nbytes, hid=8192):
    rows = nbytes // (hid * 2)
    if rows * hid * 2 == nbytes:
        return (rows, hid)
    return (nbytes // (4096 * 2), 4096)


def table_checks(C):
    """(name, ok) of the CPU-only table checks."""
    res = []

    def asc(name, tab):
        b = [r[0] for r in tab]
        res.append((f"{name}: ascending bounds ending at 1 << 62", all(x < y for x, y in zip(b, b[1:])) and b[-1] == 1 << 62))

    def blk_ok(blk, hi):
        return 32 <= blk <= hi and blk % 32 == 0

    for name in ("_AR_TABLE", "_AR_BX_TABLE"):
        t = getattr(C, name)
        asc(name, t)
        res.append((f"{name}: algorithm 2/1/3 rows, grid <= 128, block 32..512, unroll 1/2/4",
                    all(r[1] in (1, 2, 3) and 0 <= r[2] <= C.MAX_GRID and blk_ok(r[3], 512) and r[4] in (1, 2, 4) for r in t)))
        if name == "_AR_BX_TABLE":
            res.append((f"{name}: algorithm 2 only (the bf16 exchange exists for algorithm 2)", all(r[1] == 2 for r in t)))
    for name in ("_AG_TABLES",):
        for lay, t in getattr(C, name).items():
            asc(f"{name}[{lay}]", t)
            res.append((f"{name}[{lay}]: algorithm 1/2, grid <= 128, block 32..1024", all(r[1] in (1, 2) and 0 <= r[2] <= C.MAX_GRID and blk_ok(r[3], 1024) for r in t)))
    for name in ("_AR_LARGE_TABLE", "_AR_LARGE_BX_TABLE"):
        if hasattr(C, name):
            t = getattr(C, name)
            asc(name, t)
            res.append((f"{name}: block 32..512, unroll 1/2, chunk <= 1 MiB, cut blocks need >= 4 workers",
                        all(blk_ok(r[5], 512) and r[6] in (1, 2) and 16 <= r[2] <= 1 << 20 and r[2] % 16 == 0
                            and (not r[7] or r[3] >= 4) and r[1] in (0, 1) for r in t)))
    for name in ("_RS_TABLE", "_RS_BX_TABLE"):
        if hasattr(C, name):
            t = getattr(C, name)
            asc(name, t)
            res.append((f"{name}: block 32..512, unroll 1/2, chunk <= 1 MiB multiple of 16, CUT needs >= 4 workers",
                        all(blk_ok(r[3], 512) and r[4] in (1, 2) and 16 <= r[1] <= 1 << 20 and r[1] % 16 == 0 and r[2] >= 1
                            and (not r[5] or r[2] >= 4) for r in t)))
    asc("_AGL_TABLE", C._AGL_TABLE)
    res.append(("_AGL_TABLE: block 32..1024, chunk <= 1 MiB multiple of 16", all(blk_ok(r[3], 1024) and 16 <= r[1] <= 1 << 20 and r[1] % 16 == 0 and r[2] >= 1 for r in C._AGL_TABLE)))
    return res


def worker(rank, args):
    try:
        run(rank, args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)


def make(kind, shape, r, seed, dev):
    return make_input(r, shape, seed, dev, torch.bfloat16, kind)


def run(rank, args):
    dev, cpu_group, _ = init_rank(rank, args.port, nccl=False)
    from muse_topo import TopoComm
    from muse_topo import collectives as C
    from tests_h1b import RESULTS

    comm = TopoComm(cpu_group, dev, rank, WORLD, bf16_exchange=True)
    members = comm.info.members
    log(rank, "TopoComm up:", comm.info.summary, f"workspace {comm.workspace_bytes / 2**20:.1f} MiB")
    groups = set(args.groups.split(","))
    ck = lambda name, ok: check(rank, cpu_group, name, ok)  # noqa: E731
    kinds = ("randn", "negzeros") if args.quick else ("randn", "mixed", "negzeros", "zeros")

    if "tables" in groups:
        for name, ok in table_checks(C):
            ck("tables " + name, ok)

    # sizes: the verify bands / prefill messages (docs/a1_table_retune.md) and every bound of the tables (and the next row size above)
    band_T = (8, 16, 32, 64, 128, 192, 256, 320, 384, 512, 768, 1024)
    ar_sizes = set()
    for T in band_T:
        for hid in (4096, 8192):
            ar_sizes.add((T, hid))
    for tab in (C._AR_BX_TABLE,):
        for r in tab[:-1]:
            ar_sizes.add(shape_for(r[0]))
            s = shape_for(r[0])
            ar_sizes.add((s[0] + 1, s[1]) if s[0] * s[1] * 2 + s[1] * 2 <= C._LARGE_MAX_BYTES else s)
    ar_sizes = sorted(s for s in ar_sizes if 64 * KIB <= s[0] * s[1] * 2 <= 16 << 20 and (s[0] * s[1] * 2) % 128 == 0)
    if args.quick:
        ar_sizes = [s for s in ar_sizes if s[0] in band_T][::2]
    if "ar" in groups:
        for (rows, hid) in ar_sizes:
            nb = rows * hid * 2
            for kind in kinds:
                xs = [make(kind, (rows, hid), r, 91, dev) for r in range(WORLD)]
                ref = contract_ref_bx(xs, members)
                y = comm.all_reduce(xs[rank])
                ck(f"ar {rows}x{hid} ({nb / KIB:.0f} KiB) {kind} == torch emulation (bf16 exchange)", bits_equal(y, ref))
                if kind == "randn":
                    ck(f"ar {rows}x{hid} rank-identical", all_equal_across_ranks(y, cpu_group))
                    if rows % 2 == 0 and nb // 2 >= 64 * KIB:
                        yh = comm.all_reduce(xs[rank][: rows // 2].contiguous())
                        ck(f"ar {rows}x{hid} batch invariant (half message)", bits_equal(yh, y[: rows // 2]))
                del xs, ref, y
            free()
    if "rs" in groups:
        rs_sizes = {(T, hid) for T in (384, 392, 512, 768, 1024, 2048, 4096, 8192) for hid in (4096, 8192)}
        for tab in (getattr(C, "_RS_BX_TABLE", C._RS_TABLE), C._RS_TABLE):
            for r in tab[:-1]:
                rs_sizes.add(shape_for(r[0]))
        rs_sizes = sorted(s for s in rs_sizes if s[0] % 8 == 0 and 128 * KIB <= s[0] * s[1] * 2 <= (128 << 20))
        if args.quick:
            rs_sizes = [s for s in rs_sizes if s[0] * s[1] * 2 <= (16 << 20)]
        for (rows, hid) in rs_sizes:
            nb = rows * hid * 2
            tb = rows // WORLD
            for kind in kinds[:2] if nb > (32 << 20) else kinds:
                xs = [make(kind, (rows, hid), r, 92, dev) for r in range(WORLD)]
                ref = contract_ref_bx(xs, members)
                y = comm.reduce_scatter(xs[rank])
                ck(f"rs {rows}x{hid} ({nb / KIB:.0f} KiB) {kind} == rows block of the torch emulation", bits_equal(y, ref[rank * tb:(rank + 1) * tb]))
                if kind == "randn":
                    if nb <= 16 << 20:
                        ya = comm.all_reduce(xs[rank])
                        ck(f"rs {rows}x{hid} == rows block of the dispatch all_reduce", bits_equal(y, ya[rank * tb:(rank + 1) * tb]))
                        del ya
                    ck(f"rs {rows}x{hid} rank block reproducible", bits_equal(y, comm.reduce_scatter(xs[rank])))
                del xs, ref, y
            free()
    if "ag" in groups:
        cases = []  # (shape, dtype, layout dim)
        for R in (6, 48, 64, 96, 128, 256, 512, 1024):
            cases += [((R, 8192), torch.bfloat16, 0), ((R, 2304), torch.uint8, 0), ((R, 4096), torch.bfloat16, 0), ((R, 16), torch.int32, 0)]
        for T in (8, 16, 32, 64, 128, 192, 256, 320):
            cases += [((T, 512), torch.bfloat16, -1), ((T, 1024), torch.bfloat16, -1)]
        for r in C._AG_TABLES[1][:-1] + C._AGL_TABLE[:-1]:
            cases.append(((max(1, r[0] // 2048), 1024), torch.bfloat16, 0))  # a shard of about the bound's bytes
        if args.quick:
            cases = cases[::3]
        g = torch.Generator(device=dev)
        for shape, dtype, dim in cases:
            nb = shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size()
            if nb % 4 or (dim == -1 and (shape[1] * torch.empty((), dtype=dtype).element_size()) % 4):
                continue
            xs = []
            for r in range(WORLD):
                g.manual_seed(7 + r)
                v = torch.randint(0, 256, (shape[0], shape[1] * torch.empty((), dtype=dtype).element_size()), dtype=torch.uint8, device=dev, generator=g)
                v[:, ::5] = 0xFF
                xs.append(v.view(dtype).reshape(shape))
            x = xs[rank].contiguous()
            if not comm.supports("all_gather", x, dim):
                continue
            y = comm.all_gather(x, dim)
            ref = torch.cat(xs, 0 if dim == 0 else -1)
            ck(f"ag dim {dim} {tuple(shape)} {str(dtype).replace('torch.', '')} ({nb / KIB:.1f} KiB shard) == concatenation", bits_equal(y, ref))
            del xs, y, ref
        free()
    if "graph" in groups:
        seqs = {
            "c16": [("ar", (128, 8192)), ("ar", (128, 4096)), ("ag", (128, 512, -1)), ("ag", (128, 1024, -1))],
            "c32": [("ar", (256, 8192)), ("ar", (256, 4096)), ("ag", (256, 512, -1)), ("ag", (256, 1024, -1))],
            "c40": [("ar", (320, 8192)), ("ar", (320, 4096)), ("ag", (320, 512, -1)), ("ag", (320, 1024, -1))],
            "c48": [("ag", (48, 8192, 0)), ("ag", (48, 2304, 0)), ("rs", (384, 8192)), ("rs", (384, 4096))],
            "c64": [("ag", (64, 8192, 0)), ("ag", (64, 2304, 0)), ("rs", (512, 8192)), ("rs", (512, 4096)), ("ar", (512, 8192))],
            "c128": [("ag", (128, 8192, 0)), ("ag", (128, 2304, 0)), ("rs", (1024, 8192)), ("rs", (1024, 4096)), ("ar", (1024, 8192))],
        }
        for name, seq in seqs.items():
            bufs = []
            for i, (op, shp) in enumerate(seq):
                if op == "ag":
                    shape = shp[:2]
                    dtype = torch.uint8 if shape[1] == 2304 else torch.bfloat16
                    x = (rand_u8(shape, rank, i, dev) if dtype == torch.uint8 else make("mixed", shape, rank, 50 + i, dev))
                else:
                    x = make("mixed", shp, rank, 60 + i, dev)
                bufs.append((op, shp, x))

            def body():
                outs = []
                for op, shp, x in bufs:
                    if op == "ar":
                        outs.append(comm.all_reduce(x))
                    elif op == "rs":
                        outs.append(comm.reduce_scatter(x))
                    else:
                        outs.append(comm.all_gather(x, shp[2]))
                return outs

            eager = [o.clone() for o in body()]
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gouts = body()
            ok = True
            for rep in range(3):
                g.replay()
                torch.cuda.synchronize()
                ok = ok and all(bits_equal(a, b) for a, b in zip(eager, gouts))
            # interleave with eager calls again (the epoch counter keeps advancing across graph / eager)
            again = body()
            torch.cuda.synchronize()
            ok = ok and all(bits_equal(a, b) for a, b in zip(eager, again))
            ck(f"graph {name}: {len(seq)} collectives, 3 replays + eager == first eager run", ok)
            del g, gouts, again, eager, bufs
            free()
    fails = [n for n, ok in RESULTS if not ok]
    dist.barrier(cpu_group)
    log(rank, f"{len(RESULTS)} checks, {len(fails)} failures" + ("" if not fails else f": {fails[:10]}"))
    if rank == 0:
        print("ALL TESTS PASSED" if not fails else "SOME TESTS FAILED", flush=True)
    comm.close()
    sys.stdout.flush()
    os._exit(0 if not fails else 1)


def rand_u8(shape, rank, seed, dev):
    g = torch.Generator(device=dev)
    g.manual_seed(1234 + seed * 31 + rank)
    v = torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)
    v[:, ::5] = 0xFF
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--groups", default="tables,ar,rs,ag,graph")
    ap.add_argument("--port", type=int, default=29761)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
