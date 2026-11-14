"""Standalone tests of the dispatch tables of muse_topo under the production setting (bf16 exchange + medium bf16 exchange), S1-9 L9 A1.

usage: [SRC=<vLLM tree>] py_topo.sh bench/test_a1_tables.py [--quick] [--groups tables,ar,rs,ag,graph,soak] [--soak-seconds N]
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
  slab    (part of ar) the row-slab path above max_bytes (8 MiB < bytes <= 64 MiB): odd / uneven row counts of the prefill and mixed steps
          (T = 513 .. 4096 at 16 KiB rows, 8 KiB rows), the 64 MiB cap and the first row above it (4097 rows: the large algorithm): every
          result == torch emulation, == the large algorithm (slabs switched off in the test), with ``out=`` aligned / 64-byte offset /
          2-byte misaligned (guard words untouched) and a misaligned input, the slab count and size from the gate
  soak    (not in the default groups; --soak-seconds N) one CUDA graph + eager calls at the retuned sizes (reduce-scatter 1.5 .. 4 MiB, the
          1.5 / 2 MiB all-reduce rows, 2bx slabs 8.1 .. 64 MiB odd and even, a small op) replayed back to back for N seconds, bitwise
          against the references every few hundred replays; a hang ends in the kernels' spin timeout (MUSE_TOPO_TIMEOUT_S, 600 s) and
          fails the run
"""
import argparse
import os
import random
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

from bench_medium import contract_ref_bx
from common import *  # noqa: F401,F403
from tests_h1b import check, free

KIB = 1 << 10
AR_MAX_BYTES = (64 << 20) + 16 * KIB  # the slab cap (64 MiB) plus one 16 KiB row: [4097, 8192] is the first size above it


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


def slab_checks(ck, comm, C, members, rank, dev, cpu_group, rows, hid):
    """The row-slab path of an all-reduce above max_bytes (S1-9 L9 A1), or the first size above its cap: gate, slab count and size, equality with the
    torch emulation and with the large algorithm (slabs switched off), ``out=`` handling (aligned, 64-byte offset, 2-byte misaligned: the result is
    the caller's buffer, the words around it are untouched) and a misaligned input."""
    nb, rowb, n = rows * hid * 2, hid * 2, rows * hid
    xs = [make("mixed", (rows, hid), r, 93, dev) for r in range(WORLD)]
    ref = contract_ref_bx(xs, members)
    x = xs[rank]
    del xs
    per = comm._bx_slab_rows(x, nb)
    in_cap = nb <= C._AR_BX_SLABS_MAX_BYTES
    tag = f"slab {rows}x{hid} ({nb / KIB:.0f} KiB)"
    ck(f"{tag}: the gate {'takes' if in_cap else 'rejects'} the shape ({per} rows per slab)", (per > 0) == in_cap and (per == 0 or per * rowb <= comm.max_bytes))
    calls = []
    orig = comm._all_reduce
    comm._all_reduce = lambda *a, **k: (calls.append(a[0].shape[0]), orig(*a, **k))[1]
    try:
        y = comm.all_reduce(x)
    finally:
        del comm._all_reduce
    want = [min(per, rows - r0) for r0 in range(0, rows, per)] if per else []
    ck(f"{tag}: {len(want)} slab launches {want[:1]}.. covering all {rows} rows (none empty)", calls == want and sum(calls) == (rows if per else 0))
    ck(f"{tag}: == torch emulation", bits_equal(y, ref))
    saved = C._AR_BX_SLABS_MAX_BYTES
    C._AR_BX_SLABS_MAX_BYTES = 0  # slabs off on every rank: the large algorithm with the bf16 exchange serves the message
    try:
        yl = comm.all_reduce(x)
    finally:
        C._AR_BX_SLABS_MAX_BYTES = saved
    ck(f"{tag}: == the large algorithm with the bf16 exchange", bits_equal(y, yl))
    flat = torch.zeros(n + 64, dtype=torch.bfloat16, device=dev)
    for off, label in ((0, "aligned"), (32, "64-byte offset"), (1, "2-byte misaligned")):
        flat.zero_()
        o = flat[off:off + n].view(rows, hid)
        r_ = comm.all_reduce(x, out=o)
        guard = not bool(flat[:off].view(torch.int16).any()) and not bool(flat[off + n:].view(torch.int16).any())
        ck(f"{tag}: out= {label}: returns out, == emulation, the words around it untouched", r_.data_ptr() == o.data_ptr() and bits_equal(o, ref) and guard)
    xf = torch.empty(n + 1, dtype=torch.bfloat16, device=dev)
    xf[1:] = x.reshape(-1)
    ck(f"{tag}: misaligned input (2-byte offset) == emulation", bits_equal(comm.all_reduce(xf[1:].view(rows, hid)), ref))
    del x, ref, y, yl, flat, xf


def soak(ck, comm, C, members, rank, dev, cpu_group, seconds):
    """Back-to-back replays of ONE CUDA graph holding the retuned sizes (reduce-scatter 1.5 .. 4 MiB, the 1.5 / 2 MiB all-reduce rows, 2bx slabs from
    8.1 MiB to the 64 MiB cap with odd and even row counts, the first size above the cap, a small op) for ``seconds`` of wall clock, with an eager
    round of the same calls between the chunks (graph and eager calls share the epoch counter). Every result is compared bitwise with the torch
    emulation after each chunk of replays; the stop decision is rank 0's, so all ranks replay the same number of times."""
    specs = [("rs", (96, 8192)), ("rs", (128, 8192)), ("rs", (192, 8192)), ("rs", (256, 8192)), ("rs", (384, 4096)), ("rs", (512, 4096)),
             ("ar", (96, 8192)), ("ar", (128, 8192)), ("ar", (8, 8192)), ("ar", (519, 8192)), ("ar", (768, 8192)), ("ar", (1001, 8192)),
             ("ar", (1024, 8192)), ("ar", (2049, 8192)), ("ar", (4096, 8192)), ("ar", (4097, 8192))]
    bufs = []
    for i, (op, shape) in enumerate(specs):
        xs = [make("mixed", shape, r, 300 + i, dev) for r in range(WORLD)]
        ref = contract_ref_bx(xs, members)
        if op == "rs":
            tb = shape[0] // WORLD
            ref = ref[rank * tb:(rank + 1) * tb].contiguous()
        bufs.append((op, xs[rank], ref))
        del xs
        free()

    def body():
        return [comm.all_reduce(x) if op == "ar" else comm.reduce_scatter(x) for op, x, _ in bufs]

    eager = body()
    torch.cuda.synchronize()
    ok = all(bits_equal(o, ref) for o, (_, _, ref) in zip(eager, bufs))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gouts = body()
    chunk, replays, rounds, t0, tlog = 200, 0, 0, time.time(), time.time()
    while ok:
        for _ in range(chunk):
            g.replay()
        replays += chunk
        torch.cuda.synchronize()
        ok = all(bits_equal(o, ref) for o, (_, _, ref) in zip(gouts, bufs))
        eager = body()
        torch.cuda.synchronize()
        rounds += 1
        ok = ok and all(bits_equal(o, ref) for o, (_, _, ref) in zip(eager, bufs))
        ok = all(gather_all(bool(ok), cpu_group))
        stop = torch.tensor([1 if time.time() - t0 >= seconds else 0], dtype=torch.int32)
        dist.broadcast(stop, src=0, group=cpu_group)
        if time.time() - tlog >= 120:
            tlog = time.time()
            log(rank, f"soak: {replays} graph replays, {rounds} eager rounds, {time.time() - t0:.0f} s, bits ok={ok}")
        if stop.item():
            break
    ck(f"soak: {replays} graph replays of {len(specs)} collectives + {rounds} eager rounds in {time.time() - t0:.0f} s, every result == emulation", ok)
    del g, gouts, eager, bufs
    free()


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
    # the slab path (S1-9 L9 A1): bf16 messages above max_bytes (8 MiB) up to the 64 MiB cap run as row slabs. The prefill and mixed steps issue
    # [T, 8192] all-reduces at every odd T from 513 to 4096 (round-5 log: 405 distinct shapes), the c96 / c128 draft all-reduce is [768 / 1024, 8192];
    # 4097 rows (64 MiB + 16 KiB) is the first message above the cap (large algorithm), 8 KiB rows are the [T, 4096] latent shapes
    slab_T = (513, 519, 520, 600, 700, 768, 1001, 1536, 2049, 3000, 4096, 4097)
    for T in slab_T:
        ar_sizes.add((T, 8192))
    for T in (1025, 2049, 4097):
        ar_sizes.add((T, 4096))
    ar_sizes = sorted(s for s in ar_sizes if 64 * KIB <= s[0] * s[1] * 2 <= AR_MAX_BYTES and (s[0] * s[1] * 2) % 128 == 0)
    if args.quick:
        ar_sizes = [s for s in ar_sizes if s[0] in band_T or s[0] in (519, 2049, 4096, 4097)][::2]
    if "ar" in groups:
        for (rows, hid) in ar_sizes:
            nb = rows * hid * 2
            for kind in (kinds[:2] if nb > (16 << 20) else kinds):
                xs = [make(kind, (rows, hid), r, 91, dev) for r in range(WORLD)]
                ref = contract_ref_bx(xs, members)
                y = comm.all_reduce(xs[rank])
                ck(f"ar {rows}x{hid} ({nb / KIB:.0f} KiB) {kind} == torch emulation (bf16 exchange)", bits_equal(y, ref))
                if kind == "randn":
                    ck(f"ar {rows}x{hid} rank-identical", all_equal_across_ranks(y, cpu_group))
                    if rows % 2 == 0 and nb // 2 >= 64 * KIB:
                        yh = comm.all_reduce(xs[rank][: rows // 2].contiguous())
                        ck(f"ar {rows}x{hid} batch invariant (half message)", bits_equal(yh, y[: rows // 2]))
                    if nb > comm.max_bytes:
                        # a different row count splits into other slabs: the rows of the shorter message must be the same bits
                        m = rows - 3
                        ym = comm.all_reduce(xs[rank][:m].contiguous())
                        ck(f"ar {rows}x{hid} batch invariant ({m} of {rows} rows, other slab split)", bits_equal(ym, y[:m]))
                del xs, ref, y
            if nb > comm.max_bytes:
                slab_checks(ck, comm, C, members, rank, dev, cpu_group, rows, hid)
            free()
    if "rs" in groups:
        # 64 .. 320 rows: the 1.5 .. 4 MiB messages of the _RS_BX_TABLE row (and 1 MiB / 5 MiB around it); 384 .. 8192: the sequence-parallel bands and prefill
        rs_sizes = {(T, hid) for T in (64, 96, 128, 192, 256, 320, 384, 392, 512, 640, 768, 1024, 2048, 4096, 8192) for hid in (4096, 8192)}
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
                    if nb <= 64 << 20:
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
            # c96: the draft all-reduce [768, 8192] (12 MiB) is served as slabs; odd row counts of the prefill / mixed steps and the 64 MiB cap
            "c96": [("ag", (96, 8192, 0)), ("ag", (96, 2304, 0)), ("rs", (768, 8192)), ("rs", (768, 4096)), ("ar", (768, 8192))],
            "slabs": [("ar", (519, 8192)), ("ar", (1001, 8192)), ("ar", (2049, 8192)), ("ar", (4096, 8192)), ("ar", (1025, 4096))],
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
    if "soak" in groups:
        soak(ck, comm, C, members, rank, dev, cpu_group, args.soak_seconds)
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
    ap.add_argument("--soak-seconds", type=int, default=1800, help="wall clock of the soak group (not in the default groups)")
    ap.add_argument("--port", type=int, default=29761)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
