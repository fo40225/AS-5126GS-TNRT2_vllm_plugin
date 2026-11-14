"""S1-9 L9 A1: per-size sweep and before/after timing of the muse_topo dispatch tables under the bf16 exchange (the production setting).

The message sizes are the ones the production verify steps (K = 7, T = 8 x requests: replicated all-reduce / column all-gather up to
T = 320, sequence-parallel reduce-scatter / row all-gather from T = 384) and the 8192-token prefill chunks issue (docs/a1_table_retune.md
has the enumeration). Three timing protocols, all in CUDA graphs (what the model replays; the eager bands launch the same kernels):
  warm    N identical calls back to back (the protocol of the existing tables: L2-hot inputs and slots)
  spacer  a 50 us sleep kernel before each call, minus the sleep-only time
  cold    256 MiB memset (L2 flushed) + a tiny all-reduce that aligns the ranks, then the call, minus the memset + tiny all-reduce graph
Every number is the median over trials per rank; the headline is the max over ranks, and both islands ({0..3}, {4..7}) are reported.

Modes:
  sweep     candidate configurations per target against the dispatch path. Every candidate is checked bitwise on all ranks against the
            torch emulation of the arithmetic contract (all_reduce / reduce_scatter: fp32 chain per island, both island partials rounded to
            bf16, one final rounding) or the plain concatenation (all_gather) before it is timed; screening = one cold + one warm timing,
            finalists (best by the mean of the two ratios, best cold, best warm) are re-checked on four data kinds and timed in three
            interleaved rounds against the baseline with the three protocols.
  dispatch  the dispatch path (what the plugin calls) per target: three protocols, plus a digest of the output on every rank for four
            data kinds, so the rows of two host trees can be compared bit for bit (summ_a1.py before after).

usage (server, GPU lock): GPU=0,1,2,3,4,5,6,7 HOST_DIR=<host folder> <host folder>/scripts/runsrc_topo.sh <host folder>/bench/bench_a1_tables.py
       --mode sweep|dispatch --what ar,arl,rs,ag --out <jsonl> [--tag <label>] [--only <substring>]
"""
import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

from bench_h1a import GRIDS
from bench_medium import contract_ref_bx
from common import *  # noqa: F401,F403

KINDS = ("randn", "mixed", "negzeros", "zeros")
KiB = 1 << 10


# ---------------------------------------------------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------------------------------------------------
def targets(what):
    out = []
    if "ar" in what:  # medium bf16 all-reduces (the 2bx table), verify bands c1 .. c64 (T = 8 .. 512), attn-out / embedding / draft + MoE-out
        for T in (8, 16, 32, 64, 128, 192, 256, 320, 384, 512):
            out.append(dict(op="ar", shape=(T, 8192), name=f"ar_T{T}x8192"))
        for T in (8, 16, 32, 64, 128, 192, 256, 320):
            out.append(dict(op="ar", shape=(T, 4096), name=f"ar_T{T}x4096"))
    if "arl" in what:  # above max_bytes (8 MiB): the large all-reduce, c96 / c128 draft + embedding, prefill embedding
        for T in (768, 1024, 1536, 2048, 4096, 8192):
            out.append(dict(op="ar", shape=(T, 8192), name=f"ar_T{T}x8192"))
    if "rs" in what:
        for T in (64, 128, 192, 256, 384, 512, 768, 1024):
            out.append(dict(op="rs", shape=(T, 4096), name=f"rs_T{T}x4096"))
            out.append(dict(op="rs", shape=(T, 8192), name=f"rs_T{T}x8192"))
        for T in (4096, 8192):
            out.append(dict(op="rs", shape=(T, 4096), name=f"rs_T{T}x4096"))
            out.append(dict(op="rs", shape=(T, 8192), name=f"rs_T{T}x8192"))
    if "ag" in what:
        for R in (40, 48, 56, 64, 72, 80, 96, 112, 128, 512, 1024):
            out.append(dict(op="ag0", shape=(R, 8192), dtype="bf16", name=f"ag0_R{R}x8192"))
        for R in (48, 64, 96, 128, 512, 1024):
            out.append(dict(op="ag0", shape=(R, 2304), dtype="u8", name=f"ag0_R{R}x2304u8"))
        for R in (64, 128):
            out.append(dict(op="ag0", shape=(R, 4096), dtype="bf16", name=f"ag0_R{R}x4096"))
        for R in (48, 64, 128):
            out.append(dict(op="ag0", shape=(R, 16), dtype="i32", name=f"ag0_R{R}x16i32"))
        for R in (16, 32):
            out.append(dict(op="ag0", shape=(R, 2304), dtype="u8", name=f"ag0_R{R}x2304u8"))
        for T in (32, 64, 128, 192, 256, 320):
            out.append(dict(op="agc", shape=(T, 512), dtype="bf16", name=f"agc_T{T}x512"))
        for T in (16, 32, 64, 128, 192, 256, 320):
            out.append(dict(op="agc", shape=(T, 1024), dtype="bf16", name=f"agc_T{T}x1024"))
    return out


DT = {"bf16": torch.bfloat16, "u8": torch.uint8, "i32": torch.int32}


def inputs(t, kind, dev):
    dtype = DT[t.get("dtype", "bf16")]
    if dtype == torch.bfloat16:
        return [make_input(r, t["shape"], 55, dev, dtype, kind) for r in range(WORLD)]
    g = torch.Generator(device=dev)
    xs = []
    for r in range(WORLD):
        g.manual_seed(1000 + r)
        v = torch.randint(0, 256, (t["shape"][0], t["shape"][1] * dtype.itemsize), dtype=torch.uint8, device=dev, generator=g)
        v[:, ::7] = 0xFF  # sentinel-like words in the payload (all_gather must move any bits)
        xs.append(v.view(dtype) if dtype != torch.uint8 else v)
    return xs


# ---------------------------------------------------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------------------------------------------------
class Timer:
    """Graph timing under the three protocols (see the module docstring). Two cold variants:
      cold   (model-like) 256 MiB read (L2 clean-cold), then the producer of the input (a copy into x: the input is hot and dirty in
             L2 like the GEMM output it is in the model), then the aligning tiny all-reduce, then the call;
      coldw  (spike protocol of S1-9 L8) 256 MiB memset (L2 full of dirty lines), the aligning all-reduce, the call."""
    COLD_N = 6

    def __init__(self, comm, cpu_group, dev, rank, spacer_us=50.0):
        self.comm, self.cg, self.dev, self.rank = comm, cpu_group, dev, rank
        self.flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        self.flush_bf = self.flush.view(torch.bfloat16)
        self.tiny = torch.zeros(32, dtype=torch.bfloat16, device=dev)
        self.cycles = spin_cycles_for_us(spacer_us)
        self.sp_only = self._per_rank(build_graph(lambda: None, 100, self.cycles), 100, True)
        self.produce = None
        self.base = {}
        self.rebase()

    def set_input(self, x):
        """The tensor the cold protocol re-produces before each call (a copy from a spare buffer), or None."""
        self.produce = None if x is None else (lambda x=x, spare=x.clone(): x.copy_(spare))
        self.rebase()

    def _per_rank(self, g, n, quick):
        per = time_graph(g, n, self.cg, target_s=0.01 if quick else 0.05, trials=3 if quick else 5)
        return gather_all(statistics.median(per), self.cg)

    def _body(self, mode, call):
        def one():
            if mode == "cold":
                self.flush_bf.max()
                if self.produce is not None:
                    self.produce()
            else:
                self.flush.zero_()
            self.comm.all_reduce(self.tiny, out=self.tiny)
            call()

        return one

    def rebase(self):
        for mode in ("cold", "coldw"):
            one = self._body(mode, lambda: None)
            one()
            torch.cuda.synchronize()
            dist.barrier(self.cg)
            self.base[mode] = self._per_rank(build_graph(one, self.COLD_N), self.COLD_N, False)

    def warm(self, call, n, quick=False):
        return self._per_rank(build_graph(call, n), n, quick)

    def spacer(self, call, n, quick=False):
        per = self._per_rank(build_graph(call, n, self.cycles), n, quick)
        return [p - s for p, s in zip(per, self.sp_only)]

    def cold(self, call, quick=False, mode="cold"):
        one = self._body(mode, call)
        one()
        torch.cuda.synchronize()
        dist.barrier(self.cg)
        per = self._per_rank(build_graph(one, self.COLD_N), self.COLD_N, quick)
        return [p - b for p, b in zip(per, self.base[mode])]


def summ(per):
    return dict(max=round(max(per), 3), isl0=round(statistics.mean(per[:4]), 3), isl1=round(statistics.mean(per[4:]), 3))


def ncalls_for(nbytes):
    return 100 if nbytes < MiB else (20 if nbytes < 8 * MiB else 6)


# ---------------------------------------------------------------------------------------------------------------------
# one target: buffers, reference, baseline (dispatch) call, candidate list
# ---------------------------------------------------------------------------------------------------------------------
class Target:
    def __init__(self, ctx, t):
        self.ctx, self.t = ctx, t
        self.comm, self.rank, self.dev = ctx.comm, ctx.rank, ctx.dev
        self.op, self.shape = t["op"], tuple(t["shape"])
        dtype = DT[t.get("dtype", "bf16")]
        self.dtype = dtype
        self.nbytes = self.shape[0] * self.shape[1] * dtype.itemsize
        rows = self.shape[0]
        if self.op == "ar":
            self.out = torch.empty(self.shape, dtype=dtype, device=self.dev)
        elif self.op == "rs":
            self.out = torch.empty((rows // WORLD, self.shape[1]), dtype=dtype, device=self.dev)
        elif self.op == "ag0":
            self.out = torch.empty((rows * WORLD, self.shape[1]), dtype=dtype, device=self.dev)
        else:  # agc: column concatenation
            self.out = torch.empty((rows, self.shape[1] * WORLD), dtype=dtype, device=self.dev)
        self.x = torch.empty(self.shape, dtype=dtype, device=self.dev)  # persistent: the candidate closures capture it
        self.ref = None

    def load(self, kind):
        xs = inputs(self.t, kind, self.dev)
        self.x.copy_(xs[self.rank])
        if self.op == "ar":
            self.ref = contract_ref_bx(xs, self.comm.info.members)
        elif self.op == "rs":
            r = contract_ref_bx(xs, self.comm.info.members)
            n = self.shape[0] // WORLD
            self.ref = r[self.rank * n:(self.rank + 1) * n].contiguous()
        elif self.op == "ag0":
            self.ref = torch.cat(xs, 0)
        else:
            self.ref = torch.cat(xs, -1)
        del xs

    def dispatch(self):
        c = self.comm
        if self.op == "ar":
            return lambda: c.all_reduce(self.x, out=self.out)
        if self.op == "rs":
            return lambda: c.reduce_scatter(self.x, out=self.out)
        return lambda: c.all_gather(self.x, 0 if self.op == "ag0" else -1, out=self.out)

    def check(self, call):
        """Run once and compare bitwise with the reference on every rank (all-rank AND)."""
        self.out.zero_()
        call()
        torch.cuda.synchronize()
        return all(gather_all(bool(bits_equal(self.out, self.ref)), self.ctx.cg))

    def digest_out(self):
        return hashlib.sha256(bits(self.out).cpu().numpy().tobytes()).hexdigest()[:16]

    # ---- candidates -------------------------------------------------------------------------------------------------
    def candidates(self):
        c, x, out, nb = self.comm, self.x, self.out, self.nbytes
        cands = []  # (label, cfg dict, call)
        if self.op == "ar":
            if nb <= c.max_bytes and nb >= c.ar_bx_min:
                items = nb // 64
                blocks = (64, 128, 256, 512)
                unrolls = (1, 2) if nb >= 2 * MiB else (1, 2, 4)
                for block, unroll in itertools.product(blocks, unrolls):
                    if nb >= MiB and block == 64:
                        continue
                    gmax = max(1, -(-items // (block * unroll)))
                    for grid in sorted({g for g in GRIDS if g <= min(gmax, 128)} | {min(gmax, 128)}):
                        cfg = dict(kind="2bx", grid=grid, block=block, unroll=unroll)
                        cands.append((f"2bx g{grid} b{block} u{unroll}", cfg,
                                      lambda cfg=cfg: c._all_reduce(x, 2, cfg["block"], cfg["unroll"], 128, cfg["grid"], out, bx=True)))
            if c._large_ok(nb) and nb >= MiB:
                base = (1, 1 << 20, 4, 32, 512, 1, 0, 1)
                variants = [dict(zip(("mode", "chunk", "rs", "ag", "block", "unroll", "cut_rs", "cut_ag"), v)) for v in (
                    base, (0, 1 << 20, 4, 32, 256, 1, 0, 1), (1, 1 << 20, 4, 8, 512, 2, 0, 1), (1, 1 << 20, 8, 24, 512, 1, 0, 1),
                    (1, 1 << 20, 8, 8, 512, 2, 0, 1), (1, 1 << 19, 4, 8, 512, 1, 0, 1), (1, 1 << 20, 2, 8, 512, 2, 0, 1),
                    (1, 1 << 20, 4, 16, 512, 2, 0, 1), (1, 1 << 20, 16, 16, 512, 2, 0, 1), (1, 1 << 20, 4, 8, 512, 2, 1, 1),
                    (0, 1 << 20, 4, 32, 512, 1, 0, 1), (0, 1 << 20, 8, 32, 256, 1, 0, 1), (1, 1 << 19, 4, 8, 512, 2, 0, 1))]
                for v in variants:
                    cfg = dict(kind="largebx", **v)
                    cands.append((f"largebx m{v['mode']} ch{v['chunk'] >> 10}K rs{v['rs']} ag{v['ag']} b{v['block']} u{v['unroll']} c{v['cut_rs']}{v['cut_ag']}", cfg,
                                  lambda v=v: c._all_reduce_large(x, out, v["mode"], v["chunk"], v["rs"], v["ag"], v["block"], v["unroll"], True,
                                                                  v["cut_rs"], v["cut_ag"])))
            # row-chunked: n sub-calls, each through the dispatch table at the sub-size (bitwise: rows are independent)
            rows = self.shape[0]
            ns = [2, 3, 4, 8]
            if nb > c.max_bytes:
                ns.append(-(-nb // (4 * MiB)))  # slabs of at most 4 MiB (the 2bx row that is fastest per MiB)
            for n in sorted(set(ns)):
                per = -(-rows // n)
                if n > 32 or (nb <= 8 * MiB and n > 4) or per * self.shape[1] * 2 < 64 * KiB or nb < 2 * MiB:
                    continue
                pieces = [(x[r0:r0 + per], out[r0:r0 + per]) for r0 in range(0, rows, per)]

                def chunked(pieces=pieces):
                    for xp, op_ in pieces:
                        c.all_reduce(xp, out=op_)

                cands.append((f"chunked x{len(pieces)}", dict(kind="chunked", n=len(pieces)), chunked))
        elif self.op == "rs":
            if not c.supports("reduce_scatter", x):
                return cands
            seen = set()
            for ch, g, blk, un, cut in itertools.product((1 << 18, 1 << 19, 1 << 20), (2, 3, 4, 6, 8, 12, 16), (128, 256, 512), (1, 2), (0, 1)):
                # CUT needs >= 4 workers: the cut block polls 4 packs per thread ahead (rs_cut_body, Q = 4) and a worker grid smaller
                # than that never produces them before the partner's partial arrives (spin timeout, found on 2026-11-04)
                if ch * 8 > max(nb, 1 << 18) * 2 or (g, blk, un, cut, ch) in seen or (cut and g < 4):
                    continue
                seen.add((g, blk, un, cut, ch))
                cfg = dict(kind="rs", chunk=ch, grid=g, block=blk, unroll=un, cut=cut)
                cands.append((f"rs ch{ch >> 10}K g{g} b{blk} u{un} c{cut}", cfg,
                              lambda cfg=cfg: c._reduce_scatter_large(x, out, cfg["chunk"], cfg["grid"], cfg["block"], cfg["unroll"], True, cfg["cut"])))
        else:  # all_gather
            layout = 1 if self.op == "ag0" else 0
            d = 0 if layout == 1 else x.dim() - 1
            large_ok = c.large and nb % 16 == 0 and (layout == 1 or self.shape[1] * x.element_size() % 16 == 0)
            if large_ok:
                for ch, g, blk, cut in itertools.product((1 << 18, 1 << 19, 1 << 20), (2, 4, 8, 12, 16, 32), (256, 512, 1024), (0, 1)):
                    cfg = dict(kind="agl", chunk=ch, grid=g, block=blk, cut=cut)
                    cands.append((f"agl ch{ch >> 10}K g{g} b{blk} c{cut}", cfg,
                                  lambda cfg=cfg: c._all_gather_large(x, layout, out, cfg["chunk"], cfg["grid"], cfg["block"], cfg["cut"])))
            row_bytes = self.shape[1] * x.element_size()
            flag_ok = nb % 16 == 0 and (layout == 1 or row_bytes % 16 == 0) and nb <= c.ag_flag_max_bytes
            if flag_ok:
                for blk in (128, 256, 512, 1024):
                    for g in sorted({g for g in GRIDS if g <= min(128, max(1, nb // 16))}):
                        cfg = dict(kind="flag", grid=g, block=blk)
                        cands.append((f"flag g{g} b{blk}", cfg,
                                      lambda cfg=cfg: c._all_gather(x, d, layout, 2, cfg["block"], 1, 128, cfg["grid"], out)))
            if nb <= c.ag_ll_max_bytes:
                for blk, un in itertools.product((32, 64, 128, 256), (1, 2)):
                    gmax = max(1, -(-(nb // 4) // (blk * un)))
                    for g in sorted({g for g in GRIDS if g <= min(gmax, 128)} | {min(gmax, 128)}):
                        cfg = dict(kind="ll", grid=g, block=blk, unroll=un)
                        cands.append((f"ll g{g} b{blk} u{un}", cfg,
                                      lambda cfg=cfg: c._all_gather(x, d, layout, 1, cfg["block"], cfg["unroll"], 128, cfg["grid"], out)))
        return cands


# ---------------------------------------------------------------------------------------------------------------------
class Ctx:
    pass


def label_of(tg):
    return tg.t["name"]


def run_sweep(ctx, tg, args):
    rank, cg = ctx.rank, ctx.cg
    tm = ctx.timer
    tg.load("randn")
    tm.set_input(tg.x)
    disp = tg.dispatch()
    ok_base = tg.check(disp)
    n = ncalls_for(tg.nbytes)
    base_w = summ(tm.warm(disp, n, True))
    base_c = summ(tm.cold(disp, True))
    row0 = dict(target=label_of(tg), op=tg.op, bytes=tg.nbytes, shape=list(tg.shape), baseline_ok=ok_base, base_quick_warm=base_w, base_quick_cold=base_c)
    log(rank, f"== {label_of(tg)} {tg.nbytes / KiB:.0f} KiB baseline ok={ok_base} warm {base_w['max']:.1f} cold {base_c['max']:.1f}")
    scored = []
    invalid = 0
    for lab, cfg, call in tg.candidates():
        if ctx.trace:
            log(rank, "TRY", label_of(tg), lab)
        if not tg.check(call):
            invalid += 1
            continue
        w = summ(tm.warm(call, n, True))
        c_ = summ(tm.cold(call, True))
        scored.append((0.5 * (w["max"] / base_w["max"] + c_["max"] / base_c["max"]), lab, cfg, call, w, c_))
    tm.rebase()
    base_c2 = summ(tm.cold(disp, True))
    base_w2 = summ(tm.warm(disp, n, True))
    row0.update(invalid=invalid, screened=len(scored), base_quick_warm_end=base_w2, base_quick_cold_end=base_c2)
    scored_w = sorted(scored, key=lambda s: s[4]["max"])
    scored_c = sorted(scored, key=lambda s: s[5]["max"])
    scored_m = sorted(scored, key=lambda s: s[0])
    fin = []
    forced = [s for s in scored_m if s[1] in ctx.force]
    for s in scored_m[:3] + scored_c[:1] + scored_w[:1] + forced:
        if all(s[1] != f[1] for f in fin):
            fin.append(s)
    append_jsonl(ctx.out, [dict(row0, top_screen=[dict(label=s[1], score=round(s[0], 4), warm=s[4]["max"], cold=s[5]["max"]) for s in scored_m[:8]])], rank)
    if not fin:
        return
    # verification on all data kinds, then three interleaved rounds against the baseline
    verified = []
    for s in fin:
        okk = {}
        for kind in KINDS:
            tg.load(kind)
            okk[kind] = tg.check(s[3])
        tg.load("randn")
        verified.append(all(okk.values()))
    rounds = {"baseline": []}
    for s in fin:
        rounds[s[1]] = []
    for rnd in range(3):
        tm.rebase()
        for lab, call in [("baseline", disp)] + [(s[1], s[3]) for s in fin]:
            rounds[lab].append(dict(warm=summ(tm.warm(call, n)), spacer=summ(tm.spacer(call, n)), cold=summ(tm.cold(call)),
                                    coldw=summ(tm.cold(call, mode="coldw"))))
    rows = [dict(target=label_of(tg), op=tg.op, bytes=tg.nbytes, variant="baseline", rounds=rounds["baseline"])]
    for s, v in zip(fin, verified):
        rows.append(dict(target=label_of(tg), op=tg.op, bytes=tg.nbytes, variant=s[1], cfg=s[2], bitwise_all_kinds=v, rounds=rounds[s[1]]))
    append_jsonl(ctx.out, rows, rank)

    def med(lab, p, f="max"):
        return statistics.median(r[p][f] for r in rounds[lab])

    P = ("warm", "spacer", "cold", "coldw")
    b = {p: med("baseline", p) for p in P}
    log(rank, f"   baseline warm {b['warm']:.1f} spacer {b['spacer']:.1f} cold {b['cold']:.1f} coldw {b['coldw']:.1f}")
    for s, v in zip(fin, verified):
        m = {p: med(s[1], p) for p in P}
        log(rank, f"   {s[1]:44s} " + " ".join(f"{p} {m[p]:.1f} ({m[p] - b[p]:+.1f})" for p in P) + f" bitwise={v}")


def run_dispatch(ctx, tg, args):
    rank, cg = ctx.rank, ctx.cg
    tm = ctx.timer
    n = ncalls_for(tg.nbytes)
    digests, oks = {}, {}
    for kind in KINDS:
        tg.load(kind)
        disp = tg.dispatch()
        oks[kind] = tg.check(disp)
        digests[kind] = gather_all(tg.digest_out(), cg)
    tg.load("randn")
    tm.set_input(tg.x)
    disp = tg.dispatch()
    disp()
    torch.cuda.synchronize()
    rounds = []
    for rnd in range(args.rounds):
        tm.rebase()
        rounds.append(dict(warm=summ(tm.warm(disp, n)), spacer=summ(tm.spacer(disp, n)), cold=summ(tm.cold(disp)),
                           coldw=summ(tm.cold(disp, mode="coldw"))))
    row = dict(target=label_of(tg), tag=args.tag, op=tg.op, bytes=tg.nbytes, shape=list(tg.shape), bitwise_contract=oks, digests=digests, rounds=rounds)
    append_jsonl(ctx.out, [row], rank)
    med = {p: statistics.median(r[p]["max"] for r in rounds) for p in ("warm", "spacer", "cold", "coldw")}
    log(rank, f"[{args.tag}] {label_of(tg):22s} {tg.nbytes / KiB:8.0f} KiB warm {med['warm']:7.1f} spacer {med['spacer']:7.1f} cold {med['cold']:7.1f} "
              f"coldw {med['coldw']:7.1f} ok={all(oks.values())}")


def worker(rank, args):
    try:
        dev, cpu_group, _ = init_rank(rank, args.port, nccl=False)
        from muse_topo import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD, bf16_exchange=True)
        ctx = Ctx()
        ctx.comm, ctx.cg, ctx.dev, ctx.rank, ctx.out = comm, cpu_group, dev, rank, args.out
        ctx.trace = args.trace
        ctx.force = set(f for f in args.force.split(",") if f)
        log(rank, "TopoComm up", f"workspace {comm.workspace_bytes / 2**20:.1f} MiB", "tables:", os.environ.get("HOST_DIR", "?"))
        ctx.timer = Timer(comm, cpu_group, dev, rank)
        for t in targets(set(args.what.split(","))):
            if args.only and not any(o in t["name"] for o in args.only.split(",")):
                continue
            tg = Target(ctx, t)
            if tg.op == "rs" and not comm.supports("reduce_scatter", torch.empty(tg.shape, dtype=tg.dtype, device=dev)):
                continue
            (run_sweep if args.mode == "sweep" else run_dispatch)(ctx, tg, args)
            del tg
            torch.cuda.empty_cache()
        dist.barrier(cpu_group)
        log(rank, "ALL DONE")
        comm.close()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="dispatch", choices=("sweep", "dispatch"))
    ap.add_argument("--what", default="ar,arl,rs,ag")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--force", default="", help="comma list of candidate labels that are always finalists (e.g. 'chunked x4')")
    ap.add_argument("--trace", action="store_true", help="log every candidate before it runs (a deadlocking configuration kills the context)")
    ap.add_argument("--port", type=int, default=29731)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
