"""Standalone 8-process tests of the bf16 exchange of the medium all-reduce (algorithm 2 with BX, r3c), no vLLM model.

usage: py_topo.sh bench/test_medium_bx.py [--quick] [--groups a,b,...]
Groups (each check prints PASS/FAIL once, aggregated over the 8 ranks):
  emu     algorithm 2 with bf16 exchange == torch emulation (both island partials rounded to bf16, one final rounding)
          for 64 KiB .. 4 MiB, several (grid, block, unroll), randn / mixed / zeros / negzeros (mostly -0.0) inputs
  bits    random bit patterns (NaN, inf, subnormals, all-ones words): algorithm 2 BX == large algorithm BX bitwise
  det     rank-identical bits, run-to-run, eager == CUDA graph replay, batch invariance against the large algorithm
  disp    the dispatch (comm.all_reduce): which sizes / dtypes take the bf16 exchange, env switches, fp16/fp32 exact,
          the max_bytes edge (8 MiB takes algorithm 2, 8 MiB + 128 B the large algorithm) and the algorithm chosen
  seq     random mixed sequences of medium BX / exact / small / large all-reduces and all-gathers in one CUDA graph
          (epoch alignment of the shared call counter, buffer parity), and decode-layer sequences that interleave the
          BX all-reduce with the fused all_reduce_boundary / all_gather_boundary / all_gather_rmsnorm kernels on the
          shared regions (T = 8, 4 and a random mix), eager and graph replays against the Triton / torch references
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

    def seq_decode(bank):
        """Decode-layer sequences: the BX all-reduce next to the fused all_reduce_boundary / all_gather_rmsnorm /
        all_gather_boundary kernels, which share the a2/b2/c2 regions and the epoch counter with it (the real T = 8
        decode step is: embedding all-reduce, then per layer attn-out all-reduce + boundary (exact), pooler all-gather +
        RMSNorm, MoE-out all-reduce (64 KiB: BX), unpooler all-gather + boundary, and the logits all-gather). Every
        output is compared with its reference, once from the eager warm-up and once after 3 replays of a CUDA graph:
        the fused kernels with vLLM's Triton kernels applied to the exact contract / the concatenated shards, the
        all-reduces with the exact contract or the BX emulation, the all-gathers with the concatenation."""
        from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd, fused_rmsnorm_fwd

        D = 8192
        g = torch.Generator(device=dev)
        g.manual_seed(5)
        alpha = (torch.rand(D, generator=g, device=dev) * 0.9 + 0.1).float()
        beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
        post_w = (torch.randn(D, generator=g, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
        pre_w = (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
        w4 = (torch.randn(4096, generator=g, device=dev) * 0.3).to(torch.bfloat16)
        bf16 = torch.bfloat16

        def bnd_args(weighted):
            return (1e-8, 1e-6, 1.0, 1.0) if weighted else (1e-6, 1e-6, 0.0, 0.0)

        def bnd_ref(y, stream, weighted):
            post_eps, pre_eps, post_gain, pre_gain = bnd_args(weighted)
            f32 = torch.empty_like(stream)
            o, n = fused_norm_residual_norm_fwd(
                y, stream, post_w if weighted else None, pre_w, post_eps, pre_eps, alpha=alpha, beta=beta,
                post_gain_center=post_gain, pre_gain_center=pre_gain, post_input_scale=weighted, pre_input_scale=False,
                out_dtype=bf16, out_fp32=f32)
            return [o, n, f32]

        def ar_ref(xs):
            return contract_ref_bx(xs, members) if xs[0].numel() * 2 >= comm.ar_bx_min else contract_ref(xs, members)

        def stream_for(T, seed):
            gg = torch.Generator(device=dev)
            gg.manual_seed(seed)
            return (torch.randn(T, D, generator=gg, device=dev) * 2.0).float()

        def ar_item(shape, seed, kind="mixed"):
            xs = [make_input(r, shape, seed, dev, bf16, kind) for r in range(WORLD)]

            def emit(outs):
                outs.append(comm.all_reduce(xs[rank]))
            return emit, [ar_ref(xs)]

        def bank_item(p):
            def emit(outs):
                outs.append(comm.all_reduce(p[1][rank]))
            return emit, [p[3] if p[0] * KIB >= comm.ar_bx_min else p[2]]

        def gather_item(shape, seed):
            shards = [make_input(r, shape, seed, dev, bf16, "randn") for r in range(WORLD)]

            def emit(outs):
                outs.append(comm.all_gather(shards[rank], -1))
            return emit, [torch.cat(shards, dim=-1)]

        def layer_item(T, seed, weighted):
            attn = [make_input(r, (T, D), seed + 1, dev, bf16, "mixed") for r in range(WORLD)]
            pool = [make_input(r, (T, 512), seed + 2, dev, bf16, "mixed") for r in range(WORLD)]
            moe = [make_input(r, (T, 4096), seed + 3, dev, bf16, "mixed") for r in range(WORLD)]
            unpool = [make_input(r, (T, 1024), seed + 4, dev, bf16, "mixed") for r in range(WORLD)]
            st1, st2 = stream_for(T, seed + 5), stream_for(T, seed + 6)
            post_eps, pre_eps, post_gain, pre_gain = bnd_args(weighted)
            pw = post_w if weighted else None
            assert comm.supports_all_reduce_boundary(attn[rank], st1) and comm.supports_all_gather_rmsnorm(pool[rank])
            assert comm.supports_all_gather_boundary(unpool[rank], st2)

            def emit(outs):
                f1 = torch.empty_like(st1)
                o1, n1 = comm.all_reduce_boundary(attn[rank], st1, alpha, beta, pre_w, pw, post_eps, pre_eps, post_gain,
                                                  pre_gain, out_fp32=f1)
                o2 = comm.all_gather_rmsnorm(pool[rank], w4, 1e-6, 1.0)
                o3 = comm.all_reduce(moe[rank])
                f4 = torch.empty_like(st2)
                o4, n4 = comm.all_gather_boundary(unpool[rank], st2, alpha, beta, pre_w, pw, post_eps, pre_eps, post_gain,
                                                  pre_gain, out_fp32=f4)
                outs.extend([o1, n1, f1, o2, o3, o4, n4, f4])

            refs = bnd_ref(contract_ref(attn, members), st1, weighted)
            refs.append(fused_rmsnorm_fwd(torch.cat(pool, dim=-1), w4, 1e-6, gain_center=1.0, input_scale=False,
                                          use_kahan_summation=True))
            refs.append(ar_ref(moe))
            refs += bnd_ref(torch.cat(unpool, dim=-1), st2, weighted)
            return emit, refs

        def run_plan(name, items):
            refs = [r for _, rs in items for r in rs]

            def enqueue(outs):
                for emit, _ in items:
                    emit(outs)

            eager = []
            enqueue(eager)  # warm-up on the current stream, and an eager check
            torch.cuda.synchronize()
            replay = []
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                enqueue(replay)
            for _ in range(3):
                gr.replay()
            torch.cuda.synchronize()
            for tag, outs in (("eager", eager), ("graph, 3 replays", replay)):
                bad = [i for i, (a, b) in enumerate(zip(outs, refs)) if not bits_equal(a, b)]
                ck(f"decode sequence {name}: {len(items)} ops, {len(refs)} outputs, {tag} == references"
                   + (f" (outputs {bad[:6]} differ)" if bad or len(outs) != len(refs) else ""),
                   not bad and len(outs) == len(refs))
            del eager, replay, gr

        layers = 6 if not args.quick else 2
        for T in ([8, 4] if not args.quick else [8]):
            name = f"T={T}"
            items = [ar_item((T, D), 1000 + T)]  # embedding all-reduce: T = 8 is 128 KiB (BX), T = 4 is 64 KiB (BX)
            for layer in range(layers):
                items.append(layer_item(T, 2000 + 100 * T + 10 * layer, weighted=layer % 2 == 0))
            items.append(gather_item((T, 25256), 3000 + T))  # logits all-gather (404 KiB shard at T = 8: large engine)
            run_plan(name, items)
            del items
            free()
        if args.quick:
            return
        # random mix: layers of random T next to plain BX / exact / large all-reduces of the bank
        rng2 = random.Random(91)
        for trial in range(3):
            items = []
            for k in range(14):
                if rng2.random() < 0.6:
                    items.append(layer_item(rng2.choice([1, 2, 3, 4, 5, 8]), 4000 + 100 * trial + 7 * k, rng2.random() < 0.5))
                else:
                    items.append(bank_item(rng2.choice(bank)))
            run_plan(f"random mix {trial}", items)
            del items
            free()

    # ---- 1. emulation ------------------------------------------------------------------------------------------
    if "emu" in groups:
        for kib in sizes:
            shape = shape_for(kib * KIB)
            for kind in ("randn", "mixed", "zeros", "negzeros"):
                xs = [make_input(r, shape, 300 + kib, dev, torch.bfloat16, kind) for r in range(WORLD)]
                emu = contract_ref_bx(xs, members)
                exact = contract_ref(xs, members)
                if kind == "negzeros":
                    neg0 = int((emu.view(torch.int16) == -32768).sum().item())
                    ck(f"{kib} KiB negzeros: {neg0} of {emu.numel()} results are -0.0 (the input exercises the sign of zero)",
                       neg0 >= emu.numel() // 4)
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
        # (label, nbytes, shape): the KiB sizes plus the max_bytes edge: exactly max_bytes (a full phase-a slot) is the last
        # message of algorithm 2, one 128-byte pack more is the first one of the large algorithm
        cases = [(f"{kib} KiB", kib * KIB, shape_for(kib * KIB)) for kib in [32, 64, 128, 512, 1024, 2048, 4096, 8192]]
        edge = comm.max_bytes + 128
        assert comm.max_bytes == 8192 * KIB, comm.max_bytes
        cases.append(("8192 KiB + 128 B", edge, (edge // 128, 64)))
        paths = []
        comm._note = lambda op, x, algo: paths.append(algo)  # which branch of comm.all_reduce ran (instance attribute)

        def want_path(nb, medium, bx):
            if nb > comm.max_bytes:
                return "large"
            if bx and medium and nb >= comm.ar_bx_min:
                return "2bx"
            return "large" if nb >= comm.ar_large_min else 2

        for label, nb, shape in cases:
            assert int(torch.empty(shape, dtype=torch.bfloat16, device="meta").numel()) * 2 == nb, (label, shape)
            xs = [make_input(r, shape, 600 + nb // KIB, dev, torch.bfloat16, "mixed") for r in range(WORLD)]
            x = xs[rank]
            emu, exact = contract_ref_bx(xs, members), contract_ref(xs, members)
            del paths[:]
            comm.medium_bx, comm.bf16_exchange = True, True
            y_on = comm.all_reduce(x)
            comm.medium_bx = False
            y_off = comm.all_reduce(x)
            comm.medium_bx, comm.bf16_exchange = True, False
            y_exact = comm.all_reduce(x)
            comm.bf16_exchange = True
            torch.cuda.synchronize()
            want_bx_on = nb >= comm.ar_bx_min
            want_bx_off = nb >= comm.ar_large_min  # the large algorithm keeps its BX with the medium switch off
            ck(f"dispatch {label}, medium BX on: {'BX' if want_bx_on else 'exact'}",
               bits_equal(y_on, emu if want_bx_on else exact))
            ck(f"dispatch {label}, medium BX off: {'BX (large)' if want_bx_off else 'exact'}",
               bits_equal(y_off, emu if want_bx_off else exact))
            ck(f"dispatch {label}, bf16 exchange off: exact", bits_equal(y_exact, exact))
            want = [want_path(nb, True, True), want_path(nb, False, True), want_path(nb, True, False)]
            ck(f"dispatch {label}: algorithms {want} (medium BX on / medium BX off / bf16 exchange off), got {paths}",
               paths == want)
            del xs
            free()
        del comm._note
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
        seq_decode(bank)
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
