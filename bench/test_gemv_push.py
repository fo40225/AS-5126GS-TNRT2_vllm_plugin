"""8-process test of the sharded GEMV fused with the all-gather (TopoComm.gemv_all_gather_rmsnorm / gemv_all_gather_boundary).

Three layers of reference:
  1. The GEMV itself (csrc/gemv.cuh): the shard is checked bit for bit against an emulation of its summation order in torch fp32
     (``emulate_gemv``), for T = 1..8, both projections, random / outlier-heavy / zero data. The order is spelled out in gemv.cuh.
  2. The fused calls: ``gemv_all_gather_rmsnorm`` / ``gemv_all_gather_boundary`` against ``TopoComm.all_gather`` of the emulated shards
     followed by the Triton kernels of the model (the unfused production path), bitwise and identical on all ranks; with and without the
     pushes of the GEMV kernel (``comm.gemv_push``).
  3. Accuracy: the shard against the exact fp64 product (fraction of elements that are not the correctly rounded value, max error in
     bf16 ulps) next to the cuBLAS shard, to document that the order is as good as the one it replaces.
Also eager and in CUDA graphs next to other collectives (call counter and double-buffer parity, the early start of the GEMV kernel
behind the all-reduce in front of it, with the exact and the bf16-exchange (BX) all-reduce, push modes 0 .. 6), operands misaligned on
odd ranks only (activations; and the static weights / parameters, which make the call launch its GEMV kernel without programmatic
serialization), replay determinism. Also: the dispatch gates at the default MAX_ROWS (4), the rejection of the push modes that would hang
(3, 7, -1, 8), validation of the gather's operands before anything is launched, MUSE_TOPO_PDL=1, and the trigger stage as a property of
the launch (a PDL-launched sentinel kernel behind the all-reduces that carry a stage, and behind those that must not).

usage (server, under the GPU lock): GPU=0,1,2,3,4,5,6,7 SRC=<vllm tree> scripts/py_topo.sh bench/test_gemv_push.py [--quick]
"""
import argparse
import os
import sys
import traceback

import torch
import torch.multiprocessing as mp
import triton
import triton.language as tl

from common import *  # noqa: F401,F403

RESULTS = []
D = 8192
GEOM = {0: dict(N=512, K=8192), 1: dict(N=1024, K=4096)}
# warps per output row of the geometry variants (keep in sync with PoolerV* / UnpoolerV* in csrc/gemv_push.cu)
VARIANT_WPR = {0: [2, 4, 4, 2], 1: [1, 2, 2, 1]}


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


def emulate_gemv(x, w, wpr):
    """Bit-exact torch model of gemv_push_kernel's arithmetic (csrc/gemv.cuh): x [T, K] bf16, w [N, K] bf16 -> [T, N] bf16."""
    T, K = x.shape
    N = w.shape[0]
    lpl = K // 8 // wpr // 32
    xf = x.float().view(T, 1, wpr, lpl, 32, 8)
    wf = w.float().view(1, N, wpr, lpl, 32, 8)
    p = xf * wf  # exact in fp32: a product of two bf16 values has at most 16 significant bits
    acc = [torch.zeros(T, N, wpr, 32, dtype=torch.float32, device=x.device) for _ in range(4)]
    for j in range(lpl):
        for e in range(8):
            acc[e & 3] = acc[e & 3] + p[:, :, :, j, :, e]
    v = (acc[0] + acc[1]) + (acc[2] + acc[3])  # [T, N, wpr, 32]
    for half in (16, 8, 4, 2, 1):  # xor butterfly: lane i + lane i ^ half; every lane ends with the same value
        v = v[..., :half] + v[..., half:2 * half]
    v = v[..., 0]  # [T, N, wpr]
    s = v[..., 0]
    for k in range(1, wpr):
        s = s + v[..., k]
    return s.to(torch.bfloat16)


def make_params(dev, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    alpha = (torch.rand(D, generator=g, device=dev) * 0.9 + 0.1).float()
    beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
    post_w = (torch.randn(D, generator=g, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
    pre_w = (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    w4 = (torch.randn(4096, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    return alpha, beta, post_w, pre_w, w4


def make_weight(rank, kind, dev, seed, scale=0.02):
    N, K = GEOM[kind]["N"], GEOM[kind]["K"]
    g = torch.Generator(device=dev)
    g.manual_seed(seed * 7919 + rank * 31 + kind)
    return (torch.randn(N, K, generator=g, device=dev) * scale).to(torch.bfloat16)


def make_x(T, K, dev, seed, data):
    """Replicated activations: the same on every rank (the model's tokens are replicated)."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    x = torch.randn(T, K, generator=g, device=dev)
    if data == "mixed":
        e = torch.randint(-6, 7, (K,), device=dev, generator=g).float()
        x = x * torch.exp2(e)
        m = torch.rand(T, K, device=dev, generator=g) < 1e-3
        x = torch.where(m, x * 64.0, x)
    elif data == "zeros":
        s = torch.rand(T, K, device=dev, generator=g)
        x = torch.where(s < 0.5, torch.zeros_like(x), torch.where(s < 0.75, -torch.zeros_like(x), x))
    return x.to(torch.bfloat16)


@triton.jit
def _sentinel_copy(src, dst, n, WAIT: tl.constexpr, BLOCK: tl.constexpr):
    """A programmatic dependent that copies ``src`` to ``dst``: with WAIT it calls griddepcontrol.wait first (the contract of every
    kernel behind a topo all-reduce that carries a trigger stage), without it it reads as soon as it is resident."""
    if WAIT:
        tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


def sentinel(src, wait):
    """Launch _sentinel_copy as a programmatic dependent of the kernel in front of it; returns the copy."""
    dst = torch.empty_like(src)
    n = src.numel()
    _sentinel_copy[(triton.cdiv(n, 1024),)](src.view(-1), dst.view(-1), n, WAIT=wait, BLOCK=1024, launch_pdl=True)
    return dst


def run(rank, args):
    dev, cpu_group, _ = init_rank(rank, args.port, nccl=False)
    from muse_topo import TopoComm
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd, fused_rmsnorm_fwd

    comm = TopoComm(cpu_group, dev, rank, WORLD)
    log(rank, "TopoComm up:", comm.info.summary, "gemv", comm.gemv, "push", comm.gemv_push)
    assert comm.gemv, "MUSE_TOPO_GEMV=0 in the environment"
    default_variant = comm.gemv_variant
    default_push = comm.gemv_push
    default_trigger = comm.gemv_trigger
    default_max_rows = comm.gemv_max_rows

    def wpr(kind, rows=1):
        return comm.gemv_wpr(kind, rows)

    Ts = list(range(1, 9)) if not args.quick else [1, 3, 8]
    alpha, beta, post_w, pre_w, w4 = make_params(dev, 5)
    wp = make_weight(rank, 0, dev, 1)
    wu = make_weight(rank, 1, dev, 2)
    weights = {0: wp, 1: wu}

    # ---- 0. the dispatch gates at the default MAX_ROWS (the sections below run the kernels with up to 8 rows) -----------
    if "MUSE_TOPO_GEMV_MAX_ROWS" not in os.environ:
        check(rank, "default MUSE_TOPO_GEMV_MAX_ROWS is 4", TopoComm.GEMV_MAX_ROWS == 4 and default_max_rows == 4)
    stream0_ = (torch.randn(8, D, device=dev)).float()
    for T in range(0, 10):
        want = 1 <= T <= comm.gemv_max_rows
        got_rows = [comm.supports_boundary_rows(kind, T) for kind in comm.GEMV_KINDS]
        check(rank, f"supports_boundary_rows(gemv kinds) T={T} at MAX_ROWS={comm.gemv_max_rows}", got_rows == [want, want])
        if 1 <= T <= 8:
            xs = make_x(T, 8192, dev, 20 + T, "randn")
            xu_ = make_x(T, 4096, dev, 30 + T, "randn")
            check(rank, f"supports_gemv_all_gather_rmsnorm T={T}", comm.supports_gemv_all_gather_rmsnorm(xs, wp) == want)
            check(rank, f"supports_gemv_all_gather_boundary T={T}",
                  comm.supports_gemv_all_gather_boundary(xu_, wu, stream0_[:T].contiguous()) == want)
    comm.gemv = False  # MUSE_TOPO_GEMV=0: every GEMV kind declines
    check(rank, "MUSE_TOPO_GEMV=0 declines the GEMV kinds and the supports_gemv_* checks",
          not any(comm.supports_boundary_rows(kind, 1) for kind in comm.GEMV_KINDS)
          and not comm.supports_gemv_all_gather_rmsnorm(make_x(1, 8192, dev, 1, "randn"), wp)
          and not comm.supports_gemv_all_gather_boundary(make_x(1, 4096, dev, 1, "randn"), wu, stream0_[:1].contiguous())
          and comm.supports_boundary_rows("all_gather_rmsnorm", 1))
    comm.gemv = True
    comm.gemv_max_rows = 8

    # ---- 1. the GEMV shard == the emulation of its summation order, for every geometry variant ------------------------------
    for variant in range(4):
        comm.set_gemv_variant(variant)
        for kind in (0, 1):
            K, N = GEOM[kind]["K"], GEOM[kind]["N"]
            for data in ("randn", "mixed", "zeros"):
                for T in Ts:
                    x = make_x(T, K, dev, 100 + T, data)
                    ref = emulate_gemv(x, weights[kind], wpr(kind))
                    # The GEMV kernel alone (without pushes: those belong to a collective call and are consumed by its second
                    # kernel); it leaves the call counter alone.
                    e0 = comm._c.get_epoch()
                    shard = torch.empty(T, N, dtype=torch.bfloat16, device=dev)
                    comm._c.gemv_push(x, weights[kind], shard, kind, 0)
                    torch.cuda.synchronize()
                    check(rank, f"gemv shard == emulation variant={variant} kind={kind} {data} T={T}", bits_equal(shard, ref))
                    check(rank, f"gemv alone leaves the call counter variant={variant} kind={kind} {data} T={T}",
                          comm._c.get_epoch() == e0)
                    # the same row at another batch size gives the same bits (T-invariance of the order)
                    if T > 1:
                        shard1 = torch.empty(1, N, dtype=torch.bfloat16, device=dev)
                        comm._c.gemv_push(x[:1].contiguous(), weights[kind], shard1, kind, 0)
                        torch.cuda.synchronize()
                        check(rank, f"gemv row 0 independent of T variant={variant} kind={kind} {data} T={T}",
                              bits_equal(shard1, ref[:1]))
    comm.set_gemv_variant(0)

    # ---- 2. fused calls == all_gather of the emulated shards + the model's kernels ------------------------------------
    for variant in ([0, 3] if args.quick else [0, 1, 2, 3]):
        comm.set_gemv_variant(variant)
        cfgs = [(1e-8, 1e-6, 1.0, 1.0), (1e-6, 1e-6, 0.0, 0.0)] if not args.quick else [(1e-8, 1e-6, 1.0, 1.0)]
        for push in ((1, 2, 0, 4, 5, 6) if variant == 0 else (1, 2)):
            comm.gemv_push = push
            for data in ("mixed", "randn"):
                for T in Ts:
                    # pooler + pre-expert norm
                    x = make_x(T, 8192, dev, 200 + T, data)
                    shard = emulate_gemv(x, wp, wpr(0))
                    y = comm.all_gather(shard, -1)
                    for eps, gain in ((1e-6, 1.0), (1e-8, 0.0)):
                        assert comm.supports_gemv_all_gather_rmsnorm(x, wp)
                        ref = fused_rmsnorm_fwd(y, w4, eps, gain_center=gain, input_scale=False, use_kahan_summation=True)
                        got = comm.gemv_all_gather_rmsnorm(x, wp, w4, eps, gain)
                        torch.cuda.synchronize()
                        check(rank, f"gemv+ag+rmsnorm == unfused variant={variant} push={push} {data} T={T} eps={eps} gain={gain}",
                              bits_equal(got, ref) and all_equal_across_ranks(got, cpu_group))
                    # unpooler + boundary
                    xu = make_x(T, 4096, dev, 300 + T, data)
                    shard = emulate_gemv(xu, wu, wpr(1))
                    yu = comm.all_gather(shard, -1)
                    g = torch.Generator(device=dev)
                    g.manual_seed(950 + T)
                    stream = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
                    for cfg in cfgs:
                        for post_scale in (False, True):
                            for fp32 in (False, True):
                                assert comm.supports_gemv_all_gather_boundary(xu, wu, stream)
                                post_eps, pre_eps, post_gain, pre_gain = cfg
                                f32 = torch.empty_like(stream) if fp32 else None
                                ro, rn = fused_norm_residual_norm_fwd(
                                    yu, stream, post_w if post_scale else None, pre_w, post_eps, pre_eps, alpha=alpha, beta=beta,
                                    post_gain_center=post_gain if post_scale else 0.0, pre_gain_center=pre_gain,
                                    post_input_scale=post_scale, pre_input_scale=False, out_dtype=torch.bfloat16, out_fp32=f32)
                                f32b = torch.empty_like(stream) if fp32 else None
                                fo, fn = comm.gemv_all_gather_boundary(
                                    xu, wu, stream, alpha, beta, pre_w, post_w if post_scale else None, post_eps, pre_eps,
                                    post_gain if post_scale else 0.0, pre_gain, out_fp32=f32b)
                                torch.cuda.synchronize()
                                ok = bits_equal(fo, ro) and bits_equal(fn, rn) and (not fp32 or bits_equal(f32b, f32))
                                same = all_equal_across_ranks(fo, cpu_group) and all_equal_across_ranks(fn, cpu_group)
                                check(rank, f"gemv+ag+boundary == unfused variant={variant} push={push} {data} T={T} post_scale={post_scale} "
                                            f"fp32={fp32} cfg={cfg[:2]}", ok and same)
        comm.gemv_push = default_push
    comm.set_gemv_variant(0)
    comm.gemv_push = default_push

    # ---- 3. accuracy of the order against the exact product (and next to cuBLAS) ----------------------------------------
    import torch.nn.functional as F

    for kind in (0, 1):
        K = GEOM[kind]["K"]
        x = make_x(8, K, dev, 400 + kind, "mixed")
        exact = x.double() @ weights[kind].double().t()
        mine = emulate_gemv(x, weights[kind], wpr(kind))
        cub = F.linear(x, weights[kind])
        st_m = err_stats(mine, exact)
        st_c = err_stats(cub, exact)
        log(rank, f"accuracy kind={kind}: gemv.cuh {st_m} | cuBLAS {st_c}")
        # fp32 accumulation rounds a handful of elements (the exact value close to a bf16 rounding boundary) the wrong way: the order is
        # not worse than cuBLAS's if the mean error is no larger and (almost) every element is the correctly rounded value
        check(rank, f"accuracy kind={kind}: mean {st_m['mean_ulp']} ulp (cuBLAS {st_c['mean_ulp']}), "
                    f"{st_m['correctly_rounded_pct']} % correctly rounded (cuBLAS {st_c['correctly_rounded_pct']} %), max {st_m['max_ulp']} ulp",
              st_m["mean_ulp"] <= st_c["mean_ulp"] * 1.05 + 1e-3 and st_m["correctly_rounded_pct"] >= 99.9
              and st_m["max_ulp"] <= 1.5)

    # ---- 3b. push modes that would hang are rejected, in Python and in the C++ entry points (before anything is launched) ---
    xk = make_x(2, 8192, dev, 11, "randn")
    xuk = make_x(2, 4096, dev, 12, "randn")
    stk = (torch.randn(2, D, device=dev, generator=torch.Generator(device=dev).manual_seed(13)) * 2).float()
    shard_p, shard_u = torch.empty(2, 512, dtype=torch.bfloat16, device=dev), torch.empty(2, 1024, dtype=torch.bfloat16, device=dev)
    out_p = torch.empty(2, 4096, dtype=torch.bfloat16, device=dev)
    out_u, new_u = torch.empty(2, D, dtype=torch.bfloat16, device=dev), torch.empty(2, D, device=dev)
    e0 = comm._c.get_epoch()
    for bad in (3, 7, -1, 8, 12):
        try:
            comm.gemv_push = bad
            raised = False
        except ValueError:
            raised = True
        check(rank, f"TopoComm.gemv_push = {bad} raises ValueError and keeps the mode", raised and comm.gemv_push == default_push)
        calls = {
            "gemv_push": lambda: comm._c.gemv_push(xk, wp, shard_p, 0, bad),
            "gemv_all_gather_rmsnorm": lambda: comm._c.gemv_all_gather_rmsnorm(xk, wp, shard_p, out_p, w4, 1e-6, 1.0, bad, 1),
            "gemv_all_gather_boundary": lambda: comm._c.gemv_all_gather_boundary(
                xuk, wu, shard_u, stk, new_u, out_u, None, alpha, beta, None, pre_w, 1e-8, 1e-6, 0.0, 1.0, bad, 1),
        }
        for name, call in calls.items():
            try:
                call()
                msg = "no exception"
            except RuntimeError as e:
                msg = str(e)
            check(rank, f"C++ {name} rejects push mode {bad}", "bad push mode" in msg)
    torch.cuda.synchronize()
    check(rank, "rejected push modes launch nothing (call counter unchanged)", comm._c.get_epoch() == e0)
    for good in (0, 1, 2, 4, 5, 6):
        comm.gemv_push = good
    comm.gemv_push = default_push

    # ---- 3c. the gather's operands are validated before the GEMV kernel pushes: a call that throws leaves no words in the peers' slots -
    # A bad pre-expert norm weight (fp32) / a bad out_fp32 (bf16) on every rank: the call raises. If the GEMV kernel had been launched
    # before the check, its words (tagged with the epoch of the call that never happened) would sit in the peers' slots and the next call
    # at the same epoch would consume them. The next call uses other data and the ranks start it skewed in time.
    bad_w = torch.ones(4096, dtype=torch.float32, device=dev)
    bad_f32 = torch.empty(2, D, dtype=torch.bfloat16, device=dev)
    ok_all = True
    raised_all = True
    for it in range(12 if not args.quick else 4):
        for push in (1, 2):
            comm.gemv_push = push
            e0 = comm._c.get_epoch()
            try:
                comm._c.gemv_all_gather_rmsnorm(xk, wp, shard_p, out_p, bad_w, 1e-6, 1.0, push, 1)
                raised_all = False
            except RuntimeError:
                pass
            try:
                comm._c.gemv_all_gather_boundary(xuk, wu, shard_u, stk, new_u, out_u, bad_f32, alpha, beta, None, pre_w, 1e-8, 1e-6,
                                                 0.0, 1.0, push, 1)
                raised_all = False
            except RuntimeError:
                pass
            ok_all &= comm._c.get_epoch() == e0
            xg = make_x(2, 8192, dev, 3000 + 10 * it + push, "mixed")
            xug = make_x(2, 4096, dev, 3100 + 10 * it + push, "mixed")
            torch.cuda._sleep(spin_cycles_for_us(300 * rank))
            got = comm.gemv_all_gather_rmsnorm(xg, wp, w4, 1e-6, 1.0)
            fo, fn = comm.gemv_all_gather_boundary(xug, wu, stk, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
            ref = fused_rmsnorm_fwd(comm.all_gather(emulate_gemv(xg, wp, wpr(0, 2)), -1), w4, 1e-6, gain_center=1.0, input_scale=False,
                                    use_kahan_summation=True)
            ro, rn = fused_norm_residual_norm_fwd(comm.all_gather(emulate_gemv(xug, wu, wpr(1, 2)), -1), stk, post_w, pre_w, 1e-8, 1e-6,
                                                  alpha=alpha, beta=beta, post_gain_center=1.0, pre_gain_center=1.0,
                                                  post_input_scale=True, pre_input_scale=False, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            ok_all &= bits_equal(got, ref) and bits_equal(fo, ro) and bits_equal(fn, rn)
    comm.gemv_push = default_push
    check(rank, "a rejected GEMV call raises on every rank and leaves the next call (skewed ranks, other data) correct",
          raised_all and ok_all)

    # ---- 4. CUDA graphs next to other collectives: counter / parity / early start behind the all-reduce ------------------
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd  # noqa: F811

    # The same chain with every trigger stage of the all-reduce launches in front of the GEMV kernel (0: never, 1: at their start, 2 / 3: when
    # their last phase begins / ends): when the GEMV kernel becomes resident changes no bit. The stage is a property of the launch
    # (TopoComm._trigger_stage): the all-reduce of the MoE latent [T, 4096] and all_reduce_boundary carry it, here at every T up to 8.
    # Without the bf16 exchange (the test environment has no host.env) and with it (the production setting: the [8, 4096] all-reduce of
    # the padded T = 8 batch, 64 KiB, is the BX kernel with 8 CTAs of 128 threads in front of the GEMV).
    stages = sorted({default_trigger, 2, 3} if args.quick else {0, 1, 2, 3, default_trigger})
    for bf16x in (False, True):
        comm.bf16_exchange = bf16x
        for stage in stages:
            comm.gemv_trigger = stage
            for T in ([1, 2, 4, 8] if not args.quick else [1, 8]):
                xin = make_x(T, D, dev, 500 + T, "mixed")
                g = torch.Generator(device=dev)
                g.manual_seed(960 + T)
                stream0 = (torch.randn(T, D, generator=g, device=dev) * 2.0).float()
                xr = make_input(rank, (T, D), 510 + T, dev, torch.bfloat16, "mixed")
                mid = make_input(rank, (T, 4096), 520 + T, dev, torch.bfloat16, "mixed")

                def layer(use_fused):
                    """A decode-layer-like chain: attn-out all-reduce + boundary, pooler, moe all-reduce, unpooler + boundary."""
                    outs = []
                    y, s1 = comm.all_reduce_boundary(xr, stream0, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
                    xp = y
                    if use_fused:
                        pooled = comm.gemv_all_gather_rmsnorm(xp, wp, w4, 1e-6, 1.0)
                    else:
                        shard = emulate_gemv(xp, wp, wpr(0))
                        pooled = fused_rmsnorm_fwd(comm.all_gather(shard, -1), w4, 1e-6, gain_center=1.0, input_scale=False,
                                                   use_kahan_summation=True)
                    ar = comm.all_reduce(pooled + mid)
                    if use_fused:
                        o, ns = comm.gemv_all_gather_boundary(ar, wu, s1, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
                    else:
                        shard = emulate_gemv(ar, wu, wpr(1))
                        yu = comm.all_gather(shard, -1)
                        o, ns = fused_norm_residual_norm_fwd(yu, s1, post_w, pre_w, 1e-8, 1e-6, alpha=alpha, beta=beta,
                                                             post_gain_center=1.0, pre_gain_center=1.0, post_input_scale=True,
                                                             pre_input_scale=False, out_dtype=torch.bfloat16)
                    outs += [pooled, ar, o, ns]
                    return outs

                tag = f"T={T} trigger={stage} bf16x={bf16x}"
                ref = layer(False)
                torch.cuda.synchronize()
                eager = layer(True)
                torch.cuda.synchronize()
                check(rank, f"layer chain eager == unfused {tag}", all(bits_equal(a, b) for a, b in zip(eager, ref)))
                n_it = 6
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    captured = [layer(True) for _ in range(n_it)]
                for rep in range(5):
                    gr.replay()
                    torch.cuda.synchronize()
                    ok = all(bits_equal(a, b) for c in captured for a, b in zip(c, ref))
                    same = all(all_equal_across_ranks(t, cpu_group) for t in captured[-1])
                    check(rank, f"layer chain graph replay {rep} == unfused {tag}", ok and same)
    comm.bf16_exchange = False
    comm.gemv_trigger = default_trigger

    # ---- 4b. the trigger stage is a property of the launch ---------------------------------------------------------------------
    # A kernel launched with programmatic stream serialization behind a topo all-reduce. With ``wait`` (griddepcontrol.wait, the contract
    # of such a kernel) it must see the complete output behind every all-reduce at every stage (the chain above checks it for the GEMV
    # kernels, this for an arbitrary successor, and for the all-reduce that is NOT followed by a GEMV). Without the wait it may read
    # early behind a launch that carries a stage, but never behind a launch that does not: the stage 1 (kernel start) all-reduce of an
    # unrelated shape must keep the implicit trigger at its exit, with the ranks started skewed in time so that the output arrives late.
    # The output buffer is poisoned with NaN, so that an early read cannot be mistaken for the result.
    def ar_with_sentinel(x, wait, skew):
        out = torch.full_like(x, float("nan"))
        torch.cuda._sleep(spin_cycles_for_us(skew * rank))
        res = comm.all_reduce(x, out=out)
        return sentinel(res, wait), res

    for bf16x in (False, True):
        comm.bf16_exchange = bf16x
        for stage in (1, 2, 3):
            comm.gemv_trigger = stage
            for shape, armed in (((4, 4096), True), ((4, 8192), False), ((16, 4096), False), ((4, 2048), False), ((8, 4096), True)):
                if shape[0] > comm.gemv_max_rows and armed:
                    continue
                x = make_input(rank, shape, 4000 + shape[0], dev, torch.bfloat16, "mixed")
                ref = comm.all_reduce(x)
                torch.cuda.synchronize()
                stale_wait = stale_nowait = 0
                n_rep = 6 if args.quick else 16
                for it in range(n_rep):
                    seen, res = ar_with_sentinel(x, True, 60)
                    torch.cuda.synchronize()
                    stale_wait += not bits_equal(seen, ref)
                    seen, res = ar_with_sentinel(x, False, 60)
                    torch.cuda.synchronize()
                    stale_nowait += not bits_equal(seen, ref)
                check(rank, f"sentinel with griddepcontrol.wait sees the complete all-reduce {shape} stage={stage} bf16x={bf16x}",
                      stale_wait == 0)
                if not armed:
                    check(rank, f"no stage on the all-reduce {shape} (not followed by a GEMV): a sentinel without wait never reads early, "
                                f"stage={stage} bf16x={bf16x}", stale_nowait == 0)
                else:
                    log(rank, f"info: armed all-reduce {shape} stage={stage} bf16x={bf16x}: sentinel without wait read early "
                              f"{stale_nowait}/{n_rep} times (allowed: the successor has to wait)")
    comm.bf16_exchange = False
    comm.gemv_trigger = default_trigger
    # the gate itself (rank-uniform, shape / dtype / rows only)
    comm.gemv_trigger = 3
    check(rank, "trigger stage gate: the unpooler input shape and the rows <= MAX_ROWS only",
          comm._trigger_stage(4) == 3 and comm._trigger_stage(8) == 3 and comm._trigger_stage(9) == 0 and comm._trigger_stage(0) == 0
          and comm._unpooler_input(torch.empty(4, 4096, dtype=torch.bfloat16, device=dev))
          and not comm._unpooler_input(torch.empty(4, 8192, dtype=torch.bfloat16, device=dev))
          and not comm._unpooler_input(torch.empty(4, 4096, dtype=torch.float32, device=dev))
          and not comm._unpooler_input(torch.empty(4096, dtype=torch.bfloat16, device=dev)))
    comm.gemv = False
    check(rank, "trigger stage gate: none without MUSE_TOPO_GEMV", comm._trigger_stage(4) == 0)
    comm.gemv = True
    comm.gemv_max_rows = 4
    check(rank, "trigger stage gate follows MAX_ROWS", comm._trigger_stage(4) == 3 and comm._trigger_stage(5) == 0)
    comm.gemv_max_rows = 8
    comm.gemv_trigger = default_trigger

    # ---- 5. misaligned activations on odd ranks only; random mixes with other collectives -----------------------------------
    T = 1
    buf = torch.empty(8192 + 8, dtype=torch.bfloat16, device=dev)
    xm = make_x(T, 8192, dev, 600, "mixed")
    buf[(4 if rank % 2 else 0):(4 if rank % 2 else 0) + 8192] = xm.view(-1)
    xv = buf[(4 if rank % 2 else 0):(4 if rank % 2 else 0) + 8192].view(1, 8192)
    got = comm.gemv_all_gather_rmsnorm(xv, wp, w4, 1e-6, 1.0)
    ref = fused_rmsnorm_fwd(comm.all_gather(emulate_gemv(xm, wp, wpr(0)), -1), w4, 1e-6, gain_center=1.0, input_scale=False,
                            use_kahan_summation=True)
    torch.cuda.synchronize()
    check(rank, "misaligned activations on odd ranks", bits_equal(got, ref) and all_equal_across_ranks(got, cpu_group))

    # The static operands (weights, norm weights, boundary parameters) misaligned on odd ranks: the call stages aligned copies and launches
    # its GEMV kernel without programmatic serialization (the kernels read these before griddepcontrol.wait, a copy made just in front of
    # the call is not final then). Many iterations behind an all-reduce that carries a trigger stage, ranks skewed in time.
    def misaligned(t, odd):
        """A copy of ``t`` whose data pointer is 8 bytes off a 16-byte boundary when ``odd`` (same values, contiguous)."""
        if not odd:
            return t
        pad = 8 // t.element_size()
        buf = torch.empty(t.numel() + pad, dtype=t.dtype, device=t.device)
        buf[pad:] = t.reshape(-1)
        return buf[pad:].view(t.shape)

    odd = rank % 2 == 1
    comm.gemv_trigger = 3
    ok_all = True
    saw_misaligned = True
    for it in range(8 if args.quick else 40):
        T = 1 + it % 4
        xm_ = make_x(T, 8192, dev, 650 + it, "mixed")
        xu_ = make_x(T, 4096, dev, 660 + it, "mixed")
        st_ = (torch.randn(T, D, device=dev, generator=torch.Generator(device=dev).manual_seed(it)) * 2).float()
        wp_m, w4_m, wu_m = misaligned(wp, odd), misaligned(w4, odd), misaligned(wu, odd)
        al_m, be_m, pw_m, po_m = misaligned(alpha, odd), misaligned(beta, odd), misaligned(pre_w, odd), misaligned(post_w, odd)
        saw_misaligned &= (not odd) or all(t.data_ptr() % 16 != 0 for t in (wp_m, w4_m, wu_m, al_m, be_m, pw_m, po_m))
        ref_p = fused_rmsnorm_fwd(comm.all_gather(emulate_gemv(xm_, wp, wpr(0)), -1), w4, 1e-6, gain_center=1.0, input_scale=False,
                                  use_kahan_summation=True)
        ref_o, ref_n = fused_norm_residual_norm_fwd(comm.all_gather(emulate_gemv(xu_, wu, wpr(1)), -1), st_, post_w, pre_w, 1e-8, 1e-6,
                                                    alpha=alpha, beta=beta, post_gain_center=1.0, pre_gain_center=1.0,
                                                    post_input_scale=True, pre_input_scale=False, out_dtype=torch.bfloat16)
        xr_ = make_input(rank, (T, 4096), 670 + it, dev, torch.bfloat16, "mixed")
        torch.cuda._sleep(spin_cycles_for_us(40 * rank))
        comm.all_reduce(xr_)  # an all-reduce that carries the trigger stage right in front of the GEMV call
        got = comm.gemv_all_gather_rmsnorm(xm_, wp_m, w4_m, 1e-6, 1.0)
        comm.all_reduce(xr_)
        fo, fn = comm.gemv_all_gather_boundary(xu_, wu_m, st_, al_m, be_m, pw_m, po_m, 1e-8, 1e-6, 1.0, 1.0)
        torch.cuda.synchronize()
        ok_all &= bits_equal(got, ref_p) and bits_equal(fo, ref_o) and bits_equal(fn, ref_n)
        ok_all &= all_equal_across_ranks(got, cpu_group) and all_equal_across_ranks(fo, cpu_group)
    comm.gemv_trigger = default_trigger
    check(rank, "misaligned weights / norm weights / boundary parameters on odd ranks (staged, GEMV launched without PDL)",
          ok_all and saw_misaligned)

    import random

    rnd = random.Random(1234)  # the same sequence on every rank
    ok_all = True
    for it in range(args.mix if not args.quick else 40):
        T = rnd.randint(1, 8)
        op = rnd.choice(["pool", "unpool", "ar", "ag", "arb", "pool", "unpool"])
        if op == "pool":
            x = make_x(T, 8192, dev, 700 + it, "mixed")
            got = comm.gemv_all_gather_rmsnorm(x, wp, w4, 1e-6, 1.0)
            ref = fused_rmsnorm_fwd(comm.all_gather(emulate_gemv(x, wp, wpr(0)), -1), w4, 1e-6, gain_center=1.0,
                                    input_scale=False, use_kahan_summation=True)
            ok_all &= bits_equal(got, ref)
        elif op == "unpool":
            x = make_x(T, 4096, dev, 800 + it, "mixed")
            st = (torch.randn(T, D, device=dev, generator=torch.Generator(device=dev).manual_seed(it)) * 2).float()
            fo, fn = comm.gemv_all_gather_boundary(x, wu, st, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
            ro, rn = fused_norm_residual_norm_fwd(comm.all_gather(emulate_gemv(x, wu, wpr(1)), -1), st, post_w, pre_w, 1e-8, 1e-6,
                                                  alpha=alpha, beta=beta, post_gain_center=1.0, pre_gain_center=1.0,
                                                  post_input_scale=True, pre_input_scale=False, out_dtype=torch.bfloat16)
            ok_all &= bits_equal(fo, ro) and bits_equal(fn, rn)
        elif op == "ar":
            xr2 = make_input(rank, (T, D), 900 + it, dev, torch.bfloat16, "randn")
            ok_all &= bits_equal(comm.all_reduce(xr2), comm.all_reduce(xr2))
        elif op == "ag":
            sh = make_input(rank, (T, 1024), 910 + it, dev, torch.bfloat16, "randn")
            ok_all &= bits_equal(comm.all_gather(sh, -1)[:, rank * 1024:(rank + 1) * 1024], sh)
        else:
            xr2 = make_input(rank, (T, D), 920 + it, dev, torch.bfloat16, "randn")
            st = (torch.randn(T, D, device=dev, generator=torch.Generator(device=dev).manual_seed(it)) * 2).float()
            fo, fn = comm.all_reduce_boundary(xr2, st, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
            y2 = comm.all_reduce(xr2)
            ro, rn = fused_norm_residual_norm_fwd(y2, st, post_w, pre_w, 1e-8, 1e-6, alpha=alpha, beta=beta,
                                                  post_gain_center=1.0, pre_gain_center=1.0, post_input_scale=True,
                                                  pre_input_scale=False, out_dtype=torch.bfloat16)
            ok_all &= bits_equal(fo, ro) and bits_equal(fn, rn)
    torch.cuda.synchronize()
    check(rank, "random mix of fused GEMV calls and other collectives (eager)", ok_all)

    # ---- 6. the variant chosen by the number of rows (the default of the plugin) -------------------------------------------
    comm.set_gemv_variant(None)
    for T in Ts:
        x = make_x(T, 8192, dev, 1000 + T, "mixed")
        got = comm.gemv_all_gather_rmsnorm(x, wp, w4, 1e-6, 1.0)
        ref = fused_rmsnorm_fwd(comm.all_gather(emulate_gemv(x, wp, wpr(0, T)), -1), w4, 1e-6, gain_center=1.0, input_scale=False,
                                use_kahan_summation=True)
        xu = make_x(T, 4096, dev, 1100 + T, "mixed")
        st = (torch.randn(T, D, device=dev, generator=torch.Generator(device=dev).manual_seed(T)) * 2).float()
        fo, fn = comm.gemv_all_gather_boundary(xu, wu, st, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
        ro, rn = fused_norm_residual_norm_fwd(comm.all_gather(emulate_gemv(xu, wu, wpr(1, T)), -1), st, post_w, pre_w, 1e-8, 1e-6,
                                              alpha=alpha, beta=beta, post_gain_center=1.0, pre_gain_center=1.0,
                                              post_input_scale=True, pre_input_scale=False, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        check(rank, f"variant by rows (v{comm._variant_for(T)}) T={T}",
              bits_equal(got, ref) and bits_equal(fo, ro) and bits_equal(fn, rn))
    comm.set_gemv_variant(default_variant)

    # ---- 7. MUSE_TOPO_PDL=1 (every topo kernel is a programmatic dependent and triggers at its start): the gather kernel behind a GEMV kernel
    # with push mode 0 / 4 reads the shard that kernel wrote, behind griddepcontrol.wait; and the pushed modes, next to other collectives
    comm._c.set_pdl(True)
    for push in (0, 1, 2, 4, 5, 6):
        comm.gemv_push = push
        ok_all = True
        for it in range(6 if args.quick else 20):
            T = (1, 2, 4, 8)[it % 4]
            xg = make_x(T, 8192, dev, 5000 + it, "mixed")
            xug = make_x(T, 4096, dev, 5100 + it, "mixed")
            stg = (torch.randn(T, D, device=dev, generator=torch.Generator(device=dev).manual_seed(it)) * 2).float()
            xr_ = make_input(rank, (T, 4096), 5200 + it, dev, torch.bfloat16, "mixed")
            torch.cuda._sleep(spin_cycles_for_us(30 * rank))
            comm.all_reduce(xr_)
            got = comm.gemv_all_gather_rmsnorm(xg, wp, w4, 1e-6, 1.0)
            ar_ = comm.all_reduce(xr_)
            fo, fn = comm.gemv_all_gather_boundary(xug, wu, stg, alpha, beta, pre_w, post_w, 1e-8, 1e-6, 1.0, 1.0)
            ref = fused_rmsnorm_fwd(comm.all_gather(emulate_gemv(xg, wp, comm.gemv_wpr(0, T)), -1), w4, 1e-6, gain_center=1.0,
                                    input_scale=False, use_kahan_summation=True)
            ro, rn = fused_norm_residual_norm_fwd(comm.all_gather(emulate_gemv(xug, wu, comm.gemv_wpr(1, T)), -1), stg, post_w, pre_w,
                                                  1e-8, 1e-6, alpha=alpha, beta=beta, post_gain_center=1.0, pre_gain_center=1.0,
                                                  post_input_scale=True, pre_input_scale=False, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            ok_all &= bits_equal(got, ref) and bits_equal(fo, ro) and bits_equal(fn, rn) and all_equal_across_ranks(ar_, cpu_group)
        check(rank, f"MUSE_TOPO_PDL=1 gemv + gather == unfused, push={push}", ok_all)
    comm._c.set_pdl(False)
    comm.gemv_push = default_push
    comm.gemv_max_rows = default_max_rows
    comm.set_gemv_variant(default_variant)
    comm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mix", type=int, default=300)
    ap.add_argument("--port", type=int, default=29761)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
