"""Bit-exactness proof of muse_topo/csrc/boundary.cuh against the Triton kernels of the Muse Spark model.

The CUDA epilogue math (D = 8192 residual boundary, D = 4096 pre_expert_norm) must reproduce
``fused_norm_residual_norm_fwd`` and ``fused_rmsnorm_fwd`` bit for bit: new stream (fp32), out (bf16), out_fp32 and the
pre_expert_norm output, for every row, at the decode batch sizes, with random / heavy-tailed / real-checkpoint-like data.

usage (server, one GPU, under the GPU lock):
  GPU=0 SRC=<vllm tree> scripts/py_topo.sh bench/test_boundary.py [--rows N] [--ckpt-weights]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils import cpp_extension

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(os.path.dirname(HERE), "muse_topo", "csrc")


def build():
    return cpp_extension.load(
        name="muse_boundary_test", sources=[os.path.join(HERE, "boundary_test.cu")], extra_include_paths=[CSRC],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120f,code=sm_120f", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
        verbose=False)


def bits_equal(a, b):
    a = a.contiguous().view(torch.int32) if a.dtype == torch.float32 else a.contiguous().view(torch.int16)
    b = b.contiguous().view(torch.int32) if b.dtype == torch.float32 else b.contiguous().view(torch.int16)
    return bool((a == b).all())


def make_data(kind, T, D, dev, gen):
    def r(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).to(dtype)

    if kind == "normal":
        branch, stream = r(T, D, dtype=torch.bfloat16, scale=3.0), r(T, D, scale=1.0)
    elif kind == "heavy":  # outlier channels like the residual stream of a real model
        branch = r(T, D, dtype=torch.bfloat16, scale=1.0)
        stream = r(T, D, scale=1.0)
        idx = torch.randint(0, D, (T, 8), generator=gen, device=dev)
        stream.scatter_(1, idx, r(T, 8, scale=300.0))
        branch.scatter_(1, idx, r(T, 8, dtype=torch.bfloat16, scale=500.0))
    elif kind == "tiny":
        branch, stream = r(T, D, dtype=torch.bfloat16, scale=1e-18), r(T, D, scale=1e-30)
    elif kind == "zero":
        branch, stream = torch.zeros(T, D, dtype=torch.bfloat16, device=dev), torch.zeros(T, D, device=dev)
    elif kind == "big":
        branch, stream = r(T, D, dtype=torch.bfloat16, scale=3e4), r(T, D, scale=1e3)
    else:
        raise ValueError(kind)
    return branch.contiguous(), stream.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_norm_residual_norm_fwd, fused_rmsnorm_fwd

    dev = torch.device("cuda")
    ext = build()
    gen = torch.Generator(device=dev)
    fails = 0
    checks = 0
    D = 8192
    for rep in range(args.reps):
        gen.manual_seed(1000 + rep)
        alpha = (torch.rand(D, generator=gen, device=dev) * 0.9 + 0.1).float()
        beta = torch.sqrt(1.0 - alpha.clamp(max=0.99) ** 2).float()
        post_w = (torch.randn(D, generator=gen, device=dev) * 0.5 + 1.0).to(torch.bfloat16)
        pre_w = (torch.randn(D, generator=gen, device=dev) * 0.3).to(torch.bfloat16)
        post_eps, pre_eps = [(1e-8, 1e-6), (1e-6, 1e-6), (1e-5, 1e-8)][rep % 3]
        post_gain, pre_gain = [(1.0, 1.0), (0.0, 0.0), (0.5, 1.0), (-0.25, 0.75)][rep % 4]
        for kind in ("normal", "heavy", "tiny", "zero", "big"):
            for T in (1, 2, 3, 8, 24, args.rows):
                branch, stream = make_data(kind, T, D, dev, gen)
                for post_scale in (False, True):
                    for fp32_out in (False, True):
                        out_f32 = torch.empty(T, D, dtype=torch.float32, device=dev) if fp32_out else None
                        ref_out, ref_new = fused_norm_residual_norm_fwd(
                            branch, stream, post_w if post_scale else None, pre_w, post_eps, pre_eps,
                            alpha=alpha, beta=beta, post_gain_center=post_gain if post_scale else 0.0,
                            pre_gain_center=pre_gain, post_input_scale=post_scale, pre_input_scale=False,
                            out_dtype=torch.bfloat16, out_fp32=out_f32)
                        out, new, out32 = ext.boundary8192(branch, stream, alpha, beta, post_w if post_scale else None,
                                                           pre_w, post_eps, pre_eps, post_gain if post_scale else 0.0,
                                                           pre_gain, fp32_out)
                        ok = bits_equal(out, ref_out) and bits_equal(new, ref_new)
                        if fp32_out:
                            ok = ok and bits_equal(out32, out_f32)
                        checks += 1
                        if not ok:
                            fails += 1
                            nbad = (new.view(torch.int32) != ref_new.view(torch.int32)).sum().item()
                            obad = (out.view(torch.int16) != ref_out.view(torch.int16)).sum().item()
                            print(f"FAIL boundary rep={rep} {kind} T={T} post_scale={post_scale} fp32={fp32_out}: "
                                  f"new mismatches {nbad}, out mismatches {obad}", flush=True)
        # D = 4096 pre_expert_norm
        w4 = (torch.randn(4096, generator=gen, device=dev) * 0.3).to(torch.bfloat16)
        for kind in ("normal", "heavy", "tiny", "zero", "big"):
            for T in (1, 2, 3, 8, 24, args.rows):
                x, _ = make_data(kind, T, 4096, dev, gen)
                ref = fused_rmsnorm_fwd(x, w4, pre_eps, gain_center=pre_gain, input_scale=False, use_kahan_summation=True)
                got = ext.rms4096(x, w4, pre_eps, pre_gain)
                checks += 1
                if not bits_equal(got, ref):
                    fails += 1
                    print(f"FAIL rms4096 rep={rep} {kind} T={T}: {(got.view(torch.int16) != ref.view(torch.int16)).sum().item()}"
                          f" mismatches", flush=True)
        print(f"rep {rep} done: checks={checks} fails={fails}", flush=True)
    print(f"RESULT checks={checks} failures={fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
