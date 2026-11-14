#!/usr/bin/env python3
"""Where does the +500 MiB of GPU memory of a KV offload connector come from? (S1-10 probe, one GPU, run through runsrc.sh)

Steps, each followed by a reading of the device memory the way the campaign reads it (NVML used MiB via nvidia-smi for this
process's GPU, and torch.cuda.mem_get_info): context creation, a GPU KV-like tensor, cudaHostRegister of N GiB (the Simple
connector's pin_tensor), two low-priority streams + events, a batch memcpy (cuMemcpyBatchAsync, both directions) of a few blocks,
the first large batch, the native connector's Triton swap kernel.
usage: runsrc.sh gpu_overhead_probe.py --host-gib 2 [--gpu-mib 2048] [--blocks 4000]
"""
import argparse
import subprocess
import time

import torch


def nvml_used_mib():
    idx = torch.cuda.current_device()
    # CUDA_VISIBLE_DEVICES maps device 0 to a physical GPU; query by the physical index
    import os
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[idx]
    out = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out)


def rd(label, base):
    torch.cuda.synchronize()
    free, tot = torch.cuda.mem_get_info()
    used = nvml_used_mib()
    print("PROBE %-34s nvml_used=%6d MiB (+%5d)  ctx_free=%7.1f MiB" % (label, used, used - base, free / 2**20), flush=True)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-gib", type=float, default=2.0)
    ap.add_argument("--gpu-mib", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=4000)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()

    from vllm.v1.simple_kv_offload.cuda_mem_ops import build_params, copy_blocks, pin_tensor

    torch.zeros(1, device="cuda")
    base = rd("context created", 0)
    gpu = {"t%d" % i: torch.zeros((a.gpu_mib * 2**20 // 5 // 4096 // 1, 4096), dtype=torch.int8, device="cuda") for i in range(5)}
    n_gpu_blocks = gpu["t0"].shape[0]
    rd("GPU KV-like tensors (%d MiB)" % a.gpu_mib, base)
    n_cpu_blocks = int(a.host_gib * 2**30 // (5 * 4096))
    cpu = {}
    for k, g in gpu.items():
        t = torch.zeros((n_cpu_blocks, 4096), dtype=torch.int8, device="cpu")
        cpu[k] = t
    rd("CPU tensors allocated (not pinned)", base)
    if not a.no_register:
        for t in cpu.values():
            pin_tensor(t)
        rd("cudaHostRegister %.1f GiB" % a.host_gib, base)
    low, _ = torch.cuda.Stream.priority_range()
    ls, ss = torch.cuda.Stream(priority=low), torch.cuda.Stream(priority=low)
    ev = [torch.Event() for _ in range(8)]
    rd("two low-priority streams + events", base)
    sp = build_params(gpu, cpu, ss)
    lp = build_params(cpu, gpu, ls)
    rd("build_params", base)
    ids = list(range(16))
    copy_blocks(ids, ids, sp)
    torch.cuda.synchronize()
    rd("first store batch (16 blocks, 80 desc.)", base)
    copy_blocks(ids, ids, lp)
    torch.cuda.synchronize()
    rd("first load batch (16 blocks)", base)
    ids = list(range(min(a.blocks, n_gpu_blocks)))
    t0 = time.perf_counter()
    copy_blocks(ids, ids, sp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    rd("big store batch (%d blocks)" % len(ids), base)
    t0 = time.perf_counter()
    copy_blocks(ids, ids, lp)
    torch.cuda.synchronize()
    dtl = time.perf_counter() - t0
    mb = len(ids) * 5 * 4096 / 2**20
    print("PROBE bandwidth store %.1f MiB in %.3f s = %.2f GB/s; load %.3f s = %.2f GB/s" % (mb, dt, mb / 1024 / dt * 1.073741824, dtl, mb / 1024 / dtl * 1.073741824), flush=True)
    rd("after big batches", base)
    try:
        from vllm.v1.kv_offload.cpu.swap_blocks_triton import swap_blocks_batch as triton_swap  # noqa: F401
        rd("triton swap module imported", base)
    except Exception as e:  # pragma: no cover
        print("PROBE triton import failed %r" % (e,))


if __name__ == "__main__":
    main()
