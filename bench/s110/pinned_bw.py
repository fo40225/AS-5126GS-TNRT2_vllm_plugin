#!/usr/bin/env python3
"""Pinned host <-> GPU bandwidth of the KV-offload copy paths with the host buffer on a chosen NUMA node.

The buffer is built like the vLLM CPU tier: an unlinked /dev/shm file, mmap'd, pre-faulted with MADV_POPULATE_WRITE while
the memory policy of the process is bound to the target node, then cudaHostRegister'ed (vllm host_register). Placement is
verified from /proc/self/numa_maps. Copies use the same functions as vllm/v1/kv_offload/cpu/gpu_worker.py:
GPU->CPU always ops.swap_blocks_batch (cuMemcpyBatchAsync); CPU->GPU the Triton kernel (SM loads through PCIe) for pages
below 28 KiB, else ops.swap_blocks_batch.

usage (run through runsrc.sh with CUDA_VISIBLE_DEVICES set): pinned_bw.py --gpu-node N --placements local:3,same:2,cross:6
       [--sizes 4096,65536,4194304] [--seconds 3] [--start-at EPOCH] [--tag TEXT] [--out FILE]
Each placement is label:node. One JSON line per (placement, direction, page size) is printed and appended to --out.
"""
import argparse
import ctypes
import json
import mmap
import os
import re
import sys
import time

import torch

from vllm import _custom_ops as ops
from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister
from vllm.v1.kv_offload.cpu.swap_blocks_triton import THRESHOLD_BYTES
from vllm.v1.kv_offload.cpu.swap_blocks_triton import swap_blocks_batch as triton_swap

libnuma = ctypes.CDLL("libnuma.so.1")
libnuma.numa_parse_nodestring.restype = ctypes.c_void_p
libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
libnuma.numa_run_on_node.argtypes = [ctypes.c_int]


def bind_mem(node):
    libnuma.numa_set_membind(libnuma.numa_parse_nodestring(str(node).encode()))


def placement(addr):
    """Pages per node of the numa_maps line that starts at addr."""
    key = "%x" % addr
    with open("/proc/self/numa_maps") as f:
        for line in f:
            if line.startswith(key + " "):
                return {int(k[1:]): int(v) for k, v in (t.split("=") for t in line.split() if re.fullmatch(r"N\d+=\d+", t))}
    return {}


def make_region(size, node, tag):
    path = "/dev/shm/pinned_bw_%d_%s" % (os.getpid(), tag)
    bind_mem(node)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    mm.madvise(23, 0, size)  # MADV_POPULATE_WRITE
    os.unlink(path)
    os.close(fd)
    base = torch.frombuffer(memoryview(mm), dtype=torch.int8)
    ptr = base.data_ptr()
    ok = host_register(ptr, size)
    return mm, base, ptr, ok


def run_case(direction, page, total, gpu_buf, cpu_ptr, seconds, start_at):
    n = total // page
    g = torch.arange(n, dtype=torch.int64) * page + gpu_buf.data_ptr()
    gen = torch.Generator().manual_seed(1234)
    perm = torch.randperm(n, generator=gen)  # scattered CPU slots like block ids
    c = perm.to(torch.int64) * page + cpu_ptr
    sizes = torch.full((n,), page, dtype=torch.int64)
    src, dst = (g, c) if direction == "d2h" else (c, g)
    src, dst, sizes = (t.pin_memory() for t in (src, dst, sizes))
    if direction == "h2d" and page < THRESHOLD_BYTES and page % 8 == 0:
        chunk = min(1 << (page - 1).bit_length(), 8192)
        fn = lambda: triton_swap(src, dst, sizes, bytes_per_chunk=chunk)
        path = "triton"
    else:
        fn = lambda: ops.swap_blocks_batch(src, dst, sizes)
        path = "cuMemcpyBatchAsync"
    fn()
    torch.cuda.synchronize()
    if start_at:
        while time.time() < start_at:
            time.sleep(0.001)
    t0 = time.perf_counter()
    it = 0
    while time.perf_counter() - t0 < seconds:
        fn()
        torch.cuda.synchronize()
        it += 1
    dt = time.perf_counter() - t0
    return {"path": path, "iters": it, "gbps": round(it * total / dt / 1e9, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-node", type=int, required=True, help="NUMA node of this GPU (CPU is pinned there)")
    ap.add_argument("--placements", required=True)
    ap.add_argument("--sizes", default="4096,65536,4194304")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--total-mib", type=int, default=1024)
    ap.add_argument("--start-at", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    libnuma.numa_run_on_node(a.gpu_node)
    dev = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    total = a.total_mib << 20
    torch.cuda.init()
    gpu_buf = torch.empty(total, dtype=torch.int8, device="cuda")
    gpu_buf.fill_(1)
    for spec in a.placements.split(","):
        label, node = spec.split(":")
        node = int(node)
        mm, base, ptr, ok = make_region(total + (64 << 20), node, label)
        hist = placement(ptr)
        pages = sum(hist.values()) or 1
        share = {k: round(100.0 * v / pages, 1) for k, v in sorted(hist.items())}
        for page in [int(s) for s in a.sizes.split(",")]:
            for direction in ("d2h", "h2d"):
                r = run_case(direction, page, total, gpu_buf, ptr, a.seconds, a.start_at)
                rec = {"tag": a.tag, "gpu": dev, "gpu_node": a.gpu_node, "placement": label, "mem_node": node,
                       "pinned": ok, "pages_pct_by_node": share, "page_bytes": page, "dir": direction, **r}
                line = json.dumps(rec)
                print(line, flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(line + "\n")
        if ok:
            host_unregister(ptr)
        del base
        try:
            mm.close()
        except BufferError:
            pass  # exported pointers still alive; the process exits soon anyway


if __name__ == "__main__":
    main()
