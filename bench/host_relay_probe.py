"""Probe (research item, not part of the TopoComm runtime): cross-socket partner exchange through pinned host memory
versus direct P2P SM push.

usage: py_topo.sh bench/host_relay_probe.py --out <jsonl> [--sizes 65536,262144,...]
Pairs (r, r^4) exchange V bytes in both directions at the same time (the cross-island hop of the all-reduce):
  p2p      : SM stores into the partner's IPC device buffer + release flag (what the all-reduce does today)
  host_wr  : SM stores into a pinned shared-memory buffer placed on the WRITER's NUMA node, the partner loads from it
  host_rd  : same, buffer placed on the READER's NUMA node
Pinned buffers are /dev/shm files mapped by both partners, bound with mbind and cudaHostRegister'ed in each process.
Time per exchange = event-timed eager back-to-back launches / iterations (max over ranks). Every config is verified.
"""
import argparse
import ctypes
import json
import mmap
import os
import statistics
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ uint4 ld_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(int* p, int v) { asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }

// symmetric exchange: write src into tx_data (parity slot), release the flag, wait for the partner's flag, read rx_data
__global__ void exch_kernel(const uint4* src, uint4* dst, uint4* tx_data, const uint4* rx_data, int* tx_flag,
                            const int* rx_flag, int npacks, int slot_packs, int tag) {
  const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
  const int cp = (npacks + gridDim.x - 1) / gridDim.x;
  const int lo = min(npacks, b * cp), hi = min(npacks, lo + cp);
  const int64_t off = (int64_t)(tag & 1) * slot_packs;
  for (int i = lo + tid; i < hi; i += bd) st_v4(tx_data + off + i, src[i]);
  __syncthreads();
  if (tid == 0) {
    st_rel(tx_flag + b * 32, tag);
    long long n = 0;
    while (ld_acq(rx_flag + b * 32) - tag < 0) {
      if (++n > (1ll << 27)) { printf("exch timeout block %d tag %d\n", b, tag); __trap(); }
    }
  }
  __syncthreads();
  for (int i = lo + tid; i < hi; i += bd) dst[i] = ld_v4(rx_data + off + i);
}

void exch(torch::Tensor src, torch::Tensor dst, int64_t tx_data, int64_t rx_data, int64_t tx_flag, int64_t rx_flag,
          int64_t nbytes, int64_t slot_bytes, int64_t tag, int64_t grid, int64_t block) {
  exch_kernel<<<(unsigned)grid, (unsigned)block, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), (uint4*)tx_data, (const uint4*)rx_data, (int*)tx_flag,
      (const int*)rx_flag, (int)(nbytes / 16), (int)(slot_bytes / 16), (int)tag);
}

__global__ void flag_post_kernel(int* flag, int tag) { st_rel(flag, tag); }
__global__ void flag_wait_kernel(const int* flag, int tag) {
  long long n = 0;
  while (ld_acq(flag) - tag < 0) {
    if (++n > (1ll << 27)) { printf("flag_wait timeout tag %d\n", tag); __trap(); }
  }
}
void flag_post(int64_t flag, int64_t tag) {
  flag_post_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((int*)flag, (int)tag);
}
void flag_wait(int64_t flag, int64_t tag) {
  flag_wait_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((const int*)flag, (int)tag);
}

int64_t host_register(int64_t ptr, int64_t bytes) {
  return (int64_t)cudaHostRegister((void*)ptr, (size_t)bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
}
void host_unregister(int64_t ptr) { cudaHostUnregister((void*)ptr); }
"""
CPP_DECL = """
void exch(torch::Tensor src, torch::Tensor dst, int64_t tx_data, int64_t rx_data, int64_t tx_flag, int64_t rx_flag, int64_t nbytes, int64_t slot_bytes, int64_t tag, int64_t grid, int64_t block);
int64_t host_register(int64_t ptr, int64_t bytes);
void host_unregister(int64_t ptr);
void flag_post(int64_t flag, int64_t tag);
void flag_wait(int64_t flag, int64_t tag);
"""


def load_ext():
    from torch.utils.cpp_extension import load_inline

    return load_inline(name="host_relay_probe_ext", cpp_sources=[CPP_DECL], cuda_sources=[CUDA_SRC],
                       functions=["exch", "host_register", "host_unregister", "flag_post", "flag_wait"], extra_cuda_cflags=["-O3"], verbose=False)


def numa_of(rank):
    p = torch.cuda.get_device_properties(rank)
    bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    return int(open(f"/sys/bus/pci/devices/{bdf}/numa_node").read())


def addr_of(mm):
    return ctypes.addressof(ctypes.c_char.from_buffer(mm))


def mbind(addr, size, node):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    mask = (ctypes.c_ulong * 2)()
    mask[node // 64] = 1 << (node % 64)
    rc = libc.syscall(237, ctypes.c_void_p(addr), ctypes.c_size_t(size), 2, mask, 128, 0)  # SYS_mbind, MPOL_BIND
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"mbind(node {node}) failed")


def open_shm(path, size, node, create):
    fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o600)
    if create:
        os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    addr = addr_of(mm)
    if create:
        mbind(addr, size, node)
        ctypes.memset(addr, 0, size)
    return mm, addr


def worker(rank, args):
    paths = []
    try:
        dev, cpu_group, _ = init_rank(rank, args.port)
        if rank == 0:
            load_ext()
        dist.barrier(cpu_group)
        ext = load_ext()
        from muse_topo import ops

        e2 = ops.load()
        slot = max(int(s) for s in args.sizes.split(","))
        buf_bytes = 2 * slot + 128 * 1024  # 2 parity slots + per-block flags (128 B apart)
        partner = rank ^ 4
        my_node, partner_node = numa_of(rank), numa_of(partner)
        ws = ops.Workspace(e2, dev, buf_bytes)
        bases = ws.exchange(cpu_group, rank, WORLD)
        modes = {"p2p": dict(tx_data=bases[partner], rx_data=bases[rank], tx_flag=bases[partner] + 2 * slot,
                             rx_flag=bases[rank] + 2 * slot)}
        keep = []
        host = {}

        def host_view_fn(addr, nbytes):
            buf = (ctypes.c_uint8 * nbytes).from_address(addr)
            return torch.frombuffer(buf, dtype=torch.uint8)

        for variant in ("wr", "rd"):
            tx_path = f"/dev/shm/muse_probe_{args.port}_{variant}_w{rank}"
            rx_path = f"/dev/shm/muse_probe_{args.port}_{variant}_w{partner}"
            paths.append(tx_path)
            mm_tx, a_tx = open_shm(tx_path, buf_bytes, {"wr": my_node, "rd": partner_node}[variant], True)
            dist.barrier(cpu_group)
            mm_rx, a_rx = open_shm(rx_path, buf_bytes, 0, False)
            for a in (a_tx, a_rx):
                rc = ext.host_register(a, buf_bytes)
                assert rc == 0, f"cudaHostRegister rc={rc}"
            keep += [mm_tx, mm_rx]
            host[variant] = dict(tx_view=host_view_fn(a_tx, buf_bytes), rx_view=host_view_fn(a_rx, buf_bytes), tx_flag=a_tx + 2 * slot,
                                 rx_flag=a_rx + 2 * slot)
            modes["host_" + variant] = dict(tx_data=a_tx, rx_data=a_rx, tx_flag=a_tx + 2 * slot, rx_flag=a_rx + 2 * slot)
        dist.barrier(cpu_group)
        src = make_input(rank, (slot // 2,), 5, dev)
        dst = torch.empty_like(src)
        want = make_input(partner, (slot // 2,), 5, dev)
        tags = {m: 0 for m in modes}
        # copy-engine host relay: D2H on stream s1 (+ flag post per chunk), flag wait + H2D on stream s2
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        ce_tags = {"wr": 0, "rd": 0}

        def ce_exchange(variant, nbytes, nchunks):
            h = host[variant]
            ce_tags[variant] += 1
            tag = ce_tags[variant]
            par = tag & 1
            chunk = nbytes // nchunks
            cur = torch.cuda.current_stream()
            s1.wait_stream(cur)
            s2.wait_stream(cur)
            srcb, dstb = src.view(torch.uint8), dst.view(torch.uint8)
            with torch.cuda.stream(s1):
                for k in range(nchunks):
                    h["tx_view"][par * slot + k * chunk: par * slot + (k + 1) * chunk].copy_(srcb[k * chunk:(k + 1) * chunk], non_blocking=True)
                    ext.flag_post(h["tx_flag"] + 65536 + k * 128, tag)
            with torch.cuda.stream(s2):
                for k in range(nchunks):
                    ext.flag_wait(h["rx_flag"] + 65536 + k * 128, tag)
                    dstb[k * chunk:(k + 1) * chunk].copy_(h["rx_view"][par * slot + k * chunk: par * slot + (k + 1) * chunk], non_blocking=True)
            cur.wait_stream(s1)
            cur.wait_stream(s2)

        log(rank, f"probe up: numa(rank0)={my_node} numa(partner)={partner_node}; sizes {args.sizes}")
        for nbytes in [int(s) for s in args.sizes.split(",")]:
            for variant in ("wr", "rd"):
                if not args.ce:
                    break
                best = None
                for nchunks in (1, 2, 4, 8):
                    if nbytes // nchunks < 16384:
                        continue
                    dst.zero_()
                    ce_exchange(variant, nbytes, nchunks)
                    torch.cuda.synchronize()
                    ok = bool(torch.equal(dst.view(torch.uint8)[:nbytes], want.view(torch.uint8)[:nbytes]))
                    if not all(gather_all(ok, cpu_group)):
                        log(rank, f"INVALID ce_{variant} {nbytes} x{nchunks}")
                        continue
                    iters = 20
                    for _ in range(2):
                        ce_exchange(variant, nbytes, nchunks)
                    trials = []
                    for _ in range(3):
                        torch.cuda.synchronize()
                        dist.barrier(cpu_group)
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        for _ in range(iters):
                            ce_exchange(variant, nbytes, nchunks)
                        e.record()
                        e.synchronize()
                        trials.append(s.elapsed_time(e) * 1e3 / iters)
                    us = max(gather_all(statistics.median(trials), cpu_group))
                    if best is None or us < best[0]:
                        best = (us, nchunks)
                if best is not None:
                    us, nchunks = best
                    row = {"mode": "ce_host_" + variant, "bytes_per_direction": nbytes, "us": round(us, 2), "chunks": nchunks,
                           "gbps_per_gpu_per_dir": round(nbytes / us / 1e3, 2),
                           "gbps_cut_per_dir_total": round(4 * nbytes / us / 1e3, 2)}
                    append_jsonl(args.out, [row], rank)
                    log(rank, row)
            for mode, m in modes.items():
                best = None
                for grid in [int(g) for g in args.grids.split(",")]:
                    for block in [int(b) for b in args.blocks.split(",")]:
                        def launch():
                            tags[mode] += 1
                            ext.exch(src, dst, m["tx_data"], m["rx_data"], m["tx_flag"], m["rx_flag"], nbytes, slot,
                                     tags[mode], grid, block)

                        dst.zero_()
                        launch()
                        torch.cuda.synchronize()
                        ok = bool(torch.equal(dst.view(torch.uint8)[:nbytes], want.view(torch.uint8)[:nbytes]))
                        oks = all(gather_all(ok, cpu_group))
                        if not oks:
                            log(rank, f"INVALID {mode} {nbytes} {grid}x{block}")
                            continue
                        iters = 60 if nbytes <= (1 << 20) else 25
                        for _ in range(3):
                            launch()
                        trials = []
                        for _ in range(3):
                            torch.cuda.synchronize()
                            dist.barrier(cpu_group)
                            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            s.record()
                            for _ in range(iters):
                                launch()
                            e.record()
                            e.synchronize()
                            trials.append(s.elapsed_time(e) * 1e3 / iters)
                        us = max(gather_all(statistics.median(trials), cpu_group))
                        if best is None or us < best[0]:
                            best = (us, grid, block)
                if best is not None:
                    us, grid, block = best
                    row = {"mode": mode, "bytes_per_direction": nbytes, "us": round(us, 2), "grid": grid, "block": block,
                           "gbps_per_gpu_per_dir": round(nbytes / us / 1e3, 2),
                           "gbps_cut_per_dir_total": round(4 * nbytes / us / 1e3, 2)}
                    append_jsonl(args.out, [row], rank)
                    log(rank, row)
        dist.barrier(cpu_group)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="65536,262144,1048576,4194304,8388608")
    ap.add_argument("--grids", default="1,2,4,8,16,32")
    ap.add_argument("--blocks", default="256,512,1024")
    ap.add_argument("--ce", action="store_true", default=True)
    ap.add_argument("--port", type=int, default=29717)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
