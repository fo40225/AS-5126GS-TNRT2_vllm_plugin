"""Probe (research item for algorithm 4, not part of the TopoComm runtime): the cross-socket partner exchange with SM push
versus the copy engine (cudaMemcpyAsync peer copy), measured on the real 8-GPU pattern.

usage: py_topo.sh bench/ce_cut_probe.py --out <jsonl> [--sizes 262144,1048576,2097152,8388608]
Every pair (r, partner) exchanges V bytes in both directions at the same time (the cut hop of the reduce-scatter: 4 flows
per direction, same pattern as the all-reduce). One exchange = data into the partner's IPC buffer + a release flag the
partner waits for (the handoff the pipelined all-reduce needs); the data is verified on the receiving side (not timed).
  push  : SM stores (G blocks x B threads) + per-block release flags, then a wait kernel on the partner's flags
  ce    : cudaMemcpyAsync (peer) of V bytes in K pieces on the current stream, each followed by a one-thread flag kernel
          (stream order guarantees the piece has landed before its flag is released), then the wait kernel
Time per exchange = event-timed eager back-to-back iterations (max over ranks); 'gbps_cut_dir' = 4*V/time = aggregate
over the cut per direction. Eager launches, because the prefill all-reduce is eager.
"""
import argparse
import statistics
import sys
import traceback

import torch
import torch.multiprocessing as mp

from common import *  # noqa: F401,F403

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(int* p, int v) { asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }

__global__ void push_kernel(const uint4* src, uint4* tx, int* tx_flag, int npacks, int tag) {
  const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
  const int cp = (npacks + gridDim.x - 1) / gridDim.x;
  const int lo = min(npacks, b * cp), hi = min(npacks, lo + cp);
  for (int i = lo + tid; i < hi; i += bd) st_v4(tx + i, src[i]);
  __syncthreads();
  if (tid == 0) st_rel(tx_flag + b * 32, tag);
}
__global__ void flag_post_kernel(int* flag, int tag) { st_rel(flag, tag); }
__global__ void wait_kernel(const int* flag, int stride, int tag) {
  if (threadIdx.x == 0) {
    long long n = 0;
    while (ld_acq(flag + blockIdx.x * stride) - tag < 0) {
      if (++n > (1ll << 28)) { printf("wait timeout block %d tag %d\n", blockIdx.x, tag); __trap(); }
    }
  }
}

void push(torch::Tensor src, int64_t tx, int64_t tx_flag, int64_t nbytes, int64_t tag, int64_t grid, int64_t block) {
  push_kernel<<<(unsigned)grid, (unsigned)block, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)src.data_ptr(), (uint4*)tx, (int*)tx_flag, (int)(nbytes / 16), (int)tag);
}
void flag_post(int64_t flag, int64_t tag) {
  flag_post_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((int*)flag, (int)tag);
}
void wait_flags(int64_t flag, int64_t stride_ints, int64_t nblocks, int64_t tag) {
  wait_kernel<<<(unsigned)nblocks, 32, 0, at::cuda::getCurrentCUDAStream()>>>((const int*)flag, (int)stride_ints, (int)tag);
}
int64_t copy_from(torch::Tensor dst, int64_t src, int64_t nbytes) {
  return (int64_t)cudaMemcpyAsync(dst.data_ptr(), (const void*)src, (size_t)nbytes, cudaMemcpyDeviceToDevice,
                                  at::cuda::getCurrentCUDAStream());
}
int64_t memcpy_peer(int64_t dst, torch::Tensor src, int64_t src_off, int64_t nbytes) {
  return (int64_t)cudaMemcpyAsync((void*)dst, (const char*)src.data_ptr() + src_off, (size_t)nbytes, cudaMemcpyDeviceToDevice,
                                  at::cuda::getCurrentCUDAStream());
}
"""
CPP_DECL = """
void push(torch::Tensor src, int64_t tx, int64_t tx_flag, int64_t nbytes, int64_t tag, int64_t grid, int64_t block);
void flag_post(int64_t flag, int64_t tag);
void wait_flags(int64_t flag, int64_t stride_ints, int64_t nblocks, int64_t tag);
int64_t memcpy_peer(int64_t dst, torch::Tensor src, int64_t src_off, int64_t nbytes);
int64_t copy_from(torch::Tensor dst, int64_t src, int64_t nbytes);
"""


def load_ext():
    from torch.utils.cpp_extension import load_inline

    return load_inline(name="ce_cut_probe_ext", cpp_sources=[CPP_DECL], cuda_sources=[CUDA_SRC],
                       functions=["push", "flag_post", "wait_flags", "memcpy_peer", "copy_from"], extra_cuda_cflags=["-O3"], verbose=False)


def worker(rank, args):
    try:
        dev, cpu_group, _ = init_rank(rank, args.port)
        if rank == 0:
            load_ext()
        dist.barrier(cpu_group)
        ext = load_ext()
        from muse_topo import ops
        from muse_topo.collectives import TopoComm

        comm = TopoComm(cpu_group, dev, rank, WORLD, max_bytes=1 << 20)  # small workspace; only the topology is used
        partner = comm.info.partner
        e2 = ops.load()
        sizes = [int(s) for s in args.sizes.split(",")]
        slot = max(sizes)
        flags_off = 2 * slot
        ws = ops.Workspace(e2, dev, flags_off + 64 * 1024)
        bases = ws.exchange(cpu_group, rank, WORLD)
        rx_base, tx_base = bases[rank], bases[partner]
        src = make_input(rank, (slot // 2,), 5, dev)
        want = make_input(partner, (slot // 2,), 5, dev)
        tag = [0]

        def exchange(mode, nbytes, k, grid, block):
            tag[0] += 1
            t = tag[0]
            par = t & 1
            tx = tx_base + par * slot
            if mode == "push":
                ext.push(src, tx, tx_base + flags_off, nbytes, t, grid, block)
                ext.wait_flags(rx_base + flags_off, 32, grid, t)
            else:
                piece = nbytes // k
                for j in range(k):
                    ext.memcpy_peer(tx + j * piece, src, j * piece, piece)
                    ext.flag_post(tx_base + flags_off + j * 128, t)
                for j in range(k):
                    ext.wait_flags(rx_base + flags_off + j * 128, 32, 1, t)
            return par

        def verified(mode, nbytes, k, grid, block):
            """One exchange, then compare the bytes that landed in my receive slot with the partner's input."""
            par = exchange(mode, nbytes, k, grid, block)
            torch.cuda.synchronize()
            got = torch.empty(nbytes, dtype=torch.uint8, device=dev)
            ext.copy_from(got, rx_base + par * slot, nbytes)
            torch.cuda.synchronize()
            return bool(torch.equal(got, want.view(torch.uint8)[:nbytes]))

        log(rank, f"probe up: partner {partner}; sizes {sizes}")
        for nbytes in sizes:
            cfgs = [("push", 1, g, b) for g in (1, 2, 4, 8) for b in (512, 1024)] + [("ce", k, 0, 0) for k in (1, 2, 4, 8)]
            for mode, k, grid, block in cfgs:
                if mode == "ce" and nbytes // k < 65536:
                    continue
                if not all(gather_all(verified(mode, nbytes, k, grid, block), cpu_group)):
                    log(rank, f"INVALID {mode} {nbytes} k={k} {grid}x{block}")
                    continue
                iters = 40 if nbytes <= (2 << 20) else 20
                for _ in range(3):
                    exchange(mode, nbytes, k, grid, block)
                trials = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    dist.barrier(cpu_group)
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(iters):
                        exchange(mode, nbytes, k, grid, block)
                    e.record()
                    e.synchronize()
                    trials.append(s.elapsed_time(e) * 1e3 / iters)
                us = max(gather_all(statistics.median(trials), cpu_group))
                row = {"mode": mode, "bytes_per_direction": nbytes, "pieces": k if mode == "ce" else 1,
                       "grid": grid, "block": block, "us": round(us, 2),
                       "gbps_per_gpu_dir": round(nbytes / us / 1e3, 2), "gbps_cut_dir": round(4 * nbytes / us / 1e3, 2)}
                append_jsonl(args.out, [row], rank)
                log(rank, row)
        dist.barrier(cpu_group)
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
    ap.add_argument("--sizes", default="262144,1048576,2097152,8388608")
    ap.add_argument("--port", type=int, default=29725)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
