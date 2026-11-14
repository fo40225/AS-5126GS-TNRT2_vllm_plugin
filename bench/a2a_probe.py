"""Probe (research item for the large-message algorithms): aggregate egress of ONE GPU when it talks to several peers at
once, with SM stores, SM loads (pull) and the copy engine, in the exact traffic pattern of the large all-reduce.

usage: py_topo.sh bench/a2a_probe.py --out <jsonl> [--sizes 1048576,4194304]
All 8 GPUs run the same pattern at the same time (eager back-to-back launches, event timed, max over ranks). V = bytes per
flow. Flow sets: 'i1' one island peer; 'i3' the three island peers (the phase-a / all-gather pattern); 'c1' the partner
across the socket cut; 'i3c1' the three island peers + the partner. Modes:
  push_vol / push_plain : SM stores (volatile / plain 16 B), `ctas` blocks per flow x `block` threads, no flags
  pull                  : SM loads, the destination GPU reads V bytes from every source of the flow set
  ce                    : cudaMemcpyAsync peer copies, one stream per flow
  push_mix              : like push_vol but the cut flow gets `cut_ctas` blocks and the island flows `ctas` blocks each
Reported: us per iteration, GB/s per flow and total GB/s per GPU (flows * V / us). Data is verified for push and ce.
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

__device__ __forceinline__ void st_vol(void* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 ld_vol(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

struct Flows { char* p[4]; int ctas[4]; int np[4]; int n; };

template <bool VOL>
__global__ void push_kernel(const uint4* __restrict__ src, Flows f, int npacks) {
  int b = blockIdx.x, fl = 0;
  while (fl < f.n - 1 && b >= f.ctas[fl]) { b -= f.ctas[fl]; ++fl; }
  const int nc = f.ctas[fl];
  uint4* tx = reinterpret_cast<uint4*>(f.p[fl]);
  const int npk = f.np[fl] > 0 ? f.np[fl] : npacks;
  const int per = (npk + nc - 1) / nc;
  const int lo = min(npk, b * per), hi = min(npk, lo + per);
  for (int i = lo + threadIdx.x; i < hi; i += 4 * blockDim.x) {
    uint4 x[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) x[q] = (i + q * blockDim.x < hi) ? src[i + q * blockDim.x] : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (i + q * blockDim.x < hi) {
        if (VOL) st_vol(tx + i + q * blockDim.x, x[q]); else tx[i + q * blockDim.x] = x[q];
      }
  }
}

__global__ void pull_kernel(uint4* __restrict__ dst, Flows f, int npacks) {
  int b = blockIdx.x, fl = 0;
  while (fl < f.n - 1 && b >= f.ctas[fl]) { b -= f.ctas[fl]; ++fl; }
  const int nc = f.ctas[fl];
  const uint4* sx = reinterpret_cast<const uint4*>(f.p[fl]);
  uint4* d = dst + (int64_t)fl * npacks;
  const int per = (npacks + nc - 1) / nc;
  const int lo = min(npacks, b * per), hi = min(npacks, lo + per);
  for (int i = lo + threadIdx.x; i < hi; i += 4 * blockDim.x) {
    uint4 x[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) x[q] = (i + q * blockDim.x < hi) ? ld_vol(sx + i + q * blockDim.x) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (i + q * blockDim.x < hi) d[i + q * blockDim.x] = x[q];
  }
}

static Flows make_flows(std::vector<int64_t> ptrs, std::vector<int64_t> ctas, std::vector<int64_t> nbytes = {}) {
  Flows f; f.n = (int)ptrs.size();
  for (int i = 0; i < f.n; ++i) {
    f.p[i] = (char*)ptrs[i]; f.ctas[i] = (int)ctas[i];
    f.np[i] = i < (int)nbytes.size() ? (int)(nbytes[i] / 16) : 0;
  }
  return f;
}

void push(torch::Tensor src, std::vector<int64_t> ptrs, std::vector<int64_t> ctas, int64_t nbytes, int64_t block, bool vol,
          std::vector<int64_t> per_flow_bytes) {
  Flows f = make_flows(ptrs, ctas, per_flow_bytes);
  int total = 0; for (int i = 0; i < f.n; ++i) total += f.ctas[i];
  auto s = at::cuda::getCurrentCUDAStream();
  if (vol) push_kernel<true><<<total, (unsigned)block, 0, s>>>((const uint4*)src.data_ptr(), f, (int)(nbytes / 16));
  else push_kernel<false><<<total, (unsigned)block, 0, s>>>((const uint4*)src.data_ptr(), f, (int)(nbytes / 16));
}
void pull(torch::Tensor dst, std::vector<int64_t> ptrs, std::vector<int64_t> ctas, int64_t nbytes, int64_t block) {
  Flows f = make_flows(ptrs, ctas);
  int total = 0; for (int i = 0; i < f.n; ++i) total += f.ctas[i];
  pull_kernel<<<total, (unsigned)block, 0, at::cuda::getCurrentCUDAStream()>>>((uint4*)dst.data_ptr(), f, (int)(nbytes / 16));
}
int64_t memcpy_to(int64_t dst, torch::Tensor src, int64_t nbytes) {
  return (int64_t)cudaMemcpyAsync((void*)dst, src.data_ptr(), (size_t)nbytes, cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream());
}
int64_t copy_from(torch::Tensor dst, int64_t src, int64_t nbytes) {
  return (int64_t)cudaMemcpyAsync(dst.data_ptr(), (const void*)src, (size_t)nbytes, cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream());
}
"""
CPP_DECL = """
void push(torch::Tensor src, std::vector<int64_t> ptrs, std::vector<int64_t> ctas, int64_t nbytes, int64_t block, bool vol, std::vector<int64_t> per_flow_bytes);
void pull(torch::Tensor dst, std::vector<int64_t> ptrs, std::vector<int64_t> ctas, int64_t nbytes, int64_t block);
int64_t memcpy_to(int64_t dst, torch::Tensor src, int64_t nbytes);
int64_t copy_from(torch::Tensor dst, int64_t src, int64_t nbytes);
"""


def load_ext():
    from torch.utils.cpp_extension import load_inline

    return load_inline(name="a2a_probe_ext", cpp_sources=[CPP_DECL], cuda_sources=[CUDA_SRC],
                       functions=["push", "pull", "memcpy_to", "copy_from"], extra_cuda_cflags=["-O3"], verbose=False)


def worker(rank, args):
    try:
        dev, cpu_group, _ = init_rank(rank, args.port)
        if rank == 0:
            load_ext()
        dist.barrier(cpu_group)
        ext = load_ext()
        from muse_topo import ops

        e2 = ops.load()
        sizes = [int(s) for s in args.sizes.split(",")]
        V = max(sizes)
        # workspace: [8 receive slots of V][my source area V]
        ws = ops.Workspace(e2, dev, 9 * V + 4096)
        bases = ws.exchange(cpu_group, rank, WORLD)
        src = make_input(rank, (V // 2,), 5, dev)
        ws_src = bases[rank] + 8 * V
        # put my source data into the pullable area (the workspace slot every peer reads from in the pull mode)
        island = [rank ^ 1, rank ^ 2, rank ^ 3]
        partner = rank ^ 4
        ext.memcpy_to(ws_src, src, V)
        torch.cuda.synchronize()
        dist.barrier(cpu_group)
        streams = [torch.cuda.Stream() for _ in range(4)]
        dst_pull = torch.empty(4 * V, dtype=torch.uint8, device=dev)

        def flow_set(name):
            peers = {"i1": [island[0]], "i3": island, "c1": [partner], "i3c1": island + [partner]}[name]
            return peers

        def run(mode, fs, nbytes, ctas, block, cut_ctas):
            peers = flow_set(fs)
            cta_list = [cut_ctas if (p == partner and mode == "push_mix") else ctas for p in peers]
            if mode in ("push_vol", "push_plain", "push_mix"):
                ptrs = [bases[p] + rank * V for p in peers]
                ext.push(src, ptrs, cta_list, nbytes, block, mode != "push_plain", [])
            elif mode == "step_mix":  # intra flows V each, cut flow 0.75 V, one block per flow (cut_ctas blocks for the cut)
                ptrs = [bases[p] + rank * V for p in peers]
                cl = [cut_ctas if p == partner else ctas for p in peers]
                pf = [(nbytes * 3 // 4) // 16 * 16 if p == partner else nbytes for p in peers]
                ext.push(src, ptrs, cl, nbytes, block, False, pf)
            elif mode == "pull":
                ptrs = [bases[p] + 8 * V for p in peers]
                ext.pull(dst_pull, ptrs, cta_list, nbytes, block)
            elif mode == "ce_seq":  # one stream, the copies of all peers one after the other
                for p in peers:
                    ext.memcpy_to(bases[p] + rank * V, src, nbytes)
            elif mode == "ce_seq_cut":  # CE (one stream) to the island peers, SM push (1 block) to the partner, concurrently
                cur = torch.cuda.current_stream()
                streams[0].wait_stream(cur)
                with torch.cuda.stream(streams[0]):
                    ext.push(src, [bases[partner] + rank * V], [1], nbytes, 512, True, [])
                for p in island:
                    ext.memcpy_to(bases[p] + rank * V, src, nbytes)
                cur.wait_stream(streams[0])
            else:  # ce
                cur = torch.cuda.current_stream()
                for s in streams:
                    s.wait_stream(cur)
                for k, p in enumerate(peers):
                    with torch.cuda.stream(streams[k]):
                        ext.memcpy_to(bases[p] + rank * V, src, nbytes)
                for s in streams:
                    cur.wait_stream(s)

        def verify(mode, fs, nbytes, ctas, block, cut_ctas):
            peers = flow_set(fs)
            ok = True
            if mode == "pull":
                dst_pull.zero_()
                run(mode, fs, nbytes, ctas, block, cut_ctas)
                torch.cuda.synchronize()
                for k, p in enumerate(peers):
                    want = make_input(p, (V // 2,), 5, dev).view(torch.uint8)[:nbytes]
                    ok &= bool(torch.equal(dst_pull[k * nbytes:(k + 1) * nbytes], want))
                return ok
            run(mode, fs, nbytes, ctas, block, cut_ctas)
            torch.cuda.synchronize()
            dist.barrier(cpu_group)
            got = torch.empty(nbytes, dtype=torch.uint8, device=dev)
            for p in peers:  # data pushed by `p` to me sits in my slot p; only peers that target me are checked
                sender_targets_me = (rank in [p ^ 1, p ^ 2, p ^ 3] and fs in ("i1", "i3", "i3c1")) or (rank == p ^ 4 and fs in ("c1", "i3c1"))
                if fs == "i1":
                    sender_targets_me = rank == (p ^ 1)
                if not sender_targets_me:
                    continue
                nb = (nbytes * 3 // 4) // 16 * 16 if (mode == "step_mix" and p == partner) else nbytes
                ext.copy_from(got, bases[rank] + p * V, nb)
                torch.cuda.synchronize()
                want = make_input(p, (V // 2,), 5, dev).view(torch.uint8)[:nb]
                ok &= bool(torch.equal(got[:nb], want))
            return ok

        log(rank, f"probe up; island peers {island} partner {partner}; sizes {sizes}")
        cfgs = []
        for fs in ("i1", "i3", "c1", "i3c1"):
            for mode in ("push_vol", "push_plain", "pull", "ce", "ce_seq", "ce_seq_cut"):
                if mode == "ce_seq_cut":
                    if fs == "i3c1":
                        cfgs.append((mode, fs, 0, 0, 0))
                    continue
                if mode in ("ce", "ce_seq"):
                    cfgs.append((mode, fs, 0, 0, 0))
                    continue
                for ctas in (1, 2, 4, 8):
                    for block in (512, 1024):
                        cfgs.append((mode, fs, ctas, block, ctas))
        for fs in ("i3c1",):
            for ctas in (2, 4, 8):
                for cut_ctas in (1, 2):
                    cfgs.append(("push_mix", fs, ctas, 512, cut_ctas))
        for fs in ("i3c1",):
            for ctas in (1, 2, 4):
                for cut_ctas in (1, 2):
                    for block in (512, 1024):
                        cfgs.append(("step_mix", fs, ctas, block, cut_ctas))
        for nbytes in sizes:
            for mode, fs, ctas, block, cut_ctas in cfgs:
                if args.only and mode not in args.only.split(","):
                    continue
                if mode == "pull" and ctas * (1 if fs in ("i1", "c1") else 1) < 1:
                    continue
                if not all(gather_all(verify(mode, fs, nbytes, ctas, block, cut_ctas), cpu_group)):
                    log(rank, f"INVALID {mode} {fs} {nbytes} ctas={ctas} block={block}")
                    continue
                iters = 30 if nbytes <= (2 << 20) else 15
                for _ in range(3):
                    run(mode, fs, nbytes, ctas, block, cut_ctas)
                trials = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    dist.barrier(cpu_group)
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(iters):
                        run(mode, fs, nbytes, ctas, block, cut_ctas)
                    e.record()
                    e.synchronize()
                    trials.append(s.elapsed_time(e) * 1e3 / iters)
                us = max(gather_all(statistics.median(trials), cpu_group))
                nfl = len(flow_set(fs)) if mode != "step_mix" else 3.75
                row = {"mode": mode, "flows": fs, "bytes_per_flow": nbytes, "ctas_per_flow": ctas, "cut_ctas": cut_ctas if mode == "push_mix" else None,
                       "block": block, "us": round(us, 2), "gbps_per_flow": round(nbytes / us / 1e3, 2),
                       "gbps_per_gpu": round(nfl * nbytes / us / 1e3, 2)}
                append_jsonl(args.out, [row], rank)
                log(rank, row)
        dist.barrier(cpu_group)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="1048576,4194304")
    ap.add_argument("--only", default="", help="comma list of modes to run")
    ap.add_argument("--port", type=int, default=29727)
    args = ap.parse_args()
    mp.spawn(worker, args=(args,), nprocs=WORLD, join=True)
