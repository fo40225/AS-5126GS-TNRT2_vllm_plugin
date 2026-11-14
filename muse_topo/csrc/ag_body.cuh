// Flag-protocol all-gather body (algorithm 3 flag variant, also the chunk engine of the large all-gather).
//
// The body is a device function over a *virtual* grid (block b of nb) so that it can run as a standalone kernel
// (flag.cu: ag_flag_kernel, large.cu: ag_chunk_kernel) or as the second role of the fused all-reduce step kernel
// (large.cu: step_kernel). Hops, flags and buffers are those of H1a:
//   hop 1: my shard (chunk) to the partner (slow hop first) and to the 3 island peers, release flag per block
//   hop 2: forward the partner's shard to the island peers
//   last : collect the direct shards of the island peers and their forwarded copies.
// Output position of pack i of origin o (16 B units): ag_pos16(o, p0 + i, np_total, rpr, layout), i.e. a chunk
// [p0, p0 + np) of the shard of every rank lands at its place of the full output.
#pragma once

#include "common.cuh"

namespace muse_topo {

// Flag kinds of the H1a kernels (shared by flag.cu and large.cu).
enum : int { kFlagA2a = 0, kFlagA2b = 1, kFlagA2c = 2, kFlagAgDirect = 3, kFlagAgFwd = 4 };

// All-gather with flags. 16-byte packs; layout 0 needs row_bytes % 16 == 0.
__device__ __forceinline__ int64_t ag_pos16(int origin, int64_t p, int64_t np_total, int rpr, int layout) {
  if (layout == 1) return (int64_t)origin * np_total + p;
  const int64_t t = p / rpr;
  return t * (kWorld * rpr) + (int64_t)origin * rpr + (p - t * rpr);
}

// `in`: my shard chunk (np packs, contiguous). `out`: base of the full output. e: call epoch (parity + flag tag).
// cut = true: the push to the partner (the socket cut) is done by one dedicated block (ag_cut_body) instead of by every
// block; this body then only pushes to the island peers. The cut block sends the same ranges and releases the same
// per-block flags, so the receiving side is unchanged. (SM push over the cut is 20% faster from one CTA than from many.)
__device__ __forceinline__ void ag_flag_body(const Ctx& c, int b, int nb, uint64_t e, const uint4* __restrict__ in,
                                             uint4* __restrict__ out, int np, int rpr, int layout, int64_t p0,
                                             int64_t np_total, bool cut = false) {
  const long long tag = static_cast<long long>(e) + 1;
  const int par = e & 1;
  const int ip = c.ipos, me = c.rank, pr = c.partner;
  const int tid = threadIdx.x, bd = blockDim.x;
  const int cp = (np + nb - 1) / nb;
  const int lo = min(np, b * cp), hi = min(np, lo + cp);
  char* mine = c.base[me];
  const int64_t hs = c.L.h_slot >> 4;
  uint4* rx = reinterpret_cast<uint4*>(mine + c.L.h_off + par * c.L.h_par);
  uint4* txp = reinterpret_cast<uint4*>(c.base[pr] + c.L.h_off + par * c.L.h_par) + me * hs;
  uint4* txi[3];
  uint4* fwd[3];
  int peer_rank[3];
#pragma unroll
  for (int s = 0; s < 3; ++s) {
    const int peer = c.members[c.island][(ip + 1 + s) & 3];
    uint4* pb = reinterpret_cast<uint4*>(c.base[peer] + c.L.h_off + par * c.L.h_par);
    txi[s] = pb + me * hs;
    fwd[s] = pb + pr * hs;
    peer_rank[s] = peer;
  }
  // hop 1: my shard to the partner (slow hop first) and to the island peers
  for (int i = lo + tid; i < hi; i += bd) {
    const uint4 x = in[i];
    if (!cut) st_v4(txp + i, x);
    st_v4(txi[0] + i, x);
    st_v4(txi[1] + i, x);
    st_v4(txi[2] + i, x);
    out[ag_pos16(me, p0 + i, np_total, rpr, layout)] = x;
  }
  __syncthreads();
  if (tid == 0 && !cut) st_release(flag_at(c, c.base[pr], kFlagAgDirect, b, me), tag);
  if (tid >= 1 && tid <= 3) st_release(flag_at(c, c.base[peer_rank[tid - 1]], kFlagAgDirect, b, me), tag);
  // hop 2: forward the partner's shard to the island peers
  if (tid == 0) wait_flag(c, flag_at(c, mine, kFlagAgDirect, b, pr), tag, 41);
  __syncthreads();
  for (int i = lo + tid; i < hi; i += bd) {
    const uint4 v = ld_v4(rx + pr * hs + i);
    st_v4(fwd[0] + i, v);
    st_v4(fwd[1] + i, v);
    st_v4(fwd[2] + i, v);
    out[ag_pos16(pr, p0 + i, np_total, rpr, layout)] = v;
  }
  __syncthreads();
  if (tid < 3) st_release(flag_at(c, c.base[peer_rank[tid]], kFlagAgFwd, b, me), tag);
  // collect the direct shards of the island peers and their forwarded copies
  if (tid < 3) wait_flag(c, flag_at(c, mine, kFlagAgDirect, b, peer_rank[tid]), tag, 42);
  if (tid >= 8 && tid < 11) wait_flag(c, flag_at(c, mine, kFlagAgFwd, b, peer_rank[tid - 8]), tag, 43);
  __syncthreads();
  for (int i = lo + tid; i < hi; i += bd) {
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      out[ag_pos16(peer_rank[s], p0 + i, np_total, rpr, layout)] = ld_v4(rx + peer_rank[s] * hs + i);
      const int org = c.partner_of[peer_rank[s]];
      out[ag_pos16(org, p0 + i, np_total, rpr, layout)] = ld_v4(rx + org * hs + i);
    }
  }
}

// The dedicated cut block of ag_flag_body(cut = true): pushes my shard chunk to the partner block range by block range
// (the ranges of the nb worker blocks) and releases the partner's per-block flag after each range.
__device__ __forceinline__ void ag_cut_body(const Ctx& c, int nb, uint64_t e, const uint4* __restrict__ in, int np) {
  const long long tag = static_cast<long long>(e) + 1;
  const int par = e & 1;
  const int tid = threadIdx.x, bd = blockDim.x;
  const int cp = (np + nb - 1) / nb;
  uint4* txp = reinterpret_cast<uint4*>(c.base[c.partner] + c.L.h_off + par * c.L.h_par) + c.rank * (c.L.h_slot >> 4);
  for (int b = 0; b < nb; ++b) {
    const int lo = min(np, b * cp), hi = min(np, lo + cp);
    for (int i = lo + tid; i < hi; i += 4 * bd) {
      uint4 x[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) x[q] = (i + q * bd < hi) ? in[i + q * bd] : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (i + q * bd < hi) st_v4(txp + i + q * bd, x[q]);
    }
    __syncthreads();
    if (tid == 0) st_release(flag_at(c, c.base[c.partner], kFlagAgDirect, b, c.rank), tag);
  }
}

}  // namespace muse_topo
