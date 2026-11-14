// All-gather (column concatenation, LL8 protocol of ag_ll_kernel) fused with the Muse Spark epilogue kernels of boundary.cuh:
//   * epi 0: bf16 shards [T, 1024] (the unpooler columns of a rank) -> the gathered branch [T, 8192] feeds the residual
//     boundary (post-norm, gated residual, next pre-norm), T <= 32 (the LL8 slots hold 64 KiB shards);
//   * epi 1: bf16 shards [T, 512] (the pooler columns) -> [T, 4096] feeds the pre_expert_norm RMSNorm, T <= 64.
//
// One CTA of 256 threads handles one row; thread i moves the words i, i + 256, ... of the shard row exactly like the word
// loop of ag_ll_kernel (same units {data, tag}, same three stages, so the gather itself stays bit exact for every payload).
// Instead of storing the gathered words to the output tensor, the CTA stages the whole gathered row in shared memory and, after a
// barrier, every thread reads the 16-byte packs of the Triton layout of the epilogue (boundary.cuh) from there, so the gathered
// tensor never goes through global memory. The loads of the epilogue that do not depend on the gather are issued first.
#include "boundary.cuh"
#include "common.cuh"

namespace muse_topo {

template <int WPR, int EPI, bool POST_SCALE, bool FP32_OUT>
__global__ void __launch_bounds__(bnd::kRowThreads) ag_bnd_kernel(const Ctx c, const uint32_t* __restrict__ in,
                                                                  const BndParams bp, const RmsParams rp) {
  constexpr int NW = WPR / bnd::kRowThreads;  // words of the shard row per thread
  static_assert(WPR % bnd::kRowThreads == 0, "shard row must be a multiple of the block");
  __shared__ __align__(16) uint32_t srow[kWorld * WPR];  // the gathered row, 8 origins in rank order
  __shared__ float sm[66];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  const uint32_t tag = ll8_tag(e);
  const int par = e & 1;
  const int ip = c.ipos;
  const int me = c.rank;
  const int pr = c.partner;
  const int64_t gs = c.L.g_slot >> 3;  // slot size in 8-byte units
  uint2* rx = reinterpret_cast<uint2*>(c.base[me] + c.L.g_off + par * c.L.g_par);
  uint2* txp = reinterpret_cast<uint2*>(c.base[pr] + c.L.g_off + par * c.L.g_par) + me * gs;
  uint2* txi[3];
  uint2* fwd[3];
  int org[6];
#pragma unroll
  for (int s = 0; s < 3; ++s) {
    const int peer = c.members[c.island][(ip + 1 + s) & 3];
    uint2* pb = reinterpret_cast<uint2*>(c.base[peer] + c.L.g_off + par * c.L.g_par);
    txi[s] = pb + me * gs;
    fwd[s] = pb + pr * gs;
    org[s] = peer;
    org[3 + s] = c.partner_of[peer];
  }

  const bnd::Bnd8192 ba(bp);
  bnd::Bnd8192Regs regs;
  if (EPI == 0 && (bp.ld_mode == 1 || bp.ld_mode == 3)) bnd::bnd8192_prefetch_l2<POST_SCALE>(ba, row);

  TRACE(c, 0);
  // stage 1: my shard words to the partner and the island peers
  uint32_t x[NW];
#pragma unroll
  for (int k = 0; k < NW; ++k) {
    const int w = row * WPR + tid + bnd::kRowThreads * k;
    x[k] = in[w];
    st_v2(txp + w, x[k], tag);
  }
#pragma unroll
  for (int k = 0; k < NW; ++k) {
    const int w = row * WPR + tid + bnd::kRowThreads * k;
    st_v2(txi[0] + w, x[k], tag);
    st_v2(txi[1] + w, x[k], tag);
    st_v2(txi[2] + w, x[k], tag);
    srow[me * WPR + tid + bnd::kRowThreads * k] = x[k];
  }
  // The loads of the epilogue that do not depend on the gather, issued behind the pushes (see ar_bnd.cu).
  if (EPI == 0 && (bp.ld_mode == 0 || bp.ld_mode == 3)) bnd::bnd8192_load<POST_SCALE>(ba, row, regs);
  uint4 wx[bnd::kD4096Packs];
  const bnd::Rms4096 ra0(rp);
  if (EPI == 1) bnd::rms4096_load(ra0, wx);
  TRACE(c, 1);
  // stage 2: the partner's words: keep and forward
  uint2 vp[NW];
#pragma unroll
  for (int k = 0; k < NW; ++k) vp[k] = make_uint2(0, ~tag);
  {
    Spin sp;
    bool ok;
    do {
      ok = true;
#pragma unroll
      for (int k = 0; k < NW; ++k) {
        const int w = row * WPR + tid + bnd::kRowThreads * k;
        if (vp[k].y != tag) vp[k] = ld_v2(rx + pr * gs + w);
      }
#pragma unroll
      for (int k = 0; k < NW; ++k)
        if (vp[k].y != tag) ok = false;
      if (!ok) sp.tick(c, 51);
    } while (!ok);
  }
#pragma unroll
  for (int k = 0; k < NW; ++k) {
    const int w = row * WPR + tid + bnd::kRowThreads * k;
    st_v2(fwd[0] + w, vp[k].x, tag);
    st_v2(fwd[1] + w, vp[k].x, tag);
    st_v2(fwd[2] + w, vp[k].x, tag);
    srow[pr * WPR + tid + bnd::kRowThreads * k] = vp[k].x;
  }
  TRACE(c, 2);
  // stage 3: three direct words from the island peers and three forwarded ones (their partners')
  uint32_t got[NW];
#pragma unroll
  for (int k = 0; k < NW; ++k) got[k] = 0;
  {
    Spin sp;
    bool ok;
    do {
      uint2 v[NW][6];
#pragma unroll
      for (int k = 0; k < NW; ++k) {
        const int w = row * WPR + tid + bnd::kRowThreads * k;
#pragma unroll
        for (int j = 0; j < 6; ++j)
          if (!((got[k] >> j) & 1)) v[k][j] = ld_v2(rx + org[j] * gs + w);
      }
      ok = true;
#pragma unroll
      for (int k = 0; k < NW; ++k) {
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          if (!((got[k] >> j) & 1)) {
            if (v[k][j].y == tag) {
              srow[org[j] * WPR + tid + bnd::kRowThreads * k] = v[k][j].x;
              got[k] |= 1u << j;
            } else {
              ok = false;
            }
          }
        }
      }
      if (!ok) sp.tick(c, 52);
    } while (!ok);
  }
  if (EPI == 0 && (bp.ld_mode == 1 || bp.ld_mode == 2)) bnd::bnd8192_load<POST_SCALE>(ba, row, regs);
  TRACE(c, 3);
  __syncthreads();
  if (EPI == 0) {
    uint4 bx[bnd::kD8192Packs];
#pragma unroll
    for (int j = 0; j < bnd::kD8192Packs; ++j)
      bx[j] = *reinterpret_cast<const uint4*>(srow + 4 * (bnd::kRowThreads * j + tid));
    bnd::bnd8192_apply<POST_SCALE, FP32_OUT>(ba, row, bx, regs, sm, c.trace);
  } else {
    uint4 bx[bnd::kD4096Packs];
#pragma unroll
    for (int j = 0; j < bnd::kD4096Packs; ++j)
      bx[j] = *reinterpret_cast<const uint4*>(srow + 4 * (bnd::kRowThreads * j + tid));
    const bnd::Rms4096 ra(rp);
    bnd::rms4096_apply(ra, row, bx, wx, sm);
  }
  TRACE(c, 10);
  epoch_end(c, e, old);
}

void launch_ag_bnd(const Ctx& c, const void* shard, const BndParams& a, int rows, cudaStream_t s) {
  const uint32_t* i = static_cast<const uint32_t*>(shard);
  const dim3 grid(rows), block(bnd::kRowThreads);
  const RmsParams none{};
  const bool ps = a.post_w != nullptr, f32 = a.out_fp32 != nullptr;
  if (ps && f32) launch_k(ag_bnd_kernel<1024 / 2, 0, true, true>, c, grid, block, s, c, i, a, none);
  else if (ps) launch_k(ag_bnd_kernel<1024 / 2, 0, true, false>, c, grid, block, s, c, i, a, none);
  else if (f32) launch_k(ag_bnd_kernel<1024 / 2, 0, false, true>, c, grid, block, s, c, i, a, none);
  else launch_k(ag_bnd_kernel<1024 / 2, 0, false, false>, c, grid, block, s, c, i, a, none);
}

void launch_ag_rms(const Ctx& c, const void* shard, const RmsParams& a, int rows, cudaStream_t s) {
  const uint32_t* i = static_cast<const uint32_t*>(shard);
  const BndParams none{};
  launch_k(ag_bnd_kernel<512 / 2, 1, false, false>, c, dim3(rows), dim3(bnd::kRowThreads), s, c, i, none, a);
}

}  // namespace muse_topo
