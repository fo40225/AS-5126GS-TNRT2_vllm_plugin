// GEMV of the sharded pooler / unpooler of a Muse Spark decode layer, with the first stage of the all-gather that follows it as its epilogue.
//
// A decode layer computes y = x . W_r^T for this rank's 512 (pooler, K = 8192) or 1024 (unpooler, K = 4096) output columns and then gathers the
// 8 column blocks on every rank (ag_bnd.cu: the gather is fused with the pre-expert RMSNorm / the residual boundary). The all-gather kernel's first
// stage pushes the rank's block to the partner and to the three island peers; here the GEMV CTAs do that themselves, each with the columns it has
// just computed, so the pushes start while the other CTAs still reduce and the second kernel (ag_bnd_kernel with `pushed`) only forwards, collects
// and runs its epilogue. The block is also stored in plain form (the second kernel reads its own columns from there).
//
// Launch contract: this kernel and the one after it are the two halves of ONE collective call. This one is a programmatic dependent of the kernel
// in front of it in the stream (an all-reduce launched with a trigger stage, Ctx::trigger, or a plain kernel: then it simply starts after it): the
// weights, which do not depend on anything, are loaded before griddepcontrol.wait; the activations x and the call epoch are read behind it. The
// weights must therefore be final when the kernel in front of it STARTS to trigger: static model parameters are; a copy made just in front of the
// call (a misaligned weight staged by TopoComm) is not, and such a call is launched without the programmatic attribute (`pdl` = false: the kernel
// then starts after the copy has completed). It lets its own programmatic successor start at once (the pushed ag_bnd kernel, which waits for this
// grid to complete). It does not advance the call counter; the second kernel does (epoch_begin_pushed).
//
// Arithmetic: gemv.cuh (fixed fp32 order, one bf16 rounding). Not the order of cuBLAS: deterministic_different against the cuBLAS gemv.
#include "common.cuh"
#include "gemv.cuh"

namespace muse_topo {

// Geometry of a projection: K x N is the shard GEMV of one rank, W warps per CTA, WPR warps per output row (rows per CTA = W / WPR, even:
// a CTA pushes whole 32-bit words, two adjacent columns). The summation order depends on WPR (gemv.cuh), so a variant with another WPR is
// another (deterministic) rounding; the variants of one projection differ in the grid (CTAs = N * WPR / W) and the loads in flight per
// thread (K / 8 / WPR / 32 chunks of 16 B). Variant 0 is the default (MUSE_TOPO_GEMV_VARIANT), the others exist for tuning.
template <int K_, int N_, int W_, int WPR_>
struct Geom {
  static constexpr int K = K_, N = N_, W = W_, WPR = WPR_;
};
using PoolerV0 = Geom<8192, 512, 8, 2>;    // 128 CTAs x 256 threads, 16 chunks per lane
using PoolerV1 = Geom<8192, 512, 8, 4>;    // 256 CTAs x 256 threads, 8 chunks per lane
using PoolerV2 = Geom<8192, 512, 16, 4>;   // 128 CTAs x 512 threads, 8 chunks per lane
using UnpoolerV0 = Geom<4096, 1024, 8, 1>;   // 128 CTAs x 256 threads, 16 chunks per lane
using UnpoolerV1 = Geom<4096, 1024, 8, 2>;   // 256 CTAs x 256 threads, 8 chunks per lane
using UnpoolerV2 = Geom<4096, 1024, 16, 2>;  // 128 CTAs x 512 threads, 8 chunks per lane
using PoolerV3 = Geom<8192, 512, 16, 2>;     // 64 CTAs x 512 threads, 16 chunks per lane: half the activation traffic of V0 at larger T
using UnpoolerV3 = Geom<4096, 1024, 16, 1>;  // 64 CTAs x 512 threads, 16 chunks per lane

// Word of Ctx::ctl (a 512-byte device buffer; ctl[0..1] is the call counter, ctl[32] the arrival counter) that counts the finished CTAs of a
// gemv_push_kernel with push == 2.
constexpr int kLastCounter = 40;

template <class G, int T>
__global__ void __launch_bounds__(G::W * 32)
    gemv_push_kernel(const Ctx c, const __nv_bfloat16* x, const __nv_bfloat16* __restrict__ w, uint32_t* shard, int push) {
  constexpr int K = G::K, N = G::N, W = G::W, WPR = G::WPR;
  constexpr int RPC = W / WPR;      // output rows (columns of the shard) per CTA
  constexpr int PAIRS = RPC / 2;    // 32-bit words (column pairs) per token row per CTA
  constexpr int WPT = N / 2;        // words per token row of the shard
  static_assert(RPC % 2 == 0 && N % RPC == 0 && T * PAIRS <= W * 32, "geometry");
  __shared__ float part[W][T];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rg = warp / WPR, wr = warp % WPR;
  const int row = blockIdx.x * RPC + rg;

  // push: bits 0..1 = push mode (0: the gather kernel pushes, 1: every CTA pushes its words, 2: the last CTA pushes the block), bit 2 =
  // the weights are loaded behind the wait for the predecessor instead of before it (A/B switch of the early weight loads).
  const int mode = push & 3;
  const bool late = (push & 4) != 0;
  pdl_trigger();  // the pushed ag_bnd kernel may become resident now; it waits for this grid to complete
  gv::Gemv<K, T, WPR> g;
  if (!late) g.load_weights(w + (size_t)row * K, wr, lane);
  pdl_wait();  // x, and the call counter, are final
  if (late) g.load_weights(w + (size_t)row * K, wr, lane);
  const uint64_t e = ld_u64(c.ctl);
  g.accumulate(x, wr, lane);
  g.warp_reduce();
  if (lane < T) part[warp][lane] = g.warp_value(lane);
  __syncthreads();
  const uint32_t tag = ll8_tag(e);
  const int par = e & 1;
  const int64_t gs = c.L.g_slot >> 3;  // slot size in 8-byte units
  const int me = c.rank, ip = c.ipos;
  uint2* tx[4];  // partner, then the three island peers: this rank's row of their receive slots
  tx[0] = reinterpret_cast<uint2*>(c.base[c.partner] + c.L.g_off + par * c.L.g_par) + me * gs;
#pragma unroll
  for (int s = 0; s < 3; ++s)
    tx[1 + s] = reinterpret_cast<uint2*>(c.base[c.members[c.island][(ip + 1 + s) & 3]] + c.L.g_off + par * c.L.g_par) + me * gs;
  if (threadIdx.x < T * PAIRS) {
    const int t = threadIdx.x / PAIRS, p = threadIdx.x % PAIRS;
    float v0 = part[(2 * p) * WPR][t], v1 = part[(2 * p + 1) * WPR][t];
#pragma unroll
    for (int k = 1; k < WPR; ++k) {
      v0 += part[(2 * p) * WPR + k][t];
      v1 += part[(2 * p + 1) * WPR + k][t];
    }
    const __nv_bfloat162 pr = __floats2bfloat162_rn(v0, v1);
    const uint32_t word = *reinterpret_cast<const uint32_t*>(&pr);
    const int unit = t * WPT + blockIdx.x * PAIRS + p;
    shard[unit] = word;
    // mode 1: this CTA sends its words to the four destinations right away (8-byte units of a few CTAs each).
    if (mode == 1) {
#pragma unroll
      for (int d = 0; d < 4; ++d) st_v2(tx[d] + unit, word, tag);
    }
  }
  if (mode == 2) {
    // mode 2: the last CTA to finish sends the whole block, in coalesced stores. Every writer fences its words, one thread counts the
    // CTAs (the counter returns to zero in the last one, so a replayed call finds it clean), the last one fences and reads them back.
    __shared__ int is_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint32_t old = atomicAdd(c.ctl + kLastCounter, 1u);
      is_last = old == gridDim.x - 1;
    }
    __syncthreads();
    if (is_last) {
      __threadfence();
      if (threadIdx.x == 0) st_u32(c.ctl + kLastCounter, 0u);
      for (int u = threadIdx.x; u < T * WPT; u += W * 32) {
        const uint32_t word = ld_u32(shard + u);
#pragma unroll
        for (int d = 0; d < 4; ++d) st_v2(tx[d] + u, word, tag);
      }
    }
  }
}

template <class G, int T>
static void run_gemv_push(const Ctx& c, const void* x, const void* w, void* shard, int push, bool pdl, cudaStream_t s) {
  launch_kp(gemv_push_kernel<G, T>, pdl, dim3(G::N / (G::W / G::WPR)), dim3(G::W * 32), s, c,
            static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w), static_cast<uint32_t*>(shard), push);
}

template <class G>
static void dispatch_rows(const Ctx& c, const void* x, const void* w, void* shard, int rows, int push, bool pdl, cudaStream_t s) {
  switch (rows) {
    case 1: run_gemv_push<G, 1>(c, x, w, shard, push, pdl, s); break;
    case 2: run_gemv_push<G, 2>(c, x, w, shard, push, pdl, s); break;
    case 3: run_gemv_push<G, 3>(c, x, w, shard, push, pdl, s); break;
    case 4: run_gemv_push<G, 4>(c, x, w, shard, push, pdl, s); break;
    case 5: run_gemv_push<G, 5>(c, x, w, shard, push, pdl, s); break;
    case 6: run_gemv_push<G, 6>(c, x, w, shard, push, pdl, s); break;
    case 7: run_gemv_push<G, 7>(c, x, w, shard, push, pdl, s); break;
    case 8: run_gemv_push<G, 8>(c, x, w, shard, push, pdl, s); break;
    default: break;  // rejected by the caller (bindings.cpp)
  }
}

void launch_gemv_push(const Ctx& c, int kind, int variant, const void* x, const void* w, void* shard, int rows, int push, bool pdl,
                      cudaStream_t s) {
  if (kind == 0) {
    if (variant == 1) dispatch_rows<PoolerV1>(c, x, w, shard, rows, push, pdl, s);
    else if (variant == 2) dispatch_rows<PoolerV2>(c, x, w, shard, rows, push, pdl, s);
    else if (variant == 3) dispatch_rows<PoolerV3>(c, x, w, shard, rows, push, pdl, s);
    else dispatch_rows<PoolerV0>(c, x, w, shard, rows, push, pdl, s);
  } else {
    if (variant == 1) dispatch_rows<UnpoolerV1>(c, x, w, shard, rows, push, pdl, s);
    else if (variant == 2) dispatch_rows<UnpoolerV2>(c, x, w, shard, rows, push, pdl, s);
    else if (variant == 3) dispatch_rows<UnpoolerV3>(c, x, w, shard, rows, push, pdl, s);
    else dispatch_rows<UnpoolerV0>(c, x, w, shard, rows, push, pdl, s);
  }
}

}  // namespace muse_topo
