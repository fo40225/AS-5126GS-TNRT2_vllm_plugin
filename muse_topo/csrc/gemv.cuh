// Decode GEMV core of the pooler / unpooler push kernels: y[t][n] = sum_k x[t][k] * w[n][k] for T <= 8 token rows, bf16 operands,
// fp32 accumulation in a fixed order, one rounding to bf16 at the end. Not cuBLAS: the order of the additions is its own (class
// deterministic_different against the cuBLAS gemv the model used before), but it depends only on K, the number of warps per row and the
// element position, never on T, the grid or the rank, so a token row gets the same bits at every batch size.
//
// Decomposition: a CTA of W warps owns W / WPR output rows; WPR warps share a row, warp `wr` takes the slice [wr * K / WPR, (wr + 1) * K / WPR)
// of it. Lane l of a warp loads the 16-byte chunks l, l + 32, l + 64, ... of its slice (a warp instruction reads 512 contiguous bytes). The weight
// chunks of a thread are independent of every earlier kernel, so a kernel launched with programmatic dependent launch issues them before
// griddepcontrol.wait and holds them in registers (LPL * 16 B per thread: 256 B for the 8 MB shards of the model at 128 CTAs of 256 threads); only
// the activation loads wait for the producer.
//
// Reduction order of one output element (row n, token t):
//   thread: four accumulators a[e & 3], e = element position inside the chunk; chunks in ascending j; a[e] = fma(w, x, a[e]);
//           thread value v = (a0 + a1) + (a2 + a3)
//   warp:   xor butterfly over the 32 lanes (offsets 16, 8, 4, 2, 1; every lane ends with the same bits)
//   row:    the WPR warp values added in ascending warp order, then one round-to-nearest-even to bf16.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace muse_topo {
namespace gv {

// Weights: streamed once, no L1 allocation. Deliberately not ld.global.nc: ptxas moves a .nc load below griddepcontrol.wait (it treats the
// data as immutable for the kernel), which would turn the early weight loads into loads behind the wait; a plain load stays above it.
__device__ __forceinline__ uint4 ld_w(const void* p) {
  uint4 v;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
// Activations: written by the producer kernel, so a plain load that stays behind griddepcontrol.wait (a .nc load would be hoisted above it
// and read the producer's output too early). Shared by the row groups of a CTA through L1.
__device__ __forceinline__ uint4 ld_x(const void* p) {
  uint4 v;
  asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  f[0] = __uint_as_float(v.x << 16);
  f[1] = __uint_as_float(v.x & 0xffff0000u);
  f[2] = __uint_as_float(v.y << 16);
  f[3] = __uint_as_float(v.y & 0xffff0000u);
  f[4] = __uint_as_float(v.z << 16);
  f[5] = __uint_as_float(v.z & 0xffff0000u);
  f[6] = __uint_as_float(v.w << 16);
  f[7] = __uint_as_float(v.w & 0xffff0000u);
}

template <int K, int T, int WPR>
struct Gemv {
  static constexpr int CH = K / 8;            // 16-byte chunks per row
  static constexpr int SLICE = CH / WPR;      // chunks per warp slice
  static constexpr int LPL = SLICE / 32;      // chunks per lane
  static_assert(K % 8 == 0 && CH % (WPR * 32) == 0, "K must split into whole 512-byte warp loads per slice");
  static_assert(T >= 1 && T <= 8, "1 <= T <= 8");

  uint4 wv[LPL];
  float acc[T][4];

  // The weight chunks of row `wrow` (K bf16) for slice `wr`, lane `lane`.
  __device__ __forceinline__ void load_weights(const __nv_bfloat16* wrow, int wr, int lane) {
    const uint4* p = reinterpret_cast<const uint4*>(wrow) + wr * SLICE + lane;
#pragma unroll
    for (int j = 0; j < LPL; ++j) wv[j] = ld_w(p + 32 * j);
  }

  // Accumulates the products with the activations x [T][K]; returns the thread's T values in `acc[t][0]`.
  __device__ __forceinline__ void accumulate(const __nv_bfloat16* x, int wr, int lane) {
    const uint4* px = reinterpret_cast<const uint4*>(x) + wr * SLICE + lane;
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int a = 0; a < 4; ++a) acc[t][a] = 0.f;
#pragma unroll
    for (int j = 0; j < LPL; ++j) {
      float wf[8];
      unpack8(wv[j], wf);
#pragma unroll
      for (int t = 0; t < T; ++t) {
        float xf[8];
        unpack8(ld_x(px + t * CH + 32 * j), xf);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[t][e & 3] = __fmaf_rn(wf[e], xf[e], acc[t][e & 3]);
      }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t][0] = (acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3]);
  }

  // Butterfly over the lanes of the warp; every lane ends with the warp sum of its token.
  __device__ __forceinline__ void warp_reduce() {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float v = acc[t][0];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      acc[t][0] = v;
    }
  }

  // Lane t < T of the warp stores the warp sum of token t.
  __device__ __forceinline__ float warp_value(int lane) const {
    float v = 0.f;
#pragma unroll
    for (int t = 0; t < T; ++t)
      if (lane == t) v = acc[t][0];
    return v;
  }
};

}  // namespace gv
}  // namespace muse_topo
