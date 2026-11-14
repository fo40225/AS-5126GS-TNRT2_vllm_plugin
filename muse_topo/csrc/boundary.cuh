// Bit-exact CUDA mirror of the residual-boundary Triton kernels of the Muse Spark model (vLLM branch
// sm120-muse-spark-open, vllm/models/muse_spark/common/ops/fused_rmsnorm.py), used as the epilogue of the fused
// all-reduce / all-gather kernels. "Bit-exact" is defined against the SASS of the Triton kernels on sm_120a (Triton 3.7.1,
// ptxas 13.x), not against the Python source: ptxas contracts some of the multiplies of the sum of squares into FMAs, and
// the reduction tree is the one Triton generates for the block layout of the kernel.
//
// Kernels mirrored (one CTA handles one row; Triton's program = one row for D = 8192, two rows for D = 4096):
//   * _norm_residual_norm_fwd_kernel, D = 8192, bT = 1, num_warps = 32 (1024 threads, thread t owns the 8 contiguous
//     elements [8t, 8t + 8) = 16-byte pack t of the row):
//         x       = f32(branch)                  [* (f32(post_w) + post_gain)  with POST_SCALE]
//         sum_sq  = tree(thread chain)           chain = round(x1*x1); fma(x0,x0); fma(x2,x2) .. fma(x7,x7)
//         y       = x * rsqrt.approx.ftz(fma(sum_sq, 2^-13, post_eps))
//         new     = fma(alpha, stream, round(beta * y))                       (fp32, stored)
//         z       = (new * rsqrt.approx.ftz(fma(sum_sq(new), 2^-13, pre_eps))) * (f32(pre_w) + pre_gain)
//         out     = bf16_rn(z)                   [out_fp32 = f32(out)]
//     tree: xor-butterfly over the 32 lanes (offsets 16, 8, 4, 2, 1; a + b is commutative, so every lane ends with the
//     same value), the 32 warp partials go through shared memory and a second butterfly of the same shape (Triton: warp 0 alone;
//     here every warp, the value is identical).
//   * _rmsnorm_fwd_kernel, D = 4096 (pre_expert_norm), bT = 2, num_warps = 8 (256 threads; thread t owns packs t and
//     256 + t of its row; chain over the 16 elements: round(x1*x1) of pack t, then fma over x0, x2..x7 of pack t, then x0..x7
//     of pack 256 + t); tree: lane butterfly 16..1, the 8 warp partials of a row through shared memory, butterfly 4, 2, 1:
//         y = (x * rsqrt.approx.ftz(fma(sum_sq, 2^-12, eps))) * (f32(w) + gain);  out = bf16_rn(y)
//
// Everything is written with explicit rounded intrinsics so that nvcc / ptxas cannot contract or reassociate. The standalone
// proof against the real Triton kernels is bench/test_boundary.py.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "topo_types.h"

namespace muse_topo {
namespace bnd {

// Debug timeline (block 0, thread 0), compiled in only with -DMUSE_TOPO_TRACE like TRACE() in common.cuh.
#ifdef MUSE_TOPO_TRACE
#define BND_STAMP(trace, i)                                                                    \
  do {                                                                                         \
    if ((trace) && blockIdx.x == 0 && threadIdx.x == 0) {                                      \
      unsigned long long t_;                                                                   \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_));                                   \
      (trace)[i] = t_;                                                                         \
    }                                                                                          \
  } while (0)
#else
#define BND_STAMP(trace, i) \
  do {                      \
  } while (0)
#endif

constexpr int kRowThreads = 256;  // threads of a row CTA, both for D = 8192 (4 packs each) and D = 4096 (2 packs each)
constexpr int kD8192Packs = 4;
constexpr int kD4096Packs = 2;

__device__ __forceinline__ float rsqrt_ftz(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// The 8 bf16 of a 16-byte pack as floats (exact widening).
__device__ __forceinline__ void unpack_bf16x8(const uint4& p, float (&v)[8]) {
  v[0] = bf_lo(p.x);
  v[1] = bf_hi(p.x);
  v[2] = bf_lo(p.y);
  v[3] = bf_hi(p.y);
  v[4] = bf_lo(p.z);
  v[5] = bf_hi(p.z);
  v[6] = bf_lo(p.w);
  v[7] = bf_hi(p.w);
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint4 pack_bf16x8(const float (&v)[8]) {
  return make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]),
                    pack_bf16x2(v[6], v[7]));
}

// Thread chain of the first 8 elements of a pack (see the header comment): the odd element 1 starts the chain as a plain
// product, element 0 is the first FMA.
__device__ __forceinline__ float chain8(const float (&v)[8]) {
  float acc = __fmul_rn(v[1], v[1]);
  acc = __fmaf_rn(v[0], v[0], acc);
#pragma unroll
  for (int k = 2; k < 8; ++k) acc = __fmaf_rn(v[k], v[k], acc);
  return acc;
}
// Continuation over a second pack (D = 4096: elements 0..7 in order).
__device__ __forceinline__ float chain8_cont(float acc, const float (&v)[8]) {
#pragma unroll
  for (int k = 0; k < 8; ++k) acc = __fmaf_rn(v[k], v[k], acc);
  return acc;
}

template <int FROM>
__device__ __forceinline__ float bfly(float v) {
#pragma unroll
  for (int o = FROM; o >= 1; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// D = 8192 row tree. p[j] = chain of virtual Triton thread 256 * j + threadIdx.x (virtual warp 8 * j + warp). `sm`: 32 floats
// private to this call. Contains one __syncthreads; every thread of the 256-thread CTA must call it. The second stage (the butterfly
// over the 32 warp partials, run by warp 0 in the Triton kernel) is done redundantly by every warp, which saves the broadcast of its
// result: the value is the same bit for bit in every lane of every warp.
__device__ __forceinline__ float row_tree8192(float (&p)[kD8192Packs], float* sm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) {
    p[j] = bfly<16>(p[j]);
    if (lane == 0) sm[8 * j + warp] = p[j];
  }
  __syncthreads();
  return bfly<16>(sm[lane]);
}

// Pointers and constants of the D = 8192 boundary (BndParams in topo_types.h; the bf16 tensors are plain pointers there).
struct Bnd8192 {
  const float* stream;
  float* new_stream;
  __nv_bfloat16* out;
  float* out_fp32;
  const float* alpha;
  const float* beta;
  const __nv_bfloat16* post_w;
  const __nv_bfloat16* pre_w;
  float post_eps, pre_eps, post_gain, pre_gain;
  __device__ __forceinline__ Bnd8192(const BndParams& p)
      : stream(p.stream), new_stream(p.new_stream), out(static_cast<__nv_bfloat16*>(p.out)), out_fp32(p.out_fp32),
        alpha(p.alpha), beta(p.beta), post_w(static_cast<const __nv_bfloat16*>(p.post_w)),
        pre_w(static_cast<const __nv_bfloat16*>(p.pre_w)), post_eps(p.post_eps), pre_eps(p.pre_eps),
        post_gain(p.post_gain), pre_gain(p.pre_gain) {}
};

// Values a thread can load before the all-reduce / all-gather result exists (everything but the branch): the stream, the
// gate coefficients and the weights of its four packs of row `row`.
struct Bnd8192Regs {
  float stream[kD8192Packs][8];
  float alpha[kD8192Packs][8];
  float beta[kD8192Packs][8];
  uint4 post_w[kD8192Packs];
  uint4 pre_w[kD8192Packs];
};

__device__ __forceinline__ void load8f(const float* p, float (&v)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x;
  v[1] = a.y;
  v[2] = a.z;
  v[3] = a.w;
  v[4] = b.x;
  v[5] = b.y;
  v[6] = b.z;
  v[7] = b.w;
}
__device__ __forceinline__ void store8f(float* p, const float (&v)[8]) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// Load of operands that never change (gate coefficients, norm weights) ahead of griddepcontrol.wait of a programmatic dependent: a volatile
// asm with a memory clobber, so that ptxas keeps it above the wait (it moves a plain or .nc load below it).
__device__ __forceinline__ uint4 ld_early(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void load8f_early(const float* p, float (&v)[8]) {
  const uint4 a = ld_early(p), b = ld_early(p + 4);
  v[0] = __uint_as_float(a.x);
  v[1] = __uint_as_float(a.y);
  v[2] = __uint_as_float(a.z);
  v[3] = __uint_as_float(a.w);
  v[4] = __uint_as_float(b.x);
  v[5] = __uint_as_float(b.y);
  v[6] = __uint_as_float(b.z);
  v[7] = __uint_as_float(b.w);
}

// The part of bnd8192_load that does not change from call to call (gate coefficients and weights): may run before the wait for the
// producer of the branch.
template <bool POST_SCALE>
__device__ __forceinline__ void bnd8192_load_static(const Bnd8192& a, Bnd8192Regs& r) {
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) {
    const int c = 8 * (kRowThreads * j + threadIdx.x);
    load8f_early(a.alpha + c, r.alpha[j]);
    load8f_early(a.beta + c, r.beta[j]);
    r.pre_w[j] = ld_early(a.pre_w + c);
    if (POST_SCALE) r.post_w[j] = ld_early(a.post_w + c);
  }
}
// The residual stream of row `row`: written by an earlier kernel of the layer.
__device__ __forceinline__ void bnd8192_load_stream(const Bnd8192& a, int row, Bnd8192Regs& r) {
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j)
    load8f(a.stream + (int64_t)row * 8192 + 8 * (kRowThreads * j + threadIdx.x), r.stream[j]);
}

template <bool POST_SCALE>
__device__ __forceinline__ void bnd8192_load(const Bnd8192& a, int row, Bnd8192Regs& r) {
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) {
    const int64_t e = (int64_t)row * 8192 + 8 * (kRowThreads * j + threadIdx.x);
    const int c = 8 * (kRowThreads * j + threadIdx.x);
    load8f(a.stream + e, r.stream[j]);
    load8f(a.alpha + c, r.alpha[j]);
    load8f(a.beta + c, r.beta[j]);
    r.pre_w[j] = *reinterpret_cast<const uint4*>(a.pre_w + c);
    if (POST_SCALE) r.post_w[j] = *reinterpret_cast<const uint4*>(a.post_w + c);
  }
}

// EXPERIMENT: L2 prefetch of the operands of bnd8192_load (no registers, no wait).
template <bool POST_SCALE>
__device__ __forceinline__ void bnd8192_prefetch_l2(const Bnd8192& a, int row) {
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) {
    const int64_t e = (int64_t)row * 8192 + 8 * (kRowThreads * j + threadIdx.x);
    const int c = 8 * (kRowThreads * j + threadIdx.x);
    asm volatile("prefetch.global.L2 [%0];" ::"l"(a.stream + e));
    asm volatile("prefetch.global.L2 [%0];" ::"l"(a.alpha + c));
    asm volatile("prefetch.global.L2 [%0];" ::"l"(a.beta + c));
    asm volatile("prefetch.global.L2 [%0];" ::"l"(a.pre_w + c));
    if (POST_SCALE) asm volatile("prefetch.global.L2 [%0];" ::"l"(a.post_w + c));
  }
}

// The whole boundary of one row from its all-reduced / gathered branch. bx[j] = 16-byte pack 256 * j + threadIdx.x of the
// row (8 bf16). `sm`: 66 floats of shared memory. All 256 threads of the CTA must call it.
template <bool POST_SCALE, bool FP32_OUT>
__device__ __forceinline__ void bnd8192_apply(const Bnd8192& a, int row, const uint4 (&bx)[kD8192Packs],
                                              const Bnd8192Regs& r, float* sm,
                                              unsigned long long* trace = nullptr) {
  float x[kD8192Packs][8];
  float part[kD8192Packs];
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) {
    unpack_bf16x8(bx[j], x[j]);
    if (POST_SCALE) {
      float w[8];
      unpack_bf16x8(r.post_w[j], w);
#pragma unroll
      for (int k = 0; k < 8; ++k) x[j][k] = __fmul_rn(__fadd_rn(w[k], a.post_gain), x[j][k]);
    }
    part[j] = chain8(x[j]);
  }
  const float rs = rsqrt_ftz(__fmaf_rn(row_tree8192(part, sm), 0.0001220703125f, a.post_eps));
  BND_STAMP(trace, 7);
  float nw[kD8192Packs][8];
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float y = __fmul_rn(x[j][k], rs);
      nw[j][k] = __fmaf_rn(r.alpha[j][k], r.stream[j][k], __fmul_rn(r.beta[j][k], y));
    }
    store8f(a.new_stream + (int64_t)row * 8192 + 8 * (kRowThreads * j + threadIdx.x), nw[j]);
    part[j] = chain8(nw[j]);
  }
  BND_STAMP(trace, 8);
  const float rs2 = rsqrt_ftz(__fmaf_rn(row_tree8192(part, sm + 33), 0.0001220703125f, a.pre_eps));
  BND_STAMP(trace, 9);
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) {
    float w[8], z[8];
    unpack_bf16x8(r.pre_w[j], w);
#pragma unroll
    for (int k = 0; k < 8; ++k) z[k] = __fmul_rn(__fmul_rn(nw[j][k], rs2), __fadd_rn(w[k], a.pre_gain));
    const uint4 o = pack_bf16x8(z);
    const int64_t e = (int64_t)row * 8192 + 8 * (kRowThreads * j + threadIdx.x);
    *reinterpret_cast<uint4*>(a.out + e) = o;
    if (FP32_OUT) {
      float f[8];
      unpack_bf16x8(o, f);
      store8f(a.out_fp32 + e, f);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// D = 4096 RMSNorm (pre_expert_norm), weight applied after the normalization, bf16 in and out.
// ---------------------------------------------------------------------------------------------------------------------
// D = 4096 row tree: p = chain of virtual Triton thread threadIdx.x. `sm`: 8 floats private to this call. One __syncthreads; the
// second stage (butterfly 4, 2, 1 over the 8 warp partials) is done by every warp, see row_tree8192.
__device__ __forceinline__ float row_tree4096(float p, float* sm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  p = bfly<16>(p);
  if (lane == 0) sm[warp] = p;
  __syncthreads();
  return bfly<4>(sm[lane & 7]);
}

struct Rms4096 {
  __nv_bfloat16* out;            // [T, 4096] bf16
  const __nv_bfloat16* w;        // [4096]
  float eps, gain;
  __device__ __forceinline__ Rms4096(const RmsParams& p)
      : out(static_cast<__nv_bfloat16*>(p.out)), w(static_cast<const __nv_bfloat16*>(p.w)), eps(p.eps), gain(p.gain) {}
};

// The weight packs of this thread (16-byte packs 256 * c + threadIdx.x, c = 0, 1): loadable before the gather result exists.
__device__ __forceinline__ void rms4096_load(const Rms4096& a, uint4 (&wx)[kD4096Packs]) {
#pragma unroll
  for (int c = 0; c < kD4096Packs; ++c)
    wx[c] = *reinterpret_cast<const uint4*>(a.w + 8 * (kRowThreads * c + threadIdx.x));
}
// The same ahead of griddepcontrol.wait (see ld_early).
__device__ __forceinline__ void rms4096_load_early(const Rms4096& a, uint4 (&wx)[kD4096Packs]) {
#pragma unroll
  for (int c = 0; c < kD4096Packs; ++c) wx[c] = ld_early(a.w + 8 * (kRowThreads * c + threadIdx.x));
}

// bx[c] = 16-byte pack 256 * c + threadIdx.x of the row (8 bf16), c = 0, 1; wx = rms4096_load.
__device__ __forceinline__ void rms4096_apply(const Rms4096& a, int row, const uint4 (&bx)[kD4096Packs],
                                              const uint4 (&wx)[kD4096Packs], float* sm) {
  float x[kD4096Packs][8];
  unpack_bf16x8(bx[0], x[0]);
  unpack_bf16x8(bx[1], x[1]);
  const float p = chain8_cont(chain8(x[0]), x[1]);
  const float rs = rsqrt_ftz(__fmaf_rn(row_tree4096(p, sm), 0.000244140625f, a.eps));
#pragma unroll
  for (int c = 0; c < kD4096Packs; ++c) {
    const int col = 8 * (kRowThreads * c + threadIdx.x);
    float w[8], y[8];
    unpack_bf16x8(wx[c], w);
#pragma unroll
    for (int k = 0; k < 8; ++k) y[k] = __fmul_rn(__fmul_rn(x[c][k], rs), __fadd_rn(w[k], a.gain));
    *reinterpret_cast<uint4*>(a.out + (int64_t)row * 4096 + col) = pack_bf16x8(y);
  }
}

}  // namespace bnd
}  // namespace muse_topo
