// All-reduce fused with the residual boundary of the Muse Spark decoder layer (boundary.cuh), bf16 [T, 8192], T <= 64.
//
// Same arithmetic and protocol as ar2_kernel (ar_sll.cu): slice j of every row belongs to island position j, phase a pushes
// the slices a rank does not own to their owners, phase b forms P_I of the own slice and exchanges it with the partner,
// phase c pushes the final slice to the island. What differs is the decomposition: one CTA of 256 threads handles one whole
// row (slice j = packs [256 j, 256 j + 256) of the row, thread i = pack i of every slice), so that at the end of phase c every
// thread holds packs 256 j + i of the row for j = 0..3, which is exactly the "virtual thread" layout of the Triton boundary
// kernel (boundary.cuh). The epilogue then runs the boundary on the registers: the all-reduced branch never goes through
// global memory, and the loads that do not depend on it (stream, gate coefficients, weights) are issued before the
// collective starts.
//
// The element-wise result of the all-reduce part is bit for bit the one of ar2_kernel (the slicing only decides who computes
// an element, the per-element arithmetic contract is unchanged); the epilogue is bit for bit the Triton kernel
// (bench/test_boundary.py, bench/test_ar_boundary.py).
#include "boundary.cuh"
#include "common.cuh"

namespace muse_topo {

template <bool POST_SCALE, bool FP32_OUT>
__global__ void __launch_bounds__(bnd::kRowThreads) ar2_bnd_kernel(const Ctx c, const uint4* __restrict__ in,
                                                                   const BndParams bp) {
  constexpr int PK = 2;  // fp32 partial packs per bf16 input pack
  __shared__ float sm[66];
  const bnd::Bnd8192 a(bp);
  const int row = blockIdx.x;
  const int i = threadIdx.x;
  const int idx = row * bnd::kRowThreads + i;  // pack index inside every slice buffer
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  const int par = e & 1;
  const int ip = c.ipos;
  const int op = c.opos;
  if (bp.ld_mode == 1 || bp.ld_mode == 3) bnd::bnd8192_prefetch_l2<POST_SCALE>(a, row);
  TRACE(c, 0);
  char* mine = c.base[c.rank];
  const int64_t sa = c.L.a2_slot >> 4;
  const int64_t sc = c.L.c2_slot >> 4;
  uint4* rxa = reinterpret_cast<uint4*>(mine + c.L.a2_off + par * c.L.a2_par);
  uint4* rxb = reinterpret_cast<uint4*>(mine + c.L.b2_off + par * c.L.b2_par);
  uint4* rxc = reinterpret_cast<uint4*>(mine + c.L.c2_off + par * c.L.c2_par);
  uint4* txb = reinterpret_cast<uint4*>(c.base[c.partner] + c.L.b2_off + par * c.L.b2_par);

  bnd::Bnd8192Regs regs;

  // phase a: send the slices I do not own, keep my own
  uint4 x = sent4();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 xv = canon(in[(int64_t)row * 1024 + bnd::kRowThreads * j + i]);
    if (j == op) {
      x = xv;
    } else {
      uint4* tx = reinterpret_cast<uint4*>(c.base[c.owners[j]] + c.L.a2_off + par * c.L.a2_par) + ip * sa;
      st_v4(tx + idx, xv);
    }
  }
  // The operands of the epilogue that do not depend on the collective (about 128 KiB for a row) are brought in while the collective
  // is in flight; see BndParams::ld_mode for the placement. Loaded ahead of the pushes they would queue the pushes behind
  // them in the load/store unit.
  if (bp.ld_mode == 0 || bp.ld_mode == 3) bnd::bnd8192_load<POST_SCALE>(a, row, regs);
  TRACE(c, 1);
  // phase b: the three island peers' contributions to my slice
  uint4 v[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = sent4();
  {
    Spin spn;
    bool ok;
    do {
      ok = true;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (q != ip && !full(v[q])) v[q] = ld_v4(rxa + q * sa + idx);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (q != ip && !full(v[q])) ok = false;
      if (!ok) spn.tick(c, 31);
    } while (!ok);
  }
  TRACE(c, 2);
  float p[8];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float t[8];
    unpack_pack<kBF16>(q == ip ? x : v[q], t);
    if (q == 0) {
#pragma unroll
      for (int k = 0; k < 8; ++k) p[k] = t[k];
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) p[k] = p[k] + t[k];
    }
  }
  {
    uint32_t w[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      w[k] = canon(__float_as_uint(p[k]));
      p[k] = __uint_as_float(w[k]);
    }
#pragma unroll
    for (int j = 0; j < PK; ++j)
      st_v4(txb + (int64_t)PK * idx + j, make_uint4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]));
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (q != ip) st_v4(rxa + q * sa + idx, sent4());
  }
  TRACE(c, 3);
  uint4 r[PK];
#pragma unroll
  for (int j = 0; j < PK; ++j) r[j] = sent4();
  {
    Spin spn;
    bool ok;
    do {
      ok = true;
#pragma unroll
      for (int j = 0; j < PK; ++j)
        if (!full(r[j])) r[j] = ld_v4(rxb + (int64_t)PK * idx + j);
#pragma unroll
      for (int j = 0; j < PK; ++j)
        if (!full(r[j])) ok = false;
      if (!ok) spn.tick(c, 32);
    } while (!ok);
  }
  TRACE(c, 4);
  // final value of my slice, kept locally and pushed to the island peers
  uint4 bx[4];
  {
    float res[8];
#pragma unroll
    for (int j = 0; j < PK; ++j) {
      const uint32_t qw[4] = {r[j].x, r[j].y, r[j].z, r[j].w};
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float qv = __uint_as_float(qw[m]);
        const float pv = p[4 * j + m];
        res[4 * j + m] = (c.island == 0) ? (pv + qv) : (qv + pv);
      }
      st_v4(rxb + (int64_t)PK * idx + j, sent4());
    }
    const uint4 y = canon(pack_pack<kBF16>(res));
#pragma unroll
    for (int s = 1; s < 4; ++s) {
      const int peer = c.members[c.island][(ip + s) & 3];
      uint4* tx = reinterpret_cast<uint4*>(c.base[peer] + c.L.c2_off + par * c.L.c2_par) + op * sc;
      st_v4(tx + idx, y);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) bx[j] = (j == op) ? y : sent4();
  }
  TRACE(c, 5);
  trigger_stage(c, 2);
  // phase c: the other three slices from their owners
  {
    uint32_t got = 1u << op;
    Spin spn;
    bool ok;
    do {
      ok = true;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (!((got >> j) & 1)) {
          const uint4 z = ld_v4(rxc + j * sc + idx);
          if (full(z)) {
            bx[j] = z;
            st_v4(rxc + j * sc + idx, sent4());
            got |= 1u << j;
          } else {
            ok = false;
          }
        }
      }
      if (!ok) spn.tick(c, 33);
    } while (!ok);
  }
  TRACE(c, 6);
  trigger_stage(c, 3);
  if (bp.ld_mode == 1 || bp.ld_mode == 2) bnd::bnd8192_load<POST_SCALE>(a, row, regs);
  bnd::bnd8192_apply<POST_SCALE, FP32_OUT>(a, row, bx, regs, sm, c.trace);
  TRACE(c, 10);
  epoch_end(c, e, old);
}

void launch_ar2_bnd(const Ctx& c, const void* in, const BndParams& a, int rows, cudaStream_t s) {
  const uint4* i = static_cast<const uint4*>(in);
  const dim3 grid(rows), block(bnd::kRowThreads);
  const bool ps = a.post_w != nullptr, f32 = a.out_fp32 != nullptr;
  if (ps && f32) launch_k(ar2_bnd_kernel<true, true>, c, grid, block, s, c, i, a);
  else if (ps) launch_k(ar2_bnd_kernel<true, false>, c, grid, block, s, c, i, a);
  else if (f32) launch_k(ar2_bnd_kernel<false, true>, c, grid, block, s, c, i, a);
  else launch_k(ar2_bnd_kernel<false, false>, c, grid, block, s, c, i, a);
}

}  // namespace muse_topo
