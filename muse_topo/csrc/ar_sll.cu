// All-reduce kernels with the sentinel protocol (see common.cuh), algorithms 1 and 2 of the spec.
//
// Arithmetic contract (identical in both kernels, for every element e):
//   P_I[e] = ((x_{m0}[e] + x_{m1}[e]) + x_{m2}[e]) + x_{m3}[e]     in fp32, m0..m3 = ranks of island I, ascending
//   out[e] = round_to_dtype(P_0[e] + P_1[e])                          in fp32, island 0 first
// Inputs are widened exactly, the only roundings are the fp32 adds and the final conversion.
// Words that would equal the sentinel are replaced by 0xFFFFFFFE before they are sent AND before they are used
// locally, so every rank computes from identical bits (only NaN payloads can change).
#include "common.cuh"

namespace muse_topo {

// ----------------------------------------------------------------------------------------------
// Algorithm 1: 2-phase. (a) push the whole input to the 3 island peers, (b) every rank forms P_I for the whole
// message, pushes it (fp32) to its partner and combines with the partner's P.
// ----------------------------------------------------------------------------------------------
template <int D, int U>
__global__ void __launch_bounds__(512) ar1_kernel(const Ctx c, const uint4* __restrict__ in, uint4* __restrict__ out,
                                                  int np) {
  constexpr int EPW = Tr<D>::EPW;
  constexpr int EL = 4 * EPW;  // elements per 16-byte pack
  constexpr int PK = EPW;      // fp32 partial packs per input pack
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  const int par = e & 1;
  const int ip = c.ipos;
  char* mine = c.base[c.rank];
  const int64_t slot_p = c.L.a1_slot >> 4;
  uint4* rxa = reinterpret_cast<uint4*>(mine + c.L.a1_off + par * c.L.a1_par);
  uint4* rxb = reinterpret_cast<uint4*>(mine + c.L.b1_off + par * c.L.b1_par);
  uint4* txa[3];
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int peer = c.members[c.island][(ip + 1 + k) & 3];
    txa[k] = reinterpret_cast<uint4*>(c.base[peer] + c.L.a1_off + par * c.L.a1_par) + ip * slot_p;
  }
  uint4* txb = reinterpret_cast<uint4*>(c.base[c.partner] + c.L.b1_off + par * c.L.b1_par);

  TRACE(c, 0);
  const int stride = gridDim.x * blockDim.x;
  for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < np; i0 += stride * U) {
    uint4 x[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      x[k] = (i < np) ? canon(in[i]) : sent4();
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      if (i < np) {
        st_v4(txa[0] + i, x[k]);
        st_v4(txa[1] + i, x[k]);
        st_v4(txa[2] + i, x[k]);
      }
    }
    TRACE(c, 1);
    // wait for the three island peers
    uint4 v[U][4];
#pragma unroll
    for (int k = 0; k < U; ++k)
#pragma unroll
      for (int q = 0; q < 4; ++q) v[k][q] = sent4();
    {
      Spin sp;
      bool ok;
      do {
        ok = true;
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int q = 0; q < 4; ++q)
            if (q != ip && i < np && !full(v[k][q])) v[k][q] = ld_v4(rxa + q * slot_p + i);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int q = 0; q < 4; ++q)
            if (q != ip && i < np && !full(v[k][q])) ok = false;
        }
        if (!ok) sp.tick(c, 1);
      } while (!ok);
    }
    TRACE(c, 2);
    // island partial sum in fp32, fixed rank order
    float p[U][EL];
#pragma unroll
    for (int k = 0; k < U; ++k) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float t[EL];
        unpack_pack<D>(q == ip ? x[k] : v[k][q], t);
        if (q == 0) {
#pragma unroll
          for (int j = 0; j < EL; ++j) p[k][j] = t[j];
        } else {
#pragma unroll
          for (int j = 0; j < EL; ++j) p[k][j] = p[k][j] + t[j];
        }
      }
    }
    // send the partial to the partner
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      uint32_t w[EL];
#pragma unroll
      for (int j = 0; j < EL; ++j) {
        w[j] = canon(__float_as_uint(p[k][j]));
        p[k][j] = __uint_as_float(w[j]);
      }
      if (i < np) {
#pragma unroll
        for (int j = 0; j < PK; ++j)
          st_v4(txb + (int64_t)PK * i + j, make_uint4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]));
      }
    }
    TRACE(c, 3);
    // reset the consumed phase-a slots (off the critical path)
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (q != ip && i < np) st_v4(rxa + q * slot_p + i, sent4());
    }
    uint4 r[U][PK];
#pragma unroll
    for (int k = 0; k < U; ++k)
#pragma unroll
      for (int j = 0; j < PK; ++j) r[k][j] = sent4();
    {
      Spin sp;
      bool ok;
      do {
        ok = true;
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < PK; ++j)
            if (i < np && !full(r[k][j])) r[k][j] = ld_v4(rxb + (int64_t)PK * i + j);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < PK; ++j)
            if (i < np && !full(r[k][j])) ok = false;
        }
        if (!ok) sp.tick(c, 2);
      } while (!ok);
    }
    TRACE(c, 4);
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      if (i < np) {
        float res[EL];
#pragma unroll
        for (int j = 0; j < PK; ++j) {
          const uint32_t qw[4] = {r[k][j].x, r[k][j].y, r[k][j].z, r[k][j].w};
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const float qv = __uint_as_float(qw[m]);
            const float pv = p[k][4 * j + m];
            res[4 * j + m] = (c.island == 0) ? (pv + qv) : (qv + pv);
          }
          st_v4(rxb + (int64_t)PK * i + j, sent4());
        }
        out[i] = canon(pack_pack<D>(res));
      }
    }
  }
  TRACE(c, 5);
  epoch_end(c, e, old);
}

// ----------------------------------------------------------------------------------------------
// Algorithm 2: 3-phase reduce-scatter style. Slice j of the message (np/4 packs) belongs to island rank
// owners[j]. (a) push slice j to its owner, (b) the owner forms P_I[slice], exchanges it (fp32) with its partner
// (same slice index in the other island) and computes out[slice], (c) the owner pushes out[slice] to the island.
// BX (bf16 inputs only, the opt-in bf16 exchange of large.cu): both island partials are rounded to bf16 before the
// final add, so the partner exchange carries one 16 B pack per input pack instead of two and the socket cut, the
// bottleneck of the medium sizes, carries half the bytes. Every element is computed exactly like the BX path of the
// large algorithm (large.cu: rs_body), so the two give the same bits.
// ----------------------------------------------------------------------------------------------
template <int D, int U, bool BX>
__global__ void __launch_bounds__(512) ar2_kernel(const Ctx c, const uint4* __restrict__ in, uint4* __restrict__ out,
                                                  int sp) {
  constexpr int EPW = Tr<D>::EPW;
  constexpr int EL = 4 * EPW;
  constexpr int PK = BX ? 1 : EPW;  // exchanged packs per input pack
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  const int par = e & 1;
  const int ip = c.ipos;
  const int op = c.opos;
  char* mine = c.base[c.rank];
  const int64_t sa = c.L.a2_slot >> 4;
  const int64_t sc = c.L.c2_slot >> 4;
  uint4* rxa = reinterpret_cast<uint4*>(mine + c.L.a2_off + par * c.L.a2_par);
  uint4* rxb = reinterpret_cast<uint4*>(mine + c.L.b2_off + par * c.L.b2_par);
  uint4* rxc = reinterpret_cast<uint4*>(mine + c.L.c2_off + par * c.L.c2_par);
  uint4* txb = reinterpret_cast<uint4*>(c.base[c.partner] + c.L.b2_off + par * c.L.b2_par);

  TRACE(c, 0);
  const int stride = gridDim.x * blockDim.x;
  for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < sp; i0 += stride * U) {
    // phase a: send the slices I do not own, keep my own
    uint4 x[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      x[k] = sent4();
      if (i < sp) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint4 xv = canon(in[(int64_t)j * sp + i]);
          if (j == op) {
            x[k] = xv;
          } else {
            uint4* tx = reinterpret_cast<uint4*>(c.base[c.owners[j]] + c.L.a2_off + par * c.L.a2_par) + ip * sa;
            st_v4(tx + i, xv);
          }
        }
      }
    }
    TRACE(c, 1);
    // phase b: wait for the three island peers' contributions to my slice
    uint4 v[U][4];
#pragma unroll
    for (int k = 0; k < U; ++k)
#pragma unroll
      for (int q = 0; q < 4; ++q) v[k][q] = sent4();
    {
      Spin spn;
      bool ok;
      do {
        ok = true;
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int q = 0; q < 4; ++q)
            if (q != ip && i < sp && !full(v[k][q])) v[k][q] = ld_v4(rxa + q * sa + i);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int q = 0; q < 4; ++q)
            if (q != ip && i < sp && !full(v[k][q])) ok = false;
        }
        if (!ok) spn.tick(c, 11);
      } while (!ok);
    }
    TRACE(c, 2);
    float p[U][EL];
#pragma unroll
    for (int k = 0; k < U; ++k) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float t[EL];
        unpack_pack<D>(q == ip ? x[k] : v[k][q], t);
        if (q == 0) {
#pragma unroll
          for (int j = 0; j < EL; ++j) p[k][j] = t[j];
        } else {
#pragma unroll
          for (int j = 0; j < EL; ++j) p[k][j] = p[k][j] + t[j];
        }
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      if constexpr (BX) {
        // both partials are rounded to bf16 (the partner receives the rounded one, I keep the rounded one); the fp32
        // canonicalization comes first, exactly as in rs_body of large.cu
#pragma unroll
        for (int j = 0; j < EL; ++j) p[k][j] = __uint_as_float(canon(__float_as_uint(p[k][j])));
        const uint4 q = canon(pack_pack<D>(p[k]));
        if (i < sp) st_v4(txb + i, q);
        unpack_pack<D>(q, p[k]);
      } else {
        uint32_t w[EL];
#pragma unroll
        for (int j = 0; j < EL; ++j) {
          w[j] = canon(__float_as_uint(p[k][j]));
          p[k][j] = __uint_as_float(w[j]);
        }
        if (i < sp) {
#pragma unroll
          for (int j = 0; j < PK; ++j)
            st_v4(txb + (int64_t)PK * i + j, make_uint4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]));
        }
      }
      if (i < sp) {
#pragma unroll
        for (int q = 0; q < 4; ++q)
          if (q != ip) st_v4(rxa + q * sa + i, sent4());
      }
    }
    TRACE(c, 3);
    uint4 r[U][PK];
#pragma unroll
    for (int k = 0; k < U; ++k)
#pragma unroll
      for (int j = 0; j < PK; ++j) r[k][j] = sent4();
    {
      Spin spn;
      bool ok;
      do {
        ok = true;
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < PK; ++j)
            if (i < sp && !full(r[k][j])) r[k][j] = ld_v4(rxb + (int64_t)PK * i + j);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < PK; ++j)
            if (i < sp && !full(r[k][j])) ok = false;
        }
        if (!ok) spn.tick(c, 12);
      } while (!ok);
    }
    TRACE(c, 4);
    // final value of my slice, kept locally and pushed to the island peers
    uint4 y[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      y[k] = sent4();
      if (i < sp) {
        float res[EL];
        if constexpr (BX) {
          float qv[EL];
          unpack_pack<D>(r[k][0], qv);
#pragma unroll
          for (int m = 0; m < EL; ++m) res[m] = (c.island == 0) ? (p[k][m] + qv[m]) : (qv[m] + p[k][m]);
          st_v4(rxb + i, sent4());
        } else {
#pragma unroll
          for (int j = 0; j < PK; ++j) {
            const uint32_t qw[4] = {r[k][j].x, r[k][j].y, r[k][j].z, r[k][j].w};
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              const float qv = __uint_as_float(qw[m]);
              const float pv = p[k][4 * j + m];
              res[4 * j + m] = (c.island == 0) ? (pv + qv) : (qv + pv);
            }
            st_v4(rxb + (int64_t)PK * i + j, sent4());
          }
        }
        y[k] = canon(pack_pack<D>(res));
        out[(int64_t)op * sp + i] = y[k];
#pragma unroll
        for (int s = 1; s < 4; ++s) {
          const int peer = c.members[c.island][(ip + s) & 3];
          uint4* tx = reinterpret_cast<uint4*>(c.base[peer] + c.L.c2_off + par * c.L.c2_par) + op * sc;
          st_v4(tx + i, y[k]);
        }
      }
    }
    TRACE(c, 5);
    trigger_stage(c, 2);
    // phase c: receive the other three slices from their owners; each pack is stored to the output and its slot
    // reset as soon as it has arrived, so little work is left after the last arrival
    uint32_t got[U];
#pragma unroll
    for (int k = 0; k < U; ++k) got[k] = 0;
    {
      Spin spn;
      bool ok;
      do {
        uint4 z[U][4];
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (j != op && i < sp && !((got[k] >> j) & 1)) z[k][j] = ld_v4(rxc + j * sc + i);
        }
        ok = true;
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            if (j != op && i < sp && !((got[k] >> j) & 1)) {
              if (full(z[k][j])) {
                out[(int64_t)j * sp + i] = z[k][j];
                st_v4(rxc + j * sc + i, sent4());
                got[k] |= 1u << j;
              } else {
                ok = false;
              }
            }
          }
        }
        if (!ok) spn.tick(c, 13);
      } while (!ok);
    }
  }
  TRACE(c, 6);
  trigger_stage(c, 3);
  epoch_end(c, e, old);
}

// ----------------------------------------------------------------------------------------------
// host launchers
// ----------------------------------------------------------------------------------------------
template <int D, int U>
static void run_ar1(const Ctx& c, const void* in, void* out, int np, int grid, int block, cudaStream_t s) {
  launch_k(ar1_kernel<D, U>, c, grid, block, s, c, static_cast<const uint4*>(in), static_cast<uint4*>(out), np);
}
template <int D, int U, bool BX>
static void run_ar2(const Ctx& c, const void* in, void* out, int sp, int grid, int block, cudaStream_t s) {
  launch_k(ar2_kernel<D, U, BX>, c, grid, block, s, c, static_cast<const uint4*>(in), static_cast<uint4*>(out), sp);
}
template <int D, int U>
static void run_ar2_exact(const Ctx& c, const void* in, void* out, int sp, int grid, int block, cudaStream_t s) {
  run_ar2<D, U, false>(c, in, out, sp, grid, block, s);
}

#define DISPATCH_DU(FN, ...)                                \
  switch (dtype) {                                          \
    case kBF16:                                             \
      switch (unroll) {                                     \
        case 1: FN<kBF16, 1>(__VA_ARGS__); break;           \
        case 2: FN<kBF16, 2>(__VA_ARGS__); break;           \
        default: FN<kBF16, 4>(__VA_ARGS__); break;          \
      }                                                     \
      break;                                                \
    case kFP16:                                             \
      switch (unroll) {                                     \
        case 1: FN<kFP16, 1>(__VA_ARGS__); break;           \
        case 2: FN<kFP16, 2>(__VA_ARGS__); break;           \
        default: FN<kFP16, 4>(__VA_ARGS__); break;          \
      }                                                     \
      break;                                                \
    default:                                                \
      switch (unroll) {                                     \
        case 1: FN<kFP32, 1>(__VA_ARGS__); break;           \
        case 2: FN<kFP32, 2>(__VA_ARGS__); break;           \
        default: FN<kFP32, 4>(__VA_ARGS__); break;          \
      }                                                     \
      break;                                                \
  }

void launch_ar1(const Ctx& c, int dtype, const void* in, void* out, int64_t nbytes, int grid, int block, int unroll,
                cudaStream_t s) {
  const int np = static_cast<int>(nbytes >> 4);
  DISPATCH_DU(run_ar1, c, in, out, np, grid, block, s);
}

void launch_ar2(const Ctx& c, int dtype, const void* in, void* out, int64_t nbytes, int grid, int block, int unroll,
                int bx, cudaStream_t s) {
  const int sp = static_cast<int>(nbytes >> 6);  // packs per slice
  if (bx && dtype == kBF16) {
    switch (unroll) {
      case 1: run_ar2<kBF16, 1, true>(c, in, out, sp, grid, block, s); break;
      case 2: run_ar2<kBF16, 2, true>(c, in, out, sp, grid, block, s); break;
      default: run_ar2<kBF16, 4, true>(c, in, out, sp, grid, block, s); break;
    }
    return;
  }
  DISPATCH_DU(run_ar2_exact, c, in, out, sp, grid, block, s);
}

}  // namespace muse_topo
