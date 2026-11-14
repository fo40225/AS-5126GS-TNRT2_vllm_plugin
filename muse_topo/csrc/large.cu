// Large messages (algorithms 4 and 5): chunked reduce-scatter, chunked all-gather and their fused pipeline step.
//
// The message is viewed as 8 equal blocks (block r = bytes [r*E, (r+1)*E), for a [T, H] tensor with T % 8 == 0 the
// rows of rank r). A message of any size is processed in chunks of C packs taken from every block, so every
// kernel launch stays inside the workspace of the small algorithms (no extra memory):
//   reduce-scatter chunk: rank r receives the reduced chunk of block r (algorithm 5, arithmetic of ar_sll.cu)
//   all-gather chunk:     every rank sends its chunk of block r to all others (flag protocol, ag_body.cuh)
//   all-reduce = reduce-scatter + all-gather of the same chunk (algorithm 4). The fused step kernel runs the
//   reduce-scatter of chunk c and the all-gather of chunk c-1 in one launch (different blocks), which keeps the
//   socket cut busy with the fp32 partials of one chunk while the finished results of the previous chunk fan out.
//
// Reduce-scatter dataflow (the 3-phase all-reduce of ar_sll.cu without phase c, owner j handles two blocks):
//   island rank with partner-order position j owns the blocks of ranks R0 = owners[j] and R1 = partner_of[R0].
//   (a) every island rank pushes its chunk of block R0 and of block R1 to the owner (sentinel slots)
//   (b) the owner forms the island partial P_I of both blocks (fp32, ascending island rank order), sends P_I[R1]
//       to its partner R1 (fp32: exact; or rounded to bf16 with bx = 1, the opt-in, bf16 inputs only) and combines the
//       partner's P of block R0 with its own: out = round(P_0 + P_1), island 0 first.
// Every element is therefore computed from exactly the expression of the small all-reduce, by one rank, and the
// result is distributed bit for bit by the all-gather: all_reduce == all_gather(reduce_scatter) bitwise, and equal to
// the small algorithms. With bx = 1 (opt-in, bf16 inputs only: fp16 partials above 65504 would become inf) BOTH partials
// are rounded to bf16 before the final add (symmetric, so the value of an element does not depend on which rank
// finalizes it). That makes the large algorithm batch invariant by itself, but the small algorithms (all-reduce below
// 2 MiB) never round the partials, so with bx an element's bits still depend on whether its message is served by the
// large algorithm or not.
//
// Sentinel/parity rules are those of ar_sll.cu: receive regions a2 (phase a) and b2 (partials) with the parity of
// the per-GPU call counter, slots reset to the sentinel by the reader. Reverse dependencies: phase a is all-to-all
// inside the island and the partner exchange is symmetric, so a sender can only start call k+2 after every
// receiver has finished call k.
#include <type_traits>

#include "ag_body.cuh"

namespace muse_topo {

// ----------------------------------------------------------------------------------------------
// reduce-scatter body over a virtual grid (block vb of nvb)
// in: chunk start of block 0, in_bs packs between consecutive blocks; out: my output chunk (np packs).
// ----------------------------------------------------------------------------------------------
template <int D, int U, bool BX, bool CUT>
__device__ __forceinline__ void rs_body(const Ctx& c, int vb, int nvb, uint32_t par, const uint4* __restrict__ in,
                                        int64_t in_bs, uint4* __restrict__ out, int np) {
  constexpr int EPW = Tr<D>::EPW;
  constexpr int EL = 4 * EPW;     // elements per 16-byte input pack
  constexpr int PK = BX ? 1 : EPW;  // exchanged packs per input pack
  const int ip = c.ipos;
  const int op = c.opos;
  char* mine = c.base[c.rank];
  const int64_t sa = c.L.a2_slot >> 4;  // packs per source slot: sub-block 0 at [0, np), sub-block 1 at [np, 2np)
  uint4* rxa = reinterpret_cast<uint4*>(mine + c.L.a2_off + par * c.L.a2_par);
  uint4* rxb = reinterpret_cast<uint4*>(mine + c.L.b2_off + par * c.L.b2_par);
  // Where the partial for the partner goes: straight into the partner's receive region (every worker thread stores
  // across the socket cut), or with CUT into a local staging area (c2 region, all-sentinel between launches) from
  // which the dedicated cut block (rs_cut_body) pushes it. Pack order inside the exchange area: [j][i] when stored
  // by the workers (a warp writes 512 contiguous bytes per instruction), [i][j] with CUT (the cut block then reads
  // and pushes in the order the partials are produced).
  uint4* txb = CUT ? reinterpret_cast<uint4*>(mine + c.L.c2_off + par * c.L.c2_par)
                   : reinterpret_cast<uint4*>(c.base[c.partner] + c.L.b2_off + par * c.L.b2_par);
  auto xi = [&](int i, int j) -> int64_t { return CUT ? (int64_t)PK * i + j : (int64_t)j * np + i; };

  const int stride = nvb * blockDim.x;
  for (int i0 = vb * blockDim.x + threadIdx.x; i0 < np; i0 += stride * U) {
    // phase a: push both sub-blocks of every owner I am not, keep my own two
    uint4 x[U][2];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      x[k][0] = sent4();
      x[k][1] = sent4();
      if (i < np) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int r0 = c.owners[j];
          const int r1 = c.partner_of[r0];
          const uint4 v0 = canon(in[(int64_t)r0 * in_bs + i]);
          const uint4 v1 = canon(in[(int64_t)r1 * in_bs + i]);
          if (j == op) {
            x[k][0] = v0;
            x[k][1] = v1;
          } else {
            uint4* tx = reinterpret_cast<uint4*>(c.base[r0] + c.L.a2_off + par * c.L.a2_par) + ip * sa;
            st_v4(tx + i, v0);
            st_v4(tx + np + i, v1);
          }
        }
      }
    }
    // phase b: wait for the three island peers (two sub-blocks each)
    uint4 v[U][4][2];
#pragma unroll
    for (int k = 0; k < U; ++k)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        v[k][q][0] = sent4();
        v[k][q][1] = sent4();
      }
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
#pragma unroll
            for (int s = 0; s < 2; ++s)
              if (q != ip && i < np && !full(v[k][q][s])) v[k][q][s] = ld_v4(rxa + q * sa + s * np + i);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int s = 0; s < 2; ++s)
              if (q != ip && i < np && !full(v[k][q][s])) ok = false;
        }
        if (!ok) spn.tick(c, 51);
      } while (!ok);
    }
    // island partial sums in fp32, fixed island rank order
    float p[U][2][EL];
#pragma unroll
    for (int k = 0; k < U; ++k)
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float t[EL];
          unpack_pack<D>(q == ip ? x[k][s] : v[k][q][s], t);
          if (q == 0) {
#pragma unroll
            for (int j = 0; j < EL; ++j) p[k][s][j] = t[j];
          } else {
#pragma unroll
            for (int j = 0; j < EL; ++j) p[k][s][j] = p[k][s][j] + t[j];
          }
        }
    // canonicalize the partials (sent AND used locally, like ar_sll.cu), send sub-block 1 to the partner
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      uint32_t w[2][EL];
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int j = 0; j < EL; ++j) {
          w[s][j] = canon(__float_as_uint(p[k][s][j]));
          p[k][s][j] = __uint_as_float(w[s][j]);
        }
      if constexpr (BX) {
        // opt-in (bf16 inputs): both partials are rounded to bf16 (the partner receives the rounded one)
        const uint4 q1 = canon(pack_pack<D>(p[k][1]));
        const uint4 q0 = canon(pack_pack<D>(p[k][0]));
        if (i < np) st_v4(txb + i, q1);
        unpack_pack<D>(q0, p[k][0]);
      } else if (i < np) {
#pragma unroll
        for (int j = 0; j < PK; ++j)
          st_v4(txb + xi(i, j), make_uint4(w[1][4 * j], w[1][4 * j + 1], w[1][4 * j + 2], w[1][4 * j + 3]));
      }
    }
    // reset the consumed phase-a slots (off the critical path)
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int s = 0; s < 2; ++s)
          if (q != ip && i < np) st_v4(rxa + q * sa + s * np + i, sent4());
    }
    // the partner's partial of my block
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
            if (i < np && !full(r[k][j])) r[k][j] = ld_v4(rxb + xi(i, j));
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int i = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < PK; ++j)
            if (i < np && !full(r[k][j])) ok = false;
        }
        if (!ok) spn.tick(c, 52);
      } while (!ok);
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int i = i0 + k * stride;
      if (i < np) {
        float res[EL];
        if constexpr (BX) {
          float qv[EL];
          unpack_pack<D>(r[k][0], qv);
#pragma unroll
          for (int m = 0; m < EL; ++m) res[m] = (c.island == 0) ? (p[k][0][m] + qv[m]) : (qv[m] + p[k][0][m]);
          st_v4(rxb + i, sent4());
        } else {
#pragma unroll
          for (int j = 0; j < PK; ++j) {
            const uint32_t qw[4] = {r[k][j].x, r[k][j].y, r[k][j].z, r[k][j].w};
#pragma unroll
            for (int m = 0; m < 4; ++m) {
              const float qv = __uint_as_float(qw[m]);
              const float pv = p[k][0][4 * j + m];
              res[4 * j + m] = (c.island == 0) ? (pv + qv) : (qv + pv);
            }
            st_v4(rxb + xi(i, j), sent4());
          }
        }
        out[i] = canon(pack_pack<D>(res));
      }
    }
  }
}

// ----------------------------------------------------------------------------------------------
// the dedicated cut block of the reduce-scatter (CUT): pushes the staged partials to the partner in production order
// ----------------------------------------------------------------------------------------------
template <int D, bool BX>
__device__ __forceinline__ void rs_cut_body(const Ctx& c, uint32_t par, int np) {
  constexpr int PK = BX ? 1 : Tr<D>::EPW;
  uint4* stg = reinterpret_cast<uint4*>(c.base[c.rank] + c.L.c2_off + par * c.L.c2_par);
  uint4* txb = reinterpret_cast<uint4*>(c.base[c.partner] + c.L.b2_off + par * c.L.b2_par);
  const int64_t total = (int64_t)PK * np;
  constexpr int Q = 4;  // packs in flight per thread (the poll loads are L2 round trips)
  const int64_t bd = blockDim.x;
  for (int64_t i0 = threadIdx.x; i0 < total; i0 += Q * bd) {
    uint4 v[Q];
    uint32_t got = 0;
    Spin spn;
    bool ok;
    do {
      ok = true;
#pragma unroll
      for (int q = 0; q < Q; ++q) {
        const int64_t i = i0 + q * bd;
        if (i < total && !((got >> q) & 1)) {
          v[q] = ld_v4(stg + i);
          if (full(v[q]))
            got |= 1u << q;
          else
            ok = false;
        }
      }
      if (!ok) spn.tick(c, 53);
    } while (!ok);
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const int64_t i = i0 + q * bd;
      if (i < total) {
        st_v4(txb + i, v[q]);
        st_v4(stg + i, sent4());
      }
    }
  }
}

// ----------------------------------------------------------------------------------------------
// kernels
// ----------------------------------------------------------------------------------------------
// Standalone reduce-scatter chunk. With CUT the last block is the cut block.
template <int D, int U, bool BX, bool CUT>
__global__ void __launch_bounds__(512) rs_kernel(const Ctx c, const uint4* __restrict__ in, int64_t in_bs,
                                                 uint4* __restrict__ out, int np) {
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  if (CUT && blockIdx.x == gridDim.x - 1)
    rs_cut_body<D, BX>(c, e & 1, np);
  else
    rs_body<D, U, BX, CUT>(c, blockIdx.x, gridDim.x - (CUT ? 1 : 0), e & 1, in, in_bs, out, np);
  epoch_end(c, e, old);
}

// Standalone all-gather chunk. With cut the last block pushes to the partner.
__global__ void __launch_bounds__(1024) ag_chunk_kernel(const Ctx c, const uint4* __restrict__ in,
                                                        uint4* __restrict__ out, int np, int rpr, int layout,
                                                        int64_t p0, int64_t np_total, int cut) {
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  if (cut && blockIdx.x == gridDim.x - 1)
    ag_cut_body(c, gridDim.x - 1, e, in, np);
  else
    ag_flag_body(c, blockIdx.x, gridDim.x - (cut ? 1 : 0), e, in, out, np, rpr, layout, p0, np_total, cut != 0);
  epoch_end(c, e, old);
}

// Fused pipeline step. Block roles: [0, nb_rs) reduce-scatter workers, then nb_ag all-gather workers, then the cut
// blocks: if CUT and nb_rs > 0 the reduce-scatter cut block, if cut_ag != 0 and nb_ag > 0 the all-gather cut block.
// cut_ag == 2 merges both into ONE block (the socket cut is the bottleneck of the pipeline and a single block gets 9.5
// GB/s across it where two blocks only get 8 and many get 6.8): it first pushes the all-gather chunk (always ready,
// per-block flags let the partner start forwarding early), then drains the reduce-scatter partials as the workers
// produce them.
template <int D, int U, bool BX, bool CUT>
__global__ void __launch_bounds__(512) step_kernel(const Ctx c, const StepArgs a) {
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  const int b = blockIdx.x;
  const int ncw = a.nb_rs + a.nb_ag;
  if (b < a.nb_rs) {
    if (b == 0) TRACE_ROLE(c, e, 0, 0);
    rs_body<D, U, BX, CUT>(c, b, a.nb_rs, e & 1, static_cast<const uint4*>(a.rs_in), a.rs_bs,
                           static_cast<uint4*>(a.rs_out), a.rs_np);
    if (b == 0) TRACE_ROLE(c, e, 0, 1);
  } else if (b < ncw) {
    if (b == a.nb_rs) TRACE_ROLE(c, e, 1, 0);
    ag_flag_body(c, b - a.nb_rs, a.nb_ag, e, static_cast<const uint4*>(a.ag_in), static_cast<uint4*>(a.ag_out),
                 a.ag_np, a.ag_rpr, a.ag_layout, a.ag_p0, a.ag_total, a.cut_ag != 0);
    if (b == a.nb_rs) TRACE_ROLE(c, e, 1, 1);
  } else {
    const bool rs_cut = CUT && a.nb_rs > 0;
    const bool merged = rs_cut && a.nb_ag > 0 && a.cut_ag == 2;
    if (rs_cut && b == ncw) {
      TRACE_ROLE(c, e, 2, 0);
      if (merged) ag_cut_body(c, a.nb_ag, e, static_cast<const uint4*>(a.ag_in), a.ag_np);
      rs_cut_body<D, BX>(c, e & 1, a.rs_np);
      TRACE_ROLE(c, e, 2, 1);
    } else {
      TRACE_ROLE(c, e, 3, 0);
      ag_cut_body(c, a.nb_ag, e, static_cast<const uint4*>(a.ag_in), a.ag_np);
      TRACE_ROLE(c, e, 3, 1);
    }
  }
  epoch_end(c, e, old);
}

// ----------------------------------------------------------------------------------------------
// host launchers
// ----------------------------------------------------------------------------------------------
// f(std::integral_constant<int, D>, std::integral_constant<int, U>, std::bool_constant<BX>, std::bool_constant<CUT>)
template <typename F>
static void dispatch_dubc(int dtype, int bx, int unroll, int cut, F&& f) {
  auto with_d = [&](auto D) {
    auto with_bx = [&](auto BX) {
      auto with_cut = [&](auto CUT) {
        if (unroll == 1)
          f(D, std::integral_constant<int, 1>{}, BX, CUT);
        else
          f(D, std::integral_constant<int, 2>{}, BX, CUT);
      };
      if (cut)
        with_cut(std::true_type{});
      else
        with_cut(std::false_type{});
    };
    if (bx && decltype(D)::value == kBF16)
      with_bx(std::true_type{});
    else
      with_bx(std::false_type{});
  };
  switch (dtype) {
    case kBF16: with_d(std::integral_constant<int, kBF16>{}); break;
    case kFP16: with_d(std::integral_constant<int, kFP16>{}); break;
    default: with_d(std::integral_constant<int, kFP32>{}); break;
  }
}

// grid = workers (+ 1 cut block)
void launch_rs(const Ctx& c, int dtype, int bx, int cut, const void* in, int64_t in_bs, void* out, int np, int grid,
               int block, int unroll, cudaStream_t s) {
  dispatch_dubc(dtype, bx, unroll, cut, [&](auto D, auto U, auto BX, auto CUT) {
    launch_k(rs_kernel<decltype(D)::value, decltype(U)::value, decltype(BX)::value, decltype(CUT)::value>, c,
             grid + (decltype(CUT)::value ? 1 : 0), block, s, c, static_cast<const uint4*>(in), in_bs,
             static_cast<uint4*>(out), np);
  });
}

void launch_ag_chunk(const Ctx& c, const void* in, void* out, int np, int rpr, int layout, int64_t p0,
                     int64_t np_total, int cut, int grid, int block, cudaStream_t s) {
  launch_k(ag_chunk_kernel, c, grid + (cut ? 1 : 0), block, s, c, static_cast<const uint4*>(in),
           static_cast<uint4*>(out), np, rpr, layout, p0, np_total, cut);
}

void launch_step(const Ctx& c, int dtype, int bx, int unroll, const StepArgs& a, int block, cudaStream_t s) {
  dispatch_dubc(dtype, bx, unroll, a.cut_rs, [&](auto D, auto U, auto BX, auto CUT) {
    const bool rs_cut = decltype(CUT)::value && a.nb_rs > 0;
    const bool ag_cut = a.cut_ag != 0 && a.nb_ag > 0;
    const bool merged = rs_cut && ag_cut && a.cut_ag == 2;
    const int grid = a.nb_rs + a.nb_ag + (rs_cut ? 1 : 0) + ((ag_cut && !merged) ? 1 : 0);
    launch_k(step_kernel<decltype(D)::value, decltype(U)::value, decltype(BX)::value, decltype(CUT)::value>, c, grid,
             block, s, c, a);
  });
}

}  // namespace muse_topo
