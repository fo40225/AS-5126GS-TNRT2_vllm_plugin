// Flag-protocol kernels: payload stores, __syncthreads, release-stored per-block flags (see common.cuh).
// Used where the sentinel protocol's polling of the payload gets expensive (large messages) and for all-gather
// of big shards. Same arithmetic contract as ar_sll.cu (including the 0xFFFFFFFF canonicalization, so all
// algorithms produce identical bits for every input). The all-reduce kernel shares its receive regions with the
// sentinel kernels, so it restores the all-sentinel state after reading.
#include "ag_body.cuh"

namespace muse_topo {

// ----------------------------------------------------------------------------------------------
// Algorithm 2 with flags: block b owns packs [lo, hi) of every slice and runs the three phases on them.
// ----------------------------------------------------------------------------------------------
template <int D>
__global__ void __launch_bounds__(1024) ar2_flag_kernel(const Ctx c, const uint4* __restrict__ in,
                                                        uint4* __restrict__ out, int sp) {
  constexpr int EPW = Tr<D>::EPW;
  constexpr int EL = 4 * EPW;
  constexpr int PK = EPW;
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  const long long tag = static_cast<long long>(e) + 1;
  const int par = e & 1;
  const int ip = c.ipos, op = c.opos;
  const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
  const int cp = (sp + gridDim.x - 1) / gridDim.x;
  const int lo = min(sp, b * cp), hi = min(sp, lo + cp);
  char* mine = c.base[c.rank];
  const int64_t sa = c.L.a2_slot >> 4;
  const int64_t sc = c.L.c2_slot >> 4;
  uint4* rxa = reinterpret_cast<uint4*>(mine + c.L.a2_off + par * c.L.a2_par);
  uint4* rxb = reinterpret_cast<uint4*>(mine + c.L.b2_off + par * c.L.b2_par);
  uint4* rxc = reinterpret_cast<uint4*>(mine + c.L.c2_off + par * c.L.c2_par);
  uint4* txb = reinterpret_cast<uint4*>(c.base[c.partner] + c.L.b2_off + par * c.L.b2_par);

  // phase a: push the slices I do not own to their owners
  for (int i = lo + tid; i < hi; i += bd) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (j != op) {
        uint4* tx = reinterpret_cast<uint4*>(c.base[c.owners[j]] + c.L.a2_off + par * c.L.a2_par) + ip * sa;
        st_v4(tx + i, canon(in[(int64_t)j * sp + i]));
      }
    }
  }
  __syncthreads();
  if (tid < 4 && tid != op) st_release(flag_at(c, c.base[c.owners[tid]], kFlagA2a, b, c.rank), tag);
  // phase b: wait for the island peers, reduce my slice chunk, exchange with the partner
  if (tid < 4 && tid != ip) wait_flag(c, flag_at(c, mine, kFlagA2a, b, c.members[c.island][tid]), tag, 31);
  __syncthreads();
  for (int i = lo + tid; i < hi; i += bd) {
    float p[EL];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float t[EL];
      unpack_pack<D>(q == ip ? canon(in[(int64_t)op * sp + i]) : ld_v4(rxa + q * sa + i), t);
      if (q == 0) {
#pragma unroll
        for (int j = 0; j < EL; ++j) p[j] = t[j];
      } else {
#pragma unroll
        for (int j = 0; j < EL; ++j) p[j] = p[j] + t[j];
      }
    }
    uint32_t w[EL];
#pragma unroll
    for (int j = 0; j < EL; ++j) w[j] = canon(__float_as_uint(p[j]));
#pragma unroll
    for (int j = 0; j < PK; ++j)
      st_v4(txb + (int64_t)PK * i + j, make_uint4(w[4 * j], w[4 * j + 1], w[4 * j + 2], w[4 * j + 3]));
  }
  __syncthreads();
  if (tid == 0) st_release(flag_at(c, c.base[c.partner], kFlagA2b, b, c.rank), tag);
  if (tid == 1) wait_flag(c, flag_at(c, mine, kFlagA2b, b, c.partner), tag, 32);
  __syncthreads();
  for (int i = lo + tid; i < hi; i += bd) {
    // recompute my own partial (inputs are still in L2) and combine with the partner's
    float p[EL];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float t[EL];
      uint4 src;
      if (q == ip) {
        src = canon(in[(int64_t)op * sp + i]);
      } else {
        src = ld_v4(rxa + q * sa + i);
        st_v4(rxa + q * sa + i, sent4());
      }
      unpack_pack<D>(src, t);
      if (q == 0) {
#pragma unroll
        for (int j = 0; j < EL; ++j) p[j] = t[j];
      } else {
#pragma unroll
        for (int j = 0; j < EL; ++j) p[j] = p[j] + t[j];
      }
    }
    float res[EL];
#pragma unroll
    for (int j = 0; j < PK; ++j) {
      const uint4 r = ld_v4(rxb + (int64_t)PK * i + j);
      st_v4(rxb + (int64_t)PK * i + j, sent4());  // restore the sentinel state shared with the S-LL kernels
      const uint32_t qw[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float qv = __uint_as_float(qw[m]);
        const float pv = __uint_as_float(canon(__float_as_uint(p[4 * j + m])));
        res[4 * j + m] = (c.island == 0) ? (pv + qv) : (qv + pv);
      }
    }
    const uint4 y = canon(pack_pack<D>(res));
    out[(int64_t)op * sp + i] = y;
#pragma unroll
    for (int s = 1; s < 4; ++s) {
      const int peer = c.members[c.island][(ip + s) & 3];
      st_v4(reinterpret_cast<uint4*>(c.base[peer] + c.L.c2_off + par * c.L.c2_par) + op * sc + i, y);
    }
  }
  __syncthreads();
  if (tid < 4 && tid != ip) st_release(flag_at(c, c.base[c.members[c.island][tid]], kFlagA2c, b, c.rank), tag);
  // phase c: wait for the other owners and copy their slices
  if (tid < 4 && tid != op) wait_flag(c, flag_at(c, mine, kFlagA2c, b, c.owners[tid]), tag, 33);
  __syncthreads();
  for (int i = lo + tid; i < hi; i += bd) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (j != op) {
        out[(int64_t)j * sp + i] = ld_v4(rxc + j * sc + i);
        st_v4(rxc + j * sc + i, sent4());
      }
    }
  }
  epoch_end(c, e, old);
}

// ----------------------------------------------------------------------------------------------
// All-gather with flags (body in ag_body.cuh, shared with the chunked large all-gather).
// ----------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) ag_flag_kernel(const Ctx c, const uint4* __restrict__ in,
                                                       uint4* __restrict__ out, int np, int rpr, int layout) {
  uint32_t old;
  const uint64_t e = epoch_begin(c, old);
  ag_flag_body(c, blockIdx.x, gridDim.x, e, in, out, np, rpr, layout, 0, np);
  epoch_end(c, e, old);
}

void launch_ar2_flag(const Ctx& c, int dtype, const void* in, void* out, int64_t nbytes, int grid, int block,
                     cudaStream_t s) {
  const int sp = static_cast<int>(nbytes >> 6);
  const uint4* i = static_cast<const uint4*>(in);
  uint4* o = static_cast<uint4*>(out);
  switch (dtype) {
    case kBF16: launch_k(ar2_flag_kernel<kBF16>, c, grid, block, s, c, i, o, sp); break;
    case kFP16: launch_k(ar2_flag_kernel<kFP16>, c, grid, block, s, c, i, o, sp); break;
    default: launch_k(ar2_flag_kernel<kFP32>, c, grid, block, s, c, i, o, sp); break;
  }
}

void launch_ag_flag(const Ctx& c, const void* in, void* out, int64_t shard_bytes, int64_t row_bytes, int layout,
                    int grid, int block, cudaStream_t s) {
  const int np = static_cast<int>(shard_bytes >> 4);
  const int rpr = layout == 0 ? static_cast<int>(row_bytes >> 4) : 1;
  launch_k(ag_flag_kernel, c, grid, block, s, c, static_cast<const uint4*>(in), static_cast<uint4*>(out), np, rpr, layout);
}

}  // namespace muse_topo
