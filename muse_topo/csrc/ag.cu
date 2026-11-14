// All-gather kernels (algorithm 3): pure data movement, bit exact for every payload value.
//
// 2 hops: every rank pushes its shard to its 3 island peers and to its partner; the partner forwards the shard
// to its own 3 island peers. The receive slot of the data that originates at rank o is slot o on every
// destination, so each slot has exactly one writer (the origin, or the origin's partner for the forwarded copy).
//
// LL8 variant: 8-byte units {data, epoch tag}; the receiver polls the units. The output tensor is written by
// the receiver itself in either layout:
//   layout 0 (dim = -1): out[t, o*w + c] = shard_o[t, c]      out row = 8 * wpr words
//   layout 1 (dim =  0): out[o*T + t, c] = shard_o[t, c]      (rank-major, NCCL layout)
#include "common.cuh"

namespace muse_topo {

// Output position of word w of origin o: pbase + o * ostride (one integer division per word, issued early).
__device__ __forceinline__ void ag_base(int w, int nw, int wpr, int layout, int64_t& pbase, int64_t& ostride) {
  if (layout == 1) {
    pbase = w;
    ostride = nw;
  } else {
    const int t = w / wpr;
    pbase = (int64_t)t * (kWorld * wpr) + (w - t * wpr);
    ostride = wpr;
  }
}

template <int U>
__global__ void __launch_bounds__(512) ag_ll_kernel(const Ctx c, const uint32_t* __restrict__ in,
                                                    uint32_t* __restrict__ out, int nw, int wpr, int layout) {
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
  uint2* txi[3];  // my own shard to the island peers
  uint2* fwd[3];  // the partner's shard forwarded to the island peers
  int org[6];     // origins of the six words received in the last stage: 3 direct, 3 forwarded
#pragma unroll
  for (int s = 0; s < 3; ++s) {
    const int peer = c.members[c.island][(ip + 1 + s) & 3];
    uint2* pb = reinterpret_cast<uint2*>(c.base[peer] + c.L.g_off + par * c.L.g_par);
    txi[s] = pb + me * gs;
    fwd[s] = pb + pr * gs;
    org[s] = peer;
    org[3 + s] = c.partner_of[peer];
  }

  TRACE(c, 0);
  const int stride = gridDim.x * blockDim.x;
  for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < nw; i0 += stride * U) {
    uint32_t x[U];
    int64_t pbase[U], ostride[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int w = i0 + k * stride;
      x[k] = 0;
      pbase[k] = ostride[k] = 0;
      if (w < nw) {
        x[k] = in[w];
        st_v2(txp + w, x[k], tag);
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int w = i0 + k * stride;
      if (w < nw) {
        st_v2(txi[0] + w, x[k], tag);
        st_v2(txi[1] + w, x[k], tag);
        st_v2(txi[2] + w, x[k], tag);
        ag_base(w, nw, wpr, layout, pbase[k], ostride[k]);
        out[pbase[k] + me * ostride[k]] = x[k];
      }
    }
    TRACE(c, 1);
    // the partner's shard: keep it and forward it
    uint2 vp[U];
#pragma unroll
    for (int k = 0; k < U; ++k) vp[k] = make_uint2(0, ~tag);
    {
      Spin sp;
      bool ok;
      do {
        ok = true;
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int w = i0 + k * stride;
          if (w < nw && vp[k].y != tag) vp[k] = ld_v2(rx + pr * gs + w);
        }
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int w = i0 + k * stride;
          if (w < nw && vp[k].y != tag) ok = false;
        }
        if (!ok) sp.tick(c, 21);
      } while (!ok);
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int w = i0 + k * stride;
      if (w < nw) {
        st_v2(fwd[0] + w, vp[k].x, tag);
        st_v2(fwd[1] + w, vp[k].x, tag);
        st_v2(fwd[2] + w, vp[k].x, tag);
        out[pbase[k] + pr * ostride[k]] = vp[k].x;
      }
    }
    TRACE(c, 2);
    // three direct shards from the island peers and three forwarded shards (their partners'), each stored to the
    // output as soon as it arrives
    uint32_t got[U];
#pragma unroll
    for (int k = 0; k < U; ++k) got[k] = 0;
    {
      Spin sp;
      bool ok;
      do {
        uint2 v[U][6];
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int w = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < 6; ++j)
            if (w < nw && !((got[k] >> j) & 1)) v[k][j] = ld_v2(rx + org[j] * gs + w);
        }
        ok = true;
#pragma unroll
        for (int k = 0; k < U; ++k) {
          const int w = i0 + k * stride;
#pragma unroll
          for (int j = 0; j < 6; ++j) {
            if (w < nw && !((got[k] >> j) & 1)) {
              if (v[k][j].y == tag) {
                out[pbase[k] + org[j] * ostride[k]] = v[k][j].x;
                got[k] |= 1u << j;
              } else {
                ok = false;
              }
            }
          }
        }
        if (!ok) sp.tick(c, 22);
      } while (!ok);
    }
    TRACE(c, 3);
  }
  TRACE(c, 4);
  epoch_end(c, e, old);
}

void launch_ag_ll(const Ctx& c, const void* in, void* out, int64_t shard_bytes, int64_t row_bytes, int layout,
                  int grid, int block, int unroll, cudaStream_t s) {
  const int nw = static_cast<int>(shard_bytes >> 2);
  const int wpr = static_cast<int>(row_bytes >> 2);
  const uint32_t* i = static_cast<const uint32_t*>(in);
  uint32_t* o = static_cast<uint32_t*>(out);
  switch (unroll) {
    case 1: launch_k(ag_ll_kernel<1>, c, grid, block, s, c, i, o, nw, wpr, layout); break;
    case 2: launch_k(ag_ll_kernel<2>, c, grid, block, s, c, i, o, nw, wpr, layout); break;
    default: launch_k(ag_ll_kernel<4>, c, grid, block, s, c, i, o, nw, wpr, layout); break;
  }
}

}  // namespace muse_topo
