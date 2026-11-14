// Host/device shared types of the muse_topo collectives (plain C++, included by bindings.cpp and the kernels).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace muse_topo {

constexpr int kWorld = 8;
constexpr int kIsland = 4;
constexpr int kMaxBlocks = 128;  // grid limit of the flag-protocol kernels (one flag column per block)
constexpr int kFlagKinds = 16;
constexpr uint32_t kSent = 0xFFFFFFFFu;

enum Dtype : int { kBF16 = 0, kFP16 = 1, kFP32 = 2 };

// Byte offsets inside the per-rank workspace (identical on every rank). Computed once on the host.
struct WsLayout {
  // allreduce, 2-phase (algorithm 1): [parity][4 source positions][a1_slot]; partial sums [parity][b1_par]
  int64_t a1_off, a1_par, a1_slot;
  int64_t b1_off, b1_par;
  // allreduce, 3-phase (algorithm 2): phase a [parity][4][a2_slot], partials [parity][b2_par], phase c [parity][4][c2_slot]
  int64_t a2_off, a2_par, a2_slot;
  int64_t b2_off, b2_par;
  int64_t c2_off, c2_par, c2_slot;
  // all-gather LL8: [parity][8 origins][g_slot]
  int64_t g_off, g_par, g_slot;
  // flag-protocol flags: int64[kind][kMaxBlocks][8 source ranks], kind < kFlagKinds
  int64_t f_off;
  // all-gather flag protocol: [parity][8 origins][h_slot]
  int64_t h_off, h_par, h_slot;
  int64_t total;
};

struct Ctx {
  int rank, island, ipos, opos, partner;
  int pdl;  // 1: launch with programmatic stream serialization (kernels then start with griddepcontrol.wait)
  int members[2][kIsland];  // ranks of each island in arithmetic (ascending rank) order
  int owners[kIsland];      // my island: owners[j] = rank that owns slice j (partner order)
  int partner_of[kWorld];
  char* base[kWorld];       // base[r] = peer r's workspace (base[rank] is local)
  uint32_t* ctl;            // local: ctl[0..1] 64-bit call counter, ctl[32] arrival counter of the running launch
  volatile int* err;        // mapped host word, set before a spin timeout traps
  unsigned long long timeout_ns;
  unsigned long long* trace;  // debug: mapped host stamps (only with -DMUSE_TOPO_TRACE), else nullptr
  WsLayout L;
};

// Host-side workspace layout for the given limits (bytes of the largest message handled by each algorithm).
inline int64_t ws_align(int64_t v) { return (v + 255) & ~int64_t(255); }
inline WsLayout make_layout(int64_t max_bytes, int64_t a1_max, int64_t g_max, int64_t h_max) {
  WsLayout L{};
  int64_t off = 0;
  L.f_off = off;
  off += ws_align(int64_t(kFlagKinds) * kMaxBlocks * kWorld * 8);
  L.a1_slot = ws_align(a1_max);
  L.a1_par = 4 * L.a1_slot;
  L.a1_off = off;
  off += 2 * L.a1_par;
  L.b1_par = ws_align(2 * a1_max);
  L.b1_off = off;
  off += 2 * L.b1_par;
  const int64_t sl = ws_align(max_bytes / 4);
  L.a2_slot = sl;
  L.a2_par = 4 * sl;
  L.a2_off = off;
  off += 2 * L.a2_par;
  L.b2_par = ws_align(max_bytes / 2);
  L.b2_off = off;
  off += 2 * L.b2_par;
  L.c2_slot = sl;
  L.c2_par = 4 * sl;
  L.c2_off = off;
  off += 2 * L.c2_par;
  L.g_slot = ws_align(2 * g_max);
  L.g_par = kWorld * L.g_slot;
  L.g_off = off;
  off += 2 * L.g_par;
  L.h_slot = ws_align(h_max);
  L.h_par = kWorld * L.h_slot;
  L.h_off = off;
  off += 2 * L.h_par;
  L.total = off;
  return L;
}

// Launch entry points (implemented in the .cu files, called from bindings.cpp).
void launch_ar1(const Ctx& c, int dtype, const void* in, void* out, int64_t nbytes, int grid, int block,
                int unroll, cudaStream_t s);
void launch_ar2(const Ctx& c, int dtype, const void* in, void* out, int64_t nbytes, int grid, int block,
                int unroll, cudaStream_t s);
void launch_ag_ll(const Ctx& c, const void* in, void* out, int64_t shard_bytes, int64_t row_bytes, int layout,
                  int grid, int block, int unroll, cudaStream_t s);
void launch_ar2_flag(const Ctx& c, int dtype, const void* in, void* out, int64_t nbytes, int grid, int block,
                     cudaStream_t s);
void launch_ag_flag(const Ctx& c, const void* in, void* out, int64_t shard_bytes, int64_t row_bytes, int layout,
                    int grid, int block, cudaStream_t s);

// Large messages (large.cu): chunked reduce-scatter / all-gather and the fused all-reduce step.
// One kernel launch handles one chunk of the 8 blocks of the message. Arguments in 16 B packs unless noted.
struct StepArgs {
  // reduce-scatter role (blocks [0, nb_rs)): in = chunk start of block 0, in_bs = packs between consecutive blocks
  const void* rs_in;
  int64_t rs_bs;
  void* rs_out;
  int rs_np, nb_rs;
  // all-gather role (the following nb_ag blocks); in = my shard chunk, out = base of the full output
  const void* ag_in;
  void* ag_out;
  int ag_np, ag_rpr, ag_layout, nb_ag;
  int64_t ag_p0, ag_total;
  // cut_rs / cut_ag: the socket-cut pushes of the reduce-scatter / all-gather are done by one dedicated block each
  int cut_rs, cut_ag;
};
// launch_rs / launch_ag_chunk: `grid` counts the worker blocks; with `cut` one more block is launched (the cut block)
void launch_rs(const Ctx& c, int dtype, int bx, int cut, const void* in, int64_t in_bs, void* out, int np, int grid,
               int block, int unroll, cudaStream_t s);
void launch_ag_chunk(const Ctx& c, const void* in, void* out, int np, int rpr, int layout, int64_t p0,
                     int64_t np_total, int cut, int grid, int block, cudaStream_t s);
void launch_step(const Ctx& c, int dtype, int bx, int unroll, const StepArgs& a, int block, cudaStream_t s);

}  // namespace muse_topo
