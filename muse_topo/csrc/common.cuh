// Shared device/host definitions of the muse_topo collectives (8 PCIe GPUs, 2 sockets x 4 GPUs).
//
// Transport rules (see docs/topology.md): intra-island traffic is SM push stores into the peer's IPC receive
// buffer; cross-island traffic only goes rank <-> partner(rank). No remote atomics (not supported on this box).
//
// Two payload-completion protocols are used:
//  * "sentinel" (S-LL): receive slots hold the all-ones word 0xFFFFFFFF while empty. Senders store plain 16 B
//    payload packs; receivers poll the payload itself, a pack is complete when none of its four 32-bit words is
//    the sentinel, and the receiver resets the slot to the sentinel after consuming it. No fence, no flag, no
//    wire overhead. Reduction payloads are canonicalized so that no value equals the sentinel (see canon()),
//    which only changes the payload of NaNs.
//  * "LL8" (tagged): 8 B units {4 B data, 4 B epoch tag}. Used for pure data movement (all-gather), where the
//    payload must be preserved bit for bit, so no data value can be reserved.
//
// Double buffering: every receive region exists twice, selected by the parity of a per-GPU call counter
// (ctl[0..1] as one 64-bit word, advanced once per launched collective by the last warp that arrived, see
// epoch_begin/epoch_end). Reuse is safe because every sender->receiver direction has a reverse receiver->sender
// dependency inside each collective, so a sender can only start call k+2 after the receiver has finished call k.
//
// Counter width: the counter and the flag-protocol flags are 64 bit and compared with a plain `<`, so they never
// wrap (2^63 launches) and a flag cannot appear "already reached" because of an old value or the all-ones initial
// value (that is -1 as a signed 64-bit word, i.e. below every tag). With 32-bit words the signed compare broke at
// the 2^31-th launch (spin timeout, or stale data after a flag from a previous epoch) and an unsigned compare
// would still accept the all-ones initial flags once the tag has its high bit set. The 8-byte LL8 units only have
// room for a 32-bit tag, see ll8_tag().
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "topo_types.h"

namespace muse_topo {

// ------------------------------------------------------------------------------------------------
// memory primitives
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint4 ld_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_v4(void* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint2 ld_v2(const void* p) {
  uint2 v;
  asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_v2(void* p, uint32_t a, uint32_t b) {
  asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ uint32_t ld_u32(const void* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_u32(void* p, uint32_t v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_u64(const void* p) {
  unsigned long long v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_u64(void* p, unsigned long long v) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ bool full(uint4 v) { return v.x != kSent && v.y != kSent && v.z != kSent && v.w != kSent; }
__device__ __forceinline__ uint32_t canon(uint32_t w) { return w == kSent ? kSent - 1 : w; }
__device__ __forceinline__ uint4 canon(uint4 v) { return make_uint4(canon(v.x), canon(v.y), canon(v.z), canon(v.w)); }
__device__ __forceinline__ uint4 sent4() { return make_uint4(kSent, kSent, kSent, kSent); }

// LL8 tag of call epoch e. A slot of parity p is reused by every second call, so its tag advances by one per reuse;
// the 32-bit tag therefore repeats after 0xFFFFFFFE reuses (about 2^33 launches, a stale slot would have to survive
// that long unwritten). The value 0xFFFFFFFF is never used: a never-written slot holds it (workspace memset).
__device__ __forceinline__ uint32_t ll8_tag(uint64_t e) {
  return static_cast<uint32_t>((e >> 1) % 0xFFFFFFFEull) + 1u;
}

// Debug timeline (block 0, thread 0): compiled in only with -DMUSE_TOPO_TRACE.
#ifdef MUSE_TOPO_TRACE
#define TRACE(c, i)                                                         \
  do {                                                                      \
    if ((c).trace && blockIdx.x == 0 && threadIdx.x == 0) (c).trace[i] = gtimer(); \
  } while (0)
// Role timeline of the large kernels: thread 0 of the first block of a role stamps its start and end (rows indexed by
// the call epoch, 8 stamps per row, after the 64 words of TRACE above).
#define TRACE_ROLE(c, e, role, which)                                                                         \
  do {                                                                                                      \
    if ((c).trace && threadIdx.x == 0) (c).trace[64 + ((e) & 255u) * 8 + (role) * 2 + (which)] = gtimer(); \
  } while (0)
#else
#define TRACE(c, i) \
  do {              \
  } while (0)
#define TRACE_ROLE(c, e, role, which) \
  do {                                \
  } while (0)
#endif

// flag protocol: payload stores, __syncthreads, then a release store of the call tag into the receiver's flag
// column of this block; the receiver acquires the flag and, after a __syncthreads, reads the payload.
// 64-bit flags: the call counter is 64 bit, so a flag never wraps (no modular compare, no stale "ahead" flags)
__device__ __forceinline__ long long ld_acquire(const long long* p) {
  long long v;
  asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(long long* p, long long v) {
  asm volatile("st.release.sys.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// flags of (kind, block, source rank) inside workspace `base`
__device__ __forceinline__ long long* flag_at(const Ctx& c, char* base, int kind, int block, int src) {
  return reinterpret_cast<long long*>(base + c.L.f_off) + ((int64_t)kind * kMaxBlocks + block) * kWorld + src;
}

// ------------------------------------------------------------------------------------------------
// bounded spinning: a rank that never shows up must not hang the GPU silently
// ------------------------------------------------------------------------------------------------
static __device__ __noinline__ void spin_fail(const Ctx& c, int code) {
  if (c.err) *c.err = (code << 8) | (c.rank + 1);
  if ((threadIdx.x & 31) == 0)
    printf("muse_topo: spin timeout rank %d code %d block %d warp %d\n", c.rank, code, (int)blockIdx.x,
           (int)(threadIdx.x >> 5));
  __trap();
}

struct Spin {
  unsigned n = 0;
  unsigned long long t0 = 0;
  __device__ __forceinline__ void tick(const Ctx& c, int code) {
    if ((++n & 255u) == 0) {
      const unsigned long long now = gtimer();
      if (t0 == 0) {
        t0 = now;
      } else if (now - t0 > c.timeout_ns) {
        spin_fail(c, code);
      }
    }
  }
};

__device__ __forceinline__ void wait_flag(const Ctx& c, const long long* p, long long tag, int code) {
  Spin sp;
  while (ld_acquire(p) < tag) sp.tick(c, code);
}

// ------------------------------------------------------------------------------------------------
// per-GPU call counter. Every warp reads the counter once and registers its arrival; the last warp to arrive
// advances the counter for the next collective (stream order guarantees that all warps of this launch have
// read the old value before any warp of the next launch starts). Not valid with programmatic dependent launch.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ uint64_t epoch_begin(const Ctx& c, uint32_t& old) {
  if (c.pdl) {
    pdl_wait();      // the previous grid (and its counter update) has completed
    pdl_trigger();   // let the next collective's blocks become resident; they wait for this grid to finish
  } else if (c.trigger) {
    pdl_trigger();   // not a programmatic dependent itself; only a programmatic successor (gemv_push_kernel) starts early
  }
  const uint64_t e = ld_u64(c.ctl);
  uint32_t one;
  asm volatile("mov.u32 %0, 1;" : "=r"(one) : "r"((uint32_t)e));  // data dependency: the atomic waits for the load
  old = 0;
  if ((threadIdx.x & 31) == 0) old = atomicAdd(c.ctl + 32, one);
  return e;
}
__device__ __forceinline__ void epoch_end(const Ctx& c, uint64_t e, uint32_t old) {
  const uint32_t total = gridDim.x * (blockDim.x >> 5);
  if ((threadIdx.x & 31) == 0 && old == total - 1) {
    st_u32(c.ctl + 32, 0);
    st_u64(c.ctl, e + 1);
  }
}

// The collective that was started by a gemv_push_kernel (the two halves of one call): the second kernel is a programmatic dependent of the
// first, so it waits for it and for everything before it, and then takes the call epoch the first one read (it does not advance it).
// The second kernel does not trigger: its own successors start after it, as after any other topo kernel.
__device__ __forceinline__ uint64_t epoch_begin_pushed(const Ctx& c, uint32_t& old) {
  pdl_wait();
  const uint64_t e = ld_u64(c.ctl);
  uint32_t one;
  asm volatile("mov.u32 %0, 1;" : "=r"(one) : "r"((uint32_t)e));
  old = 0;
  if ((threadIdx.x & 31) == 0) old = atomicAdd(c.ctl + 32, one);
  return e;
}

// ------------------------------------------------------------------------------------------------
// element traits: a 32-bit word holds EPW elements
// ------------------------------------------------------------------------------------------------
template <int D> struct Tr;
template <> struct Tr<kBF16> {
  static constexpr int EPW = 2;
  __device__ __forceinline__ static void unpack(uint32_t w, float* f) {
    f[0] = __uint_as_float(w << 16);
    f[1] = __uint_as_float(w & 0xffff0000u);
  }
  __device__ __forceinline__ static uint32_t pack(const float* f) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(f[0], f[1]);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
};
template <> struct Tr<kFP16> {
  static constexpr int EPW = 2;
  __device__ __forceinline__ static void unpack(uint32_t w, float* f) {
    const __half2 v = *reinterpret_cast<const __half2*>(&w);
    const float2 t = __half22float2(v);
    f[0] = t.x;
    f[1] = t.y;
  }
  __device__ __forceinline__ static uint32_t pack(const float* f) {
    const __half2 v = __floats2half2_rn(f[0], f[1]);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
};
template <> struct Tr<kFP32> {
  static constexpr int EPW = 1;
  __device__ __forceinline__ static void unpack(uint32_t w, float* f) { f[0] = __uint_as_float(w); }
  __device__ __forceinline__ static uint32_t pack(const float* f) { return __float_as_uint(f[0]); }
};

// Unpack one 16-byte pack into 4*EPW floats.
template <int D>
__device__ __forceinline__ void unpack_pack(const uint4& v, float* f) {
  constexpr int E = Tr<D>::EPW;
  Tr<D>::unpack(v.x, f);
  Tr<D>::unpack(v.y, f + E);
  Tr<D>::unpack(v.z, f + 2 * E);
  Tr<D>::unpack(v.w, f + 3 * E);
}
template <int D>
__device__ __forceinline__ uint4 pack_pack(const float* f) {
  constexpr int E = Tr<D>::EPW;
  return make_uint4(Tr<D>::pack(f), Tr<D>::pack(f + E), Tr<D>::pack(f + 2 * E), Tr<D>::pack(f + 3 * E));
}

// Launch helper: <<<>>> or cudaLaunchKernelEx with the programmatic-serialization attribute (c.pdl).
template <typename... KArgs, typename... Args>
static void launch_kp(void (*kernel)(KArgs...), bool pdl, dim3 grid, dim3 block, cudaStream_t s, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kernel, args...);
}
template <typename... KArgs, typename... Args>
static void launch_k(void (*kernel)(KArgs...), const Ctx& c, dim3 grid, dim3 block, cudaStream_t s, Args... args) {
  launch_kp(kernel, c.pdl != 0, grid, block, s, args...);
}

}  // namespace muse_topo
