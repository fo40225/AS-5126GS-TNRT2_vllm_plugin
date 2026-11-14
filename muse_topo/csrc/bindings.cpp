// pybind11 module _muse_topo_C: workspace/IPC helpers and the Comm object that launches the collectives.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "topo_types.h"

namespace py = pybind11;
using namespace muse_topo;

#define CK(x)                                                                             \
  do {                                                                                    \
    cudaError_t e_ = (x);                                                                 \
    if (e_ != cudaSuccess) {                                                              \
      throw std::runtime_error(std::string(#x) + " failed: " + cudaGetErrorString(e_));   \
    }                                                                                     \
  } while (0)

static int dtype_code(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kBFloat16: return kBF16;
    case at::kHalf: return kFP16;
    case at::kFloat: return kFP32;
    default: throw std::runtime_error("muse_topo: unsupported dtype");
  }
}

// Launch geometry limits: warps are counted for the call counter (block % 32 == 0), kernels are compiled with
// __launch_bounds__(512) (sentinel / LL8) or 1024 (flag kernels).
static void check_geometry(int max_block, int grid, int block, int unroll) {
  TORCH_CHECK(block >= 32 && block <= max_block && block % 32 == 0, "bad block size ", block);
  TORCH_CHECK(grid >= 1 && grid <= 4096, "bad grid size ", grid);
  TORCH_CHECK(unroll == 1 || unroll == 2 || unroll == 4, "bad unroll ", unroll);
}

// cudaMalloc'd workspace, filled with the sentinel pattern; returns (device pointer, IPC handle bytes).
static py::tuple ws_alloc(int64_t nbytes) {
  void* p = nullptr;
  CK(cudaMalloc(&p, nbytes));
  CK(cudaMemset(p, 0xFF, nbytes));
  CK(cudaDeviceSynchronize());
  cudaIpcMemHandle_t h;
  CK(cudaIpcGetMemHandle(&h, p));
  return py::make_tuple((int64_t)p, py::bytes(reinterpret_cast<const char*>(&h), sizeof(h)));
}
static int64_t ws_open(const py::bytes& handle) {
  std::string s = handle;
  if (s.size() != sizeof(cudaIpcMemHandle_t)) throw std::runtime_error("bad IPC handle size");
  cudaIpcMemHandle_t h;
  std::memcpy(&h, s.data(), sizeof(h));
  void* p = nullptr;
  CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return (int64_t)p;
}
static void ws_close_peer(int64_t p) { CK(cudaIpcCloseMemHandle((void*)p)); }
static void ws_free(int64_t p) { CK(cudaFree((void*)p)); }

static py::dict layout_dict(int64_t max_bytes, int64_t a1_max, int64_t g_max, int64_t h_max) {
  WsLayout L = make_layout(max_bytes, a1_max, g_max, h_max);
  py::dict d;
  d["total"] = L.total;
  d["a1_off"] = L.a1_off;
  d["a2_off"] = L.a2_off;
  d["c2_off"] = L.c2_off;
  d["g_off"] = L.g_off;
  return d;
}

static constexpr size_t kTraceBytes = (64 + 256 * 8) * 8;

struct Comm {
  Ctx c;
  int device;
  int* host_err = nullptr;
  unsigned long long* host_trace = nullptr;
  uint32_t* ctl = nullptr;

  Comm(int rank, int island, int ipos, int opos, int partner, std::vector<std::vector<int>> members,
       std::vector<int> owners, std::vector<int> partner_of, std::vector<int64_t> bases, int64_t max_bytes,
       int64_t a1_max, int64_t g_max, int64_t h_max, double timeout_s, int device_)
      : device(device_) {
    std::memset(&c, 0, sizeof(c));
    c.rank = rank;
    c.island = island;
    c.ipos = ipos;
    c.opos = opos;
    c.partner = partner;
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < kIsland; ++k) c.members[i][k] = members.at(i).at(k);
    for (int k = 0; k < kIsland; ++k) c.owners[k] = owners.at(k);
    for (int r = 0; r < kWorld; ++r) {
      c.partner_of[r] = partner_of.at(r);
      c.base[r] = reinterpret_cast<char*>(bases.at(r));
    }
    c.L = make_layout(max_bytes, a1_max, g_max, h_max);
    c.timeout_ns = (unsigned long long)(timeout_s * 1e9);
    c10::cuda::CUDAGuard g(device);
    CK(cudaMalloc((void**)&ctl, 512));
    CK(cudaMemset(ctl, 0, 512));
    CK(cudaHostAlloc((void**)&host_err, 64, cudaHostAllocMapped));
    *host_err = 0;
    int* dev_err = nullptr;
    CK(cudaHostGetDevicePointer((void**)&dev_err, host_err, 0));
    c.ctl = ctl;
    c.err = dev_err;
  }
  ~Comm() { close(); }

  void close() {
    if (ctl || host_err) {
      c10::cuda::CUDAGuard g(device);
      if (ctl) cudaFree(ctl);
      if (host_err) cudaFreeHost(host_err);
      if (host_trace) cudaFreeHost(host_trace);
    }
    ctl = nullptr;
    host_err = nullptr;
    host_trace = nullptr;
  }
  int error() const { return host_err ? *(volatile int*)host_err : 0; }
  void set_timeout(double s) { c.timeout_ns = (unsigned long long)(s * 1e9); }
  // debug timeline (needs a build with MUSE_TOPO_TRACE=1): mapped host stamps written by block 0 / thread 0
  void trace_enable() {
    if (!host_trace) {
      CK(cudaHostAlloc((void**)&host_trace, kTraceBytes, cudaHostAllocMapped));
      std::memset(host_trace, 0, kTraceBytes);
      unsigned long long* d = nullptr;
      CK(cudaHostGetDevicePointer((void**)&d, host_trace, 0));
      c.trace = d;
    }
  }
  void trace_clear() {
    if (host_trace) std::memset(host_trace, 0, kTraceBytes);
  }
  // role timeline rows of the large kernels: row (epoch & 255), 8 stamps: {rs, ag, rs cut, ag cut} x {start, end}
  std::vector<unsigned long long> trace_rows() const {
    std::vector<unsigned long long> v(256 * 8, 0);
    if (host_trace)
      for (int i = 0; i < 256 * 8; ++i) v[i] = ((volatile unsigned long long*)host_trace)[64 + i];
    return v;
  }
  std::vector<unsigned long long> trace() const {
    std::vector<unsigned long long> v(16, 0);
    if (host_trace)
      for (int i = 0; i < 16; ++i) v[i] = ((volatile unsigned long long*)host_trace)[i];
    return v;
  }
  // placement of the operand loads of the fused collectives (BndParams::ld_mode in topo_types.h)
  int ld_mode = 1;
  void set_ld_mode(int m) {
    TORCH_CHECK(m >= 0 && m <= 3, "bad ld_mode ", m);
    ld_mode = m;
  }
  void set_pdl(bool on) { c.pdl = on ? 1 : 0; }
  // Kernels that are not programmatic dependents themselves still let a programmatic successor (gemv_push_kernel) start at once.
  void set_trigger(bool on) { c.trigger = on ? 1 : 0; }
  // Geometry variant of the GEMV kernels (gemv_push.cu); the summation order depends on it.
  int gemv_variant = 0;
  void set_gemv_variant(int v) {
    TORCH_CHECK(v >= 0 && v <= 3, "bad GEMV variant ", v);
    gemv_variant = v;
  }
  // Test hooks (bench/test_epoch.py), not used by TopoComm: read / force the 64-bit per-GPU call counter. Only valid while no
  // collective is in flight, with the same value on all ranks, and only moving FORWARD: the flag words and LL8 slots keep
  // the tags of earlier calls, so a smaller counter would see them as "already arrived" (stale data).
  uint64_t get_epoch() {
    c10::cuda::CUDAGuard g(device);
    uint64_t v = 0;
    CK(cudaMemcpy(&v, ctl, sizeof(v), cudaMemcpyDeviceToHost));
    return v;
  }
  void set_epoch(uint64_t v) {
    c10::cuda::CUDAGuard g(device);
    CK(cudaMemcpy(ctl, &v, sizeof(v), cudaMemcpyHostToDevice));
  }

  // ------------------------------------------------------------------------------------------------
  // Large messages (large.cu). The message is 8 equal blocks; chunks of `chunk_bytes` are taken from every block.
  // ------------------------------------------------------------------------------------------------
  int64_t large_chunk_packs(int64_t chunk_bytes, int64_t block_packs, int bx, int dt) const {
    TORCH_CHECK(chunk_bytes >= 16 && chunk_bytes % 16 == 0, "bad chunk size ", chunk_bytes);
    const int64_t cp = chunk_bytes >> 4;
    const int pk = (bx && dt == kBF16) ? 1 : (dt == kFP32 ? 1 : 2);
    TORCH_CHECK(2 * cp * 16 <= c.L.a2_slot, "chunk too large for the phase-a slots: ", chunk_bytes);
    TORCH_CHECK(pk * cp * 16 <= c.L.b2_par, "chunk too large for the partial slots: ", chunk_bytes);
    TORCH_CHECK(cp * 16 <= c.L.h_slot, "chunk too large for the all-gather slots: ", chunk_bytes);
    return std::min(cp, block_packs);
  }

  // reduce-scatter: x = 8 blocks of E bytes, out = E bytes (rank r receives the reduced block r)
  void reduce_scatter(const at::Tensor& x, at::Tensor out, int64_t chunk_bytes, int grid, int block, int unroll,
                      int bx, int cut) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous(), "bad tensors");
    c10::cuda::CUDAGuard g(x.device());
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 128 == 0 && out.numel() * out.element_size() * 8 == nbytes, "rs: bad sizes");
    const int dt = dtype_code(x);
    TORCH_CHECK(!bx || dt == kBF16, "bf16 exchange is only defined for bf16 inputs");
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    check_geometry(512, grid, block, unroll);
    TORCH_CHECK(unroll <= 2, "bad unroll for the large kernels");
    const int64_t ep = (nbytes / 8) >> 4;
    const int64_t cp = large_chunk_packs(chunk_bytes, ep, bx, dt);
    const char* xi = static_cast<const char*>(x.data_ptr());
    char* o = static_cast<char*>(out.data_ptr());
    for (int64_t p0 = 0; p0 < ep; p0 += cp) {
      const int np = static_cast<int>(std::min(cp, ep - p0));
      launch_rs(c, dt, bx, cut, xi + p0 * 16, ep, o + p0 * 16, np, grid, block, unroll, s);
    }
    CK(cudaGetLastError());
  }

  // chunk sizes (packs) of a block: uniform cp, or with ramp > 0 smaller chunks at both ends of the pipeline (the first
  // reduce-scatter and the last all-gather run alone, so a small chunk shortens the fill and the drain):
  // ramp 1: [cp/2 ... cp/2], ramp 2: [cp/4, cp/2 ... cp/2, cp/4]
  static std::vector<int64_t> chunk_sizes(int64_t ep, int64_t cp, int ramp) {
    std::vector<int64_t> pre;
    if (ramp >= 2) pre.push_back(std::max<int64_t>(1, cp / 4));
    if (ramp >= 1) pre.push_back(std::max<int64_t>(1, cp / 2));
    int64_t used = 0;
    for (int64_t v : pre) used += 2 * v;  // the same sizes mirrored at the end
    std::vector<int64_t> sizes;
    if (ramp > 0 && ep >= used + 4 * cp) {
      for (int64_t v : pre) sizes.push_back(v);
      for (int64_t rest = ep - used; rest > 0;) {
        const int64_t v = std::min(cp, rest);
        sizes.push_back(v);
        rest -= v;
      }
      for (auto it = pre.rbegin(); it != pre.rend(); ++it) sizes.push_back(*it);
    } else {
      for (int64_t rest = ep; rest > 0;) {
        const int64_t v = std::min(cp, rest);
        sizes.push_back(v);
        rest -= v;
      }
    }
    return sizes;
  }

  // all-reduce = reduce-scatter + all-gather per chunk. mode 0: two launches per chunk, mode 1: fused pipeline
  // (reduce-scatter of chunk c and all-gather of chunk c-1 in one launch, rs_grid + ag_grid blocks of `block`).
  void all_reduce_large(const at::Tensor& x, at::Tensor out, int64_t chunk_bytes, int mode, int rs_grid, int ag_grid,
                        int block, int unroll, int bx, int cut_rs, int cut_ag, int ramp) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.sizes() == x.sizes(), "bad tensors");
    c10::cuda::CUDAGuard g(x.device());
    const int64_t nbytes = x.numel() * x.element_size();
    TORCH_CHECK(nbytes % 128 == 0, "ar large: size must be a multiple of 128 bytes");
    const int dt = dtype_code(x);
    TORCH_CHECK(!bx || dt == kBF16, "bf16 exchange is only defined for bf16 inputs");
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    check_geometry(512, rs_grid, block, unroll);
    check_geometry(512, ag_grid, block, unroll);
    TORCH_CHECK(unroll <= 2 && ag_grid <= kMaxBlocks && mode >= 0 && mode <= 1, "bad geometry");
    const int64_t ep = (nbytes / 8) >> 4;  // packs per block
    const int64_t cp = large_chunk_packs(chunk_bytes, ep, bx, dt);
    const std::vector<int64_t> sizes = chunk_sizes(ep, cp, ramp);
    std::vector<int64_t> starts(sizes.size() + 1, 0);
    for (size_t i = 0; i < sizes.size(); ++i) starts[i + 1] = starts[i] + sizes[i];
    const int64_t nc = static_cast<int64_t>(sizes.size());
    const char* xi = static_cast<const char*>(x.data_ptr());
    char* o = static_cast<char*>(out.data_ptr());
    char* mine = o + (int64_t)c.rank * ep * 16;  // my output block
    if (mode == 0) {
      for (int64_t k = 0; k < nc; ++k) {
        const int64_t p0 = starts[k];
        const int np = static_cast<int>(sizes[k]);
        launch_rs(c, dt, bx, cut_rs, xi + p0 * 16, ep, mine + p0 * 16, np, rs_grid, block, unroll, s);
        launch_ag_chunk(c, mine + p0 * 16, o, np, 1, 1, p0, ep, cut_ag, ag_grid, std::min(block, 1024), s);
      }
    } else {
      for (int64_t k = 0; k <= nc; ++k) {
        StepArgs a{};
        if (k < nc) {
          const int64_t p0 = starts[k];
          a.rs_in = xi + p0 * 16;
          a.rs_bs = ep;
          a.rs_out = mine + p0 * 16;
          a.rs_np = static_cast<int>(sizes[k]);
          a.nb_rs = rs_grid;
          a.cut_rs = cut_rs;
        }
        if (k >= 1) {
          const int64_t p0 = starts[k - 1];
          a.ag_in = mine + p0 * 16;
          a.ag_out = o;
          a.ag_np = static_cast<int>(sizes[k - 1]);
          a.ag_rpr = 1;
          a.ag_layout = 1;
          a.ag_p0 = p0;
          a.ag_total = ep;
          a.nb_ag = ag_grid;
          a.cut_ag = cut_ag;
        }
        launch_step(c, dt, bx, unroll, a, block, s);
      }
    }
    CK(cudaGetLastError());
  }

  // chunked all-gather (flag protocol) of shards of any size: x = my shard, out = full output (layout as all_gather)
  void all_gather_large(const at::Tensor& x, at::Tensor out, int layout, int64_t chunk_bytes, int grid, int block,
                        int cut) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous(), "bad tensors");
    c10::cuda::CUDAGuard g(x.device());
    const int64_t shard = x.numel() * x.element_size();
    const int64_t row = x.size(-1) * x.element_size();
    TORCH_CHECK(out.numel() == x.numel() * kWorld, "bad output size");
    TORCH_CHECK(shard % 16 == 0 && (layout == 1 || row % 16 == 0), "ag large: size not supported");
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    check_geometry(1024, grid, block, 1);
    TORCH_CHECK(grid <= kMaxBlocks, "bad grid");
    const int64_t np_total = shard >> 4;
    TORCH_CHECK(chunk_bytes >= 16 && chunk_bytes % 16 == 0 && chunk_bytes <= c.L.h_slot, "bad chunk size");
    const int64_t cp = std::min<int64_t>(chunk_bytes >> 4, np_total);
    const int rpr = layout == 0 ? static_cast<int>(row >> 4) : 1;
    const char* xi = static_cast<const char*>(x.data_ptr());
    for (int64_t p0 = 0; p0 < np_total; p0 += cp) {
      const int np = static_cast<int>(std::min(cp, np_total - p0));
      launch_ag_chunk(c, xi + p0 * 16, out.data_ptr(), np, rpr, layout, p0, np_total, cut, grid, block, s);
    }
    CK(cudaGetLastError());
  }

  void all_reduce(const at::Tensor& x, at::Tensor out, int algo, int grid, int block, int unroll, int bx) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous() && out.sizes() == x.sizes(), "bad tensors");
    c10::cuda::CUDAGuard g(x.device());
    const int64_t nbytes = x.numel() * x.element_size();
    const int dt = dtype_code(x);
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    check_geometry(algo == 3 ? 1024 : 512, grid, block, unroll);
    TORCH_CHECK(!bx || algo == 2, "bf16 exchange is only defined for all-reduce algorithm 2");
    if (algo == 1) {
      TORCH_CHECK(nbytes % 16 == 0 && nbytes <= c.L.a1_slot, "ar1: size not supported");
      launch_ar1(c, dt, x.data_ptr(), out.data_ptr(), nbytes, grid, block, unroll, s);
    } else if (algo == 2) {
      TORCH_CHECK(nbytes % 64 == 0 && nbytes / 4 <= c.L.a2_slot, "ar2: size not supported");
      TORCH_CHECK(!bx || dt == kBF16, "bf16 exchange is only defined for bf16 inputs");
      launch_ar2(c, dt, x.data_ptr(), out.data_ptr(), nbytes, grid, block, unroll, bx, s);
    } else if (algo == 3) {
      TORCH_CHECK(nbytes % 64 == 0 && nbytes / 4 <= c.L.a2_slot && grid <= kMaxBlocks, "ar2 flag: size not supported");
      launch_ar2_flag(c, dt, x.data_ptr(), out.data_ptr(), nbytes, grid, block, s);
    } else {
      throw std::runtime_error("unknown all-reduce algorithm");
    }
    CK(cudaGetLastError());
  }

  // all-reduce of bf16 [T, 8192] fused with the Muse Spark residual boundary (boundary.cuh); T <= 64, one CTA per row
  void all_reduce_boundary(const at::Tensor& x, const at::Tensor& stream, at::Tensor new_stream, at::Tensor out,
                           c10::optional<at::Tensor> out_fp32, const at::Tensor& alpha, const at::Tensor& beta,
                           c10::optional<at::Tensor> post_w, const at::Tensor& pre_w, double post_eps, double pre_eps,
                           double post_gain, double pre_gain) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == 8192,
                "ar boundary: x must be a contiguous bf16 [T, 8192] CUDA tensor");
    const int rows = static_cast<int>(x.size(0));
    TORCH_CHECK(rows >= 1 && rows <= 64, "ar boundary: 1 <= T <= 64 rows, got ", rows);
    TORCH_CHECK(static_cast<int64_t>(rows) * 256 * 16 <= c.L.a2_slot && static_cast<int64_t>(rows) * 256 * 32 <= c.L.b2_par,
                "ar boundary: workspace too small");
    c10::cuda::CUDAGuard g(x.device());
    auto chk32 = [&](const at::Tensor& t, int64_t n, const char* what) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat && t.numel() == n, "ar boundary: bad ", what);
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "ar boundary: ", what, " not 16-byte aligned");
    };
    auto chk16 = [&](const at::Tensor& t, int64_t n, const char* what) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16 && t.numel() == n, "ar boundary: bad ", what);
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "ar boundary: ", what, " not 16-byte aligned");
    };
    chk32(stream, x.numel(), "stream");
    chk32(new_stream, x.numel(), "new_stream");
    chk16(out, x.numel(), "out");
    chk32(alpha, 8192, "alpha");
    chk32(beta, 8192, "beta");
    chk16(pre_w, 8192, "pre_w");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "ar boundary: x not 16-byte aligned");
    BndParams a{};
    a.stream = stream.data_ptr<float>();
    a.new_stream = new_stream.data_ptr<float>();
    a.out = out.data_ptr();
    if (out_fp32.has_value()) {
      chk32(*out_fp32, x.numel(), "out_fp32");
      a.out_fp32 = out_fp32->data_ptr<float>();
    }
    a.alpha = alpha.data_ptr<float>();
    a.beta = beta.data_ptr<float>();
    if (post_w.has_value()) {
      chk16(*post_w, 8192, "post_w");
      a.post_w = post_w->data_ptr();
    }
    a.pre_w = pre_w.data_ptr();
    a.post_eps = static_cast<float>(post_eps);
    a.pre_eps = static_cast<float>(pre_eps);
    a.post_gain = static_cast<float>(post_gain);
    a.pre_gain = static_cast<float>(pre_gain);
    a.ld_mode = ld_mode;
    launch_ar2_bnd(c, x.data_ptr(), a, rows, at::cuda::getCurrentCUDAStream());
    CK(cudaGetLastError());
  }

  // all-gather (column concatenation) of the bf16 unpooler shards [T, 1024] fused with the Muse Spark residual boundary
  // (boundary.cuh, epi 0); T <= 32 (LL8 slots)
  void ag_boundary_impl(const at::Tensor& shard, const at::Tensor& stream, at::Tensor new_stream, at::Tensor out,
                        c10::optional<at::Tensor> out_fp32, const at::Tensor& alpha, const at::Tensor& beta,
                        c10::optional<at::Tensor> post_w, const at::Tensor& pre_w, double post_eps, double pre_eps,
                        double post_gain, double pre_gain, int pushed) {
    TORCH_CHECK(shard.is_cuda() && shard.is_contiguous() && shard.scalar_type() == at::kBFloat16 && shard.dim() == 2 &&
                    shard.size(1) == 1024,
                "ag boundary: shard must be a contiguous bf16 [T, 1024] CUDA tensor");
    const int rows = static_cast<int>(shard.size(0));
    TORCH_CHECK(rows >= 1 && rows <= 32 && 2 * shard.numel() * 2 <= c.L.g_slot, "ag boundary: 1 <= T <= 32 rows, got ", rows);
    c10::cuda::CUDAGuard g(shard.device());
    const int64_t n = static_cast<int64_t>(rows) * 8192;
    auto chk32 = [&](const at::Tensor& t, int64_t k, const char* what) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat && t.numel() == k, "ag boundary: bad ", what);
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "ag boundary: ", what, " not 16-byte aligned");
    };
    auto chk16 = [&](const at::Tensor& t, int64_t k, const char* what) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16 && t.numel() == k, "ag boundary: bad ", what);
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "ag boundary: ", what, " not 16-byte aligned");
    };
    chk32(stream, n, "stream");
    chk32(new_stream, n, "new_stream");
    chk16(out, n, "out");
    chk32(alpha, 8192, "alpha");
    chk32(beta, 8192, "beta");
    chk16(pre_w, 8192, "pre_w");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(shard.data_ptr()) % 4 == 0, "ag boundary: shard not 4-byte aligned");
    BndParams a{};
    a.stream = stream.data_ptr<float>();
    a.new_stream = new_stream.data_ptr<float>();
    a.out = out.data_ptr();
    if (out_fp32.has_value()) {
      chk32(*out_fp32, n, "out_fp32");
      a.out_fp32 = out_fp32->data_ptr<float>();
    }
    a.alpha = alpha.data_ptr<float>();
    a.beta = beta.data_ptr<float>();
    if (post_w.has_value()) {
      chk16(*post_w, 8192, "post_w");
      a.post_w = post_w->data_ptr();
    }
    a.pre_w = pre_w.data_ptr();
    a.post_eps = static_cast<float>(post_eps);
    a.pre_eps = static_cast<float>(pre_eps);
    a.post_gain = static_cast<float>(post_gain);
    a.pre_gain = static_cast<float>(pre_gain);
    a.ld_mode = ld_mode;
    launch_ag_bnd(c, shard.data_ptr(), a, rows, pushed, at::cuda::getCurrentCUDAStream());
    CK(cudaGetLastError());
  }
  void all_gather_boundary(const at::Tensor& shard, const at::Tensor& stream, at::Tensor new_stream, at::Tensor out,
                           c10::optional<at::Tensor> out_fp32, const at::Tensor& alpha, const at::Tensor& beta,
                           c10::optional<at::Tensor> post_w, const at::Tensor& pre_w, double post_eps, double pre_eps,
                           double post_gain, double pre_gain) {
    ag_boundary_impl(shard, stream, new_stream, out, out_fp32, alpha, beta, post_w, pre_w, post_eps, pre_eps, post_gain,
                     pre_gain, 0);
  }

  // all-gather (column concatenation) of the bf16 pooler shards [T, 512] fused with the pre_expert_norm RMSNorm (epi 1);
  // T <= 64; returns into `out` [T, 4096]
  void ag_rmsnorm_impl(const at::Tensor& shard, at::Tensor out, const at::Tensor& w, double eps, double gain, int pushed) {
    TORCH_CHECK(shard.is_cuda() && shard.is_contiguous() && shard.scalar_type() == at::kBFloat16 && shard.dim() == 2 &&
                    shard.size(1) == 512,
                "ag rmsnorm: shard must be a contiguous bf16 [T, 512] CUDA tensor");
    const int rows = static_cast<int>(shard.size(0));
    TORCH_CHECK(rows >= 1 && rows <= 64 && 2 * shard.numel() * 2 <= c.L.g_slot, "ag rmsnorm: 1 <= T <= 64 rows, got ", rows);
    c10::cuda::CUDAGuard g(shard.device());
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() == shard.numel() * 8,
                "ag rmsnorm: bad out");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.numel() == 4096, "ag rmsnorm: bad w");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(shard.data_ptr()) % 4 == 0,
                "ag rmsnorm: misaligned operand");
    RmsParams a{};
    a.out = out.data_ptr();
    a.w = w.data_ptr();
    a.eps = static_cast<float>(eps);
    a.gain = static_cast<float>(gain);
    launch_ag_rms(c, shard.data_ptr(), a, rows, pushed, at::cuda::getCurrentCUDAStream());
    CK(cudaGetLastError());
  }
  void all_gather_rmsnorm(const at::Tensor& shard, at::Tensor out, const at::Tensor& w, double eps, double gain) {
    ag_rmsnorm_impl(shard, out, w, eps, gain, 0);
  }

  // GEMV of the sharded projection (kind 0 pooler, 1 unpooler; gemv_push.cu) into `shard`, with `push` also the first stage of the
  // all-gather. The two halves of one collective call: launch_gemv_push, then the pushed gather kernel.
  void gemv_push_check(const at::Tensor& x, const at::Tensor& w, const at::Tensor& shard, int kind) {
    const int K = kind == 0 ? 8192 : 4096, N = kind == 0 ? 512 : 1024;
    TORCH_CHECK(kind == 0 || kind == 1, "gemv: bad kind ", kind);
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == K,
                "gemv: x must be a contiguous bf16 [T, ", K, "] CUDA tensor");
    const int rows = static_cast<int>(x.size(0));
    TORCH_CHECK(rows >= 1 && rows <= 8, "gemv: 1 <= T <= 8 rows, got ", rows);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.size(0) == N && w.size(1) == K,
                "gemv: w must be a contiguous bf16 [", N, ", ", K, "] CUDA tensor");
    TORCH_CHECK(shard.is_cuda() && shard.is_contiguous() && shard.scalar_type() == at::kBFloat16 && shard.dim() == 2 &&
                    shard.size(0) == rows && shard.size(1) == N,
                "gemv: bad shard");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(shard.data_ptr()) % 4 == 0,
                "gemv: misaligned operand");
    TORCH_CHECK(2 * shard.numel() * 2 <= c.L.g_slot, "gemv: shard larger than the all-gather slots");
  }
  void gemv_push(const at::Tensor& x, const at::Tensor& w, at::Tensor shard, int kind, int push) {
    TORCH_CHECK(push >= 0 && push <= 7, "gemv: bad push mode ", push);
    gemv_push_check(x, w, shard, kind);
    c10::cuda::CUDAGuard g(x.device());
    launch_gemv_push(c, kind, gemv_variant, x.data_ptr(), w.data_ptr(), shard.data_ptr(), static_cast<int>(x.size(0)), push,
                     at::cuda::getCurrentCUDAStream());
    CK(cudaGetLastError());
  }
  void gemv_all_gather_rmsnorm(const at::Tensor& x, const at::Tensor& w, at::Tensor shard, at::Tensor out,
                               const at::Tensor& nw, double eps, double gain, int push) {
    gemv_push_check(x, w, shard, 0);
    c10::cuda::CUDAGuard g(x.device());
    launch_gemv_push(c, 0, gemv_variant, x.data_ptr(), w.data_ptr(), shard.data_ptr(), static_cast<int>(x.size(0)), push,
                     at::cuda::getCurrentCUDAStream());
    CK(cudaGetLastError());
    ag_rmsnorm_impl(shard, out, nw, eps, gain, (push & 3) != 0);
  }
  void gemv_all_gather_boundary(const at::Tensor& x, const at::Tensor& w, at::Tensor shard, const at::Tensor& stream,
                                at::Tensor new_stream, at::Tensor out, c10::optional<at::Tensor> out_fp32,
                                const at::Tensor& alpha, const at::Tensor& beta, c10::optional<at::Tensor> post_w,
                                const at::Tensor& pre_w, double post_eps, double pre_eps, double post_gain, double pre_gain,
                                int push) {
    gemv_push_check(x, w, shard, 1);
    c10::cuda::CUDAGuard g(x.device());
    launch_gemv_push(c, 1, gemv_variant, x.data_ptr(), w.data_ptr(), shard.data_ptr(), static_cast<int>(x.size(0)), push,
                     at::cuda::getCurrentCUDAStream());
    CK(cudaGetLastError());
    ag_boundary_impl(shard, stream, new_stream, out, out_fp32, alpha, beta, post_w, pre_w, post_eps, pre_eps, post_gain,
                     pre_gain, (push & 3) != 0);
  }

  // x: [T, w] shard (any dtype, bit exact); out: [8T, w] (layout 1) or [T, 8w] (layout 0)
  void all_gather(const at::Tensor& x, at::Tensor out, int layout, int algo, int grid, int block, int unroll) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && out.is_contiguous(), "bad tensors");
    c10::cuda::CUDAGuard g(x.device());
    const int64_t shard = x.numel() * x.element_size();
    const int64_t row = x.size(-1) * x.element_size();
    TORCH_CHECK(out.numel() == x.numel() * kWorld, "bad output size");
    cudaStream_t s = at::cuda::getCurrentCUDAStream();
    check_geometry(algo == 2 ? 1024 : 512, grid, block, unroll);
    if (algo == 1) {
      // the row is only used by layout 0 (column concatenation); rank-major output copies whole shards
      TORCH_CHECK(shard % 4 == 0 && (layout == 1 || row % 4 == 0) && 2 * shard <= c.L.g_slot,
                  "ag ll: size not supported");
      launch_ag_ll(c, x.data_ptr(), out.data_ptr(), shard, row, layout, grid, block, unroll, s);
    } else if (algo == 2) {
      TORCH_CHECK(shard % 16 == 0 && (layout == 1 || row % 16 == 0), "ag flag: size not supported");
      TORCH_CHECK(shard <= c.L.h_slot && grid <= kMaxBlocks, "ag flag: shard too large");
      launch_ag_flag(c, x.data_ptr(), out.data_ptr(), shard, row, layout, grid, block, s);
    } else {
      throw std::runtime_error("unknown all-gather algorithm");
    }
    CK(cudaGetLastError());
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ws_alloc", &ws_alloc);
  m.def("ws_open", &ws_open);
  m.def("ws_close_peer", &ws_close_peer);
  m.def("ws_free", &ws_free);
  m.def("layout", &layout_dict);
  py::class_<Comm>(m, "Comm")
      .def(py::init<int, int, int, int, int, std::vector<std::vector<int>>, std::vector<int>, std::vector<int>,
                    std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t, double, int>())
      .def("all_reduce", &Comm::all_reduce, py::arg("x"), py::arg("out"), py::arg("algo"), py::arg("grid"),
           py::arg("block"), py::arg("unroll"), py::arg("bx") = 0)
      .def("all_reduce_boundary", &Comm::all_reduce_boundary)
      .def("all_gather_boundary", &Comm::all_gather_boundary)
      .def("all_gather_rmsnorm", &Comm::all_gather_rmsnorm)
      .def("gemv_push", &Comm::gemv_push)
      .def("gemv_all_gather_rmsnorm", &Comm::gemv_all_gather_rmsnorm)
      .def("gemv_all_gather_boundary", &Comm::gemv_all_gather_boundary)
      .def("set_trigger", &Comm::set_trigger)
      .def("set_gemv_variant", &Comm::set_gemv_variant)
      .def("all_gather", &Comm::all_gather)
      .def("reduce_scatter", &Comm::reduce_scatter)
      .def("all_reduce_large", &Comm::all_reduce_large)
      .def("all_gather_large", &Comm::all_gather_large)
      .def("error", &Comm::error)
      .def("set_timeout", &Comm::set_timeout)
      .def("set_pdl", &Comm::set_pdl)
      .def("set_ld_mode", &Comm::set_ld_mode)
      .def("get_epoch", &Comm::get_epoch)
      .def("set_epoch", &Comm::set_epoch)
      .def("trace_enable", &Comm::trace_enable)
      .def("trace", &Comm::trace)
      .def("trace_rows", &Comm::trace_rows)
      .def("trace_clear", &Comm::trace_clear)
      .def("close", &Comm::close);
}
