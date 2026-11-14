// Test kernels around muse_topo/csrc/boundary.cuh: the residual-boundary / RMSNorm epilogue math on plain tensors, so that
// bench/test_boundary.py can compare it bit for bit with the Triton kernels of the Muse Spark model.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "boundary.cuh"

using namespace muse_topo::bnd;

template <bool PS, bool F32>
__global__ void __launch_bounds__(256) k8192(muse_topo::BndParams p, const uint4* __restrict__ branch) {
  __shared__ float sm[66];
  const Bnd8192 a(p);
  const int row = blockIdx.x;
  Bnd8192Regs r;
  bnd8192_load<PS>(a, row, r);
  uint4 bx[kD8192Packs];
#pragma unroll
  for (int j = 0; j < kD8192Packs; ++j) bx[j] = branch[(int64_t)row * 1024 + kRowThreads * j + threadIdx.x];
  bnd8192_apply<PS, F32>(a, row, bx, r, sm);
}

__global__ void __launch_bounds__(256) k4096(muse_topo::RmsParams p, const uint4* __restrict__ x) {
  __shared__ float sm[9];
  const Rms4096 a(p);
  const int row = blockIdx.x;
  uint4 bx[kD4096Packs];
#pragma unroll
  for (int c = 0; c < kD4096Packs; ++c) bx[c] = x[(int64_t)row * 512 + kRowThreads * c + threadIdx.x];
  uint4 wx[kD4096Packs];
  rms4096_load(a, wx);
  rms4096_apply(a, row, bx, wx, sm);
}

std::vector<torch::Tensor> boundary8192(torch::Tensor branch, torch::Tensor stream, torch::Tensor alpha,
                                        torch::Tensor beta, c10::optional<torch::Tensor> post_w, torch::Tensor pre_w,
                                        double post_eps, double pre_eps, double post_gain, double pre_gain,
                                        bool fp32_out) {
  const int T = branch.size(0);
  auto out = torch::empty({T, 8192}, branch.options());
  auto new_stream = torch::empty_like(stream);
  auto out32 = torch::empty({T, 8192}, stream.options());
  muse_topo::BndParams a;
  a.stream = stream.data_ptr<float>();
  a.new_stream = new_stream.data_ptr<float>();
  a.out = out.data_ptr();
  a.out_fp32 = fp32_out ? out32.data_ptr<float>() : nullptr;
  a.alpha = alpha.data_ptr<float>();
  a.beta = beta.data_ptr<float>();
  a.post_w = post_w.has_value() ? post_w->data_ptr() : nullptr;
  a.pre_w = pre_w.data_ptr();
  a.post_eps = (float)post_eps;
  a.pre_eps = (float)pre_eps;
  a.post_gain = (float)post_gain;
  a.pre_gain = (float)pre_gain;
  auto s = c10::cuda::getCurrentCUDAStream().stream();
  const uint4* b = reinterpret_cast<const uint4*>(branch.data_ptr());
  const bool ps = post_w.has_value();
  if (ps && fp32_out) k8192<true, true><<<T, 256, 0, s>>>(a, b);
  else if (ps) k8192<true, false><<<T, 256, 0, s>>>(a, b);
  else if (fp32_out) k8192<false, true><<<T, 256, 0, s>>>(a, b);
  else k8192<false, false><<<T, 256, 0, s>>>(a, b);
  return {out, new_stream, out32};
}

torch::Tensor rms4096(torch::Tensor x, torch::Tensor w, double eps, double gain) {
  const int T = x.size(0);
  auto out = torch::empty_like(x);
  muse_topo::RmsParams a;
  a.out = out.data_ptr();
  a.w = w.data_ptr();
  a.eps = (float)eps;
  a.gain = (float)gain;
  k4096<<<T, 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(a, reinterpret_cast<const uint4*>(x.data_ptr()));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("boundary8192", &boundary8192);
  m.def("rms4096", &rms4096);
}
