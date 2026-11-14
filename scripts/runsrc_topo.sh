#!/bin/bash
# Copy of /raid/vllm-runtime/sm120-dev/runsrc.sh (the original stays untouched) for the host folder: runs python from the
# dev venv against a vLLM source tree with the host folder on PYTHONPATH (muse_topo plugin discovery) and env/host.env.
# usage: [SRC=<tree>] [GPU=<ids>] [HOST_DIR=<host folder>] runsrc_topo.sh <python args...>
# The host folder defaults to the one this script lives in. Uses the shared dev caches (JIT builds of muse_c, Triton, and
# muse_topo under $VLLM_CACHE_ROOT/muse_topo/<digest>); scratch output belongs to the caller (server: work/host/).
HOST=${HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/raid/vllm-runtime/sm120-dev/.venv/bin:$PATH
SRC=${SRC:-/raid/vllm-runtime/sm120-dev/src}
export PYTHONPATH=$HOST:$SRC${PYTHONPATH:+:$PYTHONPATH}
export FLASHINFER_WORKSPACE_BASE=/raid/vllm-runtime/sm120-dev/fi_workspace
export TRITON_CACHE_DIR=/raid/vllm-runtime/sm120-dev/triton_cache
export CUTE_DSL_CACHE_DIR=/raid/vllm-runtime/sm120-dev/cute_cache
export VLLM_CACHE_ROOT=/raid/vllm-runtime/sm120-dev/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/raid/vllm-runtime/sm120-dev/inductor_cache
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
source "$HOST/env/host.env"
cd "$SRC"
exec /raid/vllm-runtime/sm120-dev/.venv/bin/python "$@"
