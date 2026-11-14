#!/bin/bash
# Server side: run python from the dev venv with the host folder on PYTHONPATH and private JIT caches.
# usage: [GPU=<ids>] [SRC=<vllm tree>] py_topo.sh <python args...>
# Scratch (JIT builds, caches, logs) lives under $MUSE_TEST_SCRATCH (created here), never in the host folder.
HOST=${HOST_DIR:-/raid/vllm-runtime/host_topo}
W=${MUSE_TEST_SCRATCH:-/raid/vllm-runtime/work/plugin-tests}
export CUDA_VISIBLE_DEVICES=${GPU:-0,1,2,3,4,5,6,7}
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/raid/vllm-runtime/sm120-dev/.venv/bin:$PATH
export PYTHONPATH=$HOST${SRC:+:$SRC}${PYTHONPATH:+:$PYTHONPATH}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$W/vllm_cache}
export TORCH_EXTENSIONS_DIR=$W/torch_ext
export TRITON_CACHE_DIR=$W/triton_cache
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
# NCCL >= 2.30 connects the cross-socket hops over the NIC (observed in RoCE mode; IB mode since 2026-11-10, NCCL_IB_DISABLE=1 still required) and fails to init unless IB is off (see env/host.env).
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export MUSE_TOPO_ENABLE=${MUSE_TOPO_ENABLE:-1}
mkdir -p "$W/logs"
exec /raid/vllm-runtime/sm120-dev/.venv/bin/python "$@"
