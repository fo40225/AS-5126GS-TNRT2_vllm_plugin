#!/bin/bash
# Copy of /raid/vllm-runtime/musetest/locked_run.sh for the host folder (the original stays untouched).
# Usage: [SRC=<vllm tree>] [SEED=<n>] locked_run_topo.sh <tag> [extra vllm args...]  (env SEED, NCCL_*, VLLM_*, MUSE_TOPO_* pass through)
# Takes the GPU lock (LOCK_HELD=1: the caller already holds it, e.g. scripts/run_h2.sh inside one lock acquisition), waits (up to 5 min) until the previous holder's GPU memory is freed,
# then runs the standard Muse Spark EP8 dummy benchmark via run_src_topo.sh of the same host folder
# (results in /raid/vllm-runtime/musetest/results/<tag>/, summarize with musetest/summarize.py).
tag=$1; shift
export HOST_DIR=${HOST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd /raid/vllm-runtime/musetest
if [ "${LOCK_HELD:-0}" = 1 ]; then LOCK=(); else LOCK=(flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock); fi
exec "${LOCK[@]}" bash -c '
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s}")
  [ "${u:-0}" -lt 2000 ] && break; sleep 5
done
exec "$HOST_DIR/scripts/run_src_topo.sh" unsloth/Muse-Spark-Open-816B-A42B-NVFP4 "$0" --enable-expert-parallel --max-num-seqs 128 \
  --kv-cache-dtype fp8 --quantization modelopt_fp4 --limit-mm-per-prompt "{\"image\":0,\"audio\":0,\"video\":0}" \
  --gpu-memory-utilization 0.95 "$@"' "$tag" "$@"
