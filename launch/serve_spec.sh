#!/bin/bash
# Production speculative launch (RadixArk DFlash2 K = 7): env/host.env + env/spec_k7.env (which sources env/offload.env), the line
#   vllm serve ... $MUSE_SPEC_SERVE_ARGS --kv-cache-memory-bytes $MUSE_SPEC_KV_BYTES $MUSE_OFFLOAD_SERVE_ARGS
# MUSE_OFFLOAD defaults to 1 (the NUMA-local KV tier, S1-10R ADOPT); MUSE_OFFLOAD=0 is the previous launch (no tier, no --numa-bind, 11,720,212,480 B).
# Before a tier-ON boot scripts/preflight_offload.sh must pass (the launch is refused otherwise); after READY scripts/check_spec_launch.sh
# (SPEC_EXPECT_FB=1, SPEC_EXPECT_OFFLOAD=$MUSE_OFFLOAD), scripts/check_offload_launch.sh and scripts/check_kv_launch.sh must pass.
# usage: [MUSE_OFFLOAD=0|1] launch/serve_spec.sh [--out DIR] [--keep-on-fail] [--keep-env] [--dry-run]
# Run it under the GPU lock when the box is shared (flock -w 43200 /raid/vllm-runtime/sm120-dev/gpu.lock ...: the server keeps the lock while it runs).
# Details, exit codes and the inputs: launch/lib.sh.
LAUNCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$LAUNCH_DIR/lib.sh"
lc_init spec "$@"
lc_source_env
lc_main
