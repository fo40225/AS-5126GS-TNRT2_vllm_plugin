#!/bin/bash
# Production non-speculative launch: serve_nonspec.sh <ns128|ns256>, env/host.env (+ env/offload.env when the tier is ON):
#   ns256: --max-num-seqs 256 --max-cudagraph-capture-size 256 --kv-cache-memory-bytes $bytes [$MUSE_OFFLOAD_SERVE_ARGS]
#   ns128: --max-num-seqs 128 (the vLLM default capture list)    --kv-cache-memory-bytes $bytes [$MUSE_OFFLOAD_SERVE_ARGS]
# The NUMA-local KV tier is ON by default for ns256 (RECOMMENDED by the S1-10R verdict) and OFF by default for ns128 (SHIPPED as a profile without an A/B:
# MUSE_OFFLOAD=1 opts in); MUSE_OFFLOAD=0 / 1 overrides either. ON: the offload-on bytes and tier of env/offload.env (preflight first); OFF: the no-tier
# bytes of env/launch.env and nothing of env/offload.env in the environment. After READY scripts/check_offload_launch.sh and scripts/check_kv_launch.sh must pass.
# usage: [MUSE_OFFLOAD=0|1] launch/serve_nonspec.sh <ns128|ns256> [--out DIR] [--keep-on-fail] [--keep-env] [--dry-run]
# Details, exit codes and the inputs: launch/lib.sh.
LAUNCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$LAUNCH_DIR/lib.sh"
lc_init nonspec "$@"
lc_source_env
lc_main
