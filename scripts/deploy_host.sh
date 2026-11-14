#!/bin/bash
# Sync the host folder to the server mirror (source of truth stays local).
# usage: [DEST=<server dir>] [FROM=<local dir>] scripts/deploy_host.sh [extra rsync args]
# FROM defaults to the folder this script lives in; DEST defaults to /raid/vllm-runtime/host_topo. Experiments that must
# not disturb the main mirror use another FROM copy and another DEST directory.
# Note: this syncs the host folder only; tools/deploy.sh syncs the vLLM source tree.
set -euo pipefail
here=${FROM:-$(cd "$(dirname "$0")/.." && pwd)}
dest=${DEST:-/raid/vllm-runtime/host_topo}
timeout 300 rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'build' \
  "$@" "$here/" "user@10.19.209.74:$dest/"
echo "synced $here -> $dest"
