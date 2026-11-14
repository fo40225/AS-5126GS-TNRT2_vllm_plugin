#!/bin/bash
# Usage: deploy_to.sh <local_git_tree> <remote_src_dir>
# Sync tracked + untracked (non-ignored) files under vllm/ tests/ benchmarks/ of a local
# vLLM checkout or worktree into a server source tree. The remote tree must already exist
# (create one with: ssh user@10.19.209.74 cp -al /raid/vllm-runtime/sm120-dev/src <dir>);
# rsync replaces changed files with new inodes, so hard-linked siblings stay untouched.
set -euo pipefail
tree=$1; dest=$2
cd "$tree"
timeout 30 ssh user@10.19.209.74 "test -d '$dest/vllm'" || { echo "missing remote tree $dest" >&2; exit 2; }
git ls-files -z --cached --others --exclude-standard -- vllm tests benchmarks \
  ':!vllm/vllm_flash_attn' ':!vllm/third_party' |
  timeout 300 rsync -a --from0 --files-from=- --ignore-missing-args ./ \
    "user@10.19.209.74:$dest/"
echo "deployed $(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty) -> $dest"
