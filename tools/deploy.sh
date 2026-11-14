#!/bin/bash
# Sync the working tree of /home/user/vllm (tracked + untracked, non-ignored files under vllm/, tests/)
# into the server source tree, keeping the precompiled artifacts there untouched.
set -euo pipefail
cd /home/user/vllm
git ls-files -z --cached --others --exclude-standard -- vllm tests benchmarks \
  ':!vllm/vllm_flash_attn' ':!vllm/third_party' |
  timeout 300 rsync -a --from0 --files-from=- --ignore-missing-args ./ \
    user@10.19.209.74:/raid/vllm-runtime/sm120-dev/src/
echo "deployed $(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)"
