#!/bin/bash
# STATIC EQUIVALENCE PROOFS of the launch scripts (S1-11 Amendment 2): for every script path the serve line, the exported environment and the environment the server process
# would inherit are byte-equal to the gated / adoption boot of the same launch (references and provenance in bench/fixtures/launch_ref/). The scripts run with --dry-run
# (the real code path up to the launch: scrub, env files, serve line, the same writers and filters as the gated boots), so no GPU and no vLLM are needed.
# Also: the env files themselves export exactly what the base commit's did (export -p, sorted), and negative controls show that the comparisons can fail.
# The one normalization: the checkout path -> /raid/vllm-runtime/host_topo (the references name the server mirror in PYTHONPATH and NCCL_GRAPH_FILE).
# Usage  bash bench/test_launch_equiv.sh   Exit 0 = all equal.
here=$(cd "$(dirname "$0")/.." && pwd)
ref=$here/bench/fixtures/launch_ref
d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
fails=0
norm() { sed "s#$here#/raid/vllm-runtime/host_topo#g" "$1"; }
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; shift; for l in "$@"; do echo "    $l"; done; fails=$((fails + 1)); }
# dry <name> <script> <profile arg or ''> [env assignments...]: the records of a clean-shell dry run in $d/<name>
dry() {
  local name=$1 script=$2 parg=$3; shift 3
  env -i PATH="$PATH" HOME="$HOME" "$@" bash "$here/launch/$script" $parg --dry-run --out "$d/$name" > "$d/$name.stdout" 2>&1 || echo "dry run $name failed: $(tail -n 2 "$d/$name.stdout")"
}
# prove <state> <script> <profile arg> <desc> [env assignments...]
prove() {
  local state=$1 script=$2 parg=$3 desc=$4; shift 4
  local name="${state}_$RANDOM"
  dry "$name" "$script" "$parg" "$@"
  if cmp -s "$d/$name/serve_cmd.txt" "$ref/${state}.serve_cmd.txt"; then :; else
    # tolerate nothing but the path: serve lines carry no checkout path, so a plain cmp must already hold
    bad "$desc: serve_cmd.txt differs from the gated boot's" "$(diff <(cat "$d/$name/serve_cmd.txt") "$ref/${state}.serve_cmd.txt" | head -4)"; return
  fi
  if [ -s "$ref/${state}.env.txt" ]; then
    if norm "$d/$name/env.txt" | cmp -s - "$ref/${state}.env.txt"; then :; else bad "$desc: env.txt differs from the gated boot's" "$(diff <(norm "$d/$name/env.txt") "$ref/${state}.env.txt" | head -6)"; return; fi
  fi
  if [ -s "$ref/${state}.pid_environ.txt" ]; then
    if norm "$d/$name/pid_environ_expected.txt" | cmp -s - "$ref/${state}.pid_environ.txt"; then :; else bad "$desc: the server's environment differs from the gated boot's pid_environ.txt" "$(diff <(norm "$d/$name/pid_environ_expected.txt") "$ref/${state}.pid_environ.txt" | head -6)"; return; fi
  fi
  ok "$desc (serve line$([ -s "$ref/${state}.env.txt" ] && echo ", env.txt")$([ -s "$ref/${state}.pid_environ.txt" ] && echo ", pid environ") byte-equal)"
}
prove spec_1 serve_spec.sh "" "spec, tier ON (default) = adopt_spec_on"
prove spec_1 serve_spec.sh "" "spec, MUSE_OFFLOAD=1 = adopt_spec_on" MUSE_OFFLOAD=1
prove spec_0 serve_spec.sh "" "spec, MUSE_OFFLOAD=0 = adopt_spec_off (and the S1-14 production spec line)" MUSE_OFFLOAD=0
prove ns256_1 serve_nonspec.sh ns256 "ns256, tier ON (default) = adopt_ns256"
prove ns256_1 serve_nonspec.sh ns256 "ns256, MUSE_OFFLOAD=1 = adopt_ns256" MUSE_OFFLOAD=1
prove ns128_1 serve_nonspec.sh ns128 "ns128, MUSE_OFFLOAD=1 = adopt_ns128" MUSE_OFFLOAD=1
prove ns128_0 serve_nonspec.sh ns128 "ns128, no tier (default) = S1-14 production pb_ns128"
prove ns128_0 serve_nonspec.sh ns128 "ns128, MUSE_OFFLOAD=0 = S1-14 production pb_ns128" MUSE_OFFLOAD=0
# ns256 OFF: the serve line is byte-equal to the 0.33.0 validation boot; that boot recorded no env.txt, so its environment is proved against the S1-14 host.env-only record
# (same launch: host.env, no profile file) and against the flags line of its status.txt; the one difference is VLLM_PLUGINS=muse_topo (host a26c2be, after that boot)
prove ns256_0 serve_nonspec.sh ns256 "ns256, MUSE_OFFLOAD=0 = 0.33.0 validation boot v033s3_ns256_1 (serve line)" MUSE_OFFLOAD=0
dry ns256_off serve_nonspec.sh ns256 MUSE_OFFLOAD=0
if norm "$d/ns256_off/env.txt" | cmp -s - "$ref/ns128_0.env.txt"; then ok "ns256 no tier: env.txt byte-equal to the S1-14 host.env-only record (ns128_0)"; else bad "ns256 no tier: env.txt vs the S1-14 record" "$(diff <(norm "$d/ns256_off/env.txt") "$ref/ns128_0.env.txt" | head -4)"; fi
# the serve line of the newer S1-10R gate boot s110r_ns256_OFF_1 (host a26c2be+) is the same line
if cmp -s "$d/ns256_off/serve_cmd.txt" "$ref/ns256_0b.serve_cmd.txt"; then ok "ns256 no tier: serve line byte-equal to the S1-10R gate boot s110r_ns256_OFF_1 too"; else bad "ns256 no tier: serve line vs s110r_ns256_OFF_1"; fi
for fl in ns256_0 ns256_0b; do
  flags=$(sed -E 's/^[^ ]+ [^ ]+ flags //; s#host_topo_v033#host_topo#; s#host_topo_s110r#host_topo#' "$ref/$fl.flags_line.txt")
  miss=0; for kv in $flags; do norm "$d/ns256_off/env.txt" | grep -q -x -F -- "$kv" || { miss=1; echo "    flags-line entry not in the script environment: $kv"; }; done
  extra=$(norm "$d/ns256_off/env.txt" | grep -v -x -F -f <(echo "$flags" | tr ' ' '\n') | tr '\n' ' ')
  # the extras: PYTHONPATH is outside the filter of that boot's flags line; VLLM_PLUGINS is outside it too (set since host a26c2be, so not set in v033s3_ns256_1, set in s110r_ns256_OFF_1)
  if [ $miss = 0 ] && [ "$extra" = "PYTHONPATH=/raid/vllm-runtime/host_topo:/raid/vllm-runtime/sm120-dev/src VLLM_PLUGINS=muse_topo " ]; then
    ok "ns256 no tier: every entry of the flags line of $fl is in the environment; the extras are PYTHONPATH and VLLM_PLUGINS=muse_topo (not in that line's filter)"
  else bad "ns256 no tier vs the flags line of $fl" "extra: '$extra'"; fi
done
# the scripts from a dirty calling shell give the same records (the scrub): the operator inputs MUSE_OFFLOAD aside, nothing inherited changes the launch
dirty=(MUSE_SPEC_MNST=0 MUSE_SPEC_GRAPH_CAP=384 MUSE_TOPO_ENABLE=0 MUSE_OFFLOAD_PER_RANK_BYTES=1 MUSE_OFFLOAD_NUMA_BIND=0 MUSE_SPEC_KV_BYTES_OVERRIDE=1 MUSE_OFFLOAD_KV_BYTES_OVERRIDE=1
  VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY=0 VLLM_MUSE_SPARK_SM120_DRAFT_FP8=0 VLLM_SERVER_DEV_MODE=0 SPEC_EXPECT_FB=0 NCCL_P2P_LEVEL=NVL NCCL_IB_DISABLE=0 NCCL_GRAPH_FILE=/x VLLM_PLUGINS=other _MUSE_X=1
  NCCL_ALGO=Ring NCCL_DEBUG=INFO NCCL_PROTO=Simple NCCL_SOCKET_IFNAME=lo VLLM_ATTENTION_BACKEND=X VLLM_LOGGING_LEVEL=DEBUG VLLM_USE_V1=0 VLLM_TORCH_PROFILER_DIR=/x CUDA_VISIBLE_DEVICES=0)
prove spec_1 serve_spec.sh "" "spec ON from a dirty shell" "${dirty[@]}"
prove spec_0 serve_spec.sh "" "spec OFF from a dirty shell" "${dirty[@]}" MUSE_OFFLOAD=0
prove ns256_1 serve_nonspec.sh ns256 "ns256 ON from a dirty shell" "${dirty[@]}"
prove ns128_0 serve_nonspec.sh ns128 "ns128 no tier from a dirty shell" "${dirty[@]}"
prove ns128_1 serve_nonspec.sh ns128 "ns128 ON from a dirty shell" "${dirty[@]}" MUSE_OFFLOAD=1
# negative controls: the comparisons can fail
dry nc_on serve_spec.sh "" MUSE_OFFLOAD=1; dry nc_off serve_spec.sh "" MUSE_OFFLOAD=0
cmp -s "$d/nc_on/serve_cmd.txt" "$d/nc_off/serve_cmd.txt" && bad "negative control: spec ON and OFF serve lines are equal" || ok "negative control: spec ON and OFF serve lines differ"
cmp -s "$d/nc_on/env.txt" "$d/nc_off/env.txt" && bad "negative control: spec ON and OFF env.txt are equal" || ok "negative control: spec ON and OFF env.txt differ"
env -i PATH="$PATH" HOME="$HOME" MUSE_SPEC_MNST=0 bash "$here/launch/serve_spec.sh" --keep-env --dry-run --out "$d/nc_keep2" > /dev/null 2>&1
cmp -s "$d/nc_keep2/serve_cmd.txt" "$ref/spec_1.serve_cmd.txt" && bad "negative control: --keep-env with MUSE_SPEC_MNST=0 gives the adoption line" || ok "negative control: --keep-env lets a dirty shell change the line (the scrub is what protects the default)"
# the same leak classes are visible in the records when the scrub is off (the unnamed NCCL_* / VLLM_* variables are in env.txt and in the environment the server would inherit)
env -i PATH="$PATH" HOME="$HOME" NCCL_ALGO=Ring VLLM_ATTENTION_BACKEND=X bash "$here/launch/serve_spec.sh" --keep-env --dry-run --out "$d/nc_keep3" > /dev/null 2>&1
if grep -q '^NCCL_ALGO=Ring' "$d/nc_keep3/env.txt" && grep -q '^VLLM_ATTENTION_BACKEND=X' "$d/nc_keep3/pid_environ_expected.txt"; then ok "negative control: --keep-env carries an unnamed NCCL_* / VLLM_* export into env.txt and the expected server environment"
else bad "negative control: --keep-env with NCCL_ALGO / VLLM_ATTENTION_BACKEND"; fi
dry nc_ns128 serve_nonspec.sh ns128 MUSE_OFFLOAD=0; dry nc_ns256 serve_nonspec.sh ns256 MUSE_OFFLOAD=0
cmp -s "$d/nc_ns128/serve_cmd.txt" "$d/nc_ns256/serve_cmd.txt" && bad "negative control: ns128 and ns256 serve lines are equal" || ok "negative control: ns128 and ns256 serve lines differ"
# ---- the env files export exactly what the base commit's did (sourcing order of the gated boots), for every path
base=da1bdad519
if git -C "$here" cat-file -e "$base^{commit}" 2> /dev/null && mkdir -p "$d/base" && git -C "$here" archive "$base" env nccl | tar -x -C "$d/base"; then
  exps() { env -i PATH="$PATH" "${@:2}" bash -c "$1" 2> /dev/null | grep -v -E '^declare -x (PWD|OLDPWD|SHLVL|_)=' | sort; }
  cmpexp() { # <desc> <env assignments...> -- <shell code with $E>
    local desc=$1; shift; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local a b; a=$(exps "E=$here/env; $1" "${envs[@]}" | sed "s#$here#X#g"); b=$(exps "E=$d/base/env; $1" "${envs[@]}" | sed "s#$d/base#X#g")
    if [ -n "$a" ] && [ "$a" = "$b" ]; then ok "exports equal to $base: $desc"; else bad "exports differ from $base: $desc" "$(diff <(echo "$a") <(echo "$b") | head -4)"; fi
  }
  cmpexp "spec ON (host.env + spec_k7.env)" -- 'HOST=$E/..; source $E/host.env; source $E/spec_k7.env; export -p'
  cmpexp "spec OFF" MUSE_OFFLOAD=0 -- 'HOST=$E/..; source $E/host.env; source $E/spec_k7.env; export -p'
  cmpexp "ns256 ON (host.env + offload.env)" MUSE_OFFLOAD_PROFILE=ns256 -- 'HOST=$E/..; source $E/host.env; source $E/offload.env; export -p'
  cmpexp "ns128 ON" MUSE_OFFLOAD_PROFILE=ns128 -- 'HOST=$E/..; source $E/host.env; source $E/offload.env; export -p'
  cmpexp "spec ON, gate order (offload.env first)" -- 'HOST=$E/..; source $E/host.env; source $E/offload.env; source $E/spec_k7.env; export -p'
  cmpexp "child shell, parent ON exports, ON again" -- 'HOST=$E/..; source $E/host.env; source $E/spec_k7.env; bash -c "HOST=\$E/..; source \$E/host.env; source \$E/spec_k7.env; export -p" 2>/dev/null'
  cmpexp "child shell, parent ON exports, MUSE_OFFLOAD=0" -- 'HOST=$E/..; source $E/host.env; source $E/spec_k7.env; export E; MUSE_OFFLOAD=0 bash -c "HOST=\$E/..; source \$E/host.env; source \$E/spec_k7.env; export -p"'
  # the launch.env file adds nothing to the environment
  a=$(env -i PATH="$PATH" bash -c "a=\$(export -p | sort); source $here/env/launch.env; b=\$(export -p | sort); [ \"\$a\" = \"\$b\" ] && echo same" 2> /dev/null)
  if [ "$a" = same ]; then ok "env/launch.env exports nothing"; else bad "env/launch.env changes the exported environment"; fi
else echo "skip: base commit $base not available in this checkout (the main history must stay reachable for the base-export proofs)"; fi
echo "$fails case(s) failed"
exit $(( fails > 0 ))
