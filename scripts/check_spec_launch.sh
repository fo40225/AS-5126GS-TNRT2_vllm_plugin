#!/bin/bash
# Confirms in the log of a speculative boot that the launch profile of env/spec_k7.env took effect.
# Usage: [SPEC_EXPECT_MNBT=8199] check_spec_launch.sh <serve.log> [expected_sp_floor [expected_graph_cap]]    (defaults max(384, cap + 1) and this shell's MUSE_SPEC_GRAPH_CAP, else 256)
# The vLLM branch logs once per process: "sequence-parallel residual stream for batches of at least <floor> tokens
# (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=<env>, CUDA graphs up to <cap> tokens)". The floor is
# max(env, 257, cap + 1): the env value alone does not make a launch (floor 513 with the default capture list).
# When serve_cmd.txt (written by the harness) sits next to the log, it must carry --max-cudagraph-capture-size <cap> too
# (not checked for cap 512, the default capture list, which has no flag).
# The capture flag is accepted as "--max-cudagraph-capture-size 256", "--max-cudagraph-capture-size=256" or a
# max_cudagraph_capture_size entry of a -cc / --compilation-config JSON (also as printf %q wrote it into serve_cmd.txt: \"max_cudagraph_capture_size\":384).
# The log line is printed once per boot (rank 0); the env is inherited by every worker, so only that floor is checked.
# S1-9 lever 2: it also checks the PAIRING of the microbatched prefill (r2e) with the KV bytes. The vLLM branch logs "MuseSpark SM120: prefill
# batches of at least <n> tokens run as 2 microbatches ..." when the overlap is active (VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE=1 of the
# profile); next to a harness serve_cmd.txt the --kv-cache-memory-bytes must then not exceed R2E_ON_MAX_BYTES (10,373,877,760), and without the
# overlap not R2E_OFF_MAX_BYTES (10,526,351,360). SPEC_EXPECT_R2E=on|off additionally demands the state (default: either, paired).
# S1-9 L4: the same pairing for the DFlash2 aux projection on each rank's rows (VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD=2, the vLLM default since the L4 merge; env/spec_k7.env
# exports it). The branch logs "the aux projection runs in the target's forward (VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD=2)" when it runs; a launch that sets the flag but cannot
# apply it logs "DRAFT_FC_SHARD=2 does not apply" and no success line, and is NOT a flag-on launch (FAIL here). The projection moves ~726 MiB of the stress transients and
# idle memory per rank, so an r2e-on launch with it may carry --kv-cache-memory-bytes up to FCSHARD_ON_MAX_BYTES (11,134,914,560); without it (flag 0, an older tree, a
# fallback) at most R2E_ON_MAX_BYTES (10,373,877,760: with the larger value S_live falls to ~630 MiB, under the 1,256 MiB floor). SPEC_EXPECT_FCSHARD=on|off demands the state.
# S1-9 L5 (B3, aligned prefill chunks): the profile also carries --max-num-batched-tokens 8199 (K = 7: the chunks are 8192 tokens = a multiple of 8 instead of 8185, no F.pad staging
# copy in sp_reduce_scatter, no 7-token tail step; -2.7 .. -3.5 % TTFT at 16k / 32k). The vLLM line "Chunked prefill is enabled with max_num_batched_tokens=<n>" must read the expected value
# (SPEC_EXPECT_MNBT, default 8199; 8192 for a launch without the flag) and, next to a harness serve_cmd.txt, the flag must be there (8199) or absent (8192). The KV bytes of the profile
# (11,134,914,560) are the same for both values; the S_live of the S1-7 stress is 1,278 MiB with 8199 (1,356.7 with 8192).
# S1-9 L7 (the DFlash2 draft in FP8, VLLM_MUSE_SPARK_SM120_DRAFT_FP8 = 1 layers / 2 layers + FP8 candidate head, VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL = cutlass | marlin): the state is read from the
# success lines of the vLLM branch, "DFlash2 draft weights in FP8 (VLLM_MUSE_SPARK_SM120_DRAFT_FP8=<1|2>, kernel <marlin|cutlass>): 30 linear layers quantized" (one per worker) and, for flag 2, "DFlash2 candidate head in FP8
# (VLLM_MUSE_SPARK_SM120_DRAFT_FP8=2)". The hook logs a "does not apply" warning instead of the first line when a layer stays bf16 (quantized draft checkpoint, a shape the kernel does not take), and the head
# copy logs "does not apply to the head" when the head is not a plain bf16 matrix (flag 2 then runs as flag 1): both FAIL here, and so does a flag set in this shell's environment with no success line.
# The KV bytes depend on the state, not on the flag alone: the idle memory at identical bytes differs per (flag, kernel) (cutlass 2: -418 MiB, cutlass 1: -616, marlin 2: -542, marlin 1: -740 against the bf16
# draft), and the log arithmetic ("670 MiB freed") is not the allocator delta (cutlass 54 MiB less, marlin 70 MiB more), so the bytes are derived per state with the S1-7 procedure: FP8_F2_MAX_BYTES (flag 2),
# FP8_F1_MAX_BYTES (flag 1); a Marlin boot is checked against the cutlass value of its flag (it leaves 124 MiB more room: safe, not derived); a FP8 boot without the aux projection on the target's rows is checked
# against R2E_ON_MAX_BYTES. SPEC_EXPECT_FP8=off|1|2 and SPEC_EXPECT_FP8_KERNEL=cutlass|marlin demand the state (default: no expectation unless this shell exports the flag).
# S1-9 L11 (sequence parallel inside the FULL decode graphs, MUSE_SPEC_GRAPH_SP_MIN_TOKENS / MUSE_SPEC_GRAPH_CAP of env/spec_k7.env, both off / 256 by default): the vLLM branch logs, once per boot and only when the graph floor is on,
# "sequence parallel inside FULL decode CUDA graphs for batches of at least <g> tokens (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=<env>; decode MoE engines serve up to <E> tokens; eager, mixed and PIECEWISE batches keep the floor of <floor> tokens)".
# The state is read from that line, never from the env: SPEC_EXPECT_GRAPH_SP=off|<g> demands it (default: this shell's MUSE_SPEC_GRAPH_SP_MIN_TOKENS if it is exported and not 0; an expected value is raised to the engine limit E + 1 of the line, the vLLM
# rule: 129 in production), else no demand), the floor on the line must be the floor of the first line and max(<env>, E + 1), and a
# set VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS without the line (a warning of the vLLM branch: no graphs, or a tree without the feature) FAILS. The default expected floor is max(384, cap + 1) (the eager floor is cap + 1 above 383).
# The line alone does not prove that a graph is sharded (it is printed from the env, also with a PIECEWISE-only or FULL-mixed capture list): the graph manager also logs, once per worker,
# "CUDA graphs: <N> FULL decode graphs are flagged full_decode_graph (token counts <lo>..<hi>), <M> other FULL graphs, <K> PIECEWISE graphs." and an armed boot FAILS unless every worker logged the same line with N > 0 and <hi> >= the graph floor (a flagged FULL decode
# graph at or above the floor is what the sharded stream is captured in). The OK line prints "flagged=<N>@<lo>..<hi>".
# A cap above 256 (A1b: the capture list reaches cap, the graph pool grows) has no derived bytes: it is checked against SPEC_CAP_MAX_BYTES=<bytes> (the S1-7 value of that cap) and FAILS without it; at cap 256 the graph floor changes no bytes (the pool shrinks 0.95 -> 0.89 GiB: the
# production maximum stays safe, exact bytes with the smaller pool are the gate's job).
# Exit 0 = profile in effect; 1 = a line with other values, no line, the flag is missing or the pairing is wrong; 2 = usage or unreadable log.
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  echo "usage: check_spec_launch.sh <serve.log> [expected_sp_floor [expected_graph_cap]]" >&2; exit 2
fi
log=$1
cap=${3:-${MUSE_SPEC_GRAPH_CAP:-256}}
# the eager floor is max(MIN_TOKENS 384, 257, cap + 1)
default_floor=384
[ "$cap" -gt 383 ] && default_floor=$((cap + 1))
floor=${2:-$default_floor}
[ -r "$log" ] || { echo "check_spec_launch: cannot read $log" >&2; exit 2; }
pattern='sequence-parallel residual stream for batches of at least [0-9]+ tokens \(VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=[0-9]+, CUDA graphs up to [0-9]+ tokens\)'
seen=$(grep -a -o -E "$pattern" "$log" | sed -E 's/.*at least ([0-9]+) tokens.*CUDA graphs up to ([0-9]+) tokens\)/\1 \2/' | sort | uniq -c)
if [ -z "$seen" ]; then
  echo "FAIL no sequence-parallel line in $log (flag off, another model, or the boot did not get that far)"; exit 1
fi
if [ "$(echo "$seen" | wc -l)" != 1 ] || ! echo "$seen" | grep -q -E "^ *[0-9]+ $floor $cap\$"; then
  echo "FAIL expected floor $floor with graphs up to $cap; found (count floor cap):"; echo "$seen"; exit 1
fi
pool=$(grep -a -o -E 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB' "$log" | sort | uniq -c | tr '\n' ';')
rc=0
cmd=$(dirname "$log")/serve_cmd.txt
flag_re="(--max-cudagraph-capture-size[ =]|max_cudagraph_capture_size[\\\"' :=]+)$cap([^0-9]|$)"
if [ "$cap" != 512 ] && [ -r "$cmd" ] && ! grep -q -E -e "$flag_re" "$cmd"; then
  echo "FAIL serve_cmd.txt has no --max-cudagraph-capture-size $cap"; rc=1
fi
# sequence parallel inside the FULL decode graphs (S1-9 L11): the state comes from the log line
graph_re='sequence parallel inside FULL decode CUDA graphs for batches of at least [0-9]+ tokens \(VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=[0-9]+; decode MoE engines serve up to [0-9]+ tokens; eager, mixed and PIECEWISE batches keep the floor of [0-9]+ tokens\)'
gsp_seen=$(grep -a -o -E "$graph_re" "$log" | sed -E 's/.*at least ([0-9]+) tokens.*MIN_TOKENS=([0-9]+); decode MoE engines serve up to ([0-9]+) tokens;.*floor of ([0-9]+) tokens\)/\1 \2 \3 \4/' | sort | uniq -c)
gsp=off
gsp_env=0
gsp_engine=
if [ -n "$gsp_seen" ]; then
  if [ "$(echo "$gsp_seen" | wc -l)" != 1 ]; then
    echo "FAIL the workers disagree on the graph floor (count graph_floor env engine_max eager_floor):"; echo "$gsp_seen"; rc=1
  else
    gsp=$(echo "$gsp_seen" | awk '{print $2}'); gsp_env=$(echo "$gsp_seen" | awk '{print $3}'); gsp_engine=$(echo "$gsp_seen" | awk '{print $4}'); gsp_eager=$(echo "$gsp_seen" | awk '{print $5}')
    if [ "$gsp_eager" != "$floor" ]; then
      echo "FAIL the graph floor line names the eager floor $gsp_eager, expected $floor"; rc=1
    fi
    if [ "$gsp" -le "$gsp_engine" ]; then
      echo "FAIL graph floor $gsp is not above the decode MoE engines' $gsp_engine tokens"; rc=1
    fi
    want_floor=$(( gsp_env > gsp_engine + 1 ? gsp_env : gsp_engine + 1 ))
    if [ "$gsp" != "$want_floor" ]; then
      echo "FAIL graph floor $gsp on the line is not max(env $gsp_env, engine limit $gsp_engine + 1) = $want_floor"; rc=1
    fi
  fi
fi
want_gsp=${SPEC_EXPECT_GRAPH_SP:-}
if [ -z "$want_gsp" ] && [ "${MUSE_SPEC_GRAPH_SP_MIN_TOKENS:-0}" != 0 ]; then
  want_gsp=$MUSE_SPEC_GRAPH_SP_MIN_TOKENS
fi
if [ -n "$want_gsp" ] && [ "$want_gsp" != off ]; then
  # an expected value is raised to the engine limit + 1 of the boot, as the vLLM branch does (the limit is on the line)
  eng=${gsp_engine:-128}
  [ "$want_gsp" -le "$eng" ] && want_gsp=$((eng + 1))
fi
if [ -n "$want_gsp" ] && [ "$want_gsp" != "$gsp" ]; then
  echo "FAIL the graph floor is $gsp in the log, expected $want_gsp (the flag was set but the boot logged no graph-floor line: no CUDA graphs, or a tree without the feature)"; rc=1
fi
if [ -z "$gsp_seen" ] && [ "${VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS:-0}" != 0 ]; then
  echo "FAIL VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS=$VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_GRAPH_MIN_TOKENS is set in this shell but the boot logged no graph-floor line"; rc=1
fi
# the graph manager's line: how many FULL decode graphs run the model with the flag (an armed floor needs one at or above it)
flagged_re='CUDA graphs: [0-9]+ FULL decode graphs are flagged full_decode_graph \(token counts (none|[0-9]+\.\.[0-9]+)\), [0-9]+ other FULL graphs, [0-9]+ PIECEWISE graphs\.'
flag_seen=$(grep -a -o -E "$flagged_re" "$log" | sed -E 's/CUDA graphs: ([0-9]+) FULL decode graphs are flagged full_decode_graph \(token counts (none|[0-9]+)(\.\.([0-9]+))?\),.*/\1 \2 \4/' | sort | uniq -c)
flagged=-
if [ -n "$flag_seen" ]; then
  if [ "$(echo "$flag_seen" | wc -l)" != 1 ]; then
    echo "FAIL the workers disagree on the number of flagged FULL decode graphs (count flagged lo hi):"; echo "$flag_seen"; rc=1
  else
    n_flag_lines=$(echo "$flag_seen" | awk '{print $1}')
    n_flagged=$(echo "$flag_seen" | awk '{print $2}'); flag_lo=$(echo "$flag_seen" | awk '{print $3}'); flag_hi=$(echo "$flag_seen" | awk '{print $4}')
    [ "$n_flagged" = 0 ] && flagged=0 || flagged="$n_flagged@$flag_lo..$flag_hi"
    n_workers_log=$(grep -a -c 'Model loading took' "$log")
    if [ "$n_workers_log" -gt 0 ] && [ "$n_flag_lines" != "$n_workers_log" ]; then
      echo "FAIL $n_flag_lines of $n_workers_log workers logged the flagged-graph line"; rc=1
    fi
  fi
fi
if [ "$gsp" != off ]; then
  if [ -z "$flag_seen" ]; then
    echo "FAIL the graph floor is armed ($gsp) but the boot logged no flagged-graph line (\"CUDA graphs: <N> FULL decode graphs are flagged ...\"): a tree without it, or the capture did not run"; rc=1
  elif [ "$flagged" = 0 ]; then
    echo "FAIL the graph floor is armed ($gsp) but no FULL decode graph is captured (flagged 0): the floor is inert; the cudagraph mode lacks FULL decode graphs"; rc=1
  elif [ "${flag_hi:-0}" -lt "$gsp" ]; then
    echo "FAIL the graph floor is armed ($gsp) but the largest flagged FULL decode graph has $flag_hi tokens: no captured graph is sharded"; rc=1
  fi
fi
# a capture list above 256 tokens has no derived bytes
cap_bytes_note=
if [ "$cap" -gt 256 ]; then
  if [ -z "${SPEC_CAP_MAX_BYTES:-}" ]; then
    echo "FAIL graphs up to $cap tokens (> 256): the graph pool is larger than the one of the derived bytes and no exact bytes exist for this capture list; pass SPEC_CAP_MAX_BYTES=<S1-7 bytes of this cap>"; rc=1
  else
    cap_bytes_note=" cap-max-bytes=$SPEC_CAP_MAX_BYTES"
  fi
fi
# r2e state of the boot and its pairing with the KV bytes
R2E_ON_MAX_BYTES=${R2E_ON_MAX_BYTES:-10373877760}
R2E_OFF_MAX_BYTES=${R2E_OFF_MAX_BYTES:-10526351360}
r2e=off
grep -a -q -E 'prefill batches of at least [0-9]+ tokens run as [0-9]+ microbatches' "$log" && r2e=on
if [ -n "${SPEC_EXPECT_R2E:-}" ] && [ "$SPEC_EXPECT_R2E" != "$r2e" ]; then
  echo "FAIL prefill overlap (r2e) is $r2e in the log, expected $SPEC_EXPECT_R2E"; rc=1
fi
# aux projection state of the boot (S1-9 L4)
FCSHARD_ON_MAX_BYTES=${FCSHARD_ON_MAX_BYTES:-11134914560}
fcs=off
grep -a -q "the aux projection runs in the target's forward" "$log" && fcs=on
if grep -a -q "DRAFT_FC_SHARD=2 does not apply" "$log"; then
  echo "FAIL the aux projection fell back to the propose step (DRAFT_FC_SHARD=2 does not apply): not a flag-on launch"; rc=1
fi
if [ -n "${SPEC_EXPECT_FCSHARD:-}" ] && [ "$SPEC_EXPECT_FCSHARD" != "$fcs" ]; then
  echo "FAIL aux projection on the target's rows is $fcs in the log, expected $SPEC_EXPECT_FCSHARD"; rc=1
fi
# DFlash2 draft in FP8 (S1-9 L7): the state comes from the success lines, never from the flag
FP8_F1_MAX_BYTES=${FP8_F1_MAX_BYTES:-11781017600}   # flag 1 (layers), cutlass: S1-7 stress S_live 1,278.4 MiB, 1,658,495 KV tokens per rank
FP8_F2_MAX_BYTES=${FP8_F2_MAX_BYTES:-11573411840}   # flag 2 (layers + FP8 head), cutlass: S1-7 stress S_live 1,278.5 MiB, 1,629,269 KV tokens per rank
fp8_layers_re='DFlash2 draft weights in FP8 \(VLLM_MUSE_SPARK_SM120_DRAFT_FP8=[12], kernel (marlin|cutlass)\): 30 linear layers quantized'
fp8=off; fp8k=-
fp8_states=$(grep -a -o -E "$fp8_layers_re" "$log" | sed -E 's/.*FP8=([12]), kernel ([a-z]+)\).*/\1 \2/' | sort | uniq -c)
workers=$(grep -a -c 'Model loading took' "$log")
if [ -n "$fp8_states" ]; then
  if [ "$(echo "$fp8_states" | wc -l)" != 1 ]; then
    echo "FAIL the workers disagree on the FP8 draft state (count flag kernel):"; echo "$fp8_states"; rc=1
  else
    fp8=$(echo "$fp8_states" | awk '{print $2}'); fp8k=$(echo "$fp8_states" | awk '{print $3}')
    n_layers=$(echo "$fp8_states" | awk '{print $1}')
    if [ "$workers" -gt 0 ] && [ "$n_layers" != "$workers" ]; then
      echo "FAIL $n_layers of $workers workers logged the FP8 draft success line"; rc=1
    fi
    n_head=$(grep -a -c -E 'DFlash2 candidate head in FP8 \(VLLM_MUSE_SPARK_SM120_DRAFT_FP8=2\)' "$log")
    if [ "$fp8" = 2 ] && [ "$n_head" != "$n_layers" ]; then
      echo "FAIL flag 2 logged the layers line $n_layers times but the candidate head line $n_head times (the head copy fell back: that is flag 1)"; rc=1
    elif [ "$fp8" = 1 ] && [ "$n_head" != 0 ]; then
      echo "FAIL flag 1 logged $n_head candidate head lines"; rc=1
    fi
  fi
fi
if grep -a -q -E 'DRAFT_FP8=[12] does not apply' "$log"; then
  echo "FAIL the FP8 draft did not apply in full (VLLM_MUSE_SPARK_SM120_DRAFT_FP8=<n> does not apply: a layer or the head stayed bf16): not an FP8 launch"; rc=1
fi
want8=${SPEC_EXPECT_FP8:-}
if [ -z "$want8" ] && [ "${VLLM_MUSE_SPARK_SM120_DRAFT_FP8:-0}" != 0 ]; then want8=$VLLM_MUSE_SPARK_SM120_DRAFT_FP8; fi
if [ -n "$want8" ] && [ "$want8" != "$fp8" ]; then
  echo "FAIL the FP8 draft is $fp8 in the log, expected $want8 (the flag was set but the success line is missing, or the head copy fell back to flag 1)"; rc=1
fi
if [ -n "${SPEC_EXPECT_FP8_KERNEL:-}" ] && [ "$fp8" != off ] && [ "$SPEC_EXPECT_FP8_KERNEL" != "$fp8k" ]; then
  echo "FAIL the FP8 draft kernel is $fp8k in the log, expected $SPEC_EXPECT_FP8_KERNEL"; rc=1
fi
if [ -r "$cmd" ]; then
  bytes=$(grep -a -o -E -e '--kv-cache-memory-bytes[ =][0-9]+' "$cmd" | head -1 | grep -o -E '[0-9]+$')
  if [ -n "$bytes" ]; then
    max=$R2E_OFF_MAX_BYTES
    if [ "$r2e" = on ]; then
      max=$R2E_ON_MAX_BYTES
      if [ "$fcs" = on ]; then
        max=$FCSHARD_ON_MAX_BYTES
        [ "$fp8" = 1 ] && max=$FP8_F1_MAX_BYTES
        [ "$fp8" = 2 ] && max=$FP8_F2_MAX_BYTES
      fi
    fi
    # a capture list above 256 tokens: the bytes derived for that cap replace the cap-256 ones
    [ "$cap" -gt 256 ] && [ -n "${SPEC_CAP_MAX_BYTES:-}" ] && max=$SPEC_CAP_MAX_BYTES
    if [ "$bytes" -gt "$max" ]; then
      echo "FAIL r2e $r2e, aux projection on the target's rows $fcs, FP8 draft $fp8/$fp8k, with --kv-cache-memory-bytes $bytes > $max (the exact bytes are derived per state: without the projection on the target's rows, unset flag 0 or an older tree, at most $R2E_ON_MAX_BYTES; FP8 flag 1 / 2: FP8_F1_MAX_BYTES $FP8_F1_MAX_BYTES / FP8_F2_MAX_BYTES $FP8_F2_MAX_BYTES)"; rc=1
    fi
  fi
fi
# prefill chunk (S1-9 L5, B3)
mnbt=${SPEC_EXPECT_MNBT:-8199}
seen_mnbt=$(grep -a -o -E 'Chunked prefill is enabled with max_num_batched_tokens=[0-9]+' "$log" | sed -E 's/.*=//' | sort | uniq -c)
if [ -z "$seen_mnbt" ]; then
  echo "FAIL no \"Chunked prefill is enabled with max_num_batched_tokens=\" line in $log (expected $mnbt)"; rc=1
elif [ "$(echo "$seen_mnbt" | wc -l)" != 1 ] || ! echo "$seen_mnbt" | grep -q -E "^ *[0-9]+ $mnbt\$"; then
  echo "FAIL expected max_num_batched_tokens=$mnbt; found (count value):"; echo "$seen_mnbt"; rc=1
fi
if [ -r "$cmd" ]; then
  if [ "$mnbt" = 8192 ]; then
    if grep -q -E -e '--max-num-batched-tokens[ =]' "$cmd" && ! grep -q -E -e '--max-num-batched-tokens[ =]8192([^0-9]|$)' "$cmd"; then
      echo "FAIL serve_cmd.txt carries a --max-num-batched-tokens other than 8192"; rc=1
    fi
  elif ! grep -q -E -e "--max-num-batched-tokens[ =]$mnbt([^0-9]|\$)" "$cmd"; then
    echo "FAIL serve_cmd.txt has no --max-num-batched-tokens $mnbt"; rc=1
  fi
fi
[ $rc = 0 ] && echo "OK floor=$floor graphs<=$cap graphsp=$gsp flagged=$flagged r2e=$r2e fcshard=$fcs fp8=$fp8/$fp8k mnbt=$mnbt$cap_bytes_note ($pool)"
exit $rc
