#!/bin/bash
# Confirms in the log of a speculative boot that the launch profile of env/spec_k7.env took effect.
# Usage: check_spec_launch.sh <serve.log> [expected_sp_floor [expected_graph_cap]]    (defaults 384 and 256)
# The vLLM branch logs once per process: "sequence-parallel residual stream for batches of at least <floor> tokens
# (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS=<env>, CUDA graphs up to <cap> tokens)". The floor is
# max(env, 257, cap + 1): the env value alone does not make a launch (floor 513 with the default capture list).
# When serve_cmd.txt (written by the harness) sits next to the log, it must carry --max-cudagraph-capture-size <cap> too
# (not checked for cap 512, the default capture list, which has no flag).
# The capture flag is accepted as "--max-cudagraph-capture-size 256", "--max-cudagraph-capture-size=256" or a
# max_cudagraph_capture_size entry of a -cc / --compilation-config JSON.
# The log line is printed once per boot (rank 0); the env is inherited by every worker, so only that floor is checked.
# S1-9 lever 2: it also checks the PAIRING of the microbatched prefill (r2e) with the KV bytes. The vLLM branch logs "MuseSpark SM120: prefill
# batches of at least <n> tokens run as 2 microbatches ..." when the overlap is active (VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_SPECULATIVE=1 of the
# profile); next to a harness serve_cmd.txt the --kv-cache-memory-bytes must then not exceed R2E_ON_MAX_BYTES (10,373,877,760), and without the
# overlap not R2E_OFF_MAX_BYTES (10,526,351,360). SPEC_EXPECT_R2E=on|off additionally demands the state (default: either, paired).
# Exit 0 = profile in effect; 1 = a line with other values, no line, the flag is missing or the pairing is wrong; 2 = usage or unreadable log.
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  echo "usage: check_spec_launch.sh <serve.log> [expected_sp_floor [expected_graph_cap]]" >&2; exit 2
fi
log=$1
floor=${2:-384}
cap=${3:-256}
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
flag_re="(--max-cudagraph-capture-size[ =]|max_cudagraph_capture_size[\"' :=]+)$cap([^0-9]|$)"
if [ "$cap" != 512 ] && [ -r "$cmd" ] && ! grep -q -E -e "$flag_re" "$cmd"; then
  echo "FAIL serve_cmd.txt has no --max-cudagraph-capture-size $cap"; rc=1
fi
# r2e state of the boot and its pairing with the KV bytes
R2E_ON_MAX_BYTES=${R2E_ON_MAX_BYTES:-10373877760}
R2E_OFF_MAX_BYTES=${R2E_OFF_MAX_BYTES:-10526351360}
r2e=off
grep -a -q -E 'prefill batches of at least [0-9]+ tokens run as [0-9]+ microbatches' "$log" && r2e=on
if [ -n "${SPEC_EXPECT_R2E:-}" ] && [ "$SPEC_EXPECT_R2E" != "$r2e" ]; then
  echo "FAIL prefill overlap (r2e) is $r2e in the log, expected $SPEC_EXPECT_R2E"; rc=1
fi
if [ -r "$cmd" ]; then
  bytes=$(grep -a -o -E -e '--kv-cache-memory-bytes[ =][0-9]+' "$cmd" | head -1 | grep -o -E '[0-9]+$')
  if [ -n "$bytes" ]; then
    max=$R2E_OFF_MAX_BYTES; [ "$r2e" = on ] && max=$R2E_ON_MAX_BYTES
    if [ "$bytes" -gt "$max" ]; then
      echo "FAIL r2e $r2e with --kv-cache-memory-bytes $bytes > $max (the exact bytes are derived per r2e state)"; rc=1
    fi
  fi
fi
[ $rc = 0 ] && echo "OK floor=$floor graphs<=$cap r2e=$r2e ($pool)"
exit $rc
