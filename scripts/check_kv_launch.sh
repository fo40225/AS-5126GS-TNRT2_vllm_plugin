#!/bin/bash
# Post-boot check of the KV sizing and the scheduler shape of a production launch (S1-11): the exact --kv-cache-memory-bytes, the KV tokens and blocks the bytes
# produce, --max-num-seqs, the capture-size cap (ns256) and the speculative state, for the three profiles and both tier states. It complements
# scripts/check_spec_launch.sh (speculative flags, pairing) and scripts/check_offload_launch.sh (tier state: bind, connector, blocks), which do not look at the KV size.
# usage: check_kv_launch.sh <spec|ns128|ns256> <1|0> <serve.log>      (1 = NUMA-local tier ON, 0 = no tier)
# Expectations come from the env files, never from the calling shell: the exact bytes (env/offload.env for the tier-ON profiles, env/spec_k7.env with MUSE_OFFLOAD=0 for the
# speculative no-tier launch, env/launch.env for the non-speculative no-tier bytes), --max-num-seqs, the cap, the largest FULL decode graph and the expected KV
# tokens / blocks (env/launch.env: LAUNCH_EXPECT_<profile>_<tier>, measured on the gated boots).
# What it reads next to the log: serve_cmd.txt (the launch line, written by the launch script or the harness; required) and cache_config_info.txt (the /metrics line
# vllm:cache_config_info that the launch script saves after READY; the block count exists only there; required). Fail-closed: a missing file or line is a FAIL.
# Checks:
#   launch line : --kv-cache-memory-bytes, --max-num-seqs; the capture flag --max-cudagraph-capture-size <cap> (ns256) or none (ns128); --speculative-config only for spec;
#   startup log : every "non-default args" line carries the same kv_cache_memory_bytes and max_num_seqs, max_cudagraph_capture_size 256 for ns256 and none for ns128;
#                 "speculative_config=None" for the non-speculative profiles (a speculative one otherwise);
#   KV size     : one "GPU KV cache size: N tokens" value = the expected tokens; cache_config_info: num_gpu_blocks, kv_cache_size_tokens and kv_cache_memory_bytes equal the
#                 expected blocks, tokens and bytes;
#   graphs      : (non-speculative) every worker logged the same "CUDA graphs: N FULL decode graphs ... (token counts 1..<hi>)" line and hi is the profile's (256 / 128). The
#                 sequence-parallel line says "CUDA graphs up to 256 tokens" for ns128 too, so the capture cap is NOT read from it.
# Exit 0 = every check passed; 1 = a mismatch or a missing line; 2 = usage or unreadable log.
if [ $# -ne 3 ]; then echo "usage: check_kv_launch.sh <spec|ns128|ns256> <1|0> <serve.log>" >&2; exit 2; fi
prof=$1; tier=$2; log=$3
case "$prof" in spec|ns128|ns256) ;; *) echo "check_kv_launch: profile must be spec, ns128 or ns256, got '$prof'" >&2; exit 2 ;; esac
case "$tier" in 0|1) ;; *) echo "check_kv_launch: tier must be 0 or 1, got '$tier'" >&2; exit 2 ;; esac
[ -r "$log" ] || { echo "check_kv_launch: cannot read $log" >&2; exit 2; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dir=$(dirname "$log")
cmd=$dir/serve_cmd.txt
cci=$dir/cache_config_info.txt
rc=0
fail() { echo "FAIL $*"; rc=1; }
# the expectations, from the env files in a clean shell
ev() { env -i PATH="$PATH" bash -c "$1" 2> /dev/null; }
src_launch="source '$here/env/launch.env'"
P=${prof^^}
case "$prof" in
  spec)
    mns=$(ev "$src_launch; echo \$LAUNCH_SPEC_MNS"); cap=; hi=
    bytes=$(ev "MUSE_OFFLOAD=$tier; source '$here/env/spec_k7.env'; echo \$MUSE_SPEC_KV_BYTES") ;;
  *)
    mns=$(ev "$src_launch; echo \$LAUNCH_${P}_MNS"); cap=$(ev "$src_launch; echo \$LAUNCH_${P}_CAP"); hi=$(ev "$src_launch; echo \$LAUNCH_${P}_GRAPH_HI")
    if [ "$tier" = 1 ]; then bytes=$(ev "MUSE_OFFLOAD=1; MUSE_OFFLOAD_PROFILE=$prof; source '$here/env/offload.env'; echo \$MUSE_OFFLOAD_KV_BYTES")
    else bytes=$(ev "$src_launch; echo \$LAUNCH_${P}_KV_BYTES"); fi ;;
esac
# the tier-ON bytes of spec come from the override of the sourced profile too (spec_k7.env with the tier ON gives the same number as offload.env)
[ "$prof" = spec ] && [ "$tier" = 1 ] && bytes=$(ev "MUSE_OFFLOAD=1; MUSE_OFFLOAD_PROFILE=spec; source '$here/env/offload.env'; echo \$MUSE_OFFLOAD_KV_BYTES")
read -r tokens blocks <<< "$(ev "$src_launch; echo \$LAUNCH_EXPECT_${prof}_${tier}")"
for v in mns bytes tokens blocks; do
  [ -n "${!v}" ] || { echo "FAIL expectation '$v' of $prof tier=$tier is unknown (env files)"; exit 1; }
done
if [ "$prof" != spec ] && [ -z "$hi" ]; then echo "FAIL the largest FULL decode graph of $prof is unknown (env/launch.env)"; exit 1; fi

# ---- launch line
if [ ! -r "$cmd" ]; then
  fail "no serve_cmd.txt next to the log: the launch line cannot be checked"
else
  v=$(grep -a -o -E -e '--kv-cache-memory-bytes[ =][0-9]+' "$cmd" | head -1 | grep -o -E '[0-9]+$')
  [ "$v" = "$bytes" ] || fail "serve_cmd.txt --kv-cache-memory-bytes '${v:-none}', expected $bytes ($prof tier=$tier)"
  v=$(grep -a -o -E -e '--max-num-seqs[ =][0-9]+' "$cmd" | head -1 | grep -o -E '[0-9]+$')
  [ "$v" = "$mns" ] || fail "serve_cmd.txt --max-num-seqs '${v:-none}', expected $mns"
  if [ "$prof" != spec ]; then
    v=$(grep -a -o -E -e '--max-cudagraph-capture-size[ =][0-9]+' "$cmd" | head -1 | grep -o -E '[0-9]+$')
    [ "$v" = "$cap" ] || fail "serve_cmd.txt --max-cudagraph-capture-size '${v:-none}', expected '${cap:-none}'"
    grep -a -q -e '--speculative-config' "$cmd" && fail "serve_cmd.txt carries --speculative-config in a non-speculative profile"
  else
    grep -a -q -e '--speculative-config' "$cmd" || fail "serve_cmd.txt has no --speculative-config in the speculative profile"
  fi
fi

# ---- startup log
nda=$(grep -a -c 'non-default args' "$log")
if [ "$nda" = 0 ]; then
  fail "no \"non-default args\" line in the log: the parsed launch arguments cannot be confirmed"
else
  seen=$(grep -a 'non-default args' "$log" | grep -a -o -E "'kv_cache_memory_bytes': [0-9]+" | sed -E 's/.*: //' | sort | uniq -c)
  echo "$seen" | grep -q -E "^ *$nda $bytes\$" || fail "the non-default args lines (count $nda) do not all carry kv_cache_memory_bytes=$bytes: $(echo $seen)"
  seen=$(grep -a 'non-default args' "$log" | grep -a -o -E "'max_num_seqs': [0-9]+" | sed -E 's/.*: //' | sort | uniq -c)
  echo "$seen" | grep -q -E "^ *$nda $mns\$" || fail "the non-default args lines (count $nda) do not all carry max_num_seqs=$mns: $(echo $seen)"
  if [ "$prof" != spec ]; then
    seen=$(grep -a 'non-default args' "$log" | grep -a -o -E "'max_cudagraph_capture_size': [0-9]+" | sed -E 's/.*: //' | sort | uniq -c)
    if [ -n "$cap" ]; then echo "$seen" | grep -q -E "^ *$nda $cap\$" || fail "the non-default args lines (count $nda) do not all carry max_cudagraph_capture_size=$cap: $(echo $seen)"
    else [ -z "$seen" ] || fail "max_cudagraph_capture_size is set in the log ($(echo $seen)) but $prof has no capture flag"; fi
  fi
fi
if [ "$prof" != spec ]; then
  grep -a -q 'speculative_config=None' "$log" || fail "no 'speculative_config=None' in the engine config line: the boot is not a plain non-speculative one"
else
  grep -a -q 'speculative_config=None' "$log" && fail "'speculative_config=None' in the log of the speculative profile"
fi

# ---- KV size
seen=$(grep -a -o -E 'GPU KV cache size: [0-9,]+ tokens' "$log" | sed -E 's/.*: ([0-9,]+) tokens/\1/; s/,//g' | sort | uniq -c)
if [ -z "$seen" ]; then fail "no 'GPU KV cache size: <N> tokens' line"
elif [ "$(echo "$seen" | wc -l)" != 1 ] || ! echo "$seen" | grep -q -E " $tokens\$"; then fail "GPU KV cache size: expected $tokens tokens; found (count value): $(echo $seen)"; fi
if [ ! -r "$cci" ]; then
  fail "no cache_config_info.txt next to the log: the block count (and the bytes the engine recorded) cannot be confirmed"
else
  v=$(grep -a -o -E 'num_gpu_blocks="[0-9]+"' "$cci" | head -1 | grep -o -E '[0-9]+')
  [ "$v" = "$blocks" ] || fail "num_gpu_blocks '${v:-none}', expected $blocks"
  v=$(grep -a -o -E 'kv_cache_size_tokens="[0-9]+"' "$cci" | head -1 | grep -o -E '[0-9]+')
  [ "$v" = "$tokens" ] || fail "kv_cache_size_tokens '${v:-none}' in cache_config_info, expected $tokens"
  v=$(grep -a -o -E 'kv_cache_memory_bytes="[0-9]+"' "$cci" | head -1 | grep -o -E '[0-9]+')
  [ "$v" = "$bytes" ] || fail "kv_cache_memory_bytes '${v:-none}' in cache_config_info, expected $bytes"
fi

# ---- FULL decode graphs of the capture cap (non-speculative)
gr=-
if [ "$prof" != spec ]; then
  seen=$(grep -a -o -E 'CUDA graphs: [0-9]+ FULL decode graphs are flagged full_decode_graph \(token counts (none|[0-9]+\.\.[0-9]+)\)' "$log" | sed -E 's/.*token counts ([^)]*)\)/\1/' | sort | uniq -c)
  if [ -z "$seen" ]; then fail "no flagged FULL decode graph line (\"CUDA graphs: <N> FULL decode graphs are flagged ...\")"
  elif [ "$(echo "$seen" | wc -l)" != 1 ]; then fail "the workers disagree on the FULL decode graph range (count range): $(echo $seen)"
  else
    n=$(echo "$seen" | awk '{print $1}'); gr=$(echo "$seen" | awk '{print $2}')
    w=$(grep -a -c 'Model loading took' "$log")
    [ "$w" -gt 0 ] && [ "$n" != "$w" ] && fail "$n of $w workers logged the FULL decode graph line"
    [ "${gr##*..}" = "$hi" ] || fail "the largest FULL decode graph is ${gr##*..} tokens, expected $hi ($prof)"
  fi
fi
[ $rc = 0 ] && echo "OK profile=$prof tier=$tier bytes=$bytes tokens=$tokens blocks=$blocks mns=$mns cap=${cap:-none} graphs=$gr"
exit $rc
