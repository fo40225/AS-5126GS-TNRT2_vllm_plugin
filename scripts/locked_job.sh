#!/bin/bash
# Server side: run a command under the shared GPU lock, detached, with a log and a DONE marker.
# usage: locked_job.sh <tag> <timeout_s> <command...>   (log: work/host/logs/<tag>.log, marker: <tag>.DONE with rc)
W=/raid/vllm-runtime/work/host
tag=$1; to=$2; shift 2
mkdir -p $W/logs; rm -f $W/logs/$tag.DONE
setsid nohup flock -w ${LOCK_WAIT_S:-43200} /raid/vllm-runtime/sm120-dev/gpu.lock bash -c '
W=/raid/vllm-runtime/work/host
tag=$1; to=$2; shift 2
echo "LOCK ACQUIRED $(date +%T)"
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s}")
  [ "${u:-0}" -lt 2000 ] && break; sleep 5
done
echo "START $tag $(date +%T) gpu_used_mib=${u:-?}"
setsid timeout -s TERM -k 20 "$to" "$@" &
pid=$!
wait $pid; rc=$?
echo "EXIT rc=$rc $(date +%T)"
ps -eo pid=,sid= | awk -v s=$pid "\$2==s{print \$1}" | xargs -r kill -TERM 2>/dev/null
sleep 3
ps -eo pid=,sid= | awk -v s=$pid "\$2==s{print \$1}" | xargs -r kill -KILL 2>/dev/null
echo "$rc" > $W/logs/$tag.DONE
' _ "$tag" "$to" "$@" > $W/logs/$tag.log 2>&1 < /dev/null &
echo "launched $tag"
