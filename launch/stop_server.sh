#!/bin/bash
# Stops a server started by launch/serve_spec.sh / serve_nonspec.sh: TERM to its whole session (the launch scripts start it with setsid), then KILL of the
# session and of its leftovers. Never a pattern kill: the session id comes from <out dir>/serve.pid.
# usage: launch/stop_server.sh <out dir of the launch>        Exit 0 = stopped (or not running), 2 = no serve.pid.
out=${1:-}
[ -r "$out/serve.pid" ] || { echo "usage: stop_server.sh <out dir with serve.pid>" >&2; exit 2; }
pid=$(head -1 "$out/serve.pid" | tr -d '[:space:]')
case "$pid" in ""|*[!0-9]*) echo "stop_server: bad pid '$pid'" >&2; exit 2 ;; esac
sid=$(ps -o sid= -p "$pid" 2> /dev/null | tr -d ' ')
[ -n "$sid" ] || { echo "stop_server: pid $pid is not running"; exit 0; }
# a recycled pid must not be taken for the server: its command line has to name vllm
tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null | grep -a -q vllm || { echo "stop_server: pid $pid is not a vllm process, nothing done" >&2; exit 2; }
kill -TERM -- "-$sid" 2> /dev/null
for _ in $(seq 1 60); do kill -0 "$pid" 2> /dev/null || break; sleep 2; done
kill -KILL -- "-$sid" 2> /dev/null
l=$(ps -eo pid,sid,pgid | awk -v g="$sid" '$2 == g || $3 == g {print $1}')
[ -n "$l" ] && kill -KILL $l 2> /dev/null
echo "stop_server: session $sid stopped"
