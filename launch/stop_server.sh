#!/bin/bash
# Stops a server started by launch/serve_spec.sh / serve_nonspec.sh: TERM to its recorded session, then KILL of the session and of its leftovers. Never a pattern kill.
# The session comes from <out dir>/server.sess, which the server's own child process wrote after setsid (pid, session, process group, start time; launch/session.sh):
# the session is signalled only when its leader is that pid with that start time, and never when it contains this script or its parent.
# usage: launch/stop_server.sh <out dir of the launch>
# Exit 0 = stopped (or not running), 2 = no serve.pid / server.sess or not a vllm process, 3 = refused to signal (leader mismatch, or a session that contains this script).
out=${1:-}
[ -r "$out/serve.pid" ] || { echo "usage: stop_server.sh <out dir with serve.pid and server.sess>" >&2; exit 2; }
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/session.sh"
pid=$(head -1 "$out/serve.pid" | tr -d '[:space:]')
case "$pid" in ""|*[!0-9]*) echo "stop_server: bad pid '$pid'" >&2; exit 2 ;; esac
sx_read_record "$out/server.sess" || { echo "stop_server: no valid $out/server.sess (the session of the server was never recorded): nothing done" >&2; exit 2; }
[ "$SX_PID" = "$pid" ] || { echo "stop_server: server.sess names pid $SX_PID, serve.pid names $pid: nothing done" >&2; exit 2; }
if sx_proc "$pid" > /dev/null; then
  # a recycled pid must not be taken for the server: its command line has to name vllm
  tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null | grep -a -q vllm || { echo "stop_server: pid $pid is not a vllm process, nothing done" >&2; exit 2; }
fi
msg=$(sx_stop "$out/server.sess" 60 2); rc=$?
echo "$msg"
[ $rc = 0 ] && echo "stop_server: session $SX_SID stopped"
exit $rc
