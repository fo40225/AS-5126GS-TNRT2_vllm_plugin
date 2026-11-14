#!/bin/bash
# Session helpers of the launch scripts (sourced by launch/lib.sh and launch/stop_server.sh, never run).
#
# The server is started as `setsid bash launch/lc_child.sh <record> vllm serve ...`; the child records its OWN identity after setsid
# ("<pid> <sid> <pgrp> <starttime>" in <out>/server.sess, written atomically) and then execs the server, so the launcher never guesses a
# session id from `ps` (which can still show the launcher's own session before the child's setsid ran). Rules enforced here:
#   - a session / process group is signalled only when it is the recorded one: its leader (pid == sid == pgrp) is the recorded pid with
#     the recorded start time (a recycled pid has another start time); a gone leader is handled below;
#   - a session or group that contains this script's own pid or its parent's is NEVER signalled (sx_guard), whatever the record says;
#   - the sweep of leftovers skips this script's pid and its parent's.

# sx_proc <pid>: "ppid pgrp sid starttime" from /proc/<pid>/stat (the comm field may hold spaces and parentheses); rc 1 when the process is gone
sx_proc() {
  local s r f
  s=$(cat "/proc/$1/stat" 2> /dev/null) || return 1
  r=${s##*) }
  f=($r)
  [ -n "${f[19]:-}" ] || return 1
  echo "${f[1]} ${f[2]} ${f[3]} ${f[19]}"
}

# sx_guard <id>: rc 0 when the session / group id may be signalled, 1 when it is not a plain id or contains this script's pid or its parent's
sx_guard() {
  local id=$1 p f pg sd
  case "$id" in ""|*[!0-9]*|0|1) return 1 ;; esac
  for p in $$ $PPID; do
    [ "$id" = "$p" ] && return 1
    f=$(sx_proc "$p") || continue
    read -r _ pg sd _ <<< "$f"
    if [ "$id" = "$pg" ] || [ "$id" = "$sd" ]; then return 1; fi
  done
  return 0
}

# sx_read_record <file>: SX_PID SX_SID SX_PGRP SX_START from a complete record; rc 1 when missing or malformed
sx_read_record() {
  local extra
  SX_PID=; SX_SID=; SX_PGRP=; SX_START=
  [ -r "$1" ] || return 1
  read -r SX_PID SX_SID SX_PGRP SX_START extra < "$1" || true
  case "$SX_PID$SX_SID$SX_PGRP$SX_START" in ""|*[!0-9]*) return 1 ;; esac
  [ -n "$SX_PID" ] && [ -n "$SX_SID" ] && [ -n "$SX_PGRP" ] && [ -n "$SX_START" ] && [ -z "$extra" ]
}

# sx_members <sid>: the pids of the session / group, minus this script's pid and its parent's
sx_members() {
  ps -eo pid=,sid=,pgid= | awk -v g="$1" -v a="$$" -v b="$PPID" '($2 == g || $3 == g) && $1 != a && $1 != b {print $1}'
}

# sx_stop <record> <term wait iterations> <poll seconds>: stops the recorded server session. Prints what it did.
#   rc 0 stopped (or nothing left), 2 no valid record, 3 refused to signal (leader mismatch or a session that contains the caller).
#   Leader alive: TERM to the group, wait for the leader, KILL to the group, KILL of the leftovers. Leader gone: only members still carrying the
#   recorded session id are signalled one by one (the kernel keeps a session id reserved while any member lives, so they are the old server's orphans).
sx_stop() {
  local rec=$1 iters=$2 poll=$3 f ppid pg sd st l i
  sx_read_record "$rec" || { echo "no valid session record $rec"; return 2; }
  sx_guard "$SX_SID" || { echo "REFUSED session $SX_SID contains this script (pid $$) or its parent ($PPID), or is not a plain id: nothing signalled"; return 3; }
  if f=$(sx_proc "$SX_PID"); then
    read -r ppid pg sd st <<< "$f"
    if [ "$sd" != "$SX_SID" ] || [ "$st" != "$SX_START" ] || [ "$SX_PID" != "$SX_SID" ]; then
      echo "REFUSED pid $SX_PID is not the recorded session leader (session $sd, start $st; recorded $SX_SID, $SX_START): nothing signalled"; return 3
    fi
    kill -TERM -- "-$SX_SID" 2> /dev/null
    for i in $(seq 1 "$iters"); do
      kill -0 "$SX_PID" 2> /dev/null || break
      sleep "$poll"
    done
    kill -KILL -- "-$SX_SID" 2> /dev/null
    echo "TERM then KILL to session $SX_SID (leader $SX_PID)"
  else
    l=$(sx_members "$SX_SID")
    if [ -n "$l" ]; then
      kill -TERM $l 2> /dev/null
      for i in $(seq 1 "$iters"); do
        [ -n "$(sx_members "$SX_SID")" ] || break
        sleep "$poll"
      done
      echo "leader $SX_PID gone; TERM to the remaining members of session $SX_SID: $(echo $l)"
    fi
  fi
  l=$(sx_members "$SX_SID")
  if [ -n "$l" ]; then kill -KILL $l 2> /dev/null; echo "KILLED leftovers of the session: $(echo $l)"; fi
  return 0
}
