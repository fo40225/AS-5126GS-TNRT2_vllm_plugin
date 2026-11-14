#!/bin/bash
# Session helpers of the launch scripts (sourced by launch/lib.sh and launch/stop_server.sh, never run).
#
# The server is started as `setsid bash launch/lc_child.sh <record> vllm serve ...`; the child records its OWN identity after setsid
# ("<pid> <sid> <pgrp> <starttime>" in <out>/server.sess, written atomically) and then execs the server, so the launcher never guesses a
# session id from `ps` (which can still show the launcher's own session before the child's setsid ran). Rules enforced here:
#   - a session / process group is signalled only when it is the recorded one: its leader (pid == sid == pgrp) is the recorded pid with
#     the recorded start time (a recycled pid has another start time); a gone leader is handled below;
#   - with the leader gone, a member is signalled only when it is provably the server's own: same boot (boot_id of the record), its working
#     directory is the launch's out dir (the server and its workers start with cd $LC_OUT) and it started no earlier than the recorded leader;
#     any other member of that session number is reported and left alone;
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

# sx_alive <pid>: rc 0 when the process exists and is not a zombie (a zombie that nobody reaps can no longer be signalled to any effect)
sx_alive() {
  local s r
  s=$(cat "/proc/$1/stat" 2> /dev/null) || return 1
  r=${s##*) }
  [ "${r%% *}" != Z ]
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

# sx_record_ok: validates the SX_* variables just set by sx_read_record / sx_parse_record (all four numbers present, a plain boot id, no extra field)
sx_record_ok() {
  case "$SX_PID$SX_SID$SX_PGRP$SX_START" in ""|*[!0-9]*) return 1 ;; esac
  case "$SX_BOOT" in *[!0-9a-f-]*) return 1 ;; esac
  [ -n "$SX_PID" ] && [ -n "$SX_SID" ] && [ -n "$SX_PGRP" ] && [ -n "$SX_START" ] && [ -z "$SX_EXTRA" ]
}

# sx_read_record <file>: SX_PID SX_SID SX_PGRP SX_START and the optional SX_BOOT (boot_id) from a complete record; rc 1 when missing or malformed
sx_read_record() {
  SX_PID=; SX_SID=; SX_PGRP=; SX_START=; SX_BOOT=; SX_EXTRA=
  [ -r "$1" ] || return 1
  read -r SX_PID SX_SID SX_PGRP SX_START SX_BOOT SX_EXTRA < "$1" || true
  sx_record_ok
}

# sx_parse_record <text>: the same as sx_read_record for a record held in memory (the launcher keeps the tuple it accepted)
sx_parse_record() {
  SX_PID=; SX_SID=; SX_PGRP=; SX_START=; SX_BOOT=; SX_EXTRA=
  [ -n "$1" ] || return 1
  read -r SX_PID SX_SID SX_PGRP SX_START SX_BOOT SX_EXTRA <<< "$1" || true
  sx_record_ok
}

# sx_members <sid>: the live (non-zombie) pids of the session / group, minus this script's pid and its parent's
sx_members() {
  ps -eo pid=,sid=,pgid=,stat= | awk -v g="$1" -v a="$$" -v b="$PPID" '($2 == g || $3 == g) && $1 != a && $1 != b && $4 !~ /^Z/ {print $1}'
}

# sx_own_members <sid> <out dir> <leader start time> <boot id>: the members that are provably the server's own (leader gone): the boot is the recorded one,
# the working directory is the out dir and the start time is not earlier than the leader's. Prints the pids; the rest is sx_members minus these.
sx_own_members() {
  local p st
  [ -n "$4" ] && [ "$4" = "$(cat /proc/sys/kernel/random/boot_id 2> /dev/null)" ] || return 0
  for p in $(sx_members "$1"); do
    [ "$(readlink -f "/proc/$p/cwd" 2> /dev/null)" = "$2" ] || continue
    st=$(sx_proc "$p") || continue
    [ "${st##* }" -ge "$3" ] && echo "$p"
  done
}

# sx_stop <record> <term wait iterations> <poll seconds> [<accepted record text>]: stops the recorded server session. Prints what it did.
#   rc 0 stopped (or nothing left), 2 no valid record, 3 refused to signal (leader mismatch, a session that contains the caller, or members of a
#   leaderless session that cannot be shown to belong to the server: they are listed and left running).
#   Leader alive: TERM to the group, wait for the leader, KILL to the group, KILL of the leftovers. Leader gone: only the provably own members
#   (sx_own_members) are signalled, one by one.
#   The launcher passes the tuple it accepted (the record text); it is used when <record> is gone or unreadable, and goes through the same guards.
sx_stop() {
  local rec=$1 iters=$2 poll=$3 mem=${4:-} src=file
  if ! sx_read_record "$rec"; then
    if [ -n "$mem" ] && sx_parse_record "$mem"; then src=memory
    else echo "no valid session record $rec"; return 2; fi
  fi
  sx_stop_tuple "$(dirname "$(readlink -m "$rec")")" "$iters" "$poll" "$src"
}

# sx_stop_tuple <out dir> <term wait iterations> <poll seconds> <source label>: the core of sx_stop, on the SX_* variables set by the caller
sx_stop_tuple() {
  local out=$1 iters=$2 poll=$3 src=$4 f ppid pg sd st l i own foreign rc=0
  [ "$src" = memory ] && echo "session record gone or unreadable: using the tuple the launcher accepted ($SX_PID $SX_SID $SX_PGRP $SX_START)"
  sx_guard "$SX_SID" || { echo "REFUSED session $SX_SID contains this script (pid $$) or its parent ($PPID), or is not a plain id: nothing signalled"; return 3; }
  if f=$(sx_proc "$SX_PID"); then
    read -r ppid pg sd st <<< "$f"
    if [ "$sd" != "$SX_SID" ] || [ "$st" != "$SX_START" ] || [ "$SX_PID" != "$SX_SID" ]; then
      echo "REFUSED pid $SX_PID is not the recorded session leader (session $sd, start $st; recorded $SX_SID, $SX_START): nothing signalled"; return 3
    fi
    kill -TERM -- "-$SX_SID" 2> /dev/null
    for i in $(seq 1 "$iters"); do
      sx_alive "$SX_PID" || break
      sleep "$poll"
    done
    kill -KILL -- "-$SX_SID" 2> /dev/null
    echo "TERM then KILL to session $SX_SID (leader $SX_PID)"
    l=$(sx_members "$SX_SID")
    if [ -n "$l" ]; then kill -KILL $l 2> /dev/null; echo "KILLED leftovers of the session: $(echo $l)"; fi
  else
    own=$(sx_own_members "$SX_SID" "$out" "$SX_START" "$SX_BOOT")
    if [ -n "$own" ]; then
      kill -TERM $own 2> /dev/null
      for i in $(seq 1 "$iters"); do
        l=$(for p in $own; do sx_alive "$p" && echo "$p"; done)
        [ -n "$l" ] || break
        sleep "$poll"
      done
      echo "leader $SX_PID gone; TERM to the remaining members of session $SX_SID that belong to this launch: $(echo $own)"
      l=$(for p in $own; do sx_alive "$p" && echo "$p"; done)
      if [ -n "$l" ]; then kill -KILL $l 2> /dev/null; echo "KILLED leftovers of the session: $(echo $l)"; fi
    fi
    foreign=$(sx_members "$SX_SID" | grep -vxF -f <(echo "$own") 2> /dev/null)
    if [ -n "$foreign" ]; then
      echo "NOT SIGNALLED: leader $SX_PID is gone and the members of session $SX_SID cannot be shown to belong to this launch (boot, cwd $out, start time): $(echo $foreign); stop them by hand if they are the server's"
      rc=3
    fi
  fi
  return $rc
}
