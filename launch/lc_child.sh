#!/bin/bash
# Runs as the server's session leader (the launch scripts start it as `setsid bash lc_child.sh <record> <vllm> <args...>`): records this process's own
# pid / session / process group / start time in <record> (written to a temporary file and renamed, so a reader never sees half a line), then replaces
# itself with the server (same pid, so the record stays valid). Exports nothing. See launch/session.sh.
rec=$1; shift
s=$(< "/proc/$$/stat")
r=${s##*) }
f=($r)
# f[0] state, f[1] ppid, f[2] pgrp, f[3] session, f[19] starttime
printf '%s %s %s %s\n' "$$" "${f[3]}" "${f[2]}" "${f[19]}" > "$rec.tmp$$" && mv -f "$rec.tmp$$" "$rec" || exit 97
exec "$@"
