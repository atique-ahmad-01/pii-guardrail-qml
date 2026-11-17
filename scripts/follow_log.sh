#!/usr/bin/env bash
# Print a log as it grows and exit once the "[tmux_run] EXIT=" line appears.
# Usage: scripts/follow_log.sh LOGFILE
log=$1
tail -n +1 -F "$log" 2>/dev/null &
tailpid=$!
until grep -q '^\[tmux_run\] EXIT=' "$log" 2>/dev/null; do sleep 2; done
sleep 1; kill "$tailpid" 2>/dev/null
