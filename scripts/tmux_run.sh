#!/usr/bin/env bash
# Run a command inside a detached tmux session and tee its output to a log file.
# Usage: scripts/tmux_run.sh SESSION LOGFILE COMMAND...
# Chain steps with && so a failure stops the run; EXIT is the status of the whole group.
# The last log line is "[tmux_run] EXIT=<code>" so scripts/follow_log.sh knows when it is done.
set -euo pipefail
session=$1; log=$2; shift 2
mkdir -p "$(dirname "$log")"; : > "$log"
tmux kill-session -t "$session" 2>/dev/null || true
tmux new-session -d -s "$session" bash -c \
  "set -o pipefail; { $*; } 2>&1 | tee -a '$log'; echo \"[tmux_run] EXIT=\${PIPESTATUS[0]}\" | tee -a '$log'"
echo "started tmux session '$session', logging to $log"
