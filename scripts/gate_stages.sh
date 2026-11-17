#!/usr/bin/env bash
# Run every gate stage in order and stop at the first failure.
# Usage: scripts/gate_stages.sh CONFIG   (run it through scripts/tmux_run.sh for long jobs)
set -euo pipefail
cfg=$1
UV=${UV:-$(command -v uv || echo .tools/uv)}
export TOKENIZERS_PARALLELISM=false
for stage in features classical vqc kernel finetune barren summary; do
  echo "=== stage: $stage ==="
  "$UV" run python scripts/run_gate.py --config "$cfg" --stage "$stage"
done
echo "=== gate complete ==="
