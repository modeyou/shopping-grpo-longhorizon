#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_CHECKPOINT="${1:?usage: bash scripts/export_grpo.sh <global_step_*/actor> [output_dir]}"
OUTPUT_DIR="${2:-$ROOT/outputs/models/grpo-merged}"
PYTHON_BIN="${GRPO_PYTHON:-${PYTHON_BIN:-$ROOT/.venv/bin/python}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$ACTOR_CHECKPOINT" ]]; then
  echo "ERROR: actor checkpoint is missing: $ACTOR_CHECKPOINT" >&2
  exit 2
fi

exec "$PYTHON_BIN" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT_DIR" \
  --trust-remote-code
