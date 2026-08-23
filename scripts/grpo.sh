#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${GRPO_PYTHON:-python}"
if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "GRPO Python is not executable: $PYTHON_BIN" >&2
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "GRPO Python is not available on PATH: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" scripts/train_grpo.py "$@"
