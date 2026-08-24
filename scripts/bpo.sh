#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${GRPO_PYTHON:-python}"
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "BPO Python is not executable: $PYTHON_BIN" >&2; exit 1; }
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "BPO Python is not available on PATH: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" scripts/train_bpo.py "$@"
