#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-base-qwen35-2b}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ASSET_DIR="${MULTITURN_ASSET_DIR:-$ROOT/data/multiturn/evaluation-dev-v2}"
TASKS="${MULTITURN_TASKS:-$ASSET_DIR/tasks.jsonl}"
GAP_OPENINGS="${MULTITURN_GAP_OPENINGS:-$ASSET_DIR/gap_openings.jsonl}"
COMPLETE_OPENINGS="${MULTITURN_COMPLETE_OPENINGS:-$ASSET_DIR/complete_openings.jsonl}"
OUTPUT_ROOT="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/multiturn/$LABEL}"
EVALUATION_LIMIT="${MULTITURN_LIMIT:-}"

SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-2b}"
SHOPPER_BASE_URL="${SHOPPER_BASE_URL:-http://127.0.0.1:8001/v1}"
SHOPPER_API_KEY="${SHOPPER_API_KEY:-local-qwen}"
SHOPPER_MODEL="${SHOPPER_MODEL:-qwen3.8-27b}"

for path in "$TASKS" "$GAP_OPENINGS" "$COMPLETE_OPENINGS"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required evaluation asset is missing: $path" >&2
    exit 2
  fi
done

limit_args=()
if [[ -n "$EVALUATION_LIMIT" ]]; then
  if [[ ! "$EVALUATION_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MULTITURN_LIMIT must be a positive integer" >&2
    exit 2
  fi
  limit_args=(--limit "$EVALUATION_LIMIT")
fi

common=(
  --expected-tasks "$TASKS"
  "${limit_args[@]}"
  --base-url "$SHOPSIM_BASE_URL"
  --model "$SERVED_MODEL_NAME"
  --llm-base-url "$LLM_BASE_URL"
  --api-key "$LLM_API_KEY"
  --max-steps 35
  --max-shopper-questions 2
  --temperature 0
  --top-p 1
  --max-tokens 512
  --context-window 24576
  --context-safety-margin 512
  --observation-token-budget 1536
  --observation-detail-token-budget 4096
  --observation-generic-token-budget 768
  --observation-search-top-k 20
  --disable-model-thinking
)

run_condition() {
  local condition="$1"
  local benchmark="$2"
  local output_dir="$OUTPUT_ROOT/$condition"
  mkdir -p "$output_dir"
  "$PYTHON_BIN" "$ROOT/scripts/evaluate_shop_benchmark.py" \
    --benchmark "$benchmark" \
    --output "$output_dir/trajectories.jsonl" \
    --summary "$output_dir/summary.json" \
    --condition "$condition" \
    "${common[@]}" \
    "${@:3}"
}

run_condition gap-ask-enabled "$GAP_OPENINGS" \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  --shopper-api-key "$SHOPPER_API_KEY" \
  --disable-shopper-thinking

run_condition gap-ask-disabled "$GAP_OPENINGS"

run_condition complete-ask-enabled "$COMPLETE_OPENINGS" \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  --shopper-api-key "$SHOPPER_API_KEY" \
  --disable-shopper-thinking

echo "multi-turn evaluation completed: $OUTPUT_ROOT"
