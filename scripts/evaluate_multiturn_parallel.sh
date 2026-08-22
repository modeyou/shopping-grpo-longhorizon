#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-base-qwen35-2b-parallel}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ASSET_DIR="${MULTITURN_ASSET_DIR:-$ROOT/data/multiturn/evaluation-dev-v2}"
TASKS="${MULTITURN_TASKS:-$ASSET_DIR/tasks.jsonl}"
GAP_OPENINGS="${MULTITURN_GAP_OPENINGS:-$ASSET_DIR/gap_openings.jsonl}"
COMPLETE_OPENINGS="${MULTITURN_COMPLETE_OPENINGS:-$ASSET_DIR/complete_openings.jsonl}"
OUTPUT_ROOT="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/multiturn/$LABEL}"
EVALUATION_LIMIT="${MULTITURN_LIMIT:-}"

SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-2b}"
SHOPPER_BASE_URL="${SHOPPER_BASE_URL:-http://127.0.0.1:8001/v1}"
SHOPPER_API_KEY="${SHOPPER_API_KEY:-local-qwen}"
SHOPPER_MODEL="${SHOPPER_MODEL:-qwen3.8-27b}"
LLM_BASE_URLS="${LLM_BASE_URLS:-}"

for path in "$TASKS" "$GAP_OPENINGS" "$COMPLETE_OPENINGS"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required evaluation asset is missing: $path" >&2
    exit 2
  fi
done

if [[ -z "$LLM_BASE_URLS" ]]; then
  echo "ERROR: LLM_BASE_URLS must contain comma-separated Actor endpoints" >&2
  exit 2
fi
IFS=',' read -r -a actor_urls <<< "$LLM_BASE_URLS"
SHARD_COUNT="${MULTITURN_SHARDS:-${#actor_urls[@]}}"
if [[ ! "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MULTITURN_SHARDS must be a positive integer" >&2
  exit 2
fi
if [[ "${#actor_urls[@]}" -ne "$SHARD_COUNT" ]]; then
  echo "ERROR: Actor endpoint count must equal MULTITURN_SHARDS" >&2
  exit 2
fi
for actor_url in "${actor_urls[@]}"; do
  if [[ -z "$actor_url" ]]; then
    echo "ERROR: LLM_BASE_URLS contains an empty endpoint" >&2
    exit 2
  fi
done

limit_args=()
manager_limit_args=()
if [[ -n "$EVALUATION_LIMIT" ]]; then
  if [[ ! "$EVALUATION_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MULTITURN_LIMIT must be a positive integer" >&2
    exit 2
  fi
  limit_args=(--limit "$EVALUATION_LIMIT")
  manager_limit_args=(--limit "$EVALUATION_LIMIT")
fi

common=(
  --expected-tasks "$TASKS"
  "${limit_args[@]}"
  --base-url "$SHOPSIM_BASE_URL"
  --model "$SERVED_MODEL_NAME"
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
  shift 2
  local condition_args=("$@")
  local output_dir="$OUTPUT_ROOT/$condition"
  local shard_root="$output_dir/shards"
  local combined="$output_dir/trajectories.jsonl"
  mkdir -p "$output_dir" "$shard_root"

  "$PYTHON_BIN" "$ROOT/scripts/manage_evaluation_shards.py" seed \
    --tasks "$TASKS" \
    --input "$combined" \
    --shard-root "$shard_root" \
    --shard-count "$SHARD_COUNT" \
    "${manager_limit_args[@]}"

  local pids=()
  local shard_index
  for ((shard_index=0; shard_index<SHARD_COUNT; shard_index++)); do
    local shard_dir="$shard_root/$shard_index"
    mkdir -p "$shard_dir"
    "$PYTHON_BIN" "$ROOT/scripts/evaluate_shop_benchmark.py" \
      --benchmark "$benchmark" \
      --output "$shard_dir/trajectories.jsonl" \
      --summary "$shard_dir/summary.json" \
      --condition "$condition" \
      --shard-count "$SHARD_COUNT" \
      --shard-index "$shard_index" \
      --llm-base-url "${actor_urls[$shard_index]}" \
      "${common[@]}" \
      "${condition_args[@]}" \
      >"$shard_dir/run.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  local index
  for ((index=0; index<SHARD_COUNT; index++)); do
    if ! wait "${pids[$index]}"; then
      echo "ERROR: $condition shard $index failed; see $shard_root/$index/run.log" >&2
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi

  "$PYTHON_BIN" "$ROOT/scripts/manage_evaluation_shards.py" merge \
    --tasks "$TASKS" \
    --shard-root "$shard_root" \
    --shard-count "$SHARD_COUNT" \
    --output "$combined" \
    "${manager_limit_args[@]}"

  "$PYTHON_BIN" "$ROOT/scripts/evaluate_shop_benchmark.py" \
    --summary-only \
    --execution-shards "$SHARD_COUNT" \
    --benchmark "$benchmark" \
    --output "$combined" \
    --summary "$output_dir/summary.json" \
    --condition "$condition" \
    --llm-base-url "${actor_urls[0]}" \
    "${common[@]}" \
    "${condition_args[@]}"
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

echo "parallel multi-turn evaluation completed: $OUTPUT_ROOT"
