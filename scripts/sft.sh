#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"

cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft/formal-v2/train.jsonl \
  --validation data/sft/formal-v2/validation.jsonl \
  --data-manifest data/sft/formal-v2/manifest.json \
  --evaluation-tasks data/evaluation/tasks.jsonl \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa

exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
