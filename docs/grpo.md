# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v4 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.8.0` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
  dynamic_sampling.py   bounded non-zero-reward sampling
```

`scripts/setup.sh` applies one SHA-256-checked patch needed to connect the
bounded dynamic sampler to veRL 0.8.0. Setup fails rather than patching an
unknown veRL version.

## Inputs

- Initial policy: `outputs/models/sft-merged`
- Train/validation: task-disjoint multi-turn gap and complete openings converted to veRL parquet
- Environment: ShopSimulator Environment v2.1
- Reward: Reward v4
- Shopper: a separate OpenAI-compatible endpoint, configured through `SHOPPER_*`

Do not reuse development or formal evaluation openings as GRPO training data. Build parquet only
from the frozen GRPO task split and its matching openings:

```bash
export PYTHONPATH=./src
python scripts/prepare_multiturn_grpo_dataset.py \
  --tasks data/multiturn/tasks/grpo_train.jsonl \
  --gap-openings data/multiturn/grpo-train-openings/gap_openings.jsonl \
  --complete-openings data/multiturn/grpo-train-openings/complete_openings.jsonl \
  --exclude-tasks data/multiturn/evaluation-dev-v2/tasks.jsonl \
  --exclude-tasks data/multiturn/evaluation-v2/tasks.jsonl \
  --mode mixed \
  --split train \
  --output data/grpo/multiturn-train.parquet
```

## Run

Inspect the resolved command first:

```bash
export SHOPPER_MODEL=deepseek-v4-flash-0731
export SHOPPER_BASE_URL="$OPENAI_BASE_URL"
export SHOPPER_API_KEY="$OPENAI_API_KEY"
bash scripts/grpo.sh --dry-run
```

Run the complete environment, dependency, manifest, patch and memory-budget
preflight without loading the model or entering veRL training:

```bash
bash scripts/grpo.sh --preflight-only
```

Train:

```bash
bash scripts/grpo.sh
```

The multi-turn harness routes `ask_shopper` to the separate Shopper endpoint. It does not send
that call to ShopSimulator and does not charge it against the 35 shopping-action steps. Every
rollout has an isolated Shopper history; clarified answers are projected into later observations.

Important defaults:

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 2 / 2 |
| Policy learning rate | `1e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 24,576 |
| Maximum training steps | 500 |
| Save / validation frequency | 50 / 50 |
| KL reward / KL loss | disabled / disabled |
| Policy entropy measurement | enabled (logging only) |

Dynamic sampling can generate at most three batches to find a useful update and
permits at most ten consecutive skipped updates. These bounds prevent an
all-equal reward batch from causing an unbounded resampling loop.

Each run also appends `training_diagnostics.jsonl` under its output directory.
`generation_batch` records contain every generated rollout, its public tool
sequence, terminal result, reward breakdown, Guard rejection reasons and group
keep/drop decision. `optimizer_step` records preserve the scalar veRL metrics,
including entropy, PPO KL, clip fractions, response lengths and effective-group
rates. `skipped_update` records make zero-signal attempts visible even though
they do not advance the optimizer step.

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
Advanced overrides may be appended after `--`:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

The reported comparison uses step 100. Select checkpoints using validation
metrics rather than assuming that the final training step is best.
