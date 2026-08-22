# Multi-turn Reward v4 LoRA SFT

## Purpose

The current SFT stage teaches Qwen3.5-2B to follow ShopSimulator's multi-turn
action protocol: ask only when information is missing, issue legal shopping tool
calls, ground actions in visible observations, select product variants, and
terminate with a valid purchase or stop decision.

`experiments/sft/summary.json` belongs to the original reference project. It is
not a training input, a split definition, or the hyperparameter source for this
multi-turn run.

## Frozen data contract

The formal dataset is generated only from the three Reward v4 re-audited pools:

- complete requests with no unnecessary question;
- replay-verified composite clarification trajectories;
- autonomous gap trajectories.

The mix contains 1,800 unique task/source-goal pairs. The split is task- and
goal-disjoint, with approximately 1,620 training rows and 180 validation rows.
Target assistant-token shares are 50% complete, 30% composite, and 20%
autonomous. Half of the selected complete examples expose the full multi-turn
tool schema, including `ask_shopper`, without inserting a synthetic question.

```bash
: "${GRPO_PYTHON:?set GRPO_PYTHON to the project Python}"
: "${MODEL_DIR:?set MODEL_DIR to the pinned local model}"

"$GRPO_PYTHON" scripts/prepare_multiturn_sft_mix.py \
  --audit-manifest outputs/multiturn-sft/v4-audit-pools-02/manifest.json \
  --evaluation-tasks data/evaluation/tasks.jsonl \
  --model "$MODEL_DIR" \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --output-dir outputs/multiturn-sft/mix-formal-1800-v4-seed20260822 \
  --total-rows 1800 \
  --validation-ratio 0.1 \
  --max-length 24576 \
  --seed 20260822 \
  --complete-token-ratio 0.5 \
  --composite-token-ratio 0.3 \
  --autonomous-token-ratio 0.2 \
  --token-share-tolerance 0.05
```

The generated manifest freezes source hashes, output hashes, selected task IDs,
policy/schema counts, token counts, split membership, model revision, and seed.
Training data must have zero overlap with the final evaluation task set.

## SwanLab setup

The 64-example smoke uses `--swanlab-mode local` and needs no cloud login.
Formal training uses `online` mode. Authenticate once on the training server with
the SwanLab CLI installed in the project environment:

```bash
: "${SWANLAB_BIN:?set SWANLAB_BIN to the project SwanLab CLI}"
"$SWANLAB_BIN" --version
"$SWANLAB_BIN" login
"$SWANLAB_BIN" verify
"$SWANLAB_BIN" ping
```

Use the interactive login prompt instead of putting the API key in shell history
or a launch script. Do not use `login --local` inside the repository because its
`.swanlab/` credentials directory can violate the clean-Git reproducibility gate.
The formal run is uploaded once from world rank zero to project
`shopping-multiturn-agentic`. The frozen two-epoch formal run name is
`qwen35-2b-sft-lora-v4-n1800-e2-seed20260822`.

## Formal recipe

| Setting | Value |
|---|---|
| Base model | Qwen3.5-2B pinned revision |
| Maximum sequence length | 24,576 |
| Epochs | 2 |
| Per-device batch size | 1 |
| GPUs | 4 (CUDA 0-3) |
| Gradient accumulation | 2 |
| Effective global batch | 8 |
| Peak learning rate | `1e-4` |
| Scheduler | 3% warmup, then linear decay |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Precision | bf16 |
| Gradient checkpointing | enabled |
| Attention / fused loss | SDPA / Liger |
| Train logging | every 5 steps |
| Validation | every 50 steps plus final validation |
| Checkpoint | every 25 steps |
| Checkpoints retained | 10 |
| Seed / data seed | 20260822 / 20260822 |

The learning rate is not constant. It warms up from zero to `1e-4`, then
linearly decays across two epochs. With 1,620 training rows and effective global
batch size 8, the run has approximately 203 optimizer steps per epoch and 406
steps in total. `checkpoint-200` is retained as the near-one-epoch candidate;
development rollouts compare it with the final two-epoch adapter rather than
selecting a model from validation loss alone.

```bash
: "${TORCHRUN:?set TORCHRUN to the project torchrun}"
: "${MODEL_DIR:?set MODEL_DIR to the pinned local model}"

export PYTHONHASHSEED=20260822
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWANLAB_MODE=online

"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=4 scripts/train_lora_sft.py \
  --model "$MODEL_DIR" \
  --train outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/train.jsonl \
  --validation outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/validation.jsonl \
  --data-manifest outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/manifest.json \
  --evaluation-tasks data/evaluation/tasks.jsonl \
  --output outputs/models/multiturn-sft-v4-1800-e2-seed20260822 \
  --max-length 24576 \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type linear \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules \
    q_proj k_proj v_proj o_proj \
    gate_proj up_proj down_proj \
    in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj \
  --dtype bf16 \
  --gradient-checkpointing \
  --liger-kernel \
  --attention-implementation sdpa \
  --logging-steps 5 \
  --eval-steps 50 \
  --save-steps 25 \
  --save-total-limit 10 \
  --seed 20260822 \
  --data-seed 20260822 \
  --full-determinism \
  --require-clean-git \
  --swanlab \
  --swanlab-mode online \
  --swanlab-project shopping-multiturn-agentic \
  --swanlab-run-name qwen35-2b-sft-lora-v4-n1800-e2-seed20260822
```

## Reproducibility contract

A completed adapter's `train_summary.json` records:

- Git commit and dirty-worktree state;
- train and validation SHA256 hashes;
- model revision and model configuration hashes;
- seed, data seed, `PYTHONHASHSEED`, world size, and visible GPUs;
- Python/platform, NVIDIA driver, Torch/CUDA/cuDNN, and package versions;
- complete arguments, step history, final metrics, runtime, and peak GPU memory.

A formal run also requires `PYTHONHASHSEED=20260822` and `--require-clean-git`; it
fails before model loading when either condition is not satisfied.

### Per-run provenance artifacts

Every formal run must preserve its exact launch provenance in the model output
directory. Do not rely on terminal history, mutable shell variables, a SwanLab
page, or a human-written experiment journal as the only record. The evidence
chain consists of:

- `run_provenance.json`: exact argv, Git commit and worktree state, local model
  revision/config hash, runtime package versions, visible GPUs, seeds, source
  paths, row counts, and SHA256 hashes;
- `data_manifest.snapshot.json`: an immutable snapshot of selected pools, quotas,
  schemas, the task-disjoint split, evaluation exclusion, and source hashes;
- `train_summary.json`: trainer-recorded arguments, step history, final metrics,
  runtime, and peak GPU memory;
- `checkpoint-*`, final adapter files, the raw launch log, and the single
  world-rank-zero SwanLab run.

Create the provenance and manifest snapshot immediately after launch while the
original command array and environment are still available. These files belong
under the ignored model output directory. Do not put machine-specific paths or
per-run journals under `experiments/` or commit them to the repository. A later
handoff must be able to reconstruct the run without the originating terminal.

The output directory must be new or empty unless an explicit checkpoint resume
is requested. SwanLab is initialized only on world rank zero, so a four-GPU run
creates one experiment rather than four duplicate runs.

`--full-determinism` requests deterministic PyTorch algorithms. Exact bitwise
identity is only expected on the same hardware, driver, CUDA, Torch,
Transformers, Liger, and Triton stack. Across stacks, reproducibility is judged
by frozen artifacts/configuration and bounded metric agreement, not byte-for-byte
floating-point equality.

## Metrics and model selection

SwanLab curves record training loss, validation loss, learning rate, gradient
norm, step time, throughput, epoch, and GPU memory. Loss is a training-health
signal, not the shopping score.

After SFT, use the frozen development rollout to report strict success,
`gold_purchase`, reward-valid and done rates, guard rejection reasons, grounded
question rate, unnecessary asking on complete requests, gap no-ask rate, reward
types, and average steps. Select the recipe using development results. Do not
run or use the final evaluation set until the SFT/GRPO recipe is frozen.

GRPO consumes a reviewed merged model, not the LoRA adapter directly. Merging is
a separate explicitly authorized step after adapter and development acceptance.
