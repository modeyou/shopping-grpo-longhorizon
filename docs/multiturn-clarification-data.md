# Multi-turn clarification data pipeline

This pipeline adds ShopSimulator-native Shopper dialogue without exposing the
private full goal to the Actor. It does not use personas, persona masking, or an
LLM critic.

## Runtime contract

- ShopSimulator owns the full instruction and goal options.
- The opening generator creates one underspecified request per task and freezes it.
- The Actor sees only the opening and may call ask_shopper zero, one, or two times.
- The Shopper answers only from the private full goal and prior dialogue.
- Missing facts receive a no-preference or uncertain answer, not an invented fact.
- Actor and Shopper LLM calls are counted separately in each trajectory.

New rows contain an opening_audit object with omitted dimensions and verbatim
omitted facts. This is provenance metadata only. The rollout sends only the
initial_request to the Actor, and SFT conversion never serializes audit fields into
training messages.

## Generate frozen openings

Run from the repository root with ShopSimulator already running:

    export PYTHONPATH=./src
    python scripts/generate_multiturn_tasks.py \
      --tasks data/grpo/train.jsonl \
      --output outputs/multiturn/openings-pilot-01.jsonl \
      --limit 10 \
      --model deepseek-v4-flash

Each new task costs one Shopper call. Existing task IDs resume without another call.
The generator fails if the underlying full-goal hash changed. The audit validator
requires each omitted fact to be copied from the full goal and absent from the opening.
Openings generated before `opening_audit` was introduced cannot be used by composite
collection. Generate them into a new file instead of resuming the legacy file; the
collector rejects such rows before making any LLM call.

## Collect replay-verified gap-positive Teacher demonstrations

    python scripts/collect_multiturn_sft_data.py \
      --tasks outputs/multiturn/openings-pilot-01.jsonl \
      --output-dir outputs/multiturn-sft/composite-pilot-01 \
      --limit 10 \
      --model deepseek-v4-flash \
      --shopper-model deepseek-v4-flash \
      --composite-teacher \
      --target-accepted 3 \
      --max-steps 35

Composite collection follows the upstream project's successful rejection-sampling
pattern. It first gives the standard shopping Teacher the complete ShopSimulator goal.
Only after that trajectory passes the unchanged Reward v3 gold gate does it spend one
call generating a question targeted at `opening_audit` and one grounded Shopper-answer
call. It then prepends that exchange to the gold action backbone and replays every
action from a fresh multi-turn reset. A row is accepted only when replay is legal and
again ends in a valid Reward v3 gold purchase.

`--target-accepted` stops after the requested number of accepted rows and supports
resuming from the same `raw.jsonl`. Failed gold backbones do not spend question or
Shopper calls. Start with three accepted rows; this is an integration pilot, not model
evaluation.

The earlier `--teacher-first-ask` mode is retained only for diagnosing autonomous
Teacher behavior. It should not be used to build the formal gap-positive SFT set: it
forces an ask but does not ensure that the question targets the frozen gap, and the
same Teacher still has to solve the entire shopping trajectory online.

Neither composite collection nor first-ask constraints are Agent runtime rules. Do not
use them for baseline evaluation, SFT evaluation, GRPO, or final evaluation. Formal SFT
data should mix replay-verified gap-positive demonstrations with successful
complete-request trajectories so that the model also learns when not to ask.

The existing Reward v3 gate remains unchanged. Only valid gold-purchase trajectories
enter the SFT, train, and validation JSONL artifacts.
