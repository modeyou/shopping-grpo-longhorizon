# Multi-turn clarification data pipeline

This pipeline adds ShopSimulator-native Shopper dialogue without exposing the
private full goal to the Actor. It does not use personas, persona masking, forced
first questions, or an LLM critic.

## Runtime contract

- ShopSimulator owns the full instruction and goal_options.
- generate_multiturn_tasks.py asks a Shopper LLM to create one underspecified
  opening per task. The opening is frozen and reused across later stages.
- The Actor sees only that opening and may call sk_shopper(question) zero, one,
  or two times.
- Each answer is generated from the private full goal. Missing facts receive an
  explicit no-preference/uncertain answer rather than a hallucinated preference.
- Actor and Shopper LLM call counts are stored separately in every trajectory.

## Produce a small pilot

Start the environment first, then generate frozen openings:

`ash
export PYTHONPATH="D:\shopping-grpo-longhorizon/src"
python scripts/generate_multiturn_tasks.py \
  --tasks data/grpo/train.jsonl \
  --output data/multiturn/pilot_tasks.jsonl \
  --limit 10 \
  --model deepseek-v4-flash
` 

Collect autonomous Teacher trajectories and build SFT JSONL artifacts:

`ash
python scripts/collect_multiturn_sft_data.py \
  --tasks data/multiturn/pilot_tasks.jsonl \
  --output-dir outputs/multiturn-sft/pilot-01 \
  --limit 10 \
  --model deepseek-v4-flash \
  --shopper-model deepseek-v4-flash \
  --max-shopper-questions 2 \
  --max-steps 35
` 

Both scripts resume by task ID. Re-running opening generation skips matching rows
and fails if the underlying ShopSimulator full-goal hash changed. The collection
output uses the existing Reward v3 gate: only valid gold_purchase trajectories
enter sft.jsonl, 	rain.jsonl, and alidation.jsonl.

The private full goal is deliberately absent from the frozen task row, Actor
messages, public reset result, and SFT row.
