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

## Collect gap-positive Teacher demonstrations

    python scripts/collect_multiturn_sft_data.py \
      --tasks outputs/multiturn/openings-pilot-01.jsonl \
      --output-dir outputs/multiturn-sft/pilot-02-forced \
      --limit 10 \
      --model deepseek-v4-flash \
      --shopper-model deepseek-v4-flash \
      --teacher-first-ask \
      --max-shopper-questions 2 \
      --max-steps 35

The first pilot showed that an untrained Teacher did not ask consistently even when
budget, capacity, or size was absent. For gap-positive SFT collection, the
teacher-first-ask flag constrains only the first Teacher tool choice. The Teacher
still writes the question, and every later choice is autonomous.

This flag is data supervision, not an Agent runtime rule. Do not use it for baseline
evaluation, SFT evaluation, GRPO, or final evaluation. Formal SFT data should mix
gap-positive demonstrations with successful complete-request trajectories so that
the model also learns when not to ask.

The existing Reward v3 gate remains unchanged. Only valid gold-purchase trajectories
enter the SFT, train, and validation JSONL artifacts.
