# Multi-turn clarification data pipeline

For a Chinese review of the runtime roles, all three Teacher-data types, SFT
acceptance, and Qwen3.8 scaling policy, see
[`multiturn-teacher-sft-review.md`](multiturn-teacher-sft-review.md).

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
- Clarification asks for shopper-owned goal information: requirements, constraints,
  preferences, compatibility, use context, or budget. It must not ask the shopper to
  report catalog facts about an unspecified product; those come from shop tools.
- Shopper answers are natural first-person paraphrases. The separate `used_facts`
  audit retains verbatim source facts, so natural wording does not weaken provenance.

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

Each successful task normally costs one Shopper call. If JSON or audit validation
fails, the generator asks the same Shopper to repair its response, up to
`--opening-attempts` attempts (default: 3). A task that still fails is reported as
`opening_error` and skipped, while later tasks continue. Existing task IDs resume
without another call, and the final summary reports `generated`, `failed`, and
`skipped_existing`. Infrastructure failures still stop the run instead of being
hidden as content failures. The generator fails if the underlying full-goal hash
changed. The audit validator requires each omitted fact to be copied from the full
goal and absent from the opening.
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
      --workers 4 \
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

For a local model served with a finite context window, set `--context-window` to the
same value as the server and enable compaction. The client counts the rendered prompt
through vLLM's `/tokenize` endpoint and, when necessary, removes only the oldest
complete assistant/tool groups. It preserves the fixed task prompt and the most recent
tool observation. The safety margin reserves room beyond `--max-tokens`; 1024 is the
current local-Qwen pilot setting.

When the Teacher endpoint and ShopSimulator have concurrent capacity,
`--workers 4` follows the same bounded scheduling policy as the original single-turn
Teacher collector. Every trajectory receives independent Actor and Shopper clients.
The scheduler never keeps more possible successes in flight than the remaining
`--target-accepted` count, appends results only in the main thread, and resumes by
task-attempt key from `raw.jsonl`. For a local vLLM launched with
`--max-num-seqs 1`, extra workers only queue requests; increase vLLM concurrency or
use a concurrent remote Teacher before expecting a throughput gain.

The earlier `--teacher-first-ask` mode is retained only for diagnosing autonomous
Teacher behavior. It should not be used to build the formal gap-positive SFT set: it
forces an ask but does not ensure that the question targets the frozen gap, and the
same Teacher still has to solve the entire shopping trajectory online.

Neither composite collection nor first-ask constraints are Agent runtime rules. Do not
use them for baseline evaluation, SFT evaluation, GRPO, or final evaluation. Formal SFT
data should mix replay-verified gap-positive demonstrations with successful
complete-request trajectories so that the model also learns when not to ask.

The common Reward v3 gate remains unchanged, but it is no longer the only gate.
Policy-specific acceptance additionally requires: grounded audited asks for
`autonomous-gap-v1`; one grounded replay-verified ask for `composite-replay-v1`; and
zero asks in standard mode for `complete-no-ask-v1`. Only rows passing both the common
and type-specific gates enter the SFT, train, and validation JSONL artifacts.

CLI policy selection is explicit:

- Frozen opening tasks with neither `--composite-teacher` nor
  `--complete-no-ask` collect `autonomous-gap-v1`.
- Frozen opening tasks with `--composite-teacher` collect
  `composite-replay-v1`.
- Full ShopSimulator tasks with `--complete-no-ask` remove the Shopper and
  `ask_shopper` tool entirely and collect `complete-no-ask-v1`.
