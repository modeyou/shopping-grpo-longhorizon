# Multi-turn composite Teacher data design

- Date: 2026-08-20
- Branch: `feat/multiturn-clarification-agent`
- Status: approved for implementation

## Problem

The first autonomous and force-first pilots asked inconsistently or asked about
the wrong missing fact, and then often failed the long shopping rollout. A single
Teacher was being asked to discover an information gap, formulate a question,
interpret the answer, and complete shopping online. The upstream project does not
solve this problem: it gives `deepseek-v4-flash` the complete request and retains
only Reward v3 gold purchases from a large rejection-sampled pool.

## Decision

Build each gap-positive SFT example from two independently verifiable parts:

1. A controlled clarification prefix. The frozen task's `opening_audit` identifies
   the omitted dimensions and verbatim facts. A Teacher writes one natural question
   targeted only at those dimensions. The existing private Shopper answers from the
   ShopSimulator full goal; unsupported questions still receive an explicit unknown
   or no-preference answer.
2. A gold shopping backbone. The standard upstream shopping Teacher receives the
   complete ShopSimulator request and standard shop tools only. Its trajectory must
   pass the unchanged Reward v3 gold-purchase gate.
3. Replay verification. The clarification prefix and shopping actions are replayed
   from a fresh reset using the public opening plus question and answer. The composed
   trajectory is accepted only if every action remains legal and Reward v3 again
   returns a valid gold purchase.

The full goal remains private provenance and is never serialized into Actor messages.
The resulting SFT messages contain the public opening, `ask_shopper`, the grounded
answer, and the verified shop actions. Complete-request gold trajectories without an
ask are mixed separately so the model also learns not to ask when the request is
already sufficient.

## Components

- `multiturn.teacher`: generate and validate a question against `opening_audit`;
  extract a replayable shop-action backbone; compose and replay the final trajectory.
- `collect_multiturn_sft_data.py`: expose composite collection as an explicit mode,
  retain resumability and audit counters, and never silently fall back to the former
  force-first online policy.
- SFT filtering: preserve the existing Reward v3 checks and additionally require one
  grounded clarification for gap-positive rows, matching source-goal hashes, and a
  successful replay.

## Failure handling

- Invalid opening audit, leaked private goal, unsupported question, failed backbone,
  illegal replay action, or non-gold replay: record deterministic rejection metadata.
- API or ShopSimulator infrastructure failure: stop before scheduling another task so
  the same run can resume safely.
- A failed task does not need to be retried immediately; collection proceeds over a
  larger task pool until the requested accepted-row target is reached.

## Minimal verification

- Unit tests for question grounding, private-goal isolation, action extraction, and
  replay rejection.
- A mock end-to-end test proving that the final SFT row contains the clarification
  turn followed by shop actions and contains no private full-goal field.
- A real 3--5 task pilot before any larger collection. Success means at least one
  replay-verified gold row and useful deterministic rejection reasons for failures;
  it is not a statistical model evaluation.

## Explicit non-goals

- No persona or persona masking.
- No LLM critic and no new reward weights.
- No change to formal evaluation behavior.
- No use of upstream committed Teacher rows as this project's generated data.
