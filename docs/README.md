# Documentation

Follow the guides in workflow order:

1. [Multi-turn data collection](multiturn-clarification-data.md) explains the
   frozen Shopper-opening and `ask_shopper` data pipeline.
2. [SFT](sft.md) trains the first useful Reward v4 shopping agent.
3. [GRPO](grpo.md) and [CARL-BPO](carl-bpo.md) describe the online RL stages.
4. [Multi-turn evaluation protocol](multiturn-evaluation.md) is the current
   authoritative Reward v4 evaluation contract, with a minimum credible tier
   and a complete five-panel tier.
5. [Final-200×3 runbook](final-200-runbook.md) is the executable final
   evaluation handoff for Base, SFT, and the selected RL checkpoint.
6. [Current experiment results](multiturn-experiment-results.md) records the
   development-set Base/SFT/RL evidence.

[Reward v3](reward-v3.md) is the historical reference-project reward contract.

[Reward v4](reward-v4.md) is the current multi-turn training and evaluation
contract. It retains the completed v3/v4 migration history and audit boundary.

[Multi-turn clarification](multiturn-clarification-data.md) documents the frozen
Shopper-opening and autonomous `ask_shopper` Teacher-data pipeline.

[Multi-turn task splits](multiturn-task-splits.md) defines the project-owned,
task-disjoint SFT candidate, GRPO, evaluation, and reserve pools frozen directly
from ShopSimulator.

[Multi-turn Teacher/SFT review](multiturn-teacher-sft-review.md) is the Chinese
study note for Actor/Shopper/environment roles, three Teacher-data types,
policy-specific SFT acceptance, Reward boundaries, and Qwen3.8 scaling.

[The original single-turn evaluation](evaluation.md),
[Final-200 Clean dataset](evaluation-dataset.md), update log, and dashboard are
historical Reward v3 reference artifacts. They are not the current Reward v4 Final-200
multi-turn benchmark and must not be merged with Reward v4 results.

[Current multi-turn experiment results](multiturn-experiment-results.md) records
the Reward v4 development baseline and subsequent SFT/GRPO paired results; it
is intentionally separate from the original reference artifacts in `experiments/`.

[CARL-BPO implementation and validation plan](carl-bpo.md) defines the landed
completion-aligned Root/Local RL design, its Linux runbook, SwanLab and local
audit contracts, and the staged validation and acceptance gates. The landed
implementation does not by itself authorize training or model selection.

[Training, Reward and evaluation decisions](training-reward-evaluation-decisions.md)
is the consolidated Chinese study note for the development/test split, five
evaluation panels, the completed Reward v4 transition, Qwen3.5-2B memory
budget, Rubric/Judge selection, and the online Shopper contract.

[Shopping Agent and Harness review](agent-harness-context-engineering.md) is the
detailed Chinese study note for the Agent/Harness boundary, veRL ToolAgentLoop,
four-rollout GRPO, page-aware context engineering, Action Guard, a complete
clarification example, current implementation gaps, and the multi-turn GRPO
acceptance checklist.
