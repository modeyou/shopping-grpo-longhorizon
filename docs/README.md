# Documentation

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
5. [Final-200 Clean evaluation dataset](evaluation-dataset.md) defines the current
   curated benchmark and its update record.

[Reward v3](reward-v3.md) is the detailed specification shared by collection,
GRPO and evaluation.

[Multi-turn clarification](multiturn-clarification-data.md) documents the frozen
Shopper-opening and autonomous `ask_shopper` Teacher-data pipeline.

[Multi-turn task splits](multiturn-task-splits.md) defines the project-owned,
task-disjoint SFT candidate, GRPO, evaluation, and reserve pools frozen directly
from ShopSimulator.

[Multi-turn Teacher/SFT review](multiturn-teacher-sft-review.md) is the Chinese
study note for Actor/Shopper/environment roles, three Teacher-data types,
policy-specific SFT acceptance, Reward boundaries, and Qwen3.8 scaling.
