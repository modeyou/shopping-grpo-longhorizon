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

The [`Personalized Agent project contract`](personalized-agent-project-contract.md)
defines the scope and staged decision boundaries for the independent,
reproducible personalization and proactive-clarification project.

The [`reference-project and ShopSimulator review`](research/reference-project-shopsimulator-review.md)
records the primary-source findings that constrain the implementation plan.

The [`personalized data contract`](personalized-data-contract.md) separates
Actor-visible context, private shopper state and audit-only task facts for the
self-generated SFT data pipeline.
