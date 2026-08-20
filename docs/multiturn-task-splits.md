# 多轮任务池冻结与数据隔离

本文说明项目如何直接从固定的 ShopSimulator 环境构建自己的多轮 SFT、GRPO 和评测任务池，
不复用参考项目现成的数据划分。

## 1. 隔离边界

任务在任何 LLM 调用之前分配到固定候选池。隔离按候选池而不是最终 accepted 轨迹执行：

- `sft_candidates` 中生成过 opening 或采集过 Teacher 轨迹的任务，即使最终 rejected，也不回流 GRPO。
- `grpo_train` 与 `grpo_validation` 不与 SFT candidate 或评测任务重叠。
- `evaluation` 在 opening、Teacher、prompt 调试和训练之前冻结，之后不得用于任何训练期调试。
- 参考项目已有 SFT、Pure V4、GRPO、validation 和 Final-200 task ID 全部排除。

原 Final-200 可以继续作为外部可比基准，但不属于本文生成的新项目内评测池。

## 2. Task ID 从哪里来

ShopSimulator v2.1 的 task ID 是运行时 goal 列表的索引，不是随机生成的新编号。冻结脚本读取内嵌的
`fine_items_eval_train_all.json.gz`，按环境相同的 ASIN 去重和非空 attributes 规则计算 goal 数量。
当前环境包含 23,421 个合法 goal，合法 task ID 范围是 `0..23420`。

脚本在抽样前验证：

- `data/environment.json` 的环境版本、Reward、Observation 和工具契约；
- ShopSimulator source commit；
- gzip 解压后产品 JSON 的 SHA-256；
- 所有 exclusion ID 都位于合法 goal 范围内。

环境源码固定 goal 顺序的 seed 为 `223`。该值与产品 hash、ShopSimulator commit 一起写入 metadata，
共同定义 task ID 到私有目标的映射。

## 3. 确定性选择

候选 ID 排除全部参考数据后，按以下 key 升序排列：

```text
SHA256("<seed>:<task_id>")
```

默认 seed 是 `20260821`。这种方法不依赖 Python `random` 的内部状态。按固定顺序切分：

1. `evaluation`: 500
2. `sft_candidates`: 3,000
3. `grpo_validation`: 500
4. `grpo_train`: 5,000
5. `reserve`: 所有剩余任务

## 4. 生成命令

```bash
cd ~/shopping-grpo
export PYTHONPATH=./src

python scripts/freeze_multiturn_task_splits.py
```

输出位于：

```text
data/multiturn/tasks/evaluation.jsonl
data/multiturn/tasks/sft_candidates.jsonl
data/multiturn/tasks/grpo_validation.jsonl
data/multiturn/tasks/grpo_train.jsonl
data/multiturn/tasks/reserve.jsonl
data/multiturn/tasks/metadata.json
```

脚本可重复运行：参数和输入完全相同时不会改写文件；只要任何既有输出不同，就会拒绝覆盖。
若确需更换 seed、数量或环境版本，应使用新的输出目录并进行一次显式的数据版本迁移。

## 5. 后续采集入口

正式 opening 只能从新 SFT candidate pool 生成：

```bash
python scripts/generate_multiturn_tasks.py \
  --tasks data/multiturn/tasks/sft_candidates.jsonl \
  --held-out-tasks data/multiturn/tasks/evaluation.jsonl \
  --output outputs/multiturn/openings-project-v1.jsonl \
  --model qwen3.8-27b \
  --llm-base-url http://127.0.0.1:8001/v1 \
  --api-key local-qwen \
  --disable-model-thinking \
  --opening-attempts 3 \
  --timeout 600 \
  --max-tokens 2048
```

A/B/C Teacher 采集均使用 `sft_candidates` 派生的 frozen opening 或完整 task manifest。GRPO 只使用
新 `grpo_train` 和 `grpo_validation`。新 `evaluation` 只在模型、prompt、Reward 和超参数冻结后运行。

此前基于 `data/grpo/train.jsonl` 生成的 opening 和轨迹保留在 `outputs/` 作为 pipeline pilot，
不合并到正式项目 SFT。
