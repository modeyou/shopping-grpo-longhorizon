# 数据目录合同

本项目只允许以下数据分层，不能再把原参考数据和当前正式训练数据放在同一目录层级。

| 目录 | 身份 | 是否可作为当前训练输入 |
|---|---|---|
| `data/reference/sft-v1/` | 原参考项目 Reward v3 SFT | 否 |
| `data/reference/grpo-v1/` | 原参考项目 Reward v3 GRPO | 否 |
| `data/sft/formal-v2/` | 当前验收的 Reward v4 多轮 SFT | 是 |
| `data/grpo/formal-v2/` | 当前验收的 Reward v4 多轮 GRPO | 是 |
| `data/multiturn/` | 共享 task split、DEV openings 和上游协议资产 | 不能直接训练 |
| `outputs/multiturn-sft/` | SFT 候选池、审计池和正式数据的构建源 | 仅作为 staging/source |
| `outputs/models/` | adapter、checkpoint 和 merged model | 模型输入，不是数据集 |
| `outputs/evaluation/` | 轨迹、summary 和评测面板 | 否 |

## 正式 SFT

规范文件固定为：

~~~text
data/sft/formal-v2/
├── train.jsonl
├── validation.jsonl
├── manifest.json
└── promotion.json
~~~

`manifest.json` 必须保持原构建 manifest 的逐字节副本；`promotion.json` 记录从 staging 目录复制到规范目录的来源和哈希。只能通过 `scripts/promote_formal_sft_data.py` 晋升，不能手工复制后修改 manifest。

## 正式 GRPO

规范文件固定为：

~~~text
data/grpo/formal-v2/
├── selection/
│   ├── train-tasks.jsonl
│   ├── validation-tasks.jsonl
│   ├── reward-audit.jsonl
│   └── selection-manifest.json
├── multiturn-train-tasks.jsonl
├── multiturn-validation-tasks.jsonl
├── multiturn-train-gap-openings.jsonl
├── multiturn-train-complete-openings.jsonl
├── multiturn-validation-gap-openings.jsonl
├── multiturn-validation-complete-openings.jsonl
├── multiturn-train.parquet
├── multiturn-validation.parquet
└── manifest.json
~~~

只有 selection schema v2 已证明全部 active task 在 Reward v4 下可达，并且最终 `shopping-multiturn-grpo-dataset-v2`、`status=accepted` 通过 reachability audit、全部哈希和 task-disjoint 校验时，`manifest.json` 才能被 `train_grpo.py` 接受。

## 禁止的模糊路径

以下松散路径不再存在，也不能重新创建：

- `data/sft/train.jsonl`
- `data/sft/validation.jsonl`
- `data/grpo/train.jsonl`
- `data/grpo/validation.jsonl`
- `data/grpo/train.parquet`
- `data/grpo/validation.parquet`

判断数据身份必须依赖规范目录与 manifest schema/status/hash，不能仅依据文件名或 `v1`、`v2` 后缀猜测。
