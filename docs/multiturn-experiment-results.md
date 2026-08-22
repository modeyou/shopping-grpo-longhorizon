# 当前多轮项目实验结果

> 本文只记录当前多轮澄清项目（Environment v2.1、Reward v4、observation/tool schema v2）的实验。原参考项目在 `experiments/` 下的 Reward v3 单轮结果不属于本文，也不与本文指标直接比较。

## Qwen3.5-2B 训练前 DEV-500 基线

本次基线用于后续 SFT、GRPO 的开发集配对比较，不是最终测试集结果。每个条件固定 500 个任务、每任务一次确定性 rollout，共 1,500 条轨迹，全部完成且无缺失任务。

### 冻结协议

- Actor：`Qwen/Qwen3.5-2B`，revision `15852e8c16360a2fea060d615a32b45270f8a8fc`
- 开发集：`data/multiturn/evaluation-dev-v2`
- tasks SHA-256：`93fda78a7a69b1ca68262a5821571055ed26d217791542723e2503fb4bc5fc3b`
- gap openings SHA-256：`fdc8bbc5ce6cfa485ac73fac1e16f8a66bd983e93ac0419a01bf587e26129588`
- complete openings SHA-256：`f32e83c1fbe9ea1add7022353358a3415c37b95beb95329446e3d1bdf4749377`
- 环境与奖励：ShopSimulator Environment v2.1、Reward v4
- 推理：temperature `0`、top_p `1`、max_steps `35`、每轮最多 `512` tokens、context window `24576`
- 上下文：不启用 compaction；observation budgets `1536/4096/768`，search top-k `20`
- Actor thinking：关闭
- ask-enabled 条件的 Shopper：`deepseek-v4-flash-0731`，最多 2 次提问，thinking 关闭

### 执行过程

评测最初由单个 Actor 服务串行运行；为了提高吞吐，在不中断已完成轨迹的前提下切换为 4 个独立 Actor 端点、4 个确定性任务分片。切换时已有轨迹按 task ID 播种到分片，随后合并并重新计算汇总：

- `gap-ask-enabled`：切换时已完成 500 条，四分片各播种 125 条。
- `gap-ask-disabled`：切换时已完成 172 条，四分片各播种 43 条。
- `complete-ask-enabled`：从 0 条开始四分片运行。
- 最终三个条件均为 `500/500`；分片合并未发现缺失或重复冲突。

大体积 `trajectories.jsonl` 和运行日志保留为服务器生成物，不提交 Git。

### 结果

| 条件 | Done | Strict success | Reward valid | Mean reward | Guards | Context overflow | Asked tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| gap + ask enabled | 41/500 (8.2%) | 2/500 (0.4%) | 41/500 (8.2%) | -0.040576 | 2,243 | 19 | 37 |
| gap + ask disabled | 74/500 (14.8%) | 1/500 (0.2%) | 73/500 (14.6%) | -0.088122 | 2,151 | 19 | 0 |
| complete + ask enabled | 53/500 (10.6%) | 3/500 (0.6%) | 52/500 (10.4%) | -0.053683 | 2,164 | 17 | 29 |
| 合计 | 168/1500 (11.2%) | 6/1500 (0.4%) | 166/1500 (11.07%) | -0.060794 | 6,558 | 55 | 66 |

Strict-success task IDs：

- gap + ask enabled：`5698`, `8100`
- gap + ask disabled：`12723`
- complete + ask enabled：`2865`, `3366`, `3856`

总 Reward 为 `-91.19071906354516`。ask-enabled 两个条件合计调用 Shopper 92 次。

### 诊断结论

基座模型的首要失败不是 Reward v4 或关键观察字段丢失，而是动作协议不稳定：大量轨迹因 `click_not_in_previous_observation` 等非法点击提前失败；三个条件共有 6,558 次 guard rejection，而 critical-footer failure 为 0。SFT 的首要目标因此是学习“只操作当前可见对象、被 guard 拒绝后重新观察和规划、持续到合法终局”，同时保留缺口请求的有根据提问与完整请求的少提问行为。
## SFT 数据冻结协议

三类原始 Teacher 数据是在 Reward v3 下采集的，原始数量为 complete 2,083、composite 1,619、autonomous 914。当前 Reward v4 项目不直接拼接这些派生 `sft.jsonl`，而是从不可变 `raw.jsonl` 重新审计：先重放仓库既有的策略级、工具级和 v3 strict-gold 门槛，再根据冻结商品数据重建 v4 goal，对教师实际购买 ASIN、实际 options 和变体价格执行 Reward v4 评分。只有 v3/v4 均为有效 `gold_purchase` 的轨迹进入新候选池。

候选池必须属于 `sft_candidates`，并显式排除当前开发集、sealed 正式评测集、GRPO train/validation 和原参考项目 Final-200。随后按 task ID 与 source-goal hash 去重，使用本地 Qwen3.5-2B chat template 检查完整渲染、assistant-loss token 和 24,576 token 上限。

初版混合目标按 assistant-loss token 计算，而不是按原始行数机械抽样：complete/composite/autonomous 为 `50%/30%/20%`。64 条 smoke 的暂定行数为 `32/20/12`；正式行数将在 v4 合格数量、跨策略重复和 token 分布统计后冻结。部分 complete 样本还将保留零提问轨迹但提供含 `ask_shopper` 的完整工具 schema，用于学习“工具可用但请求完整时不必提问”；该 schema augmentation 必须在 manifest 中单独标记。

## 正式 Reward v4 LoRA SFT

### 数据与复现绑定

正式 SFT 使用 `outputs/multiturn-sft/mix-formal-1800-v4-seed20260822`，共 1,800 条样本，按 task 隔离为 1,620 条训练和 180 条验证。训练与 `data/evaluation/tasks.jsonl` 的 task ID 重叠数为 0。

| 策略 | 行数 | Assistant tokens | Token 份额 |
|---|---:|---:|---:|
| complete-no-ask-v1 | 802 | 212,029 | 50.22% |
| composite-replay-v1 | 627 | 126,253 | 29.91% |
| autonomous-gap-v1 | 371 | 83,897 | 19.87% |

冻结哈希：

- manifest：`11be05b2d4e2cfb49529542a23030988e21ea59266cad97b48598302e56e4eeb`
- train：`bbfd477b5f4a776d64dd0c6e338829e8b76cd6529ca7806bef3def7c119ac9ae`
- validation：`40d648d4e8cf2f47bb1af6ef330907da25e2e7cbf45fe7e8856aad833652ab8b`
- 最终评测 task：`d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5`

### 正式训练

正式运行基于 Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，绑定 Git commit `8c929be9f025c34d8c9620441435d46e39dd1bc2`。使用 CUDA 0–3、bf16 LoRA、最大长度 24,576、有效全局 batch 8、2 epoch、峰值学习率 `1e-4`、3% warmup 后线性衰减。LoRA rank/alpha/dropout 为 16/32/0.05，启用 SDPA、Liger、梯度检查点和完全确定性。

输出目录：`outputs/models/multiturn-sft-v4-1800-e2-seed20260822-r2`。SwanLab project 为 `shopping-multiturn-agentic`，run name 为 `qwen35-2b-sft-lora-v4-n1800-e2-seed20260822-r2`，run ID 为 `48kdp8mk`。

| 指标 | 结果 |
|---|---:|
| Optimizer steps / epoch | 406 / 2.0 |
| 运行时间 | 6,940.93 秒 |
| 首个/最后 train loss | 0.408989 / 0.158007 |
| 最低 train loss | 0.113681 |
| Trainer 平均 train loss | 0.187540 |
| 最终/最低 eval loss | 0.182440 / 0.182392 |
| 峰值 GPU allocated memory | 16.58 GiB |
| LoRA 参数 / tensors | 16,819,200 / 372 |

最终 adapter SHA-256 为 `7a5890e51434e0415bfbcb63f8a7f18292d6c0e38c4efac0aeb1f48298095d55`。运行目录保留 `run_provenance.json`、`data_manifest.snapshot.json`、`train_summary.json` 和 `completion_audit.json`，不能用本文或 SwanLab 页面代替这些机器生成证据。

## SFT 候选 DEV-500 对比

checkpoint-200 接近第一个 epoch 结束；final-2epoch 为 step406。两个候选均先与基座合并为独立 bf16 模型，再按与 Base 相同的 dev500×3、Reward v4、四分片确定性协议评测。最终 200-task 评测集未使用。

### 严格成功率

| 模型 | Gap+Ask | Gap-NoAsk | Complete+Ask | 合计 |
|---|---:|---:|---:|---:|
| Base Qwen3.5-2B | 2/500（0.4%） | 1/500（0.2%） | 3/500（0.6%） | 6/1500（0.4%） |
| checkpoint-200 | 332/500（66.4%） | 246/500（49.2%） | 367/500（73.4%） | 945/1500（63.0%） |
| final-2epoch | 346/500（69.2%） | 245/500（49.0%） | 191/500（38.2%） | 782/1500（52.13%） |

### 稳定性与奖励

| 模型 | Done | Reward valid | Mean reward | Guards | Context overflow |
|---|---:|---:|---:|---:|---:|
| Base | 168/1500 | 166/1500 | -0.060794 | 6,558 | 55 |
| checkpoint-200 | 1,477/1500 | 1,471/1500 | 0.563517 | 63 | 1 |
| final-2epoch | 1,254/1500 | 1,251/1500 | 0.471366 | 26 | 2 |

final-2epoch 相对 checkpoint-200 在 Gap+Ask 上增加 14 个严格成功，但在 Complete+Ask 上减少 176 个严格成功、减少 233 个正常终局，并出现 239 个 error。它的 guard 更少不能单独解释为更稳定，因为大量 complete 轨迹在产生更多动作前已经提前失败。

### 澄清行为

| 模型 | Gap 提问任务 | Complete 不必要提问 |
|---|---:|---:|
| Base | 37/500（7.4%） | 29/500（5.8%） |
| checkpoint-200 | 500/500（100%） | 492/500（98.4%） |
| final-2epoch | 496/500（99.2%） | 249/500（49.8%） |

SFT 已显著修复 Base 的动作协议和不提问问题，但产生了过度提问倾向。较低 teacher-forcing eval loss 没有保证更高长程成功率，因此不能只按 loss 选择 checkpoint。

### 当前选择状态

checkpoint-200 是当前领先候选，但尚未冻结为 GRPO 基座。正在使用 SHA-256 为 `5754aaaf1a4b67c47751f4e35782866a4794d49910bbcf9651eff2d5080b2d1a` 的固定 dev200 sweep manifest 筛选 checkpoint-250、300、350。该 sweep 每个候选评测 Gap+Ask、Gap-NoAsk 和 Complete+Ask 三个条件；完成后仅对有希望超过 checkpoint-200 的候选补充完整 dev500×3。

在 sweep 和候选复核完成前，不创建 `selected-for-grpo` 记录、不启动 GRPO，也不使用最终 200-task 评测集。