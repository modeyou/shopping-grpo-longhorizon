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
| **checkpoint-325** | **345/500（69.0%）** | **264/500（52.8%）** | **361/500（72.2%）** | **970/1500（64.67%）** |
| checkpoint-350 | 348/500（69.6%） | 248/500（49.6%） | 355/500（71.0%） | 951/1500（63.40%） |
| checkpoint-375 | 343/500（68.6%） | 251/500（50.2%） | 346/500（69.2%） | 940/1500（62.67%） |
| checkpoint-406（final-2epoch） | 346/500（69.2%） | 245/500（49.0%） | 191/500（38.2%） | 782/1500（52.13%） |

### 稳定性与奖励

| 模型 | Done | Reward valid | Mean reward | Guards | Context overflow |
|---|---:|---:|---:|---:|---:|
| Base | 168/1500 | 166/1500 | -0.060794 | 6,558 | 55 |
| checkpoint-200 | 1,477/1500 | 1,471/1500 | 0.563517 | 63 | 1 |
| **checkpoint-325** | **1,490/1500** | **1,486/1500** | **0.6010** | 32 | 待细分 |
| checkpoint-350 | 1,486/1500 | 1,476/1500 | 0.5886 | **27** | 待细分 |
| checkpoint-375 | 1,485/1500 | 1,482/1500 | 0.5669 | 42 | 待细分 |
| checkpoint-406（final-2epoch） | 1,254/1500 | 1,251/1500 | 0.471366 | 26 | 2 |

final-2epoch 相对 checkpoint-200 在 Gap+Ask 上增加 14 个严格成功，但在 Complete+Ask 上减少 176 个严格成功、减少 233 个正常终局，并出现 239 个 error。它的 guard 更少不能单独解释为更稳定，因为大量 complete 轨迹在产生更多动作前已经提前失败。

### 澄清行为

| 模型 | Gap 提问任务 | Complete 不必要提问 |
|---|---:|---:|
| Base | 37/500（7.4%） | 29/500（5.8%） |
| checkpoint-200 | 500/500（100%） | 492/500（98.4%） |
| checkpoint-325 | 待轨迹细分 | 469/500（93.8%） |
| checkpoint-350 | 待轨迹细分 | 477/500（95.4%） |
| checkpoint-375 | 待轨迹细分 | 478/500（95.6%） |
| checkpoint-406（final-2epoch） | 496/500（99.2%） | 249/500（49.8%） |

SFT 已显著修复 Base 的动作协议和不提问问题，但产生了过度提问倾向。较低 teacher-forcing eval loss 没有保证更高长程成功率，因此不能只按 loss 选择 checkpoint。

### 当前选择状态

固定 dev200 sweep 已覆盖全部保留 checkpoint：200、225、250、275、300、325、350、375、400、406。manifest SHA-256 为 `5754aaaf1a4b67c47751f4e35782866a4794d49910bbcf9651eff2d5080b2d1a`。每个候选在同一批任务上执行 Gap+Ask、Gap-NoAsk 和 Complete+Ask 三个条件，各 200 条轨迹；Reward contract 固定为 v4，ask-enabled Shopper 固定为 `deepseek-v4-flash-0731`，最终 200-task 评测集未使用。10 个候选均通过轨迹数量、模型名、Reward contract 和 API 基础设施错误审计。

| Checkpoint | Gap+Ask | Gap-NoAsk | Complete+Ask | 总 strict | Gap gain | Complete 多余提问 | Mean reward | Done | Reward valid | Guards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 62.5% | 48.5% | 71.0% | 60.67% | +14.0pp | 99.0% | 0.5383 | 591/600 | 588/600 | 24 |
| 225 | 64.5% | 46.5% | 68.0% | 59.67% | +18.0pp | 91.0% | 0.5438 | 594/600 | 587/600 | 21 |
| 250 | 60.5% | 45.5% | 64.5% | 56.83% | +15.0pp | 98.5% | 0.5070 | 591/600 | 586/600 | 36 |
| 275 | 60.5% | 48.5% | 66.0% | 58.33% | +12.0pp | 96.5% | 0.5298 | 593/600 | 577/600 | 28 |
| 300 | 65.0% | 48.5% | 66.5% | 60.00% | +16.5pp | 97.5% | 0.5520 | 592/600 | 586/600 | 19 |
| **325** | **66.5%** | **54.0%** | **70.0%** | **63.50%** | **+12.5pp** | **94.5%** | **0.5807** | **598/600** | **595/600** | **8** |
| 350 | 67.0% | 51.5% | 67.5% | 62.00% | +15.5pp | 94.5% | 0.5760 | 590/600 | 585/600 | 18 |
| 375 | 68.0% | 52.0% | 66.5% | 62.17% | +16.0pp | 96.0% | 0.5551 | 596/600 | 592/600 | 8 |
| 400 | 65.0% | 49.5% | 68.0% | 60.83% | +15.5pp | 96.0% | 0.5419 | 594/600 | 592/600 | 10 |
| 406 | 66.5% | 49.5% | 67.5% | 61.17% | +17.0pp | 96.0% | 0.5512 | 592/600 | 590/600 | 16 |

checkpoint-325 是 dev200 的领先候选。它的总 strict 为 381/600，超过 checkpoint-375 的 373/600 和 checkpoint-350 的 372/600；同时取得最高 Gap-NoAsk、最高平均奖励、最高 Done、最高 Reward-valid，并以 8 次 guards 与最低值并列。checkpoint-375 的 Gap+Ask 高 1.5 个百分点，但 Complete、平均奖励和总体成功率更低，因此不如 checkpoint-325 均衡。

该 sweep 也确认训练指标并非随 step 单调改善：checkpoint-250 出现明显低谷，325 达到综合峰值，随后 Gap+Ask 仍有局部提升，但 Gap-NoAsk 与总体表现回落。所有 checkpoint 的 Complete 多余提问率仍在 91%–99%，说明 SFT 的主要剩余问题不是不会澄清，而是几乎总会澄清。

### 完整 DEV-500 复核与当前选择

dev200 排名前三的 checkpoint-325/350/375 已在与 checkpoint-200/406 相同的完整 dev500×3、Reward v4、四分片确定性协议下复核。冻结资产 manifest SHA-256 为 `b363a64628f68a588292832f6d01a5a2c5687f29e2f8f119714a46140f5fe03f`，其 tasks、gap openings、complete openings 与既有 dev500 文件逐字节相同。三个 checkpoint 均完成 1,500 条轨迹，未发现 Shopper 401/403、quota 或其他被 sweep 审计拦截的基础设施错误；最终 200-task 评测集未使用。

checkpoint-325 的总 strict 为 970/1500，超过 checkpoint-350 的 951/1500、checkpoint-200 的 945/1500、checkpoint-375 的 940/1500 和 checkpoint-406 的 782/1500。它相对 checkpoint-200 在 Gap+Ask、Gap-NoAsk 分别增加 13、18 个严格成功，Complete 减少 6 个，总计增加 25 个；平均奖励从 0.563517 提升至 0.6010，Done 增加 13，Reward-valid 增加 15，guards 减少 31。

因此 checkpoint-325 是当前开发集选择的 GRPO 基座候选。checkpoint-350 的 Gap+Ask 高 3 个任务，但其 +20.0pp Gap gain 主要来自更低的 Gap-NoAsk，并且总 strict、Complete、平均奖励、Done 和 Reward-valid 都低于 checkpoint-325。checkpoint-375 同样没有在综合指标上超过 checkpoint-325。

Complete 至少一次提问率仍为 93.8%–95.6%，但该指标只表示“问过至少一次”，不是失败率，也未从 strict success 中扣分。现有轨迹的细分结果如下：

| Checkpoint | 0/1/2 问任务 | 0 问 strict | 1 问 strict | 2 问 strict | 完全/近似重复任务 | 触发问题上限 | 提问后无购物动作 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | 8/484/8 | 4/8（50.0%） | 363/484（75.0%） | 0/8（0.0%） | 3/3 | 2 | 1 |
| **325** | **31/461/8** | **22/31（71.0%）** | 336/461（72.9%） | 3/8（37.5%） | **0/2** | 2 | **0** |
| 350 | 23/473/4 | 14/23（60.9%） | 339/473（71.7%） | 2/4（50.0%） | 1/1 | **0** | 1 |
| 375 | 22/475/3 | 13/22（59.1%） | 332/475（69.9%） | 1/3（33.3%） | **0/0** | 1 | **0** |
| 406 | 251/246/3 | 12/251（4.8%） | 179/246（72.8%） | 0/3（0.0%） | **0/0** | 1 | **0** |

五个模型的 Complete 问答均没有使用任何 `used_facts`，符合完整 opening 没有隐藏事实的协议。两问任务至多 8/500，完全或近似重复、问题上限和提问后无购物动作也都很少，因此当前主要行为不是反复提问，而是大多数模型固定先做一次无信息确认。

checkpoint-406 的 Complete 退化主要来自零问组：251 个零问任务只有 4.8% strict、5.2% Done，而一问组仍有 72.8% strict、99.2% Done。因为 Complete 的固定 Shopper 回答不增加目标信息，这种分组差异是行为相关性，不证明提问具有因果收益；它更像是模型是否进入可持续购物动作模式的标记。checkpoint-325 的零问组已经达到 71.0% strict，说明它同时保留了不依赖澄清的购物能力，这进一步支持将其选为 GRPO 基座候选。

GRPO 阶段不应把第一次 Complete 提问设为足以压倒终局成功的硬惩罚。更稳妥的优先级是：Reward v4 strict 终局质量为主；第二次无信息提问、完全或近似重复问题、`shopper_question_limit` 和提问后无购物动作承担更强约束；第一次无信息确认只承担较轻的效率成本。这样才能在减少默认确认的同时，避免复现 checkpoint-406 的零问崩溃。

未经用户单独授权，不执行 checkpoint-325 合并、不启动 GRPO，也不使用最终 200-task 评测集。
