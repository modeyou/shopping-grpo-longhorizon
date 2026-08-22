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
