# 多轮购物 Agent 评测协议

本文是当前多轮项目的权威评测说明。当前运行契约固定为
**ShopSimulator Environment v2.1 + Reward v4**。原参考项目的 Reward v3、Final-200 Clean 和
四面板报告只作为历史单轮基准，不能与
本文结果合并排名。

评测分为两层：

- **当前最小可信评测**：在冻结 DEV-500 上完成的确定性 release benchmark，也是投递简历、项目答辩
  和快速复现的默认版本。由于该集合参与过 checkpoint/方案选择，它不是未见盲测；
- **后续完整评测**：在独立 Final-200 上重跑最小评测，并增加冻结 Rubric、可选轨迹 Judge、统计检验
  和系统化 bad-case 诊断。它增强泛化证据与解释力，但不阻塞当前项目展示。

项目同时回答两个同等重要的问题：

1. RL 是否在 SFT 基础上提高了满足完整需求的最终购买成功率？
2. 模型是否学会在信息缺失时有效澄清，并在信息完整时控制多余提问？

不把所有指标加成一个不可解释的总分。

## 1. 当前冻结资产

| 用途 | 目录 | 任务数 | Reward | 状态 |
|---|---|---:|---|---|
| 当前 release benchmark | `data/multiturn/evaluation-dev-v2/` | 500 | Reward v4 | 已用于模型选择；统一报告当前结果 |
| 后续独立 Final | `data/multiturn/final-200-v1/` | 200 | Reward v4 | 已从 Final-500 结果盲抽并冻结，当前不运行 |
| 历史单轮回归 | `data/evaluation/tasks.jsonl` | 200 | Reward v3 | 只作参考项目归档 |

正式 Final-200 的关键哈希为：

| 资产 | SHA-256 |
|---|---|
| `manifest.json` | `f9a3970c9bc59374ac23741a4a4519dca44832cc55414ce55944bb1fd446551e` |
| `tasks.jsonl` | `f7f646f50c215f816fd9fc48a26e02110624b1ad17d18a9490634cfc46e99829` |
| `gap_openings.jsonl` | `fd65618c11a8b1cd2616e1efd11ba510f291124c5bc2e32567a4fe4efba4780e` |
| `complete_openings.jsonl` | `5acd0788255f6b8ee4fe8f757f20f16dfe675aa402435898de1ce5fb7c78f109` |
| `conditions.jsonl` | `5cda59afeb0050131d6e9c66f26a187093fa5dda0c2dc4244cd349507ccc8f08` |

这些是 manifest 记录的规范 LF 字节哈希；Windows checkout 若自动转换为 CRLF，直接对工作区文件执行
`Get-FileHash` 会不同，运行审计应使用仓库现有的规范化哈希逻辑。

冻结资产已验证：每个 task 都有 gap 和 complete opening；G+ 与 G− 引用同一份 gap opening；
`source_goal_hash` 全部一致；每题恰好对应三个评测条件。

当前正式 SFT、GRPO train/validation、DEV-500、Reward v4 Final-200 和旧 Reward v3 Final-200 的 task ID
两两隔离。后续训练 manifest 仍应显式列出这些集合，不能只依赖上游划分隐式保证。

## 2. 当前被比较的模型

当前 release 报告比较三个已有完整 DEV-500 结果的阶段：

1. **Base**：冻结 revision 的 `Qwen3.5-2B`；
2. **SFT**：开发集选择的 `checkpoint-325`；
3. **BPO v1**：从 SFT checkpoint-325 开始、完成 200 个 optimizer steps 的历史 RL 运行。

CARL-BPO v1 当前只有训练 validation 和 step-260 诊断，没有仓库内可复核的 DEV-500 三条件结果；
CARL-BPO v2.1 仍在运行。二者完成后可追加一行，但不能在结果不存在时把 BPO v1 数字改名为
CARL-BPO。后续运行独立 Final-200 前，必须先冻结唯一 checkpoint。

每个模型必须共享：

- task、opening 和条件映射；
- Actor system prompt、工具 schema 和 Action Guard；
- ShopSimulator、商品数据和 Reward v4；
- Shopper 模型、prompt、参数与最多提问次数；
- temperature、top-p、最大步数、上下文和 observation budgets；
- 汇总代码和固定分母规则。

## 3. 三个配对条件

| 简称 | 条件 | Actor 可见输入 | `ask_shopper` | 回答的问题 |
|---|---|---|---|---|
| G+ | `gap-ask-enabled` | 缺少关键事实的 opening | 可用 | 缺信息时能否主动澄清并完成购买？ |
| G− | `gap-ask-disabled` | 与 G+ 完全相同的 opening | 不可用 | 澄清通道带来了多少实际收益？ |
| C+ | `complete-ask-enabled` | 完整请求 | 可用 | 正常购物能力如何，是否无条件提问？ |

G+ 和 C+ 是两个同等重要的部署场景；G− 是配对诊断对照，不是第三种同权重部署场景。因此：

- 分别报告 G+、G−、C+，不只给一个平均值；
- 不把 G− 按三分之一权重加入“总体模型能力”；
- 若模型选择需要单一排序值，只允许在 DEV 上使用
  `(G+ strict success + C+ strict success) / 2` 作为选择索引，并明确它不是最终评测总分；
- G+−G− 必须同时报告两个绝对成功率；有逐题轨迹时再报告 task-level gains/losses，防止把 G− 退化
  误读成澄清进步。

每个 Actor、task、condition 只运行一次 temperature=0 的确定性 rollout，不报告 Pass@k。

## 4. Reward v4 与严格成功

Reward v4 是当前训练和评测共同使用的确定性终局标准。它把完整需求编译为 category、brand、model、
core function、option 和 price 原子，区分 hard、required 和 soft，并使用实际选择的 variant 与可验证价格
判定购买结果。

最核心指标定义为：

```text
strict_success =
  reward_version = "shopsimulator-reward-v4"
  AND trajectory status = "done"
  AND trajectory/terminal done = true
  AND terminal over = true
  AND reward_valid = true
  AND purchase_success = true
  AND reward_type = "gold_purchase"
  AND termination_reason = "gold_purchase"
```

同时报告：

- `purchase_success`：`gold_purchase` 或 `valid_alternative_purchase`；
- `done_rate` 和 termination reason；
- `reward_valid_rate`；
- mean final Reward v4；
- reward type 分布；
- infrastructure-invalid 数量和 task IDs。

固定分母内的缺失、API 失败、环境失败和无效 Reward 不能静默删除。Reward v4 不因调用
`ask_shopper` 自动加分；澄清能力由配对条件和澄清指标单独判断。

## 5. 最小可信评测

### 5.1 什么时候采用

这是当前项目默认必须完成的版本。它不调用 Rubric LLM 或轨迹 Judge，依靠环境终局、冻结 Shopper
审计和代码指标回答两个核心问题。它适用于：

- DEV-500 上统一比较 Base/SFT/BPO，并追加后续 CARL-BPO；
- 当前 GitHub release、简历、README 和答辩；
- 短期项目交付；
- 在完整 Judge 评测尚未运行时提供可信结论。

### 5.2 必须报告的指标

#### A. 最终购买结果

每个模型分别报告：

- G+ strict success；
- C+ strict success；
- G− strict success；
- 三条件 purchase success、Reward-valid、Done 和 mean Reward v4；
- reward type 与 termination reason 分布。

G+ 和 C+ 是并列主结果。G− 只用于解释澄清通道，不参与三条件等权总分。

#### B. 澄清能力

- G+ 至少提问一次的 task rate；
- G+ grounded-question task rate；
- G+ no-ask rate；
- C+ unnecessary first-ask rate；
- C+ 第二次无信息提问、完全/近似重复问题和问题上限触发率；
- 提问后无后续购物动作的 task rate；
- 推荐补充 G−→G+ strict gains、losses、ties 和对应 task IDs；
- 推荐补充 G−→G+ purchase-success gains/losses。

`grounded` 只表示 Shopper 回答使用了冻结 omitted facts，不单独证明该问题具有因果必要性。因果证据
来自同 task、同 gap opening 的 G+ 与 G− 结果迁移。

#### C. 执行可靠性

- 环境购物动作数与 Actor 总回合数；
- Action Guard rejection 及原因；
- malformed/illegal tool call；
- duplicate action、duplicate search 和 repeat loop；
- max steps；
- context overflow、observation truncation；
- Actor/Shopper API error；
- infrastructure-invalid task IDs。

### 5.3 DEV 结果解释规则

DEV-500 已经参与 SFT checkpoint 与 RL 方案选择，因此当前结果按以下边界解释：

1. 先拒绝任务缺失、Reward 版本错误、模型名错误或基础设施异常未解释的 run；
2. G+ 和 C+ strict success 形成并列主指标，Pareto 更优的候选优先；
3. 两者互有胜负时，使用 `(G+ + C+) / 2` 的 DEV 选择索引；
4. 检查 G−→G+ 净迁移不得因 G− 大幅退化而虚增；
5. 依次用 Reward-valid、Done、较低的严重澄清错误和较少 Guard 作为 tie-breaker；
6. 记录所有候选结果和选择理由；运行 Final-200 前预先冻结唯一 RL checkpoint。

RL 没有超过 SFT 也是有效实验结论，不能通过改看 Final、改分母或只挑某一条件来包装提升。

### 5.4 最小执行流程

```text
当前 release：DEV-500 × G+/G−/C+
    -> Base / SFT-325 / BPO-v1 每格 500 条
    -> 确定性汇总与 task-level 配对迁移
    -> 如实标记 frozen development benchmark

后续增强：
    -> CARL-BPO v2.1 完成后追加同协议 DEV-500
    -> 冻结唯一模型并一次性运行 Final-200
```

当前入口默认使用 DEV-v2：

```bash
export MULTITURN_ASSET_DIR="$PWD/data/multiturn/evaluation-dev-v2"
export EVAL_OUTPUT_DIR="$PWD/outputs/evaluation/release-dev500/MODEL_LABEL"
bash scripts/evaluate_multiturn_parallel.sh MODEL_LABEL
```

正式运行前应先使用小规模 limit 做基础设施 smoke；smoke 产物与正式新目录隔离，正式 run 不设置 limit。

## 6. 最小评测的统计与展示

最低交付必须给原始计数和固定分母，例如 `345/500 (69.0%)`，不能只给百分比。推荐同时给：

- success rate 的 task bootstrap 95% CI；
- Base→SFT、SFT→RL 的 paired bootstrap 差值 CI；
- G−↔G+ 二元成功迁移的 McNemar 检验；
- gains、losses 和 ties；
- 每个关键退化类型的 task IDs。

如果暂时没有实现置信区间或 G−→G+ 逐题迁移，固定分母的原始计数、绝对成功率和完整失败分布仍
构成当前最小可信评测；逐题迁移和统计推断可在完整评测阶段补充。模型之间已有配对轨迹时，应像
BPO v1 对 SFT 一样报告 gains/losses，而不是只比较聚合百分比。

推荐结果首页只放：

| 模型 | G+ strict | C+ strict | G− strict | C+ unnecessary ask | Done | Reward valid |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2/500 | 3/500 | 1/500 | 29/500 | 168/1500 | 166/1500 |
| SFT-325 | 345/500 | 361/500 | 264/500 | 469/500 | 1490/1500 | 1486/1500 |
| BPO v1 | 345/500 | 360/500 | 263/500 | 461/500 | 1487/1500 | 1483/1500 |

## 7. 完整评测体系

完整评测保留最小可信评测的全部结果，并增加五个互不覆盖的面板。LLM Judge 只增强需求级与过程级
解释，不能覆盖 Reward v4 的确定性终局结论。

### 面板 A：Reward 与终局

沿用最小评测全部 Reward v4 指标、固定分母、终局类型和基础设施无效任务。

### 面板 B：完整需求 Rubric

- 从每个 task 的私有完整目标生成一次要求候选；
- 使用冻结的本地 Qwen3.8-27B 整理 Rubric；
- 每个 task 只冻结一份，Base/SFT/RL 和三个条件共享；
- 按 hard/required/soft 报告 satisfied、violated、unknown；
- 保留 Reward–Rubric disagreement，不允许一方覆盖另一方。

Rubric LLM 只能筛选和描述代码候选，不能创造新的字段、值或隐藏要求。模型 revision、prompt、schema、
temperature、thinking 开关和产物哈希全部进入 manifest。

### 面板 C：轨迹质量 Judge

主 Judge 使用冻结的 `deepseek-v4-flash-0731`，逐轨迹分别评价：

1. search strategy；
2. candidate utilization；
3. evidence verification；
4. decision quality；
5. termination efficiency。

五维分别报告 0/1/2 分布和均值，不相加为总分。Judge 只能看到 Query、冻结 Rubric、Actor 实际看过
的 observation、Shopper 已公开回答、动作和白名单行为指标；不能看到 Reward、gold ASIN、私有
omitted facts、raw observation 或其他模型结果。

### 面板 D：澄清行为

沿用最小评测的确定性澄清指标，并由 Judge 补充问题是否具体、是否与缺口相关、回答后决策是否有可见
依据。`auditable_post_answer_action` 只表示回答后出现了可审计动作，不宣称一定存在因果使用。

### 面板 E：确定性行为与基础设施

沿用最小评测的 Guard、重复、步数、上下文和 API 指标。`infrastructure_invalid` 轨迹保留在固定分母，
但标记为 `not_judged`，不要求 LLM 猜测分数。

## 8. 完整评测的校准、成本与运行策略

在触碰 Final-200 Judge 结果前，先在 DEV 上人工复核约 50 条分层轨迹，检查：

- Rubric 是否遗漏或扩写要求；
- Judge schema 成功率；
- 是否引用不存在的 event ID；
- hard requirement 和主要错误类型的一致率；
- Reward–Rubric disagreement 是否有可解释证据。

prompt 和模型只能在 DEV 上校准。正式 Rubric、Judge 和抽样策略冻结后才能运行 Final。

Base、SFT 和一个 RL 的全量 Judge 是 1,800 条轨迹；增加一个预注册 RL 消融则为 2,400 条。按每条
12K–24K 输入、1K–2K 输出，并为重试预留 10%–20%。价格随服务变化，不在协议中写死金额；先运行
20 task × 3 condition × 1 model 的 pilot，从保存的 API `usage` 与当期单价外推。Rubric 使用本地
Qwen3.8-27B 时没有额外 API 调用费用，但需要记录 GPU 时间。当前简历版本不要求运行 Judge。

如果时间或预算有限，可以预先冻结 50–100 个分层 task 只做 Judge 诊断；这必须标成“诊断样本”，不能
冒充全量五面板结果。若启动后续独立评测，确定性核心指标仍对全部 Final-200 运行。

## 9. 完整评测产物

```text
shared/
  task_facts.jsonl
  rubric_candidates.jsonl
  rubrics.jsonl
  manifest.json

MODEL/CONDITION/
  trajectories.jsonl
  summary.json
  preprocessed.jsonl
  judges.jsonl
  evaluations.jsonl
  evaluation_summary.json
  run_manifest.json

comparison/
  deterministic-comparison.json
  paired-transitions.json
  five-panel-comparison.json
  final-report.md
```

原始轨迹和 Judge 请求保存在 `outputs/`，通常不提交 Git；仓库提交冻结配置、哈希、汇总、统计结果和
代表性 bad cases。API key、私有完整目标和 Judge 不需要的 gold 字段不得进入公开报告。

## 10. 发布门槛

### 当前 DEV-500 release

- [x] DEV-500 与正式 SFT、RL train/validation 零 task-ID 重叠；
- [x] Base、SFT-325、BPO v1 各完成 G+/G−/C+ 500 条轨迹；
- [x] Environment v2.1、Reward v4、Shopper 和推理参数保持同一合同；
- [x] 报告每格原始计数、固定分母、Done、Reward-valid 和澄清行为；
- [x] 明确标记为 frozen development benchmark，不声称未见 test-set 泛化；
- [ ] 可选增强：补齐 Base/SFT 的 G−→G+ task-level gains/losses 和置信区间。

### 后续独立完整评测

- [ ] CARL-BPO v2.1 完成，并在 DEV 上冻结唯一 checkpoint；
- [x] Final-200 已通过结果盲抽、固定 seed、父资产哈希和 manifest 冻结；
- [x] Final-200 与所有训练/开发集合零 task-ID 重叠；
- [ ] Base/SFT/选定 RL 各完成 Final-200 的 G+/G−/C+；
- [ ] Final 结果不用于重新选模型或调参；
- [ ] Rubric/Judge 已在 DEV 人工样本上校准并冻结；
- [ ] 每个 task 只有一个共享 Rubric；
- [ ] Judge 输入通过 blind guard；
- [ ] 全量或预注册诊断样本的 Judge 覆盖范围明确；
- [ ] 五面板分别报告，不输出综合总分；
- [ ] 统计检验、置信区间和 bad-case 复核完成。

## 11. 与旧评测文档的关系

- [evaluation.md](evaluation.md)、[evaluation-dataset.md](evaluation-dataset.md)、
  [evaluation-updates.md](evaluation-updates.md) 和
  [evaluation-dashboard.html](evaluation-dashboard.html) 描述原参考项目的 Reward v3 单轮评测；
- 旧 Reward v3 Final-200 可以作为额外单轮回归，但不能与当前 Reward v4 Final-200 合并分母或排名；
- 当前多轮项目的正式结论、运行边界和指标定义以本文、版本化 manifest 与代码契约为准。
