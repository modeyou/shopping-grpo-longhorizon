# 多轮购物 Agent 评测协议

当前多轮项目使用 **ShopSimulator Environment v2.1 + Reward v4**。本文只描述这套协议；
Reward v3、旧 Final-200 Clean 和旧单轮报告仅用于历史回归，不能与这里的结果合并排名。

## 先看全貌

评测不试图把所有现象压成一个总分，而是依次回答四个不同的问题：

| 层 | 组件 | 回答的问题 | 是否由 LLM 决定 |
|---|---|---|---|
| A | 确定性代码审计 | 最后买得是否正确、轨迹是否正常结束、执行是否可靠？ | 否 |
| B | Requirement Rubric | 这道题应检查哪些用户需求？ | 候选由代码产生，LLM 只整理和选择 |
| C | Trajectory Judge | 从实际动作和页面证据看，需求是否满足、过程是否合理？ | 是 |
| D | 配对与统计 | 澄清通道和训练阶段是否带来可靠变化？ | 否 |

其中 A 是当前必须报告的权威结果；B 和 C 是解释层，不能改写 A 的 Reward 结论。D 以固定任务和配对比较
避免把不同题目或不同条件混成一个不可解释的平均分。

## 1. 要证明什么

项目同时评估两件事：

1. RL 是否在 SFT 基础上提高了满足完整需求的最终购买能力？
2. 模型是否会在信息缺失时有效澄清，并在信息已经完整时避免多余提问？

评测的基本单位为：

```text
一个模型 × 一个 task × 一个条件 × 一次 temperature=0 rollout
```

同一模型在同一条件下的任务结果必须共享任务、Actor prompt、工具 schema、Action Guard、环境、商品数据、
Reward v4、Shopper 配置、最大提问次数、推理参数、上下文预算和汇总代码。任何一项变化都需要新的 run identity。

## 2. 数据集与实验边界

| 资产 | 任务数 | 作用 | 可以做什么 | 不能做什么 |
|---|---:|---|---|---|
| `evaluation-dev-v2` | 500 | 冻结开发基准 | 选 checkpoint、比较方案 | 宣称未见泛化 |
| `final-200-v1` | 200 | 与训练和 DEV 隔离的 Final | 对预先冻结模型做泛化审计 | 用结果继续选模型或调参 |
| 旧 Reward v3 Final | 200 | 历史单轮回归 | 检查旧链路 | 与 v4 多轮结果合并 |

Final-200 的 tasks、gap openings、complete openings 和 conditions 均由 manifest 固定并记录规范化哈希。
Windows checkout 的 CRLF 可能改变工作区文件哈希，因此审计必须使用仓库的规范化哈希逻辑，而非直接比较
`Get-FileHash` 的结果。

Final-200 已用于若干历史运行的回顾性审计，因此它仍是冻结资产，但不再是可用于继续挑选 checkpoint 的
未见盲测。若要证明新算法的未见泛化，必须在模型和指标冻结后另建 sealed test set。

## 3. 三个配对条件

| 简称 | Actor 看到的请求 | `ask_shopper` | 要回答的问题 |
|---|---|---|---|
| G+ | 缺少关键事实的 opening | 可用 | 缺信息时能否主动澄清并完成购买？ |
| G− | 与 G+ 完全相同的 opening | 不可用 | 澄清通道带来多少实际购买收益？ |
| C+ | 完整请求 | 可用 | 正常购物能力如何，是否仍会无意义提问？ |

G+ 和 C+ 是并列的真实部署场景；G− 是 G+ 的配对诊断对照。因此分别报告三列，**不**把三列等权平均成
“总体能力”。若必须在 DEV 上选一个模型，只可用 `(G+ strict success + C+ strict success) / 2` 作为
选择索引，并明确它不是最终总分。

例子：同一个任务缺少预算时，G+ 中模型若问到预算、利用答案完成购买，而 G− 中因不能问而失败，才构成
“澄清带来实际价值”的证据。G+ 中“问过一次”本身不算成功。

## 4. A 层：确定性代码审计（当前权威结果）

### 4.1 Reward、完成与可靠性是三件不同的事

- **Reward v4**：依据实际购买商品、选中规格和可验证的变体价格，判定 Gold、有效替代、部分满足、错误购买等；
- **Done**：轨迹是否正常结束；`done=true` 不等于买对；
- **执行可靠性**：Guard 拒绝、非法工具调用、重复动作/搜索、repeat loop、max steps、上下文截断及 Actor/
  Shopper/API 故障等。

最严格的最终购买指标是 `strict gold success`：Reward v4、轨迹正常完成、Reward 有效、购买成功、
`reward_type=gold_purchase` 且终止原因为 `gold_purchase` 必须同时成立。

同时报告：

- `purchase_success`：Gold 或全部 hard + required 满足的有效替代；
- Done rate、Reward-valid rate、平均 Reward、Reward type 和 termination reason 分布；
- infrastructure-invalid 数量和 task IDs；
- Guard、非法调用、重复、循环、超步、上下文/API 故障的次数或任务比例。

固定分母中的缺失轨迹、无效 Reward 和基础设施失败不得静默删除。Reward v4 不因调用 `ask_shopper` 直接加分；
澄清效果由 G+/G− 配对和下述行为指标判断。

### 4.2 当前最小可信报告

每个模型都应按 G+、G−、C+ 分开报告：

| 模型 | G+ strict | C+ strict | G− strict | G+−G− 迁移 | C+ 多余提问 | Done | Reward valid |
|---|---:|---:|---:|---|---:|---:|---:|
| 示例模型 | 345/500 | 361/500 | 264/500 | gains/losses/ties | 469/500 | 1490/1500 | 1486/1500 |

除最终购买外，还应报告：G+ 提问/未提问、grounded question、C+ 首次无信息提问、第二次无信息提问、重复问题、
问题上限，以及提问后未继续购物的比例。`grounded` 只说明 Shopper 的答案来自冻结事实，不说明该问题一定必要；
G−→G+ 的同题迁移才是澄清价值的配对证据。

最低交付使用原始计数和固定分母，例如 `345/500 (69.0%)`。推荐进一步给 success rate 的 task-bootstrap
95% CI、模型间 paired-bootstrap 差值 CI、G−↔G+ 的 McNemar、gains/losses/ties 与关键 bad-case task IDs。

## 5. B 层：Requirement Rubric（需求检查标准）

Rubric 是每道题预先冻结的“需求清单”，不是另一个 Reward，也不会给模型训练信号。它的流程为：

```text
任务 Query + Reward v4 constraint atoms
  → 代码提取候选全集（品类、品牌、型号、功能、规格、价格）
  → Curator 只引用 candidate_id 筛选、去重、原子化和标注 hard/soft
  → Schema 与候选/原文引用校验通过后直接冻结
  → 同一份 Rubric 供所有模型和条件共用
```

候选直接复用 Reward v4 已编译的 `constraint_atoms`，但只把它们当作可能包含冗余和错误的宽松全集。
字段、比较操作符和期望值由代码拥有；LLM 可以删除错误候选，但不能新造 candidate，也不能把 Gold 商品的
全部属性自动写成用户需求。组合 option 候选可以支持多条原子 Rubric；每项仍必须引用 Query 中由代码切分的
连续原文 anchor。Rubric 的标签是：

- `hard`：明确品类、预算上限、指定规格、否定要求等；
- `soft`：最好、优先、倾向、左右等偏好；
- `needs_review`：Curator 无法可靠判断强度；保留为诊断标记，但不要求额外审批文件。

这套标签**不同于 Reward v4** 的 `hard / required / soft`：`needs_review` 不是 `required`，不能把两套语义混写。

例子：对“必须支持热洗，预算不超过 220 元，最好白色”，Rubric 可包含“支持热洗（hard）”“价格不超过
220 元（hard）”“白色（soft）”。它让报告能说明失败是预算、功能、规格还是证据不足，而非只给一个终局标签。

候选的 `hardness_hint` 只是 Reward v4 初筛结果，Curator 必须按 Query 的实际措辞重新判断。代码会固定候选、
请求、响应和 Rubric 的 hash；未知 candidate/anchor、内容被篡改或 Schema 不合法会 fail fast。该流程与参考
项目一致，只使用一次 Flash Curator，不增加 Critic 或强制人工审批。Rubric 属于近似的诊断指标，因此正式报告
同时保留 Reward v4 的 strict success 和替代成功率，不能让 Rubric/Judge 覆盖确定性结果。

## 6. C 层：Trajectory Judge（轨迹评审）

Judge 是离线 LLM 评审器，不是环境裁判。它读取冻结 Rubric、Actor 实际可见的 observation 和动作，以及非
Reward 的确定性行为统计；它不接收 Reward、Gold ASIN 或 Reward 分项。每个结果必须通过 schema 校验：所有
Rubric 项恰好评一次、引用的 `event_id` 必须存在、禁止生成综合总分。

Judge 输出三类结果：

1. 每条 Rubric 的 `satisfied / violated / unknown / not_applicable`，附事件证据；
2. 五个互不相加的 0/1/2 分：搜索策略、候选利用、证据核验、决策质量、终止效率；
3. 澄清行为（effective / ineffective / unnecessary 等）和一个主错误、至多两个次要错误。

例如，模型恰好买到预算内且支持热洗的商品，Reward 可以成功；但若它没有打开详情或核验规格，Judge 仍可给
较低的“证据核验”分。反过来，Judge 认为需求满足但 Reward 判为错误时，结果记录为 `Reward–Rubric disagreement`
供人工复核；Judge 不能覆盖 Reward。

在 G+ 场景，完整 Rubric 可能包含 Actor 当时尚未获知的要求。它适合用于最终需求满足度审计，却会给“当时是否
应该提问”的过程判断带来事后信息风险。因此严格的后续版本应将“完整需求满足评审”和“仅基于 Actor 可见信息的
行为评审”拆成独立输入；在此之前，Judge 的澄清因果结论只能作为诊断，G+−G− 配对仍是主要证据。

Rubric/Judge 的脚本、schema 和 fail-fast 校验已经实现，但尚未将全量 Judge 产物作为既有实验结果报告。任何
未来运行都必须在 DEV 人工校准后冻结模型、prompt、schema、可见性边界和 manifest。

## 7. D 层：比较、发布与运行顺序

### DEV：用于选择，不用于泛化宣称

当前 release 的确定性比较可覆盖 Base、选定 SFT 和已有 RL run。选择规则为：

1. 拒绝 task 缺失、Reward 版本错误、模型名错误或未解释基础设施异常的 run；
2. 优先比较 G+ 与 C+ 的 strict success；
3. 两者互有胜负时才使用 DEV 选择索引；
4. 检查 G−→G+ 的绝对成功率与 gains/losses，防止 G− 退化被误读为澄清改进；
5. 再以 Reward-valid、Done、严重澄清错误和 Guard 作为 tie-breaker；
6. 记录选择理由，并冻结唯一 checkpoint。

RL 未超过 SFT 也是有效结论；不得通过改分母、只选某一条件或回看 Final 来包装提升。

### Final：用于审计冻结模型

冻结模型后，优先对 Final-200 的 G+ 条件运行 Base、SFT 与 RL，这是当前简历版本的最小可信正式评测。
Final 结果不得反过来选模型。G−/C+ 属于后续澄清机制专项评测，不阻塞当前项目交付。Rubric 只生成一次，
三个模型共享；随后分别运行 Judge。若预算允许，再补齐 G−/C+ 或抽样人工核查 Rubric/Judge 的噪声。

完整产物按以下层次保存：

```text
shared/                 # task facts、候选、冻结 Rubric 与 manifest
MODEL/CONDITION/        # trajectories、确定性汇总、Judge 结果与 run manifest
comparison/             # 条件迁移、模型比较、五面板报告
```

原始轨迹和 Judge 请求保存在 `outputs/`；公开报告只提交配置、哈希、汇总、统计和脱敏 bad cases，绝不提交
API key、私有完整目标或 Judge 不需要的 Gold 字段。

## 8. 状态清单

- [x] DEV-500、训练/验证集合与 Final-200 的 task ID 隔离；
- [x] Environment v2.1、Reward v4、条件映射和固定分母合同；
- [x] 确定性 Reward/终局、澄清和可靠性汇总代码；
- [x] Rubric/Judge 的候选约束、schema、可见性裁剪和 fail-fast 校验；
- [ ] 对计划使用的 Judge 做抽样人工核查（推荐但不阻塞 Final-200 G+）；
- [ ] 运行并发布全量 Rubric/Judge 五面板产物；
- [ ] 为未来新算法建立从未用于模型选择的 sealed test set。

## 9. 与旧文档的关系

[evaluation.md](evaluation.md)、[evaluation-dataset.md](evaluation-dataset.md)、[evaluation-updates.md](evaluation-updates.md)
和 `evaluation-dashboard.html` 描述旧 Reward v3 单轮评测。当前多轮项目以本文、版本化 manifest 和代码合同为准。
