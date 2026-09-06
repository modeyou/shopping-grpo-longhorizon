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

Rubric 是每道题、每个模型 rollout 前冻结的“需求检查标准”。它回答的是“用户到底提出了哪些可检查的
要求”，不是另一个 Reward，不参与训练，也不会根据某个模型的轨迹临时改写。其流程为：

```text
任务 Query + Reward v4 constraint atoms
  → 代码提取候选全集（品类、品牌、型号、功能、规格、价格）
  → Curator 只引用 candidate_id 筛选、去重、原子化和标注 hard/soft
  → Schema 与候选/原文引用校验通过后直接冻结
  → 同一份 Rubric 供所有模型和条件共用
```

Curator（生成 Rubric 的受约束 LLM）只读取当前 Query、代码切分的 Query 原文锚点和候选全集；它看不到任何
模型轨迹、最终购买、Reward 分数或 Gold 私有字段。Judge 则只能在 Rubric 冻结后，按这份标准评阅各模型轨迹。
也就是说，Curator 负责“先定检查题目”，Judge 负责“后看答卷”，二者不能互相代替。

### 5.1 候选由代码锁定，Curator 只做需求解释

候选来自 Reward v4 的结构化 `constraint_atoms`。每个候选由代码冻结：

| 字段 | 作用 | 示例 |
|---|---|---|
| `candidate_id` | 稳定引用 ID | `c0007` |
| `constraint_type` | 品类、品牌、功能、规格、价格等类别 | `core_function`、`option`、`price` |
| `field_path` + `operator` | 后续检查的字段与比较方式 | `purchase.price` + `lte` |
| `expected_value` | 期望值或阈值 | `220`、`支持热洗` |
| `hardness_hint` | Reward v4 的初筛提示 | `hard` / `soft` |
| `data_sources` | 该候选的结构化来源 | `instruction`、`query`、`derived.option_component` |

Curator 可以选择、删除、去重、拆分原子要求并结合 Query 重判 `hard/soft`，但不能新造 `candidate_id`、
修改字段/比较符/期望值，或仅因候选来自目标商品就把其属性写成需求。组合规格还会生成
`option_component` 候选：例如实际规格“荔枝果肉红茶 250g”可分别支持“含荔枝果肉”和“250g”，
但二者仍绑定同一个完整可购买规格。

这不是让代码替 LLM 判需求：代码负责把可检查的字段和值锁在 Reward v4 语义中，防止 LLM 凭空造规则；
Curator 仍须判断候选是否有 Query 原文的直接支持。若候选集没有合适项，Curator 不得猜造 Rubric；这是候选
覆盖边界，应修提取器或人工审计，不能由 Judge 事后补写。

### 5.2 一条冻结 Rubric 的结构

每条最终 Rubric 都保存以下可追溯字段：

| 字段 | 含义 |
|---|---|
| `rubric_id` | 本任务内稳定 ID；Judge 必须恰好评一次 |
| `description` | 面向人的原子需求重述，保留“必须/最好/左右”等强弱 |
| `candidate_ids` / `candidate_semantics` | 指向代码冻结的字段、比较符和值 |
| `hardness` / `hardness_source` | 当前 Query 下的 `hard`、`soft` 或 `needs_review` 及其来源 |
| `query_spans` / `query_anchor_ids` | Query 中逐字连续的原文证据，不能手写或改写 |
| `data_sources` / `selection_reason` | 候选来源及 Curator 选择该需求的理由 |
| `acceptance_criteria` | Judge 的共同边界：只按 Actor 可见证据判断 |

例如 Query 为“必须支持热洗，预算不超过 220 元，最好白色”，应冻结为三个可单独满足/违反的原子项：

```text
r0001  支持热洗              hard  → product.attributes contains "支持热洗"
r0002  价格不超过 220 元      hard  → purchase.price lte 220
r0003  最好白色              soft  → product.attributes contains "白色"
```

品类、颜色、材质、用途、兼容性和功能都是独立属性，不因写在同一句就合并；相反，本身不可拆分的固定概念
不会被强行拆散。

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

### 5.3 强弱与逐项判定状态

`hard/soft` 只描述用户是否允许退让，**不**描述是否容易验证；“安全性一定要好”等没有数值阈值的强制要求
仍是 `hard`。这套标签不同于 Reward v4 的 `hard / required / soft`：`needs_review` 不是 Reward v4 的
`required`，两套语义不能混写。

Judge 对每条冻结项只能输出以下状态：

| 状态 | 含义 | 不能误读为 |
|---|---|---|
| `satisfied` | 可见轨迹中存在直接支持该要求已满足的证据 | 整个任务成功 |
| `violated` | 可见轨迹中存在直接冲突证据，例如买了超预算规格 | 其他所有要求也失败 |
| `unknown` | 没有足够的 Actor 可见证据 | 失败，或可由常识/Gold 信息补成结论 |
| `not_applicable` | 在该可见终局上确实没有适用对象；应极少使用 | 掩盖没有核验的证据缺口 |

正式报告的“Rubric 满足率”固定为 `satisfied / 所有有效 Judge 已评价的 Rubric 项`；未知项不会被当作
成功，soft 项也不会加权抵消 hard 项。因此报告同时单列全部、hard、soft 三个比例。

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

### 6.1 Judge 的可见输入边界

Judge 接收 Query、冻结 Rubric、Actor 可见的逐事件轨迹（工具动作、页面 observation、Actor 文本、
真实 `event_id`）、`done/over` 终止状态，以及不包含结局奖励的过程统计（动作数、重复、Guard 拒绝、
上下文压缩）。它**不接收** Reward 总分或分项、strict/alternative 成功标签、Gold ASIN、私有完整目标、
audit raw observation，不能从未展示给 Actor 的候选中反推答案。

因此，Judge 可以评价“是否打开详情核验了价格”，却不能因为最终买到 Gold 就倒推过程合理；反过来，Judge
认为过程合理也不能推翻 Reward v4 对最终购买的确定性结论。

`judge_status=valid` 时，所有 Rubric 必须恰好评价一次，五个维度必须齐全，且所有 `evidence_event_ids`
必须真实存在。若调用或输入不可信，结果只能是 `invalid` / `not_judged`，不得保留伪造的逐项状态或维度分。
此外，输出被禁止包含 `total_score`、`overall_score` 或其他综合分。

### 6.2 五维 0/1/2 量表

五个维度是独立的粗粒度序数分：`0=明显不足或造成失败`，`1=部分做到但有关键缺口`，
`2=有针对性且有证据支持的良好执行`。它们不加权、不相加，报告只并列比较每个维度的均值和分布。

| 维度 | Judge 检查什么 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|---|
| `search_strategy` | 是否覆盖品类和关键约束，改写是否有效，是否机械重复 | 漏掉核心方向或反复无效搜索 | 找到部分方向但覆盖/改写不完整 | 有针对性覆盖关键条件，并根据页面反馈收敛 |
| `candidate_utilization` | 是否利用可见高匹配候选，比较是否必要 | 忽视明显候选，或无关乱比 | 使用候选但取舍依据不足 | 对少量相关候选做必要比较并合理利用 |
| `evidence_verification` | 购买前是否核验关键属性、规格、最终价格 | 未核验关键条件就断言/购买 | 只核验部分关键字段 | 关键属性、规格和价格都有可见核验 |
| `decision_quality` | 最终商品、规格及购买/放弃是否基于可见证据合理 | 明显错误、无支持或选错规格 | 选择可接受但有重要疑点 | 选择/放弃与已核验事实及用户优先级一致 |
| `termination_efficiency` | 是否过早购买/放弃、无效探索或耗尽步骤 | 过早终止、循环、非法动作或超步 | 能结束但有明显可避免步骤 | 足够核验后及时结束，无明显无效动作 |

### 6.3 澄清与错误归因

澄清不是天然加分：`effective` 指缺失信息确实影响选择、问题具体且回答后被用于搜索/核验/决策；
`ineffective` 指问了但没有利用回答；`unnecessary` 指公开请求已足够仍提问；`not_applicable` 指当前条件
不要求评价澄清；`unknown` 指可见证据不足。G+ 场景完整 Rubric 可能含有 Actor 提问前未知的事实，所以
Judge 的澄清结论只是过程诊断；澄清的因果收益仍以 G+−G− 同题配对为主要证据。

Judge 还会从冻结错误 taxonomy 中选择一个主错误、至多两个不重复的次要错误，例如
`premature_purchase`、`critical_evidence_missing`、`illegal_action`、`wrong_option`、`repeat_loop`。
它们是 bad-case 检索标签，不是额外的总分或对 Reward 的替代。

例如，模型恰好买到预算内且支持热洗的商品，Reward 可以成功；但若它没有打开详情或核验规格，Judge 仍可给
较低的“证据核验”分。反过来，Judge 认为需求满足但 Reward 判为错误时，结果记录为 `Reward–Rubric disagreement`
供人工复核；Judge 不能覆盖 Reward。

在 G+ 场景，完整 Rubric 可能包含 Actor 当时尚未获知的要求。它适合用于最终需求满足度审计，却会给“当时是否
应该提问”的过程判断带来事后信息风险。当前明确保持**一次单 Judge 调用**，不拆成两个评审器，以控制复杂度和
API 成本；代价是五维过程分与澄清结论只能作为诊断，G+−G− 同题配对仍是澄清因果收益的主要证据。

### 6.4 相同输入复用与确定性事实（2026-09-06）

新版 Judge 请求在送入模型前移除随机 `trajectory_id` 和 `tool_call_id`，再按 Judge 模型、prompt 版本、
解码参数和完整语义消息计算 SHA-256。所有模型和条件共用一个内容寻址缓存；相同 hash 只允许一次外部 API
调用，缓存结果在组装评测记录时绑定回各自真实轨迹 ID。缓存条目保存完整非密钥请求身份和已校验响应，损坏、
身份不匹配或超时锁都会 fail fast。

同一个 Judge 还接收代码生成的 `deterministic_facts`：

- `final_purchase`：最终成交 ASIN、商品名、品类、价格、属性和最终选项；
- `price_checks`：按冻结 Rubric 的 hard max/min、range、soft target 与最终价格得到的 pass/fail；
- `option_checks`：按冻结 Rubric 的规格轴、精确值或组合 component 与最终选项得到的 pass/fail；
- 价格或选项 Rubric 若与代码结果冲突，输出不能进入正式结果，而是触发 schema 修复并最终 fail fast。

最终订单没有记录某个规格轴时，对应 `option_checks` 为 `unavailable`。这不等于规格已满足：中途执行过
`select_option` 只能证明点击意图，不能证明该值最终进入订单，因此 r3 合同强制 Judge 将该 Rubric 标为
`unknown`。

这不会把 Reward、Gold ASIN 或成功标签泄漏给 Judge。代码只拥有可确定计算的最终状态和算术，LLM 继续负责
需求语义、过程质量和错误归因。直接调用 `evaluate_multiturn_panels.py` 时必须传入同一
`--judge-cache-dir`；使用 `run_multiturn_panel_grid.py` 时自动使用输出根目录下的
`semantic-judge-cache/`，因此任意数量模型都会共享缓存。

Rubric/Judge 的脚本、schema 和 fail-fast 校验已经实现；Final-200 G+ 已完成 Base、SFT-325 与
Root/Local RL step-200 的共享 Rubric 和全量 Judge 产物。完整结果见
[Final-200 G+ 正式评测结果](final200-evaluation-results.md)。任何未来运行都必须重新冻结模型、prompt、
schema、可见性边界和 manifest；改变其中任一项都不能与当前 Judge 均值直接混表。

2026-09-06 的 post-hoc 审计发现旧版 Judge 尚不满足模型比较的发布要求：SFT/RL 有 118 对实质相同输入，
但五维评分、Rubric 状态、澄清状态和错误标签存在高比例不一致；30 条分层人工样本还发现价格算术、最终
option 读取和组合规格语义错误。跨模型内容 hash 复用、最终状态和价格事实代码化现已实现。随后建立的
30 条结构化校准集完成 9 条人工确认，并在 r3 prompt 上通过 119/119 条断言，`release_ready=true`；
r3 已完成 Final-200 全量重跑，旧版输出只保留为 bad-case 候选。详见
[Judge Bad-case 与一致性审计](final200-judge-badcase-audit.md)。

全量重跑已于 2026-09-06 完成：在原 200 条共享 Rubric 上应用 6 条人工确认修订，并对运行中发现的
task 7704 组合 option component 做 1 条最小人工语义修订，共有 1,637 条 Rubric（hard 1,445、soft 192）。
Base、SFT、RL 共生成 600 条评测记录；SFT/RL Judge 覆盖 200/200，Base 因 7 条基础设施无效轨迹覆盖
193/200。内容寻址缓存最终包含 476 个有效唯一请求；相同语义输入跨模型直接复用。

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
- [x] Final-200 G+：Base、SFT-325、Root/Local RL step-200 的共享 Rubric 与全量 Judge；
- [x] 对 Judge 做首轮 30 条分层人工核查与相同输入一致性审计；
- [x] 实现单 Judge 的跨模型相同输入复用，以及最终状态与价格的确定性事实校验；
- [x] 固化 30 条结构化 calibration review set，并验证它能捕获旧 Judge 的 15 条已知错误；
- [x] 完成其中 9 条语义样本的人工签字，并用新版单 Judge 跑到 `release_ready=true`；
- [x] 使用校准后的 r3 Judge 全量重跑 Final-200 G+，并替换旧 Judge 比较表；
- [ ] 在新的 sealed 集补齐 G−/C+ 与全条件五面板产物；
- [ ] 为未来新算法建立从未用于模型选择的 sealed test set。

## 9. 与旧文档的关系

[evaluation.md](evaluation.md)、[evaluation-dataset.md](evaluation-dataset.md)、[evaluation-updates.md](evaluation-updates.md)
和 `evaluation-dashboard.html` 描述旧 Reward v3 单轮评测。当前多轮项目以本文、版本化 manifest 和代码合同为准。
