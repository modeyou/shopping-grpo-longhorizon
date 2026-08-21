# 多轮澄清正式评测协议

本文定义以 `Qwen3.5-2B` 为基座的多轮澄清项目如何做开发评测、训练前基座评测和最终
Base/SFT/GRPO 配对比较。协议继承参考项目的 Reward v3 与四面板轨迹评测，并增加独立的
澄清面板，但不会把原来
的单轮 `Final-200 Clean` 结果误当作多轮澄清能力。

## 1. 结论与边界

- 基座模型固定为 `Qwen3.5-2B`。Qwen3.8-27B 当前只承担 Teacher、opening generator
  和 LLM Shopper，不是本轮待训练 Actor。
- 必须在多轮 SFT 和 GRPO 之前完成基座评测；否则无法判断训练是否真正改善了澄清与购物。
- 原项目 `data/evaluation/tasks.jsonl` 的 Final-200 Clean 继续作为单轮外部可比基准。
- 新项目内 `data/multiturn/tasks/evaluation.jsonl` 是多轮候选池，目前只完成 task-ID
  隔离，还不能直接称为正式测试集。
- Reward v3 保持不变，只评价最终环境结果；澄清行为作为独立面板报告，不写回或覆盖
  Reward v3。
- Reward v4 已作为并行候选实现；在完成 v3/v4 gold replay、bad-case 抽查和任务重新冻结前，
  本协议仍以 v3 为当前正式口径，不能混合两版结果。
- Rubric 整理器固定为本地 `Qwen3.8-27B`；主 Judge 固定为
  `deepseek-v4-flash-0731`。二者都必须使用冻结 schema、prompt 和模型 revision。

## 2. 继承参考项目的哪些设计

参考项目的正式评测把结果分开报告，不计算一个不可解释的综合总分：

1. Environment Reward v3 与终局；
2. 逐任务需求 Rubric；
3. 完整轨迹质量；
4. 确定性行为、合法性、上下文与基础设施。

多轮项目完整继承以下原则：

- 固定任务分母，缺失和基础设施错误不能从分母中消失；
- Base、SFT、GRPO 使用相同任务、Actor prompt、工具协议、上下文和推理参数；
- 每个 Actor、每个条件、每个 task 只做一次确定性 rollout，不把多次尝试包装成
  Pass@1；
- Rubric 在模型之间冻结共享；
- Judge 不能看到 Reward、gold ASIN、Actor 未见的 raw observation 或其他 Actor 的结果；
- Reward 与 Rubric 冲突时并列保留，不允许一方覆盖另一方；
- 模型比较按 task ID 做配对迁移分析，而不只比较两个总体百分比。

## 3. Reward v3 在多轮项目中的职责

Reward v3 使用环境私有的完整目标评价最终购买结果：

- `gold_purchase=1.0`：购买目标 ASIN，并满足可验证要求；
- `valid_alternative_purchase=0.55`：购买不同 ASIN，但满足全部激活要求；
- 其余结果区分部分替代、错误购买、合理放弃、过早放弃、循环、步数耗尽和不可验证。

这正适合多轮任务：Actor 只看到不完整 opening，但终局仍按完整用户目标判定。Reward v3
不需要知道 Actor 是否提问，也不应因为出现 `ask_shopper` 就加分。否则完整请求上会产生
无条件提问的捷径。

需要保留的局限是：Reward v3 只能说明最终结果，不能单独回答以下问题：

- 问题是否必要、具体且与 opening 缺口相关；
- Shopper 回答是否真正补充了缺失事实；
- Actor 是否在后续搜索和决策中使用了答案；
- 如果禁止提问，同一 Actor 是否会失败。

这些问题由独立的澄清面板和 ask-enabled/ask-disabled 配对评测回答。

## 4. 正式测试集进入条件

当前新项目任务池已经在任何 LLM 调用之前确定性冻结：

```text
evaluation:       500
sft_candidates: 3,000
grpo_validation:  500
grpo_train:     5,000
```

但 task-ID 隔离不是完整的 benchmark 清洗。正式评测之前，必须对 500 个 evaluation
候选逐题执行与 Final-200 Clean 相同等级的静态或 gold replay 审计：

- gold ASIN 和正确规格真实可购买；
- required option 唯一映射到环境规格轴；
- 最终 variant price 可解析；
- Query 中出现预算时，Reward requirement compiler 实际解析出对应价格约束；
- 按 gold 规格购买能得到 Reward v3 `gold_purchase`、`reward_valid=true`；
- source goal、商品数据、环境版本和 Reward 版本 hash 一致；
- 任务与所有 SFT、GRPO、历史 benchmark 和开发评测 task ID 零重叠。

不合格任务只能按冻结顺序从 `reserve` 池确定性补入，并留下 replacement manifest；不能根据
Qwen3.5-2B、SFT 或 GRPO 的表现挑题。最终分母保持 500。

当前 `evaluation-v1` 确定性审计已经得到：

| 项目 | 数量 |
|---|---:|
| 原始候选 | 500 |
| Reward v3 gold 可达 | 306 |
| 被拒绝并替换 | 194 |
| 为取得 194 个合格补题而扫描的 frozen reserve 候选 | 307 |
| 清洗后固定分母 | 500 |

原始候选的主要拒绝原因是 `explicit_price_not_compiled=150`，其次包括 gold 购买不
可达、variant price 不可解和 option requirement 不可解。该结果说明“随机冻结且 task-ID
隔离”不足以构成可解释的 Reward benchmark。正式清洗产物位于：

```text
data/multiturn/evaluation-v1/tasks.jsonl
data/multiturn/evaluation-v1/reward_audit.jsonl
data/multiturn/evaluation-v1/metadata.json
```

随后为每道题生成一次 gap opening：

- 使用冻结的 Qwen3.8-27B、opening prompt 和确定性参数；
- 必须有非空 `omitted_dimensions`、逐字可溯源的 `omitted_facts` 和 `source_goal_hash`；
- opening 不得泄漏或改写被省略事实；
- 生成失败不能静默缩小分母，应按同一确定性 replacement 规则处理；
- opening 生成后冻结，Base、SFT、GRPO 和所有评测条件共享。

## 5. 开发评测与正式评测分开

### 5.1 开发评测

使用 `data/multiturn/tasks/grpo_validation.jsonl` 及其独立 frozen openings，用于：

- 验证 Qwen3.5-2B 服务、tool parser、Shopper 和 collector 闭环；
- 决定 `max_tokens`、context budget 等运行参数；
- 检查多轮 SFT checkpoint 和 GRPO checkpoint；
- 做 bad-case 分析和 prompt/代码修复。

它可以被反复查看，因此不能作为最终无偏测试结果。

开发集也已经按相同 Reward 可达性规则清洗为固定 500 题，其中原始候选 310 题保留、
190 题从排除正式测试集后的 frozen reserve 补入：

```text
data/multiturn/evaluation-dev-v1/tasks.jsonl
data/multiturn/evaluation-dev-v1/reward_audit.jsonl
data/multiturn/evaluation-dev-v1/metadata.json
```

### 5.2 正式评测

使用清洗并冻结后的 500 题 evaluation benchmark。正式任务不得用于：

- 数据配比、prompt、Reward 或超参数选择；
- SFT/GRPO checkpoint 选择；
- 针对 task 的 bad-case 修复；
- Shopper 或 Judge prompt 校准。

可以在训练前生成并封存 Base rollout，但如果训练过程会依据 Base 的逐题正式结果调整，
该集合就不再是盲测。项目默认先完成开发集基座评测；正式 Base/SFT/GRPO 在协议和 checkpoint
冻结后统一运行。

## 6. 三个正式条件

同一个 task 使用相同完整目标和相同 frozen opening，分别运行：

### G+：Gap / ask-enabled（主条件）

- Actor 只看到不完整 opening；
- 工具中包含 `ask_shopper`；
- 独立 LLM Shopper 只依据私有完整目标回答；
- 衡量真实多轮购物成功率和自主澄清行为。

### G−：Gap / ask-disabled（因果对照）

- Actor 看到与 G+ 完全相同的 opening；
- 隐藏 `ask_shopper`，其余工具与参数相同；
- 衡量在无法补齐信息时的购物结果。

G+ 与 G− 的 task 级成功迁移，是“澄清带来实际收益”的主要证据。它比仅统计提问率更重要。

### C+：Complete / ask-enabled（过度提问对照）

- Actor 看到完整请求；
- 仍提供 `ask_shopper`；
- 衡量模型是否在信息已经充分时无条件提问。

三种条件不是同一条件下的重复采样，不报告 Pass@k。每个条件均保持一次确定性 rollout，
并分别报告结果。

## 7. 五个互不覆盖的结果面板

### A. Reward 与终局

- Reward v3 type、validity、final reward；
- strict gold success；
- purchase success（包括 valid alternative）；
- hard gates、weighted score 和 termination reason；
- 固定分母下的缺失与基础设施无效任务。

### B. 完整需求 Rubric

- hard/soft requirement 的 satisfied、violated、unknown；
- Reward–Rubric disagreement；
- Rubric 从私有完整 Query 生成，但在 Actor 之间冻结；
- Judge 只能用 Actor 实际看过的 opening、Shopper 回答、页面证据和动作作判断。

### C. 轨迹质量

保留原项目五维：搜索策略、候选利用、证据核验、决策质量和终止效率。不要把五维相加成
总分。Judge 输入必须包含 Actor 可见的 Shopper tool response，但不得包含
`opening_audit`、`omitted_facts`、`used_facts`、Reward 或 gold 信息。

### D. 澄清行为

- ask task rate 与每题问题数分布；
- grounded ask：回答的 `used_facts` 非空且是 frozen `omitted_facts` 的子集；
- gap 上的 no-ask rate；
- complete 上的 unnecessary-ask rate；
- 首次提问位置、提问是否发生在购买前；
- Shopper 调用数、空回答、截断、API 失败；
- G+ 相对 G− 的 task 级 success gain/loss；
- 回答后搜索词、候选选择或规格判断是否出现可审计变化。

`grounded` 只证明问答命中了预先冻结的信息缺口，不等价于证明该问题在因果上必要；因果收益
由 G+/G− 配对结果负责。

### E. 确定性行为与基础设施

- 环境购物步数与总 Actor 回合数分开统计；
- Action Guard、重复调用、repeat loop 和 max steps；
- observation/context 截断；
- Actor 与 Shopper 的请求数、token 和延迟；
- infrastructure-invalid task IDs。

`ask_shopper` 不改变商店页面，也不是 ShopSimulator 环境动作。正式协议允许最多 35 个商店
动作和 2 个 Shopper 问题；问题不挤占 35 个环境动作，但问题数、Actor 回合数和延迟必须单独
报告，避免把提问当作零成本。

## 8. Base/SFT/GRPO 比较

基座为 `Qwen3.5-2B`。每个模型都运行 G+、G−、C+，并做两层配对：

1. 同一条件内：Base ↔ SFT ↔ GRPO；
2. 同一模型内：G+ ↔ G−，以及 G+ 的 gap ask rate ↔ C+ 的 unnecessary ask rate。

主要指标按优先级为：

1. G+ strict gold / purchase success；
2. G+ 相对 G− 的成功率增益和 task 迁移；
3. gap grounded-ask rate；
4. complete unnecessary-ask rate；
5. Reward/Rubric、轨迹五维和失败类型；
6. 商店步数、总回合数、Shopper 调用和延迟。

项目不声明单一“总分”。一个合理的澄清 Agent 应同时提高 G+ 购物成功、保持正的 G+−G−
增益，并控制 C+ 上的不必要提问。

## 9. 训练前基座里程碑

在启动多轮 SFT 前必须完成：

1. 多轮 evaluation pool 的 Reward-reachability 清洗脚本与 replacement manifest；
2. 开发集和正式集 opening 冻结工具；
3. collector 的 G+ / G− / C+ 模式；
4. 澄清行为确定性汇总；
5. Qwen3.5-2B 开发集 smoke；
6. Qwen3.5-2B 完整开发基座评测；
7. 协议、模型 hash、prompt、tool schema、Shopper 和推理参数 manifest。

完成这些项目后，才进入多轮 SFT。最终 benchmark 只在 checkpoint 和全部协议冻结后运行。

当前第 1、3、4 项及其 CPU 契约测试已经完成；第 2 项的工具链已具备，但正式 opening
尚未生成。后续正式入口只允许读取清洗后的
`data/multiturn/evaluation-v1/tasks.jsonl`，不得退回原始 500 候选。

opening 冻结分为两步：`generate_multiturn_tasks.py` 使用指定 Shopper 模型生成并审计
gap opening；`freeze_multiturn_openings.py` 从同一私有目标确定性提取 complete opening，
验证 `source_goal_hash`，并生成 G+/G−/C+ condition manifest。G+ 与 G− 只引用同一份
gap opening，不重复生成。

## 10. 五面板评测 v2 实现

多轮评测不再只停留在 rollout `summary.json`。当前代码已经把原单轮四面板内核升级为
`shopping-trajectory-evaluation-v2`，每条轨迹明确拼装五个面板：

1. `reward_and_terminal`：Reward v3 与终局；
2. `requirement_rubric`：逐要求判断和 Reward–Rubric disagreement；
3. `trajectory_quality`：五维 Judge 和错误类型；
4. `clarification`：代码指标与 Judge 澄清判断；
5. `deterministic`：动作、合法性、重复、上下文和基础设施。

澄清面板会确定性记录：

- Shopper 问题数和 grounded 数；
- gap no-ask 与 complete unnecessary-ask；
- 首次问题位置和购买前提问；
- 回答后搜索/候选/规格/购买动作；
- Actor/Shopper 调用数；
- G− 到 G+ 的 strict/purchase success task 迁移；
- 每个 Actor 在 C+ 上的不必要提问 task IDs。

`auditable_post_answer_action` 只表示回答后存在可见动作，不宣称因果上一定使用了答案。真正的
因果收益仍由同 task 的 G+/G− 成功迁移负责。

### 10.1 冻结共享 Rubric

Rubric 只为每个 task 生成一次，Base/SFT/GRPO 和三个条件共享：

```bash
export PYTHONPATH=./src

python scripts/freeze_multiturn_rubrics.py \
  --tasks data/multiturn/evaluation-dev-v1/tasks.jsonl \
  --output-dir outputs/evaluation/multiturn/shared-dev-v1 \
  --model qwen3.8-27b \
  --base-url http://127.0.0.1:8001/v1 \
  --api-key local-qwen
```

入口支持 `--resume`，并保存 TaskFacts、代码候选、Qwen 原始选择、最终 Rubric、请求元数据和
输入/产物 SHA-256。API key 不写入 manifest。

### 10.2 对一个 Actor/条件运行五面板

先由 `scripts/evaluate_multiturn.sh` 或同一 collector 生成原始轨迹，再执行：

```bash
export PYTHONPATH=./src
export OPENAI_BASE_URL="你的 DeepSeek API base URL"
export OPENAI_API_KEY="你的 API key"

python scripts/evaluate_multiturn_panels.py \
  --expected-tasks data/multiturn/evaluation-dev-v1/tasks.jsonl \
  --trajectories outputs/evaluation/multiturn/base-dev/gap-ask-enabled/trajectories.jsonl \
  --rubrics outputs/evaluation/multiturn/shared-dev-v1/rubrics.jsonl \
  --output-dir outputs/evaluation/multiturn/base-dev/gap-ask-enabled \
  --actor-label base \
  --condition gap-ask-enabled \
  --judge-model deepseek-v4-flash-0731
```

该入口会：

- 规范化事件并分离商店步数与 Shopper 问题；
- 先计算确定性指标；
- infrastructure-invalid 轨迹保留在固定分母，但不伪造 Judge 分数；
- 为有效轨迹调用 Judge，严格验证 rubric/event IDs 和五面板 schema；
- 保存可恢复的 `judges.jsonl`、`preprocessed.jsonl`、`evaluations.jsonl`、汇总和无密钥 manifest。

### 10.3 生成 Base/SFT/GRPO × G+/G−/C+ 配对结果

每个 Actor 根目录都必须包含三个条件的 `evaluations.jsonl`：

```bash
python scripts/compare_multiturn_evaluations.py \
  --expected-tasks data/multiturn/evaluation-dev-v1/tasks.jsonl \
  --run base=outputs/evaluation/multiturn/base-dev \
  --run sft=outputs/evaluation/multiturn/sft-dev \
  --run grpo=outputs/evaluation/multiturn/grpo-dev \
  --output outputs/evaluation/multiturn/dev-comparison.json
```

比较器会拒绝 task、condition 或 interaction mode 错配，分别输出：

- 同条件内 Base → SFT → GRPO 的配对迁移；
- 每个 Actor 内 G− → G+ 的成功获得/损失 task IDs；
- C+ 不必要提问；
- Rubric、轨迹五维、步数、Guard、重复与澄清变化；
- `composite_score=null`，明确禁止合成总分。

正式测试只能在 opening、Shopper 合约、Rubric、Judge、模型 checkpoint 和全部 manifest
冻结后运行。当前新增的是代码入口和 CPU 契约，尚未调用 Qwen/DeepSeek，也未产生正式成绩。
