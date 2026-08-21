# 多轮澄清项目：训练、Reward 与评测决策说明

本文集中整理多轮购物澄清项目在基座模型、SFT/GRPO 资源、Reward、开发集、正式评测、
Rubric/Judge 和在线 Shopper 方面的设计结论，作为后续实现与复习入口。本文不替代各模块的
详细协议；发生冲突时，以代码、版本化 manifest 和对应专题文档为准。

## 1. 状态说明

本文使用三种状态，避免把设计建议误认为已经具备的能力：

- **当前已实现**：仓库中已有代码和测试支持。
- **已确定方向**：项目采用该方向，但仍需完成实现或运行验证。
- **待确认提案**：尚未修改正式运行契约，实施前必须显式确认和版本化。

截至 2026-08-21：

| 项目 | 状态 | 说明 |
|---|---|---|
| 基座 Actor | 已确定方向 | `Qwen3.5-2B` |
| 当前环境 Reward | 当前已实现 | `shopsimulator-reward-v3` |
| Reward v4 | 当前已实现候选 | 与 v3 并行，含原子约束、价格语义和双算入口；尚未成为默认协议 |
| 教师/Baseline/Evaluation 在线 Shopper | 当前已实现 | rollout harness 管理独立 LLM Shopper |
| veRL GRPO 在线 Shopper | 待实现 | 当前 AgentLoop 尚未完成 Shopper 客户端与私有目标接入 |
| Rubric LLM | 已确定方向 | 本地 `Qwen3.8-27B`，需冻结权重和 prompt |
| 主 Judge | 已确定方向 | `deepseek-v4-flash-0731`，异常样本可升级到 Pro |
| 五面板评测 v2 | 当前已实现 | 含澄清面板、G+/G−/C+ 配对与版本化入口；尚未运行正式模型评测 |

仓库当前契约仍是 Environment v2.1 + Reward v3。Reward v4 必须作为新版本实现，不能静默
改变 v3，也不能把用不同 Reward 得到的结果混为同一条实验曲线。

## 2. 整体工作流与三方职责

项目主流程保持：

```text
数据采集 -> Base 里程碑 -> SFT -> GRPO -> 正式 Evaluation
```

多轮运行由三方组成：

| 角色 | 可见信息 | 责任 | 禁止事项 |
|---|---|---|---|
| Actor | 公开 opening、页面观察、已经公开的 Shopper 回答 | 判断是否需要澄清；搜索、核验、选规格并购买 | 不得看到完整私有目标和 gold 信息 |
| ShopSimulator | 商品状态、完整目标、规格、价格和 Reward | reset/step、页面反馈和终局判定；通过私有通道提供 Shopper context | 不调用 LLM，不直接撰写自然语言回答 |
| Shopper | 私有完整目标、允许回答的缺失事实和历史问答 | 仅回答用户拥有的偏好、预算、用途、兼容性等事实 | 不搜索商品，不替 Actor 做购买决策，不虚构偏好 |

SFT 是保存轨迹上的离线训练，本身不调用 Shopper。GRPO 和正式评测是在线 rollout；Actor
提出什么问题、后续执行什么动作都由当前策略实时决定。

## 3. 开发集、正式测试集与基座评测

### 3.1 开发集的作用

开发集用于反复调试和选择方案，包括：

- 验证模型服务、tool parser、Shopper、环境和 collector 闭环；
- 调整 context budget、`max_tokens`、重试、截断和推理参数；
- 选择 SFT/GRPO checkpoint 和超参数；
- 分析 bad case，修复 prompt、Reward、工具协议或代码；
- 校准 Rubric 和 Judge prompt；
- 做小规模消融和 GRPO 单步冒烟。

开发评测就是在开发集上按正式协议的缩小版或完整版本运行。它可以被反复查看，因此会逐渐
参与项目决策，不能再被当作无偏的最终成绩。

当前清洗后的开发集位于：

```text
data/multiturn/evaluation-dev-v1/
```

### 3.2 正式测试集的作用

正式测试集只在模型、prompt、Reward、Shopper、Judge 和 checkpoint 全部冻结后运行，用于报告
最终 Base/SFT/GRPO 结果。它不得用于：

- 调训练数据比例或超参数；
- 选 checkpoint；
- 根据逐题结果修 prompt 或代码；
- 校准 Shopper、Rubric 或 Judge。

当前清洗后的正式任务位于：

```text
data/multiturn/evaluation-v1/
```

正式测试与 SFT、GRPO、开发集必须保持 task-ID 零重叠。完整数据隔离和替补规则见
[多轮任务划分](multiturn-task-splits.md)。

### 3.3 为什么训练前必须做基座评测

基座评测回答“Qwen3.5-2B 在没有项目训练时已经会什么”。没有这个里程碑，就无法判断：

- SFT 是否提升了正确提问和工具使用；
- GRPO 是否提升了严格购买成功率；
- 改善来自训练，还是来自同时更换 prompt、Shopper 或 Reward；
- 模型是否只提高提问率，却没有提高最终购买结果。

默认先运行开发集 Base 评测。正式 Base rollout 可以提前封存，但一旦逐题正式结果被用于后续
调参，该测试集就不再盲，应重新冻结一套未被查看的正式集合。

## 4. 正式评测体系

正式评测不是一个不可解释的总分，而是五个互补面板。面板之间有联系，但回答的问题不同，
不应相加为单一分数。

### A. Environment Reward 与终局

回答“最后是否买对”：

- Reward type、validity 和数值；
- strict `gold_purchase`；
- 包含有效替代品的 purchase success；
- hard-gate、匹配分、证据覆盖和 termination reason；
- 基础设施无效任务。

### B. 完整需求 Rubric

回答“每条用户要求是否有证据满足”：

- hard/soft requirement 的 satisfied、violated、unknown；
- Reward 与 Rubric 的冲突；
- 品牌、型号、功能、规格、预算等要求级错误。

Rubric 在比较各 Actor 之前冻结共享。Judge 只能使用 Actor 实际看过的页面证据、动作、opening
和 Shopper 回答，不得看到 Reward、gold ASIN、私有 omitted facts 或其他 Actor 的结果。

### C. 轨迹质量

回答“过程是否可靠”：

1. 搜索策略；
2. 候选利用；
3. 证据核验；
4. 决策质量；
5. 终止效率。

五维分别报告，不计算简单平均总分。

### D. 澄清行为

回答“是否在正确的时候问了有效问题并使用答案”：

- gap 上的 ask/no-ask；
- complete request 上的 unnecessary ask；
- 提问是否命中预先冻结的缺口；
- Shopper 回答是否有私有事实溯源；
- 回答后的搜索、核验、规格选择是否发生可审计变化；
- G+ 相对 G- 的 task-level 成功迁移。

### E. 确定性行为与基础设施

回答“结果是否因运行异常而失真”：

- 商店动作数与 Actor 总回合数；
- 重复调用、Action Guard、repeat loop 和 max steps；
- observation/context 截断；
- Actor/Shopper 请求数、token、延迟和 API 错误；
- infrastructure-invalid task IDs。

### 4.1 三个配对条件

同一任务、同一 Actor 运行：

- **G+**：gap opening，允许 `ask_shopper`；
- **G-**：相同 gap opening，禁止 `ask_shopper`；
- **C+**：完整请求，但仍允许 `ask_shopper`。

G+ 与 G- 的成功率差和逐题迁移衡量澄清的实际收益；C+ 衡量过度提问。该方法比只统计
“模型问了多少次”更接近因果对照。

### 4.2 评测方法论与技术栈

这套体系组合了：

- 环境内确定性终局 Reward；
- 预先冻结的 requirement compiler 与 Rubric；
- 受证据约束的 LLM-as-a-Judge；
- G+/G-/C+ 配对和反事实式对照；
- task-ID 级 Base/SFT/GRPO 迁移分析；
- 轨迹错误分类和基础设施无效样本隔离；
- 数据、模型、prompt、tool schema、Shopper、Judge 和 Reward 的版本化 manifest；
- 开发集与正式测试集隔离。

详细冻结协议见[多轮正式评测协议](multiturn-evaluation.md)。

## 5. Reward v3：当前正式版本

Reward v3 是确定性的终局 Reward，不调用另一个 LLM 判断购买结果。当前主要规则为：

| 结果 | Reward |
|---|---:|
| `gold_purchase` | `1.00` |
| `valid_alternative_purchase` | `0.55` |
| `partial_alternative_purchase` | `min(0.25, -0.30 + 0.55 × S)` |
| `graceful_stop` | `-0.15` |
| `early_abstain` | `-0.35` |
| `max_steps` | `-0.50` |
| `repeat_loop` | `-0.65` |
| `wrong_purchase` | `-0.85` |
| `reward_unverifiable` | `0.00`，但 `reward_valid=false` |

硬门槛主要是类目和预算。品牌、型号、核心功能、关键选项分别使用 0.35、0.25、0.25、0.15
的固定权重。Reward v3 适合作为现有数据验收、GRPO 和评测的共同终局标准，但有以下局限：

- 价格解析词形覆盖不足；
- “230 元左右”被粗略处理为硬上限，不能表达接近程度；
- 固定维度权重未必对应用户实际强调程度；
- 某些无法解析的 option 或变体价格存在回退逻辑；
- 只评价终局，不能单独证明提问是否必要或答案是否被使用。

完整现行定义见 [Reward v3](reward-v3.md)。

## 6. Reward v4：已实现的候选版本

Reward v4 的目标是提高终局判定的可靠性和可诊断性，而不是把 LLM Judge 或 PRM 塞进环境。
实现细节和切换门槛见 [Reward v4](reward-v4.md)。完成 v3/v4 双算与 bad-case 抽查前，默认
运行契约仍是 v3。

### 6.1 约束原子化

把完整需求拆成独立约束原子，例如：

```text
类目=自动浇水器
材质=铜芯电磁阀
价格偏好=约230元
功能=雾化
功能=双路控制
```

建议的强度为：

- `hard`：必须、一定、硬性、不可缺少；
- `required`：普通明确要求；
- `soft`：最好、优先、尽量。

soft 权重可设为 0.5，其余原子权重为 1。匹配分改为：

```text
S = sum(weight_i * score_i) / sum(weight_i)
```

严格成功仍要求所有激活要求被证据覆盖且完全满足。类目、硬预算或明确 hard 约束失败产生
`wrong_purchase`；普通 required/soft 缺失进入部分匹配。

### 6.2 价格编译改进

V4 应支持：

- 预算、价格、售价、价钱、价位、总价；
- 元、块、千、k、万；
- 阿拉伯数字和中文数字；
- 以内、以下、不超过、最多、区间；
- 左右、上下、大概、约、出头、多点。

语义上区分：

- `不超过230`：硬上限；
- `200到250`：硬区间；
- `230左右`：软目标；
- `1900出头`：非对称软目标。

软目标可先使用 `max(2元, 目标价的10%)` 作为满分容差，超出后连续衰减。10% 是工程初值，
应在开发集上校准，而不是写死后直接宣称为真实用户普遍偏好。

### 6.3 规格与价格 fail-closed

- 实际购买的变体组合必须可以确定；
- 所选规格的最终价格必须能从环境证据解析；
- required option 轴无法映射时不能通过模糊回退获得严格成功；
- 关键证据不可核验时返回 `reward_unverifiable`，不得伪装成中性成功。

### 6.4 保留数值尺度

建议 V4 首版保留 v3 的终局 outcome 数值和 termination 规则，只改变需求编译、证据核验和匹配
方法。这样可以在同一批冻结轨迹上同时重算 v3/v4，定位变化来自哪里。

### 6.5 澄清诊断暂不直接加分

V4 可以输出以下诊断：

- `clarification_needed`；
- `question_grounded`；
- `question_targets_missing_constraint`；
- `answer_faithful`；
- `answer_used_after_clarification`；
- `post_clarification_progress`。

但首版不建议设置“提问一次 +0.1”。这种奖励容易诱导模型在完整请求上也无条件提问。GRPO
的终局成功已经能对有效澄清进行轨迹级信用传播；如果后续实验证明信号过稀疏，再把确定性
step shaping 或 PRM 作为 V4.1 消融实验。

### 6.6 Reward、评测指标和训练奖励的关系

- 环境 Reward：可在 GRPO 中直接优化，也用于正式评测的终局面板；
- 澄清/Rubric/轨迹面板：用于解释和诊断，不必全部合成训练标量；
- PRM/step-level reward：主要解决 GRPO 的长程信用分配，不是正式评测成立的前提；
- LLM Judge：不应直接替代确定性终局 Reward，否则会引入成本、方差和可攻击面。

## 7. 基座模型与 4×4090 资源核算

### 7.1 为什么选择 Qwen3.5-2B

项目需要同时具备：

- 中文指令和长上下文；
- 原生 tool calling；
- 能在 4×24GB RTX 4090 上完成 SFT 和在线 GRPO；
- 训练后仍可低成本部署和评测；
- 模型足够小，使多轮 rollout 和多组消融可实际完成。

Qwen3.5-2B 官方定位也包含原型开发和任务微调。它有约 2B 参数、24 层、hidden size 2048、
18 个 Gated DeltaNet 层和 6 个全注意力层，原生上下文上限 262,144。项目实际先限制在约
24,576 token，而不是直接使用理论上限。

官方模型卡：<https://huggingface.co/Qwen/Qwen3.5-2B>

### 7.2 参数与显存估算

| 项目 | 估算 |
|---|---:|
| BF16 基座权重 | 约 4.0 GB / 3.73 GiB |
| FP32 基座权重 | 约 8 GB |
| LoRA rank 16 可训练参数 | 约 1700 万，低于总参数 1% |
| LoRA 权重、梯度和 Adam 状态 | 约 0.3–0.5 GB |
| QLoRA 4-bit 基座和量化元数据 | 约 1.2–1.5 GB |
| 24,576 token × 248,320 词表的 BF16 全量 logits | 约 11.37 GiB |

标准完整微调若同时保存 BF16 参数、梯度、FP32 master weight 和 Adam m/v，仅模型状态就约
32 GB，尚未包含激活，因此单张 24GB 4090 不适合做完整参数微调。LoRA/QLoRA 才是本项目
合理路线。

### 7.3 SFT 建议

优先使用：

- BF16 LoRA；
- batch size 1；
- gradient checkpointing；
- SDPA；
- Liger fused loss；
- 先用 24,576 token 做 20-step 单卡显存冒烟。

大词表使完整 logits 占用约 11.37 GiB，Liger 很重要。若 BF16 LoRA 仍 OOM，再使用 QLoRA，
而不是一开始就牺牲全部训练精度或截断所有长轨迹。

四张 4090 合计 96GB，但不会自动成为一张 96GB 显卡。DDP SFT 会在每张卡复制模型，作用
主要是提高吞吐。当前单卡有效 batch 为 `1 × gradient_accumulation=8`；若改为四卡 DDP 并
保持等效 batch 8，应把 accumulation 改为 2。

### 7.4 GRPO 建议

当前 `configs/grpo.yaml` 的 `trainer.n_gpus_per_node=1`，不会自动使用四卡。正式 GRPO 需要先
验证：

```yaml
trainer:
  n_gpus_per_node: 4

actor_rollout_ref:
  rollout:
    tensor_model_parallel_size: 1
```

2B 模型不需要 TP=4。4090 没有 NVLink，TP=1 配合四卡 FSDP/rollout workers 通常比 TP=4
更合适。当前已有 actor/ref parameter offload 和 optimizer offload，但 24k response、reference
logprob、vLLM rollout 与反向传播仍需真实 1-update 冒烟验证。

Qwen3.5-2B 只有 6 个全注意力层。BF16 KV cache 约为：

```text
6 layers × 2(K/V) × 2 KV heads × 256 × 2 bytes
= 12 KiB/token
```

一条 24,576-token 轨迹约 288 MiB KV cache；8 条全长轨迹约 2.25 GiB，另需计算 DeltaNet
状态、vLLM 管理空间、模型权重和 CUDA 开销。

结论是：2B 在 4×4090 上做 LoRA SFT 和 GRPO 可行，但现有单卡配置必须调整，最终以单卡
SFT 冒烟和四卡 GRPO 1-update 冒烟的峰值为准。

## 8. Rubric 与 Judge 模型选择

### 8.1 Rubric：本地 Qwen3.8-27B

Rubric 负责把私有完整需求预编译成可判断的要求清单，不负责看轨迹后打分。使用本地
Qwen3.8-27B 的理由是：

- 27B 对中文约束拆分、规格和隐含偏好有足够能力；
- Rubric 是离线一次性生成，可缓存、重试和人工抽检；
- 本地生成边际 API 成本低；
- 与最终 DeepSeek Judge 保持模型家族独立。

必须冻结：

- 模型权重 SHA/revision；
- prompt 版本；
- thinking 开关、temperature 和 token 限制；
- JSON schema；
- 任务到 Rubric 的映射 hash。

Rubric 生成时只能看任务私有目标，不能看任何被评测 Actor 的输出。

### 8.2 Judge：DeepSeek-V4-Flash-0731

Flash 作为主 Judge 的理由是长上下文、结构化输出、吞吐和成本较适合批量轨迹判断。它可以
读取冻结 Rubric 和 Actor 可见证据，输出要求级判断、轨迹五维和错误类型。

官方说明：<https://help.aliyun.com/zh/model-studio/deepseek-v4-flash>

但长上下文不等于复杂判断必然可靠。建议使用级联：

- 主 Judge：`deepseek-v4-flash-0731`；
- schema 错误、证据 ID 不存在、逻辑冲突、Reward hard-gate 冲突时升级到 Pro；
- 随机抽取 5%–10% 结果交给 Pro 或人工复核。

Pro 官方说明：<https://help.aliyun.com/zh/model-studio/deepseek-v4-pro>

在开发集上先人工标注约 50 条轨迹，检查 schema 成功率、虚构证据、hard requirement 一致率和
主要错误类型一致率。Rubric/Judge prompt 只能在开发集校准，不能根据正式测试结果修改。

## 9. GRPO 在线 Shopper 与数据来源偏差

### 9.1 偏差来源

教师/SFT 轨迹中的 Shopper 主要由 Qwen3.8-27B 生成；如果 GRPO 在线改用 DeepSeek API，可能
产生 Shopper simulator shift。真正需要控制的不是模型名称，而是：

- 同一问题是否选择相同私有事实；
- 回答是否完整且不虚构；
- 无关问题是否被拒答；
- 是否泄漏未被询问的其他私有信息；
- Actor 是否利用某个 Shopper 的固定措辞或漏洞。

Actor 不需要模仿 Shopper，只需要理解回答后继续行动，因此纯措辞差异通常小于事实选择差异。
SFT 接触 Qwen、GRPO 接触 DeepSeek 也可以形成有限的回答风格增强，但前提是语义契约一致。

### 9.2 受约束的 Shopper Contract

建议 GRPO Shopper 输出私有审计结构：

```json
{
  "answer": "需要铜芯电磁阀，预算大约230元。",
  "used_fact_ids": ["material_0", "budget_0"],
  "answered_dimensions": ["材质", "预算"],
  "status": "answered"
}
```

要求：

- `used_fact_ids` 只能来自冻结 omitted facts；
- fact IDs 和 dimensions 只写私有日志，不发给 Actor；
- Actor 只收到自然语言 `answer`；
- 回答验证失败时重试，仍失败则标记基础设施错误并跳过该 GRPO 更新样本；
- 相同任务、问题、历史和 Shopper 版本可以安全缓存；
- Shopper prompt、模型版本和参数写入 rollout manifest。

### 9.3 Qwen 预生成 Answer Bank

为了减少模型切换偏差和训练时 GPU 冲突，可在关闭本地 Qwen3.8 服务之前，为每个任务预生成：

- 每个单独 omitted dimension 的回答；
- 常见非空维度组合的回答；
- 每个组合 2–3 个自然语言改写；
- 无关问题和无法回答时的标准回答。

在线 GRPO 时，DeepSeek 可以主要负责把 Actor 问题映射到 `fact_id` 子集，再从冻结的 Qwen
Answer Bank 中选择一个回答；未命中时才回退到 DeepSeek 在线生成并严格审计。这样可以：

- 保留 Qwen 教师数据的回答风格；
- 不在训练时运行占满四卡的 27B 服务；
- 减少 API token、延迟和幻觉；
- 通过多种改写防止 Actor 记忆固定模板。

若暂不实现 Answer Bank，也可以直接使用 DeepSeek，但必须先在同一批问题上做 Qwen/DeepSeek
事实级 A/B，验证 `used_fact_ids` 和回答完整性。

### 9.4 能提前生成与不能提前生成的内容

可以提前生成或冻结：

- opening、完整私有目标和 omitted facts；
- Rubric、fact IDs 和 Answer Bank；
- 商品环境和 Reward requirement；
- 对完全相同问题的确定性 Shopper 缓存；
- 仅用于里程碑比较的 Base rollout。

不能提前生成完整 GRPO Actor rollout。GRPO 是 on-policy：每次更新后 Actor 的提问、搜索和购买
动作都会变化，下一批轨迹必须由新策略在线生成。使用固定的提前轨迹反复训练将变成离线 RL、
DPO 或 RFT，而不是标准 GRPO。

### 9.5 正式比较时如何保证公平

Base、SFT、GRPO 在正式评测中必须面对完全相同的冻结 Shopper 合约、后端、Answer Bank、prompt
和参数。训练时可以接触 Qwen 与 DeepSeek 两种回答风格，但不能让三个 Actor 在正式比较时使用
不同 Shopper。

推荐正式协议为：

- SFT 数据：Qwen3.8 Shopper；
- GRPO：DeepSeek router/Shopper + 冻结 Qwen Answer Bank；
- Base/SFT/GRPO 正式评测：统一使用同一套 DeepSeek + Answer Bank Shopper；
- 小规模纯 DeepSeek 回答消融：检查 Answer Bank 是否改变结论。

## 10. GRPO 在线 Shopper 的提前实现计划

这部分不依赖 SFT 数据采集完成，可以先在开发集实现：

1. veRL AgentLoop 识别 `ask_shopper`；
2. 从环境私有通道读取 `shopper_context`，但不泄漏给 Actor；
3. 接入异步 DeepSeek Shopper 客户端、超时、重试和缓存；
4. 验证 `used_fact_ids` 与回答状态；
5. 每条轨迹最多两个 Shopper 问题；
6. Shopper 调用不占 35 个商店动作，但计入 Actor 回合、请求数和延迟；
7. 把自然语言回答追加为同一轨迹中的 tool message；
8. 保存私有审计日志，公开 Judge 输入前执行 blind guard；
9. 完成单元测试、单任务在线闭环和异常注入测试；
10. 使用 Qwen3.5-2B 做四卡、一个 task、一个 GRPO update 的资源冒烟。

只有这个闭环通过后，才适合启动正式多轮 GRPO。

## 11. 推荐执行顺序

1. 保持 Reward v3 为默认契约，先运行 Reward v3/v4 双算并评审 gained/lost bad cases；
2. 实现 GRPO 在线 Shopper Contract 和私有审计；
3. 生成或补齐 Qwen Answer Bank，并做 Qwen/DeepSeek 事实级 A/B；
4. 完成 Qwen3.5-2B 开发集 Base 评测；
5. 完成单卡 BF16 LoRA + Liger SFT 冒烟；
6. 完成四卡 GRPO 1-update 冒烟；
7. 在开发集校准 Rubric 和 DeepSeek Judge；
8. 冻结 V4 是否启用、Shopper、prompt、模型 hash 和全部评测 manifest；
9. 运行正式 Base/SFT/GRPO 配对评测。

相关详细文档：

- [多轮澄清设计](multiturn-clarification-design.md)
- [多轮教师数据与 SFT 复习](multiturn-teacher-sft-review.md)
- [多轮正式评测协议](multiturn-evaluation.md)
- [Reward v3](reward-v3.md)
- [Reward v4](reward-v4.md)
- [SFT](sft.md)
- [GRPO](grpo.md)
