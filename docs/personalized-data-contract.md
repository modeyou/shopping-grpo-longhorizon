# 个性化主动澄清数据契约

- 日期：2026-08-19
- 状态：待确认后实施
- 适用范围：自生成任务、模拟用户、教师轨迹、一次混合 SFT
- 不包含：Reward 公式、正式评测指标和 GRPO 超参数

## 1. 数据设计目标

正式数据必须让模型学会三个不同问题：

1. 当前请求已经明确了什么；
2. 用户画像能作为哪些长期偏好的先验；
3. 哪些关键本轮事实仍然未知，必须向用户提问。

这三类信息必须在数据层显式分开，不能依赖模型或评测器从自然语言中猜测其来源。

## 2. 三个信息视图

### 2.1 Agent 可见视图

在任务开始时，Agent 只能看到精简后的静态用户画像、当前购物请求、ShopSimulator 当前
observation、购物工具和最多两次 `ask_user` 调用机会。Agent 不得看到目标 ASIN、完整私有目标、预期提问
字段、标准答案或约束来源标签。调用 `ask_user` 后，只新增该问题对应的冻结用户回答。

### 2.2 Shopper 私有视图

模拟用户持有完整本轮购买目标、已经可见的事实、尚未披露的本轮事实，以及允许问题字段的冻结
回答。首版 Shopper 不是训练时自由发挥的在线 LLM；自然语言回答由 LLM API 在数据生成阶段产生
并冻结，运行时由确定性状态机返回，保证 SFT、GRPO 和评测使用相同事实。

### 2.3 验收器私有视图

验收器额外持有 ShopSimulator 原始商品与任务事实、目标商品和规格、每条约束的规范化值、证据与
信息来源，以及生成过程、模型、Prompt、seed 和数据哈希。该视图不能进入 Actor 上下文。

## 3. 信息来源与优先级

每条约束必须且只能归入一个主要来源：

```text
request_explicit     当前请求明确表达
clarification_answer 本轮提问后由用户明确回答
profile_stable_fact  画像中的高置信稳定事实，且适用对象与本轮一致
profile_preference   长期画像中的可推翻软偏好
irrelevant_profile   与本轮目标无关的画像干扰项
```

执行优先级保持为：

```text
request_explicit > clarification_answer > applicable profile_stable_fact
> profile_preference > default_assumption
```

`profile_stable_fact` 只在本轮明确为本人购买、字段确实稳定且没有冲突时才能成为操作约束，例如
本人鞋码。给他人购买或适用对象不明时不能沿用。`profile_preference` 默认不成为硬门。若当前请求
与画像冲突，当前请求必须获胜；该冲突要在任务包中显式标注，而不是让评测器临时猜测。

## 4. 首版画像 schema

正式训练只向 Agent 暴露紧凑、可验证的购物画像：

```json
{
  "profile_id": "profile-000001",
  "stable_facts": [
    {"field": "shoe_size", "value": "40", "applies_to": "self", "confidence": "high"}
  ],
  "category_preferences": [
    {"category": "运动鞋", "level": "high"}
  ],
  "brand_preferences": [
    {"category": "羽毛球鞋", "brand": "YONEX", "level": "high"}
  ],
  "budget_preferences": [
    {"category": "羽毛球鞋", "range_cny": {"min": 200, "max": 800}}
  ],
  "attribute_preferences": [
    {"category": "羽毛球鞋", "field": "function", "values": ["缓震", "耐磨"]}
  ],
  "option_preferences": [
    {"category": "鞋", "field": "size", "values": ["40"]},
    {"category": "运动鞋", "field": "color", "values": ["蓝色", "白色"]}
  ]
}
```

规则：

- 每个偏好尽量带适用品类，避免把鞋码、预算或颜色机械迁移到所有商品；
- 稳定事实首版只允许服装/鞋类尺码等少量可验证字段，并显式记录适用对象和置信度；
- 历史预算范围、品牌、颜色、材质和功能都属于偏好，不能伪装成稳定事实；
- `level` 首版只用 `high/medium/low`，其中 `low` 表示弱偏好，不表示禁止；
- 画像可包含与本轮无关的干扰项，但不能包含目标 ASIN、精确商品标题或为单个目标量身定制的唯一
  检索短语；
- 人口属性、位置、设备、支付方式和会员等级不进入首版 Agent 画像；它们可以由生成器产生用于审计，
  但不能直接决定商品要求。

## 5. 完整任务包 schema

一条正式任务采用以下逻辑结构；字段名在实现时保持稳定：

```json
{
  "schema_version": "personalized-shopping-task-v1",
  "task_id": "pca-000001",
  "source": {
    "shopsim_task_id": 123,
    "target_asin": "724988974873",
    "source_environment_version": "shopsimulator-environment-v2.1"
  },
  "scenario": "clarification_required",
  "profile": {},
  "current_request": "我想买一双平时打球穿的羽毛球鞋，要缓震耐磨。",
  "private_goal": {
    "category": "羽毛球鞋",
    "constraints": [
      {
        "constraint_id": "c-size",
        "field": "size",
        "value": "40",
        "hardness": "hard",
        "source": "clarification_answer"
      }
    ]
  },
  "clarification": {
    "should_ask": true,
    "max_questions": 2,
    "targets": [
      {
        "constraint_id": "c-size",
        "field": "size",
        "answer": "鞋码要40码。",
        "answer_facts": {"size": "40"}
      }
    ]
  },
  "conflicts": [],
  "generation": {},
  "audit": {}
}
```

`private_goal.constraints` 是唯一的完整本轮事实表。`current_request`、`profile` 和
`clarification.answer_facts` 必须逐条映射回该表，不能各自维护互相矛盾的答案。

## 6. 首版场景类型

一份混合 SFT 数据至少覆盖四类场景：

| 场景 | 请求 | 画像 | 是否应问 | 学习目标 |
|---|---|---|---:|---|
| `complete_request` | 信息完整 | 可有无关偏好 | 否 | 直接执行，不机械提问 |
| `profile_resolvable` | 缺少一个偏好或适用的稳定事实 | 画像可安全提供 | 否 | 正确利用画像 |
| `clarification_required` | 缺少一个关键本轮事实 | 画像也不能确定 | 是 | 问对字段并利用回答 |
| `profile_conflict` | 本轮明确要求与画像冲突 | 含相反历史偏好 | 否 | 当前请求覆盖画像 |

首版每条任务最多有两个 `clarification_answer` 约束，每次提问只对应一个规范字段。画像可以同时
包含多个偏好，但不得让模型通过无关字段反推出隐藏答案。

`profile_resolvable` 可以省略软偏好，也可以省略适用对象明确的高置信稳定事实。尺码只有在本轮
明确为画像本人购买时才能沿用；兼容性、数量、明确预算上限等本轮事实不能仅凭历史偏好静默当成
硬约束。字段不适用、置信度不足或影响对象不明时，应归入 `clarification_required`。

## 7. `ask_user` 数据语义

首版工具保持结构化，每条任务最多调用两次，每次只问一个主要字段：

```json
{
  "name": "ask_user",
  "arguments": {
    "field": "size",
    "question": "这次需要什么鞋码？"
  }
}
```

- `field` 必须来自任务包允许的规范字段；
- `question` 只询问一个主要事实，可以自然表达但不能夹带答案；
- 正确字段返回冻结回答，并作为 tool observation 加入后续上下文；
- 错误字段、超过两次、重复字段或当前不应提问时的调用由状态机记录，不让在线 LLM 临时编造答案；
- 是否立即终止错误提问留给交互协议实现阶段决定，本契约不提前规定 Reward。

首版候选字段来自能够映射到商品事实的规范集合：`budget`、`brand`、`function`、`material`、
`color`、`size`、`capacity`、`bundle`、`specification`。生成器只能使用当前目标商品确有证据支持的
字段。

## 8. LLM API 数据生成流水线

正式数据采用“生成器 + 确定性检查 + 独立批评器 + 环境验收”的组合，而不是一次 API 调用直接
落盘：

```text
ShopSimulator 候选任务事实
  → Task Architect：生成画像、信息分配、当前请求、冻结回答
  → Schema/事实/泄漏/冲突检查
  → Independent Critic：检查自然性、画像合理性和提问必要性
  → 人工抽检与必要修订
  → Teacher Agent + 冻结 Shopper 在真实环境中 rollout
  → 严格终局与轨迹协议验收
  → task-disjoint train/dev/eval 划分
  → manifest + hashes
```

生成和 Critic 使用独立 Prompt；可以使用同一 API 模型，但需记录完整模型名、endpoint 类型、
温度、Prompt 哈希和原始响应。若服务只提供滚动模型名，metadata 必须明确不能保证服务端 revision
完全复现。

## 9. 确定性质量门槛

任务进入教师 rollout 前必须通过：

- schema、类型、枚举、ID 唯一性和非空检查；
- 每条目标约束都能由 ShopSimulator 商品事实或原任务事实支持；
- 当前请求、画像和澄清回答之间无未声明冲突；
- `clarification_required` 的答案不出现在当前请求或画像中；
- `complete_request` 和 `profile_resolvable` 不存在关键未知字段；
- 画像不包含 ASIN、完整商品标题、唯一型号泄漏或针对目标商品拼接的搜索句；
- 当前请求不机械复制原商品标题；
- train/dev/eval 与 Final-200 按 source `task_id` 零重叠；
- 同一 source 商品或近重复请求不能跨 split 造成明显泄漏。

教师轨迹进入 SFT 前还必须通过：

- 所有购物动作在当时 observation 上合法；
- 提问次数和字段符合任务包；
- 用户回答进入上下文后，Agent 后续行为确实利用该事实；
- ShopSimulator 产生有效终局，且通过届时冻结的正式验收规则；
- 不保存目标答案、私有字段或验收标签到 Agent 可见 messages。

## 10. 产物与复现层次

建议正式目录为：

```text
data/personalized_agent/
  tasks_train.jsonl
  tasks_dev.jsonl
  tasks_eval.jsonl
  sft_train.jsonl
  sft_dev.jsonl
  manifest.json
  README.md

outputs/data_generation/<run_id>/
  raw_architect.jsonl
  raw_critic.jsonl
  rejected.jsonl
  teacher_trajectories.jsonl
  audit_summary.json
```

仓库保证两种复现：

1. **数据产物复现**：提交正式数据、manifest 和哈希，任何人能训练同一数据；
2. **生成流程复现**：提交脚本、Prompt、schema 和运行配置，能够重新执行生成流程。

远程滚动 LLM 即使温度为 0 也可能产生不同文本，因此不能承诺重新调用 API 后逐字节相同；原始
响应、正式产物和哈希是审计依据。

## 11. 与一次混合 SFT 的接口

四类场景最终统一转换为标准 chat messages：

```text
system
user(profile + current_request)
assistant(tool call: ask_user 或购物工具)
tool(user answer 或 ShopSimulator observation)
assistant(...)
...
tool(terminal observation)
```

Loss 仍只计算 assistant token。`ask_user` 调用、购物搜索、候选核验、规格选择和终止动作在同一
条多轮轨迹中共同学习，不要求拆成 A/B/C 多个训练阶段。

## 12. 实施前确认项

本契约建议冻结以下选择：

1. 原作者轨迹和画像不进入正式梯度数据；
2. 正式数据以 LLM API 生成和教师 rollout 为主，确定性检查与人工抽检负责质量收口；
3. 首版采用四种场景、最多两次单字段结构化提问和紧凑可验证画像；
4. 只有适用对象明确的高置信稳定事实可以从画像直接使用，普通长期偏好不能静默变成硬约束；
5. 首版所有场景混合为一份数据，执行一次 LoRA SFT；
6. Reward 与正式评测在交互和 SFT 数据跑通后另行设计。
