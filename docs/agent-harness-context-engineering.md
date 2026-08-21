# Shopping Agent 与 Harness：设计、上下文工程和 GRPO 复习

本文用于从方法论和工程两方面理解本项目的 Agent。重点不是某一条命令，而是回答以下问题：

- “模型”“Agent”“Harness”“环境”和“Shopper”分别是什么？
- veRL `ToolAgentLoop` 在哪里，为什么 GRPO 需要它？
- 为什么同一个 Prompt 在 GRPO 中生成多条轨迹？
- 为什么长程 Agent 不能把所有 Observation 原样塞回上下文？
- 为什么不能简单地从最早的 Token 开始截断？
- Observation Projection、Action Guard、Reward 和轨迹诊断如何组成闭环？
- 原参考项目已经实现了什么，我们的多轮澄清版本还缺什么？

本文中的“当前已实现”指截至 2026-08-21 仓库中已有代码；“待实现”表示已经确定方向，
但不能在完成代码和端到端冒烟前声称可用。

## 1. 先建立完整心智模型

### 1.1 模型不等于 Agent

语言模型只完成一次条件生成：给定 Token 序列，预测后续 Token。它不会自行管理浏览器状态、
检查按钮是否合法、调用 ShopSimulator、释放 Session 或把 Reward 交给 GRPO。

本项目里的 Agent 是以下部分共同构成的运行系统：

```text
Actor 模型
  + System Prompt / Chat Template
  + Tool Schema
  + ToolAgentLoop
  + Shopping Harness
  + ShopSimulator
  + Context Engineering
  + Action Guard
  + Reward 与诊断
```

其中只有 Actor 模型参数会被 SFT/GRPO 更新。Harness 本身不是另一个被训练的模型，而是决定
Actor 能看到什么、能做什么以及一条轨迹如何开始和结束的运行时协议。

### 1.2 五个核心角色

| 角色 | 可见信息 | 主要职责 | 不应做什么 |
|---|---|---|---|
| Actor | 当前条件指定的公开 opening、页面 Observation、已经公开的 Shopper 回答 | 决定是否澄清、搜索、核验、选规格和购买 | gap 条件不得看到被省略事实；任何条件都不得看到 gold 商品或环境私有元数据 |
| Shopper | 私有完整目标、允许回答的遗漏事实、历史问答 | 回答用户拥有的预算、用途、兼容性和偏好 | 不搜索商品，不替 Actor 购买，不虚构偏好 |
| ShopSimulator | 商品、页面状态、完整目标、Reward 所需事实 | `reset/step`、页面反馈、终局判定 | 不负责自然语言扮演用户 |
| Harness | Session、工具、上下文、Guard、终止、诊断 | 把模型输出变成可审计环境轨迹 | 不替 Actor 决定应买什么 |
| veRL | Rollout、GRPO 优势、梯度、分布式训练 | 在线采样并更新 Actor 参数 | 不理解购物业务语义 |

### 1.3 一条轨迹的抽象流程

```text
任务/opening
    ↓
Harness 创建独立 ShopSimulator Session
    ↓
Actor 生成一个 Tool Call
    ↓
Harness 校验 Schema 和 Action Guard
    ↓
执行 ask_shopper 或 ShopSimulator Action
    ↓
Harness 投影并记录 Observation
    ↓
将结果加入该轨迹自己的上下文
    ↓
重复，直到购买、主动结束、超步数或基础设施失败
    ↓
环境返回 Reward V4 和终局明细
    ↓
评测保存指标；GRPO 用同题多轨迹 Reward 更新 Actor
```

## 2. 用一个自动浇水器任务走完整闭环

假设 ShopSimulator 中的私有完整目标是：

```text
购买带大屏、雾化、智能感雨暂停、双路独立控制和25米快插式地埋喷头的
自动浇水器；必须使用铜芯电磁阀，预算约230元。
```

对 Actor 公开的 gap opening 是：

```text
我想买一款带雾化功能、能智能感雨暂停、双路独立控制的自动浇水器，
最好有大屏幕显示，并且要配25米快插式地埋喷头。
```

opening 故意省略：

```json
{
  "omitted_dimensions": ["材质", "价格"],
  "omitted_facts": [
    "浇水器必须是铜芯电磁阀的",
    "价格在230元左右"
  ]
}
```

### 2.1 正确的多轮轨迹示例

第一轮，Actor 判断缺失的是会改变选择的用户约束，而不是应从商品页查询的目录属性：

```json
{
  "name": "ask_shopper",
  "arguments": {
    "question": "您对电磁阀材质有什么要求？预算大概是多少？"
  }
}
```

Shopper 只能依据私有目标回答：

```json
{
  "answer": "需要铜芯电磁阀，预算大约230元。",
  "used_facts": [
    "浇水器必须是铜芯电磁阀的",
    "价格在230元左右"
  ]
}
```

此时这两个事实已经通过合法对话公开给 Actor。Harness 应把它们写入轨迹审计；未来实现多轮
GRPO 时，还应维护一个小型 `clarified_constraints` 状态，防止关键约束在长轨迹中丢失。

第二轮，Actor 搜索：

```json
{
  "name": "search_products",
  "arguments": {
    "query": "自动浇水器 铜芯电磁阀 双路 感雨 雾化"
  }
}
```

环境可能返回当前页 20 个商品。Harness 不会把冗长原文直接塞回模型，而是通过
Observation Projection 压缩标题和描述，同时保留当前页全部 ASIN、价格、翻页按钮和搜索状态。

第三轮，Actor 打开某个当前可见商品：

```json
{
  "name": "open_product",
  "arguments": {"asin": "677758868630"}
}
```

如果 Actor 试图打开上一页出现、但最新 Observation 已不存在的 ASIN，Action Guard 会拒绝，
并把当前合法目标重新告诉模型，而不是让错误动作污染环境。

随后 Actor 查看 Features、Attributes、选择完整规格并检查实际 variant 价格。只有商品和规格
满足要求时才调用 `buy_now`。ShopSimulator 在终局返回 Reward V4：购买正确则
`gold_purchase`，否则可能是 `wrong_purchase`、`partial_alternative_purchase` 或其他类型。

### 2.2 三种评测条件如何复用同一任务

| 条件 | Actor 初始看到的内容 | `ask_shopper` | 要回答的问题 |
|---|---|---:|---|
| G+ | gap opening | 可用 | 模型能否发现信息缺口并通过澄清提高成功率？ |
| G− | 与 G+ 完全相同的 gap opening | 禁用 | 没有澄清通道时表现如何？ |
| C+ | complete opening | 可用 | 信息完整时模型会不会仍然过度提问？ |

G+ 与 G− 必须使用同一份 gap opening，才能把结果差异归因于澄清通道，而不是两次 opening
生成的随机差异。

## 3. veRL ToolAgentLoop 到底是什么

### 3.1 它是通用循环引擎，不是完整 Shopping Harness

veRL 的 `ToolAgentLoop` 管理“生成—调用工具—加入工具结果—继续生成”的状态机。可以把它
简化为：

```text
GENERATING
  Actor 生成 assistant 文本或 Tool Call
      ↓
PROCESSING_TOOLS
  解析和执行 Tool Call
      ↓
  Tool Observation 加入上下文
      ↓
GENERATING
      ...
      ↓
TERMINATED
```

本项目的 `ShoppingToolAgentLoop` 继承 veRL 类，并在四个边界加入购物协议：

| 覆写位置 | 本项目增加的职责 |
|---|---|
| `_handle_generating_state` | 生成前检查上下文预算，限制本轮最大输出 |
| `_call_tool` | 工具返回后进行页面感知 Observation Projection |
| `_handle_processing_tools_state` | 禁止同一回合并行执行多个工具，响应终止标记 |
| `run` | 创建/关闭 Session，结算 Reward，输出轨迹诊断 |

因此，veRL ToolAgentLoop 是骨架，Shopping Harness 是装在骨架上的业务器官。

### 3.2 为什么 GRPO 不能只用普通 `while` 循环

普通评测只需保存消息和最终结果；GRPO 还必须保存并严格对齐：

```text
prompt_ids
response_mask
response_logprobs
```

- System、User、Tool Observation Token 不由 Actor 生成，通常 `response_mask=0`；
- Actor 生成的 assistant/Tool Call Token 通常 `response_mask=1`；
- 每个 Actor Token 还需要 rollout 时的 log probability。

这些数据用于计算策略比率和 GRPO loss。任何上下文裁剪、Tool Call 拼装或消息顺序错误，都可能
让梯度落到错误 Token 上。

## 4. 为什么同一个 Prompt 要采样四条轨迹

### 4.1 GRPO 使用同题组内比较

当前参考配置在训练时为每个 Prompt 设置 `n=4`、temperature 0.7。四条轨迹拥有相同任务和
初始 opening，但拥有独立的模型采样、上下文和 ShopSimulator Session：

```text
同一 Prompt
├─ Context 1 → Session 1 → Reward  1.00
├─ Context 2 → Session 2 → Reward -0.85
├─ Context 3 → Session 3 → Reward -0.65
└─ Context 4 → Session 4 → Reward  0.55
```

四条上下文不会拼接，轨迹之间也看不到彼此的搜索、提问和结果。“同时保留”只是指系统需要
同时或分批保存四套 Token、KV Cache、mask、logprob、工具轨迹和 Reward。

GRPO 完成四条 rollout 后，才在同一个 Prompt 组内计算相对优势。概念上可理解为：

```text
轨迹优势 ≈ 该轨迹 Reward − 同题组平均 Reward
```

当前配置关闭了按组标准差再次归一化，因此实际实现应以 veRL 固定版本为准，不能把上式当作
所有配置下的完整数学定义。

如果每题只有一条轨迹，它和自己的均值相同，无法获得有意义的组内优劣信号。如果四条全部
同分，也没有有效更新信号；原项目的 bounded dynamic sampling 会有限次重采，而不是无限循环。

### 4.2 为什么不是越多越好

更多 rollout 通常提高组内多样性，但成本也近似增长：

- 更多 KV Cache 和响应 Token；
- 更多 ShopSimulator Session；
- 更多 Shopper API 调用；
- 更长的 rollout 和 logprob 计算时间。

`n=4` 是原项目的资源—信号折中，不是不可更改的理论常数。Qwen3.5-2B 在 4×4090 上仍需用
真实的一次 optimizer update 冒烟决定最终并发、batch 和 rollout 数。

## 5. Shopping Harness 具体负责什么

### 5.1 Session 生命周期

每条轨迹必须独立完成：

```text
租用环境 → reset(task_id) → 多步 step(action) → 终局 → release
```

同步 HTTP 调用放在线程中，避免阻塞 veRL 异步事件循环；`ContextVar` 将当前环境和运行状态绑定到
coroutine，防止并发轨迹串 Session。任何退出路径都应在 `finally` 中释放环境。

### 5.2 Tool Call 到环境 Action

Actor 输出标准 Function Call，例如：

```json
{"name": "open_product", "arguments": {"asin": "677758868630"}}
```

Harness 将其转换为 ShopSimulator 动作：

```text
click[677758868630]
```

搜索、翻页、选择规格、查看子页、购买和主动结束都遵循同一映射。`ask_shopper` 是例外：它不应
传给 ShopSimulator 页面动作，而应由多轮 Harness 路由到独立 Shopper。

### 5.3 终止与错误边界

轨迹可能因为以下原因结束：

- 正确或错误购买；
- 合格的主动无购买结束；
- 最大环境步数；
- 重复循环；
- 连续非法动作；
- Actor 不再调用工具但环境未终局；
- 环境、模型服务或 Shopper 基础设施失败；
- 上下文无法安全容纳。

基础设施失败不能伪造为模型 Reward 0 或普通失败，否则 GRPO 会学习服务故障造成的随机噪声。

## 6. 上下文工程：出发点、危险与当前方案

### 6.1 出发点：长程轨迹会不断膨胀

一条上下文会累积：

```text
System Prompt
User opening
Actor search Tool Call
20个搜索结果
Actor open Tool Call
商品详情
Actor view_features Tool Call
大量 Features
...
```

如果反复核验多个候选，轨迹可能达到二三十步。GRPO 同题还会保留四条独立轨迹，并需要计算
logprob，因此上下文过长既可能超过模型窗口，也会迅速消耗 KV Cache 和训练显存。

上下文工程的目标不是单纯“少给模型文字”，而是同时保证：

1. 不超 Token/显存预算；
2. 不丢失当前合法动作目标；
3. 不破坏 Tool Call 结构；
4. 不破坏 GRPO Token、mask、logprob 对齐；
5. 保留完成任务真正需要的证据和约束。

### 6.2 为什么不能简单从最早 Token 开始滑动截断

简单滑动窗口可能造成四类错误：

1. **忘记任务**：删掉原始用户需求、预算或 System Prompt；
2. **破坏消息结构**：留下 Tool Observation，却删掉对应 assistant Tool Call；
3. **丢失动作目标**：商品文本仍在，但 ASIN、规格或页面按钮被截掉；
4. **破坏训练对齐**：只删 `prompt_ids`，没有同步裁剪 `response_mask` 和
   `response_logprobs`。

即使按消息删，也必须避免把 assistant/tool 配对拆开，并明确哪些固定信息永远不能删除。

### 6.3 第一层解决方案：页面感知 Observation Projection

原项目优先控制“每一步新进入上下文的内容”，而不是等上下文爆炸后才删除历史。当前预算是：

| 页面 | Token 预算 | 必须保留的内容 |
|---|---:|---|
| 搜索结果 | 1536 | 当前页全部 ASIN、价格、可点击商品、翻页/导航、搜索状态 |
| 商品详情 | 4096 | 头部关键信息、尾部规格/状态、全部动作按钮 |
| 普通/信息页 | 768 | 头部、少量尾部、全部动作 footer |
| 搜索页容量 | 20 个商品 | 不允许投影后悄悄减少当前页候选 |

投影器先识别页面类型，再压缩长标题和正文，最后重新验证：

```text
压缩前 ASIN 集合 == 压缩后 ASIN 集合
压缩前导航按钮 == 压缩后导航按钮
可见搜索商品 == 可点击商品
可见 Token <= 页面预算
```

如果无法满足安全契约，轨迹会被标为基础设施无效，而不是输出一个看似可用、实际已经丢失动作
空间的截断页面。

### 6.4 第二层解决方案：总上下文硬预算

当前 GRPO 参考配置包含以下相互关联但含义不同的边界：

```text
模型最大上下文                         24576
未启用历史压缩时的硬输入上限           23552
启用历史压缩时的目标输入预算           16384
本轮生成预留                              512
安全余量                                  512
veRL 初始 prompt 上限                    4096
veRL 一条 rollout 的最大累计 response   20480
```

这些参数属于不同边界，不能机械相加当作单一窗口；最终必须以 tokenizer 渲染后的真实 Token、
vLLM 限制和一次 GRPO update 的显存峰值验证。

### 6.5 第三层解决方案：按完整工具组同步压缩

仓库已经实现可选的安全压缩：固定保留初始 anchor，并只整组删除较旧历史：

```text
固定 anchor：System + User

可删除组1：Assistant Tool Call + 对应 Tool Observation
可删除组2：Assistant Tool Call + 对应 Tool Observation
...

至少保留最近若干完整组
```

GRPO 版本会同步裁剪 `prompt_ids`、`response_mask` 和 `response_logprobs`。但 canonical
GRPO 配置目前仍为：

```yaml
context_compaction_enable: false
```

原因不是该方向没有意义，而是任意在线裁剪都必须先证明：后续生成真正使用了相同压缩上下文，
旧策略 logprob 与训练重算仍对齐，Tool Call 边界没有损坏。正式启用前必须做 token-level 测试和
一次真实 veRL update 冒烟。

### 6.6 多轮澄清新增的上下文要求（Harness v1 已实现）

Shopper 已经公开的回答不能因为历史压缩而丢失。例如：

```text
需要铜芯电磁阀，预算约230元。
```

多轮 GRPO Harness 维护仅含“已经公开事实”的小型状态：

```json
{
  "clarified_constraints": [
    "需要铜芯电磁阀",
    "预算约230元"
  ]
}
```

它不能包含未被询问的私有目标。Harness v1 将其确定性地投影进后续 Observation；投影前先从
当前页型预算中扣除 anchor 的 Token，避免追加后突破原预算。是否进一步启用历史压缩，仍需真实
veRL update 验证 mask/logprob 对齐后决定。

## 7. Action Guard：拉 Agent 一把，而不是替 Agent 做决定

### 7.1 Guard 解决什么

模型容易生成环境无法执行的动作，例如：

- 点击上一页出现、最新页已经不存在的 ASIN；
- 在商品详情子页直接搜索；
- 把 `Description` 当成商品规格值；
- 选择当前页不存在的颜色；
- 传入 Tool Schema 之外的参数；
- 同时调用多个相互冲突工具。

当前 Guard 检查最新 Observation 中的 ASIN、可点击按钮、规格值和搜索/翻页可用性。同一回合
最多执行一个工具；连续非法调用达到上限后终止轨迹。

### 7.2 为什么 Guard 不是越多越好

推荐顺序是：

1. 先观察真实失败轨迹；
2. 尝试用 Prompt 或 Tool Description 消除错误；
3. 如果模型仍频繁触发，或错误一旦执行就会污染环境，再加入硬 Guard；
4. 保留 Guard 命中原因和次数，用数据决定是否继续存在。

`ask_shopper` 适合硬校验 Schema、非空问题、次数限制和完全重复问题；“这个问题是否有价值”
不应由脆弱的关键词 Guard 决定，而应由澄清审计、G+/G−/C+ 评测和终局购买结果共同判断。

## 8. Reward、轨迹诊断与 GRPO 更新

Harness 只接受环境返回并经过版本验证的 Reward V4，不应自行根据自然语言猜测购买是否正确。
一条 GRPO 轨迹还应输出：

- task/rollout ID、工具序列和步数；
- done、termination reason 和 infrastructure invalid；
- Reward type、validity、terminal utility、match/evidence；
- Action Guard 命中及原因；
- Observation 原始/可见 Token、压缩比例和页型；
- 最大上下文 Token、压缩次数和删除 Token；
- 多轮问题、回答、`used_facts`、grounded/over-ask 指标；
- Shopper 模型、Prompt hash、超时和重试诊断。

动态采样只保留同题 Reward 有变化且基础设施有效的组。全部同分的组没有相对优势信号；但重采
必须有上限，避免模型早期成功率过高或过低时无限等待有效组。

## 9. 原参考项目与当前多轮分支的实现状态

### 9.1 已经可以沿用

- veRL 0.8 异步 ToolAgentLoop；
- vLLM 多轨迹 rollout；
- 独立 Session、`ContextVar` 隔离和可靠释放；
- Tool Call→环境 Action 映射；
- 单工具串行执行；
- 页面感知 Observation Projection；
- Action Guard；
- 环境/Reward 版本校验；
- Reward V4 底层解析；
- bounded dynamic sampling 和训练诊断。

### 9.2 多轮 veRL Harness v1 已接入

同步评测/教师 rollout 已支持：

- `ask_shopper`；
- gap opening；
- Shopper 私有上下文；
- `used_facts` 审计；
- 问题次数限制；
- G+/G−/C+ 条件。

当前 veRL 路径已经补齐：

- `configs/tools.json` 由 canonical Python schema 生成并包含 `ask_shopper`；
- `ask_shopper` 路由到 trajectory-local Shopper，不进入 ShopSimulator，也不占购物步骤；
- Session 按 parquet 中的 gap/complete opening 元数据初始化并核对 `source_goal_hash`；
- 省略事实只保存在私有 Shopper ContextVar 中，公开 runtime/诊断不保存事实原文；
- 问题、购物步骤和拒绝次数分别计数；
- Shopper 回答公开后写入 `clarified_constraints` 并锚定到后续 Observation；
- `prepare_multiturn_grpo_dataset.py` 可从冻结 opening 生成多轮 veRL parquet；
- 启动入口固定 `environment-v4.json` / Reward V4，并要求独立 Shopper 配置。

尚未完成的是服务器上的真实 veRL 单轨迹、四轨迹和一次 optimizer update 冒烟；因此 v1 目前是
“代码契约完成、CPU 测试通过”，还不是“训练运行已经验收”。

## 10. 多轮 GRPO Harness 的实现与后续验收

### P0：正式 GRPO 前必须完成

上述代码契约已经完成；下一步是在服务器构建 task-disjoint parquet，然后完成单轨迹、单组四轨迹
和 4×4090 单 optimizer update 冒烟。真实运行通过前不启动正式长训练。

### P1：根据开发集和冒烟结果定参数

- gap/complete 训练比例，防止只学会无条件提问；
- rollouts per prompt 是否保持 4；
- 最大 Shopper 问题数；
- DeepSeek Shopper 并发、速率限制、缓存与失败重试；
- Qwen3.5-2B Chat Template 和 Tool Call Parser；
- Observation 预算与是否启用 mask-safe history compaction；
- 4×4090 上 FSDP、vLLM 显存比例和 rollout worker 数。

首版不应同时引入 Planner、长期 Memory、Reflection、多 Agent、PRM 或复杂 step-level reshaping。
先让基本多轮闭环稳定，再根据失败轨迹判断信用分配是否确实是主要瓶颈。

## 11. 常见问题与本项目结论

### Q1：四条 GRPO 轨迹都会成为同一个 Agent 的上下文吗？

不会。它们是四套相互隔离的上下文和环境 Session，只在结束后按同一个 Prompt 分组比较
Reward。系统内存/显存需要容纳四条，不代表某条轨迹能看到其他三条。

### Q2：为什么同一个 Prompt 需要多条轨迹？

GRPO 依赖同题内的相对好坏；单条轨迹与自己的组均值相同，缺少比较信号。多条随机采样使模型
可能产生不同提问、搜索和购买路径，Reward 才能告诉策略哪些 Token 序列更值得提高概率。

### Q3：四条轨迹是否共用 Shopper 回答？

不应共用实时回答。每条轨迹可能提出不同问题，也可能完全不问，因此要有独立问答历史。可以
共享冻结的事实库和缓存基础设施，但不能把另一条轨迹尚未询问的回答泄露过来。

### Q4：ShopSimulator 为什么不能直接回答澄清问题？

ShopSimulator 是确定性环境和 Reward 判定器，不是自然语言用户。它通过私有通道提供事实；
Shopper 才负责将被问到的事实转成自然、受审计的回答。这样环境语义、用户模拟和 Actor 策略
职责分离。

### Q5：为什么不能简单滑动窗口？

因为可能删任务、拆 Tool Call、丢动作目标，并破坏 GRPO Token/mask/logprob 对齐。正确优先级
是页面感知投影→总预算→必要时按完整工具组同步压缩，而不是按字符或固定 Token 粗暴截断。

### Q6：上下文越长是否一定越好？

不是。冗长页面会提高显存、延迟和注意力噪声。目标是保留“可行动目标、当前证据、用户约束和
必要历史”，而不是保留所有原文。

### Q7：Action Guard 是否越严格越好？

不是。Guard 应防止确定非法或会污染环境的动作；策略选择错误应尽量由 Reward 和训练纠正。
过多 Guard 会缩小探索空间，甚至把 Harness 的启发式偏见伪装成 Agent 能力。

### Q8：SFT 与 GRPO 中 Shopper 的作用一样吗？

SFT 是离线学习保存好的消息轨迹，训练时不调用 Shopper。GRPO 是 on-policy 在线 rollout，
Actor 每次更新后可能提出新问题，因此必须在线接入 Shopper 或受控 Answer Bank 路由。

### Q9：为什么不能提前生成完整 GRPO rollout？

因为 GRPO 是 on-policy：Actor 参数更新后，新的问题和行动分布也会变化。提前固定完整 Actor
轨迹会变成离线偏好/RFT 类数据，而不是标准在线 GRPO。可以提前冻结任务、opening、事实库和
Shopper Contract，但不能冻结未来 Actor 的实际动作。

## 12. 端到端验收清单

正式 GRPO 前至少验证：

- [x] Actor parquet Prompt 只包含当前条件指定的公开 opening；
- [x] gap 条件的省略事实和环境私有元数据不会出现在 Actor Prompt 或公开诊断；
- [x] `ask_shopper` 已接入 veRL Tool Adapter 的 trajectory-local 路由；
- [x] Shopper 回答只允许引用 opening 审计事实，并保留私有 provenance hash；
- [x] 问答历史由 `ContextVar` 按 trajectory 隔离；
- [x] `clarified_constraints` 只含已经公开的回答；
- [ ] 搜索投影完整保留当前页全部 ASIN 和按钮；
- [ ] Action Guard 使用的是最新可见 Observation；
- [ ] Tool Call 严格串行；
- [ ] context、mask、logprob 长度和分组边界一致；
- [x] 启动入口固定 Reward V4 与 `environment-v4.json`；
- [ ] 基础设施失败不进入有效 GRPO 组；
- [ ] 同题四轨迹上下文和 Session 不串线；
- [ ] 动态采样有重采/跳步上限；
- [ ] Qwen3.5-2B 完成一次真实 4 卡 optimizer update；
- [ ] Base/SFT/GRPO 正式评测使用同一冻结 Harness Contract。

## 13. 代码导航

| 主题 | 文件 |
|---|---|
| veRL Shopping AgentLoop | `src/shopping_grpo/training/grpo/adapter/agent_loop.py` |
| Session 生命周期 | `src/shopping_grpo/training/grpo/adapter/session.py` |
| veRL Tool Adapter | `src/shopping_grpo/training/grpo/adapter/tools.py` |
| 运行状态与 Reward 诊断 | `src/shopping_grpo/training/grpo/adapter/runtime.py` |
| trajectory-local 受控 Shopper | `src/shopping_grpo/training/grpo/adapter/shopper.py` |
| Observation Projection | `src/shopping_grpo/environment/projection.py` |
| 上下文窗口与同步裁剪 | `src/shopping_grpo/environment/context.py` |
| Action Guard | `src/shopping_grpo/environment/actions.py` |
| Tool Schema 与 Action 映射 | `src/shopping_grpo/environment/tools.py` |
| 教师/评测同步 rollout | `src/shopping_grpo/evaluation/rollout.py` |
| Shopper 模拟器 | `src/shopping_grpo/multiturn/shopper.py` |
| GRPO 主配置 | `configs/grpo.yaml` |
| AgentLoop 参数 | `configs/agent_loop.yaml` |
| veRL 工具配置 | `configs/tools.json` |
| 工具配置生成/漂移检查 | `scripts/generate_grpo_tool_config.py` |
| 多轮 opening→veRL parquet | `scripts/prepare_multiturn_grpo_dataset.py` |
| GRPO 启动入口 | `scripts/train_grpo.py` |
| 动态采样和指标 | `src/shopping_grpo/training/grpo/dynamic_sampling.py` |

相关复习文档：

- [多轮 Teacher/SFT 数据](multiturn-teacher-sft-review.md)
- [训练、Reward 与评测决策](training-reward-evaluation-decisions.md)
- [多轮评测协议](multiturn-evaluation.md)
- [Reward V4](reward-v4.md)
- [GRPO 使用说明](grpo.md)
