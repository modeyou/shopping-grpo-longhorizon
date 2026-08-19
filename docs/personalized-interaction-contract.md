# ShopSimulator 个性化多轮交互契约

- 日期：2026-08-19
- 状态：正式实现边界
- 目标：直接扩展 ShopSimulator，而不是离线重造任务、画像和 gold 澄清问题

## 1. 继承边界

ShopSimulator 内嵌数据已经同时提供：

- 完整购买目标 `instruction`，由环境和 Shopper 私有持有；
- 个性化初始请求 `instruction_sample`；
- 简化请求 `instruction_simple`；
- 用户画像 `user_persona`；
- 目标商品、规格、Reward 所需事实。

本项目直接继承这些基准输入并明确归因，不宣称重新生成画像或任务。内嵌数据约有 4,666 条同时带有
非空画像和简化请求，可作为个性化训练与评测任务池。正式梯度数据不使用原作者发布的 SFT/GRPO 轨迹，
而是由本项目在扩展后的环境中重新采集。

## 2. Agent 与私有状态

Persona 模式下，Agent 在 reset 后只能看到环境提供的个性化当前请求和去除 `__reasoning__` 的
`user_persona`。完整目标、目标 ASIN、Reward 事实和画像生成理由不得进入 Actor 上下文。

Shopper 私有持有完整购买目标和画像。Agent 可以使用原购物工具，或调用：

```json
{"name": "ask_user", "arguments": {"question": "这次购买对预算有什么要求？"}}
```

首版每条轨迹最多提问两次。问题使用自然语言，不预先提供 gold 字段或标准问题。Shopper 只依据私有完整目标
回答，不新增目标中不存在的要求；所有问答进入同一条轨迹上下文并完整留档。

## 3. 数据产生

```text
ShopSimulator persona task
  → Teacher Agent 在真实环境中选择 ask_user 或购物工具
  → Shopper 根据完整目标回答
  → ShopSimulator 执行搜索、核验、规格选择和购买
  → Reward v3 验收终局
  → 成功且协议合法的轨迹转换为 action-only SFT 数据
```

这里的“本项目自建数据”指本项目自行采集、验收、划分和冻结的多轮 Teacher/SFT/GRPO 轨迹；商品目录、
任务目标、画像和个性化请求属于继承的 ShopSimulator 基准数据。

## 4. 可复现性与验收

每条轨迹至少记录 task ID、persona 模式、环境版本、模型与 prompt 版本、全部 ask_user 问答、购物动作、
终局 Reward、运行配置和哈希。训练、开发和专项评测按 task ID 隔离，Final-200 Clean 不进入梯度数据。

首版不使用逐任务 LLM Critic，也不预生成 private constraint package。数据验收依赖工具协议、提问次数、
环境终局、Reward 有效性和人工抽检。提问价值、画像利用及新的 Reward 分项在 SFT 真实失败模式出现后再冻结。

## 5. 当前实现缺口

内嵌环境类已经支持 `if_persona=True`，但当前 HTTP 环境池固定以非 persona 模式创建，reset 请求中的
`if_persona` 尚未真正切换环境；环境也没有多轮 Shopper 工具。下一步只实现：

1. persona 环境启动与 reset 透传；
2. Actor-safe 的 persona observation；
3. `ask_user` 与 Shopper 会话状态；
4. Teacher rollout、轨迹验收和 SFT 转换接入。

Reward 公式和最终评测设计继续后置，不在交互闭环跑通前扩张范围。
