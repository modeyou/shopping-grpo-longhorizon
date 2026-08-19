# 多轮澄清 Agent 设计

- 日期：2026-08-19
- 分支：`feat/multiturn-clarification-agent`
- 基线：`main@28be0ed`

## 目标

在参考项目既有的 `Baseline → SFT → GRPO → Evaluation` 流水线中接入
ShopSimulator 原生的 Shopper–Agent 多轮语义。项目研究问题限定为：Agent 是否能在
购物请求缺少关键事实时提出有效问题，并利用回答提高最终严格购买成功率。

首版不使用用户画像，不遮蔽画像，不强制部署模型首轮提问，也不使用额外 LLM Critic。

## 与两个上游的边界

ShopSimulator 原生多轮流程让 Shopper 私有持有完整 `instruction` 和
`goal_options`，先生成模糊购物请求，再在 Agent 选择 `ask_shopper` 时回答问题。

参考项目的 `main` 是单轮用户协议：Actor 开头直接看到完整 `instruction`；veRL
配置中的 `multi_turn` 仅表示 Actor 与工具环境的多步循环，并不包含 Shopper。

本项目不复制 ShopSimulator 官方 `multi_eval` 的文本动作解析和独立 rollout，而是在
参考项目现有的 OpenAI function calling、Action Guard、Reward v3、SFT 转换和 veRL
适配层内实现等价交互。

## 数据契约

每个多轮任务由以下字段标识：

```json
{
  "schema_version": "shopsimulator-multiturn-task-v1",
  "task_id": 123,
  "initial_request": "我想买一款适合通勤的双肩包。",
  "source_goal_hash": "sha256...",
  "opening_model": "model-name",
  "opening_prompt_hash": "sha256..."
}
```

完整目标不写入 Actor 数据。环境 reset 后通过进程内私有通道向 Shopper 提供完整
`instruction` 和 `goal_options`；序列化轨迹的 Actor 消息和公开 `initial_result` 不得
包含这些私有字段。

初始请求由 Shopper LLM 为每个任务生成一次并冻结。后续 Baseline、SFT、GRPO 和
Evaluation 对同一 `task_id` 复用相同请求，以控制 API 成本和输入方差。生成器支持
断点续跑，并记录模型、prompt 和源目标哈希；源目标变化时旧开场立即失效。

## 运行时交互

Actor 获得原购物工具和一个开放式工具：

```json
{"name": "ask_shopper", "arguments": {"question": "您需要多大功率的？"}}
```

每个 Actor 回合仍只允许一个工具调用。`ask_shopper` 不改变商店页面，只调用一次
Shopper completion，将自然语言回答作为 tool message 追加到同一轨迹。Shopper 只根据
完整目标、规格选项和既有问答作答；目标没有答案时明确表示没有额外要求或不确定。

首版每条轨迹最多提问两次，Actor 与 Shopper API 调用各最多重试两次。轨迹记录冻结
开场标识、全部问答、Actor 调用数、Shopper 调用数、购物步骤和 Reward v3 终局。

## SFT 与验收

教师模型自主决定是否调用 `ask_shopper`，不使用 `tool_choice` 强制提问。SFT 仍只接收
`reward_valid=true` 的 `gold_purchase` 轨迹，并保留 `ask_shopper` schema、提问动作、
Shopper tool response 和后续购物动作。

正式 SFT 数据同时包含：

- 模糊请求成功轨迹，用于学习必要澄清；
- 完整请求成功轨迹，用于维持无需提问时直接购物的行为。

两类数据按 task ID 与 Final-200 Clean 隔离。第一阶段只验证数据与运行时闭环，不在尚未
获得真实分布前冻结新的 Reward 权重。

## 失败处理与最小测试

- 冻结开场缺失、hash 不匹配或私有目标缺失：任务失败，不回退到完整请求。
- Shopper 空回答、工具调用或 API 失败：轨迹记为基础设施错误，不进入 SFT。
- 超过两次提问：Action Guard 拒绝，不调用 Shopper。
- 核心测试覆盖私有目标隔离、冻结开场校验、单次问答续接、提问上限、SFT 消息保留和
  standard 单轮协议不回归。

## 暂不实施

- 用户画像与画像遮蔽；
- 固定字段选择式提问；
- 强制首轮 `ask_shopper`；
- 在线重复生成初始请求；
- 澄清过程 Reward、LLM Critic 和完整正式评测。
