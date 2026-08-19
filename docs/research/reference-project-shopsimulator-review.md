# 参考项目与 ShopSimulator 原文审阅

- 日期：2026-08-19
- 用途：正式项目规划前的事实基线
- 一手来源：ShopSimulator arXiv:2601.18225、当前仓库 `main@28be0ed`、内嵌环境源码

## 1. 参考项目实际完成的范围

当前 `shopping-grpo-longhorizon` 参考项目将 ShopSimulator 裁剪为“完整需求在开头给出的单轮、
非个性化购物任务”，再实现以下链路：

```text
教师环境 rollout → Reward v3 严格验收 → action-only LoRA SFT
→ veRL 在线 GRPO → Final-200 统一评测
```

其主要工程特征为：

- `Qwen/Qwen3.5-2B` 基座；
- OpenAI-compatible 教师在真实环境中执行工具轨迹；
- SFT 只监督 assistant token，用户与 tool observation token 被 mask；
- 结构化工具、Action Guard、observation projection 和 35 步确定性终止；
- GRPO 使用 `verl==0.8.0`、每题 4 条 rollout、在线确定性 Reward v3；
- 原始 Final-200 结果为 Base 0%、SFT 60.5%、GRPO step 100 为 62.0% strict success；
- 当前 system prompt 明确禁止追问，运行时没有 Agent—用户对话闭环。

因此，用户画像与主动澄清不是在现有策略上打开两个配置即可得到，而需要扩展任务状态、动作协议、
rollout orchestrator、数据 schema、训练样本和评测。

## 2. 参考项目的 A/B/C 是否是必要训练范式

不是。当前 Pure V4 将 1,073 条训练轨迹按难度做累计 curriculum：A 学动作协议，B 加约束任务，
C 加长程任务。简单样本因此被重复训练三次。这是参考项目后加的工程策略，不是 ShopSimulator
论文规定的训练方法，也不是多轮对话的含义。

训练器本身支持直接传入任意一份 train/validation JSONL；curriculum 参数是可选项。只要一条样本
包含完整的 user、assistant、tool 多回合消息，单次 SFT run 就能监督其中所有 assistant turns。

## 3. ShopSimulator 原文中的任务定义

论文将策略从 `a_t = π(o_t, G)` 扩展为 `a_t = π(o_t, u_t, p)`：

- `o_t`：购物环境观察；
- `u_t`：当前用户话语，允许后续澄清回答更新；
- `p`：静态用户画像；
- 多轮模式把直接与用户交流加入动作空间。

论文覆盖四种场景：Single-Turn、Multi-Turn、Single-Turn & Personalization、Multi-Turn &
Personalization。多轮与个性化是两个独立维度，可以组合但不能混为同一概念。

## 4. 原文如何构造用户画像和个性化请求

论文从已有的商品—完整指令对出发：

1. 用 LLM 生成包含长期需求线索的画像初稿；
2. 由人工审阅和修订，丰富长期偏好并避免画像过拟合目标商品；
3. 将应由长期画像提供的信息从当前指令中移除；
4. 仍将过于具体、只属于本次购买的要求保留在当前指令中；
5. 最终得到 4,726 份画像及相应改写指令。

论文画像示例包含人口属性、地区、交易特征、搜索/收藏/加购行为、类目/品牌/价格/功能/尺码/
材质/颜色/风格偏好和用户标签。其错误分析表明三个主要风险是：忽略画像、过度解读画像、混淆
短期需求和长期偏好。

本项目首版保留论文的“长期偏好与本轮需求分离”原则，但只选能够通过商品事实客观验证的画像
字段。人口属性可以出现在原始生成候选中，但不直接作为商品约束或奖励依据。

## 5. 原文如何实现多轮澄清

原文使用 LLM 模拟 Shopper：它持有完整目标，以模糊意图开场，只在 Agent 提问后逐步披露必要
属性；不会主动一次给全。购买前如果 Agent 尚未收集完整信息，Shopper 应拒绝并指出缺失内容。
Agent 的动作在 `ask_shopper` 和环境操作之间选择，并在最终购买前与用户确认。

这说明正式数据至少需要区分：

- 当前可见请求；
- 静态画像；
- Shopper 私有的完整本轮目标；
- 已披露事实与尚未披露事实；
- Agent 提问、Shopper 回答和环境动作的统一时序。

## 6. 原文训练结论与单次 SFT

原论文为每个场景收集 GPT-4.1 的成功轨迹，共 6K 条，以 batch size 32、学习率 `1e-5` 对
Qwen3-8B 做 4 epochs SFT；正文和附录没有 A/B/C curriculum。之后使用 GRPO 训练 200 steps，
每个 step 对 32 个样本各做 8 条 rollout。

论文结论是 SFT 和 RL 互补：SFT 更擅长注入工作流和冷启动，RL 更有利于偏好优化；多轮个性化
仍然最难。该证据支持本项目先采用“一份混合数据、一次 SFT run”，而不是在数据尚未验证时先
引入课程学习。是否需要 curriculum 应由各任务类型的开发集表现决定。

## 7. 当前内嵌环境对正式项目的影响

内嵌压缩数据有 23,421 条商品—指令记录，其中约 4,666 条带 `user_persona` 和非空
`instruction_simple`，可直接作为继承的个性化基准任务。但当前 HTTP 服务初始化环境时没有开启
persona 模式，项目客户端 reset 也没有多轮 Shopper 协议；当前训练/评测 prompt 反而明确禁止追问。

需要注意，冻结压缩包并不含环境源码所引用的 `instruction_sample` 字段；个性化开场请求实际存储在
非空 `instruction_simple` 中。因此 persona 环境必须保持原 task ID，只允许上述 4,666 条记录 reset，
并以 `instruction_simple` 作为 Actor 可见请求；不能重编号，也不能对缺失字段做 LLM 补写。

正式项目因此采用以下继承边界：

- 继承：冻结商品目录、购物环境动作、完整任务目标、内嵌画像、个性化请求和可验证终局；
- 不作为正式梯度数据：原作者发布的 SFT/GRPO 轨迹；
- 自行采集：persona 多轮教师轨迹、训练/开发/评测划分和审计清单；
- 自行实现：`ask_user`、Shopper 交互 orchestrator 与训练/评测所需的统一状态传递。

## 8. 规划前仍需后置的决定

以下内容不能从论文直接照搬，也不在本次审阅中冻结：

- 正式画像 schema 的最小字段和跨字段一致性规则；
- 一条任务中哪些需求进入画像、开场请求或 Shopper 私有目标；
- 一次结构化提问的具体工具协议与用户模拟器状态机；
- Teacher API、数据量、验收率目标和人工抽检比例；
- Reward 分项、权重、评测任务规模和消融矩阵。

它们应在正式实现计划中按依赖顺序逐项对齐，而不是一次性凭直觉冻结。
