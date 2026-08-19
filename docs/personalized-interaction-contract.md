# ShopSimulator 个性化多轮交互契约

- 日期：2026-08-19
- 状态：正式实现边界
- 目标：直接扩展 ShopSimulator，而不是离线重造任务、画像和 gold 澄清问题

## 1. 继承边界

ShopSimulator 内嵌数据已经同时提供：

- 完整购买目标 `instruction`，由环境和 Shopper 私有持有；
- 个性化初始请求 `instruction_simple`；
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

## 5. 已实现闭环

- `SHOPSIM_PERSONA_MODE=1` 在环境池创建时启用原生 persona 任务，客户端 reset 同时声明 persona 协议；
- Actor 只接收 `instruction_simple` 与删除 `__reasoning__` 后的画像；完整要求由环境经私有通道交给 Shopper，
  不进入 Actor 消息和轨迹的 `initial_result`；
- 个性化 Teacher 获得自然语言 `ask_user(question)` 工具，每条轨迹至多提问两次；
- 每次提问只触发一次 Shopper completion，回答写入同一消息轨迹，随后保留原商店页面状态继续购物；
- raw 轨迹记录全部问答，SFT 转换保留 `ask_user` schema 和问答消息，仍只接受 Reward v3 严格成功轨迹。

Reward 公式和最终评测设计继续后置，不在真实冒烟闭环通过前扩张范围。

## 6. 最小真实冒烟

23,421 条完整目标在 standard 与 persona 模式下保持相同 task ID；其中 4,666 条同时具有非空
`user_persona` 和 `instruction_simple`，可被 persona reset 使用，其余 ID 会被明确拒绝而不会重编号或
补写画像。`data/personalized/pilot_tasks.jsonl` 固定 5 个有效开发任务 ID，仅用于验证协议，不是正式
训练集，并且与 `data/evaluation/tasks.jsonl` 无重叠。先运行 3 条，检查问答和终局后再决定是否运行
剩余两条。

启动 persona 环境：

```bash
export SHOPSIM_PERSONA_MODE=1
bash scripts/start_environment.sh
```

在另一个终端使用一个 OpenAI-compatible endpoint；Shopper 默认复用 Teacher 的模型和 endpoint：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_API_KEY="your-key"

python scripts/collect_sft_data.py \
  --tasks data/personalized/pilot_tasks.jsonl \
  --output-dir outputs/personalized-interaction/pilot-01 \
  --personalized \
  --limit 3 \
  --model deepseek-v4-flash \
  --max-user-questions 2 \
  --max-steps 35 \
  --workers 1
```

每次 `ask_user` 会比普通购物轨迹多一次 Shopper API 调用。首轮只验收：服务无协议错误、私有字段不进入
Actor 消息、问答能被后续动作利用、轨迹可恢复保存、Reward v3 正常返回。

## 7. Pilot-01 协议验收

2026-08-19 使用 `deepseek-v4-flash` 对 task `0/1/3` 各采集一条真实轨迹。旧输出目录中残留的
task 5496 连接失败记录不属于本轮。该轮不是模型效果评测，只验收最小闭环并定位 Teacher 失败模式。

| task | 协议状态 | 提问 | Reward v3 终局 | 主要失败 |
|---:|---|---:|---|---|
| 0 | 有效 | 0 | `partial_alternative_purchase` | 核心功能满足，但选择了错误的儿童满天星规格 |
| 1 | 有效 | 0 | `repeat_loop` | 重访同一 ASIN，并重复选择同一 50W 规格 |
| 3 | 有效 | 1 | `repeat_loop` | 放弃强候选，迟到地确认已明确颜色，再重访旧商品 |

验收结论：persona reset、私有上下文隔离、Shopper 单次回答、`ask_user` 消息续接、Reward v3 和显式释放
全部通过，Actor 消息中 `__reasoning__` 泄漏为 0；严格成功为 0/3，三条均不得进入 SFT。task 3 的问题
没有新增信息，属于询问已知字段。

Pilot-02 前只做两项归因明确的修正：个性化 prompt 要求先从当前请求和画像形成需求清单、尽早判断是否
需要提问并及时购买强候选；个性化 rollout 守卫拒绝重新打开已核验 ASIN，以及在同一商品重复选择相同
规格。这些守卫调用不触碰环境，并按既有 blocked-call 规则从 SFT 消息中删除。原 standard rollout 行为
不变。

## 8. Pilot-02 Teacher 修正验收

同模型、同 task `0/1/3`、temperature 0 的 Pilot-02 使用新的需求清单 prompt 和个性化重复动作保护：

| task | 提问 | Reward v3 终局 | 新守卫触发 | 相对 Pilot-01 |
|---:|---:|---|---|---|
| 0 | 0 | `invalid_action_limit` | `product_already_inspected` | 过强守卫引发连续恢复失败 |
| 1 | 0 | `gold_purchase` | 无 | 从 `repeat_loop` 提升为严格成功 |
| 3 | 0 | `partial_alternative_purchase` | 无 | 从 `repeat_loop` 提升为有效购买 |

严格成功由 0/3 提升至 1/3，`repeat_loop` 由 2/3 降至 0/3，说明 prompt 对 task 1/3 的生存性和购买
决策有正向作用；task 1 是本轮唯一可进入 SFT 的轨迹。三条均未提问符合当前信息边界，因为原生画像已
覆盖简化请求缺失的主要约束。

`product_already_inspected` 对所有重访一刀切，会阻止比较多个候选后返回最佳商品，因此在 Pilot-02 后
撤回。个性化 prompt 改为允许有目的地返回最佳候选一次，只保留同一商品重复选择相同规格的
`option_already_selected` 守卫。原计划 Pilot-03 仅回归 task 0，实际运行时为降低偶然性仍采集了前三题。

## 9. Pilot-03 回归结果与数据判定

实际 Pilot-03 仍按 `--limit 3` 回归 task `0/1/3`，避免只观察单条轨迹：

| task | 提问 | Reward v3 终局 | SFT 判定 |
|---:|---:|---|---|
| 0 | 1 | `gold_purchase` | 拒绝：问题询问的 5 岁信息原本已在 Actor 可见画像中 |
| 1 | 0 | `partial_alternative_purchase` | 拒绝：非严格成功 |
| 3 | 0 | `partial_alternative_purchase` | 拒绝：非严格成功 |

task 0 证明重访保护撤回后，Teacher 能完成有效提问和严格购买闭环；但这条问题没有产生新信息，不能作为
澄清正样本。旧版收集器只检查 Reward，曾将它计为 1 条 accepted；从 SFT schema v2 起，完整画像条件下
出现任意 `ask_user` 都会以 `unexpected_question_on_full_persona` 拒绝。

Pilot-02 的 task 1 为 Gold，Pilot-03 同题为 Partial，尽管 temperature 为 0。这说明远端服务或策略执行仍
存在不可完全消除的运行方差；开发冒烟结果只用于协议和失败模式诊断，不作为正式显著性结论。到这里停止
继续调 prompt。

## 10. 确定性单事实遮蔽协议

原生完整画像适合训练“读取画像并直接购物”，但多数任务没有真实信息缺口，不能稳定产生有价值的
`ask_user`。为构造澄清正样本，增加 `shopsimulator-persona-mask-v1`：

1. 完整目标、原始画像、目标商品和 Reward 始终保留在 ShopSimulator/Shopper 私有侧；
2. reset 后仅对 Actor 可见的画像副本执行预登记的精确字符串遮蔽；
3. 遮蔽规则按 task ID、JSON 路径和原字符串冻结，路径或文本不完全匹配时立即失败；
4. 不调用 LLM 生成任务或 Critic，确保同一提交和 task ID 可完全复现；
5. raw 轨迹保存 mask ID、规则哈希和私有验收词，Actor 消息与 SFT 行不包含验收词。

首个低成本清单为 `data/personalized/masked_pilot_tasks.jsonl`，固定 task `0/1/2`，分别遮蔽儿童年龄、灯泡
功率和机器型号。三条都不与 Final-200 Clean 重叠。

SFT schema v2 按条件验收：

- `full_persona`：要求 Reward v3 Gold，且不得提问；
- `single_fact_mask`：要求 Reward v3 Gold、至少一次提问，并且 Shopper 回答命中该 mask 的预登记事实；
- 其余协议、隐私、终局和去重规则保持不变。

这套规则验证“模型发现缺口并通过用户回答恢复了事实”，严格成功则验证最终购物结果。3 条 masked pilot
仍只是数据管线冒烟，不用来报告模型能力；正式阶段再冻结更大任务集以及 matched no-ask 对照。
