# 多轮澄清教师数据与 SFT 复习笔记

本文统一说明真实运行架构、教师数据采集、SFT 筛选和扩量策略。它是概念总览；具体命令参见
[`multiturn-clarification-data.md`](multiturn-clarification-data.md)，最初的研究假设参见
[`multiturn-clarification-design.md`](multiturn-clarification-design.md)。

## 1. 真实运行时到底由谁回答澄清问题

是的，当前原生多轮 rollout（教师采集、Baseline 和正式 Evaluation）有一个独立的 **LLM
Shopper** 回答 `ask_shopper`。ShopSimulator 本身不生成自然语言答案；它通过私有通道提供完整目标，
rollout harness 再调用 Shopper LLM 作答。SFT 是对已保存 messages 的离线训练，本身不调用 Shopper。

当前 veRL/GRPO adapter 仍是原参考项目的商店工具闭环：`configs/tools.json` 尚未注册
`ask_shopper`，训练工具适配器也尚未管理 Shopper 客户端和私有目标。因此“多轮 GRPO 在线调用
Shopper”是下一阶段待实现功能，不应写成当前能力。实现后仍沿用下面同一三方协议。

```text
冻结的模糊请求 ──公开──> Actor
                         │
                         ├─ 商店工具 ──> ShopSimulator ──> 页面观察、选项、Reward v3
                         │
                         └─ ask_shopper
                                │
ShopSimulator 私有完整目标 ──> LLM Shopper ──自然语言回答──> Actor
```

三方职责如下：

| 组件 | 持有什么 | 负责什么 | 不负责什么 |
|---|---|---|---|
| Actor | 公开请求、页面观察、Shopper 已公开的回答 | 判断是否需要问；搜索、浏览、选择并购买 | 不能看到完整 gold goal；不能让 Shopper 替它查商品目录 |
| ShopSimulator 环境 | 商品状态、完整任务目标、goal options、Reward v3 | reset/step、页面反馈、购买判定；通过私有字段把目标交给 Shopper | 不调用 LLM，不直接撰写澄清回答 |
| LLM Shopper | 环境私下提供的完整目标和历史问答 | 只按目标回答偏好、约束、兼容性、用途或预算；未知则说未知 | 不搜索商品、不报告未指定商品的材质/价格等目录事实、不替 Actor 决策 |

因此“Shopper 是环境的一部分”是协议层面的说法；工程实现上，它是 rollout harness 管理的独立
LLM 客户端。正式比较不同 Actor 时应固定同一个 Shopper 模型、prompt 和采样参数，否则结果同时混入
了 Shopper 质量变化。

## 2. 数据是怎样产生的

### 2.1 冻结开场

开场生成器只运行一次：从环境私有完整目标生成一个有意缺少 1–2 个购买关键事实的公开请求，并记录
`opening_audit.omitted_dimensions`、逐字的 `omitted_facts` 和 `source_goal_hash`。Actor 只看到
`initial_request`；审计字段只留在 raw 轨迹中，不进入训练 messages。

同一个 task ID 在 Baseline、教师采集、训练后评测和配对实验中复用同一开场。完整目标变化导致哈希
不一致时必须拒绝，而不是临时改用完整请求。

### 2.2 三类教师数据

#### A. 自主缺口正例（`autonomous-gap-v1`）

Actor 只看模糊开场，自主决定是否以及何时调用 `ask_shopper`；LLM Shopper 在线回答；Actor 随后继续
与真实 ShopSimulator 交互并购买。这类数据与正式推理分布最一致，能训练“发现缺口—提问—继续行动”，
但对教师模型的自主提问和长程购物能力要求最高，合格率可能较低。

#### B. 复合缺口正例（`composite-replay-v1`）

1. 标准购物教师看到完整目标，产生 Reward v3 gold-purchase 动作主干。
2. 独立问题生成器针对冻结的 `opening_audit` 生成一个问题。
3. LLM Shopper 依据被省略事实生成回答，并返回私有 `used_facts` 溯源。
4. 把问答放到主干之前，从新的多轮 reset 逐步重放所有动作。
5. 只有重放仍合法并再次 gold purchase 才接收。

它是可执行、可审计的合成示范，适合稳定补充高质量问答前缀；但它没有证明主干教师自主发现了缺口，
也没有证明后续动作在因果上使用了回答。因此必须保留 `composite-replay-v1` 标签，不能把它当作自主
澄清评测结果。

#### C. 完整请求零提问正例（`complete-no-ask-v1`）

Actor 看到完整请求，直接与 ShopSimulator 交互并成功购买，整条轨迹不得出现 `ask_shopper`。它用于
防止模型学成“所有任务开头都问一句”，并教授何时无需澄清。

`--teacher-first-ask` 仅用于诊断教师是否能在被强制首问后完成任务，不进入正式数据混合。

## 3. 从 raw 轨迹筛选为 SFT

采集时所有成功和失败尝试都追加到 `raw.jsonl`。在线调用同一个确定性验收函数，仅用于统计
`--target-accepted` 是否达到；失败轨迹仍被保留。派生阶段再次从 raw 全量重建 accepted/rejected、SFT、
train、validation 和 reject stats，并排除 evaluation task ID 与重复 task ID。

所有类型首先通过共同硬门槛：

- 无模型、工具或环境错误，状态和环境均真正结束；
- 轨迹实际执行 `Buy Now`；
- Reward v3 为 `gold_purchase`、`reward_valid=true`、`purchase_success=true`，终止原因一致；
- 每个 Actor 回合至多一个工具调用；动作必须和当时页面观察一致；
- held-out 评测任务不得进入训练集，同一 task ID 只保留一个接受样本。

然后按 `teacher_policy` 追加类型级门槛：

| 类型 | 已实现的额外验收 |
|---|---|
| 自主缺口正例 | 完整目标哈希一致；存在冻结缺口；至少问一次；每次接受的问答均有非空 `used_facts`，且逐字来自该开场的 `omitted_facts`；无被 guard 拦截的调用 |
| 复合缺口正例 | `replay_verified`；目标哈希一致；恰好一次问答；问题/回答有缺口溯源；无被拦截调用；重放再次 gold purchase |
| 完整请求零提问 | 标准交互模式；`shopper_questions` 为空；步骤中不存在 `ask_shopper`；最终 gold purchase |

SFT 转换只保留公开对话和工具轨迹，移除教师私有 reasoning、opening/clarification audit、guard 审计和原始
终局 Reward 文本；终局工具消息统一为“购买已完成。”。训练 loss 仍只落在 assistant 动作上。

### 单条验收能说明什么，不能说明什么

`used_facts` 能证明回答来自预先冻结的缺口，并能过滤零提问、无关提问或无依据回答；gold purchase 能证明
整条轨迹最后成功。两者合在一起仍不能严格证明“若不问就会失败”。真正的澄清因果收益必须在同一冻结
开场上做配对评测：

- ask-enabled：Actor 可以调用 `ask_shopper`；
- ask-disabled：隐藏或拒绝 `ask_shopper`，其他配置完全相同；
- 报告 gold success 差值，同时报告提问率、grounded-ask 率、问题数、步数和失败类型。

## 4. Qwen3.8-27B 教师与扩量原则

本地部署后，Qwen3.8-27B 的边际 API 成本接近零，所以可以使用 rejection sampling 生成远多于最终所需
的数据。只要验收规则可靠，增加尝试数通常会增加合格样本的**绝对数量**。但“生成越多”不自动等于
“数据越好”：同一任务的近重复成功轨迹、固定失败模式和教师偏差也会一起增长。

推荐扩量顺序：

1. 先用冻结的同一小批任务对 DeepSeek 与 Qwen 做 A/B pilot，Shopper 固定为同一个模型。
2. 分别统计三类数据的 raw 数、严格接受率、任务覆盖率、重复率、平均步数、提问率和 reject reasons。
3. 确认 Qwen 的 OpenAI-compatible tool calling、`qwen3_xml` 解析和 Reward v3 闭环稳定后再扩大任务覆盖。
4. 优先扩大不同 task ID 和缺口维度的覆盖，再增加同一 task 的 attempts；正式 SFT 仍按 task ID 去重。
5. A、B、C 三类先分目录、分标签保存；在验证集上比较混合比例后再冻结训练配方。
6. 保留所有 raw 和配置，使任何 accepted 集都能从 raw 确定性重建；不要只保存成功样本。

建议把“无限生成”改写为“预算允许范围内持续 rejection sampling，直到新增任务覆盖和验证收益趋于饱和”。
若合格率低，先依据 reject reasons 修复教师闭环或任务分层，而不是仅靠重复采样掩盖系统性失败。

当前 Qwen3.8/vLLM 接入还确认了两项兼容要求：购物 Actor 的单回合输出上限需高于原 DeepSeek
配置的 512，否则长 reasoning 可能在工具调用前被截断；Qwen chat template 只允许 system message
出现在消息序列开头，因此 Shopper 规则和私有事实必须合并到同一个 system message。两者都是服务协议
适配，不是 Reward 或购物能力失败。长度截断应记录为 `model_output_truncated`，HTTP 4xx 不应自动重试。
开场生成还必须包含显式 user turn，不能只发送 system-only 请求。

## 5. Reward v3 是否需要修改

当前结论是：**SFT 采集和正式评测不修改 Reward v3**。

Reward v3 回答的是“最终买到的商品是否满足环境私有完整目标”，它适合作为三类数据共同的终局硬门和
跨版本可比的正式评测标准。`ask_shopper` 不改变商店状态；缺口真实性、回答溯源和是否应当提问已经由
类型级验收与独立指标负责。把“问了问题”直接加进 Reward v3 会产生无条件提问的捷径，并破坏和原项目
结果的可比性。

多轮 GRPO 接通之后，可以实验一个**训练侧独立 shaping 信号**，但不能修改或重命名环境 Reward v3：

```text
R_train = R_v3_terminal + λg · R_grounded_clarification - λu · P_unnecessary_ask
```

其中 bonus 只能用于有冻结缺口、来源哈希一致且 `used_facts` 命中缺口的提问；完整请求上的提问可作为
unnecessary ask。权重必须小、封顶，并分别记录 `R_v3_terminal` 和 shaping 分量。是否启用要通过以下
消融决定，而不是现在预设：

1. SFT 后策略仅用原生 Reward v3 做多轮 GRPO；
2. 同配置加入小幅 clarification shaping；
3. 在同一 held-out 开场上比较 gold success、grounded-ask、unnecessary-ask 和 ask-enabled 相对
   ask-disabled 的成功增益。

如果原生 Reward v3 已能让 gold success 和有效澄清同步上升，就不增加 shaping；只有出现明确的信用
分配不足时才加入。正式 Evaluation 始终只报告未修改的 Reward v3，并把澄清指标分栏报告。

## 6. 当前结论

- 教师采集、Baseline 和 Evaluation 已是 Actor + ShopSimulator + 独立 LLM Shopper 的在线三方闭环；
  多轮 GRPO adapter 尚待接入同一闭环。
- 复合流程已经基本定型，但定位是可执行的 synthetic bootstrap，不是自主能力证明。
- 三类数据均需共同 gold gate 和各自的类型级验收；它们不应在 raw 阶段混为一种策略。
- Qwen3.8-27B 可用于低成本扩量，但应先完成小规模 A/B 和类型级统计，再按任务多样性扩张。
- 澄清的因果有效性最终由 ask-enabled / ask-disabled 配对评测回答，而不是由 SFT 接收规则替代。
- Reward v3 保持为稳定终局标准；若以后需要澄清信用分配，只在训练侧以可消融、可分解的小幅 shaping
  实验处理。


cd ~/shopping-grpo

QWEN38_ENV="/home/gjx/.venvs/qwen38"

if [ -e "$QWEN38_ENV" ]; then
  echo "Existing environment found: $QWEN38_ENV"
else
  uv venv "$QWEN38_ENV" --python 3.12
fi

source "$QWEN38_ENV/bin/activate"

echo "virtual environment: $VIRTUAL_ENV"
which python
python --version



uv pip install \
  --python /home/gjx/.venvs/qwen38/bin/python \
  -U "vllm>=0.17.0" \
  --torch-backend=auto



/home/gjx/.venvs/qwen38/bin/python - <<'PY'
import torch
import transformers
import vllm

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("vLLM:", vllm.__version__)
print("transformers:", transformers.__version__)
PY
