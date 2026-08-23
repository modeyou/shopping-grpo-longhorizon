# GRPO 强化学习方案

本文定义当前多轮购物项目唯一支持的强化学习阶段。运行时合同固定为 ShopSimulator Environment v2.1、Reward v4、observation v2 和 tool schema v2。最终正式测试集始终封存，不能用于 reward 设计、checkpoint 选择或超参数选择。

## 目标与起点

GRPO 从 SFT checkpoint-325 开始。它在冻结 DEV-500 三条件评测上的结果为：

| 条件 | Strict success |
|---|---:|
| Gap + Ask | 69.0% |
| Gap - Ask | 52.8% |
| Complete + Ask | 72.2% |
| 合计 | 970/1500（64.67%） |

Done 为 1490/1500、Reward-valid 为 1486/1500、guard rejection 为 32。由此可见，SFT 已基本解决工具协议崩溃；RL 的第一目标是提高正确商品购买率和 Reward v4 strict gold success，而不是继续把主要精力放在工具调用格式上。

指标优先级如下：

1. Gap+Ask、Gap-NoAsk、Complete+Ask 三条件 strict gold success 和总 strict success。
2. 原生 Reward v4 平均终局效用、purchase success、Done、Reward-valid。
3. Gap+Ask 相对 Gap-NoAsk 的增益，以及提问是否 grounded。
4. Complete 的重复提问、第二次无信息提问、问题上限和提问后无购物动作。
5. guard、上下文溢出、基础设施错误和效率指标。

Complete 第一次无信息确认不是硬失败，也不能压倒一次正确购买。项目需要减少机械确认，但不能以复现 checkpoint-406 的零问崩溃为代价。

## 实现边界与输入合同

veRL 固定使用 0.8.0，本仓库不复制 veRL 源码。项目代码只维护 AgentLoop/ShopSimulator adapter、有限兼容补丁和有界动态采样。环境补丁必须通过 SHA-256 与 marker 预检，未知 veRL 安装直接拒绝。

GRPO 输入必须满足：

- 初始策略是选定的 SFT checkpoint-325 合并模型，不能使用 final-2epoch 代替。
- train 与 validation 按 task ID 隔离，并与开发集、sealed 正式评测集零重叠。
- 环境固定为 Environment v2.1，Reward 固定为 v4。
- ask_shopper 使用独立的 OpenAI-compatible Shopper endpoint，不占用 35 个购物动作预算。
- 每条 rollout 有隔离的 Shopper history，澄清回答只通过公开 observation 投影进入 Actor 上下文。

正式 GRPO 数据不再直接沿用旧 `data/grpo/train.parquet`、`validation.parquet` 或其 Reward v3 metadata。旧 `data/multiturn/tasks/grpo_train.jsonl` 只作为 5,000 个 task ID 的候选池，固定 SHA-256 为 `c5aecc973fb15bd6e37b90c7fa0c4c292573f3fe14aff5d1f27ce9eb3c446c5b`。旧 `grpo_validation.jsonl` 与当前 DEV-500 重叠 395/500，因此不进入正式流程。

### 正式 GRPO 数据冻结协议

当前协议固定如下：

- 从旧 5,000-task reservoir 重新排除正式 SFT train/validation、冻结 DEV-500 和 sealed Final-200。
- seed 固定为 `20260823`，按 `sha256(seed:task_id)` 升序排列。
- 排序后前 200 个 task 作为 validation，随后 1,000 个 task 作为 train，其余仅记录为 unused。
- 每个 task 生成一个经过审计的 gap opening；complete opening 从同一个 ShopSimulator 私有目标确定性派生。
- train 因此包含 1,000 gap + 1,000 complete，共 2,000 条 prompt；validation 包含 200 gap + 200 complete，共 400 条 prompt。
- 任务选择、排除集、opening、Environment v2.1 / Reward v4 manifest、Parquet 和全部 SHA-256 都写入 `data/grpo/manifest.json`。
- `train_grpo.py` 会强制验证 `shopping-multiturn-grpo-dataset-v2`、`status=accepted`、Reward v4、路径、哈希、行数和 task-disjoint 审计；不满足时拒绝启动。

第一步只冻结 active task，不调用 LLM：

~~~bash
export PYTHONPATH=./src
GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo/bin/python

"$GRPO_PYTHON" scripts/select_multiturn_grpo_tasks.py \
  --reservoir data/multiturn/tasks/grpo_train.jsonl \
  --expected-reservoir-sha256 c5aecc973fb15bd6e37b90c7fa0c4c292573f3fe14aff5d1f27ce9eb3c446c5b \
  --exclude sft-train=outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/train.jsonl \
  --exclude sft-validation=outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/validation.jsonl \
  --exclude dev500=data/multiturn/evaluation-dev-v2/tasks.jsonl \
  --exclude final200=data/evaluation/tasks.jsonl \
  --seed 20260823 \
  --train-count 1000 \
  --validation-count 200 \
  --output-dir data/grpo/selection-v2
~~~

随后分别用 `generate_multiturn_tasks.py` 为 `train-tasks.jsonl` 和 `validation-tasks.jsonl` 生成冻结 gap openings，再用 `freeze_multiturn_openings.py` 派生 complete openings。最后仅通过以下命令发布正式 Parquet 与 accepted manifest：

~~~bash
"$GRPO_PYTHON" scripts/finalize_multiturn_grpo_dataset.py \
  --selection-manifest data/grpo/selection-v2/selection-manifest.json \
  --train-gap-openings data/grpo/selection-v2/train-gap-openings.jsonl \
  --train-complete-openings data/grpo/selection-v2/train-openings/complete_openings.jsonl \
  --validation-gap-openings data/grpo/selection-v2/validation-gap-openings.jsonl \
  --validation-complete-openings data/grpo/selection-v2/validation-openings/complete_openings.jsonl \
  --environment-manifest data/environment-v4.json \
  --output-dir data/grpo
~~~

这些命令定义来源与验收合同；只有 task selection 是本地确定性操作，opening 生成仍需单独确认 Shopper endpoint 后执行。本节不表示已经生成或启动训练。

## 默认优化配置

除 A/B 的 reward profile 外，下面参数保持完全一致：

| 设置 | 值 |
|---|---:|
| 算法 | GRPO |
| 每 prompt rollout 数 | 4 |
| temperature / top-p | 0.7 / 0.9 |
| train / validation batch | 2 / 2 |
| 学习率 | 1e-6 |
| data / PPO mini-batch seed | 20260823 / 20260823 |
| LoRA rank / alpha | 16 / 32 |
| 最大模型长度 | 24,576 |
| KL reward / KL loss | 关闭 / 关闭 |
| entropy | 只记录，不直接加奖励 |
| 动态采样最多生成批次 | 3 |
| 连续跳过更新上限 | 10 |

动态采样只保留同一 prompt 内训练效用存在差异的组。全常数组可以跳过；真实基础设施或 Reward 不可验证轨迹会使对应组无效。模型失败不是基础设施失败：A 中为 0，B 中为 -0.75。

每次训练都在输出目录写入 training_diagnostics.jsonl。generation_batch 记录公开动作、终局、原生/训练奖励、guard 与组选择；optimizer_step 记录 entropy、PPO KL、clip fraction、响应长度和有效组比例；skipped_update 记录没有推进 optimizer 的零信号尝试。


## 两个受控实验

这里的“第一轮”和“第二轮”表示两个从同一 SFT checkpoint 独立启动的实验，不是 epoch，也不是先后续训关系。

### A：原生 Reward v4

- 初始模型：同一个 checkpoint-325。
- reward profile：none。
- 模型自身未完成、非法动作上限和提前结束作为有效的 0 分失败样本参与组内比较。
- 环境/API 失败和 Reward 不可验证轨迹仍排除。
- 先做 1-step smoke，再运行到 step 10 和 step 50。
- 保存 step 0、25、50；step 10 只做快速诊断，不要求永久保留模型。

### B：bounded-v1

- 从同一个 checkpoint-325 重新开始，不能接 A 的 checkpoint。
- 除 reward profile 外，数据、seed、rollout 参数、优化器参数和验证协议与 A 相同。
- Reward v4 的 reward_version、reward_type、reward_valid 和原生终局效用不变。
- 模型自身未完成失败记为 -0.75。
- 第一次提问不罚；第二次及以后每次 0.02。
- Shopper 拒绝每次 0.03，guard rejection 每次 0.005，重复动作每次 0.01。
- 有效终局的行为成本总上限为 0.10，不能盖过 Reward v4 的商品决策信号。
- 同样执行 1/10/50 step，并保存 step 0、25、50。

代码同时记录 native_terminal_utility 与实际 terminal_utility/total。前者用于回答“环境原生表现是否提升”，后者才是优化器和动态采样使用的训练信号。任何报告都必须同时展示两者，不能把 shaped reward 当作 Reward v4 原生结果。

### 选择与延长

A/B 在 step 50 使用同一冻结开发子集配对比较。优先按总 strict、三条件最差项、原生平均效用和基础设施有效性选择；shaped reward 只能作为训练诊断，不能作为最终选择指标。

胜者继续到 150–200 step，并在 100、150、200 做里程碑验证。只有当 200 附近仍稳定上升且 KL、熵、长度、无效组率没有恶化，才考虑延长到 300；当前不预先承诺 500 step。

## Reward 边界

Reward v4 是环境合同，不修改为 step-level reward。训练期 bounded shaping 只提供有限的 credit assignment，不能改变 gold_purchase 的严格成功定义。

不采用逐步奖励的原因是：

- 搜索、点击和提问是否正确依赖后续商品与终局，单步启发式容易奖励错误路径。
- 当前核心目标是正确购买，而不是更短或更像模板的轨迹。
- 强行为惩罚可能让模型学会不提问或提前停止。

未来如需 step-level 信号，只能以不改变终局排序的 potential-based 或严格有界辅助项做独立消融，并继续报告原生 Reward v4。

## 五面板评测如何使用

完整评测框架的五个面板分别回答不同问题：

| 面板 | 内容 | 用途 |
|---|---|---|
| A Reward and terminal | Reward v4、终局、strict/purchase、效用 | 主要模型选择 |
| B Requirement rubric | 冻结 rubric 下各需求满足情况 | 判断买对了哪些约束、错在哪里 |
| C Trajectory quality | Judge 对计划、证据、恢复、终止的评估 | 分析长程策略质量 |
| D Clarification | 是否提问、grounding、次数、时机与无效提问 | 分析多轮澄清价值 |
| E Deterministic | guard、上下文、步骤、错误和基础设施 | 排除协议与运行故障 |

Rubric 不是另一个 reward。它把任务要求冻结成可核对维度；Judge 根据公开轨迹和 rubric 评估不能由 Reward v4 单个标量解释的过程质量。Judge 结果用于诊断和论文分析，不直接参与 GRPO 优化，也不替代 strict success。

使用节奏：

- 1-step smoke：只检查 Reward/terminal 与 deterministic 合同。
- step 10：跑便宜的 A、D、E 面板和 reward 审计，发现明显方向错误。
- step 25/50：运行同一冻结开发子集；step 50 对 A/B 运行完整五面板。
- 胜者 100/150/200：完整五面板；以配对任务差异分析退化和提升。
- 最终模型选择完成后：只执行一次 sealed 正式评测，并运行完整五面板。

因此，之前的快速 DEV sweep 没有每次调用 rubric/judge 是有意的成本控制，不代表这两个面板无用。

## BPO 决策门

当前只完成 GRPO，不在本阶段承诺完整 BPO。GRPO 运行期间可以在独立分支研究 BPO，但不得改变 A/B 的冻结协议。

到 step 50 后再决定：

- 如果同一 prompt 内存在稳定的成功/失败轨迹对，而绝对 reward 标度或组归一化限制明显，再考虑完整 BPO。
- 如果主要问题仍是探索不到成功轨迹，BPO 没有足够偏好对，优先改 rollout 探索和数据覆盖。
- 如果 bounded-v1 已显著提高 strict 且没有行为退化，不为“方法更新”额外增加框架风险。

## 复现要求

每个运行必须绑定：

- Git commit 和工作区状态。
- checkpoint-325 的模型/adapter 哈希。
- train、validation parquet 与环境 manifest 哈希。
- Reward profile 和全部系数。
- seed、rollout n、采样参数、batch、学习率、LoRA、最大长度。
- Shopper model、endpoint 标识与 API 可用性；不得记录密钥。
- run_contract.json、training_diagnostics.jsonl、训练日志、checkpoint 列表和验证输出。

正式比较要求 A/B 使用相同 task IDs、相同 opening、相同 data/PPO seed 和相同评测参数。异步 GPU rollout 不承诺 bitwise 完全确定，因此必须使用配对任务与重复审计判断差异。任何基础设施无效率异常的运行不能参与模型优劣结论。

## 入口

先检查解析结果：

~~~bash
bash scripts/grpo.sh --reward-profile none --dry-run
~~~

只做完整预检：

~~~bash
bash scripts/grpo.sh --reward-profile none --preflight-only
~~~

A 使用：

~~~bash
bash scripts/grpo.sh --reward-profile none
~~~

B 使用：

~~~bash
bash scripts/grpo.sh --reward-profile bounded-v1
~~~

高级 veRL 覆盖参数仍放在双横线之后。启动脚本会把 reward profile 写入公开审计输出，每条 trajectory 也会记录 shaping_profile、native_terminal_utility、实际训练效用和各惩罚分量。

未经用户单独授权，不启动训练、不合并模型，也不运行最终 200-task 评测。
