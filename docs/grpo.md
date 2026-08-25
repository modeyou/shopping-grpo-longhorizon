# GRPO 强化学习方案

数据目录身份与禁止混用规则见 `docs/data-layout.md`。

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

正式 GRPO 的规范目录为 `data/grpo/formal-v2`。原参考项目的 Reward v3 文件已隔离到 `data/reference/grpo-v1`，不得作为本项目正式 GRPO 输入。旧 `data/multiturn/tasks/grpo_train.jsonl` 只作为 5,000 个 task ID 的候选池，固定 SHA-256 为 `c5aecc973fb15bd6e37b90c7fa0c4c292573f3fe14aff5d1f27ce9eb3c446c5b`。旧 `grpo_validation.jsonl` 与当前 DEV-500 重叠 395/500，因此不进入正式流程。

### 正式 GRPO 数据冻结协议

当前协议固定如下：

- 从旧 5,000-task reservoir 重新排除正式 SFT train/validation、冻结 DEV-500 和 sealed Final-200。
- seed 固定为 `20260823`，先按 `sha256(seed:task_id)` 升序排列，再使用与 DEV-500 相同的 Reward v4 gold-purchase 审计逐项过滤不可达任务。
- 从审计通过的有序任务中取前 200 个作为 validation，随后 1,000 个作为 train；其余 Reward v4 可达任务记录为 unused，不可达任务单独写入 `reward-audit.jsonl`。
- 每个 task 生成一个经过审计的 gap opening；complete opening 从同一个 ShopSimulator 私有目标确定性派生。
- train 因此包含 1,000 gap + 1,000 complete，共 2,000 条 prompt；validation 包含 200 gap + 200 complete，共 400 条 prompt。
- 任务选择、排除集、opening、Environment v2.1 / Reward v4 manifest、Parquet 和全部 SHA-256 都写入 `data/grpo/formal-v2`。
- `train_grpo.py` 会强制验证 `shopping-multiturn-grpo-dataset-v2`、`status=accepted`、Reward v4、reachability audit、路径、哈希、行数和 task-disjoint 审计；不满足时拒绝启动。

第一步审计 reservoir 并冻结 active task，不调用 LLM，也不访问运行中的 ShopSimulator API：

~~~bash
export PYTHONPATH=./src
GRPO_PYTHON="${GRPO_PYTHON:-$(command -v python)}"

"$GRPO_PYTHON" scripts/select_multiturn_grpo_tasks.py \
  --reservoir data/multiturn/tasks/grpo_train.jsonl \
  --expected-reservoir-sha256 c5aecc973fb15bd6e37b90c7fa0c4c292573f3fe14aff5d1f27ce9eb3c446c5b \
  --products environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz \
  --environment-manifest data/environment-v4.json \
  --exclude sft-train=data/sft/formal-v2/train.jsonl \
  --exclude sft-validation=data/sft/formal-v2/validation.jsonl \
  --exclude dev500=data/multiturn/evaluation-dev-v2/tasks.jsonl \
  --exclude final200=data/evaluation/tasks.jsonl \
  --seed 20260823 \
  --train-count 1000 \
  --validation-count 200 \
  --output-dir data/grpo/formal-v2/selection
~~~

selector 会写出 selection schema v2、完整 `reward-audit.jsonl`、拒绝原因计数，以及商品数据的压缩/解压哈希。所有 JSON/JSONL artifact 固定使用 UTF-8 + LF，确保 Windows/Linux 逐字节一致。2026-08-23 的真实 5,000-task 审计得到 3,916 个 Reward v4 可达任务、1,084 个不可达任务；预期 train task SHA-256 为 `c8dedc11f4bc0f22e6b7776d80f9e9b17c82447c6389d1b6e1837d68803f3826`，validation task SHA-256 为 `85df9a00cdf4c8b56514eb8ce621266bb84f2cd52fede471ac8c7500e0deeb66`，reward audit SHA-256 为 `b75882737c67bbd3eb148627c7ea2d6928445862027da4b8f6bf2eacb920f575`。正式服务器运行必须独立复现这些计数与哈希后才能继续。

随后分别用 `generate_multiturn_tasks.py` 为 `train-tasks.jsonl` 和 `validation-tasks.jsonl` 生成冻结 gap openings。为保持与 DEV-500 一致，opening generator 固定使用 `qwen3.8-27b`、temperature 0、thinking 关闭，以及仓库当前 `OPENING_PROMPT_HASH`（DEV-500 为 `9fac425b31f44721e95d9bc1bb1a5d42da79ee305cbd5356001368de8ed0769b`）。再用 `freeze_multiturn_openings.py` 确定性派生 complete openings。最后仅通过以下命令发布正式 Parquet 与 accepted manifest：

~~~bash
"$GRPO_PYTHON" scripts/finalize_multiturn_grpo_dataset.py \
  --selection-manifest data/grpo/formal-v2/selection/selection-manifest.json \
  --train-gap-openings data/grpo/formal-v2/selection/train-gap-openings.jsonl \
  --train-complete-openings data/grpo/formal-v2/openings/train/complete_openings.jsonl \
  --validation-gap-openings data/grpo/formal-v2/selection/validation-gap-openings.jsonl \
  --validation-complete-openings data/grpo/formal-v2/openings/validation/complete_openings.jsonl \
  --environment-manifest data/environment-v4.json \
  --output-dir data/grpo/formal-v2
~~~

正式数据已经发布并由仓库验证器接受。唯一训练入口为 `data/grpo/formal-v2/manifest.json`，其中绑定：

- train：1,000 tasks / 2,000 rows，Parquet SHA-256 `38f41370264277c76c106f5970a7d0560f745ad77dcfee6bfc108fa9c1720f41`；
- validation：200 tasks / 400 rows，Parquet SHA-256 `575fe9b20ae6c24259144b05ad130fd032d260d171a68c95294566521fc7cae4`；
- opening generator：`qwen3.8-27b`、temperature 0、thinking 关闭、prompt SHA-256 `9fac425b31f44721e95d9bc1bb1a5d42da79ee305cbd5356001368de8ed0769b`；
- Reward v4 reachability：3,916/5,000 可达，正式选择的 1,200 个任务全部可达；
- SFT train/validation、DEV-500 与 sealed Final-200 overlap 均为 0。

这些命令定义来源与验收合同；task selection 与 Reward v4 可达性审计都是本地确定性操作。GRPO 训练期的 `ask_shopper` 仍使用独立 DeepSeek API，不能与 opening generator 混为同一模型来源。正式数据已经完成，但本节不表示已经启动训练。

## 默认优化配置

除 A/B 的 reward profile 外，下面参数保持完全一致：

| 设置 | 值 |
|---|---:|
| 算法 | GRPO |
| 每 prompt rollout 数 | 4 |
| temperature / top-p | 0.7 / 0.9 |
| train / validation batch | 2 / 2 |
| 学习率 | 1e-6 |
| warmup | 10 optimizer steps |
| scheduler | cosine，500-step 固定 horizon，最低学习率 1e-7 |
| data / PPO mini-batch seed | 20260823 / 20260823 |
| LoRA rank / alpha | 16 / 32 |
| memory kernels | fused kernels + Liger + remove-padding |
| 最大模型长度 | 24,576 |
| KL reward / KL loss | 关闭 / 关闭 |
| entropy | 只记录，不直接加奖励 |
| 动态采样最多生成批次 | 3 |
| 连续跳过更新上限 | 10 |

正式 A/B 把“本阶段停止位置”和“学习率调度总长度”分开：第一阶段都在 step 50 停止，但 optimizer scheduler 始终按 500 steps 计算。这样 step 50 是同一条长期训练曲线上的决策 checkpoint，而不是已经衰减到末端的短实验。veRL 0.8 原生会把两者绑定，因此正式运行必须先应用仓库的 scheduler-horizon 补丁；预检会验证补丁、10-step warmup、cosine、500-step horizon、Liger、SDPA、remove-padding 和零 DataLoader 子进程。

动态采样只保留同一 prompt 内训练效用存在差异的组。全常数组可以跳过；真实基础设施或 Reward 不可验证轨迹会使对应组无效。模型失败不是基础设施失败：A 中为 0，B 中为 -0.75。

每次训练都在输出目录写入 training_diagnostics.jsonl。generation_batch 记录公开动作、终局、原生/训练奖励、guard 与组选择；optimizer_step 记录 entropy、PPO KL、clip fraction、响应长度和有效组比例；skipped_update 记录没有推进 optimizer 的零信号尝试。


## 两个受控实验

这里的“第一轮”和“第二轮”表示两个从同一 SFT checkpoint 独立启动的实验，不是 epoch，也不是先后续训关系。

### A：原生 Reward v4

- 初始模型：同一个 checkpoint-325。
- reward profile：none。
- 模型自身未完成、非法动作上限和提前结束作为有效的 0 分失败样本参与组内比较。
- 环境/API 失败和 Reward 不可验证轨迹仍排除。
- A1 smoke 已完成一次 optimizer update；正式 A 从原始 checkpoint-325 独立运行到 step 50。
- 保存 step 25 和 step 50；step 50 运行冻结 validation。

### B：bounded-v1

- 从同一个 checkpoint-325 重新开始，不能接 A 的 checkpoint。
- 除 reward profile 外，数据、seed、rollout 参数、优化器参数和验证协议与 A 相同。
- Reward v4 的 reward_version、reward_type、reward_valid 和原生终局效用不变。
- 模型自身未完成失败记为 -0.75。
- 第一次提问不罚；第二次及以后每次 0.02。
- Shopper 拒绝每次 0.03，guard rejection 每次 0.005，重复动作每次 0.01。
- 有效终局的行为成本总上限为 0.10，不能盖过 Reward v4 的商品决策信号。
- B1 smoke 已完成一次 optimizer update；正式 B 同样从原始 checkpoint-325 独立运行到 step 50，并保存 step 25/50。

代码同时记录 native_terminal_utility 与实际 terminal_utility/total。前者用于回答“环境原生表现是否提升”，后者才是优化器和动态采样使用的训练信号。任何报告都必须同时展示两者，不能把 shaped reward 当作 Reward v4 原生结果。

### 选择与延长

A/B 在 step 50 使用同一冻结开发子集配对比较。优先按总 strict、三条件最差项、原生平均效用和基础设施有效性选择；shaped reward 只能作为训练诊断，不能作为最终选择指标。

胜者从自己的 step-50 checkpoint 恢复 optimizer 与 scheduler 状态，继续到 150–200 step，并在 100、150、200 做里程碑验证。不能重新 warmup，也不能把 step 50 当作新的 scheduler step 0。只有当 200 附近仍稳定上升且 KL、熵、长度、无效组率没有恶化，才考虑延长到 300；500 只是固定调度 horizon，不等于预先承诺一定执行 500 次更新。

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

## 正式 A/B 入口

当前服务器的 GPU 1 被其他服务占用。正式 GRPO 必须显式保留物理卡
`0,2,3,4`，不要写成进程内的逻辑编号 `0,1,2,3`，也不要手工启动 Ray：

~~~bash
export CUDA_VISIBLE_DEVICES=0,2,3,4
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
unset RAY_ADDRESS
~~~

运行时 hook 会把 Ray 分配的物理 ID 映射为 CUDA 掩码内的逻辑 ordinal：
`0→0、2→1、3→2、4→3`。正式 preflight 会拒绝未显式设置的掩码、重复或非四卡
掩码、Ray 自动重写掩码、可见卡数量不符，以及任一可见卡空闲显存低于 20 GiB
的启动。这样既不会误占物理 GPU 1，也不会把物理 GPU 4 错当成不存在的
进程内 `cuda:4`。

先在项目 Python 环境中安装两个经过固定源校验的 veRL 补丁。第二个补丁可逆地恢复第一个补丁并校验其 SHA-256，因此未知或被手改的 veRL 文件会被拒绝：

~~~bash
"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py
"$GRPO_PYTHON" scripts/apply_verl_scheduler_horizon_patch.py
"$GRPO_PYTHON" scripts/apply_verl_scheduler_horizon_patch.py --check
~~~

以下变量只存在于当前 shell/tmux，不写入仓库；API key 不会写入 run contract：

~~~bash
export PYTHONPATH=./src
export GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo/bin/python
export GRPO_MODEL="$PWD/outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325"
export SHOPPER_MODEL=deepseek-v4-flash-0731
export SHOPPER_BASE_URL='你的 OpenAI-compatible endpoint'
export SHOPPER_API_KEY='你的密钥'
export SWANLAB_API_KEY='你的 SwanLab 密钥'
~~~

A/B 必须使用两个全新且不同的输出目录。先分别预检，不会加载模型或启动训练：

~~~bash
"$GRPO_PYTHON" scripts/run_formal_grpo_ab.py \
  --arm a \
  --model "$GRPO_MODEL" \
  --output outputs/models/grpo-a-native-v4-step50 \
  --preflight-only

"$GRPO_PYTHON" scripts/run_formal_grpo_ab.py \
  --arm b \
  --model "$GRPO_MODEL" \
  --output outputs/models/grpo-b-bounded-v1-step50 \
  --preflight-only
~~~

确认预检输出同时包含 `scheduler_total_training_steps=500` 与 `stage_total_training_steps=50` 后，一次只启动一个 arm：

~~~bash
"$GRPO_PYTHON" scripts/run_formal_grpo_ab.py \
  --arm a \
  --model "$GRPO_MODEL" \
  --output outputs/models/grpo-a-native-v4-step50

# A 完成并释放 Ray/GPU 后，才启动 B。
"$GRPO_PYTHON" scripts/run_formal_grpo_ab.py \
  --arm b \
  --model "$GRPO_MODEL" \
  --output outputs/models/grpo-b-bounded-v1-step50
~~~

入口固定使用 4 张 GPU、formal-v2 数据、seed 20260823、SwanLab project `shopping-multiturn-agentic`、step 25/50 checkpoint、step 50 validation。A/B 唯一的训练语义差异是 `reward-profile=none` 与 `bounded-v1`；输出路径和 run name 只用于身份隔离。

选出胜者后，使用同一 arm、同一输出目录和其中的 step-50 checkpoint 显式恢复，并把阶段终点改为 200；入口会拒绝跨输出目录 checkpoint，也不会覆盖初始 `run_contract.json`：

~~~bash
"$GRPO_PYTHON" scripts/run_formal_grpo_ab.py \
  --arm a \
  --model "$GRPO_MODEL" \
  --output outputs/models/grpo-a-native-v4-step50 \
  --resume-from-checkpoint outputs/models/grpo-a-native-v4-step50/global_step_50 \
  --stage-end 200
~~~

上例的 `--arm a` 只是格式示例；若 B 胜出，必须同时替换 arm、输出目录和 checkpoint。恢复启动会另写 `run_contract.resume-global_step_50.json`，并由 veRL 加载 model、optimizer、scheduler 和 dataloader 状态。

## 通用调试入口

先检查解析结果：

~~~bash
export GRPO_PYTHON="$(command -v python)"
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
