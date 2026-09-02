# 从 SFT checkpoint-200 重启 GRPO 的改进方案

状态：已确认，按本文实施。本文定义的新实验取代 `docs/grpo.md` 中以 SFT checkpoint-325 为起点的旧运行方案；历史运行资产和结论保持不变。

## 1. 结论与实验目标

新 GRPO run 从 **SFT checkpoint-200 的 merged model** 全新启动，不从旧 GRPO step-175 续训。算法继续使用原生 Reward v4 GRPO；本次只针对已经观测到的严重 `ask_shopper` 拒绝循环做训练样本隔离，并启用最新版监控与结束验收。

核心目标仍然是提高正确购买率和 `gold_purchase` strict success，而不是单独优化提问率。`ask_shopper` 的调用本身不加分，也不引入提问 shaping。

本次实验要回答两个问题：

1. 较早的 SFT checkpoint-200 是否在已经掌握购物协议的同时，保留了比 checkpoint-325 更适合 GRPO 探索的策略支持。
2. 排除严重拒绝循环后，GRPO 是否能在不被超长病态轨迹污染梯度的情况下，提高相对 SFT-200 的购买成功率。

## 2. 已证实事实与尚未验证的假设

### 已证实事实

- SFT checkpoint-200 在冻结 DEV-500×3 上的总 strict 为 `945/1500 = 63.00%`；checkpoint-325 为 `970/1500 = 64.67%`。因此 SFT-200 的端点分数不是现有 checkpoint 中最优。
- 旧 GRPO run 截至 optimizer step 175 的诊断中，1400 条已训练轨迹有 52 条出现 `ask_shopper` 拒绝，涉及 36/350 个训练 group。
- 这些轨迹共出现 969 次拒绝；其中 25 条长循环轨迹贡献 887 次，占 91.5%。单条轨迹最多拒绝 36 次。
- 52 条带拒绝轨迹中有 25 条获得正 advantage；至少 4 条最终 `gold_purchase` 的轨迹会把前面的非法提问 token 一并正向强化。
- 当前 loss 使用 `token-mean`。因此只要 advantage 非零，35–36 次拒绝形成的长 response 就可能占用显著 token 权重。
- 旧 SFT-325 与 GRPO-100 的 Final-200×3 对比为基本持平、轻微退化，差异不显著；该结果不能证明 GRPO 有提升。

### 尚未验证的假设

- checkpoint-200 比 checkpoint-325 有更高的 rollout 多样性或更低的 zero-variance group 比例。
- 拒绝循环是 Final 结果未提升的重要原因。现有数据证明它是实际训练污染风险，但没有逐轨迹 response-mask token 数，不能量化它对总梯度的精确贡献。
- 从 checkpoint-200 重训会优于 checkpoint-325。该选择是新的 RL learnability 假设，不是由现有端点评测直接支持的结论。

因此，必须保留 step-0 validation，并把新 GRPO checkpoint 首先与它自己的 SFT-200 基线配对比较。

## 3. 冻结设计

### 3.1 初始化模型

- 起点：正式 SFT run 的 `checkpoint-200`。
- GRPO 输入必须是 checkpoint-200 与原 base 合并后的完整 bf16 Hugging Face 模型，不能把 LoRA adapter 目录直接作为 `--model`。
- 建议服务器路径：

  ```text
  /home/gjx/shopping-grpo/outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-200
  ```

  启动前必须核实该路径确实是 merged artifact，并保存 `merge_manifest.json`、模型权重 SHA-256、原 SFT adapter checkpoint 和 base model 身份。路径名称不能代替内容校验。
- 旧 GRPO step-175 只作为诊断与 provenance 资产保留，不复制其 LoRA、optimizer、scheduler、dataloader 或 RNG 状态。

### 3.2 Harness 与提问规则

Harness 保持现状：

- 每条轨迹最多成功调用 `ask_shopper` 两次。
- `ask_shopper` 不计入 ShopSimulator 的 35 个购物 action step。
- 空问题、重复问题、超过两次以及开始购物后的提问继续由训练 Harness 拒绝。
- 拒绝不立即终止 episode，也不增加连续拒绝上限；轨迹仍受最多 40 assistant turns 限制。
- 不为了与评测 Harness 形式一致而修改训练 Harness。

### 3.3 严重拒绝循环的训练侧隔离

冻结规则：

```text
若同一条 rollout 的 shopper_rejections >= 3：
    将该 rollout 标记为 policy pathology / optimizer-ineligible；
    丢弃它所属的整个 prompt uid group；
    由现有动态采样继续生成新的完整 group；
    不对原 prompt 做单条缺口补采。
```

选择阈值 `>=3` 的依据：旧 run 中该阈值涉及 32 条轨迹、27 个训练 group，却覆盖 943/969（97.3%）次拒绝。它集中隔离真正的循环，同时不因一次或两次偶发拒绝就删除整个 group。

该规则只改变 optimizer eligibility，不改变：

- ShopSimulator Reward v4 数值；
- `reward_valid` 和基础设施有效性定义；
- episode 的实际终止方式；
- validation/final evaluation 的计分方式。

实现时不重写动态采样控制流。训练与 validation 共用的 AgentLoop 只输出中性的 pathology 标志和明确原因；训练侧 group selector 再通过现有 `sampling_invalid -> whole group drop -> next generation batch` 通道触发过滤。必须保留独立字段，例如：

```json
{
  "policy_pathology": true,
  "policy_pathology_reason": "shopper_rejections_gte_3",
  "shopper_rejections": 35,
  "infrastructure_invalid": false,
  "reward_valid": true
}
```

不能把这种模型行为伪装成 API、ShopSimulator 或 Reward 基础设施错误。validation 仍保留并计分这种轨迹；只有训练 selector 将它解释为 optimizer-ineligible。即使训练选择通道名仍叫 `sampling_invalid`，公开诊断和结束报告也必须单独统计 `policy_pathology`。

过滤后的 group 不进入 advantage、old-log-prob actor update 或 response-token loss。被过滤的原始 rollout 必须保留在 `training_diagnostics.jsonl` 中。

### 3.4 动态采样保持不变

- 每个 prompt group 固定 `n=4`。
- 每个 optimizer update 需要 2 个完整、reward-varying、optimizer-eligible 的 group。
- 最多生成 3 个候选 batch；不足时跳过该 update，不缩成单 group 或不完整 group。
- constant-reward、现有 infrastructure/reward-invalid 以及新的 rejection-pathology group 都使用现有整组过滤路径。
- 不实现 pending group、局部补采、跨 generation batch 拼同一 prompt 或动态修改 `rollout.n`。
- 被跳过的 update 不推进 global step。

这会牺牲一部分生成效率，但实现简单，且不会让病态长序列进入 optimizer。

### 3.5 Reward 与优化器

保持原生 Reward v4 配置：

| 设置 | 冻结值 |
|---|---:|
| Reward | `shopsimulator-reward-v4` |
| reward shaping | `none` |
| advantage estimator | `grpo` |
| rollout `n` | 4 |
| train batch | 2 prompts |
| temperature / top-p | `0.7 / 0.9` |
| advantage std normalization | 关闭 |
| loss aggregation | `token-mean` |
| PPO clip low/high | `0.20 / 0.20` |
| KL reward / KL loss | 关闭 / 关闭 |
| entropy coefficient | 0 |
| LoRA rank / alpha | `16 / 32` |
| learning rate | `1e-6` |
| scheduler | constant |
| warmup | 3%，即 15 个真实 optimizer updates |
| total optimizer updates | 500 |
| seed | 20260823 |

本次不同时引入 Clip-Higher、sequence-mean loss、提问奖励、拒绝惩罚、长度 shaping、BPO 或 turn-level credit assignment。否则无法判断 SFT-200 起点和循环隔离分别是否有用。

`actor.calculate_entropy=true` 只用于监控；`entropy_coeff=0`，不把 entropy 加入目标函数。

### 3.6 与早期“行为修复 + pilot”建议的取舍

早期诊断方案是在后续决策冻结前提出的探索性方案。本轮保留其中不改变学习目标的
修复：拒绝循环诊断、原始/训练样本可观测性、正确的 entropy 指标别名和独立
SwanLab project。以下建议不并入本轮：

- 不在第三次拒绝时终止 episode，也不新增 `shopper_rejection_loop` 终止原因；
- 不增加 Complete 首问成本、逐次拒绝惩罚或固定负 Reward；
- 不先做 no-ask SFT/DPO 行为修复；
- 不强制每个 update 接受一个 Gap group 和一个 Complete group；
- 不把两次 50-step pilot 设为启动 500-step 的前置门槛。

原因不是这些方向一定无效，而是它们会同时改变 Harness、训练 Reward、初始化模型或
采样分布，无法与本轮“换用 SFT-200 起点并隔离严重拒绝循环”的主要变量分开归因。
Gap/Complete 的 raw 与 optimizer 分布仍应记录；若新 run 仍无提升，再把模式平衡或
条件化 no-ask 修复作为下一轮单变量实验。

## 4. 数据、验证与实验身份

- Train：`data/grpo/formal-v2/multiturn-train.parquet`，1000 tasks / 2000 rows。
- Validation：`data/grpo/formal-v2/multiturn-validation.parquet`，200 tasks / 400 rows。
- Manifest：`data/grpo/formal-v2/manifest.json`。
- 训练只使用 gap 和 complete 两种 opening；本次不新增 Gap-NoAsk 采样。
- `val_before_train=true`，得到 SFT-200 在同一 GRPO validation 协议下的 step-0 基线。
- 每 50 个真实 optimizer updates 跑同一份完整 validation；validation 不过滤拒绝循环，模型产生什么就按什么计分。
- 每 25 steps 保存完整 checkpoint，并保留全部 step 25–500 checkpoint。

新 run 必须使用新的 output directory、experiment name 和 SwanLab run ID，不能 resume 旧 run，也不能复用旧 run 的曲线身份。run name 应明确包含 `sft200`，例如：

```text
grpo-sft200-native-v4-500-s20260823-<timestamp>
```

SwanLab 使用独立 project `shopping-multiturn-grpo-sft200`。旧 run 继续保留在
`shopping-multiturn-agentic`，不得移动或重命名；新 project 由正式训练首次
`swanlab.init` 时创建。

由于本方案是在查看旧 Final-200 结果后制定，同一 Final-200 对新方案不再是完全盲测，只能称为 held-out benchmark。若需要对新算法作严格盲测声明，必须另行冻结从未查看的新测试集。

## 5. 监控合同

不设置基于训练质量曲线的自动 health gate。以下指标持续记录，训练结束统一验收；只有 NaN/Inf、OOM、schema/hash 错误、设备错误、无法形成任何 optimizer update 等运行时完整性故障才中止。

### 5.1 原始生成与过滤

必须区分 generated 与 trained，不能只看进入 optimizer 后的干净 batch：

- `policy_pathology/trajectory_count`
- `policy_pathology/group_count`
- `policy_pathology/group_rate`
- `policy_pathology/shopper_rejections_total`
- `policy_pathology/shopper_rejections_max`
- `policy_pathology/dropped_groups`
- `trajectory/shopper_question_mean`
- `trajectory/shopper_rejection_mean`
- `trajectory/gap_count` / `trajectory/gap_rate`（optimizer batch）
- `trajectory/complete_count` / `trajectory/complete_rate`（optimizer batch）
- `group/generated`
- `group/trained`
- `group/effective_ratio`
- `group/all_equal_ratio`
- `group/sampling_invalid`
- `group/resample_batches`
- `shopping_dynamic_sampling/skipped_updates_total`

若 SwanLab 暂时只能显示进入 optimizer 后的 trajectory 聚合，`training_diagnostics.jsonl` 中的原始 generation batch 仍是权威来源；结束报告必须从原始记录重新聚合，不能用过滤后的均值代替。
结束摘要还必须分别输出 generated、optimizer 和 pathology group 的
Gap/Complete/other 计数，防止动态过滤造成的训练模式偏斜被总体均值隐藏。

### 5.2 优化器与长度

- policy loss、grad norm、learning rate；
- PPO approx KL、lower/upper clip fraction；
- advantage 和 return 的 min/mean/max；
- entropy；
- prompt/response/total token mean/max；
- 每个 update 的有效 response token 数；
- rollout、old-log-prob、actor update、validation 和 checkpoint 耗时。

veRL 0.8 实际上报的 entropy key 是 `actor/entropy_loss`；监控别名
`optimization/entropy` 必须读取该字段。`entropy_coeff=0` 仍保持不变，名称中的
`loss` 不代表它被加入优化目标。

重点检查长轨迹是否仍获得异常大的有效 token 占比。虽然 `>=3` 拒绝的 group 应当完全排除，但一次或两次拒绝仍可能进入训练。

### 5.3 Reward 与购物行为

- terminal utility、native utility、strict、purchase success；
- gold/wrong/partial/unfinished/repeat-loop 等 termination type；
- Done、Reward-valid、infrastructure-invalid；
- guard rejection、repeat action、max-steps/overlong；
- gap/complete 分面的 ask rate、shopper rejection rate 和 strict success。

训练中不因为单项曲线越过经验阈值而自动停止。异常只记录并调查，最终与 step-0 和历史 run 一起验收。

## 6. Checkpoint 选择与效果判定

### 6.1 在线选择

主选择指标保持：

```text
validation/selection/balanced_strict_success_rate
```

它是 gap strict 与 complete strict 的等权平均。不能仅凭 Gap ask rate、`G+−G−`、平均 Reward 或训练 reward 选择 checkpoint。

同时报告：

- gap strict、complete strict；
- purchase、Done、Reward-valid；
- gap ask rate、complete ask rate；
- shopper rejection 与循环；
- checkpoint 相对 step-0 SFT-200 的逐题 gain/loss。

best checkpoint 由冻结 validation 选择，不默认使用 step-500。

### 6.2 正确的比较关系

主比较：

```text
新 GRPO best checkpoint vs 新 run 的 SFT-200 step-0
```

次比较：

```text
新 GRPO best checkpoint vs SFT-325
新 GRPO best checkpoint vs 旧 GRPO-100
```

后两项只能回答最终系统孰优，不能单独归因于 GRPO 改进，因为初始化 checkpoint 已改变。

对 strict success 使用同题配对转换和 exact McNemar 检验；同时报告绝对差值、gain/loss 数和置信区间。没有显著差异时，按“持平/证据不足”描述，不能仅凭点估计宣称提升。

## 7. 实施完成条件

实施代码在启动训练前必须满足：

1. 单元测试证明 `shopper_rejections=0/1/2` 不触发 pathology，`>=3` 触发。
2. 任一 rollout 触发 pathology 时，整个四轨迹 uid group 都不出现在最终 optimizer batch。
3. 被过滤 group 的 Reward v4、`reward_valid` 与 `infrastructure_invalid` 原值保持不变，并有独立 exclusion reason。
4. 过滤发生后，现有动态采样生成新的完整 group；不产生三轨迹 group。
5. 被过滤轨迹仍完整写入 diagnostics，但不出现在 actor update 的 token/advantage 对齐数据中。
6. validation 不应用该过滤规则。
7. run contract 记录 SFT-200 merged model、Git commit、配置、数据/模型 hash、过滤阈值和补丁身份。
8. 全部相关测试、veRL patch `--check` 和 `--preflight-only` 实际通过。

实现与验证完成后，仍需先在服务器核验 merged model、运行环境和 run contract；不停止或删除历史 run，也不在本地自动启动训练。

## 8. 训练结束验收

结束报告只做一次统一验收，不做质量分级：

- 实际提交的 optimizer updates 数、跳过次数和 wall-clock；
- checkpoint 25–500 的完整性及 latest pointer；
- 全程 loss/grad/advantage/return/KL/entropy 是否 finite；
- raw generated、filtered、trained group/trajectory/token 数；
- `>=3` 拒绝循环的轨迹数、group 数、总拒绝数、最大值及随 step 趋势；
- 确认 pathology group 进入 optimizer 的数量严格为 0；
- step-0/50/.../500 validation 完整性和主/分面曲线；
- validation 选出的唯一 best checkpoint 及冻结依据；
- best checkpoint 相对 SFT-200 step-0 的配对统计；
- 与 SFT-325、旧 GRPO-100 的次级对比；
- 模型、数据、代码、配置、环境和 checkpoint provenance。

工程完成与学习成功必须分开陈述：训练完整结束但 strict 没有改善，仍然是一次有效且可解释的实验，只是不能称为 GRPO 提升。

## 9. 启动前命令草案

以下只作为审核后的服务器执行模板。实际路径、Git commit、GPU 和 merged model 必须先重新核验。

```bash
cd /home/gjx/shopping-grpo-grpo

export PYTHONPATH=./src
export CUDA_VISIBLE_DEVICES=0,2,3,4
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
unset RAY_ADDRESS

GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo-grpo/bin/python
SFT200_MODEL=/home/gjx/shopping-grpo/outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-200
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="grpo-sft200-native-v4-500-s20260823-$RUN_TAG"
RUN_OUT="$PWD/outputs/models/$RUN_NAME"

"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py --check
"$GRPO_PYTHON" scripts/apply_verl_fused_ppo_grad_patch.py --check

"$GRPO_PYTHON" scripts/run_formal_grpo.py \
  --model "$SFT200_MODEL" \
  --output "$RUN_OUT" \
  --experiment-name "$RUN_NAME" \
  --seed 20260823 \
  --preflight-only
```

preflight、模型身份和最终 run contract 经过人工核对后，才能移除 `--preflight-only` 启动正式训练。
