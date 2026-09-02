# GRPO 正式长训练方案

数据身份规则见 `docs/data-layout.md`。本文定义当前多轮购物项目唯一支持的强化学习阶段：ShopSimulator Environment v2.1、Reward v4、observation v2、tool schema v2。sealed Final-200 不得用于 reward、checkpoint 或超参数选择。

## 已冻结决策

本轮不再进行 BPO、Reward A/B、bounded shaping、OPSD 或 off-policy replay。正式训练只有一条路线：

> 从 SFT checkpoint-325 merged model 出发，使用原作者 `upstream/main` 的 GRPO 算法和原生 `shopsimulator-reward-v4`，连续执行 500 个真实 optimizer updates。

用户选择不另做 1-step optimizer smoke。正式训练前仍必须执行不更新参数的确定性 preflight；第一个正式 update 同时承担运行时验收。preflight 失败时不得靠降低 batch、关闭 fused kernel 或跳过合同校验绕过。

原作者仓库在本地 remote 中名为 `upstream`，算法基准是：

- <https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/main/configs/grpo.yaml>
- <https://github.com/YYHDBL/shopping-grpo-longhorizon/blob/main/docs/grpo.md>

## 目标与起点

checkpoint-325 的 DEV-500 三条件 strict success 为：Gap+Ask 69.0%、Gap-NoAsk 52.8%、Complete+Ask 72.2%，合计 970/1500（64.67%）；Done 1490/1500，Reward-valid 1486/1500。RL 的首要目标是提高正确商品购买和 strict gold success，不是继续把主要精力放在工具格式上。

模型选择优先级：

1. 三条件 strict gold success、总 strict success 和最差条件。
2. 原生 Reward v4 终局效用、purchase、Done、Reward-valid。
3. Gap 提问增益、grounding；Complete 重复或无信息提问。
4. guard、上下文、基础设施、采样有效率和效率。

## 算法合同：复现上游 GRPO

除 Reward、初始模型、正式数据和硬件适配外，优化目标保持原作者配置：

| 设置 | 冻结值 |
|---|---:|
| advantage estimator | `grpo` |
| train batch / rollout n | 2 / 4 |
| temperature / top-p | 0.7 / 0.9 |
| PPO mini-batch / micro-batch per GPU | 2 / 1 |
| gradient accumulation | 2 |
| advantage std normalization | 关闭 |
| loss aggregation | `token-mean` |
| PPO clip low / high | 0.20 / 0.20 |
| KL in reward / KL loss | 关闭 / 关闭 |
| entropy coefficient | 0；保留 entropy 监控 |
| LoRA rank / alpha | 16 / 32 |
| learning rate | `1e-6` |
| warmup | 3%，即 15 optimizer steps |
| scheduler | veRL 原生 `constant`；warmup 后保持 `1e-6` |
| prompt / response / total 上限 | 4,096 / 20,480 / 24,576 tokens |
| optimizer updates | 500 |
| data / PPO seed | 20260823 / 20260823 |

必须移除此前 A/B 引入的 10-step warmup、cosine decay、最低学习率 1e-7 和 scheduler-horizon patch。恢复训练必须恢复 model、optimizer、scheduler、global step、dataloader 和 RNG，不能重新 warmup。

### 动态采样

- 每步严格需要 2 个有效 prompt groups，每组 4 条 rollout。
- `terminal_utility` 必须存在真实差异；全常数组不产生 advantage。
- 最多生成 3 个候选 batch，不允许降级成单 group 或缩小 batch。
- infra/Reward 不可验证轨迹必须排除；正常未完成、买错和策略失败仍按 Reward v4 训练。
- 连续 10 次无法形成完整更新时停止，skipped update 不推进 global step。

500 updates 至少对应 1,000 个有效 groups 和 4,000 条有效 rollout，不含被丢弃或追加生成的轨迹。

## Reward 与数据合同

唯一训练标量是 Reward v4 `terminal_utility`，不增加提问、重复动作、unfinished、guard、长度 shaping，也不改成 step-level reward。模型失败是有效训练结果；环境租约、API、parser、Reward schema 等基础设施无效结果必须排除。

唯一训练 manifest 是 `data/grpo/formal-v2/manifest.json`：

| artifact | tasks / rows | SHA-256 |
|---|---:|---|
| `multiturn-train.parquet` | 1,000 / 2,000 | `38f41370264277c76c106f5970a7d0560f745ad77dcfee6bfc108fa9c1720f41` |
| `multiturn-validation.parquet` | 200 / 400 | `575fe9b20ae6c24259144b05ad130fd032d260d171a68c95294566521fc7cae4` |

两份数据均为每个 task 的 gap/complete 各一行；全部任务通过 Reward v4 reachability，并与 SFT、DEV-500、Final-200 零重叠。opening 固定为 qwen3.8-27b、temperature 0、thinking 关闭、prompt hash `9fac425b31f44721e95d9bc1bb1a5d42da79ee305cbd5356001368de8ed0769b`。

在线 validation 始终使用完整冻结集：200 tasks，每个 task 保留 gap/complete 两行，共 400 rows。step 0 和之后每 50 个真实 optimizer updates 使用完全相同的 task IDs、Parquet、采样参数和 Reward 合同；不再创建 50-task 子集，也不在训练中切换验证数据。

veRL 0.8 不按 data.val_batch_size 切分 AgentLoop validation，而是把完整 400 rows 分发给 8 个 AgentLoopWorker；每个 worker 又并发执行自己的样本分片。为避免瞬时耗尽 ShopSimulator 的 20 个 lease 槽位，ShoppingToolAgentLoop 在取得环境之前使用 worker 进程内共享 semaphore，把每个 worker 的在途环境限制为 2，因此全局理论峰值为 16。限制从 session.start() 前持续到 session.close() 后，覆盖成功、异常与取消路径；它不缩减 validation、不改变 rollout、Reward 或 optimizer。preflight 必须验证 8 × 2 = 16 ≤ 20，run contract 必须记录 worker 数、每 worker 上限与预期槽位数。

## veRL 显存与算子适配

原作者使用单张 96 GiB GPU；本项目使用四张 24 GiB RTX 4090，冻结以下运行时适配：

正式 GRPO 必须使用独立 Python 环境（建议 `/home/gjx/.venvs/shopping-grpo-grpo`），不得复用已经安装 BPO entropy、snapshot、branch 或 advantage patch 的环境。环境建立后固定依赖版本并写出完整 inventory；此后只安装本文列出的 GRPO/common 补丁。

| 设置 | 冻结值 |
|---|---:|
| `use_fused_kernels` | `true` |
| `fused_kernel_options.impl_backend` | `torch` |
| `use_remove_padding` | `true` |
| `use_liger` | `true` |
| `liger-kernel` | `>=0.8.2`，版本写入 run contract |
| attention | `sdpa` |
| vLLM memory / max sequences | 0.45 / 8 |
| DataLoader workers | 0 |
| FSDP optimizer / reference param offload | 开启 |

三个开关必须通过 veRL 模型配置生效，项目入口不得再次直接 monkey-patch Hugging Face 模型：

1. Liger 只接管 RMSNorm、SwiGLU、RoPE 等模型内部算子，必须关闭它自己的 `fused_linear_cross_entropy`。
2. veRL fused PPO output head 分块计算选中 token 的 log-prob，避免完整 `sequence × vocabulary` logits 常驻显存。
3. remove-padding 使 actor forward、log-prob、response mask 与 fused output 的 no-padding 布局一致。

`fused=true + remove-padding=false` 不受支持，会使 no-padding log-probs 与 padding 恢复元数据不一致。历史 veRL issue 曾报告 Liger 与 fused 同开异常；当前官方实现已经分离模型内部 kernel 和 PPO output head，并明确声明二者兼容，见 <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>。因此不能只看 YAML 开关，必须验证实际调用路径。

### veRL 0.8.0 fused backward 回移植

固定 veRL 0.8.0 的 `FusedLinearForPPOFunction.backward` 有独立的静默梯度风险：custom forward 对非连续 hidden states 执行 `flatten` 可能产生副本，backward 若检查副本的 `requires_grad`，会漏掉原输入梯度。

必须将 BPO 环境已验证的修复抽取为 GRPO/common patch，以 `ctx.needs_input_grad` 决定 hidden-state 和 vocabulary-weight 梯度。补丁严格绑定 veRL 版本、原文件 SHA-256、唯一 marker 和可恢复备份，不得携带 BPO advantage、entropy、snapshot 或 branch patch。

正式 preflight 不产生 optimizer update，但必须：

- 用非连续 hidden states 跑真实 `FusedLinearForPPO` forward/backward；
- 断言输入梯度存在、finite 且绝对值和大于 0；
- 校验 Qwen3.5 torch fused forward、position IDs、remove-padding offsets、response mask；
- 校验 Liger fused CE 没有覆盖 veRL PPO output head；
- 校验最终 Hydra config 的 fused/Liger/remove-padding 均为 true。

用户选择不做 optimizer smoke，所以 preflight 不能证明真实 24K rollout 一定成功。正式训练第一个 update 是线上硬验收；出现 OOM、shape mismatch、零梯度、NaN/Inf 或 patch 不匹配时立即停止，不能在同一输出目录修改配置后续跑。

## GPU 与 Ray 合同

当可用物理 GPU 为 `0,2,3,4` 时：

```bash
export CUDA_VISIBLE_DEVICES=0,2,3,4
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
unset RAY_ADDRESS
```

Ray 由训练入口创建，不手工启动。worker hook 将物理 ID 映射为 mask 内逻辑 ordinal：`0,2,3,4 -> 0,1,2,3`。preflight 要求四张可见卡、无重复 ID、每张至少 20 GiB 空闲，并拒绝残留 GRPO/Ray 集群。

GPU 编号属于启动时事实，不永久假定 GPU 1 一定被占用。资源变化时重新审计，不能误杀任务范围外的进程。

## SwanLab 正式监控

正式训练强制使用：

```yaml
trainer:
  logger: [console, swanlab]
  project_name: shopping-multiturn-agentic
```

run name 为 `grpo-native-v4-500-s20260823-<timestamp>`，必须包含 GRPO、Reward、总步数和 seed。新训练以 experiment name 作为稳定 SwanLab run ID，并使用 `resume=allow`；断点续训必须显式传入原 run ID，并使用 `resume=must`，否则 launcher 拒绝启动，避免曲线静默分叉。`SWANLAB_MODE=online`，本地目录为 `<output>/swanlab`。API key 只从 tmux 环境变量读取，不写入命令、日志、run contract 或 Git。veRL `Tracking` 会把每次 `logger.log(data=metrics, step=global_step)` 同时发往 console 和 SwanLab。

### 必须进入 SwanLab 的指标

| 类别 | 关键指标 |
|---|---|
| optimizer | global step、policy loss、grad norm、learning rate、PPO KL、clip fraction、advantage min/mean/max、return min/mean/max |
| Reward v4 | terminal utility min/mean/max、native utility、strict、purchase、match score、evidence coverage、各 Reward 分量 |
| dynamic sampling | generated/accepted/filtered/constant/all-zero/invalid groups、generation/resample batches、generated trajectories、consecutive skips |
| trajectory | Done、平均步数、max-steps、overlong、repeat loop/action、Shopper 提问/拒绝、guard 拒绝、observation 截断、context compaction、model failure、partial purchase |
| infrastructure | infra-invalid、reward-unverifiable、Shopper/API/parser/guard 错误及原因 |
| length | prompt/response mean/max、总序列最大值、observation 原始/投影 token、投影截断次数 |
| performance | rollout、reward、old-log-prob、actor update、checkpoint、validation 耗时及 tokens/s |
| hardware | 各物理 GPU used/total/peak、utilization、CPU RAM、Ray object store |

`training_diagnostics.jsonl` 是逐 rollout 权威审计，至少包含 `generation_batch`、`optimizer_step`、`skipped_update`。SwanLab 只上传标量，不上传 omitted facts、Shopper 私有上下文、API key 或完整敏感轨迹。

veRL 原生键保持不变，同时增加稳定的 `optimization/*`、`length/*`、`performance/*` 别名以及三个健康量：`monitor/critical_metric_present_ratio`、`monitor/observed_metrics_all_finite`、`monitor/entropy_available`。本次 `actor.calculate_entropy=false` 且 entropy coefficient 为 0，为避免额外显存与计算，不为监控重新计算 entropy；因此正常值是 `monitor/entropy_available=0`，这不是指标缺失故障。

SwanLab 系统监控不能替代峰值记录。正式 launcher 每 30 秒把物理 GPU `index,memory.used,memory.total,utilization.gpu` 写入 `<output>/gpu_telemetry.csv`，训练结束生成每卡峰值摘要。

### 硬停止条件

出现以下任一情况必须停止并保留现场：

- 已启用的 loss、grad norm、advantage、return 或 KL 出现 NaN/Inf；
- update 没有有限非零梯度；
- CUDA OOM、invalid device ordinal、fused/unpadding shape mismatch；
- Reward/environment/schema/hash 不符；
- sampling-invalid trajectory 进入训练 group；
- 连续 10 次 skipped update；
- 连接既有 Ray 集群或可见 GPU 集合改变；
- global step、checkpoint、SwanLab step 不一致。

constant-group 比例、max-steps/overlong、response length、entropy 下降、KL/clip fraction 突增、API 重试和 validation 退化只告警，先调查而不自动当作模型失败。

## Checkpoint、validation 与模型选择

- `val_before_train=true`，记录 step-0 在线 validation。
- 每 25 optimizer steps 保存 checkpoint，全部保留。
- step 0 及每 50 steps 跑同一份完整 200-task/400-row validation。
- DEV-500 只评测该完整 validation 选出的少量候选。
- sealed Final-200 在 checkpoint 冻结后只运行一次。

在线 checkpoint 排序的主指标是 `validation/selection/balanced_strict_success_rate`，即 gap 与 complete 两类 `reward/strict_mean` 的等权平均，避免样本行数或模式比例改变排序。必须同时查看 `validation/gap/reward/strict_mean`、`validation/complete/reward/strict_mean`、purchase/done/invalid rate、`validation/gap/trajectory/shopper_ask_rate` 与 `validation/complete/trajectory/shopper_ask_rate`；后者分别反映缺信息时是否提问和信息完整时是否多问。每次验证的逐样本 generation 与 Reward extra info 保存到 `<output>/validation`，因此已经完成的新版本 validation 可以离线重聚合，无需再次调用模型。

保存频率 25 是故障恢复适配，不改变算法。500 steps 正常产生 step 25 至 step 500 共 20 份计划 checkpoint，全部保留。checkpoint 必须包含 model/LoRA、optimizer、scheduler、global step、dataloader 和 RNG；恢复后 global step 必须连续。

启动 preflight 必须按一份实际 checkpoint 的保守估算检查磁盘容量，并额外预留日志、validation、SwanLab 和临时写入空间；空间不足以保留20份完整 checkpoint 时拒绝启动，训练中不自动删除旧 checkpoint。

失败或强制退出时不得清理、覆盖或复用该运行目录。launcher 必须保留最后一个已经完整提交的 checkpoint、`run_contract.json`、训练日志、`training_diagnostics.jsonl`、SwanLab 本地目录和 GPU telemetry，并写出失败摘要。`SIGINT`/`SIGTERM` 应先请求训练进程有序停止；但 `SIGKILL`、整机掉电、CUDA 硬死或进程处于不可中断内核状态时不可能可靠保存内存中的当前 update，因此合同只保证最近一次已完成的 25-step checkpoint，最坏损失 24 个尚未保存的 optimizer updates。半写入或验收失败的 checkpoint 不得作为恢复点，`latest_checkpoint.json` 只指向最近一份通过完整性校验的 checkpoint。

## 复现与未来 off-policy 数据

run contract 必须记录 Git/工作区、模型与 adapter hash、全部 Parquet/manifest hash、环境/Reward/tool/parser 版本、算法/kernel/batch/seed/GPU/Ray/Shopper/SwanLab 配置，以及 Python、CUDA、driver、PyTorch、Transformers、veRL、vLLM、Liger 和 SwanLab 版本。还要记录补丁目标、原始/补丁 SHA-256、marker、GPU telemetry、20份计划 checkpoint 和 validation 清单。SwanLab 服务端 URL 只有初始化 run 后才产生，因此从训练日志提取到完成/失败摘要；最后完整 checkpoint 同样由运行期摘要和 `latest_checkpoint.json` 记录。

为未来独立 off-policy/replay 保存公开 prompt/token IDs、公开 trajectory、response mask、生成 checkpoint/global step、采样参数、behavior old log-prob、Reward v4 分量和环境版本；这些记录不改变本次 on-policy GRPO。

## 实施顺序

1. 用单一 native-v4 500-step launcher 替换 A/B、bounded-v1 和 cosine scheduler 正式入口。
2. 抽取通用 fused-PPO gradient patch，加入 veRL/Liger/remove-padding preflight 与测试。
3. 绑定完整 200-task/400-row validation，验证 step 0/50/.../500 使用同一份 Parquet。
4. 接入 SwanLab、训练诊断、GPU telemetry、checkpoint 完整性指针和完成/失败摘要。
5. 只做 dry-run/preflight，核对最终命令和 run contract，不产生 update。
6. 在服务器全新输出目录直接启动 500-step 正式训练。

未经用户单独授权，不启动训练、不合并模型，也不运行 sealed Final-200。

## 正式入口与执行顺序

代码只保留一个正式入口：`scripts/run_formal_grpo.py`。旧 A/B launcher、bounded-v1 正式入口和 scheduler-horizon patch 已删除。正式入口固定 native Reward v4、500 updates、完整 Validation-200、SwanLab 以及每 25 步 checkpoint；命令行不能把它降级成另一条算法路线。

正式入口同时强制 clean Git worktree；必须先提交并推送本次代码与正式数据 manifest，不能用未记录的工作区差异启动长训练。

在独立 GRPO Python 环境中先安装项目依赖，再安装并验证两个 GRPO 补丁：

```bash
export PYTHONPATH=./src
GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo-grpo/bin/python

"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py
"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py --check
"$GRPO_PYTHON" scripts/apply_verl_fused_ppo_grad_patch.py
"$GRPO_PYTHON" scripts/apply_verl_fused_ppo_grad_patch.py --check
```

不得在该环境安装 BPO entropy、snapshot、branch、advantage 或 BPO 聚合补丁。

启动前在同一个 tmux 中配置资源和密钥：

```bash
export CUDA_VISIBLE_DEVICES=0,2,3,4
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
unset RAY_ADDRESS

export SHOPPER_MODEL=deepseek-v4-flash-0731
export SHOPPER_BASE_URL=<OpenAI-compatible-Shopper-endpoint>
read -rsp 'SHOPPER_API_KEY: ' SHOPPER_API_KEY; echo
export SHOPPER_API_KEY
read -rsp 'SWANLAB_API_KEY: ' SWANLAB_API_KEY; echo
export SWANLAB_API_KEY
```

先做不更新参数的完整 preflight；通过后对同一空输出目录启动正式训练：

```bash
RUN_TAG=$(date +%Y%m%d-%H%M%S)
GRPO_OUT="$PWD/outputs/models/grpo-native-v4-500-s20260823-$RUN_TAG"
GRPO_LAUNCH_LOG="$PWD/outputs/grpo/logs/grpo-native-v4-500-s20260823-$RUN_TAG.log"
mkdir -p "$(dirname "$GRPO_LAUNCH_LOG")"

"$GRPO_PYTHON" scripts/run_formal_grpo.py \
  --model outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325 \
  --output "$GRPO_OUT" \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  --preflight-only

nohup "$GRPO_PYTHON" scripts/run_formal_grpo.py \
  --model outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325 \
  --output "$GRPO_OUT" \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  > "$GRPO_LAUNCH_LOG" 2>&1 < /dev/null &
```

训练父进程同时写出 `training.log`、`gpu_telemetry.csv`、`run_contract.json`、`training_diagnostics.jsonl`、`latest_checkpoint.json` 和带 UTC 时间戳的 `run_summary.*.json`。`latest_checkpoint.json` 只有在 veRL tracker 已提交且对应 `global_step_N/actor` 含实际文件时才更新。
