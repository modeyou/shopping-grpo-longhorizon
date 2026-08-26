# BPO 训练原始信息交接包

本文用于把本项目 BPO 实现、正式训练和评测证据交给另一个对话独立复核。本文优先记录原始路径、冻结参数、运行输出和已发生异常；除“待复核问题”外，不预设 BPO 配置正确或错误，也不要求接手者接受已有分析结论。

## 0. 本次交接的核心分析目标

用户希望接手者重点回答的不是“BPO 有没有跑完”，而是：

> BPO 已经完成 200 个 optimizer steps，梯度、checkpoint、Reward v4 和评测链路均有审计证据，
> 但冻结 dev500 的完整购买成功率没有超过 SFT checkpoint-325。问题可能出在哪里，下一轮应如何
> 改进，才能实现 RL 在 SFT 基础上继续显著提高最终购买成功率的项目目标？

本项目的主要优化目标是完整购物严格成功率，而不是单独降低 XML 错误、减少提问、改善工具格式
或让训练 loss 更平滑。辅助指标只有在解释或支持严格购买成功率变化时才具有决策意义。

接手者需要把下列四类解释分开，不要用“训练稳定”替代“训练有效”：

1. **实现或梯度语义问题**：tree、分叉状态、reward、advantage、mask、loss aggregation 或 optimizer
   虽能运行，但实际优化的 token/行为不是预期目标；
2. **训练目标或数据问题**：Reward v4 的连续分数、组内相对优势、有效树过滤、训练任务分布或
   最大熵分叉与 strict gold purchase 不对齐；
3. **超参数与优化强度问题**：LR、scheduler、KL/reference 设置、PPO clip、M/K、batch、训练预算
   或 LoRA 容量使更新过弱、过强或高方差；
4. **评测与统计问题**：当前差值可能处于随机波动范围，但至少没有证据表明 BPO 已产生项目要求的
   显著提升；需要说明怎样用最小成本区分“确实无效”和“评测噪声”。

最终建议必须落到可证伪的最小对照实验：每个实验只改变一个主要因素，写明预期现象、成功/停止
条件、需要的 steps/trees/returns、GPU 时间和 Shopper API 成本。不能只建议“再多跑一些步”。

## 1. 仓库与边界

- 分支：`feat/bpo2`
- 本文编写时 HEAD：`d049d3f`
- 服务器仓库：`~/shopping-grpo`
- Windows 仓库：`D:\shopping-grpo-longhorizon`
- 当前运行契约：ShopSimulator Environment v2.1、Reward v4、observation v2、tool schema v2
- 严格成功：`reward_type == "gold_purchase" and reward_valid == true`
- 正式最终 200 题没有用于训练，也没有用于本轮 BPO 选型。
- `outputs/` 下的模型、日志、SwanLab 缓存和分析 JSON 是服务器生成物，通常不提交 Git。
- 本文编写时工作区存在用户自己的未跟踪目录 `tmp/`，不要删除或提交。

## 2. 正式训练输入

### 2.1 模型与数据

```text
起点模型：outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325
训练数据：data/grpo/formal-v2/multiturn-train.parquet
验证数据：data/grpo/formal-v2/multiturn-validation.parquet
数据清单：data/grpo/formal-v2/manifest.json
数据环境清单：data/environment-v4.json
BPO 运行时环境清单：data/environment-bpo-v1.json
```

formal-v2 数据的已冻结规模：训练 1000 个任务、2000 行（gap/complete 各 1000）；验证 200 个任务、400 行（gap/complete 各 200）。

### 2.2 正式运行标识与服务器产物

```text
运行标签：bpo-native-v4-step200-r1600-seed20260823-20260826-113750-r3
随机种子：20260823
SwanLab run ID：4cmh0p3k
SwanLab 页面：https://swanlab.cn/@mode/shopping-multiturn-agentic/runs/4cmh0p3k/chart
运行目录：outputs/models/bpo-native-v4-step200-r1600-seed20260823-20260826-113750-r3
最终 checkpoint：<运行目录>/global_step_200
导出模型：outputs/models/bpo-native-v4-step200-r1600-seed20260823-export
```

服务器上应优先保留并检查：

```text
<运行目录>/run_contract.json
<运行目录>/step0_validation_contract.json
<运行目录>/training_diagnostics.jsonl
<运行目录>/latest_checkpointed_iteration.txt
<运行目录>/global_step_*/
outputs/bpo/logs/<同一运行标签>.log
outputs/bpo/step0-validation-cache/
```

## 3. 当前代码中的冻结参数

以下值抄自当前 `configs/bpo.yaml` 和 `configs/bpo_agent_loop.yaml`，接手者应再次读取文件确认，不能只信本文。

| 项目 | 当前值 |
|---|---:|
| `data.train_batch_size` | 2 |
| `actor.ppo_mini_batch_size` | 2 |
| `actor.ppo_micro_batch_size_per_gpu` | 1 |
| sibling count `K` | 4 |
| branch count | 1 |
| 每步有效树 | 2 |
| 每步 sibling returns | 8 |
| 正式 optimizer steps | 200 |
| 正式有效树预算 | 400 |
| 正式有效 return 预算 | 1600 |
| learning rate | `1e-6` |
| warmup | 10 steps |
| scheduler | cosine |
| scheduler horizon | 500 |
| min LR ratio | 0.1 |
| PPO clip | 0.2 |
| `upstream_lambda` | 0.95 |
| rollout temperature / top-p | 0.7 / 0.9 |
| validation temperature / top-p | 0.0 / 1.0 |
| LoRA rank / alpha | 16 / 32 |
| `use_kl_loss` | false |
| `use_kl_in_reward` | false |
| entropy coefficient | 0.0 |
| actor `calculate_entropy` | false |
| BPO 分叉熵 | 独立 vLLM 首 token 全词表探针 |
| `use_remove_padding` | true |
| `use_fused_kernels` | true |
| `use_liger` | true |
| attention | SDPA |
| gradient checkpointing | true |
| FSDP parameter/optimizer offload | true |
| rollout `n` | 4 |
| vLLM memory utilization | 0.45 |
| vLLM max sequences | 8 |
| GPU | 4 张 |
| max prompt / response / sequence | 4096 / 20480 / 24576 |
| AgentLoop workers | 2 |
| AgentLoop max steps | 35 |
| Shopper questions 上限 | 2 |
| dynamic sampling target/minimum | 2 / 2 |
| dynamic sampling full batch | true |
| soft warning / max candidate batches | 10 / 30 |
| checkpoint | step 10；之后每 25 steps |
| validation | step 0、10；之后每 50 steps |

正式 Reward profile 是 `none`，即原生 Reward v4，没有 bounded shaping。

## 4. BPO 实现数据流索引

接手者应从这些文件直接追踪实现，不要把 `docs/bpo.md` 当作实现证据：

```text
configs/bpo.yaml
configs/bpo_agent_loop.yaml
scripts/bpo.sh
scripts/train_bpo.py
scripts/check_bpo_runtime.py
scripts/apply_verl_bpo_patch.py
scripts/apply_verl_dynamic_sampling_patch.py
src/shopping_grpo/training/bpo/agent_loop.py
src/shopping_grpo/training/bpo/branching.py
src/shopping_grpo/training/bpo/session.py
src/shopping_grpo/training/bpo/advantage.py
src/shopping_grpo/training/bpo/runtime.py
src/shopping_grpo/training/bpo/step0_validation.py
src/shopping_grpo/training/bpo/entropy_patch.py
src/shopping_grpo/training/bpo/xml_tool_parser_patch.py
src/shopping_grpo/training/bpo/fused_ppo_grad_patch.py
src/shopping_grpo/training/grpo/dynamic_sampling.py
```

现有实现声称的流程是：在线生成 backbone；在 assistant action 边界保存快照并计算下一 token 精确熵；排除最终 action 后选最大熵边界；从同一快照生成 backbone continuation 加 3 个 clone；K=4 sibling 用 leave-one-out return baseline；分叉前 action 乘 `0.95^d`；最后进入 PPO clip LoRA actor update。该段是实现目标描述，仍需用源码和真实诊断验证。

## 5. 正式运行原始审计数字

`scripts/audit_bpo_formal_run.py` 已接受该运行，记录如下：

| 项目 | 数值 |
|---|---:|
| optimizer steps | 200/200 |
| 有效树 | 400/400 |
| 有效 sibling returns | 1600/1600 |
| 跳过 optimizer update | 0 |
| 生成的 backbone 数 | 1,732 |
| 生成的 branch rollout 数 | 5,196 |
| 总 rollout 数 | 6,928 |
| 生成回复 token | 1,478,990 |
| 环境 transition | 30,255 |
| Shopper API 调用 | 880 |
| wall time | 8:57:06 |

由上述计数直接计算：400 棵有效树来自 1,732 棵候选 backbone，有效树/候选 backbone 比例约为 23.09%。这只是计数比值，不解释造成无效候选的原因。

## 6. 训练期间验证曲线

SwanLab 正确指标命名空间为 `val-shopping/summary/*`。云端因收尾异常只保存到 step 199；step 200 数字来自服务器最终日志中的 `Final validation metrics`。

| step | strict | purchase | mean reward |
|---:|---:|---:|---:|
| 0 | 0.6875 | 0.6925 | 0.647794 |
| 10 | 0.6975 | 0.7050 | 0.663147 |
| 50 | 0.6850 | 0.6925 | 0.648229 |
| 100 | 0.6875 | 0.6950 | 0.652001 |
| 150 | 0.6975 | 0.7050 | 0.673723 |
| 200 | 0.6975 | 0.7050 | 0.6772444853 |

step 200 其他验证原始数字：

```text
done rate: 0.9900
reward valid rate: 0.9950
mean steps: 4.57
sampling invalid rate: 0.0050
infrastructure invalid rate: 0.0000
reward unverifiable rate: 0.0050
Shopper question rate: 1.0000
```

训练稳定性与采样耗时的 SwanLab 导出数字：

```text
actor/ppo_kl mean: 0.00586
actor/ppo_kl max: 0.01410
gradient norm mean: 0.2567
gradient norm max: 0.8397
actor loss range: [-0.321, 0.387]
clip fraction max: 0.00562
candidate batches mean: 4.33
candidate batches median: 4
candidate batches p95: 9
candidate batches max: 13
seconds to full batch mean: 159.9
seconds to full batch median: 135.8
seconds to full batch p95: 243.6
seconds to full batch max: 956
slow-batch warnings: 6
unique branch actions mean: 2.62 / 4
unique tool sequences mean: 2.33 / 4
sibling return std mean: 0.415
sibling return range mean: 0.934
```

本地 SwanLab 导出文件（若仍保留）：

```text
outputs/analysis/swanlab-bpo-step200-4cmh0p3k/swanlab-history.json
outputs/analysis/swanlab-bpo-step200-4cmh0p3k/swanlab-analysis.md
```

导出代码：

```text
scripts/export_swanlab_run_metrics.py
src/shopping_grpo/evaluation/swanlab_history.py
```

## 7. 冻结 dev500 三面板结果

评测使用 `outputs/evaluation/checkpoint-sweep-dev500-v1/assets`，每个条件 500 题，三个条件共 1500 条；`final_evaluation_used=false`。

| 模型 | gap ask | gap no-ask | complete | total | gap gain | unnecessary ask | mean reward | done | reward valid | guards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BPO step 200 | 0.690 | 0.526 | 0.720 | 0.645 | +0.164 | 0.922 | 0.5983 | 1487/1500 | 1483/1500 | 49 |
| SFT checkpoint-325 | 0.690 | 0.528 | 0.722 | 0.647 | +0.162 | 0.938 | 0.6010 | 1490/1500 | 1486/1500 | 32 |

BPO 减 SFT：

```text
gap ask: +0.0000
gap no-ask: -0.0020
complete: -0.0020
total: -0.0013
gap gain: +0.0020
unnecessary ask: -0.0160
mean reward: -0.0027
```

产物路径：

```text
outputs/evaluation/bpo-native-v4-step200-r1600-dev500-v1/results/evaluation_results.json
outputs/analysis/bpo-step200-vs-sft325-dev500/paired-diagnostics.json
```

## 8. dev500 配对翻转原始记录汇总

### 8.1 条件级计数

```text
gap-ask-enabled: gains=9, losses=9, net=0, guard_delta=+9, question_delta=-2, reward_delta=-0.0021
gap-ask-disabled: gains=3, losses=4, net=-1, guard_delta=+3, question_delta=0, reward_delta=-0.0066
complete-ask-enabled: gains=6, losses=7, net=-1, guard_delta=+5, question_delta=-9, reward_delta=+0.0005
```

### 8.2 reward type 转移

```text
gap-ask-enabled
7 loss gold_purchase -> partial_alternative_purchase
5 gain partial_alternative_purchase -> gold_purchase
3 gain repeat_loop -> gold_purchase
1 loss gold_purchase -> wrong_purchase
1 loss gold_purchase -> repeat_loop
1 gain unknown -> gold_purchase

gap-ask-disabled
3 gain partial_alternative_purchase -> gold_purchase
2 loss gold_purchase -> partial_alternative_purchase
2 loss gold_purchase -> wrong_purchase

complete-ask-enabled
4 loss gold_purchase -> unknown
3 loss gold_purchase -> partial_alternative_purchase
3 gain partial_alternative_purchase -> gold_purchase
3 gain unknown -> gold_purchase
```

### 8.3 task IDs

```text
gap-ask-enabled gains: 4582, 2865, 11943, 14882, 20506, 9311, 22118, 232, 18073
gap-ask-enabled losses: 6655, 22557, 8007, 9422, 13735, 13300, 18055, 852, 11107

gap-ask-disabled gains: 7790, 10500, 6895
gap-ask-disabled losses: 20563, 2149, 9807, 2068

complete-ask-enabled gains: 16382, 5100, 18946, 7361, 5559, 13604
complete-ask-enabled losses: 10159, 8268, 11281, 23225, 6777, 4282, 14535
```

## 9. 已发生的错误和补丁历史

以下是运行过程中实际遇到过的错误，不代表当前 HEAD 仍会复现：

1. `manual_seed expected a long, but got str`：seed Hydra 类型问题。
2. actor 长序列完整词表 `log_softmax` OOM：随后启用 fused kernels，并增加 fused PPO input-gradient backport。
3. XML tool-call 参数缺少 `>` 导致解析失败：增加 tolerant XML parser patch。
4. 稀疏物理 GPU `CUDA_VISIBLE_DEVICES=0,2,3,4` 与 Ray accelerator ID 映射不一致：增加 physical-to-logical 映射预检。
5. ShopSimulator 曾错误启动成 Reward v3：更新代码后必须确认服务实际返回 Environment v2.1 / Reward v4。
6. 合成 CPU preflight diagnostics 曾与真实训练 diagnostics 混淆：正式进程启动前删除 launcher 创建的合成诊断文件。
7. 诊断审计曾直接阻断正常训练：之后改成非阻断观测；正式验收由独立审计脚本执行。
8. 正式运行结束时出现：

```text
RuntimeError: cannot join current thread
```

堆栈位于 veRL `Tracking.__del__` 调用 SwanLab `finish()`；错误发生前已经出现 200/200、step 200 checkpoint 和最终验证日志。随后代码增加显式、幂等的 tracking finish。云端 run 状态仍显示 `CRASHED`，云端历史只到 step 199。

相关近期提交：

```text
c001b46 feat(bpo): freeze step-zero validation and enrich run metrics
758467b feat(bpo): diagnose actor mask and backward path
f268b68 fix: support real one-step BPO diagnostics
f414865 fix: restore fused PPO input gradients
f134962 fix: make BPO optimizer audit non-blocking
c805541 fix: finalize BPO tracking and document step 200
9838409 feat: export SwanLab training history
ae4c1d5 fix: report SwanLab validation namespace
e334c0f feat: evaluate exported RL checkpoints on frozen dev500
29dafbd feat: audit paired BPO dev500 flips
d049d3f fix: validate dev500 reward contract by condition
```

## 10. 已纠正的旧审计参数

`docs/bpo-independent-audit.md` 曾错误沿用一次早期实验方案：

```text
target=2 / minimum=1 / max_batches=3
```

该参数不适用于本次正式训练。正式运行已经由 `run_contract.json`、200 条
`optimizer_step` 诊断以及 `scripts/audit_bpo_formal_run.py` 确认使用：

```text
target=2 / minimum=2 / require_full_batch=true / soft_warning=10 / max_batches=30
```

因此，本次正式 BPO 的确定事实是：每个 optimizer step 必须凑满 2 棵有效树，只有完整的
8 个 sibling returns 才能更新；第 10 个候选 batch 告警，最多尝试 30 个 candidate batches，
不允许用 1 棵树降级更新。旧审计文档中的相应章节已同步修正。

## 11. 待独立复核的问题（不是既定结论）

1. 200 步无明显 dev500 增益究竟是算法本身、超参数、训练数据/Reward 信号，还是实现偏差。
2. `M=2` 棵有效树/step、`K=4` sibling 的 batch、mask、loss aggregation 和梯度尺度是否正确。
3. `loss_agg_mode=token-mean` 是否适合不同 continuation 长度的 BPO tree。
4. `upstream_lambda=0.95` 与组内 LOO advantage 的前缀梯度是否按预期抵消或产生有效信号。
5. 当前没有 KL reward、没有 actor KL loss、没有 reference policy；这与论文主实验不同，是否需要小规模对照。
6. 学习率 `1e-6`、warmup 10、cosine horizon 500，但正式只跑 200 步；是否导致有效更新过保守。
7. 每步严格凑满 2 棵有效树、最多 30 个候选 batch，是否值得保留；候选有效率低的主要原因是什么。
8. 最大首 token 熵是否总能选到与购物决策相关的边界；分叉位置分布、工具类型和失败类型之间是否存在偏差。
9. Reward v4 的连续奖励、strict success 与 BPO sibling 相对优势是否对齐。
10. `complete-ask-enabled` 中 `gold_purchase -> unknown` 的 4 条轨迹、`gap-ask-disabled` 中 `gold_purchase -> wrong_purchase` 的 2 条轨迹应逐条审计。
11. 训练 validation 全程 Shopper question rate 为 100%，需要核对验证任务组成、指标定义和策略行为。
12. 是否只调整 BPO 配置做小规模可归因实验，还是继续延长训练；不得同时改变算法、Reward、LR、K/M 和数据后再把差异归因给单一因素。

## 12. 可直接复制给新对话的提示词

```text
请作为独立审查者分析 shopping-grpo 项目的 BPO 正式训练配置与后续改进，不要先接受原对话的结论。

核心问题是：这次 BPO 已真实完成 200 个 optimizer steps、400 棵有效树和 1600 个 sibling
returns，但冻结 dev500 的 strict purchase success 没有超过 SFT checkpoint-325。RL 的项目目标
是在 SFT 基础上显著提高完整购买成功率；请诊断为什么本次训练没有实现这个目标。不要把“代码能跑完、
loss 有限、训练稳定”当作“训练目标有效”的证明。

先读取：
1. AGENTS.md
2. docs/bpo-training-handoff.md
3. docs/bpo.md
4. docs/bpo-formal-results.md
5. docs/bpo-diagnostics.md
6. docs/bpo-independent-audit.md（动态采样章节已更新为正式参数）
7. configs/bpo.yaml
8. configs/bpo_agent_loop.yaml

然后直接审查这些实现：
- scripts/bpo.sh
- scripts/train_bpo.py
- scripts/check_bpo_runtime.py
- scripts/audit_bpo_formal_run.py
- src/shopping_grpo/training/bpo/
- src/shopping_grpo/training/grpo/dynamic_sampling.py
- BPO、dynamic sampling、fused gradient、XML parser 相关 tests 和 veRL patch 安装脚本

正式运行是 full-bpo-v1，从 SFT checkpoint-325 独立开始，Reward v4 profile=none，seed=20260823，200 optimizer steps，每步 2 棵有效树，每棵 K=4，共 400 棵有效树/1600 returns。请把服务器的 run_contract.json、training_diagnostics.jsonl、完整日志、SwanLab history 和 dev500 paired-diagnostics.json 视为一手证据；文档声明只是索引。

任务：
1. 判断 tree 构造、entropy 分叉、snapshot clone、LOO advantage、mask、token-mean PPO loss、fused/remove-padding/Liger 路径和 optimizer update 是否实现正确。
2. 核对正式 run 的 `run_contract.json` 与当前 YAML 是否一致；动态采样正式值已经确定为
   `target=2/minimum=2/require_full_batch=true/max_batches=30`，不要再把早期的
   `minimum=1/max_batches=3` 当作候选解释。
3. 从“实现/梯度语义、训练目标/数据、超参数/优化强度、评测噪声”四类原因分别分析为什么
   strict purchase success 没有提升，并给出按可能性和影响排序的假设。
4. 用原始训练曲线、候选树计数、分叉位置/多样性、reward 方差和 dev500 配对翻转支撑判断；
   把“有代码证据”“有运行证据”“仅假设”分开写。
5. 重点评估 Reward v4 与 strict gold purchase 的一致性、有效树过滤是否产生选择偏差、最大熵
   分叉是否命中购买关键决策、LOO/upstream credit assignment、token-mean loss、LR、scheduler
   horizon、KL/reference、M/K、full-batch dynamic sampling 和 200-step 预算。
6. 给出最少数量、可归因、预算可控的下一轮 BPO 对照实验。每个实验只改变一个主要变量，写明
   预期现象、成功阈值、提前停止条件以及 steps/trees/returns/GPU-hour/API 成本。不能只建议延长训练。
7. 明确说明什么结果可以证明 BPO 值得继续，什么结果应停止 BPO 并把资源转向 GRPO。
8. 不要启动训练、合并模型或使用正式 final200，除非我之后明确授权。

报告格式：
- 原始证据清单
- 配置/实现 findings（按严重程度）
- 已验证与未验证
- 对“实现错误、超参数不合适、训练信号不合适、样本预算不足”四类解释分别给证据
- 推荐的最小实验矩阵、停止条件、预计 return/GPU-hour/API 成本
- 需要我从服务器补充的精确文件或命令
```

## 13. 其他相关文档与工具

```text
docs/bpo.md                         BPO 设计与运行手册
docs/bpo-formal-results.md          正式运行与 step 200 结果
docs/bpo-diagnostics.md             actor mask/loss/backward 诊断说明
docs/bpo-independent-audit.md       只读代码审计指南，动态采样章节已按正式参数修正
docs/data-layout.md                 正式数据与历史数据目录边界
docs/grpo.md                        GRPO 对照路线
scripts/audit_bpo_formal_run.py     正式运行验收
scripts/export_swanlab_run_metrics.py SwanLab 历史导出
scripts/audit_bpo_dev500_diagnostics.py dev500 配对翻转审计
scripts/run_standalone_checkpoint_evaluation.py 导出 checkpoint 的独立评测
scripts/audit_standalone_checkpoint_evaluation.py 独立评测验收
```
