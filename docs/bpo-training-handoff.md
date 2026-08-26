# BPO 训练分析交接

本文供另一个对话独立分析：为什么 BPO 已完成正式训练，却没有在 SFT 基础上提高最终购买成功率。

第一轮只分析可能原因、证据强弱和需要补充的信息，与用户讨论后再决定实验。不要直接修改代码、启动训练、合并模型或使用正式 final200。

## 1. 核心问题

项目的 RL 目标是提高完整购物严格成功率：

```text
reward_type == "gold_purchase" and reward_valid == true
```

BPO 已完成 200 个 optimizer steps，训练链路、checkpoint 和 Reward v4 均通过审计，但冻结 dev500 的 strict success 没有超过 SFT checkpoint-325。

不要预设原因只有固定几类。实现语义、Reward、信用分配、长轨迹决策、探索与分叉、数据分布、
优化强度、环境/Shopper 随机性和评测统计都只是初始线索；它们可能相互作用，也可能存在当前
文档没有想到的 BPO 或 agent RL 特有原因。接手者可以重新建立原因图谱，不必沿用本文分类。

“训练能跑完、loss 有限、曲线稳定”不能替代“购买成功率得到提升”。

## 2. 仓库和证据边界

```text
分支：feat/bpo2
本文对应提交前基线：d049d3f
服务器仓库：~/shopping-grpo
Windows 仓库：D:\shopping-grpo-longhorizon
运行契约：ShopSimulator Environment v2.1 / Reward v4 / observation v2 / tool schema v2
```

正式 final200 没有用于训练或本轮选型。模型、日志、SwanLab 缓存和分析 JSON 位于服务器 `outputs/`，通常不提交 Git。

### 需要直接阅读的代码

```text
configs/bpo.yaml
configs/bpo_agent_loop.yaml
scripts/bpo.sh
scripts/train_bpo.py
scripts/check_bpo_runtime.py
scripts/audit_bpo_formal_run.py
src/shopping_grpo/training/bpo/
src/shopping_grpo/training/grpo/dynamic_sampling.py
tests/test_bpo_*.py
tests/test_verl_bpo_*.py
tests/test_verl_dynamic_sampling*.py
```

### 相关文档和分析工具

```text
docs/bpo.md
docs/bpo-formal-results.md
docs/bpo-diagnostics.md
docs/bpo-independent-audit.md
docs/data-layout.md
scripts/export_swanlab_run_metrics.py
scripts/audit_bpo_dev500_diagnostics.py
scripts/run_standalone_checkpoint_evaluation.py
```

文档是索引，实际证据以源码、正式 `run_contract.json`、`training_diagnostics.jsonl`、日志和评测 JSON 为准。

## 3. 正式运行输入与标识

```text
起点模型：outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325
训练数据：data/grpo/formal-v2/multiturn-train.parquet
验证数据：data/grpo/formal-v2/multiturn-validation.parquet
数据清单：data/grpo/formal-v2/manifest.json
数据环境清单：data/environment-v4.json
BPO 运行时清单：data/environment-bpo-v1.json
随机种子：20260823
Reward profile：none
运行标签：bpo-native-v4-step200-r1600-seed20260823-20260826-113750-r3
SwanLab run ID：4cmh0p3k
SwanLab：https://swanlab.cn/@mode/shopping-multiturn-agentic/runs/4cmh0p3k/chart
```

formal-v2 数据：

```text
train：1000 tasks / 2000 rows / gap 1000 / complete 1000
validation：200 tasks / 400 rows / gap 200 / complete 200
```

服务器产物：

```text
RUN=outputs/models/bpo-native-v4-step200-r1600-seed20260823-20260826-113750-r3
$RUN/run_contract.json
$RUN/step0_validation_contract.json
$RUN/training_diagnostics.jsonl
$RUN/latest_checkpointed_iteration.txt
$RUN/global_step_*/
outputs/bpo/logs/<run-tag>.log
outputs/bpo/step0-validation-cache/
outputs/models/bpo-native-v4-step200-r1600-seed20260823-export
```

## 4. 正式配置摘要

以下值应再与正式 `run_contract.json` 核对：

| 项目 | 值 |
|---|---:|
| optimizer steps | 200 |
| trees / step | 2 |
| sibling count K | 4 |
| branch count | 1 |
| returns / step | 8 |
| 总有效 tree / return | 400 / 1600 |
| train / mini / micro batch | 2 / 2 / 1 |
| LR | `1e-6` |
| warmup / scheduler | 10 / cosine |
| scheduler horizon / min ratio | 500 / 0.1 |
| PPO clip / upstream lambda | 0.2 / 0.95 |
| rollout temperature / top-p | 0.7 / 0.9 |
| LoRA rank / alpha | 16 / 32 |
| KL reward / actor KL loss | false / false |
| reference policy | 未启用 |
| loss aggregation | token mean |
| fused / Liger / remove padding | true / true / true |
| max prompt / response / sequence | 4096 / 20480 / 24576 |
| vLLM memory utilization / max seqs | 0.45 / 8 |
| GPU / AgentLoop workers | 4 / 2 |
| max environment steps / questions | 35 / 2 |

正式 dynamic sampling：

```text
target=2
minimum=2
require_full_batch=true
soft_warning_gen_batches=10
max_num_gen_batches=30
```

每步必须凑满 2 棵有效树，不能用 1 棵树降级更新。旧审计曾写过 `minimum=1/max_batches=3`，该值错误且不适用于正式运行；`docs/bpo-independent-audit.md` 已修正。

## 5. 正式运行原始计数

`scripts/audit_bpo_formal_run.py` 已接受：

| 项目 | 数值 |
|---|---:|
| optimizer steps | 200/200 |
| 有效 trees | 400/400 |
| 有效 sibling returns | 1600/1600 |
| 跳过 update | 0 |
| 候选 backbone | 1,732 |
| branch rollouts | 5,196 |
| 总 rollouts | 6,928 |
| response tokens | 1,478,990 |
| environment transitions | 30,255 |
| Shopper API calls | 880 |
| wall time | 8:57:06 |

`400 / 1732 = 23.09%`，这里只表示有效 tree 与候选 backbone 的计数比例，不说明拒绝原因。

## 6. validation 与训练监控

SwanLab 正确 validation 命名空间为 `val-shopping/summary/*`。云端因收尾异常只到 step 199；step 200 来自服务器最终日志。

| step | strict | purchase | mean reward |
|---:|---:|---:|---:|
| 0 | 0.6875 | 0.6925 | 0.647794 |
| 10 | 0.6975 | 0.7050 | 0.663147 |
| 50 | 0.6850 | 0.6925 | 0.648229 |
| 100 | 0.6875 | 0.6950 | 0.652001 |
| 150 | 0.6975 | 0.7050 | 0.673723 |
| 200 | 0.6975 | 0.7050 | 0.6772444853 |

step 200：

```text
done rate=0.9900
reward valid rate=0.9950
mean steps=4.57
sampling invalid rate=0.0050
infrastructure invalid rate=0
reward unverifiable rate=0.0050
Shopper question rate=1.0000
```

SwanLab 训练摘要：

```text
actor/ppo_kl mean=0.00586, max=0.01410
gradient norm mean=0.2567, max=0.8397
actor loss range=[-0.321, 0.387]
clip fraction max=0.00562
candidate batches mean/median/p95/max=4.33/4/9/13
seconds to full batch mean/median/p95/max=159.9/135.8/243.6/956
slow-batch warnings=6
unique branch actions mean=2.62/4
unique tool sequences mean=2.33/4
sibling return std mean=0.415
sibling return range mean=0.934
```

导出文件（如服务器仍保留）：

```text
outputs/analysis/swanlab-bpo-step200-4cmh0p3k/swanlab-history.json
outputs/analysis/swanlab-bpo-step200-4cmh0p3k/swanlab-analysis.md
```

## 7. 冻结 dev500 三面板

每个条件 500 题，共 1500 条，`final_evaluation_used=false`。

| 模型 | gap ask | gap no-ask | complete | total | gap gain | unnecessary ask | mean reward | done | reward valid | guards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BPO-200 | 0.690 | 0.526 | 0.720 | 0.645 | +0.164 | 0.922 | 0.5983 | 1487/1500 | 1483/1500 | 49 |
| SFT-325 | 0.690 | 0.528 | 0.722 | 0.647 | +0.162 | 0.938 | 0.6010 | 1490/1500 | 1486/1500 | 32 |

BPO 减 SFT：

```text
gap ask=+0.0000
gap no-ask=-0.0020
complete=-0.0020
total=-0.0013
gap gain=+0.0020
unnecessary ask=-0.0160
mean reward=-0.0027
```

原始产物：

```text
outputs/evaluation/bpo-native-v4-step200-r1600-dev500-v1/results/evaluation_results.json
outputs/analysis/bpo-step200-vs-sft325-dev500/paired-diagnostics.json
```

配对 strict flips：18 gains、20 losses。

```text
gap-ask-enabled: gains=9, losses=9
  loss gold -> partial=7, wrong=1, repeat=1
  gain partial -> gold=5, repeat -> gold=3, unknown -> gold=1

gap-ask-disabled: gains=3, losses=4
  loss gold -> partial=2, wrong=2
  gain partial -> gold=3

complete-ask-enabled: gains=6, losses=7
  loss gold -> unknown=4, partial=3
  gain partial -> gold=3, unknown -> gold=3
```

具体 task IDs 和逐条 trajectory 字段在 `paired-diagnostics.json` 中，不在本文重复。

## 8. 已发生过的异常

这些异常不等于当前 HEAD 仍会复现：

1. Hydra seed 被解析为字符串：`manual_seed expected a long, but got str`。
2. actor 长序列全词表 `log_softmax` OOM；随后启用 fused kernels 和 fused PPO input-gradient backport。
3. XML tool-call 畸形参数；随后加入 tolerant parser patch。
4. 稀疏物理 GPU 与 Ray accelerator ID 映射错误；随后加入 physical-to-logical 预检。
5. ShopSimulator 曾启动为 Reward v3；正式运行要求实测 v2.1/v4。
6. CPU 合成诊断曾与真实训练诊断混淆；launcher 后来会在正式训练前删除合成诊断。
7. 诊断审计曾阻断正常训练；后来改成非阻断记录，正式验收由独立脚本完成。
8. step 200 完成后 SwanLab 收尾出现 `RuntimeError: cannot join current thread`。错误前已有 200/200、checkpoint 和 final validation；云端 run 显示 `CRASHED` 且只记录到 step 199，随后代码加入显式幂等 finish。

## 9. 开放分析线索

以下只用于帮助定位证据，不是原因清单或思考边界。接手者应主动补充、合并或推翻这些方向：

- Reward v4 连续奖励与 strict gold purchase 是否错位；
- 稀疏终局 reward 如何落到 token，并经 LOO advantage、response mask 和 PPO loss 分配给动作；
- 分叉前共享长前缀的零和 advantage 与 `upstream_lambda=0.95` 是否造成信用抵消；
- 长轨迹中的搜索、询问、选择和购买存在多步依赖时，单分叉 suffix 是否覆盖真正的因果决策；
- 分叉过早或过晚、35-step 上限、上下文长度、工具 observation 和累积错误是否改变学习信号；
- 有效树过滤是否选择了容易产生 return 差异、但不代表关键购买能力的任务；
- 最大首 token entropy 是否经常选到协议或工具格式边界，而不是购买关键决策；
- K=4 的 sibling 多样性、return 方差和 LOO advantage 是否足以提供稳定方向；
- `token-mean` 是否对长短 continuation 产生不合适的权重；
- 无 KL/reference 与论文主实验差异是否重要；
- LR、cosine horizon 500、只训练 200 步、LoRA rank 16 是否使更新过弱；
- full-batch 2 trees 与 23.09% 候选有效率是否引入选择偏差或高成本；
- backbone 由当前策略在线生成、每棵树只选择一个最大熵分叉，是否带来探索覆盖或状态分布偏差；
- ShopSimulator、Shopper 回答和 rollout sampling 的随机性是否淹没 sibling 间的可学习差异；
- validation question rate 100% 是否说明数据、指标或行为存在偏差；
- dev500 的 -2/1500 是否主要是统计波动；
- `gold -> wrong` 和 `gold -> unknown` 的损失轨迹是否暴露一致的策略退化。

## 10. 可直接复制给新对话的提示词

```text
请开放、独立地分析 shopping-grpo 的正式 BPO 训练为什么没有在 SFT checkpoint-325 基础上提高完整购买 strict success。先讨论可能性，不要直接设计或启动下一轮实验，也不要受现有文档的原因分类限制。

先读取 AGENTS.md、docs/bpo-training-handoff.md、docs/bpo.md、docs/bpo-formal-results.md、docs/bpo-diagnostics.md、configs/bpo.yaml、configs/bpo_agent_loop.yaml。然后直接审查 scripts/train_bpo.py、scripts/check_bpo_runtime.py、scripts/audit_bpo_formal_run.py、src/shopping_grpo/training/bpo/、src/shopping_grpo/training/grpo/dynamic_sampling.py 和相关 tests。

正式运行事实：
- full-bpo-v1，从 SFT checkpoint-325 开始；
- Reward v4 profile=none，seed=20260823；
- 200 optimizer steps，每步严格 2 棵有效树，每棵 K=4；
- 400 棵有效树、1600 sibling returns；
- dynamic sampling 是 target=2/minimum=2/require_full_batch=true/max_batches=30；
- dev500：BPO total strict=0.645，SFT=0.647。

请：
1. 先根据源码、运行证据和相关研究建立自己的原因图谱；可以重新分类，也要主动寻找本文未列出的解释。
2. 搜索并核对 BPO 原论文及近期长轨迹 agent RL、branching rollout、credit assignment 的一手资料；明确论文做法、项目适配和可能偏差。
3. 特别追踪 Reward 从终局结果到 tree return、LOO advantage、上游信用、token mask 和 PPO loss 的完整链路。
4. 分析 agent 长轨迹的决策跨度、分叉覆盖、探索偏差、上下文/工具状态、累积错误和 horizon 对学习的影响。
5. 检查 tree、entropy branch、snapshot clone、mask、token-mean loss、fused/remove-padding/Liger 和 optimizer 数据流，但不要假设工程实现是唯一问题来源。
6. 对每种可能原因列出支持证据、反对证据、潜在影响和证据缺口；按证据强弱排序，保留无法归类的新假设。
7. 列出需要我补充的具体服务器文件或只读命令，并先和我讨论改进方向。
8. 在讨论确认前不要改代码、启动训练、合并模型或使用 final200。

输出：
- 原始证据
- findings
- 开放原因图谱、可能性排序及正反证据
- 未验证项
- 需要补充的信息
- 需要与我讨论的关键选择
```
