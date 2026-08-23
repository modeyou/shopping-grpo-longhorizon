# 多轮 Reward v4 LoRA SFT

## 目标

当前 SFT 阶段用于训练 Qwen3.5-2B 遵循 ShopSimulator 多轮动作协议：只在信息确实缺失时向 Shopper 提问，只调用合法购物工具，以当前可见观察为依据执行动作，正确选择商品变体，并以有效购买或停止决策结束轨迹。

`experiments/sft/summary.json` 属于原参考项目，不是当前多轮项目的训练输入、数据划分或超参数来源。当前项目只遵循：

```text
Baseline → SFT → GRPO → Evaluation
```

运行契约固定为 Environment v2.1、Reward v4、observation v2 和 tool schema v2。

## 冻结数据契约

正式数据只来自三个经过 Reward v4 重新审计的候选池：

- `complete-no-ask-v1`：完整需求且没有不必要提问；
- `composite-replay-v1`：经过重放验证的复合澄清轨迹；
- `autonomous-gap-v1`：自主缺口澄清轨迹。

正式混合包含 1,800 个唯一 task/source-goal 对，按 task 隔离为 1,620 条训练样本和 180 条验证样本。三类策略按 assistant-loss token 份额冻结为约 50%/30%/20%，实际行数为：

| 策略 | 行数 | Assistant tokens | Token 份额 |
|---|---:|---:|---:|
| complete-no-ask-v1 | 802 | 212,029 | 50.22% |
| composite-replay-v1 | 627 | 126,253 | 29.91% |
| autonomous-gap-v1 | 371 | 83,897 | 19.87% |

802 条 complete 样本中，401 条仅提供购物工具，401 条提供包含 `ask_shopper` 的完整多轮工具 schema，但目标轨迹仍保持零提问。这一 schema augmentation 用于学习“工具可用不等于必须调用”。

正式数据目录：

```text
outputs/multiturn-sft/mix-formal-1800-v4-seed20260822
```

冻结哈希：

| 产物 | SHA-256 |
|---|---|
| manifest.json | `11be05b2d4e2cfb49529542a23030988e21ea59266cad97b48598302e56e4eeb` |
| train.jsonl | `bbfd477b5f4a776d64dd0c6e338829e8b76cd6529ca7806bef3def7c119ac9ae` |
| validation.jsonl | `40d648d4e8cf2f47bb1af6ef330907da25e2e7cbf45fe7e8856aad833652ab8b` |
| data/evaluation/tasks.jsonl | `d99112a20ef47534c27a32e4b38229bf048dcc6b06fef2e3e919aac3093662f5` |

训练与验证数据和最终评测 task 集的 task ID 重叠数必须为 0。正式数据 manifest 的 schema 为 `shopping-multiturn-sft-mix-v2`，并冻结候选池哈希、输出哈希、任务 ID、策略/schema 数量、token 数量、数据划分、模型 revision、随机种子和最终评测排除审计。

数据生成命令：

```bash
: "${GRPO_PYTHON:?请设置项目 Python}"
: "${MODEL_DIR:?请设置固定模型目录}"

"$GRPO_PYTHON" scripts/prepare_multiturn_sft_mix.py \
  --audit-manifest outputs/multiturn-sft/v4-audit-pools-02/manifest.json \
  --evaluation-tasks data/evaluation/tasks.jsonl \
  --model "$MODEL_DIR" \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --output-dir outputs/multiturn-sft/mix-formal-1800-v4-seed20260822 \
  --total-rows 1800 \
  --validation-ratio 0.1 \
  --max-length 24576 \
  --seed 20260822 \
  --complete-token-ratio 0.5 \
  --composite-token-ratio 0.3 \
  --autonomous-token-ratio 0.2 \
  --token-share-tolerance 0.05
```

## SwanLab 配置

64 条 smoke 使用 `--swanlab-mode local`，不需要云端登录。正式训练使用 `online` 模式，只在 world rank 0 初始化一次 SwanLab，避免四卡生成四个重复实验。

在训练服务器的项目环境中完成一次交互式登录：

```bash
: "${SWANLAB_BIN:?请设置项目 SwanLab CLI}"
"$SWANLAB_BIN" --version
"$SWANLAB_BIN" login
"$SWANLAB_BIN" verify
"$SWANLAB_BIN" ping
```

不要把 API key 写入命令历史或启动脚本，也不要在仓库内使用 `login --local`，否则 `.swanlab/` 凭据目录可能破坏 clean-Git 复现门槛。

正式项目名：`shopping-multiturn-agentic`。本次成功运行的 run name 为：

```text
qwen35-2b-sft-lora-v4-n1800-e2-seed20260822-r2
```

SwanLab run ID：`48kdp8mk`。SwanLab 页面只用于曲线监控，不能代替本地冻结产物和哈希审计。

## 正式训练配置

| 配置 | 值 |
|---|---|
| 基座模型 | Qwen3.5-2B，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 最大序列长度 | 24,576 |
| Epoch | 2 |
| GPU | CUDA 0–3，共 4 卡 |
| 单卡训练/验证 batch | 1/1 |
| 梯度累积 | 2 |
| 有效全局 batch | 8 |
| 峰值学习率 | `1e-4` |
| 调度器 | 3% warmup，随后线性衰减 |
| LoRA rank/alpha/dropout | 16/32/0.05 |
| 精度 | bf16 |
| 梯度检查点 | 启用 |
| 注意力/融合 loss | SDPA/Liger |
| 训练日志 | 每 5 step |
| 验证 | 每 50 step，并在结束后再验证一次 |
| Checkpoint | 每 25 step，最多保留 10 个 |
| Seed/data seed | 20260822/20260822 |
| 完全确定性 | 启用 |

学习率不是常数：先从 0 warmup 到 `1e-4`，再在两轮训练内线性衰减。1,620 条训练样本和有效全局 batch 8 对应每个 epoch 约 203 个 optimizer step，总计 406 step。

正式训练输出：

```text
outputs/models/multiturn-sft-v4-1800-e2-seed20260822-r2
```

正式启动命令：

```bash
: "${TORCHRUN:?请设置项目 torchrun}"
: "${MODEL_DIR:?请设置固定模型目录}"

export PYTHONHASHSEED=20260822
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWANLAB_MODE=online

"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=4 scripts/train_lora_sft.py \
  --model "$MODEL_DIR" \
  --train outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/train.jsonl \
  --validation outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/validation.jsonl \
  --data-manifest outputs/multiturn-sft/mix-formal-1800-v4-seed20260822/manifest.json \
  --evaluation-tasks data/evaluation/tasks.jsonl \
  --output outputs/models/multiturn-sft-v4-1800-e2-seed20260822-r2 \
  --max-length 24576 \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type linear \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules \
    q_proj k_proj v_proj o_proj \
    gate_proj up_proj down_proj \
    in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj \
  --dtype bf16 \
  --gradient-checkpointing \
  --liger-kernel \
  --attention-implementation sdpa \
  --logging-steps 5 \
  --eval-steps 50 \
  --save-steps 25 \
  --save-total-limit 10 \
  --seed 20260822 \
  --data-seed 20260822 \
  --full-determinism \
  --require-clean-git \
  --swanlab \
  --swanlab-mode online \
  --swanlab-project shopping-multiturn-agentic \
  --swanlab-run-name qwen35-2b-sft-lora-v4-n1800-e2-seed20260822-r2
```

## 正式训练结果

本次成功运行绑定 Git commit `8c929be9f025c34d8c9620441435d46e39dd1bc2`，完成 406 optimizer step 和 2.0 epoch。

| 指标 | 结果 |
|---|---:|
| 运行时间 | 6,940.93 秒（约 1 小时 55 分 41 秒） |
| 首个/最后记录 train loss | 0.408989/0.158007 |
| 最低记录 train loss | 0.113681 |
| Trainer 汇报平均 train loss | 0.187540 |
| 最终/最低 eval loss | 0.182440/0.182392 |
| 峰值 GPU allocated memory | 16.58 GiB |
| LoRA 可训练参数 | 16,819,200 |
| Adapter tensors | 372 |

验证 loss 依次为：

```text
0.223017, 0.205616, 0.196346, 0.191286, 0.189146,
0.186900, 0.184479, 0.182392, 0.182440
```

loss 持续下降只能说明 teacher-forcing 目标仍在改善，不能替代长程购物 rollout。开发集结果已经证明：final-2epoch 虽然 loss 更低，但 complete 条件可能发生明显行为退化。

保留的 checkpoint 为：

```text
200, 225, 250, 275, 300, 325, 350, 375, 400, 406
```

最终 adapter SHA-256：

```text
7a5890e51434e0415bfbcb63f8a7f18292d6c0e38c4efac0aeb1f48298095d55
```

## 复现与产物契约

正式运行必须设置 `PYTHONHASHSEED=20260822` 并启用 `--require-clean-git`；任一条件不满足时应在加载模型前失败。模型输出目录必须保留：

- `run_provenance.json`：完整 argv、Git commit、工作树状态、模型 revision/config 哈希、运行包版本、可见 GPU、随机种子、数据路径、行数和 SHA-256；
- `data_manifest.snapshot.json`：候选池、配额、schema、task-disjoint 划分、最终评测排除和输入哈希的不可变快照；
- `train_summary.json`：Trainer 参数、step 历史、最终指标、运行时间和峰值显存；
- `completion_audit.json`：训练完成状态和候选 checkpoint；
- `checkpoint-*`、最终 adapter、原始日志和唯一的 world-rank-zero SwanLab run。

不要依赖终端历史、可变 shell 变量、SwanLab 页面或人工实验日志作为唯一记录。输出目录必须为新目录或空目录，除非明确执行 checkpoint resume。

`--full-determinism` 会请求确定性 PyTorch 算法；只有硬件和软件栈完全相同时才预期逐 bit 一致。跨软件栈复现以冻结产物、配置和指标容差为准。

## 开发集评测与模型选择

SFT 后使用冻结的 `data/multiturn/evaluation-dev-v2` 比较模型，主指标 strict success 要求完整终局、`gold_purchase` 且 `reward_valid=true`。同时报告完成率、Reward v4 有效率、reward type、guard 原因、gap 提问率、grounded question、complete 不必要提问率、上下文截断和平均步数。

当前完整 dev500×3 结果：

| 模型 | Gap+Ask strict | Gap-NoAsk strict | Complete+Ask strict | 总 strict |
|---|---:|---:|---:|---:|
| Base Qwen3.5-2B | 0.4% | 0.2% | 0.6% | 0.4% |
| checkpoint-200 | 66.4% | 49.2% | 73.4% | 63.0% |
| final-2epoch | 69.2% | 49.0% | 38.2% | 52.13% |

为定位第二个 epoch 内的行为拐点，随后在同一个冻结 dev200 子集上评测全部 10 个保留 checkpoint。每个 checkpoint 固定执行 Gap+Ask、Gap-NoAsk、Complete+Ask 三个条件，各 200 个任务，共 600 条轨迹。全部候选通过 Reward v4、轨迹数量、模型名和基础设施错误审计；最终 200-task 评测集未使用。dev200 sweep manifest SHA-256：

```text
5754aaaf1a4b67c47751f4e35782866a4794d49910bbcf9651eff2d5080b2d1a
```

| Checkpoint | Gap+Ask | Gap-NoAsk | Complete+Ask | 总 strict | Mean reward | Done | Reward valid | Guards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 62.5% | 48.5% | 71.0% | 60.67% | 0.5383 | 591/600 | 588/600 | 24 |
| 225 | 64.5% | 46.5% | 68.0% | 59.67% | 0.5438 | 594/600 | 587/600 | 21 |
| 250 | 60.5% | 45.5% | 64.5% | 56.83% | 0.5070 | 591/600 | 586/600 | 36 |
| 275 | 60.5% | 48.5% | 66.0% | 58.33% | 0.5298 | 593/600 | 577/600 | 28 |
| 300 | 65.0% | 48.5% | 66.5% | 60.00% | 0.5520 | 592/600 | 586/600 | 19 |
| **325** | **66.5%** | **54.0%** | **70.0%** | **63.50%** | **0.5807** | **598/600** | **595/600** | **8** |
| 350 | 67.0% | 51.5% | 67.5% | 62.00% | 0.5760 | 590/600 | 585/600 | 18 |
| 375 | 68.0% | 52.0% | 66.5% | 62.17% | 0.5551 | 596/600 | 592/600 | 8 |
| 400 | 65.0% | 49.5% | 68.0% | 60.83% | 0.5419 | 594/600 | 592/600 | 10 |
| 406 | 66.5% | 49.5% | 67.5% | 61.17% | 0.5512 | 592/600 | 590/600 | 16 |

checkpoint-325 是 dev200 的领先候选：总 strict、Gap-NoAsk、平均奖励、Done 和 Reward-valid 均为最佳，Complete 接近最佳，guards 与最低值并列。checkpoint-350/375 的 Gap+Ask 略高，但总体更不均衡。所有候选在 Complete 条件仍有 91%–99% 的不必要提问率，说明 SFT 已建立澄清能力但尚未学会充分抑制多余提问；这是 Reward v4 GRPO 需要继续优化的主要行为。

checkpoint-325 仍需执行完整 dev500×3 同协议复核，才能正式冻结为 `selected-for-grpo`。复核前不启动 GRPO，最终 200-task 评测集继续封存。

GRPO 只能使用经过开发集审查并显式选定的 merged model，不能直接使用 LoRA adapter。模型合并与 GRPO 训练都属于单独授权步骤。
