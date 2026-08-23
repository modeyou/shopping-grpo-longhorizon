# BPO 正式方案与运行手册

本文只描述本项目独立的 BPO 路线。BPO 不复用 `configs/grpo.yaml`，不覆盖 GRPO
checkpoint，也不改变 ShopSimulator Reward v4。正式实现位于：

- `configs/bpo.yaml`：训练参数；
- `configs/bpo_agent_loop.yaml`：分支 AgentLoop 参数；
- `src/shopping_grpo/training/bpo/`：分支、优势和 veRL 适配；
- `scripts/bpo.sh`、`scripts/train_bpo.py`：启动入口；
- `scripts/apply_verl_bpo_patch.py`：精确熵探针补丁。

## 1. 目标

BPO 的第一目标仍然是提高完整购物成功率，而不是只修工具格式。所有 sibling 都在
Environment v2.1、Reward v4、observation v2、tool schema v2 下完成真实购物；训练信号取
Reward v4 终局效用。严格成功仍定义为 `gold_purchase` 且 `reward_valid=true`。

正式起点为 SFT `checkpoint-325` 的合并模型，训练数据为
`data/grpo/formal-v2/multiturn-train.parquet`，验证数据为
`data/grpo/formal-v2/multiturn-validation.parquet`。两者都必须由同一
`data/grpo/formal-v2/manifest.json` 绑定，且不得与 `data/evaluation/tasks.jsonl` 重叠。

## 2. 算法冻结

实现采用完整 one-boundary BPO，而不是 BPO-lite：

1. 对一个 prompt 运行一条 backbone trajectory；
2. 每个真实 assistant action 生成前，在 ShopSimulator 服务端保存不透明快照；
3. 正常生成 action，跳过固定 XML 协议 token，定位第一个语义决策 token；
4. 对该 token 位置额外执行一次 `max_tokens=1` 的全词表 log-prob 探针，计算精确
   Shannon entropy；
5. 每条 backbone 只保留熵最高的一个 action 边界；熵相同时选择更早边界；
6. 从同一环境、Shopper history、运行诊断和 token 前缀克隆 3 条 continuation；
7. backbone continuation 加 3 条 clone 构成 K=4 sibling returns；
8. sibling 内使用 leave-one-out baseline：

   `A_i = R_i - mean(R_j, j != i)`

9. 分叉 action 及其后续 token 使用完整 `A_i`；共享前缀中距离分叉点 `d` 个 action 的
   token 使用 `0.95^d * A_i`；工具 observation 与 padding 永远 mask 为 0；
10. 使用 PPO clip 0.2 更新 LoRA actor。

不同 backbone 会独立选择自己的最高熵边界，因此分叉位置可以不同。每个 prompt 的正式
return budget 固定为 4，不因轨迹 action 数增加。

算法依据为论文 *Branching Policy Optimization: Sandbox-Native Language Agent
Reinforcement Learning*：<https://arxiv.org/abs/2607.14171>。项目实现额外冻结了单边界、
K=4 和服务端快照契约，以适配 4×24GB GPU 与 ShopSimulator。

## 3. 快照与隔离契约

快照只存在 ShopSimulator 服务端，训练进程只收到随机 `snapshot_id`。快照包括页面、
session、Reward tracker、历史与可点击项；clone 会申请新 slot。Shopper history、问题次数、
`clarified_constraints` 和公共运行诊断在 trajectory 本地深拷贝。

正式 train batch 为 2；每组同时最多占用 1 backbone + 3 clone，因此运行期最多需要 8 个
ShopSimulator slot。BPO AgentLoop worker 固定为 2，保证 veRL 按 K=4 重复后的每组 sibling
不会被切到不同 worker。不要把它改回 GRPO 曾使用的 8 workers。

ShopSimulator 更新后必须重启服务；旧进程没有 `snapshot`、`clone` 和 `drop_snapshot`
接口，不能运行 BPO。

## 4. 正式参数

| 项目 | 冻结值 |
|---|---:|
| backbone | SFT checkpoint-325 merged model |
| Reward | 原生 `shopsimulator-reward-v4`，不做 bounded shaping |
| train batch | 2 prompts |
| sibling count K | 4 |
| branch count M | 1 |
| return budget | 4 / prompt |
| 分叉选择 | 最大精确全词表 entropy，最早边界打破平局 |
| upstream λ | 0.95 |
| PPO clip | 0.20 |
| rollout temperature / top-p | 0.7 / 0.9 |
| LoRA rank / alpha | 16 / 32 |
| learning rate | 1e-6，warmup ratio 0.03 |
| 正式 optimizer updates | 10 |
| checkpoint / validation | 每 5 updates |
| GPU | 0–3，共 4 张 |
| vLLM max sequences | 8 |
| AgentLoop workers | 2 |

`use_fused_kernels=true` 与 `use_liger=true` 是正式显存方案。它们避免完整大词表 logits
长期驻留并降低模型内部显存，影响的是 kernel、吞吐和显存，不改变 BPO 分支、Reward、
advantage 或评测定义。由于两者组合依赖固定 veRL/Transformers 环境，仍必须先通过一次
1-step smoke，确认 loss 有限、四卡工作且无 OOM。

全词表 logits 只在单 token entropy probe 内生成，并立即在 vLLM 服务进程中归约为一个
标量；完整向量不会进入 AgentLoop output 或训练 batch。

## 5. 安装补丁与预检

在项目 Python 环境中执行：

```bash
cd ~/shopping-grpo
export PYTHONPATH=./src
export GRPO_PYTHON=/path/to/project-python

"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py
"$GRPO_PYTHON" scripts/apply_verl_bpo_patch.py
```

BPO 补丁只修改固定 `verl==0.8.0` 的 `vllm_async_server.py`。脚本校验官方源码 SHA256、
自动备份、幂等检查并拒绝未知版本。

配置 Shopper API key 后执行只读预检：

```bash
read -rsp 'SHOPPER_API_KEY: ' SHOPPER_API_KEY
echo
export SHOPPER_API_KEY
export SHOPPER_BASE_URL='https://your-endpoint/compatible-mode/v1'
export SHOPPER_MODEL='deepseek-v4-flash-0731'

bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$BPO_OUTPUT" \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  --preflight-only
```

预检会拒绝：非 Reward v4、错误数据 manifest、非四卡、非 K=4/M=1、worker 分组错误、
缺失精确熵补丁、未启用两项 OOM 配置、动态采样补丁缺失以及 Shopper/SwanLab 配置错误。

## 6. 1-step smoke

smoke 只验证完整链路，不作为实验结果，也不保留 checkpoint：

```bash
bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$BPO_SMOKE_OUT" \
  --experiment-name bpo-native-v4-smoke1 \
  --logger console \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  -- \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.test_freq=-1
```

smoke 必须确认：

- 每个 `bpo_group_id` 恰有 sibling 0–3；
- 四条轨迹具有相同分叉前缀与同一 `bpo_branch_action`；
- 三个 clone 的 ShopSimulator slot 不同，Shopper history 不串线；
- entropy 为有限标量，完整词表向量没有写入输出；
- Reward 全部为 v4，基础设施无效样本不会伪装成正常终局；
- advantage 数值有限，工具/padding mask 为 0；
- optimizer 确实更新一次，四卡无 OOM。

## 7. 正式训练与监控

确认 smoke 后使用全新输出目录启动；默认配置即 10 updates：

```bash
bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$BPO_FORMAL_OUT" \
  --experiment-name bpo-native-v4-formal10-seed20260824 \
  --logger swanlab \
  --seed 20260824 \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL"
```

正式启动会在输出目录写 `shopping-bpo-run-contract-v1`，记录 Git commit、模型/数据/
配置 SHA256、seed、分支参数与两项 OOM 配置，不记录 API key。

不要让 BPO 与 GRPO 同时占用 GPU 0–3、同一个 Ray runtime 或同一批 ShopSimulator slots。
当前可以在 GRPO 运行期间开发/提交 BPO 代码，但正式 BPO smoke 应在 GRPO 完全退出后执行。

## 8. 评测与停止规则

10 updates 完成后，只在冻结 dev500 的三个面板评测候选 checkpoint：

- gap + ask enabled；
- gap + ask disabled；
- complete + ask enabled。

先比较严格购买成功率、Reward v4 有效率、mean terminal utility，再检查提问率、重复提问、
guard rejection、上下文溢出和基础设施无效率。不要在方法选择阶段使用 final200；只有最终
方法和 checkpoint 完全冻结后，才允许一次正式 final200 评测。

出现以下任一情况立即停止，不把结果解释成算法效果：快照状态不一致、sibling 前缀不同、
全词表概率质量不约等于 1、Reward 非 v4、数据重叠、OOM、NaN/Inf、连续动态采样跳过或
ShopSimulator slot 泄漏。
