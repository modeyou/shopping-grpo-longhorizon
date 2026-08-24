# BPO 正式方案与运行手册

本文只描述本项目独立的 BPO 路线。BPO 使用自己的配置、运行入口和输出目录，
不覆盖 GRPO checkpoint，也不修改 `configs/grpo.yaml`。

主要文件：

- `configs/bpo.yaml`：正式训练配置；
- `configs/bpo_agent_loop.yaml`：分叉 AgentLoop 注册；
- `src/shopping_grpo/training/bpo/`：分叉、优势估计和 veRL 适配；
- `scripts/bpo.sh`、`scripts/train_bpo.py`：独立启动入口；
- `scripts/apply_verl_bpo_patch.py`：精确全词表熵补丁。

## 1. 训练目标

BPO 的首要目标仍然是提高完整购物成功率，而不是只改善工具格式。所有 sibling
都在 ShopSimulator Environment v2.1、Reward v4、observation v2 和 tool schema v2
下完成真实购物。严格成功仍定义为：

```text
reward_type == gold_purchase and reward_valid == true
```

正式起点是 SFT `checkpoint-325` 合并模型。训练和验证数据分别为：

```text
data/grpo/formal-v2/multiturn-train.parquet
data/grpo/formal-v2/multiturn-validation.parquet
```

二者必须由 `data/grpo/formal-v2/manifest.json` 绑定，并与
`data/evaluation/tasks.jsonl` 保持零重叠。

## 2. 冻结的算法方案

实现采用完整 one-boundary BPO，而不是 BPO-lite：

1. 每个 prompt 先生成一条 backbone trajectory；
2. 每次 assistant action 生成前，在 ShopSimulator 服务端保存快照；
3. 正常生成 action，并定位第一个非固定协议的语义 token；
4. 在该位置执行一次 `max_tokens=1`、`top_p=1`、`temperature=1` 的全词表
   log-prob 探针，计算精确 Shannon entropy；
5. 每条 backbone 只保留 entropy 最大的一个 action 边界，熵相同选择更早边界；
6. 从同一个环境、Shopper history、运行诊断和 token 前缀克隆三条 continuation；
7. backbone 与三个 clone 构成 K=4 sibling returns；
8. sibling 内使用 leave-one-out baseline：

```text
A_i = R_i - mean(R_j, j != i)
```

9. 分叉 action 及其后续 action 使用完整 `A_i`；分叉前第 `d` 个 action 使用
   `0.95^d * A_i`；工具 observation 和 padding 始终由 response mask 置零；
10. 使用 PPO clip 0.2 更新 LoRA actor。

不同 backbone 独立选择各自的最大熵边界，因此不同 rollout 的分叉点可以不同。
每个 prompt 的总 return budget 固定为 4，不随 action 数增加。

算法依据是论文 *Branching Policy Optimization: Sandbox-Native Language Agent
Reinforcement Learning*：<https://arxiv.org/abs/2607.14171>。项目进一步冻结了
单边界、K=4 和服务端快照契约，以适配四张 24GB GPU 与 ShopSimulator。

## 3. 快照与状态隔离

快照只保存在 ShopSimulator 服务端，训练进程只接收随机 `snapshot_id`。快照覆盖：

- 服务端 session 与 Reward tracker；
- 浏览器 URL、页面内容与可执行动作；
- 环境历史、购物车和任务状态；
- trajectory 本地运行状态与已澄清约束；
- Shopper history、问题次数和调用计数。

每个 clone 会申请独立 ShopSimulator slot。正式 train batch 为 2，每组最多同时占用
一条 backbone 和三个 clone，因此最多需要 8 个 slot。AgentLoop worker 固定为 2，
确保 veRL 按 K=4 重复后，每组 sibling 不会被切分给不同 worker。

更新代码后必须重启 ShopSimulator；旧服务进程没有 `snapshot`、`clone` 和
`drop_snapshot` 接口，不能运行 BPO。

## 4. 正式参数

| 参数 | 冻结值 |
|---|---:|
| backbone | SFT checkpoint-325 merged model |
| Reward | 原生 `shopsimulator-reward-v4`，无 bounded shaping |
| train batch | 2 prompts |
| sibling count K | 4 |
| branch count M | 1 |
| return budget | 4 / prompt |
| 分叉选择 | 最大精确全词表 entropy，最早边界打破平局 |
| upstream lambda | 0.95 |
| PPO clip | 0.20 |
| rollout temperature / top-p | 0.7 / 0.9 |
| LoRA rank / alpha | 16 / 32 |
| learning rate | `1e-6`，warmup ratio `0.03` |
| 正式 optimizer updates | 10 |
| checkpoint / validation | 每 5 updates |
| GPU | 4 张 |
| vLLM max sequences | 8 |
| AgentLoop workers | 2 |

`use_fused_kernels=true` 和 `use_liger=true` 是正式显存方案。两者改变算子、吞吐和
显存占用，但不改变 BPO 分叉、Reward、advantage 或评测定义。由于 fused kernel 与
Liger 依赖固定软件版本，仍必须用 1-step smoke 验证 loss 有限、四卡工作且没有 OOM。

全词表 logits 只在单 token entropy probe 中产生，并立即在 vLLM 服务进程中归约成
一个标量；完整向量不会进入 AgentLoop output 或训练 batch。

## 5. 安装补丁与预检

在项目 Python 环境中执行：

```bash
cd ~/shopping-grpo
export PYTHONPATH=./src
export GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo/bin/python

"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py
"$GRPO_PYTHON" scripts/apply_verl_bpo_patch.py
```

BPO 补丁只修改固定 `verl==0.8.0` 的 `vllm_async_server.py`。脚本校验官方源码
SHA256、创建备份、执行幂等检查，并拒绝未知版本。

配置 Shopper API：

```bash
read -rsp 'SHOPPER_API_KEY: ' SHOPPER_API_KEY
echo
export SHOPPER_API_KEY
export SHOPPER_BASE_URL='https://your-endpoint/compatible-mode/v1'
export SHOPPER_MODEL='deepseek-v4-flash-0731'
```

只读预检：

```bash
bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$BPO_OUTPUT" \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  --preflight-only
```

预检会拒绝非 Reward v4、错误 manifest、非四卡、非 K=4/M=1、worker 分组错误、
缺少精确熵补丁、未启用 fused kernels/Liger、动态采样补丁缺失和监控配置错误。

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
- 四条轨迹具有相同分叉前缀和相同 `bpo_branch_action`；
- 三个 clone 的 slot 不同，Shopper history 不串线；
- entropy 为有限标量，全词表向量未写入输出；
- Reward 全部为 v4，基础设施无效样本不伪装成普通失败；
- advantage 有限，工具与 padding mask 为零；
- optimizer 确实更新一次，四卡无 OOM。

## 7. 正式训练

确认 smoke 后使用全新输出目录启动默认 10 updates：

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

正式启动会写入 `shopping-bpo-run-contract-v1`，记录 Git commit、模型、数据与配置
SHA256、seed、分叉参数和两项显存配置，但不会记录 API key。

BPO 与 GRPO 不得同时占用 GPU 0–3、同一 Ray runtime 或同一批 ShopSimulator slots。
可以在 GRPO 运行期间开发 BPO 代码，但 BPO smoke 必须等 GRPO 完全退出后执行。

## 8. 评测与停止规则

10 updates 后只在冻结 dev500 的三个面板评测候选 checkpoint：

- gap + ask enabled；
- gap + ask disabled；
- complete + ask enabled。

优先比较严格购买成功率、Reward v4 有效率和 mean terminal utility，再检查提问率、
重复提问、guard rejection、上下文溢出和基础设施无效率。方法选择阶段不得使用
final200；只有最终方法和 checkpoint 完全冻结后，才执行一次正式 final200。

出现以下任一情况立即停止，不能解释为算法效果：快照状态不一致、sibling 前缀不同、
全词表概率质量不约等于 1、Reward 非 v4、数据重叠、OOM、NaN/Inf、连续动态采样
跳过或 ShopSimulator slot 泄漏。
