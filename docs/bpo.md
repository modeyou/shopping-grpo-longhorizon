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

这里有两份用途不同、不得混用的环境清单：`data/environment-v4.json` 是 formal-v2
Parquet 已冻结绑定的数据/Reward 契约，保持原字节与 SHA256 不变；
`data/environment-bpo-v1.json` 是 BPO 运行时契约，额外绑定快照接口修改后的
`pack_api.py` 和 `snapshot_store.py`。训练入口先用前者验证数据血缘，再用后者验证
当前运行时代码，不能通过改写 formal manifest 或跳过哈希来迁移数据。

## 2. 冻结的算法方案

实现采用完整 one-boundary BPO，而不是 BPO-lite：

1. 每个 prompt 先生成一条 backbone trajectory；
2. 每次 assistant action 生成前，在 ShopSimulator 服务端保存快照；
3. 在与快照完全相同的 action 起始边界执行一次 `max_tokens=1`、`top_p=1`、
   `temperature=1` 的全词表 log-prob 探针，计算首 token 的精确 Shannon entropy；
4. 再从这个边界正常生成 backbone action；熵探针不得预先条件化任何 backbone action token；
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

### 2.1 Backbone 从哪里来

Backbone 不是离线人工构造的教师轨迹，也不是固定的数据文件。每次训练时，当前
actor 从 Parquet 中的一条 prompt 出发，在当时的策略参数、ShopSimulator 状态和
Shopper 响应下在线生成一条完整轨迹。它既用于提供一条 sibling continuation，也用于
发现本题最值得探索的分叉位置。因此 actor 更新以后，即使再次抽到同一任务，backbone
及其分叉点也可能改变。

训练数据的单位仍是 prompt/task-condition；BPO 优化与比较的单位则是由该 prompt
在线产生的一个 K=4 sibling group，而不是把 4 条轨迹视为互不相关的样本。

### 2.2 分叉点怎样确定

候选点是每次 assistant 即将生成下一次 action 的边界，例如获得搜索结果、商品详情、
Shopper 回答或其他工具 observation 之后。实现不会预先指定“必须在询问之后”或“必须在
某种工具之后”分叉，而是在 backbone 的所有有效 action 边界上测量下一 action 首 token
的精确全词表熵，选择熵最大的边界。这里严格采用论文的“first-token distribution at every
decision boundary”：熵计算、环境快照和三条 clone 使用同一个条件状态，不再跳过协议 token
后用已经生成的 backbone 前缀重新条件化。

不同 backbone 独立选择，所以不同 prompt、不同训练时刻乃至同一任务的不同在线 rollout
都可能落在不同位置。在同一个 sibling group 内，4 条 continuation 必须共享完全相同的
快照、Shopper history 和 token 前缀，只从选中的边界开始重新采样。

K=4 不是四种预定义行为类别。它是一次 backbone continuation 加三次独立随机 continuation；
它们可能表现为不同的提问、搜索、点击或购买路径，也可能偶然重复。我们比较的是同一起点
下实际采样出的四个终局回报。

### 2.3 Backbone 的过滤与异常处理

正式数据首先受 manifest、Reward v4 可达性、train/validation/final 零重叠约束。运行时还受
4096 prompt、20480 response、35 个购物步骤、最多 2 次 Shopper 提问以及 tool schema、guard、
session 隔离约束。正常的错误动作、无购买、错误购买和终止失败属于策略结果，应保留为训练
信号；HTTP/API 故障、状态恢复不一致、快照泄漏、非 Reward v4 和其他基础设施无效组不得当作
负奖励进入优化。

若一个 group 的四个 return 没有可辨别差异，组内相对 advantage 为零。动态采样可以继续抽取
新 group，但最多尝试 3 个生成 batch，并在连续 10 次无法形成有效更新时停止，避免无界重试。

### 2.4 BPO、GRPO、critic 与 PPO 的关系

当前两条路线都没有训练 critic/value model，也都不是用绝对 return 直接更新 actor。二者都在
组内构造相对 advantage，并使用相同类型的 PPO clipped policy loss 稳定 LoRA actor 更新：

| 项目 | GRPO | 当前 BPO |
|---|---|---|
| critic/value model | 无 | 无 |
| 比较组 | 同一原始 prompt 的 K 条完整轨迹 | 同一分叉快照的 K 条 continuation |
| baseline | 组内相对基线 | sibling leave-one-out 均值 |
| 主要归因范围 | 整条轨迹 | 分叉 action 及分叉后的差异行为 |
| actor loss 外壳 | PPO clip | PPO clip |

所以 BPO 的主要创新在 rollout 拓扑与信用分配，不是另换一个带 critic 的优化器。配置中的
`upstream_lambda=0.95` 会给分叉前 action 写入衰减权重；但同组共享前缀完全相同，LOO advantage
之和为零时，这部分梯度在理想条件下会大幅抵消。第一版应把它理解为保留的上游信用接口，
实际主要学习信号仍来自分叉 action 和分叉后的不同 continuation，不能夸大前缀传播效果。

### 2.5 K=4、显存和速度预期

K 增大不仅增加采样，还增加完整序列的 log-prob、reference forward 和 actor backward，显存与
训练计算都会上升。因此四张 24GB 4090 的正式第一版固定 K=4，与 GRPO 使用同一 return budget，
保证比较公平并控制 OOM 风险；不是因为 BPO 理论上只能使用 4 条。

若平均分叉发生在轨迹比例 `f` 处，纯环境生成长度可粗略写为
`L + 3(1-f)L = (4-3f)L`，小于 GRPO 的约 `4L`。不过 BPO 还要付出逐 action 单 token 全词表熵
探针、服务端深拷贝/恢复和先完成 backbone 再生成 clones 的串行依赖；训练阶段又仍需处理四条
重建后的完整序列。因此它可能减少重复的前缀环境交互和 Shopper API 调用，但不保证 wall-clock
一定更快。正式目标是提高关键决策附近的样本效率，并在相同 K=4 预算下提升完整购买成功率。

只有 10 个 optimizer updates 的首轮属于“可运行性 + 方向性”实验，足以淘汰错误实现、观察
分叉分布和判断相对 GRPO 是否有信号，但不足以单独证明算法稳定优越。是否扩展训练应由冻结
dev500 三面板结果、基础设施有效率和分叉诊断共同决定。

## 3. 快照与状态隔离

快照只保存在 ShopSimulator 服务端，训练进程只接收随机 `snapshot_id`。快照覆盖：

- 服务端 session 与 Reward tracker；
- 浏览器 URL、页面内容与可执行动作；
- 环境历史、购物车和任务状态；
- trajectory 本地运行状态与已澄清约束；
- Shopper history、问题次数和调用计数。

每个 clone 会申请独立 ShopSimulator slot。正式 train batch 为 2，每组最多同时占用
三个 clone；backbone slot 会在 clone 前释放，允许被某个 clone 安全复用。两个 prompt
并行恢复时最多有 6 个 clone slot。AgentLoop worker 固定为 2，
确保 veRL 按 K=4 重复后，每组 sibling 不会被切分给不同 worker。

预检会实际执行一次 source snapshot、三次 clone 和四次相同搜索动作，只有四条恢复路径产生
字节等价的去 session-id transition 时才通过。完整 rollout 还会在两个位置 fail-closed：

- AgentLoop 返回前验证 sibling 0–3、同一 group/branch/entropy、完全相同的分叉前 token 与 mask，
  以及三个并发 clone 的独立 lease；
- advantage 计算前在 veRL `DataProto` 上再次验证这些字段，并输出每组 return、LOO advantage
  及其零和审计；任一不一致都不会进入 optimizer。

### 3.1 与原论文逐项对齐及项目适配

论文算法要求：先完成一条 backbone；在 action 边界按首 token 全词表熵选点；从同一 `s_t`
恢复并得到 K 个条件独立 sibling；用其他 K−1 个 return 的均值作 LOO baseline；以
`lambda=0.95` 向分叉前传播；最后进入 PPO clip tree loss。当前实现逐项对应这些接口。

项目与论文实验规模的差异是有意冻结的资源适配，不冒充论文原配置：论文主实验使用多分叉、
更大的 return budget、8 张 A100 和数千次更新；本项目首版固定 `M=1, K=4`、四张 4090 和
10 次更新，只用于验证 Shopping 场景下是否有方向性收益。论文实验还使用 `2e-6` 学习率、
`beta=0.05` KL 和 batch 128；本项目为显存与首轮归因固定为 `1e-6`、`beta=0`、batch 2。
`beta=0` 是论文无偏性定理明确覆盖的特例，但不是论文主实验超参数。论文没有公开可核对的官方实现仓库，
因此“对齐”以论文公式、算法 1 和 veRL 0.8 官方数据流为准，而不是依据同名但不同算法的仓库。

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
| entropy 状态 | action 起始边界的首 token 分布；不得条件化已生成 action token |
| rollout 审计 | `exact-tree-v1`，不通过则禁止 advantage/optimizer |
| upstream lambda | 0.95 |
| PPO clip | 0.20 |
| rollout temperature / top-p | 0.7 / 0.9 |
| LoRA rank / alpha | 16 / 32 |
| learning rate | `1e-6`，warmup ratio `0.03` |
| 正式 optimizer updates | 10 |
| checkpoint / validation | 每 5 updates |
| GPU | 4 张 |
| vLLM GPU memory utilization | 0.45 |
| vLLM max sequences（每个单卡引擎） | 8 |
| AgentLoop workers | 2 |

正式显存方案冻结为 `use_fused_kernels=true`、`use_liger=true`、
`use_remove_padding=true`、`gpu_memory_utilization=0.45` 和 `max_num_seqs=8`。Liger、
remove-padding 和 batch 预算沿用 GRPO A1U/B1 在同一 SFT checkpoint-325、四张 4090 上
完成 optimizer update 的已验证路径；BPO 首次走到 actor update 时又证明非 fused output head
会在长序列全词表 `log_softmax` 处 OOM，因此 BPO 额外冻结 fused output head。它们改变算子、
吞吐和显存占用，但不改变 BPO 分叉、Reward、advantage 或评测定义。仍必须由新的 1-step
smoke 验证该组合真正完成有限 loss 的 optimizer update，不能把 GRPO smoke 当作 BPO smoke。

`0.45/8` 要求四张可见 GPU 在启动前保持独占和充足余量。rollout 前的
`checkpoint_manager.update_weights` 会由 FSDP 临时召集完整参数、合并 LoRA 并同步给 vLLM；
外部 GPU 进程可能使该阶段在尚未进入 snapshot clone、四 sibling、advantage 或 optimizer update
之前就 OOM，这属于运行资源冲突而不是 BPO 算法失败。正式预检要求恰好四张可见 GPU 且每张
至少有 20 GiB 空闲；有外部任务的物理 GPU 必须通过 `CUDA_VISIBLE_DEVICES` 排除。

当四张可见卡的物理编号不连续时，Ray 仍可能返回物理 accelerator ID，而 CUDA 进程只能接受
掩码后的逻辑编号。BPO worker 启动钩子固定执行 `physical ID -> CUDA_VISIBLE_DEVICES 列表位置`
映射，例如 `0,2,3,4 -> 0,1,2,3`；预检会验证该映射和钩子 marker，避免把物理 GPU 4
错误地传给只有四张可见卡的进程并触发 `invalid device ordinal`。

ShopSimulator 快照只保存可恢复的权威状态。`text_to_clickable` 是从浏览器页面派生的
BeautifulSoup 对象缓存，包含循环父子引用，因此不得深拷贝；clone 恢复 `page_source` 后调用
`get_available_actions()` 重建。快照预检会先执行一次搜索再创建快照，以覆盖真实 rollout
中已经生成 clickable 对象图的状态，而不只测试刚 reset 的空缓存。

单 token entropy probe 的全词表 logits 会立即在 vLLM 服务进程中归约成一个标量；完整
向量不会进入 AgentLoop output 或训练 batch。actor update 仍需计算被选 token 的 log-prob，
因此启用 fused output head，避免为完整长序列常驻 sequence-by-vocabulary logits。
`actor.calculate_entropy=false`：BPO 已由独立的一 token 精确探针记录分叉熵，且训练 entropy
系数为 0，不要求 trainer 保存或汇总常规 actor entropy；torch fused output head 仍可能在
内部生成紧凑的逐 token entropy，这是该算子实现细节，不是 BPO 分叉熵。Liger 与 fused
output head 分工不同，二者与 remove-padding 同时开启。

`fused=true + remove-padding=false` 不属于支持配置：fused forward 返回 no-padding
`log_probs`，而 dense 分支与 PPO loss 的 `no_padding_2_padding()` 元数据无法对齐，会触发
`sequence_offsets[-1] != values.shape[0]`。因此预检必须把 fused 与 remove-padding 同时冻结为
true；不能为了规避该断言关闭 fused，也不能为了规避非 fused 全词表 OOM 关闭 remove-padding。

## 5. 安装补丁与预检

在项目 Python 环境中执行：

```bash
cd ~/shopping-grpo
export PYTHONPATH=./src
export GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo/bin/python

"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py
"$GRPO_PYTHON" scripts/apply_verl_bpo_patch.py
```

预检不再只判断补丁 marker。动态采样补丁必须精确匹配冻结的
`ray_trainer.py` SHA256；精确熵补丁必须精确匹配冻结的
`vllm_async_server.py` 唯一 SHA256；该值由冻结的原始 veRL 0.8 文件和确定性 V2
变换实时推导。V2 明确按
`0 * log(0) = 0` 处理 vLLM 对不可能 token 返回的 `logprob=-inf`，同时拒绝
`NaN/+inf`；补丁升级器只允许从 SHA256 已验证的原始备份升级。
此外，预检会在加载模型权重前，用真实 veRL `DataProto` 在 CPU 上执行一次
K=4 BPO advantage 分发，确认 `AgentLoopWorker` 与 `compute_advantage` 两个
运行时 hook 已生效，并验证 advantage/return 的形状与有限性。这样 estimator
guard、错误补丁版本和 hook 未安装会在占用四张 GPU 之前直接失败。

BPO 补丁只修改固定 `verl==0.8.0` 的 `vllm_async_server.py`。脚本校验官方源码
SHA256、创建备份、执行幂等检查，并拒绝未知版本。

动态采样补丁同样固定 `verl==0.8.0`：先应用已经过 GRPO A1U/B1 验证的 V4 patch，
再执行唯一、确定性的 estimator guard 重写，使它显式接受 `GRPO` 或 `BPO`。被接纳的
trajectory 可能来自多个 generation batch；现有补丁已在 concat 和 balance 后重新提取 reward。
BPO outcome return 对全部 token reward 求和；
`response_mask` 只控制 policy-gradient 落点，不能把位于工具/环境 token 上的终局奖励清零。
任一 K=4 sibling group 在对齐后仍无 reward 差异时直接拒绝 optimizer update。最终
`ray_trainer.py` SHA256 固定为
`a2132ecbce6ca55fcd3a61f615b925b4a0c7a2192c69cd3e4faf8046124b334b`；
旧 `9fc8...` 版本只能在存在已验证原始备份时升级，未知源码仍拒绝修改。

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
缺少精确熵补丁、显存组合不是 `fused=true + Liger=true + remove-padding=true + vLLM 0.45/8`、
四张可见 GPU 中任一卡空闲显存低于 20 GiB、动态采样补丁缺失和监控配置错误。

## 6. 1-step smoke

smoke 只验证完整链路，不作为实验结果，也不保留 checkpoint：

BPO 启动器必须独占并自行创建本地 Ray runtime。若此前为诊断手动执行过 `ray start`，先用
对应 Ray 可执行文件执行 `ray stop --force`，确认无 BPO/GRPO Ray 任务后再
`unset RAY_ADDRESS`；不得把正式训练连接到启动时缺少 BPO 环境变量的外部 Ray head。
启动器会拒绝非空 `RAY_ADDRESS`，避免远端 `TaskRunner` 到运行阶段才出现环境插值错误。

先冻结本次名称和空输出目录，并只做不加载训练权重的预检：

```bash
cd ~/shopping-grpo
export PYTHONPATH=./src
export GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3

export BPO_MODEL="$PWD/outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325"
export BPO_SMOKE_NAME="bpo-native-v4-smoke1-$(date +%Y%m%d-%H%M%S)"
export BPO_SMOKE_OUT="$PWD/outputs/models/$BPO_SMOKE_NAME"
export BPO_SMOKE_LOG="$PWD/outputs/bpo/logs/$BPO_SMOKE_NAME.log"
export BPO_SMOKE_PID_FILE="$PWD/outputs/bpo/logs/$BPO_SMOKE_NAME.pid"

mkdir -p "$BPO_SMOKE_OUT" "$(dirname "$BPO_SMOKE_LOG")"

bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$BPO_SMOKE_OUT" \
  --experiment-name "$BPO_SMOKE_NAME" \
  --logger console \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  --preflight-only
```

预检通过后，用相同变量启动真正的 1-step smoke：

```bash
nohup bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$BPO_SMOKE_OUT" \
  --experiment-name "$BPO_SMOKE_NAME" \
  --logger console \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  -- \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  > "$BPO_SMOKE_LOG" 2>&1 < /dev/null &

BPO_SMOKE_PID=$!
printf '%s\n' "$BPO_SMOKE_PID" > "$BPO_SMOKE_PID_FILE"
echo "BPO_SMOKE_PID=$BPO_SMOKE_PID"
tail -F "$BPO_SMOKE_LOG"
```

另开终端实时查看：

```bash
watch -n 5 '
date
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
echo "--- BPO process ---"
ps -eo pid,ppid,etime,cmd | grep -E "train_bpo.py|shopping_grpo.training.bpo|verl.trainer.main_ppo" | grep -v grep || true
echo "--- latest diagnostics ---"
tail -n 5 '"$BPO_SMOKE_OUT"'/training_diagnostics.jsonl 2>/dev/null || true
'
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
SHA256、seed、分叉参数、数据环境清单、BPO 运行时清单和显存配置，但不会记录 API key。

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

## 9. 与 GRPO 分支安全切换

不要在当前 GRPO 分支上直接执行 `git pull origin feat/bpo2`，否则 Git 会尝试把 BPO
提交合并进当前分支。先确认 tracked worktree 干净，再切换目标分支：

```bash
git status --short --branch
git fetch origin
git switch feat/bpo2
git pull --ff-only origin feat/bpo2
```

从 BPO 返回 GRPO 时，先停止 BPO/Ray 进程，并在仍位于 `feat/bpo2` 时恢复仓库外的
veRL entropy patch，然后再切换代码：

```bash
"$GRPO_PYTHON" scripts/apply_verl_bpo_patch.py --restore
git switch feat/multiturn-clarification-agent
git pull --ff-only origin feat/multiturn-clarification-agent
"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py
```

反向进入 BPO 时，在切到 `feat/bpo2` 后重新应用两个补丁。`data/`、`outputs/` 中被
Git 忽略的本地产物通常会跨分支保留；若未跟踪文件与目标分支文件同名，Git 会拒绝切换，
此时必须先核对和归档，不能强制覆盖。

ShopSimulator 进程会继续使用启动时已加载的 Python 代码，不会随 `git switch` 自动刷新。
两个方向切换后都要重启服务：BPO 需要含 `snapshot`、`clone`、`drop_snapshot` 的版本；
GRPO 为严格复现也应重启到对应分支代码。切换 Git 分支不等于切换完整运行环境。
