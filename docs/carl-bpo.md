# CARL-BPO 改进方案与验证标准

> **名称**：CARL-BPO（Completion-Aligned Root–Local Branching Policy Optimization）
> **中文名**：完成率对齐的 Root–Local 分支策略优化
> **状态**：CARL-BPO v1 作为历史对照；v2 已完成代码实现，等待服务器 preflight 与再次确认后运行
> **起点模型**：SFT `checkpoint-325`

本文把正式 `full-bpo-v1` 的失败分析落成一套做减法后的 RL 改进方案。CARL-BPO
不是 BPO 原论文的逐字复现，也不声称是新的通用算法；它是针对 shopping-grpo
验收目标和 SFT-325 能力缺口构造的宏观/局部双层训练方案：

```text
完成率对齐的训练 return
          ↓
Root K=4 全局组 + Local K=4 局部组
          ↓
目标对比分级筛选
          ↓
Root episode LOO + Local suffix-only sibling LOO
          ↓
标准 PPO token loss，Root/Local group 等权
```

本文定义方案、实现约束和验证门槛。第 1–16 节记录 CARL-BPO v1 的设计与当前实现；
第 17 节记录正式运行暴露的问题及已经确认的 v2 替换方案。本文不会自动启动训练、合并模型，
也不会使用 `final200`。

## 1. 问题与目标

正式运行 `full-bpo-v1` 从 SFT `checkpoint-325` 开始，完成 200 个 optimizer steps、
400 棵有效树和 1600 个 sibling returns，但冻结 dev500 上：

```text
BPO total strict = 0.645
SFT total strict = 0.647
```

训练链路具有非零梯度和参数更新，因此问题不能简化为“优化器没有工作”。现有证据更支持：

1. 原生 Reward v4 的连续 utility 与最终验收优先级没有完全对齐；
2. 单一 Local entropy 分叉不能覆盖首轮搜索和完整策略；
3. SFT-325 的主要缺口集中在商品选择、规格选择以及搜索/恢复，而不是工具格式；
4. 大量有效树只比较 failure utility，消耗了有限 optimizer slots；
5. 共享前缀上的 Local LOO 信用缺乏有效的非抵消梯度；
6. 不能仅凭训练强度不足解释训练前后行为指标基本不变。

CARL-BPO 的两个并列主要验收指标是：

```text
gold_purchase
gold_purchase + valid_alternative_purchase
```

其中完整任务完成率优先，gold 命中保留额外偏好。`mean utility`、失败类型、轨迹长度和
工具效率用于辅助判断，不能替代两个主要指标。

## 2. 设计原则

每个进入方案的组件必须同时满足：

1. 对应一个已经观察到的失败模式；
2. 能说明它在哪个数据或梯度环节产生作用；
3. 不与另一个组件重复表达同一目标；
4. 能通过离线重算、单元测试或运行诊断验证；
5. 不能为了使用仓库已有功能而默认启用。

因此，第一版不引入 learned critic、PRM、MCTS、完整 `bounded-v1`、action-mean loss、
正负 advantage 重平衡或同一 backbone 的强制 `M=2`。

## 3. 训练目标与 return

### 3.1 Reward v4 保持为事实判定器

Reward v4 继续原样输出：

- `reward_type`；
- `reward_valid`；
- `purchase_success`；
- `terminal_utility`；
- 原子约束、价格、证据覆盖和失败原因。

评测仍直接使用 Reward v4，不把 CARL-BPO 的训练 return 当作新的评测 Reward。

### 3.2 Completion-aligned train return

只在训练数据流中增加 `train_return`：

\[
S(\tau)=
\begin{cases}
1.25, & \texttt{gold\_purchase} \\
1.00, & \texttt{valid\_alternative\_purchase} \\
0.10\,\mathrm{clip}(U,-1,1), & \text{其他可验证正常终局} \\
-0.075, & \text{模型导致的未完成}
\end{cases}
\]

其中 `U` 是 Reward v4 的原生 `terminal_utility`。基础设施错误、Reward 不可验证、
结构审计失败和无法恢复的 snapshot 错误不得变成负样本，必须继续排除。

按正式采样中观测到的 Reward v4 值，训练分数应为：

| 终局类型 | `train_return` |
|---|---:|
| `gold_purchase` | `1.25` |
| `valid_alternative_purchase` | `1.00` |
| `partial_alternative_purchase` | 约 `[-0.025, 0.0215]` |
| `max_steps` | `-0.05` |
| `repeat_loop` | `-0.065` |
| 模型未完成 | `-0.075` |
| `wrong_purchase` | `-0.085` |

这一映射保证成功与失败之间存在明确间隔，同时保留失败内部的少量 utility 排序。

### 3.3 不采用完整 bounded-v1

现有正式日志的离线反事实重算显示：

- 1600 条 accepted rollout 中有 77 条 model failure；
- 210 条正常终局会受到 `bounded-v1` 行为惩罚；
- 经过 `0.1` utility 权重后，受影响正常轨迹的平均分数变化约为 `0.001995`，最大为 `0.01`；
- 没有 model failure 的 group 中，行为惩罚造成的 sibling 两两排序翻转为 0；
- 观察到的 29 个排序翻转全部来自把 model failure 从 0 改为 `-0.75`；
- 额外提问惩罚在 accepted rollout 中没有触发。

因此只保留“模型未完成必须为有限负例”的语义，不启用额外提问、Shopper 拒绝、Guard
拒绝和重复动作微惩罚。上述行为继续记录为诊断指标。

### 3.4 Advantage 不按组内标准差归一

保持：

```yaml
norm_adv_by_std_in_grpo: false
```

否则 failure utility 的微小差异会被重新放大到与 completion contrast 接近的尺度，破坏
训练 return 的优先级设计。

## 4. Root–Local 采样拓扑

每个 optimizer step 使用两个独立 prompt slot；每个 slot 只包含一个 `M=1, K=4`
comparison group：

```text
prompt A → Root group  → 4 条从初始状态独立生成的完整轨迹
prompt B → Local group → 1 条 backbone + 同一 snapshot 的 3 条 sibling
```

因此每步仍为：

```text
2 groups × K=4 = 8 terminal returns
```

Root 和 Local 不要求来自同一任务或同一 backbone。批内应使用两个独立数据行，并记录
task/condition 标识，避免误把两个组拼成论文意义上的多分叉树。

### 4.1 Root group

Root group 在初始环境状态、相同 prompt 和相同公开 Shopper opening 下采样四条独立完整
轨迹。四条轨迹从第一个模型 action 开始即可不同。

Root group 负责：

- 首轮搜索词；
- 是否需要提问；
- 初始搜索、打开商品和完整购买策略；
- 早期错误对最终结果的全局信用；
- Local 分叉无法训练到的分叉前动作。

Root group 本质上是以初始状态为锚点的 episode-level RLOO group。

### 4.2 Local group

Local group 保留服务端 snapshot clone：先生成当前 actor 的一条 backbone，在选定动作边界
恢复完全相同的环境、Shopper history、可见 observation、上下文和 token 前缀，再生成三条
sibling continuation。backbone continuation 是第四条 sibling。

Local group 负责同一状态下的动作比较，避免把不同搜索历史和不同页面状态混入 baseline。

### 4.3 `M` 保持为 1

CARL-BPO 第一版不把 Root 和 Local 绑定到同一 backbone，因此每棵树 `M=1`。若两个
prompt 都执行 `M=2, K=4`，独立轨迹数会从每步 8 增至 14，约增加 75% rollout 成本，
但在 Local 前缀 mask 为 0 后不会新增一种 advantage。

如果以后增加预算，优先增加独立 task-condition 的有效 group 数或 optimizer steps，而不是
先增加同一 backbone 的分叉点数。

## 5. Local 分叉位置

Local 先按语义阶段覆盖，再在同一阶段内用 entropy 排序。初始软目标为：

| 阶段 | 软目标 | 典型决策 |
|---|---:|---|
| 商品选择 | 40% | 打开哪个商品、是否返回搜索结果 |
| 规格选择 | 35% | 颜色、尺寸、容量、套装等 option |
| 搜索与恢复 | 25% | 改写查询、换候选、错误后回退 |

实现上用 20 个 Local step 的确定性轮换（8/7/5）近似 40/35/25，统计口径仍是滚动目标而不是单步硬配额。
某阶段没有合法边界时可以回退到其他阶段，但必须
记录 `stage_unavailable` 和实际分布。终局 `buy_now` 不单设配额；Root 已覆盖完整购买策略，
Local 只有在它属于可用高价值边界时才选择。

entropy 只回答“同一语义阶段内选择哪个边界”，不单独决定所有分叉预算。具体要求：

1. 分叉必须位于模型 action 边界；
2. 不得位于 observation、padding 或纯环境 token；
3. 同阶段优先选择 entropy 更高的合法边界；
4. entropy 相同选择更早边界；
5. 无合法边界的 Local 候选不能伪装成 Root group。

## 6. 候选 group 筛选

每个结构 slot 分别维护候选，按以下层级选择：

1. **Completion contrast**：siblings 的成功完成标记不同；
2. **Gold contrast**：均完成，但存在 gold 与 valid alternative 对比；
3. **Failure utility contrast**：均未完成，但 `train_return` 非常量；
4. **Discard**：常量、invalid、unverifiable、snapshot/prefix 审计失败。

同层可以使用 `train_return` range 作为排序指标，但不能跨层用方差覆盖目标优先级。

不设置 failure-only 20% 硬上限。正式旧日志的 1732 棵候选树按新目标重算为：

| 候选类别 | 数量 |
|---|---:|
| completion contrast | 266 |
| gold contrast | 6 |
| failure/同层 utility contrast | 165 |
| constant | 1295 |

在已观察候选流中，目标 contrast 只有 272 棵，无法在不额外扩大采样的情况下让400个槽位中的
failure group 低于20%。正确策略是：在固定 generation budget 内优先目标 contrast；Root 与
当前指定语义阶段的 Local 都必须存在有效 contrast 才能更新。达到硬上限后不得用 constant、
invalid 或错误阶段 Local 填槽，也不能训练不完整 optimizer batch。

必须记录：

- `group_type=root|local`；
- Local semantic stage；
- `completion_contrast`、`gold_contrast`；
- score range 和 sibling reward types；
- candidate rank、accepted/fallback/drop reason；
- 每步 generation batches 和累计候选成本；
- unique branch actions、tool sequences 和 terminal results。

## 7. 信用分配

### 7.1 Root episode LOO

对 Root group 的四条完整轨迹：

\[
A_i^{root}=S_i-\frac{1}{K-1}\sum_{j\ne i}S_j
\]

同一轨迹的所有模型 action token 使用该 episode advantage；observation、环境 token 和 padding
保持 mask=0。

### 7.2 Local sibling LOO

对同一 snapshot 的四条 Local continuation：

\[
A_i^{local}=S_i-\frac{1}{K-1}\sum_{j\ne i}S_j
\]

Local policy mask 必须满足：

```text
分叉前全部模型 token：0
分叉 action 及其后续模型 action token：1
observation / environment / padding：0
```

Local 不传播 `upstream_lambda`。在单分叉 K=4 group 中，共享前缀相同且 LOO advantages 求和
为零，理想情况下共享 token 的梯度相互抵消；Root group 已提供真正的早期动作对比。因此继续
广播 Local advantage 只会增加实现复杂度和数值依赖。

## 8. PPO loss

保留标准 PPO clipped token objective。对 group `g` 的有效 policy token 集合 `M_g`：

\[
L_g=\frac{\sum_{(i,t)\in M_g}L^{clip}_{i,t}}
{\max(|M_g|,1)}
\]

最终：

\[
L=\frac{1}{2}L_{root}+\frac{1}{2}L_{local}
\]

不做：

- action 内 token mean；
- trajectory 内 action mean；
- 正、负 advantage 分开归一；
- 人为把一正三负重加权为一比一；
- 每组 advantage 标准差归一。

Root/Local 等权只控制两个采样单元的权重，不改变每个序列动作内部的 token policy gradient。

## 9. 第一版保持不变的训练项

除非实现前审计发现不兼容，第一版保持：

- SFT `checkpoint-325`；
- Reward v4；
- `K=4`；
- LoRA rank 16、alpha 32；
- peak learning rate `1e-6`；
- warmup 10 steps；
- cosine decay 按 500-step horizon 冻结，`min_lr_ratio=0.1`；
- PPO clip `0.2`；
- entropy coefficient `0`；
- 不增加 learned value head；
- 不增加 KL loss；
- fused kernels、remove-padding 和 Liger 只在通过 reference-equivalence 后保留。

单次正式训练上限冻结为 500 optimizer steps：

```text
500 Root groups + 500 Local groups = 1000 accepted groups
1000 groups × K=4 = 4000 accepted terminal returns
```

500 是运行与采样预算上限，不是最终 checkpoint 的默认选择。增加预算用于覆盖更多独立
task-condition 和 Local 语义阶段，不把 `M` 改为2，也不提高 peak LR。由于 cosine scheduler
依赖总步数，500-step run 在 step 200 的 LR 高于旧 200-step run；因此不能把两者的同名
checkpoint 当成相同更新强度。scheduler horizon、warmup 和 minimum LR 必须写入 run contract，
运行中不得追改。

checkpoint 固定每25个 optimizer steps 保存，确定性 validation 保持较低频率：

```text
checkpoint: 25、50、75、……、475、500
validation: step 0、10、50、100、150、200、250、300、350、400、450、500
```

step 0 只有 validation；step 10 是早期健康门槛。20个定期 checkpoint 全部保留，避免普通故障
丢失超过24个已提交更新；若120批仍不能形成完整 Root/Local pair，终止前还要额外保存最后一个
已提交 optimizer step。validation 仍按预注册规则选择模型，不因保存频率提高而增加反复挑选。
step-0 validation 继续保留：它给出同一模型哈希、同一400-row validation parquet 和同一
harness 下的 SFT-325 起点，使后续曲线能够判断净提升。结果按完整输入 contract SHA256 缓存；
精确命中时只回放标量，不重新生成400条轨迹，任一模型、数据、配置或运行实现变化都会换 key。

## 10. 明确删除或延期的组件

| 组件 | 处理 | 原因 |
|---|---|---|
| 完整 `bounded-v1` | 删除 | 行为惩罚过小且没有独立排序作用 |
| model failure 保持 0 | 修复 | 会让未完成优于部分负 utility 终局 |
| 同 backbone `M=2` | 延期 | 成本增加且不产生新的信用类型 |
| entropy-only 分叉 | 删除 | 不确定性不等于终局影响 |
| failure-only 20% 硬上限 | 删除 | 当前候选供给不支持 |
| Local upstream propagation | 删除 | 单分叉共享前缀 LOO 理想梯度抵消 |
| action-mean loss | 删除 | 改变标准序列 policy-gradient 权重 |
| 正负 advantage 等权 | 删除 | 扭曲 LOO 的数量和幅度信息 |
| 同时提高 LR | 延期 | 无法区分信号修复与更新强度影响 |
| learned critic / PRM / MCTS | 延期 | 增加模型、标注、显存和验证复杂度 |

## 11. 实现边界与落地状态

训练实现已经落地在：

```text
configs/bpo.yaml
configs/bpo_agent_loop.yaml
scripts/train_bpo.py
scripts/check_bpo_runtime.py
scripts/audit_bpo_formal_run.py
src/shopping_grpo/training/bpo/agent_loop.py
src/shopping_grpo/training/bpo/branching.py
src/shopping_grpo/training/bpo/advantage.py
src/shopping_grpo/training/bpo/reward.py
src/shopping_grpo/training/bpo/runtime.py
src/shopping_grpo/training/grpo/dynamic_sampling.py
patches/verl-0.8.0-shopping-dynamic-sampling.patch
相关 tests
```

实现不得改变 Environment v2.1、Reward v4、observation v2、tool schema v2、数据 manifest
或训练/评测零重叠契约。实现落地不构成启动训练、合并模型或运行 final200 的授权。

## 12. SwanLab 可观测性方案

SwanLab 用于实时观察、checkpoint 决策和运行后导出，但不是唯一证据源。正式结论仍以
`run_contract.json`、`training_diagnostics.jsonl`、完整日志、checkpoint 记录和冻结评测
JSON 为准。SwanLab 指标缺失、收尾状态异常或网络中断不得覆盖本地机器证据。

### 12.1 运行身份与冻结元数据

建议运行身份：

```text
project: shopping-multiturn-agentic
experiment: carl-bpo-v1-step500-r4000-seed20260823
tags: carl-bpo, root-local, reward-v4, sft325, seed-20260823
```

SwanLab config 和本地 run contract 必须共同记录：

- Git commit 和 dirty-worktree audit；
- 起点模型路径与 SHA256；
- train/validation manifest 与环境 manifest SHA256；
- CARL-BPO 配置和 AgentLoop 配置 SHA256；
- Reward v4、train-return schema/version 和 `1.25/1.00` 映射；
- Root/Local 各1组、`M=1`、`K=4`；
- seed、500-step horizon、warmup、cosine 和 minimum LR；
- checkpoint/validation steps；
- fused/remove-padding/Liger 及 reference-equivalence 结果；
- Shopper model、endpoint identity 和公开超参数，但不记录 API key。

正式启动前必须完成 SwanLab API 真鉴权。仅有环境变量不算通过；登录接口返回失败或连接异常
时应在模型加载前终止。实际 run 由 veRL logger 在训练进程中创建。运行结束由主线程显式、
幂等调用 `finish()`，并核对云端最后 step、validation、checkpoint 和本地日志一致。

### 12.2 指标命名空间

底层 console 与 `training_diagnostics.jsonl` 继续使用 `bpo_*`、`carl_*` 和 veRL 原始字段，
保证正式审计不依赖云端展示别名。SwanLab 后端在 `Tracking.log()` 边界单独投影为五个顶级
命名空间；console 不投影，未进入清单的冗余训练标量不上传 SwanLab。当前稳定展示如下：

| SwanLab 板块 | 内容 | 主要来源（本地保留原名） |
|---|---|---|
| `validation/*` | gold、combined completion、效用、有效性与三个条件 | `val-shopping/*`、`val-core/*` |
| `sampling/*` | Root/Local、候选效率、contrast、Local阶段与accepted budget | `bpo_batch/*`、`bpo_sampling/*`、`group/*`、`bpo_stage/*`、`carl_budget/*` |
| `credit/*` | train return、native utility、sibling差异、branch位置与熵 | `reward/*`、`bpo_return/*`、`bpo_branch/*` |
| `optimization/*` | loss、LR、grad norm、KL、clip、entropy、更新/跳过状态 | `actor/*`、`training/*`、`shopping_dynamic_sampling/*` |
| `runtime/*` | rollout/token、环境与Shopper调用、timing与perf | `bpo_cost/*`、`rollout/*`、`timing_s/*`、`perf/*` |

不得为了名字整齐重复上传同一批高频标量。SwanLab 保存聚合曲线；逐树 return、LOO、mask、
prefix hash、梯度张量计数和参数 delta 以 `training_diagnostics.jsonl` 与完整日志为权威证据。

#### A. 预算与成本：`sampling/*`、`runtime/*`

- optimizer step；
- accepted Root/Local groups 单步与累计值；
- accepted terminal returns 单步与累计值；
- generated candidate groups、实际生成轨迹和 response tokens；
- environment transitions、Shopper API calls；
- step wall time、generation time、validation time；
- seconds to first accepted group 和 full Root/Local batch。

SwanLab 必须同时显示 accepted budget 与实际 generated cost。`R4000` 只表示进入 optimizer 的
4000 个 returns，不能隐藏被筛掉候选所消耗的 rollout、token 和外部 API 成本。

#### B. 采样组成：`sampling/*`

- candidate groups 总量与每步 accepted Root/Local 数量；
- completion、gold、failure、constant 和 invalid candidate 数量；
- goal-contrast accepted share；
- generation batches per step、max-batch hit 和 slow-batch warning；
- sibling train-return range、unique branch actions、unique tool sequences；
- Root/Local 的累计 accepted 数量。

当前补丁没有稳定导出“按 Root/Local 拆分的全部候选数”，因此不能从总 candidate groups 反推
两类 acceptance rate；需要时应从本地 rollout diagnostics 离线统计，不能把 accepted 50/50
误写成候选供给50/50。

#### C. Local 阶段：`sampling/*`、`credit/*`

对 `product`、`option`、`search_recovery` 分别记录：

- accepted group 数与由历史曲线计算的 rolling accepted share；
- `stage_unavailable` 和 cross-stage fallback 数；
- entropy、分叉相对位置和 prefix step 分布。

面板同时显示软目标 `40/35/25` 与实际滚动比例，不能只记录最终累计值。

#### D. Return 与信用：`credit/*` 与本地 LOO 审计

- Reward v4 native utility 与 CARL `train_return` 分开记录；
- gold、valid alternative、partial、wrong、repeat、model failure 计数；
- Root/Local train-return mean/min/max/range；
- completion、gold、failure contrast 的绝对 LOO advantage mass及share，由逐组 return 和
  `contrast_type` 离线重算；
- Root/Local 非零 advantage group 和 token 比例；
- 每组 `loo_sum_abs_max`；
- 正、负 advantage mass只作诊断，不用于重加权。

不得把训练 return 命名为 Reward v4 total，避免训练目标与评测事实判定混淆。

#### E. Mask 与 loss：本地 actor-batch 审计与 `optimization/*`

- Root first-action trainable coverage；
- Root/Local active policy tokens；
- Local prefix nonzero policy tokens；
- observation/padding nonzero policy tokens；
- 联合 batch 的标准 token-mean actor loss、PPO ratio、clipfrac 和 approx KL；
- Root/Local 等权由两组 policy-weight sum、mask support 和单元测试审计，不伪造两个独立
  fused minibatch loss；
- fused/reference loss difference（只在审计步骤记录）。

`approx KL` 必须标注为新旧policy的PPO batch诊断，不能写成对SFT-325的reference KL。
若增加只读的policy-to-SFT drift指标，应使用独立命名 `carl_drift/sft_reference_kl`。

#### F. 优化器：`optimization/*` 与本地 optimizer audit

- actual learning rate及scheduler progress；
- gradient norm；
- 首个合格更新的 nonzero-gradient parameter count 和 changed-parameter count；
- clipfrac与gradient norm的滚动统计；
- skipped update、NaN/Inf和overflow计数。

参数级梯度与 delta 审计只在首次更新及首个正学习率更新执行，避免每步复制 FSDP 参数造成额外
开销；普通 step 使用 veRL 聚合优化指标。当前审计记录 changed-parameter tensor count，不宣称
提供每步 LoRA delta norm。

500-step horizon下必须画出真实LR，而不是只在run config中记录peak `1e-6`。

#### G. Validation：`validation/*`

在 step `0/10/50/.../500` 记录：

- `summary/strict_success_rate`，对应有效 `gold_purchase`；
- `summary/purchase_success_rate` 与其等价别名 `summary/combined_completion_rate`，对应
  `gold_purchase + valid_alternative_purchase`；
- native terminal utility；
- reward-valid、done、invalid、unverifiable和model failure；
- partial、wrong、repeat、max-steps；
- 平均steps/actions和Shopper questions；
- gap-ask-enabled、gap-ask-disabled、complete-ask-enabled三个条件分项。

validation 使用冻结400-row parquet、`n=1`、temperature 0确定性轨迹，不生成CARL树，也不把
validation结果混入on-policy return。

#### H. 系统资源：SwanLab system metrics

保留每张GPU的utilization、显存、温度和功耗，以及CPU、内存、磁盘和网络。系统指标用于解释
吞吐或故障，不能作为算法效果证据。

### 12.3 面板布局

SwanLab 官方按第一个 `/` 前缀自动分组。当前 tracking patch 只向 SwanLab 上传以下五个顶级
前缀，因此新 run 会自动生成五个自定义指标板块；SwanLab 自带的 GPU/CPU system 板块仍独立
存在，不计入这五个自定义板块：

1. **Primary validation**：combined、gold和三个条件分项；
2. **Sampling contract**：Root/Local、contrast组成、fallback、阶段分布和采样成本；
3. **Credit contract**：advantage mass、LOO sum、mask覆盖和Root/Local loss；
4. **Optimization health**：LR、loss、ratio、clipfrac、KL、gradient和参数delta；
5. **Runtime cost**：tokens、环境/Shopper调用、step时间和GPU/CPU资源。

主要 validation 曲线不得和高频 training reward 共用纵轴；native utility 与 train return
也不得共用纵轴，图例必须明确标注二者含义。

### 12.4 记录频率与高基数数据

- 每个optimizer step记录聚合训练标量；
- 每个generation batch只累加计数，step结束后向SwanLab提交聚合值；
- validation和checkpoint按冻结节点记录；
- 前1步和每50步记录一次reference/mask扩展审计；
- per-task、per-group、prefix hash、完整action和错误文本保留在本地
  `training_diagnostics.jsonl`，不作为SwanLab高基数series上传。

SwanLab的step轴统一使用completed optimizer step；candidate generation batch使用单独计数指标，
不得伪装成global step，validation也必须落在对应checkpoint step。

### 12.5 SwanLab告警与决策门槛

以下条件视为阻断性错误，并同时写入本地诊断：

```text
Root/Local不完整batch
constant/invalid group进入optimizer
Local prefix或observation出现非零policy token
Root first-action trainable coverage < 100%
LOO sum abs max > 1e-6
skipped update、NaN/Inf
首步无非零gradient，或首个正学习率step无参数delta
```

以下为窗口告警，不单凭一条曲线自动宣告训练失败：

- goal-contrast group share低于60%；
- goal-contrast advantage-mass share低于90%；
- 至少100个Local groups后，阶段比例偏离 `40/35/25` 超过10个百分点；
- 连续命中 `max_num_gen_batches` 或full-batch时间显著增长；
- clipfrac、KL或gradient norm相对前50步基线出现持续数量级变化；
- 两个连续validation节点的combined和gold都比此前最佳点估计低超过1pp。

告警只触发检查或预注册的暂停门槛，不允许查看曲线后临时改变Reward、LR、分叉比例或训练上限。

### 12.6 运行后导出与完整性审计

run结束后必须用 `scripts/export_swanlab_run_metrics.py --all-custom` 导出：

```text
outputs/analysis/<run-tag>/swanlab-history.json
outputs/analysis/<run-tag>/swanlab-analysis.md
```

history exporter 已把 decision steps 从旧 `0/10/50/100/150/200` 扩展到
`0/10/50/.../500`，重要指标过滤器同时识别 `bpo_*` 与 `carl_*`。

完整性审计要求：

- SwanLab最后completed step等于本地optimizer记录；
- checkpoint和validation节点无缺失；
- 累计Root/Local groups均为500，accepted returns为4000；
- 云端summary与本地最终validation一致；
- 如云端run状态异常但本地完成，必须记录最后一致step和缺失区间，不能静默修补历史。

## 13. 验证方法与通过标准

验证按阶段进行；前一阶段失败时不得进入下一阶段。

### V0：静态设计验证

不加载模型、不访问服务器。

检查：

1. 遍历所有 Reward v4 terminal types，计算 `train_return`；
2. 验证 `gold > valid alternative > any failure`；
3. 验证 normal failure 仍保持预期 utility 排序；
4. 验证 infra-invalid 和 unverifiable 被排除；
5. 改变 questions/rejections/repeats 等行为字段不得改变 `train_return`；
6. model failure 必须稳定映射为 `-0.075`。

通过标准：全部为确定性断言，任何一项失败即阻断实施。

### V1：正式旧日志离线重放

输入为现有 `results-bpo/training_diagnostics.jsonl`，只读重算，不生成新轨迹。

检查：

1. 重新得到 400 accepted groups、1600 accepted returns；
2. 重算 completion/gold/failure/constant 类别；
3. 重算每组 LOO advantage 和绝对 advantage mass；
4. 模拟新的分级筛选和 generation-budget fallback；
5. 报告目标 contrast 的 group share、advantage-mass share 和额外采样需求；
6. 验证行为 shaping 删除后，除 model failure 外不会发生目标外排序变化。

当前旧树按新 return 重算的参考值为：

```text
completion contrast groups = 249
gold contrast groups       = 6
failure groups             = 145
completion + gold 的绝对 LOO advantage mass share ≈ 98.10%
```

通过标准：重放脚本输出应与上述参考值一致；浮点聚合允许 `1e-6` 绝对误差。若实现方案无法解释
任何差异，必须先修复审计，不进入 runtime test。

### V2：单元与属性测试

必须覆盖：

- Root K=4 从相同初始公开状态生成但 continuation 独立；
- Local K=4 的 prompt、snapshot、Shopper history、prefix hash 完全一致；
- 每组 sibling index 恰好为 `0..3`；
- Root mask 覆盖全部模型 action，Local prefix mask 全零；
- observation、环境 token 和 padding 始终 mask=0；
- 每组 LOO advantage 求和绝对值小于 `1e-6`；
- constant group 不进入 optimizer；
- invalid/unverifiable 不成为负样本；
- Root/Local loss 各自归一后严格等权；
- action 长度变化不会触发额外的 action-level 重加权；
- dynamic sampling 达到预算上限时执行已定义 fallback，不形成不完整 batch。

通过标准：相关测试全部通过，不允许 xfail 或跳过核心路径。

### V3：reference loss 与 kernel 等价验证

用同一小批固定输入分别运行：

1. unfused、无 remove-padding、无 Liger 的 reference loss；
2. 正式 fused/remove-padding/Liger 路径。

比较：

- active token 数；
- Root/Local group loss；
- 总 loss；
- LoRA 参数梯度方向和梯度范数；
- optimizer step 后的参数 delta。

推荐通过门槛：

```text
mask/token support            完全一致
loss absolute difference      <= 1e-3（bf16）
gradient cosine similarity    >= 0.999
gradient norm relative error  <= 1%
两条路径均有非零参数 delta
```

不满足时优先使用 reference 路径定位，不得以“loss 有限”替代等价验证。

### V4：正式运行的启动硬门槛

不再要求单独启动 1-step smoke。正式 SwanLab run 在 step-0 validation 后直接进入训练，并把
最初更新作为硬门槛；无需重复初始化模型、Ray、ShopSimulator 或外部日志。

必须观察到：

```text
accepted Root groups  = 1
accepted Local groups = 1
K per group           = 4
terminal returns      = 8
skipped updates       = 0
```

同时要求：

- Root 和 Local 都存在非零 advantage；
- Root 首动作具有非零 policy mask；
- Local 分叉前 policy mask 为 0，分叉及后缀非零；
- 两组 LOO sum 均小于 `1e-6`；
- loss、ratio、KL 诊断、gradient norm 均为有限值；
- 至少一个 LoRA 参数具有非零梯度和非零更新；
- snapshot clone、Shopper state、prefix/token 审计全部通过；
- 无 infrastructure-invalid 和 reward-unverifiable 被送入 optimizer。

结构、采样、mask、LOO 和有限性契约在第1个 optimizer step 验证，并继续作为每步不变量。
非零梯度在第1步硬检查。由于10-step warmup的零索引首步学习率可以为0，parameter delta 在
第1个正学习率 step 硬检查，通常是紧接的下一步；通过后停止昂贵的参数快照审计。任一核心
条件失败都会抛出异常并终止正式 run，不允许只写 warning 后继续训练。

### V5：单次正式训练的在线门槛

正式 run contract 必须在启动前冻结：seed、step 上限、checkpoint/validation 间隔、数据
manifest、模型哈希、Reward profile、Root/Local 比例和筛选规则。运行中不得根据曲线临时改参。

每个 checkpoint 窗口报告：

- Root/Local accepted group 各占 50%；
- completion、gold、failure contrast 数量；
- 目标 contrast 的绝对 LOO advantage-mass share；
- Local stage 的目标/实际分布和 unavailable/fallback 次数；
- candidate acceptance rate 和 generation batches；
- active policy tokens、正负 advantage mass、clipfrac、KL、gradient norm；
- reward type、purchase success、strict、invalid 和 model failure；
- 分叉位置、动作/工具序列多样性和轨迹长度。

训练健康门槛：

```text
每步完整 Root+Local batch                  100%
constant/invalid group 进入 optimizer       0
目标 contrast 的绝对 advantage mass share   >= 90%，目标 >= 95%
Local prefix 非零 policy token              0
Root 首动作可训练覆盖率                      100%
skipped optimizer updates                    0
NaN/Inf                                      0
```

目标 contrast group share 不设硬失败比例，但必须达到旧运行的 `63.75%` 作为参考下限，目标为
`>=70%`。低于60%应暂停并审计采样；`60%–63.75%` 只能继续到下一个预注册 checkpoint，不能
宣称筛选已改善。

Local 阶段比例按至少100个 Local groups 的滚动窗口检查；与 `40/35/25` 的偏差超过10个百分点
时必须说明是 stage unavailable、候选不足还是选择器错误，不能静默偏移。

### V6：checkpoint 选择

不在 dev500 上反复挑 checkpoint。checkpoint 只使用冻结的训练 validation parquet，按启动前
写入 run contract 的规则选择：

1. 先最大化 `gold + valid alternative` completion；
2. completion 相同或处于预设容差内时，选择 gold 更高者；
3. 两项接近时选择 invalid 更低、模型未完成更少者；
4. 仍相同则选择更早 checkpoint。

只允许选中的一个 checkpoint 进入一次 dev500 配对评测。训练最后一步不自动等于最佳模型，
也不得自动使用 `final200`。

### V7：SFT-325 配对验收

CARL-BPO 与 SFT checkpoint-325 必须使用完全相同的：

- dev500 task-condition 集合；
- Environment/Reward/observation/tool schema；
- Shopper 设置、seed 和 ask 条件；
- evaluation harness 和超参数。

主要报告：

```text
Δ combined = Δ(gold_purchase + valid_alternative_purchase)
Δ gold     = Δ(gold_purchase)
```

必须进行逐任务配对统计，并给出 paired bootstrap 95% CI；二元成功还应报告 McNemar 检验及
gain/loss 转移表。

验收分级：

- **通过**：至少一个主要指标的 paired 95% CI 下界大于0，另一个指标的下界不低于 `-1.0pp`；
- **方向性改善但未确认**：两个点估计都大于0，但 CI 均跨0；不得宣称训练已显著提高；
- **不通过**：任一主要指标明确退化超过 `1.0pp`，或两项均无正向点估计；
- **工程无效**：提升伴随 invalid/unverifiable、模型未完成或 Reward 契约异常增加，不能视为成功。

辅助报告但不替代主要指标：

- `purchase_success`、strict 和 reward-valid；
- mean utility；
- partial、wrong、repeat、unknown/model failure；
- 商品选择错误、规格错误、搜索未覆盖和恢复失败；
- 平均 action/step 数、提问和 Guard/Shopper rejection；
- 三种 ask/complete 条件的分项结果。

只有通过 dev500 配对门槛并经用户再次明确批准后，才讨论是否执行最终评测、合并模型或更新
项目正式基线。

## 14. 已确认的产品与预算选择

第一版已确认：

1. gold 与 valid alternative 的训练分数采用 `1.25 : 1.00`；
2. 单次正式训练上限为500 optimizer steps；
3. peak LR保持 `1e-6`，按500-step horizon执行10步warmup和cosine decay；
4. checkpoint/validation只用于预注册规则选择，不默认使用step 500。

其余第一版组件不再作为自由组合的实验开关，避免再次形成无法解释的“方法缝合”。

## 15. 研究依据与项目偏差

- BPO：同 snapshot sibling baseline、分叉 rollout 和 `1+M(K-1)` 成本模型；CARL-BPO
  不采用其 Local upstream propagation，也不在第一版使用同 backbone 多分叉。
  <https://arxiv.org/abs/2607.14171>
- GiGPO：episode-level macro group 与 state-level micro group 的两级信用思想；CARL-BPO
  使用额外 Root rollout 获得 macro group，不实现跨轨迹 state hashing。
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html>
- ARPO：高 entropy tool-use boundary 的自适应探索；CARL-BPO 只把 entropy 用作语义阶段
  内部排序。
  <https://openreview.net/pdf?id=TX4k7BF6aO>
- APPO：entropy-only 不足以表示对未来结果的影响；CARL-BPO 用 SFT 能力缺口限定阶段，
  但不引入其 learned branching score。
  <https://arxiv.org/abs/2606.12384>
- POAD：action/token 粒度需要严格推导，不能用简单 action mean 替代；CARL-BPO 第一版
  保留标准 token PPO。
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc09efb501c801ed92e181e26a885c2d-Abstract-Conference.html>

## 16. 当前实现与 Linux 运行手册

本节是当前 `feat/bpo2` 的唯一可执行 CARL-BPO 手册。`bpo.md` 中的 200-step、R1600、
`M=2` 和手写 Hydra smoke override 属于已完成的 `full-bpo-v1`，不得复制到新运行。

### 16.1 继承的底层运行契约

CARL-BPO 改变采样拓扑、训练 return、mask 和组间 loss 权重，但继续继承已经验证的基础设施：

- `verl==0.8.0`，先安装 dynamic-sampling/tracking patch，再安装 entropy、XML parser 和
  fused-PPO gradient patch；
- 四张干净的24GB GPU，`CUDA_VISIBLE_DEVICES` 映射后每张至少20GiB空闲；
- `fused=true + remove-padding=true + Liger=true`，vLLM `0.45/8`；
- launcher 独占并创建本地 Ray runtime，拒绝非空 `RAY_ADDRESS`；
- ShopSimulator Environment v2.1，且服务端必须包含 snapshot/clone/session-reset 修复；
- Reward v4、observation v2、tool schema v2 和 formal-v2 manifest；
- 输出目录必须不存在或为空，诊断和正式训练不得复用目录；
- Shopper 与 SwanLab 在加载训练权重前进行真实鉴权；
- step-0 validation 使用按输入哈希寻址的 cache，不能无条件复用旧结果。

若机器上曾手动执行 `ray start`，先确认没有其他任务依赖该 Ray 集群，再使用同一 Python 环境的
`ray stop --force`，并执行 `unset RAY_ADDRESS`。不要停止不属于本次训练的 Ray 或 GPU 任务。

### 16.2 环境与补丁

```bash
cd /path/to/shopping-grpo-longhorizon

git fetch origin
git switch feat/bpo2
git pull --ff-only origin feat/bpo2

export GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset RAY_ADDRESS

export BPO_MODEL="$PWD/outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325"
export SHOPPER_MODEL=deepseek-v4-flash-0731
export SHOPPER_BASE_URL='https://your-endpoint/compatible-mode/v1'

read -rsp 'SHOPPER_API_KEY: ' SHOPPER_API_KEY
echo
export SHOPPER_API_KEY

read -rsp 'SWANLAB_API_KEY: ' SWANLAB_API_KEY
echo
export SWANLAB_API_KEY

"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py
"$GRPO_PYTHON" scripts/apply_verl_bpo_patch.py
"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py --check
"$GRPO_PYTHON" scripts/apply_verl_bpo_patch.py --check
```

补丁修改当前虚拟环境中的 veRL 源码并建立可验证备份，不属于 Git 工作区。升级 veRL、切换虚拟
环境或恢复补丁后必须重新应用并通过 `--check`。补丁脚本拒绝未知源码哈希，不得绕过。

### 16.3 不训练预检

预检会访问 ShopSimulator、Shopper API 和 SwanLab，并用临时文件验证 step-0 cache 的写入与
回放钩子；正式的内容寻址 cache 由实际 step-0 validation 生成。预检不执行 optimizer update，
也不产生候选模型。每次使用新的空输出目录：

```bash
export CARL_PREFLIGHT_NAME="carl-bpo-v1-preflight-$(date +%Y%m%d-%H%M%S)"
export CARL_PREFLIGHT_OUT="$PWD/outputs/bpo/$CARL_PREFLIGHT_NAME"

bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$CARL_PREFLIGHT_OUT" \
  --experiment-name "$CARL_PREFLIGHT_NAME" \
  --logger swanlab \
  --seed 20260823 \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  --preflight-only
```

通过标志为最后出现 `BPO runtime preflight passed`，同时此前所有 manifest、patch、snapshot、
fused gradient、scheduler、step-0 cache、GPU headroom 和 API 鉴权检查均成功。

### 16.4 可选的独立 1-step 诊断

`--diagnostic-steps 1` 入口继续保留给故障定位，但不再是正式训练的前置阶段。若人工调用，它会
建立独立 SwanLab run、关闭 validation/checkpoint，并使用与正式运行相同的启动硬门槛；正常
流程直接执行下一节。不要使用旧 `bpo.md` 的手写 Hydra override。

### 16.5 正式 500-step 训练

正式运行必须使用新的空目录。不要覆盖诊断目录，不要传 `trainer.total_training_steps`、Reward、
LR、Root/Local 比例或 checkpoint 频率 override：

```bash
export CARL_NAME="carl-bpo-v1-step500-r4000-seed20260823-$(date +%Y%m%d-%H%M%S)"
export CARL_OUT="$PWD/outputs/models/$CARL_NAME"
export CARL_LOG="$PWD/outputs/bpo/logs/$CARL_NAME.log"
export CARL_PID_FILE="${CARL_LOG%.log}.pid"

mkdir -p "$CARL_OUT" "$(dirname "$CARL_LOG")"

nohup bash scripts/bpo.sh \
  --model "$BPO_MODEL" \
  --output "$CARL_OUT" \
  --experiment-name "$CARL_NAME" \
  --logger swanlab \
  --seed 20260823 \
  --shopper-model "$SHOPPER_MODEL" \
  --shopper-base-url "$SHOPPER_BASE_URL" \
  >"$CARL_LOG" 2>&1 </dev/null &

CARL_PID=$!
printf '%s\n' "$CARL_PID" >"$CARL_PID_FILE"
echo "CARL_PID=$CARL_PID"
tail -F "$CARL_LOG"
```

默认配置已经冻结500 optimizer steps、1000 accepted groups、4000 accepted returns、10-step
warmup 和500-step cosine horizon。正式 run 会先执行或精确复用 step-0 validation，随后在同一
SwanLab run 内对首步非零梯度和首个正学习率参数 delta 执行硬门槛。通过后继续训练；失败则
立即退出。500是上限，不代表自动选择 `global_step_500`。

### 16.6 完成后审计与 SwanLab 导出

训练进程正常退出后，先审计完整运行契约；审计不会合并或导出模型：

```bash
"$GRPO_PYTHON" scripts/audit_bpo_formal_run.py \
  --output "$CARL_OUT" \
  --log "$CARL_LOG"
```

唯一完整通过标志是：

```text
CARL-BPO-N500-R4000 FORMAL RUN ACCEPTED
```

随后用 SwanLab 页面显示的真实 `username/project/run_id` 导出全部标量：

```bash
export CARL_SWAN_RUN='username/shopping-multiturn-agentic/run_id'
export CARL_ANALYSIS_OUT="$PWD/outputs/analysis/$CARL_NAME"

"$GRPO_PYTHON" scripts/export_swanlab_run_metrics.py \
  --run-path "$CARL_SWAN_RUN" \
  --output-dir "$CARL_ANALYSIS_OUT" \
  --all-custom
```

审计通过后仍只按 V6 使用冻结 validation 选择一个 checkpoint。未经再次确认，不合并模型、
不执行 dev500 配对评测，也不使用 final200。

## 17. CARL-BPO v2：目标优先候选池与可验证 Local 覆盖

本节是对当前 v1 正式运行暴露问题的确认修订。v2 不是在 v1 上增加兼容分支；实现时应删除
被替换的“首个有效 group 立即接受”、observation 字符串阶段分类和静默跨阶段 fallback 路径，
只保留本节定义的新合同。

### 17.1 当前运行证据与问题归属

当前正式运行身份为：

```text
SwanLab run: mode/shopping-multiturn-agentic/n6xjq1pd
run name: carl-bpo-v1-step500-r4000-seed20260823-20260830-001314
```

2026-08-30 11:44 的只读 SwanLab 快照显示运行处于 `RUNNING`，已完成 254/500 个 optimizer
steps、508 个 accepted groups 和 2032 个 accepted returns；所有步骤均为完整的 1 Root + 1 Local，
没有 skipped update、NaN/Inf 或数值失控。冻结在线 validation400 为：

| step | gold purchase | gold + valid alternative | mean utility |
|---:|---:|---:|---:|
| 0 | 0.6850 | 0.6925 | 0.646669 |
| 150 | 0.7050 | 0.7125 | 0.671511 |
| 200 | 0.7050 | 0.7125 | 0.673669 |
| 250 | 0.7025 | 0.7075 | 0.673013 |

step 250 相比起点仍分别高 1.75pp 和 1.50pp；相比当前最佳点只低 0.25pp 和 0.50pp，
尚未触发第 12.5 节的停止条件。v1 运行可以继续作为诊断/消融证据，但不得被描述为已经完整
验证本节的 v2 选择器。

已经确认的问题分为三类：

| 问题 | 当前证据 | 归属 |
|---|---|---|
| contrast 优先级只在单个 generation batch 内排序 | 每批只有 1 Root + 1 Local；代码遇到每类首个非恒定 group 就立即接受，不会被后续更高优先级候选替换 | v1 代码未完整实现第 6 节意图 |
| Local 实际覆盖严重偏离 40/35/25 | step 254 的目标累计为 104/90/60，实际为 product 216、option 38、search_recovery 0；124/254 个 Local 发生 fallback | v1 设计允许软 fallback，同时分类实现过窄 |
| accepted contrast 构成不可从 SwanLab 复核 | 2200 个候选中有 completion 476、gold 5、failure 260，但页面没有 508 个 accepted groups 的 contrast 构成 | 可观测性未完整落地 |

Gold contrast 的候选供给只有 5/2200。它不是 Reward 或 optimizer 故障，而是尚未解决的目标信号
稀缺问题。v2 必须保证 Gold contrast 一旦出现就不会被较早到达的普通 completion 或 failure
候选浪费。

### 17.2 v2 保持不变的合同

v2 只修复“哪些 comparison groups 进入 optimizer”和“Local 决策阶段如何获得真实覆盖”。以下
部分冻结不变，避免把数据合同修复与优化器调参混在一起：

- Reward v4 继续作为事实判定器；
- completion-aligned train return 保持 gold `1.25`、valid alternative `1.0` 及当前失败映射；
- 每步 1 Root + 1 Local，每组 `M=1, K=4`；
- Root episode LOO；
- Local suffix-only LOO，`upstream_lambda=0`；
- Root/Local 等权的标准 token-mean PPO loss；
- 当前 LoRA、LR、10-step warmup、500-step cosine horizon 和 `clip_grad=1`；
- fused/remove-padding/Liger 及已有 reference-equivalence 合同；
- 500 optimizer steps 仍只是预算上限，不预先指定最终 checkpoint。

### 17.3 每步分槽候选池

每个 optimizer step 在 actor 更新前建立两个只属于当前步骤的候选池：

```text
Root pool
Local pool
```

每个 generation batch 仍生成一个 Root group 和一个 Local group。通过 snapshot、prefix、Reward、
return 和完整性审计的 group 先进入对应 pool，不再立即进入 optimizer。每个 pool 独立维护当前
最佳候选，后续更高优先级候选必须能够替换旧候选。

正式优先级为：

```text
Gold contrast
> Completion contrast
> Failure utility contrast
> constant / invalid / unverifiable / audit failure（丢弃）
```

Gold 优先于普通 Completion 的原因是 strict gold 与 combined completion 同为主要验收指标，而
Gold contrast 极其稀少；Gold group 的 siblings 仍然都是有效完成轨迹，不是用 strict 指标替换
任务完成率。同一 contrast 层内依次比较：

1. `train_return` range；
2. unique branch actions；
3. unique tool sequences；
4. 更早生成的候选作为确定性 tie-break。

候选获取使用两段预算：

1. Root 和 Local 都获得 goal contrast（Gold 或 Completion）后立即结束采样；
2. 到第 10 批仍未形成 goal pair，则每个 pool 选择当前最佳有效候选；
3. 若任一 pool 连有效 Failure contrast 都没有，保留另一 pool 已找到的候选并继续采样；
4. 第120批仍无法形成完整 Root/Local pair 时才硬停止，不训练 pending group，并先保存最后一个
   已提交 optimizer step。

真实 rollout、环境 snapshot 和 token batch 绝不能跨 optimizer step 携带。actor 更新后旧候选已经
变成陈旧的 off-policy 数据。允许跨步骤保留的只有累计计数、阶段欠账和成本统计。

### 17.4 Local 阶段合同

v2 删除 `search_recovery` 名称，统一使用三个结构化阶段：

| 阶段 | 长期目标 | 结构化决策 |
|---|---:|---|
| `product` | 40% | 打开、切换、比较商品 |
| `option` | 35% | 颜色、尺寸、容量、套装等 option 选择 |
| `search_strategy` | 25% | 初始/改写查询、重新搜索、返回结果页、翻页和错误恢复 |

阶段必须根据该边界生成出的结构化 backbone action/tool 与明确的环境状态分类，不得再依赖
observation 是否包含 `error`、`失败`、`无效` 等自由文本。至少满足：

```text
search_products / back_to_search / prev_page -> search_strategy
select_option 及等价 option action           -> option
open_product / view_features / view_description
及等价商品查看、比较、切换 action              -> product
```

`buy_now`、`ask_shopper` 和无法识别的动作不计入三个 Local target pool；它们保留为诊断候选。
Root 已负责购买与提问的完整策略，不能用这些动作伪造 Local 的 product/search/option 覆盖。

目标阶段不再用固定 20-step 数组机械轮换。driver 在每一步根据累计目标和实际 accepted 数量选择
欠账最大的阶段；确定性 tie-break 使用 `product -> option -> search_strategy`。该调度携带的是统计
欠账，不携带旧 rollout。

Local 获取期间分别维护 target-stage pool 和 cross-stage diagnostic pool。只有 target-stage pool
可以填充当前 Local slot；跨阶段候选用于供给诊断，不能静默代替目标阶段。120 批仍没有目标阶段
有效候选时写入完整诊断并硬停止，而不是继续扩大 product 占比。

在实现硬配额前，必须先用 v1 的 `training_diagnostics.jsonl` 做离线重放。若新的结构化分类仍然
无法为某阶段提供足够候选，应增加基于任务能力标签的 stage-aware task routing；不得重新引入
静默 fallback。任务路由只决定为某一 Local target 选择哪类训练任务，不改变 Reward 或 return。

### 17.5 SwanLab 与本地审计

SwanLab 在现有五个顶级板块内增加以下低基数聚合指标：

```text
sampling/accepted_root_contrast/*
sampling/accepted_local_contrast/*
sampling/accepted_goal_contrast_share
sampling/accepted_failure_fallback_share
sampling/local_target/*
sampling/local_actual/*
sampling/local_stage_debt/*
sampling/local_cross_stage_rejected
sampling/local_acquisition_failure
sampling/reservoir_replacements
sampling/batches_to_goal_pair
```

`training_diagnostics.jsonl` 对每个候选保存 generation batch、Root/Local、target stage、actual
stage、contrast type、priority tuple、accepted/replaced/rejected、reason、最终 rank、四个 train
returns 和 LOO advantage mass。SwanLab 的候选总数与 accepted 构成必须分开命名，不能再用前者
推断后者。

### 17.6 实现前只读审计

实现代码前先完成 v1 当前运行日志的只读审计。审计不修改模型、不启动训练，只把每一步的候选
流按 v2 规则离线重放，回答：

1. v1 实际 accepted Root/Local 中 Completion、Gold、Failure 各有多少；
2. 跨批候选池会替换多少个早到的 Failure group；
3. v2 选择器的 accepted goal-contrast share；
4. 新结构化阶段定义下的 product/option/search_strategy 供给；
5. 第10批质量窗口与第120批硬上限的完整 pair 成功率、平均候选批次和最坏成本；
6. 是否需要 stage-aware task routing。

离线重放的实现准入门槛为：

```text
predicted accepted goal-contrast share >= 60%
predicted Failure fallback share <= 40%
三个 Local 阶段都有真实候选
rolling-100 Local accepted share 与 40/35/25 各自偏差 <= 10pp
平均 candidate batches <= 10
正式运行不得在120批以前因单个目标阶段供给不足丢弃已提交 optimizer 更新
Gold contrast 一旦出现，预测接受率 = 100%
```

若日志证据不足以重放新的结构化动作分类，必须明确列出缺失字段，并使用原始 rollout diagnostics
补齐；不能用 SwanLab 聚合曲线猜测。

#### 17.6.1 step-260 快照审计结果

服务器快照 `carl-bpo-v1-diagnostics-20260830-120034.tar.gz` 已通过清单 SHA-256 校验。
`training_diagnostics.jsonl` 包含 260 个完整 optimizer steps、1145 个 generation batches，JSONL
没有损坏行；另有 step 261 的未完成采样事件，以下统计只使用已经存在 `optimizer_step` 的 1–260。

v1 实际 accepted contrast 构成为：

| group | Gold | Completion | Failure | goal share |
|---|---:|---:|---:|---:|
| Root | 1 | 184 | 75 | 71.15% |
| Local | 1 | 150 | 109 | 58.08% |
| 合计 | 2 | 334 | 184 | 64.62% |

因此 v1 总体已经达到 `goal share >= 60%`，但 Local 仍有 41.92% 的 accepted groups 只比较
Failure utility。1145 个已生成候选批次中共有 5 个 Gold contrast，v1 只接受 2 个。

在不增加任何 rollout、只利用“一个 slot 已接受而另一个 slot 尚未接受”期间已经生成的同类型
候选时，按 v2 pool priority 离线选择得到：

| group | Gold | Completion | Failure | goal share |
|---|---:|---:|---:|---:|
| Root | 4 | 201 | 55 | 78.85% |
| Local | 1 | 161 | 98 | 62.31% |
| 合计 | 5 | 362 | 153 | 70.58% |

即无需新增生成成本就能把 31 个 Failure slots 替换为 goal contrast，并接受全部 5 个 Gold
contrast；另有 33 个同层候选会因 return range 或行为多样性更优而替换。该结果精确来自日志中
已经真实生成的候选，不依赖分布假设。

按结构化 backbone action 检查每条 Local backbone 的非终局决策边界，1145 条候选轨迹的阶段
可用性为：

| 阶段 | 至少存在一个该阶段边界 | 占全部 Local backbones |
|---|---:|---:|
| product | 1138 | 99.39% |
| option | 1116 | 97.47% |
| search_strategy | 1145 | 100.00% |

这说明 v1 的 `search_recovery=0` 主要不是 backbone 没有搜索/恢复边界，而是旧分类和选择逻辑
没有把结构化动作正确分槽。当前证据不支持立即增加 stage-aware task routing；v2 第一版先实现
结构化分类、每阶段 entropy retention 和硬 target pool，只有运行门槛证明供给不足时才增加路由。

用 1145 个真实批次的 Root/Local 联合候选频率作 IID 供给估计：

| 预算 | 形成 goal pair 的估计概率 | 形成任意有效 pair 的估计概率 |
|---:|---:|---:|
| 10 batches | 78.72% | 95.06% |
| 30 batches | 99.47% | 99.993% |

在 10-batch 质量窗口内，Root 至少出现一个 goal contrast 的估计概率为 95.26%，Local 为
82.65%；goal pair 的截断期望采样成本约 6.17 batches。它支持保留第17.3节的10-batch质量窗口，
但不足以证明30批可以作为硬上限。

这些概率不是精确反事实重放。v1 一旦接受首个完整 pair 就停止生成，日志没有后续本可产生的
候选；同时只保存最终选中边界的 entropy 和 suffix returns，没有保存同一 backbone 所有边界的
反事实 suffix returns。因此：

- 64.62% 和 70.58% 是精确审计结果；
- 三阶段“边界存在率”是精确的结构化 action 审计，但不等于该边界一定产生非恒定 return；
- 10/30批概率是有明确 IID 假设的供给估计，不能写成正式运行保证或硬停止依据。

审计结论为“允许进入代码实现”，但不宣布新训练已获准。v2.1 实现采用120-batch硬停止、完整
候选诊断和第 17.7 节测试；仍需通过服务器 preflight，并用首次正式运行的真实 target-stage
suffix returns 验证剩余缺口。

### 17.7 代码验证与新正式运行

实现必须添加以下确定性测试：

1. 第一批 Failure、第二批 Completion，最终接受 Completion；
2. 第一批 Completion、第二批 Gold，最终接受 Gold；
3. Root 和 Local pool 不能互相替代；
4. 更高优先级候选能够替换旧候选；
5. constant、invalid、unverifiable 永远不能进入 optimizer；
6. 三类结构化 action 的阶段分类符合第 17.4 节；
7. 非目标 Local 不能填充目标 slot；
8. 阶段欠账调度在 20、100、500 个 accepted Local 后达到 40/35/25；
9. 候选不能跨 optimizer step 复用；
10. Root/Local mask、LOO、等权 loss 与当前 reference 完全一致；
11. SwanLab accepted 指标与本地逐候选审计聚合完全一致；
12. 10批质量窗口、120批硬停止和终止前紧急 checkpoint 合同成立。

修复后的正式运行必须重新从 SFT `checkpoint-325` 开始，不能从 v1 RL checkpoint resume；否则
无法区分 v2 数据选择器的效果与 v1 已发生的更新。当前 v1 run 只作为对照和候选供给证据。
未经再次确认，不启动新训练、不合并模型、不执行 dev500 配对评测，也不使用 `final200`。

### 17.8 v2 实现落点

当前代码实现与本节方案的对应关系如下：

- `dynamic_sampling.py`：40/35/25 加权公平账本、Gold/Completion/Failure 排序、Root/Local
  reservoir 替换和 10-batch 停止判定；
- `bpo/agent_loop.py`：按实际结构化 tool call 分类 `product`、`option`、`search_strategy`；
- `bpo/runtime.py`：只接收 driver 分配的 Local target，禁止 worker 自行轮换；
- veRL dynamic-sampling patch V6：每步候选池、120-batch缺槽硬停止、终止前紧急 checkpoint、optimizer selection
  诊断、resume 后有效 return 预算恢复；
- `audit_bpo_formal_run.py`：要求 500 条 authoritative selection 记录，并验收最终
  `200/175/125` Local 覆盖；
- SwanLab 继续只投影到 `validation / sampling / credit / optimization / runtime` 五个板块，
  v2 reservoir、goal/failure 接受量和阶段覆盖统一归入 `sampling`。

### 17.9 v2 step-48 供给失败与 v2.1 修订

首个 v2 正式运行完成47次 optimizer 更新后，在未提交的第48步达到30个 generation batches。
该步共生成30个 Root 和30个 Local：Root 有14个有效候选；29个目标为 `option` 的 Local 全部
为 constant return，唯一有 contrast 的 Local 实际属于 `search_strategy`，因阶段不匹配被正确拒绝。
因此这不是环境并发、snapshot clone、invalid rollout 或 optimizer 故障，而是硬目标 Local 的
低概率供给尾部事件。旧实现将离线 IID 估计误当成30批运行保证，且最近定期 checkpoint 只有
step 10，导致已提交的 step 11–47 无法恢复。

v2.1 不改变 Reward、return、LOO、mask、PPO loss、Root/Local 结构或40/35/25阶段目标，只修订
采样可靠性与持久化合同：

1. 第10批仍是质量窗口；若尚未凑齐 pair，继续使用同一步、同一策略版本和同一候选池采样；
2. 已找到的 Root 或目标阶段 Local 保留到该 optimizer step 凑齐，不跨 optimizer step 复用；
3. 第120批才是硬上限，仍不允许错误阶段、constant、invalid 或不完整 batch fallback；
4. 每25个已提交 optimizer steps 保存 checkpoint，并保留全部20个定期 checkpoint；
5. 第120批终止前额外保存最后一个已提交 optimizer step，pending pair 永不训练；
6. 新正式运行从 SFT `checkpoint-325` 的 step 0 开始，形成连续的新 SwanLab 曲线。
