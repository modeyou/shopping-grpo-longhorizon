# BPO 独立代码审计指南

本文用于把 `feat/bpo2` 交给一个没有参与实现的 Codex 审查。目标是发现会导致错误
rollout、错误 advantage、无效 optimizer update、资源泄漏或 GRPO 回归的问题，而不是重复
实现者的结论。

可直接把下面这段交给 Codex Spark：

```text
请严格按照 docs/bpo-independent-audit.md 对当前 feat/bpo2 的 CARL-BPO v3 做独立、只读审计。
不要修改代码，不要启动训练，不要合并模型，也不要运行正式 200 题评测。
先读取 AGENTS.md，再记录 HEAD 并检查 v3 相关源码、配置与补丁；不要把文档声明当作实现证据。
执行指南中安全的静态检查和测试，并按规定格式输出 findings。若没有发现问题，也必须列出
已验证范围、未验证范围和残余风险。不要因为已有测试通过就省略源码数据流审查。
```

## 1. 审计边界

- 被审计分支：`feat/bpo2` 当前远端 HEAD；先记录完整 commit SHA。
- 实现锚点：v3 首个实现提交 `987668a`，同时必须审查其后的修复直至当前 HEAD；commit只用于
  定位，算法事实仍以当前源码为准。
- 算法依据：BPO提供snapshot branching背景；当前v3的Root/Local两级信用以GiGPO为方法锚点，
  但动作边界、canonical semantic gate与action-balanced loss均为本项目适配，必须按源码单独审查。
- 运行契约：ShopSimulator Environment v2.1、Reward v4、observation v2、tool schema v2。
- 不在本次审计范围：启动 BPO/GRPO 训练、模型合并、正式 200 题评测、修改 formal 数据。
- 审计必须只读；发现问题后先报告，不在同一轮自动修复。

先执行：

```bash
git status --short --branch
git rev-parse HEAD
git diff --check b156e19..HEAD
git diff --stat b156e19..HEAD
git diff b156e19..HEAD -- \
  configs/bpo.yaml \
  patches/verl-0.8.0-shopping-dynamic-sampling.patch \
  scripts/apply_verl_bpo_patch.py \
  scripts/apply_verl_bpo_tool_parser_patch.py \
  scripts/apply_verl_dynamic_sampling_patch.py \
  scripts/check_bpo_runtime.py \
  scripts/check_grpo_runtime.py \
  scripts/train_bpo.py \
  src/shopping_grpo/training/bpo \
  src/shopping_grpo/training/grpo/dynamic_sampling.py \
  tests
```

若工作区不干净，先在报告中列出非 HEAD 修改；不得自行丢弃或覆盖。

## 2. 必须逐项验证的算法契约

### 2.1 分叉边界

沿 `ShoppingBPOAgentLoop.run_tree()` 到 `validate_tree_outputs()` 和
`audit_bpo_rollout_batch()` 追踪完整数据流，验证：

1. entropy 探针、环境 snapshot 和 sibling continuation 使用同一个 action 前状态；
2. 探针不包含尚未生成的 backbone action token；
3. backbone 最终 action 被排除，最高熵选择只发生在非终局 action 集合；
4. 在线只保留两个候选 snapshot 的做法在“最终 action 可能成为最高熵点”时仍等价于从全部
   非终局候选中取最大值；检查 tie-break 是否仍选择较早 action；
5. 少于两个 action 的 backbone 必须 fail closed，不能伪造 tree；
6. `bpo_backbone_action_count` 和 `bpo_branch_relative_position` 在 AgentLoop 返回前和
   advantage 前都受到一致性检查。

重点寻找 off-by-one：action 起始边界、最终 action 定义、clone 的 action 列表可能与 backbone
长度不同，不能混为一谈。

### 2.2 K=4 tree 与 advantage

验证每个 group 恰好包含 sibling `0..3`，共享完全相同的 prompt、分叉前 token/mask、prefix
hash 和 branch metadata；三个 clone 必须拥有互异的环境 lease。

验证 reward 和梯度计算：

- return 对完整 `token_level_rewards` 求和，不能被 actor `response_mask` 丢掉终局环境奖励；
- 每个 sibling 的 baseline 是其他 K−1 个 return 的均值；
- LOO advantage 数值正确且组内和接近 0；
- Root 的所有真实 action 获得 episode LOO；Local 只有 branch action 获得 sibling LOO；
- Local prefix和suffix必须同时退出policy numerator与denominator；
- action内先做token mean，group内做action mean，Root/Local policy mass严格各0.5；
- constant、NaN/Inf、metadata 不完整和 prefix 漂移不能进入 optimizer；
- critic 和 reference policy 没有被意外启用。

### 2.3 Snapshot 生命周期与并发

检查所有正常和异常路径是否最终完成：source snapshot 删除、clone lease 释放、source lease
释放、ContextVar 恢复。特别审查：

- final action snapshot 不会再被 clone；
- terminal runtime state 或 clone 返回 `done=true` 时 fail closed，并释放刚取得的 slot；
- 三个 clone 使用 `gather(return_exceptions=True)` 后，单个 sibling 失败不会让其他 lease 留在后台；
- source lease 保留到 clone 全部结束是否会引入新的死锁、slot 枯竭或状态耦合；
- `train_batch_size=2, K=4` 峰值为 2 个 source + 6 个 clone，必须与服务器 slot 配置兼容；
- retained snapshot 的去重和删除逻辑不会重复删除仍在使用的 snapshot。

不要仅依据 mock；同时阅读 `pack_api.py`、`snapshot_store.py`、环境客户端的
`snapshot/clone/drop_snapshot/release` 实现。

## 3. 动态采样审计

正式 v3 的冻结值是
`target=2/minimum=2/require_full_batch=true/quality_search=10/max_batches=120`。Root与Local候选
分别进入保留池，不得用单棵 K=4 tree 完成较小更新；只有凑满1棵Root和1棵Local、共8个
sibling terminal outcomes后才能进入 optimizer。逐分支验证：

1. `target=2/minimum=2/require_full_batch=true/max_batches=120` 在 Hydra 合并配置、run contract
   和 trainer 实际执行路径中保持一致；
2. 0 个有效 group 的路径不会对空列表执行 `DataProto.concat()`；
3. 只有 1 个有效 group 时会跨 generation batch 正确保留，且不会提前进入 balance、log-prob、
   advantage 或 actor update；
4. 第10个 candidate batch结束quality search但不改变严格双组要求；第120个仍未凑齐时必须
   fail closed，不能单树更新或伪造第二棵树；
5. 2 个有效 group 的正式路径不会混入额外或半个 sibling group；每次 optimizer batch 必须是
   2 棵树和 8 个 sibling returns；
6. ready、告警和硬停止路径的计数器、profiling、replica sleep/wake、checkpoint 与 global step
   语义正确，未完成的 step 不计数；
7. GRPO 未显式设置 BPO 的 full-batch 参数时，原 GRPO 动态采样语义不能被共享 patch 改坏；
8. 从多个 generation batch concat 后，所有 tensor/non-tensor 字段仍逐 trajectory 对齐。

审查仓库 patch 时还要把它实际应用到冻结的官方 veRL 0.8 `ray_trainer.py` 副本，确认所有 hunk
唯一命中、结果可编译，并且派生 SHA256 与安装器计算一致。不得只搜索 marker 字符串。

## 4. XML 工具解析补丁审计

验证 `apply_verl_bpo_patch.py` 直接作为脚本执行时，不会导入 site-packages 中可能存在的同名
`scripts` 包；必须按同目录文件路径加载 XML 安装器。

验证 XML 变换只处理单个 `<parameter=...` 内容缺少 `>` 的情况：跳过坏参数，保留同一 tool
call 的其他合法参数；后续 tool schema 仍负责拒绝缺少必填参数的调用。检查它不会：

- 吞掉整个 tool call 或后续 tool call；
- 将空参数名当成合法参数；
- 把畸形工具调用记成成功执行；
- 隐藏与该锚点无关的 parser 异常。

同时验证 entropy 和 XML 两个安装目标各自具有：冻结版本检查、精确源码锚点、独立备份、未知
源码拒绝、幂等 apply/check、可验证 restore、写入后 Python 编译。

## 5. 诊断可观测性审计

从 AgentLoop `extra_fields` 一直追到 `training_diagnostics.jsonl`，确认所有 BPO 字段与 trajectory
逐行对齐。尤其验证：

- `bpo_branch_semantic_action_sha256` 来自规范化的单个tool name+arguments，不混入reasoning、
  XML空白或工具observation；
- Local必须全部semantic-valid且至少包含2个不同semantic action；token hash不能替代该门槛；
- `bpo_action_token_starts/ends` 能重建真实action，Local policy support只覆盖branch action；
- `bpo_unique_tool_sequence_count`、终止原因、完整错误和错误类型可用于区分 sibling 相同、
  reward constant、XML 错误和基础设施错误；
- metadata 缺失时明确记录 incomplete 或 fail closed，不产生误导性的多样性数字；
- 诊断中没有 Shopper API key、隐藏 omitted facts 或其他私有数据。

## 6. 安全测试命令

在具有项目 veRL 环境的服务器执行以下测试；这些命令不启动训练：

```bash
export PYTHONPATH=./src
: "${GRPO_PYTHON:?请设置为安装了项目固定依赖的 Python 可执行文件}"

"$GRPO_PYTHON" scripts/apply_verl_dynamic_sampling_patch.py --check
"$GRPO_PYTHON" scripts/apply_verl_bpo_patch.py --check

"$GRPO_PYTHON" -m pytest \
  tests/test_bpo_*.py \
  tests/test_verl_dynamic_sampling.py \
  tests/test_verl_dynamic_sampling_patch.py \
  tests/test_verl_bpo_entropy_patch.py \
  tests/test_verl_bpo_xml_tool_parser_patch.py \
  tests/test_merge_lora_adapter.py \
  tests/test_standalone_checkpoint_evaluation.py \
  -q
```

验收要求是所有列出的测试均通过；不要冻结历史测试数量，因为 v3 测试集合会随实现修复增加。
若本地缺少 veRL、Hydra、
OmegaConf、Torch 或服务器依赖，必须把对应项写成 `NOT RUN`；不得用纯单元测试替代并宣称完整
集成通过。

另外进行只读契约检查：

```bash
git grep -nE 'data/evaluation/tasks.jsonl|shopsimulator-reward-v4|environment-v2.1' -- \
  configs/bpo.yaml scripts/train_bpo.py src/shopping_grpo/training/bpo docs/bpo.md
git status --short --branch
```

不要执行 `scripts/bpo.sh` 的真实运行、`verl.trainer.main_ppo`、模型 merge 或 evaluation。

## 7. 报告格式

报告必须以 findings 开头，按严重度排序：

- `S0`：数据污染、错误 Reward/环境契约、会训练错误目标或不可恢复破坏；
- `S1`：会造成错误 tree/advantage、OOM/死锁/泄漏、optimizer 无法更新或 GRPO 回归；
- `S2`：诊断失真、重要异常未 fail closed、可复现性或维护性明显不足；
- `S3`：低风险质量问题。

每条 finding 必须包含：

1. 严重度和一句话标题；
2. 精确 `file:line`；
3. 可复核的代码路径或最小复现证据；
4. 对正式 BPO/GRPO 的具体影响；
5. 最小修复方向，但不要直接改代码。

findings 后依次给出：

- `Verified`：实际验证过的契约和命令；
- `Not verified`：因环境或权限未覆盖的内容；
- `Residual risks`：即使没有 finding 仍存在的运行风险；
- `Verdict`：只能是 `BLOCKED`、`READY FOR PREFLIGHT` 或 `READY FOR 1-STEP SMOKE`。

没有 findings 时也不能只写“LGTM”；必须给出证据充分的 `Verified / Not verified /
Residual risks / Verdict`。
