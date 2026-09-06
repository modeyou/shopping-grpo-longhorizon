# Final-200 G+ 正式评测结果

## 结论

在冻结的 200 条未见任务、Gap + Ask（G+）条件下，Root/Local 双分支 RL step-200 相对
SFT checkpoint-325 的严格命中率由 `61.0%` 提升到 `62.5%`，计入 Reward v4 判定的合格替代品后，
购买成功率由 `62.5%` 提升到 `64.5%`。RL 同时取得更高的总体 Rubric 满足率和硬约束
Rubric 满足率，但软约束满足率和 Judge 决策质量略低于 SFT。

这是一个小幅正向结果，而不是统计显著的全面胜出。SFT 到 RL 的配对 strict 转移为 5 条
失败转成功、2 条成功转失败，净增 3/200（+1.5pp）；exact McNemar `p=0.4531`。

## 冻结协议

- 数据集：`data/multiturn/final-200-v1/tasks.jsonl`，共 200 条未见任务。
- 条件：只评测 G+，即任务存在信息缺口，Actor 可以调用 `ask_shopper` 获取冻结回答。
- 模型：未适配 Base、SFT checkpoint-325、Root/Local 双分支 RL step-200。
- 终局判定：ShopSimulator Reward v4；所有比率固定以 200 为分母。
- 购买成功：`gold_purchase + valid_alternative_purchase`；`partial_alternative_purchase` 不计成功。
- Rubric：先由 Reward v4 从冻结任务资产抽取候选约束，再由 `deepseek-v4-flash` Curator
  对照 Query 选择、合并并冻结为共享 Rubric。三组模型使用完全相同的 1,628 条 Rubric。
- Judge：`deepseek-v4-flash-0731`，只读取 Actor 可见轨迹与共享 Rubric，不读取 Reward 结果、
  Gold 商品私有字段或未展示给 Actor 的信息。
- Rubric 冻结版本：`shopping-requirement-rubric-v5`；Rubric SHA-256：
  `73cf38a93d861ad173663aab892c25e52e194e567ce08e2f361d825ad8c5b69e`。

评测面板把三类信号并列保留，不把它们合成单一总分：代码硬检查负责终局购买、工具合法性和
执行状态；逐需求 Rubric 负责约束满足；轨迹 Judge 负责搜索、候选利用、证据核验、决策和终止质量。

## 主要结果

| 模型 | Strict / Gold | 合格替代 | 购买成功 | Mean Reward | Reward valid | Judge 覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 1/200（0.5%） | 0/200（0.0%） | 1/200（0.5%） | -0.0540 | 20/200（10.0%） | 193/200（96.5%） |
| SFT-325 | 122/200（61.0%） | 3/200（1.5%） | 125/200（62.5%） | 0.5401 | 199/200（99.5%） | 200/200（100%） |
| Root/Local RL step-200 | **125/200（62.5%）** | **4/200（2.0%）** | **129/200（64.5%）** | **0.5708** | 199/200（99.5%） | 200/200（100%） |

RL 相对 SFT 的 strict、购买成功率和 Mean Reward 分别为 `+1.5pp`、`+2.0pp` 和
`+0.0307`。SFT 与 RL 另有 49/200 和 48/200 条 `partial_alternative_purchase`，这些样本均未被
加入购买成功率。

## Rubric 与 Judge 结果

| 模型 | 全部 Rubric 满足 | 硬约束满足 | 软约束满足 | Search | Candidate | Evidence | Decision | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 772/1580（48.9%） | 718/1394（51.5%） | 54/186（29.0%） | 1.021 | 0.772 | 0.373 | 0.135 | 0.026 |
| SFT-325 | 1346/1628（82.7%） | 1214/1438（84.4%） | **132/190（69.5%）** | 1.190 | 0.940 | 0.500 | **1.120** | 0.975 |
| Root/Local RL step-200 | **1364/1628（83.8%）** | **1239/1438（86.2%）** | 125/190（65.8%） | **1.255** | **0.950** | **0.505** | 1.090 | **1.000** |

Judge 五个维度各自取 0/1/2 分，表中是有效 Judge 样本的均值，不计算综合总分。Base 只有
193 条有效 Judge，故其 Rubric 分母为 1,580；SFT 和 RL 均覆盖全部 200 条轨迹及 1,628 条
共享 Rubric。

相对 SFT，RL 的搜索策略、候选利用、证据核验和终止效率分别变化 `+0.065`、`+0.010`、
`+0.005`、`+0.025`，决策质量变化 `-0.030`。硬约束 Rubric 的任务级比较为 12 条改善、
178 条不变、10 条变差；平均硬约束违反率下降 0.02 条/任务。RL 的有效澄清为 116/200，
高于 SFT 的 98/200；无效澄清由 68 降到 55，不必要澄清由 31 降到 25。

## 结果边界

Base 的 0.5% 不是普通语言能力基线：它在当前 Agent Harness 中有大量非法动作和异常终止，
Reward-valid 也只有 10%。它说明未经后训练的模型没有掌握当前工具协议，可用于展示 SFT 对
Agent 可执行性的修复，但不应被表述成同等执行条件下的纯模型能力差距。

SFT 与 RL 的轨迹均有观察压缩标记；这是统一 Harness 下的可见上下文压缩，不是评测缺失。
Judge 将部分 Reward 成功轨迹归因为过早购买或证据核验不足，因此 Reward 终局成功与过程质量
需要并列报告，不能互相覆盖。SFT 有 12 条、RL 有 16 条 Reward/Rubric 分歧任务，适合作为后续
badcase 审计入口。

当前可以准确表述为：Root/Local 双分支 RL 在冻结 Final-200 G+ 上取得小幅终局提升，并改善
硬约束满足、搜索策略和澄清有效性；不能表述为“显著超过 SFT”或“所有过程维度均提升”。

## 可复核产物

- 对比汇总：`outputs/evaluation/final200-gplus-reference-v1/comparison-gplus.json`
- 共享 Rubric：`outputs/evaluation/final200-gplus-reference-v1/shared-rubrics/`
- 三组 Judge 结果：`outputs/evaluation/final200-gplus-reference-v1/judge/`
- 冻结输入轨迹：`outputs/evaluation/final200-gplus-reference-v1/selected-artifacts/`

