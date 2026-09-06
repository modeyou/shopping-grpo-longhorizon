# Final-200 G+ 正式评测结果

## 结论

在冻结的 200 条未见任务、Gap + Ask（G+）条件下，Root/Local 双分支 RL step-200 相对
SFT checkpoint-325 的严格命中率由 `61.0%` 提升到 `62.5%`，计入 Reward v4 判定的合格替代品后，
购买成功率由 `62.5%` 提升到 `64.5%`。这是当前可以正式使用的确定性结果。校准后的单 Judge 已于
2026-09-06 完成全量重跑；RL 相比 SFT 的五个过程维度均小幅上升，但幅度有限，作为诊断结果与
Reward v4 终局结果并列报告，不表述为统计显著的过程能力提升。

这是一个小幅正向结果，而不是统计显著的全面胜出。SFT 到 RL 的配对 strict 转移为 5 条
失败转成功、2 条成功转失败，净增 3/200（+1.5pp）；exact McNemar `p=0.4531`。

## 冻结协议

- 数据集：`data/multiturn/final-200-v1/tasks.jsonl`，共 200 条未见任务。
- 条件：只评测 G+，即任务存在信息缺口，Actor 可以调用 `ask_shopper` 获取冻结回答。
- 模型：未适配 Base、SFT checkpoint-325、Root/Local 双分支 RL step-200。
- 终局判定：ShopSimulator Reward v4；所有比率固定以 200 为分母。
- 购买成功：`gold_purchase + valid_alternative_purchase`；`partial_alternative_purchase` 不计成功。
- Rubric：先由 Reward v4 从冻结任务资产抽取候选约束，再由 `deepseek-v4-flash` Curator
  对照 Query 选择、合并并冻结为共享 Rubric；6 个校准任务使用人工确认版本，task 7704 的组合规格
  在全量运行中触发确定性冲突后拆为人工确认的原子规格。三组模型使用完全相同的 1,637 条 Rubric。
- Judge：`deepseek-v4-flash-0731`，读取 Actor 可见轨迹与共享的完整需求 Rubric，不直接读取 Reward 结果或
  Gold 商品私有字段。Rubric 可能编码 Actor 提问前尚未知的隐藏约束，因此可用于终局需求满足审计，不能
  把五维过程分或澄清标签单独解释为因果收益。当前设计保持一次单 Judge 调用，不拆成两个评审器。
- Rubric 冻结版本：`shopping-requirement-rubric-v5`；Rubric SHA-256：
  `7a1d28ce8fa6b61923748c44ae748a0e0d7377c352a694a9d71eab36a01022c3`。

评测面板把三类信号并列保留，不把它们合成单一总分：代码硬检查负责终局购买、工具合法性和
执行状态；逐需求 Rubric 负责约束满足；轨迹 Judge 负责搜索、候选利用、证据核验、决策和终止质量。
Rubric 的冻结字段、Curator 与 Judge 的输入隔离、逐项状态以及五维 0/1/2 量表，见
[多轮购物 Agent 评测协议](multiturn-evaluation.md) 第 5、6 节。

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

> **可靠性更新（2026-09-06）：** 旧 Judge 曾对 118 对相同语义输入给出不一致结果，因此不再使用。
> 新版 `trajectory-judge-v2-single-cache-r3` 以语义 hash 跨模型共享结果，30×2 校准集达到
> 119/119 断言通过、`release_ready=true`，并已完成本表对应的 Final-200 全量重跑。
> 全量过程中 task 7704 暴露出 Curator 将多个原子要求绑定到组合 option component 的问题；本次按
> Actor 可见最终规格人工拆分该任务的 7 个 option 语义后重建结果。该处理保留了成本最低的基本可信版本，
> 其余未触发合同冲突的组合 component 留作后续审计，不影响 Reward v4 主结果。

| 模型 | 全部 Rubric 满足 | 硬约束满足 | 软约束满足 | Search | Candidate | Evidence | Decision | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 571/1589（35.9%） | 551/1401（39.3%） | 20/188（10.6%） | 1.016 | 0.725 | 0.269 | 0.047 | 0.026 |
| SFT-325 | 1370/1637（83.7%） | 1227/1445（84.9%） | 143/192（74.5%） | 1.250 | 0.925 | 0.570 | 1.185 | 1.055 |
| Root/Local RL step-200 | **1399/1637（85.5%）** | **1251/1445（86.6%）** | **148/192（77.1%）** | **1.265** | **0.940** | **0.615** | **1.215** | **1.065** |

Judge 五个维度各自取 0/1/2 分，表中是有效 Judge 样本的均值，不计算综合总分。Base 只有
193 条有效 Judge，故其 Rubric 分母为 1,589；SFT 和 RL 均覆盖全部 200 条轨迹及 1,637 条
共享 Rubric。

校准后输出中，RL 相对 SFT 的 Search、Candidate、Evidence、Decision、Termination 分别变化
`+0.015`、`+0.015`、`+0.045`、`+0.030`、`+0.010`。配对任务中，Decision 为 14 条改善、
10 条变差、176 条不变；Evidence 为 23 条改善、15 条变差、162 条不变。有效澄清为
`102 → 106`，无效澄清为 `66 → 62`，不必要澄清为 `29 → 28`。这些均是小幅描述性变化。

### Judge 诊断解读（2026-09-06）

五个维度是互不相加的 0/1/2 序数分；不能把它们与 Reward strict 合成总分，也不能把极小均值差当作
稳定能力变化。RL 的 Evidence 从 `0.570` 升到 `0.615`，Decision 从 `1.185` 升到 `1.215`，说明本次
共享 Judge 下既有更多关键属性、规格和价格核验，也有略多轨迹把证据落实到最终决策，但提升都小于 0.05 分。

主要错误结构基本未改变：`premature_purchase: 71 → 72`、
`candidate_comparison_insufficient: 32 → 33`、`critical_evidence_missing: 30 → 28`、
`wrong_option: 16 → 16`、`budget_violation: 8 → 6`。因此更稳妥的解释是 RL 在少量任务上改善了终局选择和
证据核验，但“较早购买、候选比较不足”仍是共同瓶颈。

主错误是 Judge 在冻结 taxonomy 中选择的一个最能解释该轨迹低质量的根因；次错误至多两个，是同时存在但
不占主导的因素，均需引用真实 `event_id`。这些是 LLM 诊断标签，不是 Reward v4 的确定性终局结论。
当前已完成 30 条结构化校准集、9 条人工语义确认和相同输入一致性分析；校准后的 r3 Judge 已通过
119/119 条断言并完成 Final-200 全量重跑。以上差异仍是诊断性证据，不是训练算法的因果证明。

## 结果边界

Base 的 0.5% 不是普通语言能力基线：它在当前 Agent Harness 中有大量非法动作和异常终止，
Reward-valid 也只有 10%。它说明未经后训练的模型没有掌握当前工具协议，可用于展示 SFT 对
Agent 可执行性的修复，但不应被表述成同等执行条件下的纯模型能力差距。

SFT 与 RL 的轨迹均有观察压缩标记；这是统一 Harness 下的可见上下文压缩，不是评测缺失。
Judge 将部分 Reward 成功轨迹归因为过早购买或证据核验不足，因此 Reward 终局成功与过程质量
需要并列报告，不能互相覆盖。SFT 与 RL 均有 15 条 Reward/Rubric 分歧任务，适合作为后续
badcase 审计入口。

当前可以准确表述为：Root/Local 双分支 RL 在冻结 Final-200 G+ 上取得小幅 Reward v4 终局提升；不能
表述为“显著超过 SFT”；校准 Judge 支持“Rubric 满足率与五维过程分均小幅上升”的描述，但不应写成
显著或已证明的因果提升。

## 可复核产物

- 对比汇总：`outputs/evaluation/final200-gplus-r3/comparison-gplus.json`
- 共享 Rubric：`tmp/final200-rubrics-minimal-v1/`
- 三组 Judge 结果：`outputs/evaluation/final200-gplus-r3/{base,sft-325,carl-bpo-v3-step200}/`
- 语义 Judge 缓存：`outputs/evaluation/final200-gplus-r3/semantic-judge-cache/`
- 冻结输入轨迹：`outputs/evaluation/final200-gplus-reference-v1/selected-artifacts/`
