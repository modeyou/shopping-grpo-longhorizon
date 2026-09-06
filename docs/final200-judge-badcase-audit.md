# Final-200 G+ Judge Bad-case 与一致性审计

> 审计日期：2026-09-06
> 对象：SFT checkpoint-325 与 Root/Local RL step-200
> Judge：`deepseek-v4-flash-0731`，`trajectory-judge-v2-draft-r1`

## 结论

当前 Judge 输出可以用于定位候选 bad case，但**不能用于证明 SFT 与 RL 的小幅过程指标差异**。
Reward v4 的 strict、合格替代品和终局结果不受本审计影响；受影响的是 Rubric 满足率、五维 Judge 均值、
澄清状态和主次错误计数的模型间解释。

主要依据不是主观感受，而是一个可复现的一致性反例：200 个 SFT/RL 配对任务中，有 118 对 Judge 输入在
移除无语义的 `trajectory_id`、`tool_call_id` 后完全相同。这 118 对本应得到同一判定，但实际结果为：

| 输出 | 相同输入却不同 | 不一致率 |
|---|---:|---:|
| Search | 33/118 | 28.0% |
| Candidate | 28/118 | 23.7% |
| Evidence | 45/118 | 38.1% |
| Decision | 32/118 | 27.1% |
| Termination | 46/118 | 39.0% |
| 澄清状态 | 54/118 | 45.8% |
| 至少一条 Rubric 状态 | 48/118 | 40.7% |
| 主/次错误组合 | 106/118 | 89.8% |

这说明 `temperature=0` 没有带来足够的实际确定性；不同模型目录分别调用 Judge，使相同轨迹获得了不同评分。
任何小于这一级别噪声的均值差都不能解释为训练效果。

## 对原始均值的敏感性分析

作为敏感性检查，对 118 对相同输入强制复用 SFT 的同一判定，仅在剩余 82 对真正不同的输入上保留各自
Judge 输出：

| 维度 | SFT 原始 | RL 原始 | 相同输入共享判定后的 RL | 修正后 RL−SFT |
|---|---:|---:|---:|---:|
| Search | 1.190 | 1.255 | 1.220 | +0.030 |
| Candidate | 0.940 | 0.950 | 0.920 | -0.020 |
| Evidence | 0.500 | 0.505 | 0.480 | -0.020 |
| Decision | 1.120 | 1.090 | 1.080 | -0.040 |
| Termination | 0.975 | 1.000 | 0.945 | -0.030 |

原始 Candidate、Evidence、Termination 的正增量在强制一致后全部翻转，说明这些增量主要来自 Judge
随机差异，而非轨迹变化。Search 在敏感性检查后仍为正，但仍未经重复 Judge 或人工标注校准，暂不发布为
模型能力结论。

同样处理后，RL 的 hard Rubric satisfied 从原始 1,239 条降为 1,231 条（SFT 为 1,214），effective
澄清从原始 116 条降为 103 条（SFT 为 98）。方向没有完全消失，但幅度明显缩小，且不同输入部分仍包含
独立 Judge 噪声。

## 30 条分层人工审计

样本覆盖 Decision 下降、`premature_purchase`、`wrong_option`、预算错误和 Reward–Rubric disagreement，
共 30 个去重 task。按“证据是否能直接支持 Judge 结论”分为三组。

### A. 明确的真实模型/过程问题（14 条）

`615, 1770, 1822, 2338, 5420, 6497, 8799, 9880, 11619, 12466, 15566, 21554, 22605, 22950`

- task 22605：RL 连续切换大量互斥选项，最终 `repeat_loop`、未购买；这是明确的 RL 退化。
- task 1770：要求粉色+蓝色组合，最终只选烟灰蓝。
- task 1822：要求 100cm，最终选择 80cm。
- task 6497：澄清回答要求中号，最终选择 S。
- task 615/5420/11619/15566：最终价格分别为 109>100、21>20、2800>2500、365>350。
- task 22950：2025 款车辆最终选择 20–22 款选项。

这些案例支持一个真实结论：模型经常在只打开一个候选后下单，最终价格、隐藏规格和组合选项仍是主要薄弱点。

### B. 明确的 Judge 或 Rubric 错误（10 条）

`2240, 2386, 2672, 4479, 4681, 6025, 13020, 14960, 19873, 20755`

- task 2386：最终选项明确为“需另接开关”，Judge 却读取此前临时选择的“人体感应”并判 `wrong_option`。
- task 4681/2672/6025/13020/20755：分别把 `412≤440`、`27≤30`、`227≤230`、`239≤240`、
  `119≤120` 判成超预算。
- task 4479：用户要求买 2 罐、每罐 400g，最终规格正是“400g*2罐”，却被判违反“400g一罐”；
  冻结 Rubric 又要求“800g”单项，组合规格语义错误。
- task 2240：商品标题明确为“2025年二级建造师”，Judge/Rubric 却按“一级建造师”判品类冲突。
- task 14960：Judge 使用“尺码最小（XXL）”作为要求，需回查候选语义。
- task 19873：实质相同输入在 SFT 中五维均为 1，在 RL 中五维均为 2，是纯评分不一致样本。

预算主错误共有 7 条，其中 4 条是真实超预算、3 条是明确算术错误；当前 `budget_violation` 主错误计数
不能直接用于模型比较。

### C. Reward、隐藏条件或证据边界的混合问题（6 条）

`973, 2057, 7126, 7153, 14412, 18004`

- task 7126/7153：Judge 认为可见商品基本满足需求，但 Reward 为 partial alternative，需要逐 atom 核对。
- task 14412：Actor Query 未出现黑色，Judge 使用完整 Rubric 判红色违反黑色要求；适合终局需求审计，
  但不能直接评价 Actor 当时的可见决策。
- task 18004：商品显示支持定制但未选择具体尺寸，应区分 `unknown`、未澄清与真正 `violated`。
- task 973：Reward 接受女装套装为 valid alternative，Judge/Rubric 的品类描述出现矛盾语义。
- task 2057：SFT/RL 都购买同一 Gold 商品和规格，过程近似相同，但 Decision 分别为 1/0。

## 当前发布边界

仍可正式使用：

- Reward v4 strict：SFT 61.0%，RL 62.5%；
- Reward v4 购买成功率：SFT 62.5%，RL 64.5%；
- 配对 strict 为 5 gains、2 losses，McNemar `p=0.4531`；
- 真实轨迹中的循环、预算超限、选错最终规格等确定性 bad case。

暂不用于简历或模型优劣结论：

- 五维 Judge 的小幅均值差；
- hard/soft Rubric satisfied 的模型间百分点差；
- effective/ineffective/unnecessary 澄清计数差；
- Judge 主错误、次错误的模型间计数差。

## 修复优先级

1. **相同输入只 Judge 一次（已实现，待重跑）。** 新版单 Judge 请求移除 `trajectory_id`、
   `tool_call_id`，按模型、prompt、解码参数和完整语义消息生成 SHA-256，并使用跨模型共享缓存；并发运行也通过
   每个 hash 的原子锁避免重复调用。对现有三组 600 条 G+ 输入离线计算得到 482 个唯一请求，SFT/RL 的
   118 对相同输入会节省 118 次调用。
2. **确定性事实交回代码（价格与最终状态已实现）。** 单 Judge 现在接收代码生成的 `final_purchase`、
   `price_checks` 与 `option_checks`。价格上限、下限、区间和近似区间由代码计算；最终选项按规格轴的精确值或
   组合 component 检查。LLM 若给出冲突的 Rubric 状态，schema 重试仍不一致则 fail fast，不再把中途临时
   选择当作终局状态。
3. **保持一个 Judge。** 不拆 Requirement/Process 两个调用，继续一次性输出逐项 Rubric、五维过程分、澄清
   和错误标签，以控制实现与 API 成本。完整 Rubric 可能包含 Actor 提问前未知信息的限制继续明确披露：五维
   与澄清只作诊断，不能单独证明因果收益。
4. **校准后再全量重跑。** 固定 30 条人工 gold，验证逐项状态、五维分和错误标签；非相同轨迹至少做重复
   Judge 或成对 A/B 盲评。校准未通过前不再为 200×3 支付全量 API 成本。

## 结构化校准门槛（2026-09-06）

30 条审计样本已固化到
`data/evaluation/judge-calibration-review-v1/cases.jsonl`。其中 21 条原有代码断言；另外 9 条涉及 Rubric
语义、Reward–Rubric 边界或隐藏条件，已于 2026-09-06 完成人工确认并写入 `human_review`，因此当前
`manual_review_task_count=0`。人工审查依据见 `docs/judge-calibration-manual-review.md`。

人工确认发现 6 条不是 Judge 推理问题，而是冻结 Rubric 本身需要修订：task 973、2057、2240、4479、
14960、18004。它们要求独立的 `shopping-multiturn-rubric-calibration-v1`，在新版 Rubric 生成前由
`rubric_revision_task_count=6` 明确阻止 `release_ready`，不能通过把人工标记清零来假装通过。

这 6 条修订已经结构化写入
`data/evaluation/judge-calibration-review-v1/rubric-revisions.jsonl`。生成器
`scripts/apply_judge_calibration_rubric_revisions.py` 会验证修订任务集合与确认表完全一致，输出 30 条校准
Rubric，并将人工修改标记为 `review.status=human_approved`、
`rubric_source=human_review_revision`；不会原地覆盖 Final-200 v10 Rubric。2026-09-06 本地生成检查结果为
30 条有效 Rubric、6 条人工修订，修订待办降为 0。

```powershell
$env:PYTHONPATH = "src"
python scripts/apply_judge_calibration_rubric_revisions.py `
  --cases data/evaluation/judge-calibration-review-v1/cases.jsonl `
  --base-rubrics outputs/evaluation/final200-gplus-reference-v1/shared-rubrics/rubrics.jsonl `
  --revisions data/evaluation/judge-calibration-review-v1/rubric-revisions.jsonl `
  --output-dir tmp/judge-calibration-rubrics-v1 `
  --force
```

人工确认前，旧 Judge 产物的离线校准结果为 `89` 条自动断言中 `74` 条通过、`15` 条失败，另有 `9`
条人工待复核。确认结果固化后，新加入的人工 gold 状态断言也会检查旧 Judge；当前共 `119` 条自动断言、
`21` 条失败、`0` 条人工待确认、`6` 条 Rubric 修订待完成，`release_ready=false`。
失败项准确覆盖 task 2386 的临时 option 误读、五个价格 Rubric 算术错误、三个虚假
`budget_violation`、task 19873 相同输入不同输出，以及若干真实规格违反被误判 satisfied，证明校准器能拦截
本次已知问题，而不是对旧结果无条件放行。

离线复核命令：

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_judge_calibration.py `
  --cases data/evaluation/judge-calibration-review-v1/cases.jsonl `
  --rubrics outputs/evaluation/final200-gplus-reference-v1/shared-rubrics/rubrics.jsonl `
  --run sft-325=outputs/evaluation/final200-gplus-reference-v1/judge/sft-325/gap-ask-enabled `
  --run carl-bpo-v3-step200=outputs/evaluation/final200-gplus-reference-v1/judge/carl-bpo-v3-step200/gap-ask-enabled `
  --output outputs/evaluation/judge-calibration-report.json `
  --force
```

`automated_failure_count`、`manual_review_task_count` 和 `rubric_revision_task_count` 必须同时为 0，
`release_ready` 才能为 true。

### 校准结论（2026-09-06）

`trajectory-judge-v2-single-cache-r3` 已完成两模型 30×2 校准：30 个任务、119 条断言全部通过，
`automated_failure_count=0`、`manual_review_task_count=0`、
`rubric_revision_task_count=0`，最终 `release_ready=true`。

r2 首轮结果为 116/119，唯一剩余问题是 task 7126 的 SFT Judge 把终局未记录的颜色规格判为
`satisfied`，而 RL 判为 `unknown`。r3 将 `option_checks=unavailable` 固化为必须输出 `unknown`，避免用
中途 `select_option` 点击替代最终订单证据。升级时对旧缓存逐条执行新版合同：46 个唯一请求中 44 个
通过并安全迁移，只有 task 7126 和 task 22605 被拒绝并重新请求 API。最终 SFT、RL 各 30 条中均有
29 条缓存命中、1 条新请求。

### 全量运行结论（2026-09-06）

校准后的 r3 Judge 已完成 Final-200 G+ 的 Base、SFT-325、Root/Local RL step-200 全量运行：三组均为
200 条完整评测，SFT/RL Judge 覆盖 100%，Base 的 7 条基础设施无效轨迹按协议记为 `not_judged`。
SFT→RL 的五维均值变化为 Search `+0.015`、Candidate `+0.015`、Evidence `+0.045`、Decision
`+0.030`、Termination `+0.010`；Rubric 总满足率为 `83.7% → 85.5%`。

运行到 task 7704 时，合同发现 Curator 生成的组合 component `3倍免打孔` 与可见最终规格
`高清+3倍…打孔 | 免打孔` 发生假阴性。该任务经人工确认后，将 7 个组合 option 语义拆为原子匹配值，
再利用共享缓存重建三组结果。为控制时间和成本，本轮未扩展为全部组合 component 的人工审计，因此 Judge
部分定位为“经过校准、具备基本可信度的诊断结果”；Reward v4 的 strict/alternative 主结果不受该限制。

在调用新版 Judge 前，可先从现有 Final 产物生成两模型各 30 条的最小输入集：

```powershell
$raw = "outputs/evaluation/final200-gplus-reference-v1/selected-artifacts/outputs/evaluation/final200-v1"
$pilot = "outputs/evaluation/judge-calibration-v1"
python scripts/prepare_judge_calibration_inputs.py `
  --cases data/evaluation/judge-calibration-review-v1/cases.jsonl `
  --tasks data/multiturn/final-200-v1/tasks.jsonl `
  --rubrics outputs/evaluation/final200-gplus-reference-v1/shared-rubrics/rubrics.jsonl `
  --trajectory "sft-325=$raw/sft-325/gap-ask-enabled/trajectories.jsonl" `
  --trajectory "carl-bpo-v3-step200=$raw/carl-bpo-v3-step200/gap-ask-enabled/trajectories.jsonl" `
  --output-dir "$pilot/inputs" `
  --allow-blind-final
```

随后两次 `evaluate_multiturn_panels.py` 必须都传入同一个
`--judge-cache-dir "$pilot/semantic-judge-cache"`。60 个模型×任务单元中有 14 对相同语义输入，因此最多发生
46 个唯一首次 Judge 请求；schema 重试会额外调用。先跑 SFT、再跑 RL 时，第二次运行会直接命中这 14 个缓存，
不需要重复付费。

当前仓库提供一条安全入口，交互式读取 Key、顺序运行两模型、共享语义缓存并自动执行校准门禁；Key 只保留
在该 PowerShell 进程环境中，不写磁盘：

```powershell
cd D:\shopping-grpo-longhorizon
.\scripts\run_judge_calibration.ps1
```
