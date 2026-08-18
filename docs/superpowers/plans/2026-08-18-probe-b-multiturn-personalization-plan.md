# B 探针实现计划：澄清 + 个性化价值验证（Pilot）

- 日期：2026-08-18
- 分支：`dev/innovation`
- 关联设计：`docs/superpowers/specs/2026-08-18-probe-b-multiturn-personalization-design.md`
- 预计工期：2~3 天（无 GPU，纯 API + 环境服务）
- 目标：跑通两条对比臂（基线 vs 1~2 轮澄清），用预设判据决定"是否正式立项 B"

---

## 阶段 0：前置与环境（~0.5 天）

- [ ] 确认在 `dev/innovation` 分支，创建 `probes/` 目录：
  ```
  probes/
    data/             # 生成的任务数据
    outputs/          # 轨迹与结果
    user_simulator.py # 受控版用户模拟器
    build_tasks.py    # 任务欠约束化构造
    runner.py         # 双臂运行器
    metrics.py        # 指标计算
    README.md         # 结论与报告
  ```
- [ ] 确认 ShopSimulator 环境服务可启动（`scripts/start_environment.sh`，端口 5700），并可用 `task_id` 恢复任务与 TaskFacts。
- [ ] 确认教师 LLM API 可用（DeepSeek V4 Flash），key 写入环境变量，**不入库**。
- [ ] 确认 `data/sft/train.jsonl` 可读，字段结构已核实（messages 内含 system/user(Instruction:)/assistant/tool）。

## 阶段 1：任务欠约束化构造（离线，~0.5 天）

文件：`probes/build_tasks.py`

1. 读取 `data/sft/train.jsonl`，提取 `task_id` + `Instruction` 原文。
2. 选择 20~30 个任务（过滤条件：Query 含至少 2 个可隐藏约束维度，如品牌/型号/规格/预算；优先挑有多解空间的）。
3. 对每个任务生成：
   - `clear_query`：原 Query；
   - `under_query`：隐藏 1~2 个约束（如"预算 300 元"→"预算不太高"、去掉品牌名）；
   - `fake_profile`：被隐藏约束的结构化表示（如 `{budget: 300, brand: "某牌"}`）；
   - `hidden_fields`：记录隐藏了哪些维度。
4. 输出 `probes/data/tasks.json`：
   ```json
   [{"task_id": 1535, "clear_query": "...", "under_query": "...", "fake_profile": {...}, "hidden_fields": [...]}, ...]
   ```
5. 人工抽查 3~5 条：确认 `under_query` 确实"有歧义/多解"，且 `fake_profile` 与 TaskFacts 一致（从环境按 task_id 恢复核对）。

**验收**：`tasks.json` 生成成功；抽查 3~5 条通过；离线可复现（固定 seed）。

## 阶段 2：受控版用户模拟器（~0.5 天）

文件：`probes/user_simulator.py`

- 输入：`fake_profile` + 教师的一句话澄清问题（文本）。
- 规则解析：按关键词/字段表匹配问题意图（预算/品牌/型号/规格/用途等）；匹配到哪项就只返回哪项。
- 输出：模板化自然中文回答（如问预算 → "预算大概 300 元左右吧，别超太多就行。"）。
- 处理边界：问题不匹配任何已知字段 → 返回中性回答（如"这个我不太确定，你按合适的来就行"）并标记 `unknown_asked`；同时只答所问、不主动补充其他字段。
- 单元测试（`probes/test_user_simulator.py`）：
  - [ ] 问预算只回预算，不问就不给；
  - [ ] 问品牌只回品牌；
  - [ ] 无关问题走中性分支；
  - [ ] 0 随机性（同输入同输出）。

**验收**：单测全过。

## 阶段 3：双臂运行器（~1 天）

文件：`probes/runner.py`

- 复用参考项目的环境客户端/动作定义（`src/shopping_grpo/environment/`）与 Action Guard 校验思路，**不改作者源码**，只 import 复用。
- 模式：对每个任务跑两条臂：
  - **基线臂**：系统提示 = 单轮版（沿用参考 prompt 的"不得追问"语义），输入 = `under_query`；
  - **澄清臂**：系统提示 = 允许追问版（最多 2 轮），输入 = `under_query`；模型若选择"ask"动作 → 调 `user_simulator` → 把回答作为新的 user 消息注入 → 继续，最多 N=2 轮后强制回到购物。
- 统一参数：temperature 等保持一致；最大步数沿用参考（35 步）。
- 输出：`probes/outputs/{task_id}_baseline.jsonl` / `..._clarify.jsonl`（含完整轨迹）。

**验收**：两臂各在 20~30 任务上跑完；轨迹字段完整；澄清臂确实出现了 ask→answer 交互（抽样看 3 条）。

## 阶段 4：指标计算与判据（~0.5 天）

文件：`probes/metrics.py`

- 从环境返回的 Reward 判定：严格成功（gold_purchase）、错误购买（wrong_purchase）、放弃/超步等。
- 输出对比表（每任务 + 汇总）：
  | 指标 | 基线臂 | 澄清臂 |
  |---|---|---|
  | 严格成功率（模糊子集） | | |
  | 错误购买率 | | |
  | 平均步数 | | |
  | 平均澄清轮数 | — | |
  | 清晰任务成功率（对照） | | |
- 对照判据表（设计文档 §5）输出 PASS/FAIL 结论。

**验收**：指标可自动计算；结果写入 `probes/outputs/summary.json`。

## 阶段 5：报告与决策（~0.5 天）

- 写 `probes/README.md`：
  - 方法（简短）+ 对比表 + 结论（是否立项 B）
  - 失败模式分析（草率购买 / 弱约束执行 / 澄清滥用 / 用户模拟器失真）
  - 诚实边界（探针验证的是"澄清对欠约束指令的价值"，非真实画像个性化）
- 决策：PASS → 进入 B 正式设计（profile 字段、澄清协议、Rubric 扩展、多轮数据构造）；FAIL → 记录后主线转 A（Step-level Reward）。

**验收**：README 可读、可作作品集材料。

---

## 风险与注意
- **任务太"清楚"导致澄清无增益**：阶段 1 的抽查就是防这个；必要时扩到 50 任务或把 `under_query` 隐藏得更狠。
- **环境服务不可用**：探针依赖 5700 服务；若机器连不上，先在能跑的环境机上执行阶段 3~5（纯脚本部分可先离线）。
- **教师 API 费用**：20~30 任务 × 2 臂 ≈ 40~60 条轨迹，成本可控；先跑 5 条冒烟再全量。

## 完成定义（DoD）
- `tasks.json`、`user_simulator.py`(含测试)、`runner.py`、`metrics.py` 就绪且通过验收项；
- 两臂结果 + `summary.json` + `README.md` 完成；
- 全部提交到 `dev/innovation`。