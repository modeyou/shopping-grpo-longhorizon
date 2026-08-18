# Probe B V2 实现计划

依据：`docs/superpowers/specs/2026-08-18-probe-b-oracle-information-value-v2-design.md`

## 目标

在不调用付费 API 的前提下，把 25 个单隐藏字段任务、No-Ask/Oracle-Ask 配对运行、完整轨迹记录、成本硬上限和配对指标实现并通过离线测试。真实运行仍需用户单独授权。

## 实现顺序

1. 新增 `probes/data/tasks_v2.json`，按 13 budget / 6 color / 3 brand / 3 origin 冻结单字段任务。
2. 新增 `probes/task_schema.py`，校验来源、评测集隔离、单字段、泄漏、Oracle 一致性和字段分布，并提供隐藏字段满足判断。
3. 将 `probes/runner.py` 改为 V2 配对运行器：两臂共用正式提示词和工具；Oracle 对话在首次模型动作前固定注入；复用正式 Action Guard/rollout；按 run_id 隔离输出；支持断点续跑、50 次尝试和 `max_llm_calls`。
4. 将 `probes/metrics.py` 改为按 run_id 分析完整有效任务对，区分严格成功、真实 `wrong_purchase`、行为失败和基础设施无效，并执行预登记的三项门槛。
5. 新增 `tests/test_probe_b_v2.py`，覆盖任务不变量、两臂唯一差异、Oracle 不占购物步数、lease 释放、成本上限、断点续跑和配对判据。
6. 更新 `probes/README.md`，给出零成本校验、Mock、真实一对冒烟和剩余 24 对命令；只实际运行前三类离线检查，不执行真实 API 命令。

## 完成标准

- V2 数据静态校验通过；
- V2 专项测试和 Mock 端到端测试通过；
- 现有相关评测 rollout/action tests 不回归；
- 工作区只包含预期文件；
- 输出一条需要用户显式确认后才能执行的真实一对冒烟命令。
