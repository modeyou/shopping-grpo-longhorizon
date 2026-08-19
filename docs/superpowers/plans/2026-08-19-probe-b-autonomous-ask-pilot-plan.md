# Probe B V3 Autonomous-Ask Pilot 实施计划

日期：2026-08-19

依据：`docs/superpowers/specs/2026-08-19-probe-b-autonomous-ask-pilot-design.md`

## 目标

新增一个与 V2 隔离的 10 题 `autonomous_ask` Pilot：每题只在第一次购物动作前开放一次本地 `ask_user` 工具；复用 V2 No-Ask/Oracle 结果，离线验证后只把真实执行命令交给用户。

## 实施顺序

1. 在 `tests/test_probe_b_autonomous.py` 先冻结选题顺序、关键词分类、正确/错误回答、一次性工具边界和无泄漏要求。
2. 新建 `probes/autonomous_runner.py`，复用 V2 的环境覆盖、调用预算、真实客户端、轨迹有效性和 JSON 写入逻辑；新增 Autonomous 客户端适配器、V3 manifest 与单任务断点续跑。
3. 扩充测试，覆盖 mock 完整 10 题、invalid 中止、调用预算和同 run ID 续跑。
4. 新建 `probes/autonomous_metrics.py`，严格加载 V2 主记录与 `18637` 补充记录，并生成 No-Ask / Autonomous / Oracle 三条件报告。
5. 增加指标测试，覆盖受限 supplemental 替换、重复/错误引用拒绝、PASS/FAIL/INCOMPLETE 和 Oracle 差距恢复率。
6. 更新 `probes/README.md`，只提供 Linux 的 validate、1 题冒烟、同 run ID 续跑到 10、指标和诊断命令。
7. 执行专项 pytest、完整 mock、指标生成、Python 编译和 `git diff --check`；不调用真实 API，不连接服务器环境。

## 完成标准

- V2 入口和冻结文件语义不变；
- 10 条 mock 均完成且每条恰有一次正确提问；
- 提问不计环境步数，模型可见消息不含 `clear_query`；
- V2 supplemental 只替换同 key 的 invalid；
- 真实模式必须显式 `--run-id`、`--reference-run-dir` 和 `--allow-real-api`；
- README 命令可先冒烟再安全续跑，默认最多 200 次 LLM HTTP 尝试。
