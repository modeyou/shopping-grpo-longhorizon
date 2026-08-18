# Probe B V2：Oracle 信息价值低成本筛查

本探针只回答：从购物指令中隐藏一个当前约束后，在首次模型购物动作前固定披露该约束，是否比不披露更容易完成任务。

关联文档：

- 设计：`docs/superpowers/specs/2026-08-18-probe-b-oracle-information-value-v2-design.md`
- 实现计划：`docs/superpowers/plans/2026-08-18-probe-b-oracle-information-value-v2-plan.md`

V2 不测试自主提问，也不测试用户画像。`tasks.json`、`user_simulator.py`、`build_tasks.py` 和 `_gen_tasks_v2.py` 是 V1 历史文件，不进入 V2 运行。

## V2 文件

```text
probes/
  data/tasks_v2.json  # 25 个单隐藏字段任务
  task_schema.py      # 零 API 校验与隐藏字段满足判断
  runner.py           # No-Ask / Oracle-Ask 配对运行器
  metrics.py          # 配对指标与预登记门槛
  outputs/v2/<run_id>/
    manifest.json
    trajectories.jsonl
    metrics_summary.json
    metrics_summary.md
```

## 零成本检查

```powershell
# 1. 检查训练来源、评测集隔离、单字段、泄漏、Oracle 一致性和字段分布
python probes/runner.py --mode validate

# 2. 跑完整 25×2 Mock 链路，不连接 ShopSimulator 或模型 API
python probes/runner.py --mode mock --run-id mock-v2

# 3. 验证配对汇总；Mock 结果会标记 NOT_APPLICABLE_MOCK
python probes/metrics.py --run-id mock-v2

# 4. 专项单元测试
python -m pytest tests/test_probe_b_v2.py -q
```

## 真实一对冒烟

真实模式需要：

- ShopSimulator Environment v2.1 已运行在 `http://127.0.0.1:5700`；
- 设置 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和 `OPENAI_MODEL`；
- 用户明确批准这次付费调用。

阿里云百炼（华北 2 / 北京）配置：

```bash
read -rsp "百炼 API Key: " DASHSCOPE_API_KEY; echo
export DASHSCOPE_API_KEY
export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
export OPENAI_BASE_URL="https://ws-66q3vmu9ebhzahay.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="deepseek-v4-flash-0731"
export SHOPSIM_BASE_URL="http://127.0.0.1:5700"
```

Runner 会为该百炼 DeepSeek V4 模型显式发送 `enable_thinking: false`，避免思考 Token 增加成本；API Key 不会写入 manifest 或轨迹。

批准后只运行一个配对任务：

```bash
python probes/runner.py --mode real --run-id oracle-v2-real --limit-pairs 1 --allow-real-api
python probes/metrics.py --run-id oracle-v2-real
```

这两条有效轨迹直接计入最终 25 对。人工确认消息顺序、Oracle 注入、环境步数和 Reward 明细后，用同一个 `run_id` 断点续跑剩余任务：

```bash
python probes/runner.py --mode real --run-id oracle-v2-real --limit-pairs 25 --allow-real-api
python probes/metrics.py --run-id oracle-v2-real
```

Runner 会跳过已经尝试的 task-arm 组合。首轮最多 50 次轨迹尝试、默认最多 700 次模型 HTTP 请求；基础设施异常会停止运行，不自动补跑，也不会提高调用上限。

## 固定判据

只有 25 个完整有效真实任务对才能做决策，并且三项必须同时满足：

1. Oracle 相对 No-Ask 净增加至少 3 个严格成功任务；
2. Oracle 隐藏字段满足率更高；
3. Oracle 的真实 `reward_type == "wrong_purchase"` 数量不高于 No-Ask。

结果仍然只是低成本机制筛查，不是统计显著性结论。通过后才加入 Autonomous-Ask；用户画像实验排在自主澄清机制之后。
