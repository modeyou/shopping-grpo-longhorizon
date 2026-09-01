不会丢失：正式训练使用 `nohup ... >"$CARL_LOG" 2>&1`，控制台的 stdout 和 stderr 都保存在完整日志里。建议另开一个 SSH 窗口同时监控两条通道。

### 1. 监控控制台严重问题

```bash
tail -n 0 -F "$CARL_LOG" |
grep --line-buffered -Ei \
'Traceback|RayTaskError|RuntimeError|ERROR|CUDA out of memory|nonfinite|no_nonzero_gradients|no_parameter_delta|full_batch_timeout|optimizer audit warning|diagnostic write skipped'
```

这会显示真正需要处理的异常，不会被普通 step 输出淹没。

### 2. 监控结构化训练事件

```bash
DIAG="$CARL_OUT/training_diagnostics.jsonl"

tail -n 0 -F "$DIAG" |
jq --unbuffered -c '
  select(
    .event == "slow_full_batch_warning" or
    .event == "full_batch_timeout" or
    .event == "skipped_update" or
    (
      .event == "bpo_optimizer_backward" and
      .audit.accepted != true
    )
  )
'
```

已有历史记录可以直接扫描：

```bash
jq -c '
  select(
    .event == "slow_full_batch_warning" or
    .event == "full_batch_timeout" or
    .event == "skipped_update" or
    (
      .event == "bpo_optimizer_backward" and
      .audit.accepted != true
    )
  )
' "$DIAG"
```

### 3. 检查训练是否仍存活

```bash
if kill -0 "$CARL_PID" 2>/dev/null; then
  echo "训练仍在运行，PID=$CARL_PID"
else
  echo "训练进程已经退出"
  tail -n 120 "$CARL_LOG"
fi
```

需要立即处理的主要是：

- `Traceback`、`RayTaskError`、`RuntimeError`
- OOM 或 nonfinite
- optimizer audit 失败
- `full_batch_timeout`
- 进程意外退出

以下通常只是观察项，不应立即停训：

- `slow_full_batch_warning`
- goal advantage share 偏低
- KL、clipfrac、grad norm 短期波动
- validation 单点下降
- 普通库的 deprecation warning

因此最好同时保留：SwanLab 看趋势、JSONL 看结构化契约、完整日志捕获第三方/Ray/CUDA异常。







### 4. 只读实时聚合监控

运行中的训练不热更新代码，也不向同一个 SwanLab run 注入第二个写入端。独立 sidecar 只读取
`training_diagnostics.jsonl`，每 60 秒输出一次摘要并原子更新 JSON 快照：

```bash
export BPO_LIVE_MONITOR="$CARL_OUT/bpo-live-monitor.json"

"$GRPO_PYTHON" scripts/monitor_bpo_training.py \
  --diagnostics "$CARL_OUT/training_diagnostics.jsonl" \
  --interval 60 \
  --output "$BPO_LIVE_MONITOR"
```

一次性快照使用：

```bash
"$GRPO_PYTHON" scripts/monitor_bpo_training.py \
  --diagnostics "$CARL_OUT/training_diagnostics.jsonl" \
  --once \
  --output "$CARL_OUT/bpo-live-monitor.json"
```

监控口径：

- `accepted_sibling_terminal_outcomes_total` 是进入 optimizer 的 group 数乘以 K，不再称作
  effective returns；
- task 重复以 group 为采样单位，组内 K 个 sibling 的同 task 对照不计为重复；同时报告
  task 和 task-condition-stage 两种覆盖率、最大复用次数与 effective count；
- 当前已经启动的 v3 只保存 action start/end，sidecar 将其明确标为 span proxy。这个跨度包含
  工具/环境 token，不冒充 actor-token 梯度质量；
- 新代码产生的后续 run 会在 `bpo_actor_batch.token_mass` 中记录每个 group、sibling、action 的
  精确 actor-token 数，并给出 action/sibling/group 三层 max-share、CV 与 HHI；
- 当前 run 没有保存 entropy probe token identity，sidecar 必须显示 unavailable。后续 run 会记录
  sampled token、argmax token 及 argmax token type；这些字段只用于诊断，不参与分叉选择；
- `slow_full_batch_warning`、`full_batch_timeout`、`skipped_update` 和 optimizer rejection 从 JSONL
  直接累计，其中后三者或非有限 actor batch 才设置 `alerts.blocking=true`。
