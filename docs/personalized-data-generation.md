# 个性化任务数据生成

本阶段只生成画像、当前请求、私有目标和冻结澄清回答，不采集 SFT 教师购物轨迹，也不调用
Reward。生成结果默认写入被 Git 忽略的 `outputs/`，人工审阅通过后才会晋升到正式 `data/`。

## 1. 导出候选源事实

从与 Final-200 隔离的 GRPO train pool 中固定抽取一批 ShopSimulator 商品与任务事实。20 条通过任务
建议先导出 80 条候选，为 API 或 Critic 拒绝留余量：

```bash
cd ~/shopping-grpo

python scripts/export_personalized_sources.py \
  --task-pool data/grpo/train.jsonl \
  --output-dir outputs/personalized-data/pilot-sources-01 \
  --count 80 \
  --seed 20260819
```

输出：

```text
outputs/personalized-data/pilot-sources-01/
  source_tasks.jsonl
  manifest.json
```

导出行不包含原作者 `user_persona` 内容，只记录是否存在参考画像。正式画像由下一步重新生成。

## 2. 配置 API

使用 OpenAI-compatible Chat Completions 接口：

```bash
export OPENAI_BASE_URL="https://ws-66q3vmu9ebhzahay.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="deepseek-v4-flash"
read -s -p "Bailian API key: " OPENAI_API_KEY
export OPENAI_API_KEY
echo
```

Key 不写入命令历史、输出文件或 manifest。脚本会记录 endpoint 和模型名，但不会记录密钥。

## 3. 生成 20 条 Pilot

```bash
python scripts/generate_personalized_tasks.py \
  --source-tasks outputs/personalized-data/pilot-sources-01/source_tasks.jsonl \
  --output-dir outputs/personalized-data/pilot-tasks-01 \
  --target-accepted 20
```

正式 Pilot 默认执行两次独立调用：

1. Architect 生成任务；
2. Critic 只做 accept/reject，不替 Architect 静默改写答案。

代码固定 source ID、目标 ASIN、schema、场景和 provenance。LLM 生成的数据必须先通过 schema、
隐藏信息、问题数量、source evidence 和场景语义检查，才会交给 Critic。

四种场景按已验收任务轮转，避免因某类任务被拒绝而最终失衡；`clarification_required` 任务轮流要求
1 个和 2 个缺失字段，每次 `ask_user` 仍只询问一个规范字段。

## 4. 中断续跑

额度耗尽、网络异常或 provider 错误会停止进程，但不会把当前 source 标成数据拒绝。恢复额度后使用
完全相同的参数：

```bash
python scripts/generate_personalized_tasks.py \
  --source-tasks outputs/personalized-data/pilot-sources-01/source_tasks.jsonl \
  --output-dir outputs/personalized-data/pilot-tasks-01 \
  --target-accepted 20 \
  --resume
```

`--resume` 会核对 source hash、模型、endpoint、Prompt 和生成配置；配置不一致时拒绝混跑。

## 5. 输出审计

```text
outputs/personalized-data/pilot-tasks-01/
  run_config.json
  raw_architect.jsonl
  raw_critic.jsonl
  attempts.jsonl
  accepted_tasks.jsonl
  summary.json
```

- `raw_*`：provider 完整响应，用于 token/费用与错误审计；
- `attempts.jsonl`：每个 source 的接受或拒绝原因；
- `accepted_tasks.jsonl`：通过代码和 Critic 的私有任务包；
- `summary.json`：完成状态、API 调用数和正式任务哈希。

快速检查：

```bash
cat outputs/personalized-data/pilot-tasks-01/summary.json

python - <<'PY'
import collections, json
path = "outputs/personalized-data/pilot-tasks-01/accepted_tasks.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
print("rows", len(rows))
print("scenarios", collections.Counter(row["scenario"] for row in rows))
print("question_counts", collections.Counter(
    len(row["clarification"]["targets"]) for row in rows
))
PY
```

`--skip-critic` 只用于接口调试，不用于正式数据。

## 6. 本阶段验收边界

20 条 Pilot 完成后先人工审阅：

- 每种场景至少抽看 2 条；
- 所有两问任务逐条检查两个问题是否都必要；
- 检查画像稳定事实是否只用于本人且高置信的尺码；
- 检查品牌、预算、颜色和功能没有被写成不可推翻的事实；
- 检查当前请求自然、没有目标标题或隐藏答案泄漏；
- 检查 source evidence 确实支持每条私有约束。

通过后再估算正式数据量、API 成本和轨迹长度，并设计 Teacher + Shopper + ShopSimulator 的 SFT
轨迹采集。不要在此时直接把 `accepted_tasks.jsonl` 当成 SFT messages。
