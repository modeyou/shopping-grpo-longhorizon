# Final-200×3 运行手册

本文给出 CARL-BPO v2.1 训练完成后的最短评测路径。最终目标是让 Base、SFT-325 和唯一选定的
CARL-BPO checkpoint 在同一份未见 Final-200 上各运行 G+、G−、C+，共 1,800 条轨迹。

## 1. 已冻结资产

最终资产位于 `data/multiturn/final-200-v1/`。它从 Reward v4 Final-500 父集合中按
`sha256-rank-without-replacement-v1` 和固定 seed `shopping-final-200-v1` 无放回选择 200 个 task，
选择过程不读取任何模型结果。

```text
task count                         200
condition count                    600
selected task IDs SHA-256          f7f646f50c215f816fd9fc48a26e02110624b1ad17d18a9490634cfc46e99829
manifest SHA-256                   f9a3970c9bc59374ac23741a4a4519dca44832cc55414ce55944bb1fd446551e
```

在训练服务器上先检查资产；检查不调用模型或 API：

```bash
python scripts/freeze_multiturn_final_subset.py --check
```

不要使用 `--refresh` 改变已经提交的 Final-200。该参数只用于资产首次生成阶段的跨平台规范化。

## 2. 训练结束后先选唯一 RL checkpoint

checkpoint 只按训练前冻结的 validation 规则选择，不在 DEV-500 或 Final-200 上 sweep：

1. 最大化 `gold + valid alternative` completion；
2. 处于预设容差内时选择 gold 更高者；
3. 再比较 invalid 和模型未完成；
4. 仍相同则选择更早 checkpoint。

把选择理由、validation 数字和 checkpoint step 写入运行记录。只导出这个 checkpoint：

```bash
RL_RUN="$PWD/outputs/models/CARL_BPO_RUN"
RL_STEP=SELECTED_STEP
RL_SOURCE="$RL_RUN/global_step_$RL_STEP"
RL_MODEL="$PWD/outputs/models/carl-bpo-v2.1-step-$RL_STEP-export"

test -d "$RL_SOURCE/actor"
test ! -e "$RL_MODEL"
bash scripts/export_grpo.sh "$RL_SOURCE/actor" "$RL_MODEL"
```

## 3. 先运行 RL-only DEV-500×3

沿用 [CARL-BPO](carl-bpo.md) V6/V7 和
`scripts/run_standalone_checkpoint_evaluation.py` 的 dev 模式。只允许唯一选定的 checkpoint 运行一次
DEV-500；根据 DEV 结果可以决定是否发布该模型，但不得切换到另一个 checkpoint 后再试。

DEV 验收通过后冻结以下内容：模型目录及哈希、Actor/Shopper 模型名、prompt、Reward v4、环境版本、
最大步数、上下文预算、温度和评测代码 commit。此后才能打开 Final-200。

## 4. 单模型 Final-200 入口

先设置服务和模型路径。Base、SFT、RL 都必须使用相同的环境变量与 GPU 布局：

```bash
export PYTHONPATH=./src
export GRPO_PYTHON=/path/to/python
export VLLM_BIN=/path/to/vllm
export SHOPSIM_BASE_URL=http://127.0.0.1:5700
export SHOPPER_BASE_URL=https://YOUR_SHOPPER_ENDPOINT/v1
export SHOPPER_API_KEY=YOUR_KEY
export SHOPPER_MODEL=deepseek-v4-flash-0731

FINAL_ASSETS="$PWD/data/multiturn/final-200-v1"
```

每个模型使用一个新的输出目录。下面先以 RL 为例；Base/SFT 只替换 label、model 和输出路径，且不传
`--source-checkpoint`：

```bash
LABEL=carl-bpo-v2.1-step-$RL_STEP
OUT="$PWD/outputs/evaluation/final200-v1/$LABEL"
ACTORS="$PWD/outputs/evaluation/actors/final200-v1-$LABEL"

FINAL_CMD=(
  "$GRPO_PYTHON" -u scripts/run_standalone_checkpoint_evaluation.py
  --evaluation-role final
  --model "$RL_MODEL"
  --model-name "$LABEL"
  --source-checkpoint "$RL_SOURCE"
  --assets "$FINAL_ASSETS"
  --output-root "$OUT"
  --actor-log-root "$ACTORS"
  --vllm-bin "$VLLM_BIN"
  --actor-ports 18102 18103 18104 18105
  --gpu-indices 0 1 2 3
  --startup-timeout 900
)

# 不启动 Actor、不调用 Shopper 的命令与资产预检
"${FINAL_CMD[@]}" --dry-run

# 真实检查 Environment v2.1 / Reward v4 和 Shopper 鉴权；只产生一次极小 API 请求
"${FINAL_CMD[@]}" --preflight-only

# 正式运行
"${FINAL_CMD[@]}"

# 进程中断后恢复；入口会先核对 evaluation_plan.json，模型或配置变化将拒绝恢复
"${FINAL_CMD[@]}" --resume
```

完成日志必须包含 `STANDALONE CHECKPOINT EVALUATION COMPLETED`，且
`evaluation_results.json` 必须记录：

```text
schema_version        shopping-final-model-evaluation-v1
evaluation_role       final
final_evaluation_used true
reward_contract       shopsimulator-reward-v4
expected_tasks        200 / condition
asset manifest        f9a3970c9bc59374ac23741a4a4519dca44832cc55414ce55944bb1fd446551e
```

建议运行顺序为 Base → SFT → RL。发生 API/环境错误时先修复并从原输出恢复，不得删除失败 task 或改变
固定分母。

## 5. 三模型离线汇总

三次运行全部完成后执行：

```bash
python scripts/summarize_final_evaluations.py \
  --run base=outputs/evaluation/final200-v1/base \
  --run sft=outputs/evaluation/final200-v1/sft-325 \
  --run rl=outputs/evaluation/final200-v1/carl-bpo-v2.1 \
  --output-dir outputs/evaluation/final200-v1/comparison
```

输出包括：

- `final_report.md`：可直接摘入 README 的结果表；
- `final_comparison.json`：三条件原始计数、Wilson 95% CI、模型间逐题 gains/losses/ties 和每条件
  exact McNemar 检验。三条件共享 task ID，因此合计 flips 只作描述，不跨条件计算显著性。

这一步完全离线，不调用 Shopper 或 Judge。Final-200 完成后不能因为结果不理想而重选 checkpoint、修改
Reward/prompt 或重新抽取任务；任何新方案必须使用新版本号，并把本次 Final 视为已公开开发证据。
