# CARL-BPO v3 训练与评测交接

> 更新时间：2026-09-03
>
> 权威分支：`feat/bpo2`
>
> 当前候选：CARL-BPO v3 `global_step_200` merged
>
> Final-200：未使用

本文只记录当前可执行状态。v1/v2/v2.1 的历史设计与失败分析见
[CARL-BPO 主文档](carl-bpo.md)，旧 `full-bpo-v1` 结果见
[BPO 正式训练结果](bpo-formal-results.md)。事实优先级为源码与配置、服务器原始产物、本文，
最后才是历史文档描述。

## 1. 当前结论

CARL-BPO v3 已实现并从 SFT-325 训练到 step 200。训练在 checkpoint 保存后由人工停止，以便执行
DEV-500×3；不是崩溃，也不是500-step正式运行完成。当前 checkpoint 可以继续训练。

最重要的新结论不是“RL没有更新”，而是此前评测模型选错：

- `scripts/export_grpo.sh` 导出的顶层 `model.safetensors` 仍是 SFT-325 base；
- RL 更新保存在 `lora_adapter/adapter_model.safetensors`；
- 直接评测 `carl-bpo-v3-step200-export` 等于漏掉 LoRA；
- 合并 adapter 后模型 SHA-256 改变，100个固定 action 中99个出现非零 decision-log-prob delta；
- 合并模型在 DEV-500×3 的 strict total 为 `0.655`，SFT-325 为 `0.647`。

因此，所有后续 scoring、DEV 或 Final 评测必须使用带 `-merged` 的目录。

## 2. 当前运行身份

```text
run name: carl-bpo-v3-step500-r4000-seed20260823-20260901-193508
SwanLab run: mode/shopping-multiturn-agentic/sconfuhu
start model: outputs/models/sft-checkpoint-sweep-dev200-v1/checkpoint-325
saved checkpoint: global_step_200
configured horizon: 500
save frequency: 25
validation steps: 0, 10, 50, 100, 150, 200, ..., 500
```

服务器应保留：

```text
outputs/models/<run>/run_contract.json
outputs/models/<run>/training_diagnostics.jsonl
outputs/models/<run>/latest_checkpointed_iteration.txt
outputs/models/<run>/global_step_200/
outputs/bpo/logs/<run>.log
```

`audit_bpo_formal_run.py` 的 N500/R4000 成功标志只适用于完整500步运行，不能用于当前中间状态。

## 3. 当前算法合同

权威实现位于 `configs/bpo.yaml`、`src/shopping_grpo/training/bpo/`、
`src/shopping_grpo/training/grpo/dynamic_sampling.py` 和 veRL dynamic patch。

- 每步1个 Root K=4和1个 Local K=4，共8个 sibling terminal outcomes；
- Root 对每条完整轨迹的所有真实 action 使用 episode LOO；
- Local 从同一 snapshot 比较4条 continuation，但只有当前 branch action 进入 policy loss；
- Local prefix/suffix 的 policy support为0；
- Local 至少包含2个不同 canonical semantic tool action；
- Local stage仅为 `product`、`option`、`search_strategy`，500组目标为200/175/125；
- action内、group内等权，Root/Local policy mass各0.5；
- train return为 gold `1.25`、valid alternative `1.0`、model failure `-0.075`，其余正常终局
  `0.1 * clip(terminal_utility, -1, 1)`；
- 不按组内标准差归一 advantage；
- 10批为 quality-search边界，120批为无法凑齐严格 Root+Local 的应急硬停止；
- 500步是上限，不是默认模型选择；checkpoint每25步保存，validation主要每50步执行。

“accepted sibling terminal outcomes”只是 accepted groups乘K，不是 effective sample size。

## 4. 正确导出与合并

```bash
cd ~/shopping-grpo

export GRPO_PYTHON=/home/gjx/.venvs/shopping-grpo/bin/python
export MERGE_PYTHON="$PWD/.venv/bin/python"

export RL_RUN="$PWD/outputs/models/carl-bpo-v3-step500-r4000-seed20260823-20260901-193508"
export RL_STEP=200
export RL_SOURCE="$RL_RUN/global_step_$RL_STEP"
export RL_EXPORT="$PWD/outputs/models/carl-bpo-v3-step200-export"
export RL_MERGED="$PWD/outputs/models/carl-bpo-v3-step200-merged"

test -d "$RL_SOURCE/actor"
test ! -e "$RL_EXPORT"
test ! -e "$RL_MERGED"

bash scripts/export_grpo.sh "$RL_SOURCE/actor" "$RL_EXPORT"

test -f "$RL_EXPORT/model.safetensors"
test -f "$RL_EXPORT/lora_adapter/adapter_model.safetensors"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=src "$MERGE_PYTHON" \
  scripts/merge_lora_adapter.py \
  --base-model "$RL_EXPORT" \
  --adapter "$RL_EXPORT/lora_adapter" \
  --output "$RL_MERGED" \
  --bf16
```

当前已得到：

```text
SFT-325 SHA-256:   a6bd209090d8fef4a842639af3b2f403467794d40023925362910d99fd8338b0
export base:       a6bd209090d8fef4a842639af3b2f403467794d40023925362910d99fd8338b0
merged v3 step200: 622204f176539806e412475ef192b04700ab31112c0cc4f15748605b83dfe1f6
```

## 5. DEV-500×3 结果

| model | gap ask | gap no-ask | complete ask | strict total | gap gain | unnecessary ask | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT-325 | 0.690 | 0.528 | 0.722 | 0.647 | +0.162 | 0.938 | 0.6010 |
| v3 step200 merged | 0.700 | 0.528 | 0.736 | 0.655 | +0.172 | 0.924 | 0.6138 |
| delta | +0.010 | 0.000 | +0.014 | +0.008 | +0.010 | -0.014 | +0.0128 |

逐题 strict flips：

```text
gap-ask-enabled:      gains 13, losses 8, net +5
gap-ask-disabled:     gains 4,  losses 4, net  0
complete-ask-enabled: gains 14, losses 7, net +7
total descriptive:   gains 31, losses 19, net +12
```

完整性：v3 done `1492/1500`、reward-valid `1485/1500`、guards 29；SFT 分别为
`1490/1500`、`1486/1500`、32。当前尚缺 merged 模型的
`gold_purchase + valid_alternative_purchase` 原始计数和逐条件显著性检验，因此不能把点估计写成
最终结论。

## 6. 已完成的 Local 语义审计

现有 step1–200 diagnostics 中，已复核150个 product/option Local tree，覆盖137个 task：

| class | count |
|---|---:|
| stable success action | 48 |
| single success, unreplicated | 15 |
| mixed return under same semantic action | 25 |
| failure only | 62 |

48个 stable-success tree中41个动作语义正确、7个错误。结论是：v3 Local 中存在真实、方向正确的
动作对照，但仍有大量树不能仅凭 return contrast 证明当前 branch action 的稳定因果作用。

## 7. 下一项最低成本审计

已经完成的 fixed-state scoring覆盖50个DEV状态和100个correct/wrong候选；merged 模型下99/100的
decision log-prob发生非零变化。下一步只需离线聚合现有 JSONL：

1. 分别统计 product、option、Complete ask-suppression 的
   `delta[(logp_correct - logp_wrong)]`；
2. 报告 margin扩大/缩小/不变的状态数、中位数和分位数；
3. 与 selected Local tree的 correct/wrong语义结果交叉；
4. 从 merged DEV summary补出 gold、valid alternative和combined原始计数。

该审计不生成 rollout、不更新模型、不访问 Final-200。完成后再决定是从 step200续训、冻结
step200，还是把工作转向SFT数据与能力补强。

## 8. 当前禁止误用

- 不评测 `*-export/model.safetensors`；只评测 `*-merged`。
- 不把 `+12/1500` 描述成统计显著或 Final 提升。
- 不为解释当前结果临时改 Reward、K、temperature/top-p、stage比例或 loss权重。
- 不在 DEV-500 上继续 sweep多个 checkpoint。
- 不打开 Final-200，直到唯一 checkpoint、merged模型哈希和评测合同正式冻结。
