# Reward v4：原子约束与价格语义

Reward v4 是与 Reward v3 并存的确定性终局奖励候选。它不调用 LLM、不覆盖 v3，也不属于
PRM 或 step-level shaping。正式切换前必须完成同任务 v3/v4 gold replay、重新冻结开发/测试
任务，并保证 Base/SFT/GRPO 使用同一版本。

## 1. 版本与启用方式

| 项目 | Reward v3 | Reward v4 |
|---|---|---|
| Reward version | `shopsimulator-reward-v3` | `shopsimulator-reward-v4` |
| Feature version | `shopping-reward-features-v1` | `shopping-reward-features-v2` |
| 默认配置 | `configs/environment.json` | `configs/environment-v4.json` |
| 当前默认 | 是 | 否，候选版本 |

启用 v4 时，在启动 ShopSimulator 前显式选择配置：

```bash
export SHOP_ENV_CONFIG="$PWD/environments/ShopSimulator/shop_env/configs/environment-v4.json"
bash scripts/start_environment.sh
```

启动 GRPO 的终端还必须设置
`SHOP_REWARD_VERSION=shopsimulator-reward-v4`，AgentLoop 会在 reset 时核对服务端实际版本，
不匹配即终止，防止误连 v3 环境。

未设置 `SHOP_ENV_CONFIG` 时继续使用 v3，因此历史轨迹和已有 SFT 数据不会被静默重解释。

## 2. 需求原子

v4 只从任务已有 instruction、`attributes`、`instruction_options` 和目标商品类目中编译约束，
不从目标商品价格反推预算。每个原子包含：

```text
atom_id, dimension, strength, weight, requirement, source
```

支持 `category`、`brand`、`model`、`core_function`、`option` 和 `price`。

- `hard`：类目，以及同一语句中有“必须、一定、务必、硬性、不可缺少”等标记的要求；
- `required`：没有软硬标记的明确要求；
- `soft`：同一语句中有“最好、优先、尽量、希望、偏好、倾向”等标记的要求，以及
  “约/左右”价格目标。

`hard` 和 `required` 权重为 1，`soft` 权重为 0.5。编译结果和逐原子比较写入 Reward
evidence，便于审计哪些要求导致结果变化。

## 3. 价格约束

价格编译器版本为 `shopping-price-constraint-v2`，支持阿拉伯数字、常规中文数字、元/块、
千/k/万：

| 表达 | 编译结果 |
|---|---|
| `不超过230元`、`230元以内` | `hard_max` |
| `至少4千元` | `hard_min` |
| `价格200到250元` | `hard_range` |
| `230元左右`、`预算大约两百三十元` | `soft_target` |
| `预算230元` | `hard_max` |

`soft_target` 使用确定性的 ±10% 区间，最小容差为 5 元。例如 230 元左右编译为
207–253 元。价格只使用实际已选 variant 的可验证价格；无法解析 variant price 时，hard
价格约束使 Reward 无效，soft 价格约束只降低证据覆盖率。

## 4. 评分与终局分类

每个原子得到 `pass / fail / unverifiable`：

```text
S = Σ(weight_i × passed_i) / Σ(weight_i)
C = Σ(weight_i × verifiable_i) / Σ(weight_i)
```

终局顺序：

1. 任一 hard 原子不可验证：`reward_unverifiable`；
2. 任一 hard 原子失败：`wrong_purchase=-0.85`；
3. 所有原子通过且 ASIN 为目标：`gold_purchase=1.0`；
4. 所有原子通过且 ASIN 不同：`valid_alternative_purchase=0.55`；
5. 其余：`partial_alternative_purchase=min(0.25, -0.30 + 0.55 × S)`。

放弃、循环和步数耗尽数值暂时沿用 v3，让本轮变化集中在购买结果判定。

## 5. v3/v4 双算

切换正式协议前，对相同 frozen task 做 gold replay：

```bash
export PYTHONPATH=./src

python scripts/compare_reward_versions.py \
  --tasks data/multiturn/evaluation-dev-v1/tasks.jsonl \
  --output-dir outputs/reward-comparison/development-v1
```

输出包括逐 task 比较、汇总、SHA-256 manifest、v4 新增/丢失 eligible task IDs，以及 v4
修复的价格编译 task IDs。不能仅因 v4 让更多任务 eligible 就采用；必须抽查 gained/lost
bad cases，排除错误解析。

## 6. 与 GRPO 和评测的关系

- 环境终局、评测严格成功和 GRPO runtime 已能识别 v3/v4；
- 同一次 run 只能使用一个 Reward 版本，manifest 记录实际版本；
- 现有 Teacher SFT 数据仍按 v3 gold purchase 验收，应使用双算报告重新审计；
- v4 不奖励“是否提问”，澄清质量继续由 G+/G−/C+ 和五面板评测负责；
- PRM/step-level reward 以后作为独立 GRPO shaping 版本，不写入 v4 终局标签。

## 7. 正式采用门槛

1. 编译和终局单元测试通过；
2. 开发集和正式候选池完成 v3/v4 双算；
3. gained/lost 样本完成抽查；
4. v4 gold replay 得到固定任务集和 replacement manifest；
5. openings、Rubric、Base/SFT/GRPO 全部绑定 v4 manifest；
6. 更新 runtime contract 后，才把默认配置从 v3 切换到 v4。
