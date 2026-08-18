# B 探针：澄清 + 个性化价值验证

> 目的：在投入大工程前，验证"在欠约束/有歧义的购物指令上，允许 Agent 做 1~2 轮澄清，能否显著提升任务成功、降低错误购买，且不伤害清晰任务"。
>
> 关联文档：`docs/superpowers/specs/2026-08-18-probe-b-multiturn-personalization-design.md`

## 目录结构

```
probes/
  build_tasks.py        # 阶段1：从 data/sft 构造欠约束任务（clear/under query + fake_profile）
  user_simulator.py     # 阶段2：受控版用户模拟器（规则+模板，只答所问）
  test_user_simulator.py# 用户模拟器单元测试
  runner.py             # 阶段3：双臂运行器（基线 vs 澄清），mock/real 两种模式
  metrics.py            # 阶段4：指标计算 + 判据 PASS/FAIL
  data/tasks.json       # 生成的 25 个欠约束任务
  data/tasks_review.md  # 人工抽查清单（未勾选）
  outputs/              # 轨迹与结果（git-ignore 可加）
  README.md             # 本文档
```

## 快速开始

```bash
# 1) 离线冒烟（不依赖环境/API，验证链路）
python probes/runner.py --mode mock --limit 3

# 2) 真实运行（需 ShopSimulator 服务 + LLM API）
python probes/runner.py --mode real --limit 5   # 先小批量
python probes/runner.py --mode real             # 全量 25 任务 × 2 臂

# 3) 指标与判据
python probes/metrics.py
```

## 真实运行前置

- ShopSimulator 环境服务：`bash scripts/start_environment.sh`（端口 5700，无 GPU）
- LLM API：`RealLLM` 走 OpenAI 兼容接口，通过环境变量提供：
  - `OPENAI_BASE_URL`（如 DeepSeek 的 `https://api.deepseek.com/v1`）
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`（如 `deepseek-chat`）
- **注意**：真实模式的 `RealEnv` 依赖环境的 `observation_state` 结构，首次运行前
  建议先手动 reset+step 一次确认字段（见 `runner.py` 中兜底逻辑）。

## 指标与判据（预先固定）

| 指标 | 判据 |
|---|---|
| 严格成功率（模糊子集） | 澄清臂 ≥ 基线 +10pp |
| 错误购买率 | 澄清臂 ≤ 基线 |
| 平均澄清轮数 | ≤ 2 轮 |
| 平均步数 | ≤ 1.3x 基线 |
| 清晰任务成功率（对照） | 两臂相当 |

## 当前进度

- [x] 阶段1：任务欠约束化（25 任务生成，`tasks_review.md` 待人工抽查勾选）
- [x] 阶段2：受控用户模拟器 + 单测（5/5 PASS）
- [x] 阶段3：双臂运行器 + mock 冒烟通过
- [x] 阶段4：指标计算与判据
- [ ] 阶段3 真实运行（需环境服务 + API key）
- [ ] 阶段5 结论填写（下表）

## 结果（待真实运行后填写）

> 占位：跑完 `--mode real` + `metrics.py` 后，把对比表与结论贴到这里。

## 诚实边界

- 探针验证的是**"澄清对欠约束指令的价值"**（机制层面），**不等于**真实用户画像个性化的完整验证；
- 本探针用"从原 Query 隐藏约束"代理用户偏好（`fake_profile`），真正的结构化用户画像（年龄/性别/消费层级/品牌偏好等）需在后续自建多轮数据时按 ShopSimulator 论文构造；
- mock 模式 reward 为 None，指标仅验证流程，不代表真实效果。
