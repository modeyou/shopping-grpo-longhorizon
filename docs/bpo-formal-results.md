# BPO 正式训练结果

## 冻结运行

- 算法：`full-bpo-v1`
- 起点：SFT `checkpoint-325`
- 奖励：ShopSimulator Reward v4，原生 profile `none`
- 随机种子：`20260823`
- 训练规模：200 个优化器步；每步接收 2 棵有效树，每棵树包含 4 个 sibling return
- 运行标签：`bpo-native-v4-step200-r1600-seed20260823-20260826-113750-r3`
- SwanLab 运行 ID：`4cmh0p3k`
- 服务器检查点：`outputs/models/<run-tag>/global_step_200`

模型权重、完整日志与 SwanLab 本地缓存属于服务器生成物，不提交到 Git。本文件记录可复核的实验结论；机器证据仍以 `run_contract.json`、`training_diagnostics.jsonl`、完整日志和检查点记录为准。

## 完成情况审计

| 项目 | 结果 |
|---|---:|
| 训练进度 | 200/200 |
| 检查点记录 | 200 |
| 有效树 | 400/400 |
| 有效回报数 | 1600/1600 |
| 跳过优化器更新 | 0 |
| 主干轨迹总数 | 1,732 |
| 分支轨迹总数 | 5,196 |
| 轨迹总数 | 6,928 |
| 生成回复 token 数 | 1,478,990 |
| 环境交互次数 | 30,255 |
| Shopper API 调用次数 | 880 |
| 总运行时间 | 8:57:06 |

## Step 200 冻结验证结果

验证使用正式 GRPO validation parquet 中的 400 条 gap/complete 样本，不是最终 200-task 测试集。

| 指标 | 结果 |
|---|---:|
| 严格成功率 | 69.75% |
| 购买成功率 | 70.50% |
| 平均奖励 | 0.6772444853 |
| 正常结束率 | 99.00% |
| 奖励有效率 | 99.50% |
| 平均步数 | 4.57 |
| 采样无效率 | 0.50% |
| 基础设施无效率 | 0.00% |
| 奖励不可验证率 | 0.50% |
| Shopper 提问率 | 100.00% |

## SwanLab 收尾事件

最终日志已经包含 `Training Progress: 200/200`、step 200 训练指标、检查点保存记录和 `Final validation metrics`。随后出现的 `RuntimeError: cannot join current thread` 来自 veRL `Tracking.__del__` 在 SwanLab terminal worker 线程中再次调用 `finish()`，不是训练、保存或验证失败。

项目补丁已将 Tracking 收尾改为幂等操作，并在训练主线程写入最后一条训练或验证指标后显式调用 `finish()`。旧 SwanLab 运行不强制回填；本地日志与本文件共同保存 step 200 结果。
