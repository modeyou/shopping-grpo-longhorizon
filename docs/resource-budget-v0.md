# 4×RTX 4090 资源预算 v0

- 状态：静态预估；20 条数据 Pilot 后用实测更新
- 模型：Qwen3.5-2B
- 硬件：4×RTX 4090 24GB，PCIe 通信，无 NVLink

## 当前判断

一次 LoRA/QLoRA SFT 可行。GRPO 对 2B 模型也具备可行性，但不能把四张 24GB 卡视为一张连续
96GB 显存，也不能直接复用参考项目的单卡 96GB、24K context 配置。

首轮资源策略：

- SFT 从 8K–12K context、batch size 1、gradient checkpointing 和一次混合训练开始；
- 只有真实轨迹 P95 明显超过 12K 时才提高 context；
- GRPO 保留 LoRA、CPU offload 和 `n=4` 起点，但先完成 1 trajectory、5 steps、20 steps 三档冒烟；
- 依据 20-step 实测再决定四卡 colocate 或划分 rollout/training GPU；
- 4090 PCIe 通信可能降低 FSDP/TP 吞吐，项目以跑通和可复现为优先，不预设论文级速度。

## 需要在数据 Pilot 后测量

| 项目 | 决策用途 |
|---|---|
| 轨迹 input/response token 的均值、P95、最大值 | 冻结 SFT/GRPO context |
| Architect/Critic 输入输出 token 和接受率 | 计算正式 API 成本与候选池大小 |
| Teacher 成功率和平均环境步数 | 计算 SFT 轨迹采集成本 |
| SFT 单 step 时间与四卡峰值显存 | 冻结 batch、累积步数和 epoch |
| GRPO 5/20-step 时间、显存、KV cache 和有效 group 比例 | 冻结卡分配、rollout 数和训练步数 |
| checkpoint、原始轨迹和日志增长速度 | 冻结磁盘保留策略 |

## 当前磁盘建议

在没有实测前，为基础模型、SFT merged model、GRPO actor/checkpoint、vLLM cache、原始 API 响应和
轨迹日志预留至少 150GB 可用空间；若保留多个 GRPO checkpoint，建议 250GB。该数字是容量保护线，
不是最终使用量。

## 冻结时点

完整资源预算在以下条件满足后冻结：

1. 20 条个性化任务 Pilot 完成；
2. 至少 5 条 Teacher 多轮购物轨迹跑通；
3. 对这些轨迹完成 tokenizer 长度统计；
4. 在目标 4×4090 服务器完成一次 SFT forward/backward smoke；
5. 完成 5-step GRPO smoke。

在这些测量完成前，不承诺正式数据条数、SFT context、epoch 或 GRPO steps。
