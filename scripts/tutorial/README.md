# Tutorial Interactive Scripts

教程 ch2–ch5 的配套交互脚本。每个脚本都支持 `--sample` 模式（读取预置真实数据，无需 API key），让你在不跑训练的情况下亲手验证每一章的概念。

## 快速开始

```bash
cd Trinity-RFT

# 第 2 章：查看 trajectory 内部 — 对比 base policy 和训练后的行为变化
python scripts/tutorial/ch2_inspect_trajectory.py --sample --compare

# 第 3 章：看 reward 打分流程 — 从 raw checks 到 trajectory reward
python scripts/tutorial/ch3_score_trajectory.py --sample

# 第 4 章：GRPO advantage 计算 — 组内相对优势
python scripts/tutorial/ch4_compute_advantage.py --sample

# 第 5 章：LoRA 参数量与训练配置
python scripts/tutorial/ch5_inspect_training.py --sample
```

## 目录结构

```
scripts/tutorial/
├── README.md                           # 本文件
├── sample_data/
│   └── simple_085/
│       ├── step1_session.json          # base policy 的 trajectory（score=0.33）
│       ├── step1_result.json           # 对应的 verifier 打分结果
│       ├── step19_session.json         # step 19 policy 的 trajectory（score=1.0）
│       ├── step19_result.json          # 对应的 verifier 打分结果
│       └── group_rewards.json          # G=8 组的 reward 列表（供 ch4）
├── ch2_inspect_trajectory.py           # 解析 trajectory 的 ReAct 循环
├── ch2_rollout_single.py              # 在线跑一条 trajectory（需 API key）
├── ch3_score_trajectory.py            # 展示 reward 评分全流程
├── ch4_compute_advantage.py           # GRPO advantage 计算演示
└── ch5_inspect_training.py            # LoRA 参数与训练配置检查
```

## 设计原则

- **全部以 simple_085**（asyncio TCP echo server）为主线，贯穿各章
- simple_085 是 v22 实验中改进最明显的任务：score 从 0.50 → 0.83（+33 pp）
- 预置数据从真实的 v22 checkpoint 提取，不是模拟数据
- 提供 step 1（base policy，失败案例）和 step 19（训练后，成功案例）两条 trajectory 对比

## 两种使用模式

| 模式 | 命令 | 需要 | 适合 |
|---|---|---|---|
| **离线**（推荐） | `--sample` | 无（仅需 Python 3） | 阅读教程时随手验证 |
| **在线** | `--session <path>` | ch1 跑完的 checkpoint | 分析自己的实验数据 |

## 样本数据说明

`sample_data/simple_085/` 中的数据来自 v22 实验（`0608-v22-grad-accum`）：

- **step1_session.json**: `step_-1_rollout/simple_085/` 中 score=0.33 的一条 trajectory
  - 模型用 `asyncio.run()` 启动 server → 与 sandbox event loop 冲突 → 失败
- **step19_session.json**: `step_17_rollout/simple_085/` 中 score=1.0 的一条 trajectory
  - 模型用 `asyncio.new_event_loop()` 正确启动 + sleep 等待就绪 → 成功
- **group_rewards.json**: step 1 时同一个 prompt 的 G=8 条 trajectory 的 reward：
  `[0.33, 0.33, 0.67, 0.67, 0.67, 0.67, 1.0, 1.0]`

注意：session.json 中的 token 级数据（`_model_trajectory` 中的 `token_ids`、`logprobs`）已被替换为长度元信息，以减小文件体积。对话内容（`agent.memory.content`）保留完整。
