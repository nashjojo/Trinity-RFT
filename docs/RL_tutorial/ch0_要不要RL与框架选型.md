# 第 0 章：要不要 RL？框架怎么选？

> 在你 pip install 任何东西之前，先用 5 分钟读完这一章。如果你的任务不该用 RL，再多漂亮的曲线也没用。

---

## 0.1 你真的需要 RL 吗

很多人来到 RL，是因为看到几张漂亮的 reward 曲线。但**真正适合上 RL 的任务不多**——大多数情况下，更便宜、更可控的方法是：

| 你的现状 | 推荐方法 | 为什么不是 RL |
|---|---|---|
| 任务有清晰的"标准答案"，且你有一批高质量答案 | **SFT**（Supervised Fine-Tuning） | RL 在这里没有信息增量，反而引入梯度噪音 |
| 任务靠改 prompt 就能从 30% 提到 80% | **Prompt Engineering** | RL 一晚上不如你改 prompt 改 1 小时 |
| 任务输出主观（写作风格、对话好坏），没有可计算的对错 | **DPO / RLHF**（用人类偏好）| 没有 verifier，标准 RL 跑不动 |
| 任务可以被一个**程序**自动判分，且模型偶尔能做对、偶尔做错 | ✅ **RL** | 这是 RL 真正的甜区 |

最后一行就是本 tutorial 关心的场景。具体地说，三个条件必须**同时满足**：

1. **可验证的 reward**：能写一段代码 / 跑一组测试，给任意一条 trajectory 打 0–1 的分。判分要**确定性**——同一条 trajectory 跑两次得分必须一样。
2. **非平凡的初始 pass rate**：base policy 在 G 次采样里**至少做对 1 次**（>0%），且**没有全做对**（<100%）。前者保证有正样本可学，后者保证有改进空间。
3. **多步决策 / 工具使用**（可选但有趣）：如果任务是单步问答，RL 退化成 RLVR，本 tutorial 第 §1.0 已论证它不够"agentic"；如果任务需要多步工具调用 + 环境状态变化，RL 的优势会非常明显。

**反例自检**：

- "我想让模型写出更幽默的回答" → ❌ 没法自动判分，去查 DPO；
- "我想让模型在 GSM8K 上从 70% 提到 75%" → ⚠️ 可以用 RL，但收益小，可能 SFT 更香；
- "我想让模型在 sandbox 里用 sqlite + shell 完成一个 KV 存储任务，任务正确性能被 checklist 自动判分" → ✅ 来对地方了。

---

## 0.2 主流 RL 框架对比

如果你已经决定要做 RL，下一个问题是：用哪个框架？2026 年初的开源生态主要有这几个：

| 框架 | 强项 | 短板 | 适合场景 |
|---|---|---|---|
| **[Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)**（本 tutorial 选用）| Multi-step agentic 原生支持、与训练后端解耦、yaml 即配 | 文档较新，案例还在补充 | Agentic RL（多步工具使用） |
| [verl](https://github.com/volcengine/verl) | 性能强、FSDP/Megatron 调优深入 | 配置层级深、新手起步陡 | 大规模 RLHF |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 起步简单、社区活跃 | Single-step 偏 RLHF，agentic 支持弱 | 经典 RLHF / GSM8K 类任务 |
| [TRL](https://github.com/huggingface/trl) | HF 生态最熟、入门最浅 | 大规模训练性能不如 verl | 单卡 / 小规模实验 |

> **本 tutorial 选 Trinity-RFT 的核心理由**：它把"trainer 算法"和"训练后端"解耦——你写的 yaml 永远只描述"算法 + 数据"，至于这条 forward / backward 是在 Tinker 云端跑还是在你本地的 TuFT 上跑，trinity 完全无感知。这让"零本地 GPU 入门"和"有了 GPU 切换到自部署"是同一份 yaml、改一行 `base_url` 就完成。

---

## 0.3 训练后端：Tinker vs TuFT

Trinity-RFT 自己**不做** forward / backward / weight update，它通过 [tinker SDK](https://tinker.thinkingmachines.ai/) 把这些请求转发到训练后端。后端两选一：

| 后端 | 部署方式 | 你需要 | 适合 |
|---|---|---|---|
| **[Tinker](https://tinker.thinkingmachines.ai/) 云服务** | 注册账号，拿 API key | 0 张 GPU | **入门首选** ⭐ |
| **[TuFT](https://github.com/modelscope/TuFT)**（自部署）| 在自己机器上起 server | 计算资源（GPU 集群）| 自有数据敏感 / 想自己改训练栈 |

**对教程读者的推荐**：

- 你**第一次**跑 RL → 直接用 Tinker，省去本地配 vLLM / FSDP / weight sync 的两天踩坑时间；
- 跑通之后想做改动（自己改 reward / 加 trick / 换模型 size）→ 切换到 TuFT，把 [`queries_simple.train.tinker.ch1_repro.yaml`](../../examples/copaw_rl/queries_simple.train.tinker.ch1_repro.yaml) 里 `base_url` 从 Tinker 换成 `http://localhost:10610` 即可，**算法层 yaml 一字不改**。

---

## 0.4 本 tutorial 用到的栈

为了让你**今天就能开始**，我们把所有选择都钉死：

| 维度 | 选择 |
|---|---|
| 框架 | [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) |
| 算法 | multi-step GRPO（G=8） |
| Base model | `Qwen/Qwen3-4B-Thinking-2507`（4B + LoRA rank=8） |
| 训练后端 | Tinker（默认）/ TuFT（可选） |
| Sandbox | E2B（真实 shell + 文件系统） |
| 数据集 | `queries_simple v20_top8`（8 个真实编程任务） |
| 训练长度 | 19 step（一晚上） |
| 评估指标 | `rollout/score/mean`（0–100，64 traj 均值） |

第 1 章我们就用这套栈，**不解释任何原理**地把实验跑起来，让你先看到 `score 72 → 82 (+10pp)` 这条曲线。然后从第 2 章开始一层一层把黑盒拆开。

---

## 0.5 这一章你应该带走的

✅ **判断 RL 适不适合你的任务**：可验证 reward + 非平凡 pass rate + （可选）多步决策。
✅ **Trinity-RFT 的关键卖点**：算法和后端解耦，零 GPU 入门 / 自部署可平滑切换。
✅ **本 tutorial 的固定栈**：Trinity-RFT + GRPO + Qwen3-4B + Tinker + E2B + queries_simple v20_top8。

❌ **这一章你不需要懂**：什么是 PPO / GRPO / advantage / KL penalty / forward-backward。这些是后面几章的事。

---

**返回**：[教程总览](./README.md) ｜ **下一章**：[第 1 章：5 分钟跑通一个最小训练循环](./ch1_5分钟跑通.md)
