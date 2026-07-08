# RL Tutorial：从黑盒到白盒的一条线

> 📌 本教程所在仓库是 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 的修改版，专门用于配套本教程；改动集中在 `docs/RL_tutorial/` 与 `examples/copaw_rl/`。完整 fork 说明见[项目主页](https://github.com/nashjojo/Trinity-RFT/tree/public/copaw_rl)。

> 这是一份**真正能在一晚上跑通**的 Agentic-RL 入门教程。7 章按"黑盒 → 拆解 → 实验"的顺序：先让你看到曲线、有直觉，再一层一层打开内部、最后动手改参数跑 ablation。

> 💡 **离线 HTML 版**：本目录内置构建好的网页版教程（入口 [`index.html`](./index.html)，带章节导航栏，完全离线可用）。markdown 与网页版内容一一对应；md 更新后运行 [`build_html.py`](./build_html.py) 可重新生成。

---

## 整体结构

```
第 0 章：要不要 RL？框架选型？        ← 在你 pip install 之前
    ↓
第 1 章：5 分钟跑通最小训练循环        ← 黑盒模式，看 score 72→82 的曲线，建立直觉
    ↓
第 2 章：拆开黑盒 — 单次 rollout 内部
第 3 章:拆开黑盒 — reward 怎么算
第 4 章：拆开黑盒 — GRPO advantage 怎么来
第 5 章：拆开黑盒 — 模型权重怎么更新     ← 4 章拆解，每章带学员回到第 1 章用新理解重看
    ↓
第 6 章：改黑盒 — 换任务、换模型、换 reward    ← 真正学到 RL 直觉的方式
```

---

## 章节链接

| 章 | 标题 | 模式 | 大约耗时 |
|---|---|---|---|
| [第 0 章](./ch0_要不要RL与框架选型.md) | 要不要 RL？框架选型？ | 决策 | 5 分钟阅读 |
| [第 1 章](./ch1_5分钟跑通.md) | 5 分钟跑通最小训练循环 | 黑盒 / 操作 | 10 分钟准备 + 9 小时挂着跑 |
| [第 2 章](./ch2_单次rollout内部.md) | 单次 rollout 到底发生了什么 | 拆解 | 15 分钟阅读 |
| [第 3 章](./ch3_reward怎么算.md) | reward 怎么算的 | 拆解 | 15 分钟阅读 |
| [第 4 章](./ch4_GRPO_advantage.md) | GRPO advantage 怎么来 | 拆解 | 15 分钟阅读 |
| [第 5 章](./ch5_权重更新.md) | 模型权重怎么更新 | 拆解 | 15 分钟阅读 |
| [第 6 章](./ch6_改黑盒做实验.md) | 换任务、换模型、换 reward | 实验 | 每个 ablation 9 小时 |

---

## 写作约定

每一章末尾都有：

- ✅ **这一章你应该带走的**（3–4 条）
- ❌ **你还不需要懂**（指明哪些内容延后到哪一章）
- 💡 **留给自己的问题**（自然过渡到下一章）

**这样你永远知道**：

1. 当前应该理解到什么程度（不焦虑）；
2. 没理解的部分会在哪一章解决（不挫败）；
3. 下一章为什么要读（不迷茫）。

---

## 配套素材

| 文件 | 用途 |
|---|---|
| [`docs/2026-06-08_agentic_rl_v22_tutorial.md`](../2026-06-08_agentic_rl_v22_tutorial.md) | 本教程的“长篇精炼版”，单文档完整呈现 |
| [`docs/_v22_first20_rollout_score.png`](../_v22_first20_rollout_score.png) | 第 1 章 / 第 2 章引用的核心曲线图 |
| [`scripts/tutorial/minimal_rl_loop.py`](../../scripts/tutorial/minimal_rl_loop.py) | 训练循环 4 步骨架（ch2-ch5 对照阅读） |
| [`examples/copaw_rl/queries_simple.train.tinker.ch1_repro.yaml`](../../examples/copaw_rl/queries_simple.train.tinker.ch1_repro.yaml) | baseline yaml（ch1 复现配置，`run.sh` 默认使用） |
| [`examples/copaw_rl/queries_simple/data/v20_top8/tasks.json`](../../examples/copaw_rl/queries_simple/data/v20_top8/tasks.json) | 8-task 数据集 |

---

## Trinity-RFT 是什么

**[Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)** 是一个把 trainer 算法和训练后端**解耦**的 RL 框架——你写的 yaml 永远只描述"算法 + 数据"，至于这条 forward / backward 是在 Tinker 云上跑还是本地 TuFT 上跑，trinity 完全无感知。这让"零本地 GPU 入门"和"有了 GPU 切换到自部署"是同一份 yaml、改一行 `base_url` 就完成。

→ 详细对比与选型理由见 [第 0 章](./ch0_要不要RL与框架选型.md)。

---

## 反馈与贡献

发现教程里的错误 / 不清楚的地方，欢迎给 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 提 issue。让 Agentic RL 从"少数大厂能做的事"变成"每个有兴趣的工程师都能上手的事"，需要每个读者的反馈。

---

**开始阅读**：[第 0 章：要不要 RL？框架选型？](./ch0_要不要RL与框架选型.md)
