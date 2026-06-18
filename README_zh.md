[**上游 Trinity-RFT**](https://github.com/agentscope-ai/Trinity-RFT) | [**教程总索引**](./docs/RL_tutorial/README.md) | [**常见问题**](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/faq.html)

> **📌 本仓库是《Agentic-RL 入门教程：从黑盒到白盒的一条线》的配套代码与文档。** 教程教你如何做 agentic RL，其中 Trinity-RFT / Tinker / TuFT / E2B 都只是选用的工具——学会方法论后，换框架、换后端都适用。
> 代码基于 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 修改，本仓库相对上游的改动集中在 [`docs/RL_tutorial/`](./docs/RL_tutorial/)（7 章教程）与 [`examples/copaw_rl/`](./examples/copaw_rl/)（教程配套 baseline 配置、数据集与脚本）。
> 关于**教程内容与配套代码**的问题，欢迎在本仓库提 issue；Trinity-RFT **框架本体**的问题请到 [上游仓库](https://github.com/agentscope-ai/Trinity-RFT/issues)。

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01lvLpfw25Pl4ohGZnU_!!6000000007519-2-tps-1628-490.png" alt="Trinity-RFT" style="height: 90px;">
</div>

<h2 align="center">Agentic-RL 入门教程</h2>

<p align="center">从黑盒到白盒的一条线 —— 一晚上跑通真正的多步 Agentic-RL</p>

<div align="center">

[![paper](http://img.shields.io/badge/cs.LG-2505.17826-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.17826)
[![doc](https://img.shields.io/badge/Docs-blue?logo=markdown)](https://agentscope-ai.github.io/Trinity-RFT/)
[![pypi](https://img.shields.io/pypi/v/trinity-rft?logo=pypi&color=026cad)](https://pypi.org/project/trinity-rft/)
![license](https://img.shields.io/badge/license-Apache--2.0-000000.svg)

</div>

## 📚 教程总览

> 这是一份**真正能在一晚上跑通**的 Agentic-RL 入门教程。7 章按"黑盒 → 拆解 → 实验"的顺序：先让你看到曲线、有直觉，再一层一层打开内部、最后动手改参数跑 ablation。

| 章 | 标题 | 模式 | 大约耗时 |
|---|---|---|---|
| [第 0 章](./docs/RL_tutorial/ch0_要不要RL与框架选型.md) | 要不要 RL？框架选型？ | 决策 | 5 分钟阅读 |
| [第 1 章](./docs/RL_tutorial/ch1_5分钟跑通.md) | 5 分钟跑通最小训练循环 | 黑盒 / 操作 | 10 分钟准备 + 9 小时挂着跑 |
| [第 2 章](./docs/RL_tutorial/ch2_单次rollout内部.md) | 单次 rollout 到底发生了什么 | 拆解 | 15 分钟阅读 |
| [第 3 章](./docs/RL_tutorial/ch3_reward怎么算.md) | reward 怎么算的 | 拆解 | 15 分钟阅读 |
| [第 4 章](./docs/RL_tutorial/ch4_GRPO_advantage.md) | GRPO advantage 怎么来 | 拆解 | 15 分钟阅读 |
| [第 5 章](./docs/RL_tutorial/ch5_权重更新.md) | 模型权重怎么更新 | 拆解 | 15 分钟阅读 |
| [第 6 章](./docs/RL_tutorial/ch6_改黑盒做实验.md) | 换任务、换模型、换 reward | 实验 | 每个 ablation 9 小时 |

完整介绍、写作约定与配套素材见 [教程总索引](./docs/RL_tutorial/README.md)。

**教程配套素材**：

| 素材 | 路径 | 用途 |
|---|---|---|
| Baseline 训练配置 | [`examples/copaw_rl/queries_simple.train.tinker.v22.yaml`](./examples/copaw_rl/queries_simple.train.tinker.v22.yaml) | 第 1 章跑通用 yaml（Tinker 后端） |
| 8-task 数据集 | [`examples/copaw_rl/queries_simple/data/v20_top8/tasks.json`](./examples/copaw_rl/queries_simple/data/v20_top8/tasks.json) | 8 个真实编程任务 |
| 批量跑 ablation 脚本 | [`examples/copaw_rl/entry/batch_run.py`](./examples/copaw_rl/entry/batch_run.py) | 第 6 章跑实验用 |
| 长篇精炼版 | [`docs/2026-06-08_agentic_rl_v22_tutorial.md`](./docs/2026-06-08_agentic_rl_v22_tutorial.md) | 单文档完整呈现 |

## ✨ 这个教程为什么不同

市面上很多"Agentic RL"教程其实是 **GSM8K + Calculator** / **MATH + Python REPL** 这类*单步、verifier 判分*的设置，本质只是 RLVR，不算真 agentic。本教程用的数据集 `queries_simple v20_top8` 包含 8 个**真实的小型工程任务**，模型必须在 [E2B sandbox](https://e2b.dev/) 里多轮调用 shell / 文件读写 / grep 等工具，真正创建、修改、验证文件与服务，判分依赖**容器内的真实运行结果**。

> 详见 [第 1 章 §1.1：为什么这是真正的 Agentic-RL 任务](./docs/RL_tutorial/ch1_5分钟跑通.md)。

## 🛠️ 教程技术栈

> 本教程的核心是**教你做 agentic RL 的方法论**，下面的工具只是载体——学会后换框架、换后端、换模型都适用。

| 角色 | 工具 | 说明 |
|---|---|---|
| 训练框架 | [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) | 把 trainer 算法与训练后端解耦，yaml 只描述"算法 + 数据" |
| 训练后端 | [Tinker](https://tinker.thinkingmachines.ai/) 云服务（默认）/ [TuFT](https://github.com/modelscope/TuFT)（自部署） | 负责实际的 forward/backward/权重更新；0 GPU 入门选 Tinker，有 GPU 选 TuFT |
| Sandbox | [E2B](https://e2b.dev/) | 真实 shell + 文件系统，模型在里面多步调用工具完成任务 |
| 算法 | multi-step GRPO（G=8） | group-relative advantage，无需 value 网络 |
| Base model | `Qwen/Qwen3-4B-Thinking-2507`（4B + LoRA rank=8） | |
| 数据集 | `queries_simple v20_top8`（8 个真实编程任务） | |

> 各工具的对比与选型理由见 [第 0 章 §0.2–§0.4](./docs/RL_tutorial/ch0_要不要RL与框架选型.md)。

## 🚀 快速开始（教程版）

3 步看到第一条曲线（约 10 分钟准备 + 9 小时挂着跑）：

1. **拿到 Tinker API key** — 在 [Tinker](https://tinker.thinkingmachines.ai/) 注册账号（0 张 GPU 即可入门）
2. **准备配置** — 复制 [`examples/copaw_rl/queries_simple.train.tinker.v22.yaml`](./examples/copaw_rl/queries_simple.train.tinker.v22.yaml)，填入 API key 与模型路径
3. **启动训练** — `trinity run --config queries_simple.train.tinker.v22.yaml`，等一晚上看 `rollout/score/mean` 从 **72.4 → 82.4（+10pp）**

> 完整步骤（含环境安装、数据准备、出图脚本）见 [第 1 章：5 分钟跑通](./docs/RL_tutorial/ch1_5分钟跑通.md)。

---

## 附录 A：工具栈各组件简介

> 下面是教程选用工具的简要介绍，完整文档见各项目主页。教程核心是 agentic RL 方法论，这些工具均可替换。

**Trinity-RFT**（训练框架）：通用、灵活的 LLM 强化微调框架，将 RFT 解耦为 Explorer / Trainer / Buffer 三模块协同运行。本仓库代码基于它修改。完整文档见 [上游仓库](https://github.com/agentscope-ai/Trinity-RFT) 与 [官方文档站](https://agentscope-ai.github.io/Trinity-RFT/)。

**Tinker / TuFT**（训练后端）：负责实际的 forward/backward/权重更新——Trinity-RFT 自己不做这些，通过 [Tinker SDK](https://tinker.thinkingmachines.ai/) 转发到后端。[Tinker](https://tinker.thinkingmachines.ai/) 是云端服务（0 GPU 入门），[TuFT](https://github.com/modelscope/TuFT) 是自部署方案（需要 GPU 集群）；切换只需改 yaml 里一行 `base_url`，算法层 yaml 一字不改。

**E2B**（sandbox）：提供真实 shell + 文件系统的容器，模型在里面多轮调用 `execute_shell_command` / `read_file` / `write_file` / `edit_file` / `grep_search` 等工具完成任务，判分依赖容器内真实运行结果（文件内容、服务可访问性、sqlite 表等）。

<details><summary>Trinity-RFT 支持的 RFT 模式、算法与安装（精简，完整列表见上游）</summary>

- **RFT 模式**：同步/异步、on-policy/off-policy、在线/离线；rollout 与训练可分离、可独立扩展；支持经验回放。
- **算法**：PPO、GRPO、SFT、DPO、CHORD、REC 系列、RLOO、REINFORCE++、GSPO、TOPR、sPPO、AsymRE、CISPO、SAPO、On-Policy Distillation、JSD 等，完整列表与配置见 [上游算法模块](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/algorithm.py)。
- **训练后端**：[Tinker](https://tinker.thinkingmachines.ai/) 云服务（无 GPU 入门）/ [TuFT](https://github.com/modelscope/TuFT)（自部署）。
- **安装**：`pip install -e ".[vllm,flash_attn]"`（有 GPU）或 `pip install -e ".[tinker]"`（无 GPU），详见上游 [快速上手](https://github.com/agentscope-ai/Trinity-RFT#quick-start)。

</details>

## 附录 B：致谢与引用

本教程基于 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 构建，感谢上游团队的开源贡献。Trinity-RFT 还基于 verl、vLLM、FSDP、Megatron-LM、Data-Juicer、AgentScope、Ray 等优秀开源项目（完整致谢见上游 README）。

如果本仓库对你的研究有帮助，请引用 Trinity-RFT 技术报告：

```bibtex
@misc{trinity-rft,
      title={Trinity-RFT: A General-Purpose and Unified Framework for Reinforcement Fine-Tuning of Large Language Models},
      author={Xuchen Pan and Yanxi Chen and Yushuo Chen and Yuchang Sun and Daoyuan Chen and Wenhao Zhang and Yuexiang Xie and Yilun Huang and Yilei Zhang and Dawei Gao and Yaliang Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2505.17826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17826},
}
```

## 贡献指南

* **教程与配套代码**（`docs/RL_tutorial/`、`examples/copaw_rl/`、README 等）：欢迎在本仓库提 issue / PR，包括教程勘误、配置改进、新 ablation 实验。
* **Trinity-RFT 框架本体**（`trinity/` 核心代码、算法、buffer 等）：请贡献到 [上游仓库](https://github.com/agentscope-ai/Trinity-RFT)，参见上游 [CONTRIBUTING.md](https://github.com/agentscope-ai/Trinity-RFT/blob/main/CONTRIBUTING.md)。
