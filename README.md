[**中文主页**](https://github.com/agentscope-ai/Trinity-RFT/blob/main/README_zh.md) | [**Tutorial Index**](./docs/RL_tutorial/README.md) | [**Upstream Trinity-RFT**](https://github.com/agentscope-ai/Trinity-RFT)

> **📌 This repo hosts the code and docs accompanying the tutorial *Agentic-RL: From Black-Box to White-Box*.** The tutorial teaches you how to do agentic RL — Trinity-RFT / Tinker / TuFT / E2B are just the chosen tools; once you learn the methodology, swapping the framework or backend is straightforward.
> The code is based on [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT); changes relative to upstream are concentrated in [`docs/RL_tutorial/`](./docs/RL_tutorial/) (a 7-chapter tutorial, in Chinese) and [`examples/copaw_rl/`](./examples/copaw_rl/) (tutorial-accompanying baseline configs, dataset and scripts).
> For issues about the **tutorial and accompanying code**, feel free to open an issue here; for the Trinity-RFT **framework itself**, please file them at the [upstream repo](https://github.com/agentscope-ai/Trinity-RFT/issues).

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01lvLpfw25Pl4ohGZnU_!!6000000007519-2-tps-1628-490.png" alt="Trinity-RFT" style="height: 90px;">
</div>

<h2 align="center">Agentic-RL Tutorial</h2>

<p align="center">From black-box to white-box — get a real multi-step Agentic-RL run going in one night</p>

<div align="center">

[![paper](http://img.shields.io/badge/cs.LG-2505.17826-B31B1B?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.17826)
[![doc](https://img.shields.io/badge/Docs-blue?logo=markdown)](https://agentscope-ai.github.io/Trinity-RFT/)
[![pypi](https://img.shields.io/pypi/v/trinity-rft?logo=pypi&color=026cad)](https://pypi.org/project/trinity-rft/)
![license](https://img.shields.io/badge/license-Apache--2.0-000000.svg)

</div>

> **Note:** The tutorial chapters are written in Chinese. The structure below mirrors [README_zh.md](./README_zh.md).

## What you get in one night

> **Zero GPUs** + one overnight run = the model actually learns new skills.

|  | Highlight |
|---|---|
| **Low barrier** | Tinker cloud — no local GPU needed. Register API key, copy yaml, bash — 10 min setup |
| **Truly agentic** | Not GSM8K + Calculator. 8 real engineering tasks: the model multi-turn invokes shell/file tools inside containers, graded on real runtime results |
| **Visible behavior change** | After 19 steps: from "write buggy code and give up" to "check port status → kill conflict → retry until success" |

<div align="center">
  <img src="./docs/_v22_first20_rollout_score.png" alt="score 72.4 → 82.4" width="600">
  <p><b>rollout/score/mean: 72.4 → 82.4 (+10 pp)</b>, pass rate 50% → 59%</p>
</div>

## 📚 Tutorial Overview

> A hands-on Agentic-RL tutorial you can actually finish in one night. 7 chapters follow a "black-box → dissect → experiment" path: first see the curve and build intuition, then peel back each layer, and finally run your own ablations.

| Ch | Title (zh) | Mode | Approx. time |
|---|---|---|---|
| [Ch.0](./docs/RL_tutorial/ch0_要不要RL与框架选型.md) | 要不要 RL？框架选型？ | Decision | 5 min read |
| [Ch.1](./docs/RL_tutorial/ch1_5分钟跑通.md) | 5 分钟跑通最小训练循环 | Black-box / ops | 10 min prep + 9h unattended |
| [Ch.2](./docs/RL_tutorial/ch2_单次rollout内部.md) | 单次 rollout 到底发生了什么 | Dissect | 15 min read |
| [Ch.3](./docs/RL_tutorial/ch3_reward怎么算.md) | reward 怎么算的 | Dissect | 15 min read |
| [Ch.4](./docs/RL_tutorial/ch4_GRPO_advantage.md) | GRPO advantage 怎么来 | Dissect | 15 min read |
| [Ch.5](./docs/RL_tutorial/ch5_权重更新.md) | 模型权重怎么更新 | Dissect | 15 min read |
| [Ch.6](./docs/RL_tutorial/ch6_改黑盒做实验.md) | 换任务、换模型、换 reward | Experiment | 9h per ablation |

Full intro, writing conventions and accompanying materials: [Tutorial Index (zh)](./docs/RL_tutorial/README.md).

**Accompanying materials**:

| Material | Path | Usage |
|---|---|---|
| Baseline training config | [`examples/copaw_rl/queries_simple.train.tinker.v22.yaml`](./examples/copaw_rl/queries_simple.train.tinker.v22.yaml) | yaml for Ch.1 (Tinker backend) |
| 8-task dataset | [`examples/copaw_rl/queries_simple/data/v20_top8/tasks.json`](./examples/copaw_rl/queries_simple/data/v20_top8/tasks.json) | 8 real programming tasks |
| Minimal RL loop | [`scripts/tutorial/minimal_rl_loop.py`](./scripts/tutorial/minimal_rl_loop.py) | Conceptual 4-step skeleton mapping to Ch.2–5 |
| Batch ablation runner | [`examples/copaw_rl/entry/batch_run.py`](./examples/copaw_rl/entry/batch_run.py) | For Ch.6 experiments |
| Condensed long-form | [`docs/2026-06-08_agentic_rl_v22_tutorial.md`](./docs/2026-06-08_agentic_rl_v22_tutorial.md) | Single-doc complete presentation |

## 🛠️ Tutorial Tech Stack

> The core of this tutorial is **teaching agentic-RL methodology** — the tools below are just carriers; once learned, you can swap the framework, backend or model.

| Role | Tool | Notes |
|---|---|---|
| Training framework | [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) | Decouples trainer algorithm from training backend; yaml describes only "algorithm + data" |
| Training backend | [Tinker](https://tinker.thinkingmachines.ai/) cloud (default) / [TuFT](https://github.com/modelscope/TuFT) (self-hosted) | Runs the actual forward/backward/weight updates; 0-GPU onboarding via Tinker, self-hosted via TuFT |
| Sandbox | [E2B](https://e2b.dev/) | Real shell + filesystem; the model multi-turn invokes tools inside it to complete tasks |
| Algorithm | multi-step GRPO (G=8) | Group-relative advantage, no value network needed |
| Base model | `Qwen/Qwen3-4B-Thinking-2507` (4B + LoRA rank=8) | |
| Dataset | `queries_simple v20_top8` (8 real programming tasks) | |

> Comparison and rationale for each tool: [Ch.0 §0.2–§0.4 (zh)](./docs/RL_tutorial/ch0_要不要RL与框架选型.md).

## 🚀 Quick Start (tutorial edition)

3 steps to your first curve (~10 min prep + 9h unattended):

1. **Get a Tinker API key** — register at [Tinker](https://tinker.thinkingmachines.ai/) (no GPU needed to start)
2. **Prepare the config** — copy [`examples/copaw_rl/queries_simple.train.tinker.v22.yaml`](./examples/copaw_rl/queries_simple.train.tinker.v22.yaml), fill in your API key and model path
3. **Start training** — `trinity run --config queries_simple.train.tinker.v22.yaml`, leave it overnight and watch `rollout/score/mean` go from **72.4 → 82.4 (+10pp)**

> Full steps (env install, data prep, plotting script) in [Ch.1: 5-minute run (zh)](./docs/RL_tutorial/ch1_5分钟跑通.md).

---

## Appendix A: Tools in the stack

> Brief intros to the tools used by the tutorial; full docs are at each project's homepage. The tutorial's core is agentic-RL methodology, and all these tools are replaceable.

**Trinity-RFT** (training framework): a general-purpose, flexible framework for LLM reinforcement fine-tuning (RFT), decoupling RFT into three coordinating modules — Explorer / Trainer / Buffer. This repo's code is based on it. Full docs at the [upstream repo](https://github.com/agentscope-ai/Trinity-RFT) and [official docs](https://agentscope-ai.github.io/Trinity-RFT/).

**Tinker / TuFT** (training backend): run the actual forward/backward/weight updates — Trinity-RFT does not do these itself, but forwards them to a backend via the [Tinker SDK](https://tinker.thinkingmachines.ai/). [Tinker](https://tinker.thinkingmachines.ai/) is a cloud service (0-GPU onboarding); [TuFT](https://github.com/modelscope/TuFT) is a self-hosted option (needs a GPU cluster). Switching only requires changing one `base_url` line in the yaml — the algorithm-layer yaml stays untouched.

**E2B** (sandbox): provides a container with a real shell + filesystem, where the model multi-turn invokes `execute_shell_command` / `read_file` / `write_file` / `edit_file` / `grep_search` and other tools to complete tasks. Grading depends on real runtime results inside the container (file contents, service reachability, sqlite tables, etc.).

<details><summary>Trinity-RFT supported RFT modes, algorithms and install (summary; full list upstream)</summary>

- **RFT modes**: sync/async, on-policy/off-policy, online/offline; rollout and training separable and independently scalable; experience replay supported.
- **Algorithms**: PPO, GRPO, SFT, DPO, CHORD, REC series, RLOO, REINFORCE++, GSPO, TOPR, sPPO, AsymRE, CISPO, SAPO, On-Policy Distillation, JSD, etc. Full list and configs at the [upstream algorithm module](https://github.com/agentscope-ai/Trinity-RFT/tree/main/trinity/algorithm/algorithm.py).
- **Training backends**: [Tinker](https://tinker.thinkingmachines.ai/) cloud (no-GPU onboarding) / [TuFT](https://github.com/modelscope/TuFT) (self-hosted).
- **Install**: `pip install -e ".[vllm,flash_attn]"` (with GPU) or `pip install -e ".[tinker]"` (no GPU). See upstream [Quick Start](https://github.com/agentscope-ai/Trinity-RFT#quick-start).

</details>

## Appendix B: Acknowledgements & Citation

This tutorial is built on [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT); thanks to the upstream team for open-sourcing it. Trinity-RFT further builds on verl, vLLM, FSDP, Megatron-LM, Data-Juicer, AgentScope, Ray and other excellent open-source projects (full acknowledgements in the upstream README).

If this repo helps your research, please cite the Trinity-RFT technical report:

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

## Contributing

* **Tutorial & accompanying code** (`docs/RL_tutorial/`, `examples/copaw_rl/`, READMEs): issues / PRs are welcome in this repo, including tutorial errata, config improvements and new ablation experiments.
* **Trinity-RFT framework itself** (`trinity/` core code, algorithms, buffer, etc.): please contribute to the [upstream repo](https://github.com/agentscope-ai/Trinity-RFT); see upstream [CONTRIBUTING.md](https://github.com/agentscope-ai/Trinity-RFT/blob/main/CONTRIBUTING.md).
