# 贡献指南

感谢你对本教程仓库的关注！本仓库是 [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) 的修改版，核心内容是 [`docs/RL_tutorial/`](./docs/RL_tutorial/) 下的 Agentic-RL 入门教程与 [`examples/copaw_rl/`](./examples/copaw_rl/) 下的配套代码。

## 在哪里贡献

请根据你要贡献的内容选择对应的仓库。

### 1. 教程与配套代码 → 本仓库

欢迎在本仓库提 issue / PR，包括：

| 关注点 | 目录 | 可做的事 |
|---|---|---|
| **教程内容** | `docs/RL_tutorial/` | 勘误、补充解释、新增章节、改进示例 |
| **配套配置与数据** | `examples/copaw_rl/` | 改进 baseline yaml、补充数据集、新增实验配置 |
| **实验脚本** | `examples/copaw_rl/entry/` | 批量跑 ablation、出图、评估脚本优化 |
| **文档与报告** | `docs/2026-*.md`、README | 实验记录、图表、主页改进 |

### 2. Trinity-RFT 框架本体 → 上游仓库

`trinity/` 核心代码（算法、buffer、explorer、trainer 等）来自上游 Trinity-RFT。涉及框架本身的改动（新算法、新工作流、bug 修复、性能优化等）请贡献到上游：

- 上游仓库：https://github.com/agentscope-ai/Trinity-RFT
- 上游贡献指南：[upstream CONTRIBUTING.md](https://github.com/agentscope-ai/Trinity-RFT/blob/main/CONTRIBUTING.md)
- 上游开发者文档：[Developer Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_overview.html)

## 提交前检查

1. **代码风格**：本仓库使用 `pre-commit` 维护代码质量，提交前运行：
   ```bash
   pre-commit run --all-files
   ```
2. **测试**：如改动涉及 `trinity/` 或脚本代码，请运行：
   ```bash
   python -m pytest tests/
   ```
3. **PR 说明**：写清**动机**（为什么改）和**实现**（怎么做的）。

## 反馈

- **教程 / 配套代码问题**：请在本仓库提 issue。
- **Trinity-RFT 框架问题**：请到 [上游 issues](https://github.com/agentscope-ai/Trinity-RFT/issues)。
- **重大改动**：请先开 issue 讨论设计，再动手实现。

感谢帮助改进这份教程！
# Contributing to Trinity-RFT

Thank you for your interest in Trinity-RFT! Our framework is built on a decoupled architecture consisting of the **Explorer**, **Trainer**, and **Buffer**. We welcome all forms of contributions—from core feature enhancements and new algorithms to documentation and bug reports.

## Where to Contribute

Trinity-RFT provides modular interfaces for different technical interests. Please refer to our [Developer Guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/develop_overview.html) for detailed implementation standards:

| Focus Area | Interface/Code Directory | Potential Tasks |
| :--- | :--- | :--- |
| **Agentic Workflows** | `Workflow` | Implementing multi-turn dialogs, ReAct workflows, or domain-specific agent training capabilities (e.g., Coding, Math). |
| **RL Algorithms** | `Algorithm` | Integrating new RL algorithms (e.g., RLOO, GSPO) or optimizing loss functions and advantage estimations. |
| **Data & Experience** | `Operator`, `Selector` | Designing data cleaning, selection, reward modeling, or experience augmentation and replay strategies. |
| **Use Cases** | `Examples` | Sharing new usages and improving built-in demonstrated configurations. |
| **General Utility** | `docs/`, `tests/` | Improving documentation, adding translations, fixing bugs, or enhancing CLI/GUI tools. |

## How to Start: The "Plugin-First" Approach

To minimize friction and keep the core codebase stable, we recommend the **Plugin-First** workflow for new features:

1. **Develop**: Create your custom module in the `trinity/plugins/` directory.
2. **Auto-Load**: Trinity-RFT automatically detects and registers modules in this directory at runtime without requiring changes to the framework's internal code.
3. **Integrate**: Once your feature is stable and verified, submit a Pull Request to "graduate" your code from `plugins/` into the formal modules (e.g., `trinity/algorithm/` or `trinity/common/workflows/`).

## Submission Checklist

To ensure a smooth review process, please complete the following:

1. **Registration**: If moving code from a plugin to the core framework, register it in the corresponding `__init__.py` mapping.
2. **Testing**: Add or update unit tests in the `tests/` directory. Verify your changes by running:
   ```bash
   python -m pytest tests/
   ```
3. **Code Style**: We use `pre-commit` to maintain code quality. Run the following before committing:
   ```bash
   pre-commit run --all-files
   ```
4. **Description**: Provide a clear PR title and a description that explains the **motivation** (why this change is needed) and the **implementation** (how it works).

---

## Additional Guidelines

- **Bug Reports & Feature Requests**: Please use [GitHub Issues](https://github.com/agentscope-ai/Trinity-RFT/issues). For bugs, include reproduction steps, environment info, and error logs.
- **Major Changes**: For significant architectural changes or large features, please open an issue first to discuss the design with the maintainers.
- **Documentation**: We highly value improvements to our tutorials, docstrings, and translations.

*For a deep dive into the framework's architecture, please refer to the [Full Doc](https://agentscope-ai.github.io/Trinity-RFT/en/main/index.html).*

**Thank you for helping us build a better Reinforcement Fine-Tuning framework!**
