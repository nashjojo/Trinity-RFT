#!/usr/bin/env python3
"""最小 Agentic-RL 训练循环骨架 — 把 ch1~ch5 的概念串成一条线。

这不是一个可执行的训练脚本，而是一个注释完整的概念地图：
用 < 50 行伪代码展示 Trinity-RFT 一个 training step 里到底发生了什么。

对应关系：
  Step 1 (rollout)   → 第 2 章：单次 rollout 内部
  Step 2 (reward)    → 第 3 章：reward 怎么算
  Step 3 (advantage) → 第 4 章：GRPO advantage
  Step 4 (update)    → 第 5 章：权重更新

用法：直接阅读本文件即可，不需要运行。
"""

# ─── 超参（第 1 章 yaml 里的配置）───────────────────────────
G = 8           # 每个 prompt 采样 8 条 trajectory
num_prompts = 8 # 每步取 8 个不同任务
total_steps = 19
lr = 5e-6
clip_eps = 0.2
eps = 0.1       # GRPO 防除零


def one_training_step(prompts, model):  # noqa — 伪代码，仅供阅读
    """一个训练步的完整流程（伪代码）。"""

    # === Step 1: Rollout — 第 2 章 ===
    # 8 个 prompt × G=8 = 64 条 trajectory，每条在 E2B sandbox 中多轮 ReAct
    for prompt in prompts:
        trajectories = [rollout(prompt, model, sandbox="e2b") for _ in range(G)]
        # 每条 trajectory ≈ 10 次 tool call（mkdir → write_file → run → verify...）

    # === Step 2: Reward — 第 3 章 ===
    # Verifier 在 sandbox 内跑 3 个 check → score = passed_checks / total_checks
    rewards = [verifier.run_checks(traj) for traj in trajectories]
    # 例：[0.33, 0.33, 0.67, 0.67, 0.67, 0.67, 1.0, 1.0]
    # over_length / over_time 的 trajectory → reward 强制置 0

    # === Step 3: GRPO Advantage — 第 4 章 ===
    # 组内相对归一化：不需要 critic 网络，用 group mean 当 baseline
    mean = sum(rewards) / G
    std = stdev(rewards)
    advantages = [(r - mean) / (std + eps) for r in rewards]
    # advantage > 0 → 鼓励；< 0 → 抑制；全相同 → 不学

    # === Step 4: Weight Update — 第 5 章 ===
    # PPO surrogate loss + clip → 只更新 LoRA（15M params / ~31 MB）
    ratio = exp(new_logprobs - old_logprobs)
    loss = -min(ratio * adv, clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    lora_params -= adam_update(grad(loss), lr=lr)
    # base 4B 参数永远冻结，只传 ~31 MB LoRA 增量回 explorer


# 循环 19 步 → rollout/score/mean 从 72.4 → 82.4 (+10 pp)
