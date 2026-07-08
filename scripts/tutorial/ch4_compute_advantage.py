#!/usr/bin/env python3
"""第 4 章配套脚本：GRPO advantage 计算演示。

用法：
  # 方式 1：用预置的 simple_085 样本（8 条 trajectory 的 rewards）
  python scripts/tutorial/ch4_compute_advantage.py --sample

  # 方式 2：自定义 rewards 列表
  python scripts/tutorial/ch4_compute_advantage.py --rewards 1.0,1.0,0.67,0.67,0.33,0.33,0.33,0.33

  # 方式 3：试试全相同 reward 会怎样
  python scripts/tutorial/ch4_compute_advantage.py --rewards 0.67,0.67,0.67,0.67,0.67,0.67,0.67,0.67

本脚本完整展示 GRPO advantage 的计算过程，让你直观理解
"组内相对"是什么意思、为什么不需要 critic 网络。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data" / "simple_085"

# Trinity 默认 epsilon
DEFAULT_EPSILON = 0.1
# 最小有效 std 阈值
STD_THRESHOLD = 0.05


def compute_grpo_advantage(
    rewards: list[float],
    epsilon: float = DEFAULT_EPSILON,
    verbose: bool = True,
) -> list[float]:
    """计算 GRPO advantage 并打印详细过程。"""
    G = len(rewards)
    mean = sum(rewards) / G
    variance = sum((r - mean) ** 2 for r in rewards) / G
    std = variance ** 0.5

    if verbose:
        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │ Step 1: Group Statistics                                │")
        print(f"  └─────────────────────────────────────────────────────────┘")
        print()
        print(f"    G (group size) = {G}")
        print(f"    rewards = {[f'{r:.4f}' for r in rewards]}")
        print(f"    mean(rewards) = {mean:.4f}")
        print(f"    std(rewards)  = {std:.4f}")
        print(f"    epsilon       = {epsilon}")
        print()

    # Compute advantages
    advantages = [(r - mean) / (std + epsilon) for r in rewards]

    if verbose:
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │ Step 2: Compute Advantages                             │")
        print(f"  └─────────────────────────────────────────────────────────┘")
        print()
        print(f"    advantage_i = (reward_i - mean) / (std + epsilon)")
        print(f"                = (reward_i - {mean:.4f}) / ({std:.4f} + {epsilon})")
        print(f"                = (reward_i - {mean:.4f}) / {std + epsilon:.4f}")
        print()

        for i, (r, a) in enumerate(zip(rewards, advantages)):
            if a > 0.5:
                label = "← strong ENCOURAGE"
            elif a > 0:
                label = "← mild ENCOURAGE"
            elif a > -0.5:
                label = "← mild SUPPRESS"
            else:
                label = "← strong SUPPRESS"
            print(f"    Trajectory {i+1}: reward={r:.4f} → adv = {a:+.4f}  {label}")

        print()

    return advantages


def health_check(rewards: list[float], advantages: list[float], verbose: bool = True):
    """检查这组 reward 是否有有效的训练信号。"""
    G = len(rewards)
    mean = sum(rewards) / G
    variance = sum((r - mean) ** 2 for r in rewards) / G
    std = variance ** 0.5

    std_ok = std > STD_THRESHOLD
    has_positive = any(a > 0 for a in advantages)
    has_negative = any(a < 0 for a in advantages)
    valid = std_ok and has_positive and has_negative

    if verbose:
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │ Step 3: Health Check                                   │")
        print(f"  └─────────────────────────────────────────────────────────┘")
        print()
        print(f"    reward_std = {std:.4f} {'> ' if std_ok else '< '}{STD_THRESHOLD} "
              f"{'✓' if std_ok else '✗ (group内无差异，无法学习！)'}")
        print(f"    has positive advantage: {'✓' if has_positive else '✗'}")
        print(f"    has negative advantage: {'✓' if has_negative else '✗'}")
        print(f"    valid_ratio contribution: {'1.0 ✓' if valid else '0.0 ✗ (这组不产生梯度)'}")
        print()

        if not std_ok:
            print("    ⚠️  所有 trajectory 得分相同 → 无法区分好坏 → 这一组白跑")
            print("       解决: 增大 G（更多采样）或检查任务难度是否合适")
            print()
        elif valid:
            print("    ✓ 健康：组内有差异，RL 能学到「什么行为相对更好」")
            print()

    return valid


def main():
    parser = argparse.ArgumentParser(
        description="第 4 章：GRPO advantage 计算演示"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="使用预置的 simple_085 样本数据（8 条 trajectory 的 rewards）"
    )
    parser.add_argument(
        "--rewards", type=str, default="",
        help="自定义 rewards（逗号分隔），例如: 1.0,1.0,0.67,0.67,0.33,0.33,0.33,0.33"
    )
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help=f"GRPO epsilon（防除零，默认: {DEFAULT_EPSILON}）"
    )
    args = parser.parse_args()

    if args.sample:
        data = json.loads((SAMPLE_DIR / "group_rewards.json").read_text())
        rewards = data["rewards"]
        task_id = data["task_id"]
        step = data["step"]

        print("\n" + "█" * 60)
        print("  第 4 章：GRPO Advantage 计算")
        print("  用 simple_085 step 1 的真实 G=8 组数据演示")
        print("█" * 60)
        print()
        print(f"  Task: {task_id} (asyncio TCP echo server)")
        print(f"  Step: {step}")
        print(f"  G = {len(rewards)} trajectories from same prompt")

        advantages = compute_grpo_advantage(rewards, epsilon=args.epsilon)
        health_check(rewards, advantages)

        # 额外：展示如果用 0/1 reward 会怎样
        print("  " + "─" * 56)
        print("  对比实验：如果 reward 退化为 0/1 (只看是否满分)？")
        print("  " + "─" * 56)
        binary_rewards = [1.0 if r >= 0.99 else 0.0 for r in rewards]
        print(f"\n    binary rewards = {binary_rewards}")
        binary_adv = compute_grpo_advantage(binary_rewards, epsilon=args.epsilon)
        health_check(binary_rewards, binary_adv)

        print("  结论:")
        print("    • 多 check reward: 每条 trajectory 都有区分度，梯度信号丰富")
        print("    • 0/1 reward: 只有 2 条满分 vs 6 条非满分，信号粗糙")
        print("    • 这就是第 3 章强调「reward 颗粒度决定能不能学」的数值验证")
        print()

    elif args.rewards:
        try:
            rewards = [float(x.strip()) for x in args.rewards.split(",")]
        except ValueError:
            print("错误: --rewards 格式不正确，应为逗号分隔的数字")
            return 1

        if len(rewards) < 2:
            print("错误: 至少需要 2 条 trajectory 的 reward")
            return 1

        print(f"\n{'='*60}")
        print(f"  GRPO Advantage (自定义 rewards)")
        print(f"{'='*60}")
        print(f"\n  Input: G={len(rewards)} trajectories")

        advantages = compute_grpo_advantage(rewards, epsilon=args.epsilon)
        health_check(rewards, advantages)

    else:
        parser.print_help()
        print("\n示例:")
        print("  # 预置数据")
        print("  python scripts/tutorial/ch4_compute_advantage.py --sample")
        print()
        print("  # 自定义：所有 reward 相同（无信号）")
        print("  python scripts/tutorial/ch4_compute_advantage.py --rewards 0.5,0.5,0.5,0.5")
        print()
        print("  # 自定义：极端分化")
        print("  python scripts/tutorial/ch4_compute_advantage.py --rewards 1.0,1.0,0.0,0.0")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
