#!/usr/bin/env python3
"""第 3 章配套脚本：展示 reward 评分全流程。

用法：
  # 方式 1：用预置的 simple_085 样本
  python scripts/tutorial/ch3_score_trajectory.py --sample

  # 方式 2：指定一个 result_simple.json
  python scripts/tutorial/ch3_score_trajectory.py --result path/to/result_simple.json

本脚本展示从 raw checks → task score → trajectory reward 的完整流程，
让你看到 reward 是怎么一层一层算出来的。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data" / "simple_085"

# simple_085 的 check 定义（与 test_case_simple_085.py 一致）
CHECKS_DEFINITION = [
    {
        "name": "anchor_ok",
        "description": "VERIFICATION_ANCHOR == 'CPW-SIMPLE-085'",
        "what_it_tests": "agent 是否正确写入了 anchor 标记文件",
    },
    {
        "name": "port_listen",
        "description": "127.0.0.1:18085 端口可达",
        "what_it_tests": "asyncio TCP server 是否在正确端口监听",
    },
    {
        "name": "output_has_ping",
        "description": "/opt/cpw_simple_085/output.txt 包含 'ping CPW'",
        "what_it_tests": "client 发送的消息是否被 server echo 并写入文件",
    },
]


def print_reward_layers(result: dict, label: str = ""):
    """分层展示 reward 计算过程。"""
    task_id = result.get("task_id", "unknown")
    score = result.get("score", 0)
    passed = result.get("passed", False)
    checks = result.get("checks", [])
    agent_call_seconds = result.get("agent_call_seconds", 0)
    over_length = result.get("over_length", False)
    # v22 实验 MAX_TRAJ_SECONDS=1200s，默认 300s 只是代码里的 fallback
    max_traj_seconds = 1200
    over_time = agent_call_seconds > max_traj_seconds

    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  Reward Calculation for {task_id}")
    print(f"{'='*60}\n")

    # Layer 1: Verifier Checks
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Layer 1: Verifier Checks                                │")
    print("  │ (回放 sandbox 内 verifier 产出的结果，sandbox 已销毁)      │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print(f"    simple_085 的 3 个 check：")
    print(f"      1. anchor_ok     — anchor 文件存在且内容 == 'CPW-SIMPLE-085'")
    print(f"      2. port_listen   — 127.0.0.1:18085 端口可达")
    print(f"      3. output_has_ping — /opt/cpw_simple_085/output.txt 含 'ping CPW'")
    print()

    passed_count = 0
    total_count = len(checks) if checks else len(CHECKS_DEFINITION)

    if checks:
        for c in checks:
            mark = "✓" if c.get("passed") else "✗"
            status = "PASSED" if c.get("passed") else "FAILED"
            if c.get("passed"):
                passed_count += 1
            detail = c.get("detail", "")
            print(f"    [{mark}] {c.get('name', '?')}: {status}")
            if detail:
                print(f"        detail: {detail}")
    else:
        # result_simple.json 不含 per-check 详情，根据 score 和 check 定义推断
        # （实际打分发生在 sandbox 内，sandbox 已销毁，无法重跑）
        print(f"    根据记录的 score={score:.4f} 推断 check 通过情况：")
        print()
        for i, cd in enumerate(CHECKS_DEFINITION):
            check_passed = i < round(score * len(CHECKS_DEFINITION))
            mark = "✓" if check_passed else "✗"
            status = "PASSED" if check_passed else "FAILED"
            if check_passed:
                passed_count += 1
            print(f"    [{mark}] {cd['name']}: {status}")
            print(f"        测试内容: {cd['what_it_tests']}")

    print()

    # Layer 2: Task Score
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Layer 2: Task Score                                     │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print(f"    score = passed_checks / total_checks")
    print(f"         = {passed_count} / {total_count}")
    print(f"         = {score:.4f}")
    print()
    print(f"    可能取值: {{0/3, 1/3, 2/3, 3/3}} = {{0.000, 0.333, 0.667, 1.000}}")
    print()

    # Layer 3: Trajectory Reward
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ Layer 3: Trajectory Reward (截断惩罚)                   │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()
    print(f"    over_length (token 超限): {over_length}")
    print(f"    over_time (耗时 > {max_traj_seconds}s): {over_time} (实际: {agent_call_seconds:.1f}s)")
    print()

    if over_length or over_time:
        final_reward = 0.0
        reason = "over_length" if over_length else "over_time"
        print(f"    ⚠️  {reason} = True → reward 强制置 0")
        print(f"    Final trajectory reward = 0.0")
    else:
        final_reward = score
        print(f"    无截断 → reward = task_score")
        print(f"    Final trajectory reward = {final_reward:.4f}")

    print()
    return final_reward


def main():
    parser = argparse.ArgumentParser(
        description="第 3 章：展示 reward 评分全流程"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="使用预置的 simple_085 样本数据"
    )
    parser.add_argument(
        "--result", type=str, default="",
        help="指定 result_simple.json 路径"
    )
    args = parser.parse_args()

    if args.sample:
        # 展示 step1（失败）和 step19（成功）两条
        r1 = json.loads((SAMPLE_DIR / "step1_result.json").read_text())
        r19 = json.loads((SAMPLE_DIR / "step19_result.json").read_text())

        print("\n" + "█" * 60)
        print("  第 3 章：reward 怎么算的")
        print("  用 simple_085 的真实数据展示 4 层 reward 计算")
        print("█" * 60)

        reward1 = print_reward_layers(r1, label="CASE A: Base policy (Step 1) — 失败案例")
        reward19 = print_reward_layers(r19, label="CASE B: Trained policy (Step 19) — 成功案例")

        # 对比总结
        print("=" * 60)
        print("  对比总结")
        print("=" * 60)
        print()
        print(f"    Base policy  reward = {reward1:.4f}  (1/3 checks)")
        print(f"    Step 19      reward = {reward19:.4f}  (3/3 checks)")
        print(f"    Δ reward = +{reward19 - reward1:.4f}")
        print()
        print("  关键 insight:")
        print("    • 多 check 设计让 reward 是连续的 (0.33 而非 0)")
        print("    • 即使 base policy 从未拿满分，也能区分「过 1 个 check」和「过 0 个」")
        print("    • 这个梯度信号让 GRPO 有东西可学（见第 4 章）")
        print()
        print("  下一步: 这些 reward 怎么变成 advantage？")
        print("    python scripts/tutorial/ch4_compute_advantage.py --sample")
        print()

    elif args.result:
        result_path = Path(args.result)
        if not result_path.exists():
            print(f"错误: 文件不存在: {result_path}")
            return 1
        r = json.loads(result_path.read_text())
        print_reward_layers(r)
    else:
        parser.print_help()
        print("\n提示: 使用 --sample 查看预置数据，或用 --result 指定你的文件")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
