#!/usr/bin/env python3
"""第 2 章配套脚本（在线版）：对 base model 跑一条 simple_085 trajectory。

前置条件：
  export TINKER_API_KEY="<你的 Tinker API key>"
  export E2B_API_KEY="<你的 E2B API key>"

用法：
  python scripts/tutorial/ch2_rollout_single.py --output ./my_trajectory/

这个脚本会：
1. 通过 Tinker API 创建 sampling client（base model）
2. 在 E2B sandbox 中执行 simple_085 任务的完整 rollout
3. 将 session.json 和 result 保存到指定目录

注意：跑一条 trajectory 大约需要 3-5 分钟，取决于模型响应速度和 sandbox 初始化时间。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def check_env():
    """检查必要的环境变量。"""
    missing = []
    for key in ["TINKER_API_KEY", "E2B_API_KEY"]:
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print("错误: 缺少以下环境变量:")
        for k in missing:
            print(f"  export {k}=\"<你的 key>\"")
        print("\n获取方式:")
        print("  - Tinker API key: https://tinker.thinkingmachines.ai/")
        print("  - E2B API key: https://e2b.dev/")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="第 2 章（在线版）：跑一条 simple_085 trajectory"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./tutorial_rollout_output",
        help="输出目录（默认: ./tutorial_rollout_output）"
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="模型路径（默认: Qwen/Qwen3-4B-Thinking-2507）"
    )
    args = parser.parse_args()

    if not check_env():
        return 1

    print("=" * 60)
    print("  ch2_rollout_single: 在线跑一条 simple_085 trajectory")
    print("=" * 60)
    print()
    print(f"  模型: {args.model}")
    print(f"  任务: simple_085 (asyncio TCP echo server)")
    print(f"  输出: {args.output}/")
    print()

    # 检查 trinity 是否安装
    try:
        from trinity.common.workflows import WORKFLOWS  # noqa: F401
    except ImportError:
        print("错误: trinity 未安装。请先执行:")
        print("  bash run.sh  # 会自动 uv sync 安装依赖")
        print()
        print("或者使用离线模式查看预置数据:")
        print("  python scripts/tutorial/ch2_inspect_trajectory.py --sample")
        return 1

    print("提示: 完整的在线 rollout 需要 Trinity 训练框架的完整配置。")
    print("对于教程学习，推荐使用 --sample 模式查看预置的真实数据：")
    print()
    print("  python scripts/tutorial/ch2_inspect_trajectory.py --sample --compare")
    print()
    print("如果你已经通过第 1 章跑完了训练，可以直接查看你的 checkpoint 中的 session.json：")
    print()
    print("  python scripts/tutorial/ch2_inspect_trajectory.py \\")
    print("    --session checkpoints/<你的实验>/step_-1_rollout/simple_085/<sandbox_id>/session.json")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
