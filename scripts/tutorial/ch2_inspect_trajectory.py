#!/usr/bin/env python3
"""第 2 章配套脚本：查看一条 trajectory 内部的 ReAct 循环。

用法：
  # 方式 1：用预置的 simple_085 样本（无需 API key）
  python scripts/tutorial/ch2_inspect_trajectory.py --sample

  # 方式 2：同时对比 step1（失败）和 step19（成功）
  python scripts/tutorial/ch2_inspect_trajectory.py --sample --compare

  # 方式 3：用你自己的 session.json
  python scripts/tutorial/ch2_inspect_trajectory.py --session path/to/session.json

本脚本解析 session.json 中的 agent.memory.content，pretty-print 整条
trajectory 的 ReAct 循环，让你看到模型每一步在干什么。
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample_data" / "simple_085"


def _content_to_text(content) -> str:
    """将 content（str 或 list[dict]）转为可读纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    output = block.get("output", [])
                    if isinstance(output, list):
                        for o in output:
                            if isinstance(o, dict) and o.get("type") == "text":
                                parts.append(o.get("text", ""))
                    elif isinstance(output, str):
                        parts.append(output)
        return "\n".join(parts)
    return str(content)


def _extract_tool_name(content) -> str:
    """从 tool_result content 中提取工具名称。"""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return block.get("name", "")
    return ""


def _find_tool_result(all_content: list, call_id: str) -> str | None:
    """在所有 message 中找到与指定 call_id 匹配的 tool_result 文本。"""
    if not call_id:
        return None
    for msg in all_content:
        role, msg_content = get_role_and_content(msg)
        if role in ("tool", "system") and isinstance(msg_content, list):
            for block in msg_content:
                if (isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and block.get("id") == call_id):
                    output = block.get("output", [])
                    if isinstance(output, list):
                        parts = []
                        for o in output:
                            if isinstance(o, dict) and o.get("type") == "text":
                                parts.append(o.get("text", ""))
                        return "\n".join(parts)
                    elif isinstance(output, str):
                        return output
    return None


def truncate(text: str, max_len: int = 200) -> str:
    """截断长文本用于展示。"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... ({len(text)} chars total)"


def _format_arg_value(value: str, max_len: int = 60) -> str:
    """格式化 tool call 参数值：转义换行 + 截断 + 标注总长度。"""
    # 替换换行为可见的 \n，保持单行显示
    escaped = value.replace('\n', '\\n').replace('\r', '\\r')
    if len(value) <= max_len:
        return escaped
    # 截断显示，标注原始长度（字节数/行数）
    lines = value.count('\n') + 1
    truncated = escaped[:max_len]
    return f"{truncated}... ({len(value)} chars, {lines} lines)"


def _format_tool_result(text: str) -> str:
    """格式化 tool result 输出：分离 stdout/stderr，只显示关键信息，保持单行。"""
    if not text:
        return "(empty)"

    # 分离 stdout 和 stderr
    if "[stderr]" in text:
        parts = text.split("[stderr]", 1)
        stdout = parts[0].strip()
        stderr = parts[1].strip() if len(parts) > 1 else ""
    else:
        stdout = text.strip()
        stderr = ""

    # 提取 stderr 中的关键错误行（最后一行通常是核心错误）
    if stderr:
        stderr_lines = [l.strip() for l in stderr.split('\n') if l.strip()]
        # 找关键错误行：Error/Exception 结尾的行
        error_line = ""
        for line in reversed(stderr_lines):
            if "Error" in line or "Exception" in line:
                error_line = line
                break
        if not error_line and stderr_lines:
            error_line = stderr_lines[-1]

        stdout_short = stdout.replace('\n', ' ').strip()
        if stdout_short:
            return f"{truncate(stdout_short, 40)} [stderr: {truncate(error_line, 80)}]"
        else:
            return f"[stderr: {truncate(error_line, 100)}]"

    # 无 stderr：把换行替换为空格，保持单行显示
    one_line = stdout.replace('\n', ' ').replace('\r', ' ').strip()
    return truncate(one_line, 120)


def extract_tool_calls(content) -> list[dict]:
    """从 assistant message content 中提取 tool_call 信息。"""
    calls = []
    if isinstance(content, str):
        # 有些格式用 <tool_call> 标签
        import re
        for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL):
            try:
                calls.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                calls.append({"raw": m.group(1)[:100]})
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    calls.append({
                        "name": block.get("name", "?"),
                        "input": block.get("input", {}),
                    })
    return calls


def extract_thinking(content) -> str:
    """从 content 中提取 <think>...</think> 部分。"""
    import re
    if isinstance(content, str):
        m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if m:
            return m.group(1).strip()
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                return block.get("text", "").strip()
    return ""


def get_role_and_content(msg) -> tuple[str, object]:
    """兼容多种 message 格式，返回 (role, content)。

    content 可能是 str 或 list[dict]（content blocks）。
    """
    if isinstance(msg, dict):
        return msg.get("role", "?"), msg.get("content", "")
    elif isinstance(msg, list):
        # AgentScope 格式：list of sub-messages，取第一个有效 dict
        if msg and isinstance(msg[0], dict):
            role = msg[0].get("role", "?")
            content = msg[0].get("content", "")
            return role, content
    return "?", ""


def print_trajectory(session_data: dict, label: str = ""):
    """Pretty-print 一条 trajectory 的 ReAct 循环。"""
    agent = session_data.get("agent", {})
    memory = agent.get("memory", {})
    content = memory.get("content", [])
    trajectory = agent.get("_model_trajectory", [])

    # 从 result 获取 score（如果有的话）
    task_id = agent.get("name", "unknown")

    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"  Task: {task_id}")
    print(f"  Messages: {len(content)}")
    print(f"  LLM calls: {len(trajectory)}")
    if trajectory:
        total_prompt = sum(s.get("_prompt_token_ids_length", 0) for s in trajectory)
        total_resp = sum(s.get("_token_ids_length", 0) for s in trajectory)
        print(f"  Total tokens: prompt\u2248{total_prompt:,}, response\u2248{total_resp:,}")
    print(f"{'='*60}\n")

    step_idx = 0
    for i, msg in enumerate(content):
        role, msg_content = get_role_and_content(msg)

        if role == "user" and i == 0:
            # Task prompt（太长，只显示开头）
            text = _content_to_text(msg_content)
            print(f"  [System] Task prompt ({len(text)} chars)")
            first_line = text.split('\n')[0][:80] if text else ""
            print(f"           {first_line}...")
            print()
            continue

        if role == "assistant":
            step_idx += 1

            # content 可能是 list[dict] (content blocks) 或 str
            if isinstance(msg_content, list):
                # 结构化 content blocks
                thinking_text = ""
                tool_calls_found = []
                plain_text = ""

                for block in msg_content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        txt = block.get("text", "")
                        # 提取 thinking 部分
                        th = extract_thinking(txt)
                        if th:
                            thinking_text = th
                        else:
                            plain_text += txt
                    elif btype == "tool_use":
                        tool_calls_found.append({
                            "name": block.get("name", "?"),
                            "input": block.get("input", {}),
                            "id": block.get("id", ""),
                        })

                # 打印 thinking
                if thinking_text:
                    print(f"  [Step {step_idx}] Assistant (thinking):")
                    for line in textwrap.wrap(truncate(thinking_text, 150), width=70):
                        print(f"           {line}")
                elif plain_text and not tool_calls_found:
                    print(f"  [Step {step_idx}] Assistant (text only, no tool call):")
                    print(f"           {truncate(plain_text, 150)}")
                elif not thinking_text:
                    print(f"  [Step {step_idx}] Assistant:")

                # 打印 tool calls + 配对的 results
                if tool_calls_found:
                    for tc in tool_calls_found:
                        name = tc["name"]
                        call_id = tc.get("id", "")
                        args = tc.get("input", {})
                        if isinstance(args, dict):
                            arg_str = ", ".join(
                                f'{k}="{_format_arg_value(str(v), 60)}"'
                                for k, v in list(args.items())[:3]
                            )
                        else:
                            arg_str = _format_arg_value(str(args), 100)
                        print(f"           Tool call: {name}({arg_str})")
                        # 配对显示对应的 result
                        result_text = _find_tool_result(content, call_id)
                        if result_text is not None:
                            formatted = _format_tool_result(result_text)
                            if "error" in result_text.lower() or "Error" in result_text or "stderr" in result_text:
                                print(f"             \u2514\u2192 \u26a0\ufe0f  {formatted}")
                            else:
                                print(f"             \u2514\u2192 {formatted}")
            else:
                # 纯字符串 content
                text = str(msg_content)
                thinking = extract_thinking(text)
                if thinking:
                    print(f"  [Step {step_idx}] Assistant (thinking):")
                    for line in textwrap.wrap(truncate(thinking, 150), width=70):
                        print(f"           {line}")

                tool_calls = extract_tool_calls(text)
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get("name", "?")
                        args = tc.get("input", tc.get("arguments", {}))
                        if isinstance(args, dict):
                            arg_str = ", ".join(
                                f'{k}="{truncate(str(v), 60)}"'
                                for k, v in list(args.items())[:3]
                            )
                        else:
                            arg_str = truncate(str(args), 100)
                        print(f"           Tool call: {name}({arg_str})")
                elif not thinking:
                    print(f"  [Step {step_idx}] Assistant (text only, no tool call):")
                    print(f"           {truncate(text, 150)}")
            print()

        elif role in ("tool", "system") and i > 0:
            # Tool results are shown inline with tool calls above;
            # skip them here to avoid duplication.
            pass

    print(f"  {'\u2500'*56}")
    print(f"  Summary: {step_idx} assistant turns, {len(content)} total messages")
    print()


def print_result(result_path: Path):
    """打印 result_simple.json 的评分结果。"""
    if not result_path.exists():
        return
    r = json.loads(result_path.read_text())
    score = r.get("score", 0)
    passed = r.get("passed", False)
    llm_calls = r.get("agent_llm_calls", "?")
    call_sec = r.get("agent_call_seconds", "?")
    checks = r.get("checks", [])

    print(f"  Score: {score:.4f} ({'PASSED' if passed else 'FAILED'})")
    print(f"  LLM calls: {llm_calls}, total time: {call_sec}s")
    if checks:
        print(f"  Checks:")
        for c in checks:
            mark = "✓" if c.get("passed") else "✗"
            print(f"    [{mark}] {c.get('name', '?')}: {c.get('detail', '')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="第 2 章：查看 trajectory 内部的 ReAct 循环"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="使用预置的 simple_085 样本数据"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="对比 step1（失败）和 step19（成功）两条 trajectory"
    )
    parser.add_argument(
        "--session", type=str, default="",
        help="指定 session.json 路径"
    )
    parser.add_argument(
        "--result", type=str, default="",
        help="指定 result_simple.json 路径（可选，用于显示评分）"
    )
    args = parser.parse_args()

    if args.sample:
        if args.compare:
            # 对比两条
            print("\n" + "█" * 60)
            print("  对比：base policy (step 1) vs trained policy (step 19)")
            print("  任务：simple_085 — asyncio TCP echo server")
            print("█" * 60)

            # Step 1 (失败)
            s1 = json.loads((SAMPLE_DIR / "step1_session.json").read_text())
            print_trajectory(s1, label="BASE POLICY (Step 1) — score=0.33, 1/3 checks")
            print_result(SAMPLE_DIR / "step1_result.json")

            # Step 19 (成功)
            s19 = json.loads((SAMPLE_DIR / "step19_session.json").read_text())
            print_trajectory(s19, label="TRAINED POLICY (Step 19) — score=1.00, 3/3 checks")
            print_result(SAMPLE_DIR / "step19_result.json")

            # 对比总结
            print("=" * 60)
            print("  关键行为变化:")
            print("  ┌─────────────────┬──────────────────────────┬──────────────────────────┐")
            print("  │ 行为            │ Base policy              │ Step 19 policy          │")
            print("  ├─────────────────┼──────────────────────────┼──────────────────────────┤")
            print("  │ async 写法      │ SyntaxError/属性错误    │ 正确 async def 写法    │")
            print("  │ 连接前检查    │ 不检查，直接放弃      │ ps -ef|grep LISTEN 确认 │")
            print("  │ 错误处理        │ 忽略 stderr，放弃      │ kill 冲突进程，重试    │")
            print("  │ 任务完成        │ 未完成（text only）  │ grep 验证 output 正确  │")
            print("  └─────────────────┴──────────────────────────┴──────────────────────────┘")
            print()
        else:
            # 只看 step1
            s1 = json.loads((SAMPLE_DIR / "step1_session.json").read_text())
            print_trajectory(s1, label="BASE POLICY (Step 1) — simple_085 失败案例")
            print_result(SAMPLE_DIR / "step1_result.json")

    elif args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            print(f"错误: 文件不存在: {session_path}")
            return 1
        data = json.loads(session_path.read_text())
        print_trajectory(data, label=f"从 {session_path.name}")
        if args.result:
            print_result(Path(args.result))
        else:
            # Try to find result_simple.json in same directory
            result_path = session_path.parent / "result_simple.json"
            if result_path.exists():
                print_result(result_path)
    else:
        parser.print_help()
        print("\n提示: 使用 --sample 查看预置数据，或用 --session 指定你自己的文件")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
