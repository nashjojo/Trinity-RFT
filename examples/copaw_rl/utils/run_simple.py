#!/usr/bin/env python3
"""run_simple.py — sandbox 内执行的轻量级 runner（queries_simple 数据集专用）。

与 utils/run.py 的差别：
- 不依赖 OSS 上的预定义 task：query / fields 直接由 host 通过 --input-json 注入；
- 不调用 grader：由本目录上传的 test_case_simple_XXX.py 在容器内做确定性校验，
  退出码 0=passed / 1=failed；
- 输出 /root/result_simple.json（含 passed / score / checks / agent_metrics），
  Trinity workflow 拉回后转 reward。

输入 JSON 格式（host 端通过 stdin 或 --input-json-file 注入）::

    {
      "task_id": "simple_001",
      "query": "...预先注入了固定字段的 query 文本...",
      "fields": {"SITE_PORT": "18001", ...},
      "user_id": "default",                     // 可选
      "url": "http://127.0.0.1:8088",          // 可选；qwenpaw HTTP API
      "provider_base_url": "...",
      "provider_model_id": "...",
      "provider_api_key": "...",
      "test_cases_root": "/root/test_cases",   // host 上传后的目录
      "result_file": "/root/result_simple.json"
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

# agent_runner 是 sandbox image 内置模块（与 run.py 同一来源）。
from agent_runner import RL_PROVIDER_NAME, call_agent  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("run_simple")


# ---------------------------------------------------------------------------
# qwenpaw HTTP server 探活 / 拉起
# ---------------------------------------------------------------------------
def _qwenpaw_alive(url: str, timeout: int = 3) -> bool:
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url.rstrip("/") + "/api/version")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= int(r.getcode()) < 500
    except Exception:
        return False


def _ensure_qwenpaw(url: str, max_wait: int = 90) -> bool:
    """qwenpaw HTTP API 必须已就绪；若没起来，尝试用 supervisorctl 或后台启动。"""
    if _qwenpaw_alive(url):
        log.info("qwenpaw 已存活: %s", url)
        return True

    log.warning("qwenpaw 未就绪，尝试拉起 ...")
    # 优先 supervisorctl（template 里通常是 supervisord 管理）；失败则直接 nohup
    try:
        subprocess.run(
            ["supervisorctl", "restart", "app"], check=False, capture_output=True, timeout=20
        )
    except Exception as e:
        log.warning("supervisorctl restart app 失败: %s", e)

    # 兜底：直接后台跑 qwenpaw app（与 sandbox_utils.py 末尾一致）
    if not _qwenpaw_alive(url):
        try:
            log.info("尝试 nohup qwenpaw app ...")
            with open("/app/qwenpaw-app.log", "ab", buffering=0) as fp:
                subprocess.Popen(
                    ["bash", "-lc", "qwenpaw app"],
                    stdout=fp,
                    stderr=fp,
                    start_new_session=True,
                )
        except Exception as e:
            log.warning("nohup qwenpaw app 启动失败: %s", e)

    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _qwenpaw_alive(url):
            log.info("qwenpaw 启动成功: %s", url)
            return True
        time.sleep(2)
    log.error("qwenpaw 在 %ds 内仍未就绪", max_wait)
    return False


# ---------------------------------------------------------------------------
# 输入参数 / 配置加载
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="queries_simple 沙箱内 runner")
    p.add_argument(
        "--input-json-file",
        default="",
        help="包含 task_id/query/fields/provider 等的 JSON 文件；为空则读 stdin",
    )
    p.add_argument(
        "--result-file",
        default="/root/result_simple.json",
        help="覆盖输入 JSON 中的 result_file 路径",
    )
    return p.parse_args()


def load_input(path: str) -> dict:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    log.info("从 stdin 读取输入 JSON ...")
    return json.loads(sys.stdin.read())


# ---------------------------------------------------------------------------
# Session 文件定位 + final 文本提取
# ---------------------------------------------------------------------------
_SESSIONS_ROOT = "/app/working/workspaces/default/sessions"


def locate_session_file(user_id: str, session_id: str) -> str:
    candidates = [
        f"{_SESSIONS_ROOT}/{user_id}_{session_id}.json",
        f"{_SESSIONS_ROOT}/console/{user_id}_{session_id}.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"session 文件不存在；尝试过: {candidates}")


def extract_final_text(session: dict) -> str:
    """从 session.json 抽取 agent 的最终回复文本（用于事后排查 / DONE 检测）。"""
    agent = session.get("agent", {}) or {}
    traj = agent.get("_model_trajectory", []) or []
    if traj:
        last = traj[-1]
        for part in last.get("response", []) or []:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "") or ""
        return ""
    # 回退 memory.content
    mem = (agent.get("memory", {}) or {}).get("content", []) or []
    for item in reversed(mem):
        if isinstance(item, list) and item and isinstance(item[0], dict):
            msg = item[0]
            if msg.get("role") == "assistant":
                for part in msg.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "") or ""
                break
    return ""


def count_llm_calls(session: dict) -> int:
    agent = session.get("agent", {}) or {}
    traj = agent.get("_model_trajectory", []) or []
    if traj:
        return len(traj)
    mem = (agent.get("memory", {}) or {}).get("content", []) or []
    return sum(
        1
        for item in mem
        if isinstance(item, list) and item and isinstance(item[0], dict)
        and item[0].get("role") == "assistant"
    )


# ---------------------------------------------------------------------------
# input.yaml 渲染（直接用预设 fields，不再依赖 agent 输出）
# ---------------------------------------------------------------------------
def render_input_yaml(task_id: str, fields: dict) -> str:
    """生成 test_case_simple_XXX 期望的 YAML（顶层对象，KEY: VALUE 格式）。

    common.parse_yaml_or_output 用的是 yaml.safe_load + str(v).strip()。
    我们这里只输出标量字段，避免特殊字符注入。
    """
    out_lines: list[str] = [f"TASK_ID: {task_id}"]
    for k, v in fields.items():
        # 用 JSON 字符串避免引号 / 反斜杠 / 冒号转义；safe_load 能识别
        out_lines.append(f"{k}: {json.dumps(str(v), ensure_ascii=False)}")
    return "\n".join(out_lines) + "\n"


# ---------------------------------------------------------------------------
# 在容器内执行 test_case_simple_XXX.py
# ---------------------------------------------------------------------------
def run_test_case(
    task_id: str,
    fields: dict,
    test_cases_root: str,
    workdir: str,
) -> dict:
    """渲染 input.yaml + 调用 test_case_simple_XXX.py，返回标准化结果。

    test_cases_root 期望布局：
      <root>/test_case_simple/test_case_simple_XXX.py
      <root>/test_case_simple/common.py            (代理，from test_case_use 取符号)
      <root>/test_case_use/common.py               (真正实现)
    """
    script = Path(test_cases_root) / "test_case_simple" / f"test_case_{task_id}.py"
    if not script.is_file():
        return {
            "passed": False,
            "score": 0.0,
            "status": "missing_test_case",
            "reason": f"test case 脚本不存在: {script}",
        }

    Path(workdir).mkdir(parents=True, exist_ok=True)
    yaml_path = Path(workdir) / "input.yaml"
    result_path = Path(workdir) / "case_result.json"
    yaml_path.write_text(render_input_yaml(task_id, fields), encoding="utf-8")

    # 让脚本的 `from common import *` 找到 test_case_simple/common.py
    cmd = [
        sys.executable,
        str(script),
        "--yaml-file",
        str(yaml_path),
        "--result-file",
        str(result_path),
    ]
    log.info("执行 test_case: %s", " ".join(shlex.quote(c) for c in cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "passed": False,
            "score": 0.0,
            "status": "timeout",
            "reason": str(e),
            "stdout": (e.stdout or "")[-2000:] if hasattr(e, "stdout") else "",
            "stderr": (e.stderr or "")[-2000:] if hasattr(e, "stderr") else "",
        }

    inner: dict = {}
    if result_path.is_file():
        try:
            inner = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("解析 case_result.json 失败: %s", e)

    passed = bool(inner.get("passed", proc.returncode == 0))
    checks = inner.get("checks", []) or []
    score = float(sum(1 for c in checks if c.get("passed"))) / max(1, len(checks))
    if not checks:
        # checks 为空兜底为 0/1（防止意外）
        score = 1.0 if passed else 0.0

    return {
        "passed": passed,
        "score": score,
        "status": "passed" if passed else "failed",
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[-4000:],
        "stderr": (proc.stderr or "")[-4000:],
        "yaml_file": str(yaml_path),
        "case_result_file": str(result_path),
        "inner_report": inner,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    cfg = load_input(args.input_json_file)

    task_id = str(cfg["task_id"])
    query = str(cfg["query"])
    fields = dict(cfg.get("fields") or {})
    user_id = str(cfg.get("user_id") or "default")
    url = str(cfg.get("url") or "http://127.0.0.1:8088")
    provider_base_url = cfg.get("provider_base_url") or None
    provider_model_id = str(cfg.get("provider_model_id") or "")
    provider_api_key = str(
        cfg.get("provider_api_key")
        or os.getenv("TINKER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    test_cases_root = str(cfg.get("test_cases_root") or "/root/test_cases")
    result_file = args.result_file or str(cfg.get("result_file") or "/root/result_simple.json")
    session_id = cfg.get("session_id") or f"{task_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    workdir = str(cfg.get("workdir") or f"/tmp/cpw_simple/{task_id}")

    summary: dict = {
        "task_id": task_id,
        "session_id": session_id,
        "user_id": user_id,
        "passed": False,
        "score": 0.0,
        "status": "init",
        "agent_call_seconds": 0.0,
        "agent_llm_calls": 0,
    }

    try:
        if not provider_model_id:
            raise ValueError("provider_model_id 不能为空")

        # 1) qwenpaw 必须就绪
        if not _ensure_qwenpaw(url):
            summary.update(status="qwenpaw_not_ready", reason="qwenpaw HTTP server 未就绪")
            return _dump_and_exit(result_file, summary)

        # 1.5) 清理 qwenpaw default workspace 里的 BOOTSTRAP.md。
        # qwenpaw v1.1.7 的 "qwenpaw init --defaults" 会在 default workspace
        # 创建 BOOTSTRAP.md，触发 agent 进入“引导模式”并把“引导质问”拼
        # 到 user_input 前面，污染真实任务 prompt（导致 simple_004 这类
        # task 被带偏到聊天引导场景，根本不会开始干活）。
        # 每次 task 运行前在 sandbox 内部 unlink 该文件，避免引导模式。
        for _bootstrap_path in (
            "/app/working/workspaces/default/BOOTSTRAP.md",
            "/app/working/workspace/default/BOOTSTRAP.md",  # 兼容可能的别名
        ):
            try:
                if os.path.exists(_bootstrap_path):
                    os.unlink(_bootstrap_path)
                    log.info("unlink %s to skip qwenpaw onboarding", _bootstrap_path)
            except OSError as _e:
                log.warning("unlink %s failed: %s", _bootstrap_path, _e)

        # 2) 调 agent
        log.info("调用 agent: task=%s session=%s", task_id, session_id)
        # 硬超时：避免 agent 陷入死循环（例如反复重试同一错误）拖死 RL
        # 训练。默认 300s，可由 host 环境变量 MAX_AGENT_SECONDS 覆盖。
        # 超时不是基础设施问题，不该走 transient skip；走 over_time 路径
        # (workflow 会看 agent_call_seconds > MAX_TRAJ_SECONDS 判 over_time, reward=0)。
        max_agent_seconds = int(os.environ.get("MAX_AGENT_SECONDS", "300"))

        class _AgentTimeout(Exception):
            pass

        def _alarm_handler(signum, frame):
            raise _AgentTimeout(f"call_agent exceeded {max_agent_seconds}s")

        import signal as _signal
        _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.alarm(max_agent_seconds)
        agent_timed_out = False
        t0 = time.time()
        try:
            call_agent(
                url=url.rstrip("/"),
                user_input=query,
                session_id=session_id,
                user_id=user_id,
                provider_name=RL_PROVIDER_NAME,
                provider_base_url=provider_base_url,
                provider_api_key=provider_api_key,
                provider_model_id=provider_model_id,
            )
        except _AgentTimeout as e:
            agent_timed_out = True
            log.warning("call_agent 强制超时: %s", e)
            # 不 return；后续仍读部分 session.json 拿部分 trajectory。
            # status 不设为 transient，让 workflow 看 agent_call_seconds 超限判 over_time。
        except Exception as e:
            log.exception("call_agent 异常")
            summary.update(status="call_agent_error", reason=f"{type(e).__name__}: {e}")
            _signal.alarm(0)
            return _dump_and_exit(result_file, summary)
        finally:
            _signal.alarm(0)
        summary["agent_call_seconds"] = round(time.time() - t0, 2)
        if agent_timed_out:
            summary["agent_timed_out"] = True

        # 3) 收集 session
        try:
            session_file = locate_session_file(user_id, session_id)
        except FileNotFoundError as e:
            summary.update(status="session_missing", reason=str(e))
            return _dump_and_exit(result_file, summary)
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        summary["session_file"] = session_file
        # 顺手 export 一份到 /root，便于 Trinity 拉回
        try:
            shutil.copy2(session_file, "/root/session.json")
            summary["session_export"] = "/root/session.json"
        except Exception as e:
            log.warning("session export 失败: %s", e)
        final_text = extract_final_text(session_data)
        summary["agent_llm_calls"] = count_llm_calls(session_data)
        summary["final_text_tail"] = final_text[-1500:]
        summary["done_marker"] = "【DONE】" in final_text

        # 4) 跑确定性 test_case（YAML 来自预设 fields，与 agent 输出无关）
        case = run_test_case(task_id, fields, test_cases_root, workdir)
        summary.update(
            passed=case["passed"],
            score=case["score"],
            status=case["status"],
            test_case_exit_code=case.get("exit_code"),
            test_case_stdout_tail=case.get("stdout", "")[-1500:],
            test_case_stderr_tail=case.get("stderr", "")[-1500:],
            test_case_inner=case.get("inner_report"),
            test_case_result_file=case.get("case_result_file"),
        )

        # 防御：agent 一次 LLM 调用都没成功。常见原因：provider
        # （TuFT/上游 OpenAI 服务）重启 / 5xx / 超时。这种是 transient
        # 故障，不应该被当作 reward=0 训练信号污染下游。明确标
        # 记为 no_llm_call，workflow 层可据此重试或丢弃。
        if int(summary.get("agent_llm_calls") or 0) == 0:
            summary["status"] = "no_llm_call"
            summary["reason"] = (
                "agent_llm_calls=0：qwenpaw HTTP 调用后 session trajectory 为空，"
                "大概率是上游 LLM provider (TuFT/OpenAI) transient 不可用。"
            )
    except Exception as e:
        log.exception("run_simple 顶层异常")
        summary.update(status="error", reason=f"{type(e).__name__}: {e}")

    return _dump_and_exit(result_file, summary)


def _dump_and_exit(result_file: str, summary: dict) -> int:
    Path(result_file).parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(
        "[result] task=%s passed=%s score=%.3f status=%s -> %s",
        summary.get("task_id"),
        summary.get("passed"),
        float(summary.get("score") or 0.0),
        summary.get("status"),
        result_file,
    )
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
