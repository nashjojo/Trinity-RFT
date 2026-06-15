import argparse
import base64
import hashlib
import json
import os
import pickle
import shlex
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

# shlex.quote 别名，便于在拼接容器路径时转义
shlex_quote = shlex.quote

import httpx
import numpy as np
from e2b import (
    CommandExitException,
    NotFoundException,
    Sandbox,
    SandboxException,
    TimeoutException,
)


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_sandbox_info(sandbox_id, token, domain, logger):
    """Get sandbox info via API to retrieve dashboard URL"""
    try:
        url = f"https://api.{domain}/sandboxes/{sandbox_id}"
        headers = {"X-API-KEY": token, "X-Generate-Dashboard-Url": "true"}

        resp = httpx.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json()

            dashboard_url = None

            if "metadata" in data:
                metadata = data["metadata"]
                if isinstance(metadata, dict):
                    if "sandbox.alicloud.com/dashboard-url" in metadata:
                        dashboard_url = metadata["sandbox.alicloud.com/dashboard-url"]
                    elif "dashboard_url" in metadata:
                        dashboard_url = metadata["dashboard_url"]

            if not dashboard_url and "dashboard_url" in data:
                dashboard_url = data["dashboard_url"]

            if dashboard_url:
                dashboard_url = dashboard_url.replace(".vpc.", ".")
                logger.info(f"\n{Colors.HEADER}{Colors.BOLD}🌐 Dashboard URL:{Colors.ENDC}")
                logger.info(f"{Colors.OKGREEN}{Colors.UNDERLINE}{dashboard_url}{Colors.ENDC}\n")
            else:
                logger.warning(f"{Colors.WARNING}Dashboard URL not found in metadata{Colors.ENDC}")

            return data
        else:
            logger.error(
                f"{Colors.FAIL}Failed to get sandbox info: {resp.status_code}{Colors.ENDC}"
            )
            return None

    except Exception as e:
        logger.error(f"{Colors.WARNING}Could not fetch dashboard URL: {e}{Colors.ENDC}")
        return None


def connect_sandbox(sandbox_id, token, domain, logger):
    """Connect to existing sandbox"""
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    os.environ["E2B_API_KEY"] = token

    sandbox = Sandbox.connect(sandbox_id)

    logger.info(f"    {Colors.OKGREEN}✓ Connected to sandbox{Colors.ENDC}")

    return sandbox


def create_sandbox(token, domain, template, logger) -> Sandbox:
    """Create sandbox with authorization header"""
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    os.environ["E2B_API_KEY"] = token

    sandbox = Sandbox.create(
        template=template,
        timeout=3600,  # 1 hour ; TODO: make this configurable
        headers={
            "template": template,
        },
    )

    logger.info(
        f"    {Colors.OKGREEN}✓ Sandbox created{Colors.ENDC} (ID: {Colors.BOLD}{sandbox.sandbox_id}{Colors.ENDC})"
    )

    return sandbox


def update_sandbox_files(sandbox: Sandbox, template, logger):
    """递归同步 utils/ 下所有 .py 和 .sh 文件到 /root/，保留目录结构。

    - 顶层模块（如 bench_client.py）→ /root/bench_client.py
    - 包目录（如 copaw_eval/__init__.py）→ /root/copaw_eval/__init__.py
    - md5_maps.json 中的 key 用 POSIX 相对路径（如 "copaw_eval/__init__.py"）。
    - 跳过 __pycache__ 等运行时产物。
    """
    utils_dir = Path(__file__).parent.parent / "utils"
    local_md5_map = {}
    for file in utils_dir.rglob("*"):
        if not file.is_file() or file.suffix not in {".py", ".sh"}:
            continue
        rel_parts = file.relative_to(utils_dir).parts
        if any(p.startswith((".", "__pycache__")) for p in rel_parts):
            continue
        rel_path = file.relative_to(utils_dir).as_posix()
        with open(file, "rb") as f:
            file_md5 = hashlib.file_digest(f, "md5").hexdigest()
        local_md5_map[rel_path] = file_md5

    # get md5 map in sandbox
    files = " ".join(local_md5_map.keys())
    try:
        result = sandbox.commands.run(f"md5sum {files}")
        stdout = result.stdout.strip()
    except CommandExitException as e:
        stdout = e.stdout.strip()
    md5_map = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "No such file or directory" in line:
            rel_path = line.split(": ")[1]
            md5_map[rel_path] = None
        else:
            md5, rel_path = line.split("  ")
            md5_map[rel_path] = md5

    # upload files that are missing or different
    for rel_path, md5 in local_md5_map.items():
        if md5_map.get(rel_path, None) != md5:
            logger.info(f"Uploading {rel_path} to sandbox")
            with open(utils_dir / rel_path, "r") as f:
                sandbox.files.write(f"/root/{rel_path}", f)


def update_sandbox_dir(sandbox: Sandbox, host_dir: Path, sandbox_dir: str, logger):
    """递归同步 host_dir 到 sandbox_dir，md5 增量判重。

    与 update_sandbox_files 不同：不限 .py/.sh 类型，适用于 test_cases/ 这种
    需保证目录完整同步的场景。sandbox_dir 不存在时会自动创建。
    """
    host_dir = Path(host_dir)
    if not host_dir.is_dir():
        logger.warning(f"[sync] host 目录不存在: {host_dir}")
        return

    local_md5_map: dict[str, str] = {}
    for file in host_dir.rglob("*"):
        if not file.is_file():
            continue
        rel_parts = file.relative_to(host_dir).parts
        if any(p.startswith((".", "__pycache__")) for p in rel_parts):
            continue
        rel_path = file.relative_to(host_dir).as_posix()
        with open(file, "rb") as f:
            local_md5_map[rel_path] = hashlib.file_digest(f, "md5").hexdigest()

    if not local_md5_map:
        logger.warning(f"[sync] host 目录空: {host_dir}")
        return

    # 确保 sandbox 上的目标目录存在
    try:
        sandbox.commands.run(f"mkdir -p {shlex_quote(sandbox_dir)}")
    except Exception as e:
        logger.warning(f"[sync] mkdir {sandbox_dir} 失败: {e}")

    file_list = " ".join(
        shlex_quote(f"{sandbox_dir.rstrip('/')}/{rel}") for rel in local_md5_map.keys()
    )
    try:
        result = sandbox.commands.run(f"md5sum {file_list}", timeout=60)
        stdout = result.stdout.strip() if result.stdout else ""
    except CommandExitException as e:
        stdout = (e.stdout or "").strip()
    except Exception as e:
        logger.warning(f"[sync] md5sum 失败，全量重传: {e}")
        stdout = ""

    md5_map: dict[str, str | None] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "No such file or directory" in line:
            # 格式例如: 'md5sum: /root/test_cases/x.py: No such file or directory'
            seg = line.split(": ")
            if len(seg) >= 2:
                full_path = seg[-2]
                rel = full_path[len(sandbox_dir.rstrip("/")) + 1 :]
                md5_map[rel] = None
        elif "  " in line:
            md5, full_path = line.split("  ", 1)
            rel = full_path[len(sandbox_dir.rstrip("/")) + 1 :]
            md5_map[rel] = md5

    uploaded = 0
    for rel_path, md5 in local_md5_map.items():
        if md5_map.get(rel_path) == md5:
            continue
        target = f"{sandbox_dir.rstrip('/')}/{rel_path}"
        # 预创建子目录
        sub = os.path.dirname(target)
        if sub:
            try:
                sandbox.commands.run(f"mkdir -p {shlex_quote(sub)}")
            except Exception as e:
                logger.warning(f"[sync] mkdir {sub} 失败: {e}")
        with open(host_dir / rel_path, "rb") as f:
            sandbox.files.write(target, f)
        uploaded += 1
    logger.info(f"[sync] {host_dir} -> {sandbox_dir}: 上传 {uploaded}/{len(local_md5_map)} 个文件")


def _wait_for_commands_api(sandbox: Sandbox, logger, max_retries: int = 10, delay: float = 3.0) -> bool:
    """Retry a lightweight command until the commands API is ready.

    E2B has a known race: ``is_running()`` returns True but ``commands.run()``
    still raises "Sandbox is still pending".  This helper bridges that gap.
    """
    for i in range(1, max_retries + 1):
        try:
            result = sandbox.commands.run("echo ready", timeout=15)
            if "ready" in (result.stdout or ""):
                logger.info(f"    {Colors.OKGREEN}✓ Commands API ready (probe {i}){Colors.ENDC}")
                return True
        except Exception as e:
            logger.warning(f"    [cmd-probe {i}/{max_retries}] Commands API not ready: {e}")
            if i < max_retries:
                time.sleep(delay)
    logger.error(f"    {Colors.FAIL}✗ Commands API did not become ready after {max_retries} probes{Colors.ENDC}")
    return False


def get_or_create_sandbox(sandbox_id, token, domain, template, logger) -> Tuple[Sandbox, bool]:
    """Get existing sandbox or create new one"""
    if sandbox_id:
        logger.info(
            f"\n{Colors.OKCYAN}[1] Connecting to existing sandbox:{Colors.ENDC} {Colors.BOLD}{sandbox_id}{Colors.ENDC}"
        )
        sandbox = connect_sandbox(sandbox_id, token, domain, logger)
        get_sandbox_info(sandbox_id, token, domain, logger)
        _wait_for_commands_api(sandbox, logger)
        update_sandbox_files(sandbox, template, logger)
        return sandbox, False
    else:
        logger.info(
            f"\n{Colors.OKCYAN}[1] Creating sandbox with template:{Colors.ENDC} {Colors.BOLD}{template}{Colors.ENDC}"
        )
        sandbox = create_sandbox(token, domain, template, logger)
        logger.info(f"\n{Colors.OKCYAN}[2] Waiting for sandbox to be ready...{Colors.ENDC}")
        max_attempts = 60
        for attempt in range(1, max_attempts + 1):
            try:
                is_running = sandbox.is_running()
                status_str = (
                    f"{Colors.OKGREEN}Running{Colors.ENDC}"
                    if is_running
                    else f"{Colors.WARNING}Not Running{Colors.ENDC}"
                )
                logger.info(f"    [{attempt}/{max_attempts}] Sandbox status: {status_str}")
                if is_running:
                    logger.info(f"    {Colors.OKGREEN}✓ Sandbox is now running!{Colors.ENDC}")
                    get_sandbox_info(sandbox.sandbox_id, token, domain, logger)
                    _wait_for_commands_api(sandbox, logger)
                    update_sandbox_files(sandbox, template, logger)
                    return sandbox, True
            except Exception as e:
                logger.error(
                    f"    [{attempt}/{max_attempts}] {Colors.WARNING}Failed to check status:{Colors.ENDC} {e}"
                )

            if attempt < max_attempts:
                time.sleep(2)

        logger.warning(
            f"    {Colors.WARNING}Warning: Sandbox did not reach Running state within timeout{Colors.ENDC}"
        )
        get_sandbox_info(sandbox.sandbox_id, token, domain, logger)
        _wait_for_commands_api(sandbox, logger)
        update_sandbox_files(sandbox, template, logger)
        return sandbox, True


def run_with_reconnect(sandbox: Sandbox, cmd, envs, logger, max_retries=5):
    # 1. 后台启动命令
    handle = sandbox.commands.run(
        cmd,
        background=True,
        envs=envs,
        timeout=1000,  # about 16 min; was 3600
        request_timeout=1800,
    )
    pid = handle.pid

    for attempt in range(max_retries):
        try:
            # 2. 等待命令完成（此处维持 streaming 接收输出）
            result = handle.wait(
                on_stdout=lambda data: logger.info(f"[stdout]: {data.rstrip()}"),
                on_stderr=lambda data: logger.info(f"[stderr]: {data.rstrip()}"),
            )
            return result
        except TimeoutException as e:
            logger.warning(f"连接断开 (attempt {attempt + 1}): {e}")
            time.sleep(3)
            if attempt != max_retries - 1:
                # 3. 重连到 sandbox 和进程
                handle = sandbox.commands.connect(
                    pid,
                    timeout=1200,
                    request_timeout=1800,
                )
            else:
                raise e
    return None


def _tinker_provider_api_key() -> str:
    return os.environ.get("TINKER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "tml-tuft-dev-key"


def _append_tinker_provider_api_key(cmd: str) -> str:
    if "--provider-api-key" in cmd:
        return cmd
    return f"{cmd} --provider-api-key {_tinker_provider_api_key()}"


def launch_run_py(
    sandbox: Sandbox, cmd: str, oss_config, dashscope_api_key, logger, envs={}, raise_error=False
):
    assert dashscope_api_key, "DASHSCOPE_API_KEY is required to run the workflow"
    dashscope_api_keys = dashscope_api_key.split(",")
    dashscope_api_key = np.random.choice(dashscope_api_keys).item()
    t0 = time.perf_counter()
    envs.update(
        {
            "OSS_ACCESS_KEY_ID": oss_config["access_key_id"],
            "OSS_ACCESS_KEY_SECRET": oss_config["access_key_secret"],
            "OSS_REGION": oss_config["region"],
            "OSS_ENDPOINT": oss_config["endpoint"],
            "OSS_BUCKET_NAME": oss_config["bucket_name"],
            "DASHSCOPE_API_KEY": dashscope_api_key,
        }
    )
    # When ``TINKER_API_KEY`` is set, inject it as ``OPENAI_API_KEY`` so that
    # ``run.py`` inside the E2B sandbox can authenticate against the TuFT
    # OpenAI-compatible API (or any other OpenAI-protocol service) that Trinity
    # passes via ``--provider-base-url``.
    tinker_api_key = os.environ.get("TINKER_API_KEY")
    if tinker_api_key:
        envs["OPENAI_API_KEY"] = tinker_api_key
    # auto_eval.py 注入的每请求 sampling kwargs（JSON 字符串），透传给沙箱里的 run.py
    gen_kwargs = os.environ.get("AUTO_EVAL_GENERATE_KWARGS")
    if gen_kwargs:
        envs["AUTO_EVAL_GENERATE_KWARGS"] = gen_kwargs
    try:
        logger.info(f"Running command in sandbox: {cmd}")
        logger.info(f"Sandbox envs: {envs}")
        result = run_with_reconnect(sandbox, cmd, envs, logger)
        run_outputs = result.stdout + "\n" + result.stderr
    except CommandExitException as e:
        logger.info("run.py exited with non-zero exit code: %s", e.exit_code)
        logger.info("Error stdout: %s", e.stdout.strip())
        logger.info("Error stderr: %s", e.stderr.strip())
        run_outputs = e.stdout + "\n" + e.stderr
        if raise_error:
            raise e
    latency_seconds = time.perf_counter() - t0
    return latency_seconds, run_outputs


def _download_file(sandbox: Sandbox, remote_path: str, logger, format: Optional[str] = None):
    for attempt in range(30):
        try:
            content = sandbox.files.read(remote_path, format=format)
            return content
        except SandboxException as e:
            logger.warning(f"Attempt {attempt + 1}/30: {remote_path} not ready, retrying... ({e})")
            time.sleep(2)
    raise FileNotFoundError(f"{remote_path} not found in sandbox after multiple attempts")


def _setup_otel_envs(otel_config: dict = {}):
    key_map = {
        "endpoint": "OTEL_EXPORTER_OTLP_ENDPOINT",
        "service_name": "OTEL_SERVICE_NAME",
        "arms_license_key": "OTEL_ARMS_LICENSE_KEY",
        "arms_project": "OTEL_ARMS_PROJECT",
        "cms_workspace": "OTEL_CMS_WORKSPACE",
    }
    envs = {}
    for k, v in key_map.items():
        if v in os.environ:
            envs[v] = os.environ[v]
        if otel_config.get(k, None) is not None:
            envs[v] = str(otel_config[k])
    return envs


def run_workflow(
    sandbox: Sandbox,
    task_id,
    oss_config,
    otel_config,
    dashscope_api_key,
    api_server_url,
    model_path,
    logger,
):
    cmd = _append_tinker_provider_api_key(
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-base-url {api_server_url} --provider-model-id {model_path}"
    )
    envs = {}
    enable_otel = otel_config.pop("enable", False)
    if enable_otel:
        logger.info("Restarting qwenpaw app with LOONGSUITE_PYTHON_SITE_BOOTSTRAP=True")
        sandbox.commands.run("pkill -f '[q]wenpaw app' || true", timeout=30)
        qwenpaw_envs = dict(envs)
        qwenpaw_envs["LOONGSUITE_PYTHON_SITE_BOOTSTRAP"] = "True"
        result = sandbox.commands.run(
            "qwenpaw app |& tee /app/qwenpaw-app.log",
            background=True,
            envs=qwenpaw_envs,
        )
        for stdout, stderr, _ in result:
            if stdout:
                logger.debug(f"[qwenpaw app stdout]: {stdout.strip()}")
                if "http://127.0.0.1:8088" in stdout:
                    logger.info("qwenpaw app restarted with pid %d", result.pid)
                    break
            if stderr:
                logger.debug(f"[qwenpaw app stderr]: {stderr.strip()}")
        # 注入 OTEL 相关环境变量
        cmd += " --enable-otel"
        envs.update(_setup_otel_envs(otel_config))
    _, _ = launch_run_py(
        sandbox, cmd, oss_config, dashscope_api_key, logger, envs=envs, raise_error=True
    )

    content = _download_file(sandbox, "/root/dataset.pkl", logger, format="bytes")
    dataset = pickle.loads(content)
    return dataset


def _save_and_extract_zip(sandbox: Sandbox, remote_path: str, task_dir: str, logger):
    """Download a zip from the sandbox, extract it, and delete the zip."""
    filename = os.path.basename(remote_path)
    zip_path = os.path.join(task_dir, filename)
    try:
        data = sandbox.files.read(remote_path, format="bytes")
        with open(zip_path, "wb") as f:
            f.write(data)
        extract_dir = os.path.join(task_dir, filename.replace(".zip", ""))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        os.remove(zip_path)
        logger.info("Extracted %s -> %s", filename, extract_dir)
    except NotFoundException as e:
        logger.warning(f"{Colors.WARNING}{filename} not found: {e}{Colors.ENDC}")
    except zipfile.BadZipFile as e:
        logger.warning(f"{Colors.WARNING}{filename} is not a valid zip: {e}{Colors.ENDC}")


def run_eval_workflow(
    sandbox: Sandbox,
    task_id: str,
    oss_config: dict,
    dashscope_api_key: str,
    api_server_url: str,
    model_path: str,
    model_label: str,
    checkpoint_job_dir: str,
    logger,
):
    cmd = _append_tinker_provider_api_key(
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-base-url {api_server_url} --provider-model-id {model_path} --evaluation"
    )
    latency_seconds, _ = launch_run_py(sandbox, cmd, oss_config, dashscope_api_key, logger)

    if model_label:
        task_dir = os.path.join(checkpoint_job_dir, model_label, task_id)
    else:
        task_dir = os.path.join(checkpoint_job_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    logger.info("Saving summary to %s", task_dir)

    summary = sandbox.files.read("/root/summary.json")
    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        f.write(summary)
    summary_data = json.loads(summary)
    logger.info("summary.json content: %s", summary_data)

    session = sandbox.files.read("/root/session.json")
    with open(os.path.join(task_dir, "session.json"), "w") as f:
        f.write(session)

    _save_and_extract_zip(sandbox, "/root/screenshots.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/workspace_files.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/qwenpaw_log.zip", task_dir, logger)

    score = summary_data.get("summary", {}).get("avg_score", -1) if summary_data else -1
    status = "PASS" if score == 100 else "FAIL" if score >= 0 else "ERROR"
    steps = -1
    duration_seconds = -1.0
    response_length = -1
    if summary_data and summary_data.get("tasks"):
        task_steps = [t.get("steps", -1) for t in summary_data["tasks"] if t.get("steps", -1) >= 0]
        if task_steps:
            steps = sum(task_steps)
        durs = [
            t.get("duration_seconds")
            for t in summary_data["tasks"]
            if t.get("duration_seconds") is not None
        ]
        if durs:
            duration_seconds = sum(durs)
        else:
            duration_seconds = latency_seconds
        total_chars = 0
        for t in summary_data["tasks"]:
            for step in t.get("trajectory") or []:
                total_chars += len((step.get("thought") or ""))
            total_chars += len((t.get("final_text") or ""))
        response_length = total_chars
    if duration_seconds < 0:
        duration_seconds = latency_seconds
    tag = f"[{model_label or model_path}] {task_id}" if (model_label or model_path) else task_id
    print(
        f"[{status}] {tag} — score: {score}, steps: {steps}, 输出(计费): {response_length}, time: {duration_seconds:.1f}s"
    )
    return {
        "task": task_id,
        "model": model_label or model_path,
        "score": score,
        "status": status,
        "steps": steps,
        "response_length": response_length,
        "latency_seconds": round(latency_seconds, 2),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds >= 0 else -1,
    }


def run_teacher_workflow(
    sandbox: Sandbox, task_id, oss_config, dashscope_api_key, model_id, logger
):
    cmd = (
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-name dashscope --provider-model-id {model_id} "
        f"--provider-api-key {dashscope_api_key} --evaluation"
    )
    latency_seconds, run_outputs = launch_run_py(
        sandbox, cmd, oss_config, dashscope_api_key, logger
    )

    trajectory_file = sandbox.files.read("/root/tests/traj.json")
    trajectory = json.loads(trajectory_file)
    return trajectory_file, trajectory, run_outputs


def run_teacher_eval_workflow(
    sandbox: Sandbox,
    task_id: str,
    oss_config: dict,
    dashscope_api_key: str,
    model_id: str,
    model_label: str,
    checkpoint_job_dir: str,
    logger,
):
    """Teacher model evaluation: uses DashScope built-in provider instead of
    --provider-base-url, and collects the same summary/session/screenshots
    artifacts as run_eval_workflow."""
    cmd = (
        f"python run.py --task-id {task_id} --oss-prefix {oss_config['prefix']} "
        f"--provider-name dashscope --provider-model-id {model_id} "
        f"--provider-api-key {dashscope_api_key} --evaluation"
    )
    latency_seconds, _ = launch_run_py(sandbox, cmd, oss_config, dashscope_api_key, logger)

    if model_label:
        task_dir = os.path.join(checkpoint_job_dir, model_label, task_id)
    else:
        task_dir = os.path.join(checkpoint_job_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    logger.info("Saving summary to %s", task_dir)

    summary = sandbox.files.read("/root/summary.json")
    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        f.write(summary)
    summary_data = json.loads(summary)
    logger.info("summary.json content: %s", summary_data)

    session = sandbox.files.read("/root/session.json")
    with open(os.path.join(task_dir, "session.json"), "w") as f:
        f.write(session)

    _save_and_extract_zip(sandbox, "/root/screenshots.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/workspace_files.zip", task_dir, logger)
    _save_and_extract_zip(sandbox, "/root/qwenpaw_log.zip", task_dir, logger)

    score = summary_data.get("summary", {}).get("avg_score", -1) if summary_data else -1
    status = "PASS" if score == 100 else "FAIL" if score >= 0 else "ERROR"
    steps = -1
    duration_seconds = -1.0
    response_length = -1
    if summary_data and summary_data.get("tasks"):
        task_steps = [t.get("steps", -1) for t in summary_data["tasks"] if t.get("steps", -1) >= 0]
        if task_steps:
            steps = sum(task_steps)
        durs = [
            t.get("duration_seconds")
            for t in summary_data["tasks"]
            if t.get("duration_seconds") is not None
        ]
        if durs:
            duration_seconds = sum(durs)
        else:
            duration_seconds = latency_seconds
        total_chars = 0
        for t in summary_data["tasks"]:
            for step in t.get("trajectory") or []:
                total_chars += len((step.get("thought") or ""))
            total_chars += len((t.get("final_text") or ""))
        response_length = total_chars
    if duration_seconds < 0:
        duration_seconds = latency_seconds
    tag = f"[{model_label or model_id}] {task_id}" if (model_label or model_id) else task_id
    print(
        f"[{status}] {tag} — score: {score}, steps: {steps}, 输出(计费): {response_length}, time: {duration_seconds:.1f}s"
    )
    return {
        "task": task_id,
        "model": model_label or model_id,
        "score": score,
        "status": status,
        "steps": steps,
        "response_length": response_length,
        "latency_seconds": round(latency_seconds, 2),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds >= 0 else -1,
    }


def _inject_llm_seed(sandbox: Sandbox, seed: int, logger):
    """Inject LLM seed into sandbox: patch openai SDK + restart qwenpaw.

    Writes a Python hook file + .pth trigger into the sandbox's
    site-packages so that the qwenpaw process (which uses openai SDK)
    automatically adds ``seed=<value>`` to every ``chat.completions.create()``
    call.  This enables deterministic T=1 evaluation.

    Call flow:
      qwenpaw (Python) starts
        → site-packages/llm_seed.pth triggers import
        → _llm_seed_hook.py patches openai SDK
        → all subsequent openai calls include seed
        → TuFT OAI API forwards seed to vLLM
        → vLLM uses seed for deterministic sampling
    """
    logger.info(f"[seed_inject] injecting LLM_SEED={seed} into sandbox")

    # 1) Write the hook module
    hook_code = f'''"""Auto-injected by Trinity eval: add seed={seed} to all openai calls."""
import os as _os
_seed = int(_os.environ.get("LLM_SEED", "{seed}"))
try:
    import openai.resources.chat.completions as _cc
    _orig_sync = _cc.Completions.create
    _orig_async = _cc.AsyncCompletions.create

    def _patched_create(self, *args, **kwargs):
        kwargs.setdefault("seed", _seed)
        return _orig_sync(self, *args, **kwargs)

    async def _patched_async_create(self, *args, **kwargs):
        kwargs.setdefault("seed", _seed)
        return await _orig_async(self, *args, **kwargs)

    _cc.Completions.create = _patched_create
    _cc.AsyncCompletions.create = _patched_async_create
    import sys
    print(f"[llm_seed_hook] patched openai SDK: seed={{_seed}}", file=sys.stderr)
except Exception as _e:
    import sys
    print(f"[llm_seed_hook] WARNING: patch failed: {{_e}}", file=sys.stderr)
'''
    sandbox.files.write(
        "/app/venv/lib/python3.11/site-packages/_llm_seed_hook.py",
        hook_code,
    )

    # 2) Write .pth file to auto-import the hook on Python startup
    sandbox.files.write(
        "/app/venv/lib/python3.11/site-packages/llm_seed.pth",
        "import _llm_seed_hook\n",
    )

    # 3) Kill existing qwenpaw (which doesn't have the hook)
    try:
        sandbox.commands.run("pkill -f 'qwenpaw' || true", timeout=10)
    except Exception as e:
        logger.warning(f"[seed_inject] pkill qwenpaw: {e}")
    time.sleep(2)

    # 4) Restart qwenpaw with LLM_SEED env (hook will auto-load via .pth)
    try:
        result = sandbox.commands.run(
            "qwenpaw app &> /app/qwenpaw-app-seeded.log",
            background=True,
            envs={"LLM_SEED": str(seed)},
        )
        logger.info(f"[seed_inject] qwenpaw restarted with seed={seed}, pid={result.pid}")
    except Exception as e:
        logger.error(f"[seed_inject] qwenpaw restart failed: {e}")
        raise

    # 5) Wait for qwenpaw HTTP API ready
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            probe = sandbox.commands.run(
                "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8088/api/version', timeout=3)\"",
                timeout=10,
            )
            if probe.exit_code == 0:
                logger.info(f"[seed_inject] qwenpaw ready (seed={seed})")
                return
        except Exception:
            pass
        time.sleep(3)
    logger.warning("[seed_inject] qwenpaw did not become ready in 90s, proceeding anyway")


def run_simple_workflow(
    sandbox: Sandbox,
    task_id: str,
    query: str,
    fields: dict,
    api_server_url: str,
    model_path: str,
    dashscope_api_key: Optional[str],
    test_cases_host_dir: Path,
    model_label: str,
    checkpoint_job_dir: str,
    logger,
) -> dict:
    """queries_simple 专用：注入 query/fields 到 sandbox 中跑 run_simple.py。

    不依赖 OSS：仅同步 utils/ 与 test_cases/ 进 sandbox，然后用 stdin 把
    input JSON 传给 /root/run_simple.py，跑完拉回 /root/result_simple.json。
    """
    # 1) 同步 test_cases/（common.py + test_case_simple/* + test_case_use/*）
    update_sandbox_dir(sandbox, test_cases_host_dir, "/root/test_cases", logger)

    # 2) 构造传给 run_simple.py 的输入 JSON
    tinker_api_key = (
        os.environ.get("TINKER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "tml-tuft-dev-key"
    )
    payload = {
        "task_id": task_id,
        "query": query,
        "fields": fields,
        "user_id": "default",
        "url": "http://127.0.0.1:8088",
        "provider_base_url": api_server_url,
        "provider_model_id": model_path,
        "provider_api_key": tinker_api_key,
        "test_cases_root": "/root/test_cases",
        "result_file": "/root/result_simple.json",
    }
    # LLM_SEED 评估模式：固定 session_id 消除 prompt 中唯一的动态 token
    # （qwenpaw system prompt 里嵌入了 session_id，含时间戳秒数 → 1 token 差异）
    llm_seed = os.environ.get("LLM_SEED")
    if llm_seed:
        payload["session_id"] = f"{task_id}_eval_seed{llm_seed}"
    payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    # 3) 环境变量：rdashscope_api_key 是可选的，但 qwenpaw 内部某些 fallback 可能有用
    envs: dict = {"OPENAI_API_KEY": tinker_api_key}
    if dashscope_api_key:
        keys = [k for k in dashscope_api_key.split(",") if k.strip()]
        envs["DASHSCOPE_API_KEY"] = np.random.choice(keys).item() if keys else ""

    # 透传 host 上的 length / time penalty 阈值到 sandbox。run_simple.py
    # 会读 MAX_AGENT_SECONDS 启用 signal.alarm 硬超时，防止 agent
    # 陷入死循环拖死训练。
    for k in ("MAX_AGENT_SECONDS", "MAX_STEP_TOKENS", "MAX_TRAJ_SECONDS"):
        v = os.environ.get(k)
        if v:
            envs[k] = v

    # ---- LLM_SEED 注入：固定 seed 可复现评估 ----
    # 当 host 环境变量 LLM_SEED 被设置时（仅 eval 场景），
    # 向 sandbox 内注入 openai SDK 的 seed monkey-patch，
    # 让 qwenpaw 的每次 LLM 调用都带上 seed=<value>。
    # TuFT OAI API 直接透传 seed 给 vLLM SamplingParams。
    llm_seed = os.environ.get("LLM_SEED")
    if llm_seed:
        _inject_llm_seed(sandbox, int(llm_seed), logger)

    cmd = (
        f"echo {payload_b64} | base64 -d | "
        f"python /root/run_simple.py --result-file /root/result_simple.json"
    )
    logger.info(f"Running run_simple.py in sandbox for task={task_id}")
    t0 = time.perf_counter()
    try:
        run_with_reconnect(sandbox, cmd, envs, logger)
    except CommandExitException as e:
        # 脚本 exit≠0 也正常产出 result_simple.json，不报错
        logger.info("run_simple.py exit_code=%s; 仕拉取结果文件", e.exit_code)
    latency_seconds = round(time.perf_counter() - t0, 2)

    # 4) 拉回 result_simple.json
    # 同一 task_id 可能被重复 sample N 次（难度评测 / GRPO），必须用唯一
    # 后缀区分，否则后面的会覆盖前面。用 sandbox_id 作为天然唯一标识。
    run_tag = f"{int(time.time())}_{getattr(sandbox, 'sandbox_id', 'nosbx')}"
    if model_label:
        task_dir = os.path.join(checkpoint_job_dir, model_label, task_id, run_tag)
    else:
        task_dir = os.path.join(checkpoint_job_dir, task_id, run_tag)
    # 诊断日志：明确拉回路径 + actor cwd
    logger.info(
        f"[diag] checkpoint_job_dir={checkpoint_job_dir!r} task_dir={task_dir!r} "
        f"abs={os.path.abspath(task_dir)!r} cwd={os.getcwd()!r}"
    )
    os.makedirs(task_dir, exist_ok=True)
    logger.info(
        f"[diag] after makedirs: isdir={os.path.isdir(task_dir)} "
        f"parent_isdir={os.path.isdir(os.path.dirname(task_dir))} "
        f"checkpoint_root_isdir={os.path.isdir(checkpoint_job_dir)}"
    )

    result_data: dict = {}
    result_local = os.path.join(task_dir, "result_simple.json")
    try:
        result_text = sandbox.files.read("/root/result_simple.json")
        with open(result_local, "w", encoding="utf-8") as f:
            f.write(result_text)
        result_data = json.loads(result_text)
        logger.info(
            f"[diag] result_simple.json written: path={result_local!r} "
            f"isfile={os.path.isfile(result_local)} "
            f"size={os.path.getsize(result_local) if os.path.isfile(result_local) else -1}"
        )
    except Exception as e:
        logger.error(f"拉取 /root/result_simple.json 失败: {type(e).__name__}: {e}")
        result_data = {"task_id": task_id, "passed": False, "score": 0.0, "status": "pull_failed"}

    # 顺手拉一份 session.json（供事后排查）
    try:
        session_text = sandbox.files.read("/root/session.json")
        with open(os.path.join(task_dir, "session.json"), "w", encoding="utf-8") as f:
            f.write(session_text)
    except Exception as e:
        logger.warning(f"session.json 拉取失败: {e}")

    score = float(result_data.get("score") or 0.0)
    passed = bool(result_data.get("passed", False))
    status = "PASS" if passed else ("FAIL" if result_data.get("status") == "failed" else "ERROR")
    tag = f"[{model_label or model_path}] {task_id}" if (model_label or model_path) else task_id
    print(
        f"[{status}] {tag} — passed={passed}, score={score:.3f}, "
        f"agent_call_seconds={result_data.get('agent_call_seconds', 0)}, "
        f"agent_llm_calls={result_data.get('agent_llm_calls', 0)}, "
        f"e2e={latency_seconds:.1f}s"
    )
    return {
        "task": task_id,
        "model": model_label or model_path,
        "score": score * 100.0,  # 与 run_eval_workflow.summary.avg_score 统一量纲
        "passed": passed,
        "status": status,
        "latency_seconds": latency_seconds,
        "agent_call_seconds": float(result_data.get("agent_call_seconds") or 0.0),
        "agent_llm_calls": int(result_data.get("agent_llm_calls") or 0),
        "test_case_exit_code": result_data.get("test_case_exit_code"),
        "reason": result_data.get("reason"),
        "task_dir": task_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.environ.get("E2B_API_KEY", None))
    parser.add_argument("--domain", type=str, default=os.environ.get("E2B_DOMAIN", None))
    parser.add_argument("--template", type=str, default=os.environ.get("E2B_TEMPLATE", None))
    parser.add_argument(
        "--enable-otel", action="store_true", help="Whether to set up OpenTelemetry in the sandbox"
    )
    args = parser.parse_args()

    from trinity.utils.log import get_logger

    logger = get_logger()
    sandbox, created = get_or_create_sandbox(None, args.token, args.domain, args.template, logger)

    utils_dir = Path(__file__).parent.parent / "utils"

    sandbox.commands.run("mkdir patch")
    patch_dir = utils_dir / "patch"
    for patch_file in patch_dir.glob("*.patch"):
        logger.info(f"Uploading patch {patch_file.name} to sandbox...")
        with open(patch_file, "r") as f:
            sandbox.files.write(f"/root/patch/{patch_file.name}", f)

    if args.enable_otel:
        otel_envs = _setup_otel_envs()
        logger.info(f"OTEL envs for sandbox: {otel_envs}")
    else:
        otel_envs = {}
    try:
        result = sandbox.commands.run(
            "pip uninstall qwenpaw -y && "
            "pip install qwenpaw==v1.1.7 && "
            "pip install oss2 pytest py-openjudge pytest-asyncio && "
            "patch /app/venv/lib/python3.11/site-packages/qwenpaw/agents/react_agent.py < /root/patch/react_agent.patch && "
            "patch /app/venv/lib/python3.11/site-packages/qwenpaw/agents/tools/browser_control.py < /root/patch/browser_control.patch && "
            "patch /app/venv/lib/python3.11/site-packages/agentscope/model/_openai_model.py < /root/patch/openai_model.patch && "
            "patch /app/venv/lib/python3.11/site-packages/agentscope/model/_model_response.py < /root/patch/model_response.patch && "
            "apt-get update && "
            "apt-get install -y xfce4 xfce4-goodies x11vnc openbox xvfb novnc websockify supervisor dbus-x11 && "
            "rm -rf /var/lib/apt/lists/* && "
            "echo '100.118.58.9    copaw-dataset.oss-cn-beijing-internal.aliyuncs.com' >> /etc/hosts && "
            "bash /root/setup_otel.sh && "
            "pip install --no-cache-dir 'wrapt<2' && "
            "pip install opentelemetry-instrumentation-openai && "
            "qwenpaw init --defaults --accept-security",
            envs=otel_envs,
            on_stdout=lambda data: logger.info(f"[stdout]: {data.rstrip()}"),
            on_stderr=lambda data: logger.info(f"[stderr]: {data.rstrip()}"),
            timeout=3600,
        )
    except CommandExitException as e:
        logger.info("Error stdout: %s", e.stdout.strip())
        logger.info("Error stderr: %s", e.stderr.strip())
        raise e

    try:
        result = sandbox.commands.run(
            "qwenpaw app &> /app/qwenpaw-app.log",
            background=True,
        )
        logger.info("qwenpaw app started with pid %d", result.pid)
    except CommandExitException as e:
        logger.info("Error starting qwenpaw app. stdout: %s", e.stdout.strip())
        logger.info("Error starting qwenpaw app. stderr: %s", e.stderr.strip())
        raise e

    try:
        result = sandbox.commands.run(
            "bash /root/start-vnc.sh &> /root/start-vnc.log",
            background=True,
        )
        logger.info("VNC server started with pid %d", result.pid)
    except CommandExitException as e:
        logger.info("Error starting VNC server. stdout: %s", e.stdout.strip())
        logger.info("Error starting VNC server. stderr: %s", e.stderr.strip())
        raise e
