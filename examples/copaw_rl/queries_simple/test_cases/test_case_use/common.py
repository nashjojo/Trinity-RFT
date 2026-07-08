#!/usr/bin/env python3
from __future__ import annotations
import base64
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from urllib import request, error

try:
    import yaml
except Exception:
    yaml = None

@dataclass
class CheckResult:
    name:str
    passed:bool
    detail:str

_EXEC_ENV_TYPE = "LOCAL"
_ECS_REGION_ID = ""
_ECS_INSTANCE_ID = ""


def _refresh_exec_env(data: dict):
    """根据解析得到的 YAML/output 数据刷新模块级执行环境变量。

    将 EXEC_ENV_TYPE / ECS_REGION_ID / ECS_INSTANCE_ID 从输入字典同步到
    模块全局变量，供后续 run_cmd 判断命令在本地还是在 ECS 内执行。
    未提供时默认使用 LOCAL 执行模式。
    """
    global _EXEC_ENV_TYPE, _ECS_REGION_ID, _ECS_INSTANCE_ID
    _EXEC_ENV_TYPE = data.get("EXEC_ENV_TYPE", "LOCAL").strip().upper() or "LOCAL"
    _ECS_REGION_ID = data.get("ECS_REGION_ID", "").strip()
    _ECS_INSTANCE_ID = data.get("ECS_INSTANCE_ID", "").strip()


def parse_yaml_or_output(yaml_file:str='', output_file:str=''):
    """解析 agent 最终输出，返回大写键的字段字典。

    支持两种输入源，可叠加使用：
      - yaml_file: 标准 YAML 文件，顶层必须为对象；
      - output_file: 任意文本文件，优先截取 <OUTPUT>...</OUTPUT> 段，
        按 "KEY: VALUE" 逐行解析，自动去除首尾引号。
    两者同时传入时，output_file 的值会覆盖 yaml_file。
    解析完成后会补齐 EXEC_ENV_TYPE/ECS_REGION_ID/ECS_INSTANCE_ID 默认值，
    并调用 _refresh_exec_env 同步到模块执行环境。
    """
    data={}
    if yaml_file:
        if yaml is None:
            raise RuntimeError('未安装 PyYAML')
        with open(yaml_file,'r',encoding='utf-8') as f:
            y = yaml.safe_load(f) or {}
        if not isinstance(y, dict):
            raise ValueError('YAML 顶层必须是对象')
        data.update({str(k).strip().upper(): str(v).strip() for k,v in y.items() if v is not None})
    if output_file:
        txt = open(output_file,'r',encoding='utf-8').read()
        m = re.search(r"<OUTPUT>([\s\S]*?)</OUTPUT>", txt, re.IGNORECASE)
        block = m.group(1) if m else txt
        for line in block.splitlines():
            line=line.strip()
            if not line or ':' not in line:
                continue
            k,v = line.split(':',1)
            data[k.strip().upper()] = v.strip().strip('"').strip("'")
    data.setdefault("EXEC_ENV_TYPE", "LOCAL")
    data.setdefault("ECS_REGION_ID", "")
    data.setdefault("ECS_INSTANCE_ID", "")
    _refresh_exec_env(data)
    return data

def norm_url(x:str):
    """规范化 endpoint URL。

    - 去掉前后空白；
    - 若未带 http/https scheme，默认补 http://；
    - 去掉结尾多余的 '/'；
    - 空字符串会抛出 ValueError。
    """
    x=x.strip()
    if not x:
        raise ValueError('endpoint 为空')
    if not x.startswith(('http://','https://')):
        x='http://'+x
    return x.rstrip('/')

def http_get(url:str, timeout:int=25):
    """以 GET 方式访问 url，返回 (status_code, body)。

    - 网络层错误（URLError/TimeoutError）返回 (0, 错误描述)，方便下游断言；
    - HTTPError 会把实际状态码和响应体返回，不视为异常；
    - 统一带上 User-Agent，避免部分站点拒绝默认 UA。
    """
    req=request.Request(url, method='GET', headers={'User-Agent':'cloudpaw-testcase/1.0'})
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return int(r.getcode()), r.read().decode('utf-8','ignore')
    except error.HTTPError as e:
        return int(e.code), e.read().decode('utf-8','ignore')
    except (error.URLError, TimeoutError) as e:
        return 0, f'urlopen_error: {type(e).__name__}: {e}'

def http_head(url:str, timeout:int=25):
    """以 HEAD 方式访问 url，仅返回 status_code。

    - 仅需探活 / 校验状态码时使用，避免下载 body；
    - 网络错误统一返回 0，HTTPError 返回对应状态码。
    """
    req=request.Request(url, method='HEAD', headers={'User-Agent':'cloudpaw-testcase/1.0'})
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return int(r.getcode())
    except error.HTTPError as e:
        return int(e.code)
    except (error.URLError, TimeoutError):
        return 0

def _run_local_cmd(cmd:str, timeout:int=90):
    """在本地评测机执行 shell 命令，返回 (returncode, stdout, stderr)。

    使用 shlex.split 解析命令，避免 shell 注入；超时由调用方控制，
    超时会直接抛出 subprocess.TimeoutExpired。

    命令不存在（容器缺 iproute2/docker/psql 等）时映射为 rc=127，
    与 shell 的 "command not found" 行为一致，避免整个 test_case
    脚本因 FileNotFoundError 直接崩溃而丢失结构化 checks。
    """
    try:
        p=subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or '', p.stderr or ''
    except FileNotFoundError as e:
        missing = e.filename or (shlex.split(cmd)[:1] or [''])[0]
        return 127, '', f'command not found: {missing}'


def _extract_invoke_id(text: str) -> str:
    """从 RunCommand 的返回 JSON 文本中提取 InvokeId。

    使用正则以容忍 aliyun CLI 可能输出的非严格 JSON/多余日志，
    未匹配到时返回空串，由调用方做错误处理。
    """
    m = re.search(r'"InvokeId"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else ""


def _b64decode_text(s: str) -> str:
    """安全地对 base64 文本进行解码。

    ECS RunCommand 返回的 Output 为 base64 编码的 stdout，
    非 base64 或解码异常时原样返回，避免吞掉已可读的文本。
    """
    if not s:
        return ""
    try:
        return base64.b64decode(s).decode("utf-8", "ignore")
    except Exception:
        return s


def _parse_invocation_result(payload: str):
    """解析 DescribeInvocations 的返回，抽取关键执行状态字段。

    返回 dict 包含：invoke_status / invocation_status / instance_status /
    exit_code / stdout / stderr。解析失败或无 Invocation 记录时返回 None。
    其中 stdout 会自动对 base64 Output 进行解码。
    """
    try:
        data = json.loads(payload)
    except Exception:
        return None
    invs = data.get("Invocations", {}).get("Invocation", [])
    if not invs:
        return None
    inv = invs[0] or {}
    invoke_status = str(inv.get("InvokeStatus", ""))
    invocation_status = str(inv.get("InvocationStatus", ""))
    instances = inv.get("InvokeInstances", {}).get("InvokeInstance", [])
    inst = (instances[0] if instances else {}) or {}
    instance_status = str(inst.get("InstanceInvokeStatus", ""))
    exit_code = inst.get("ExitCode")
    out_text = _b64decode_text(str(inst.get("Output", "")))
    err_text = str(inst.get("ErrorInfo", "")) or str(inst.get("OssOutputErrorInfo", ""))
    return {
        "invoke_status": invoke_status,
        "invocation_status": invocation_status,
        "instance_status": instance_status,
        "exit_code": exit_code,
        "stdout": out_text,
        "stderr": err_text,
    }


def _run_cmd_via_ecs(cmd: str, timeout: int = 90):
    """通过 aliyun ecs RunCommand 在目标 ECS 实例内执行 shell 命令。

    流程：
      1. 将命令 base64 编码后调用 RunCommand 下发；
      2. 解析出 InvokeId；
      3. 轮询 DescribeInvocations，直到 Invocation 与 Instance 都 Finished；
      4. 返回 (returncode, stdout, stderr)，其中 stdout 已做 base64 解码。
    前置条件：模块级 _ECS_REGION_ID / _ECS_INSTANCE_ID 必须已设置，
    否则返回 (2, '', error_msg)。轮询超时返回 124。
    """
    if not _ECS_REGION_ID or not _ECS_INSTANCE_ID:
        return 2, "", "EXEC_ENV_TYPE=ECS but ECS_REGION_ID/ECS_INSTANCE_ID missing"

    content_b64 = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    submit = (
        f"aliyun ecs RunCommand --RegionId {_ECS_REGION_ID} --InstanceId.1 {_ECS_INSTANCE_ID} "
        f"--Type RunShellScript --ContentEncoding Base64 --CommandContent {content_b64} "
        f"--Timeout {timeout}"
    )
    rc, out, err = _run_local_cmd(submit, timeout=min(120, timeout + 30))
    if rc != 0:
        return rc, out, err
    invoke_id = _extract_invoke_id(out)
    if not invoke_id:
        return 2, out, "RunCommand missing InvokeId"

    # Poll invocation state and collect output.
    query = (
        f"aliyun ecs DescribeInvocations --RegionId {_ECS_REGION_ID} --InvokeId {invoke_id} "
        f"--IncludeOutput true --PageSize 1"
    )
    deadline = time.time() + max(30, timeout + 60)
    last_out = ""
    last_err = ""
    while time.time() < deadline:
        qrc, qout, qerr = _run_local_cmd(query, timeout=60)
        last_out, last_err = qout, qerr
        if qrc != 0:
            time.sleep(2)
            continue
        parsed = _parse_invocation_result(qout)
        if not parsed:
            time.sleep(2)
            continue
        # Wait until command really finished on instance.
        if parsed["invoke_status"] != "Finished" or parsed["instance_status"] != "Finished":
            time.sleep(2)
            continue
        exit_code = parsed["exit_code"]
        if isinstance(exit_code, int):
            rc = exit_code
        else:
            rc = 0 if parsed["invocation_status"] == "Success" else 1
        stdout = parsed["stdout"]
        stderr = parsed["stderr"]
        return rc, stdout, stderr
        time.sleep(2)
    return 124, last_out, last_err or "DescribeInvocations timeout"


def run_cmd(cmd:str, timeout:int=90):
    """统一的命令执行入口，根据 EXEC_ENV_TYPE 自动路由本地或 ECS。

    路由规则：
      - EXEC_ENV_TYPE == 'ECS' 且命令不是以 'aliyun ' 开头时，
        通过 RunCommand 在目标 ECS 内执行（数据面校验）；
      - 其他情况（含所有 aliyun 云控制面调用、LOCAL 模式）
        统一在本地评测机执行。
    返回值与 _run_local_cmd 一致：(returncode, stdout, stderr)。
    """
    # Keep cloud control-plane calls local. Execute data-plane shell checks in ECS when requested.
    if _EXEC_ENV_TYPE == "ECS" and not cmd.strip().startswith("aliyun "):
        return _run_cmd_via_ecs(cmd, timeout=timeout)
    return _run_local_cmd(cmd, timeout=timeout)


def extract_last_json(text: str):
    """从可能含噪音前缀的 stdout 中提取最后一个可解析的 JSON。

    场景：container 注入了 loongsuite site bootstrap 等装飾，
    被测脚本启动时会先打 INFO 日志到 stdout，紧接其后才是真正
    的 JSON 输出。直接 json.loads(整串) 会失败；该工具会：
      1) 先尝试整串（干净 stdout 下的乐观路径）；
      2) 按行从上往下逐步找 JSON 起点，直到找到可解析的后缀。
    解析失败时抛 ValueError（由调用方 try/except 转 json_ok=False）。
    """
    if not text:
        raise ValueError("empty text")
    stripped = text.rstrip()
    # 乐观路径：整串已是 JSON
    try:
        return json.loads(stripped)
    except Exception:
        pass
    lines = stripped.split('\n')
    for i in range(len(lines)):
        candidate = '\n'.join(lines[i:]).strip()
        if not candidate or candidate[0] not in '{[':
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("no valid JSON object/array found in text")

def dump_report(task_id, checks, result_file='final_eval_result.json', extra=None):
    """汇总 checks 并输出最终评测报告。

    - 所有 check 全部 passed 才视为任务通过；
    - 报告字段包含：task_id / passed / checks / exec_env_used /
      evidence_source（local_runner 或 ecs_runcommand）；
    - extra 用于追加任务特有字段（如 base_url 等）；
    - 结果会同时打印到 stdout 并写入 result_file；
    - 返回 0 表示通过，1 表示失败，供脚本作为进程退出码使用。
    """
    passed = all(c.passed for c in checks)
    report={
        'task_id':task_id,
        'passed':passed,
        'checks':[c.__dict__ for c in checks],
        'exec_env_used': _EXEC_ENV_TYPE,
        'evidence_source': 'ecs_runcommand' if _EXEC_ENV_TYPE == 'ECS' else 'local_runner',
    }
    if extra: report.update(extra)
    text=json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if result_file:
        with open(result_file,'w',encoding='utf-8') as f: f.write(text+'\n')
    return 0 if passed else 1
