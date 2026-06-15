#!/usr/bin/env python3
"""共享 test_case 校验原语。

simple_019..100 的 test_case_simple_XXX.py 通过 `from helpers import *` 复用以下
原语，避免大量样板代码。每个原语接受字段字典 f 和必要参数，返回 CheckResult。

设计目标：
- 每个原语都"参数化 + 防御式"：传入路径不存在/解析失败时给清晰 detail，不抛异常；
- detail 字段尽量给出实际值与期望值，便于 E2B 验证脚本失败时定位；
- 仅依赖 stdlib + common.py 中已导出的 CheckResult / parse_yaml_or_output 等符号。
"""
from __future__ import annotations
import csv as _csv
import hashlib
import json as _json
import os
import re
import socket
from typing import Callable, Iterable

from common import CheckResult, run_cmd  # noqa: F401


# ---------------------------------------------------------------------------
# anchor / 路径基础
# ---------------------------------------------------------------------------
def check_anchor(f: dict, expected: str) -> CheckResult:
    """VERIFICATION_ANCHOR == expected。"""
    actual = (f.get("VERIFICATION_ANCHOR") or "").strip()
    return CheckResult(
        "anchor_ok",
        actual == expected,
        f"anchor={actual!r}, expected={expected!r}",
    )


def check_file_exists(name: str, path: str) -> CheckResult:
    ok = bool(path) and os.path.isfile(path)
    return CheckResult(name, ok, f"path={path!r}, exists={ok}")


def check_dir_exists(name: str, path: str) -> CheckResult:
    ok = bool(path) and os.path.isdir(path)
    return CheckResult(name, ok, f"path={path!r}, exists={ok}")


def check_file_size_min(name: str, path: str, min_bytes: int) -> CheckResult:
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}")
    size = os.path.getsize(path)
    return CheckResult(
        name, size >= min_bytes, f"size={size}, min={min_bytes}"
    )


def check_file_lines_min(name: str, path: str, min_lines: int) -> CheckResult:
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            n = sum(1 for _ in fh)
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"read fail: {e!r}")
    return CheckResult(name, n >= min_lines, f"lines={n}, min={min_lines}")


def check_file_contains(
    name: str, path: str, keyword: str, encoding: str = "utf-8"
) -> CheckResult:
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}")
    try:
        body = open(path, "r", encoding=encoding, errors="ignore").read()
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"read fail: {e!r}")
    return CheckResult(
        name, keyword in body,
        f"keyword={keyword!r} found={keyword in body}, body_len={len(body)}",
    )


def check_files_byte_eq(name: str, a: str, b: str) -> CheckResult:
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return CheckResult(name, False, f"missing: a={os.path.isfile(a)} b={os.path.isfile(b)}")
    try:
        ah = hashlib.sha256(open(a, "rb").read()).hexdigest()
        bh = hashlib.sha256(open(b, "rb").read()).hexdigest()
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"hash fail: {e!r}")
    return CheckResult(name, ah == bh, f"a_sha256={ah[:16]}.. b_sha256={bh[:16]}.., equal={ah==bh}")


# ---------------------------------------------------------------------------
# CSV 校验
# ---------------------------------------------------------------------------
def _read_csv(path: str) -> list[list[str]]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        return list(_csv.reader(fh))


def check_csv_header_eq(name: str, path: str, cols: list[str]) -> CheckResult:
    """表头与 cols 完全一致（顺序、大小写）。"""
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}")
    try:
        rows = _read_csv(path)
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"read fail: {e!r}")
    header = [c.strip() for c in rows[0]] if rows else []
    return CheckResult(name, header == cols, f"header={header}, expected={cols}")


def check_csv_header_contains(name: str, path: str, cols: Iterable[str]) -> CheckResult:
    """表头（小写）包含 cols 中所有字段。"""
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}")
    try:
        rows = _read_csv(path)
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"read fail: {e!r}")
    header = [c.strip().lower() for c in rows[0]] if rows else []
    missing = [c for c in cols if c.lower() not in header]
    return CheckResult(
        name, not missing,
        f"header={header}, missing={missing}",
    )


def check_csv_rows_eq(name: str, path: str, n_data: int) -> CheckResult:
    """数据行数（不含表头）== n_data。"""
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}")
    try:
        rows = _read_csv(path)
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"read fail: {e!r}")
    actual = max(0, len(rows) - 1)
    return CheckResult(
        name, actual == n_data, f"data_rows={actual}, expected={n_data}",
    )


def check_csv_data_subset(
    name: str,
    path: str,
    expected_rows: list[dict],
    key_cols: list[str] | None = None,
) -> CheckResult:
    """csv 数据行（按 dict 解析）必须以集合方式覆盖 expected_rows。"""
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
            actual = list(_csv.DictReader(fh))
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"read fail: {e!r}")
    if key_cols is None:
        key_cols = sorted(expected_rows[0].keys()) if expected_rows else []

    def keyfn(r):
        return tuple((c, str(r.get(c, "")).strip()) for c in key_cols)

    actual_keys = {keyfn(r) for r in actual}
    expected_keys = {keyfn(r) for r in expected_rows}
    missing = expected_keys - actual_keys
    return CheckResult(
        name, not missing,
        f"missing_rows={len(missing)}, expect={len(expected_keys)}, actual={len(actual_keys)}",
    )


# ---------------------------------------------------------------------------
# JSON 校验
# ---------------------------------------------------------------------------
def check_json_valid(name: str, path: str) -> tuple[CheckResult, object]:
    if not (path and os.path.isfile(path)):
        return CheckResult(name, False, f"missing: {path!r}"), None
    try:
        data = _json.loads(open(path, "r", encoding="utf-8").read())
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"parse fail: {e!r}"), None
    return CheckResult(name, True, f"type={type(data).__name__}"), data


def check_json_array_min(name: str, path: str, min_len: int) -> CheckResult:
    res, data = check_json_valid(name + "_valid", path)
    if not res.passed:
        return CheckResult(name, False, res.detail)
    if not isinstance(data, list):
        return CheckResult(name, False, f"not array, type={type(data).__name__}")
    return CheckResult(
        name, len(data) >= min_len, f"len={len(data)}, min={min_len}",
    )


def check_json_dict_has(name: str, path: str, keys: list[str]) -> CheckResult:
    res, data = check_json_valid(name + "_valid", path)
    if not res.passed:
        return CheckResult(name, False, res.detail)
    if not isinstance(data, dict):
        return CheckResult(name, False, f"not dict, type={type(data).__name__}")
    missing = [k for k in keys if k not in data]
    return CheckResult(name, not missing, f"missing_keys={missing}")


# ---------------------------------------------------------------------------
# 服务/网络
# ---------------------------------------------------------------------------
def check_port_listening(name: str, port: int, host: str = "127.0.0.1") -> CheckResult:
    try:
        with socket.create_connection((host, int(port)), timeout=3) as s:
            s.close()
        ok = True
        detail = f"{host}:{port} reachable"
    except Exception as e:  # noqa: BLE001
        ok = False
        detail = f"{host}:{port} unreachable: {e!r}"
    return CheckResult(name, ok, detail)


def check_pgrep(name: str, pattern: str) -> CheckResult:
    rc, out, _ = run_cmd(f"pgrep -fa {pattern}", timeout=5)
    return CheckResult(
        name, rc == 0 and bool(out.strip()),
        f"pattern={pattern!r}, rc={rc}, pids_head={out.strip()[:80]!r}",
    )


# ---------------------------------------------------------------------------
# Hash / 编解码
# ---------------------------------------------------------------------------
def sha256_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_sha256_file(name: str, payload: str, checksum_path: str) -> CheckResult:
    """checksum_path 第一行包含的 64 hex 必须 == sha256(payload)。"""
    if not (os.path.isfile(payload) and os.path.isfile(checksum_path)):
        return CheckResult(
            name, False,
            f"missing: payload={os.path.isfile(payload)}, checksum={os.path.isfile(checksum_path)}",
        )
    try:
        first = open(checksum_path, "r", encoding="utf-8").read().strip().splitlines()[0]
        m = re.search(r"[0-9a-fA-F]{64}", first)
        if not m:
            return CheckResult(name, False, f"no 64-hex in line: {first!r}")
        recorded = m.group(0).lower()
        actual = sha256_of(payload)
        return CheckResult(
            name, recorded == actual,
            f"recorded={recorded[:16]}.. actual={actual[:16]}.., equal={recorded==actual}",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(name, False, f"err: {e!r}")


# ---------------------------------------------------------------------------
# 数值/字符串 utility
# ---------------------------------------------------------------------------
def to_float(v) -> float | None:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_predicate(name: str, ok: bool, detail: str = "") -> CheckResult:
    return CheckResult(name, bool(ok), detail)
