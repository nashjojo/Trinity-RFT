#!/usr/bin/env python3
"""Task simple_098 的销毁前功能验证脚本（sqlite KV 存储）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part5.py

校验项：
  - db_size: DB 存在
  - value_match: OUTPUT == CPW-VAL-98
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_098'
EXPECTED_ANCHOR = 'CPW-SIMPLE-098'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_file_size_min('db_size', (f.get('DB_FILE') or "").strip(), 100))

    import sqlite3 as _s
    _o=(f.get("OUTPUT_TXT") or "").strip(); _d=(f.get("DB_FILE") or "").strip()
    try:
        _v=open(_o).read().rstrip("\n")
        _row=_s.connect(_d).execute("SELECT v FROM kv WHERE k=?", ("cpw:simple:98",)).fetchone()
        _ok=(_v=="CPW-VAL-98" and _row and _row[0]=="CPW-VAL-98")
        _det=f"output={_v!r} db_value={_row[0] if _row else None!r}"
    except Exception as _ex: _ok=False; _det=f"err={_ex!r}"
    checks.append(check_predicate("value_match", _ok, _det))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
