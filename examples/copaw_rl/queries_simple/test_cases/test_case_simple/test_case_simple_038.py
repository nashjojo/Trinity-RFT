#!/usr/bin/env python3
"""Task simple_038 的销毁前功能验证脚本（字母频次直方图）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part2.py

校验项：
  - header: 表头规范
  - rows: 26 行
  - freq_correct: 频次与重算一致
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_038'
EXPECTED_ANCHOR = 'CPW-SIMPLE-038'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_csv_header_eq('header', (f.get('OUTPUT_CSV') or "").strip(), ['letter', 'count']))
    checks.append(check_csv_rows_eq('rows', (f.get('OUTPUT_CSV') or "").strip(), 26))

    import csv as _c
    from collections import Counter as _Co
    _i=(f.get("INPUT_TXT") or "").strip(); _o=(f.get("OUTPUT_CSV") or "").strip()
    _ok=False; _det=""
    try:
        _t=open(_i).read().lower(); _exp=_Co(ch for ch in _t if ch.isalpha())
        _rows=list(_c.DictReader(open(_o)))
        _act={r["letter"]:int(r["count"]) for r in _rows}
        _seq=[r["letter"] for r in _rows]
        _ok=(_seq==list("abcdefghijklmnopqrstuvwxyz") and all(_exp.get(ch,0)==_act.get(ch,0) for ch in _act))
        _det=f"alpha_total={sum(_exp.values())} sample={dict(list(_act.items())[:5])}"
    except Exception as _ex: _det=f"err={_ex!r}"
    checks.append(check_predicate("freq_correct", _ok, _det))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
