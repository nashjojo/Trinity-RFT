#!/usr/bin/env python3
"""Task simple_019 的销毁前功能验证脚本（CSV 双条件过滤）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part1.py

校验项：
  - input_csv_exists: INPUT_CSV 存在
  - input_csv_header: 表头与规格一致
  - input_csv_lines: 行数 ≥ 9
  - output_csv_exists: OUTPUT_CSV 存在
  - output_csv_correct: OUTPUT == filter(INPUT)
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_019'
EXPECTED_ANCHOR = 'CPW-SIMPLE-019'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_file_exists('input_csv_exists', (f.get('INPUT_CSV') or "").strip()))
    checks.append(check_csv_header_eq('input_csv_header', (f.get('INPUT_CSV') or "").strip(), ['name', 'age', 'city', 'score']))
    checks.append(check_file_lines_min('input_csv_lines', (f.get('INPUT_CSV') or "").strip(), 9))
    checks.append(check_file_exists('output_csv_exists', (f.get('OUTPUT_CSV') or "").strip()))

    import csv as _c
    _ic=(f.get("INPUT_CSV") or "").strip(); _oc=(f.get("OUTPUT_CSV") or "").strip()
    _ok=False; _det=""
    try:
        _rows=list(_c.DictReader(open(_ic)))
        _exp=[(r["name"],r["age"],r["city"],r["score"]) for r in _rows if int(r.get("age","0"))>30 and r.get("city","").strip()=="Beijing"]
        _act=[(r["name"],r["age"],r["city"],r["score"]) for r in _c.DictReader(open(_oc))]
        _ok=(_exp==_act); _det=f"expected={_exp}; actual={_act}"
    except Exception as _ex:
        _det=f"err={_ex!r}"
    checks.append(check_predicate("output_csv_correct", _ok, _det))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
