#!/usr/bin/env python3
"""Task simple_075 的销毁前功能验证脚本（多关键字一次替换）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part4.py

校验项：
  - output_exists: OUTPUT 存在
  - replace_correct: 替换结果与重算一致
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_075'
EXPECTED_ANCHOR = 'CPW-SIMPLE-075'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_file_exists('output_exists', (f.get('OUTPUT_TXT') or "").strip()))

    import json as _j, re as _r
    _i=(f.get("INPUT_TXT") or "").strip(); _ru=(f.get("RULES_JSON") or "").strip(); _o=(f.get("OUTPUT_TXT") or "").strip()
    _ok=False; _det=""
    try:
        _txt=open(_i).read(); _rules=_j.loads(open(_ru).read())
        _keys=sorted(_rules.keys(), key=lambda k:-len(k))
        _pat=_r.compile("|".join(_r.escape(k) for k in _keys))
        _exp=_pat.sub(lambda m: _rules[m.group(0)], _txt)
        _act=open(_o).read()
        _ok=(_exp==_act); _det=f"len_exp={len(_exp)} len_act={len(_act)}"
    except Exception as _ex: _det=f"err={_ex!r}"
    checks.append(check_predicate("replace_correct", _ok, _det))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
