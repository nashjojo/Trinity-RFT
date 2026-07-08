#!/usr/bin/env python3
"""Task simple_070 的销毁前功能验证脚本（凯撒密码加解密）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part4.py

校验项：
  - cipher_exists: CIPHER 存在
  - caesar_correct: 移位/恢复正确
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_070'
EXPECTED_ANCHOR = 'CPW-SIMPLE-070'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_file_exists('cipher_exists', (f.get('CIPHER') or "").strip()))

    _p=(f.get("PLAIN") or "").strip(); _c=(f.get("CIPHER") or "").strip(); _r=(f.get("RECOVERED") or "").strip()
    _k=int((f.get("SHIFT") or "0").strip() or 0)
    _ok=False; _det=""
    def _ca(s,k):
        o=[]
        for ch in s:
            if ch.isupper(): o.append(chr((ord(ch)-65+k)%26+65))
            elif ch.islower(): o.append(chr((ord(ch)-97+k)%26+97))
            else: o.append(ch)
        return "".join(o)
    try:
        _pl=open(_p).read(); _ci=open(_c).read(); _re=open(_r).read()
        _ok=(_ci==_ca(_pl,_k) and _re==_pl); _det=f"cipher_match={_ci==_ca(_pl,_k)} recovered_match={_re==_pl}"
    except Exception as _ex: _det=f"err={_ex!r}"
    checks.append(check_predicate("caesar_correct", _ok, _det))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
