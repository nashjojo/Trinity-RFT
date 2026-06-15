#!/usr/bin/env python3
"""Task simple_059 的销毁前功能验证脚本（增量备份按 sha256）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part3.py

校验项：
  - delta_exists: DELTA 存在
  - delta_correct: 差异集合一致
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_059'
EXPECTED_ANCHOR = 'CPW-SIMPLE-059'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_dir_exists('delta_exists', (f.get('DELTA_DIR') or "").strip()))

    import os as _o, hashlib as _h
    _s=(f.get("SRC_DIR") or "").strip(); _b=(f.get("BAK_DIR") or "").strip(); _d=(f.get("DELTA_DIR") or "").strip()
    _ok=False; _det=""
    try:
        def _scan(d):
            o={}
            for r,_,fs in _o.walk(d):
                rel=_o.path.relpath(r,d)
                for n in fs:
                    rp=n if rel=="." else _o.path.join(rel,n)
                    o[rp]=_h.sha256(open(_o.path.join(r,n),"rb").read()).hexdigest()
            return o
        _S=_scan(_s); _B=_scan(_b); _D=_scan(_d)
        _exp={k for k,v in _S.items() if _B.get(k)!=v}
        _ok=(set(_D.keys())==_exp and all(_D[k]==_S[k] for k in _D))
        _det=f"expected={_exp} actual={set(_D.keys())}"
    except Exception as _ex: _det=f"err={_ex!r}"
    checks.append(check_predicate("delta_correct", _ok, _det))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
