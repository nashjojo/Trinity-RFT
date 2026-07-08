#!/usr/bin/env python3
"""Task simple_094 的销毁前功能验证脚本（stdlib HTTP 多 path 路由）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part5.py

校验项：
  - port_listen: 端口监听
  - routes_correct: 三路由 200/200/404
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_094'
EXPECTED_ANCHOR = 'CPW-SIMPLE-094'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_port_listening('port_listen', int((f.get('SITE_PORT') or "").strip() or '0')))

    from urllib.request import urlopen as _uo
    from urllib.error import HTTPError as _HE
    _p=(f.get("SITE_PORT") or "").strip()
    _ok=False; _det=""
    try:
        _a=_uo(f"http://127.0.0.1:{_p}/a/", timeout=3); _ab=_a.read().decode()
        _b=_uo(f"http://127.0.0.1:{_p}/b/", timeout=3); _bb=_b.read().decode()
        _cc=0
        try: _uo(f"http://127.0.0.1:{_p}/c/", timeout=3)
        except _HE as _e: _cc=_e.code
        _ok=(_a.status==200 and "CPW-A-94" in _ab and _b.status==200 and "CPW-B-94" in _bb and _cc==404)
        _det=f"a={_ab[:60]!r} b={_bb[:60]!r} c_code={_cc}"
    except Exception as _ex: _det=f"err={_ex!r}"
    checks.append(check_predicate("routes_correct", _ok, _det))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
