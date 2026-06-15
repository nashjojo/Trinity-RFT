#!/usr/bin/env python3
"""Task simple_085 的销毁前功能验证脚本（asyncio TCP echo）。

本文件由 _extended/build.py 自动生成，请勿手工修改。
spec source: examples/copaw_rl/queries_simple/_extended/specs_part5.py

校验项：
  - port_listen: 端口监听
  - output_has_ping: OUTPUT_FILE 含 ping CPW
返回退出码：0 全部通过 / 1 至少一项未通过。
"""
from __future__ import annotations
import argparse
from common import *
from helpers import *

TASK_ID = 'simple_085'
EXPECTED_ANCHOR = 'CPW-SIMPLE-085'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml-file", default="")
    ap.add_argument("--output-file", default="")
    ap.add_argument("--result-file", default="final_eval_result.json")
    a = ap.parse_args()
    f = parse_yaml_or_output(a.yaml_file, a.output_file)
    checks = []
    checks.append(check_anchor(f, EXPECTED_ANCHOR))
    checks.append(check_port_listening('port_listen', int((f.get('SERVER_PORT') or "").strip() or '0')))
    checks.append(check_file_contains('output_has_ping', (f.get('OUTPUT_FILE') or "").strip(), 'ping CPW'))
    return dump_report(TASK_ID, checks, a.result_file, _extra(f))


def _extra(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("TASK_ID",)}


if __name__ == "__main__":
    raise SystemExit(main())
