#!/usr/bin/env python3
"""test_case_simple 的公共辅助入口。

为避免代码重复，这里将同级 test_case_use/common.py 的公共符号
（CheckResult / parse_yaml_or_output / run_cmd / http_get / http_head /
norm_url / dump_report 等）整体再导出，供本目录下的各个
test_case_simple_XXX.py 直接 `from common import *` 使用。

单层沙箱任务全部为 LOCAL 执行，因此不需要 ECS RunCommand 相关辅助；
但为保持与 test_case_use 一致的风格，仍沿用同一套 API。
"""
from __future__ import annotations
import os
import sys

_USE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_case_use")
if _USE_DIR not in sys.path:
    sys.path.insert(0, _USE_DIR)

# 从 test_case_use/common.py 重新导出所有符号
# 注意：必须先将模块注册到 sys.modules，否则 dataclass 在构造时
# 通过 sys.modules[cls.__module__] 回查会失败（AttributeError: 'NoneType'...）。
import importlib.util

_MOD_NAME = "_tcuse_common"
_spec = importlib.util.spec_from_file_location(
    _MOD_NAME, os.path.join(_USE_DIR, "common.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _mod
_spec.loader.exec_module(_mod)  # type: ignore

# 将 _mod 中的公开符号注入到本模块命名空间
for _name in dir(_mod):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_mod, _name)
