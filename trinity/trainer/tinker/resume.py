"""TuFT cross-session resume helper.

Backport from /mnt/workspace/kaixiang/TuFT/examples/cross_session_resume_demo.py
(2026-06-07 解决 Trinity 与 TuFT 的 sequence conflict 问题).

Background
----------
默认 ``ServiceClient.create_training_client_from_state_with_optimizer_async``
路径会撞 ``tinker.ConflictError: Sequence conflict``：

    1. 内部走 ``create_lora_training_client + load_state_with_optimizer``，先
       创建一个**新** ``training_run_id``（next_seq_id=1）
    2. 但 ``load_checkpoint`` 在 server 端把 sequence guard 套到 path 解析出的
       **老** ``training_run_id`` 上，期望 seq_id == old.next_seq_id（远大于 1）

正确的恢复方式（TuFT 端 2026-06-07 上线）：让 caller 直接以**老
training_run_id 作为自己的 model_id**，所有 RPC 的 sequence guard 都套在同一
record 上，client counter 与 server next_seq_id 自然同步。Adam (m, v) state
已由 TuFT 端 ``_restore_from_checkpoints`` 在 server 启动时加载到 backend
GPU lora slot，client 接管后立即可用。

Reference
---------
- TuFT demo: ``/mnt/workspace/kaixiang/TuFT/examples/cross_session_resume_demo.py``
- 设计文档: ``docs/2026-06-07_TuFT_resume_checkpoint.md`` §8
"""

from __future__ import annotations

import re
from typing import Tuple

from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.lib.public_interfaces.training_client import TrainingClient

# Regex: tinker://<uuid>/weights/<name>  →  group(1) = <uuid> = training_run_id
_URI_RUN_ID_RE = re.compile(r"^tinker://([0-9a-fA-F-]{36})/")
_EXPECTED_SEQ_RE = re.compile(r"expected (\d+)")


def parse_training_run_id_from_uri(weight_uri: str) -> str:
    """从 tinker URI 抽出 training_run_id (UUID 部分)。"""
    m = _URI_RUN_ID_RE.match(weight_uri)
    if m is None:
        raise ValueError(
            f"failed to parse training_run_id from URI: {weight_uri!r}; "
            "expected form 'tinker://<uuid>/weights/<name>'"
        )
    return m.group(1)


def probe_expected_next_seq_id(
    sc: ServiceClient,
    training_run_id: str,
    weight_uri: str,
    *,
    probe_seq: int = 999,
) -> int:
    """无副作用 probe：故意发一个肯定大于当前 next_seq_id 的 seq_id 触发
    ConflictError，从 detail 抠出 server 端的真实 ``next_seq_id``。

    sequence guard 在 ``_operation`` 之前检查，conflict 时 next_seq_id 不会被
    推进，所以 probe 完全无副作用。

    如果 server 端 next_seq_id 已经超过 ``probe_seq``，会抛 RuntimeError——把
    ``probe_seq`` 调高即可。
    """
    probe = TrainingClient(holder=sc.holder, model_seq_id=0, model_id=training_run_id)
    probe._request_id_counter = probe_seq - 1  # _get_request_id() 返回 998 → seq_id=999
    probe._turn_counter = probe_seq - 1
    try:
        probe.load_state_with_optimizer(weight_uri).result(timeout=30)
    except Exception as exc:  # noqa: BLE001
        m = _EXPECTED_SEQ_RE.search(str(exc))
        if m is None:
            raise RuntimeError(
                f"probe failed to extract `expected` from error: {exc}"
            ) from exc
        return int(m.group(1))
    raise RuntimeError(
        f"probe should have raised ConflictError but succeeded; "
        f"is server's next_seq_id really > {probe_seq}? "
        "raise probe_seq and retry."
    )


def take_over_training_client(
    sc: ServiceClient,
    weight_uri: str,
    *,
    training_run_id: str | None = None,
) -> Tuple[TrainingClient, int]:
    """构造一个接管已存在 training_run 的 ``TrainingClient``。

    内部步骤：
      1. 从 URI 抽出 training_run_id（如未显式提供）
      2. probe 出 server 端当前 next_seq_id
      3. 手工构造 TrainingClient(model_id=老 training_run_id)
      4. patch _request_id_counter / _turn_counter，使下一次 RPC 发出的
         seq_id 与 server next_seq_id 对齐

    返回 ``(client, expected_next_seq_id)``。返回的 client 可以直接用
    ``forward_backward`` / ``optim_step`` / ``save_state`` /
    ``load_state_with_optimizer``。
    """
    if training_run_id is None:
        training_run_id = parse_training_run_id_from_uri(weight_uri)

    expected = probe_expected_next_seq_id(sc, training_run_id, weight_uri)
    patch = expected - 1
    client = TrainingClient(holder=sc.holder, model_seq_id=0, model_id=training_run_id)
    client._request_id_counter = patch
    client._turn_counter = patch
    return client, expected
