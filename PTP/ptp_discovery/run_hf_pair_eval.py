from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, Mapping


if __package__ is None or __package__ == "":
    ptp_discovery_dir = os.path.dirname(os.path.abspath(__file__))
    ptp_root = os.path.abspath(os.path.join(ptp_discovery_dir, ".."))
    repo_root = os.path.abspath(os.path.join(ptp_root, ".."))
    for path in (repo_root, ptp_root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

from ptp_discovery.pref_loss_coevo_loop import _evaluate_pair_worker
from ptp_discovery.runtime_trace import RuntimeTrace


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one stage3 HF pair evaluation in an isolated subprocess.")
    p.add_argument("--payload", required=True, type=str)
    p.add_argument("--result", required=True, type=str)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    task_dir = os.path.dirname(os.path.abspath(str(args.result))) or os.getcwd()
    trace = RuntimeTrace(
        os.path.join(task_dir, "runtime_status.json"),
        role="hf_pair_worker",
        heartbeat_interval_s=float(os.environ.get("HF_RUNTIME_TRACE_HEARTBEAT_S", "15")),
    )
    trace.start(
        extra={
            "payload_path": os.path.abspath(str(args.payload)),
            "result_path": os.path.abspath(str(args.result)),
        }
    )
    trace.install_signal_handlers()
    payload: Dict[str, Any] = {}
    try:
        payload_raw = _load_json(str(args.payload))
        if not isinstance(payload_raw, dict):
            raise ValueError(f"Invalid payload JSON (expected dict): {args.payload}")

        payload = dict(payload_raw)
        trace.heartbeat(
            extra={
                "generation": payload.get("generation"),
                "pair_index": payload.get("pair_index"),
                "g_id": payload.get("g_id"),
                "f_id": payload.get("f_id"),
                "device": payload.get("device_physical_str", payload.get("device_str")),
                "stage": "evaluate_pair_worker",
            },
            force=True,
        )
        try:
            record = _evaluate_pair_worker(payload)
        except Exception as exc:  # noqa: BLE001
            fixed = dict(payload)
            physical_device = str(fixed.get("device_physical_str") or fixed.get("device_str") or "")
            fixed["device"] = physical_device
            fixed["device_str"] = physical_device
            fixed["pair_ok"] = False
            fixed["pair_reason"] = "child_exception"
            fixed["high_fidelity_error"] = f"{type(exc).__name__}: {exc}"
            fixed["high_fidelity_traceback"] = traceback.format_exc()
            fixed["score"] = float("inf")
            record = fixed
            trace.fail(
                reason="child_exception",
                exc=exc,
                exit_code=1,
                extra={
                    "generation": payload.get("generation"),
                    "pair_index": payload.get("pair_index"),
                    "g_id": payload.get("g_id"),
                    "f_id": payload.get("f_id"),
                },
            )
        else:
            trace.finish(
                reason="completed",
                exit_code=0,
                extra={
                    "pair_ok": bool(record.get("pair_ok")),
                    "pair_reason": record.get("pair_reason"),
                    "score": record.get("score"),
                    "generation": payload.get("generation"),
                    "pair_index": payload.get("pair_index"),
                    "g_id": payload.get("g_id"),
                    "f_id": payload.get("f_id"),
                },
            )
        _atomic_write_json(str(args.result), dict(record))
        return 0
    except Exception as exc:  # noqa: BLE001
        fixed = dict(payload) if isinstance(payload, dict) else {}
        physical_device = str(fixed.get("device_physical_str") or fixed.get("device_str") or "")
        fixed["device"] = physical_device
        fixed["device_str"] = physical_device
        fixed["pair_ok"] = False
        fixed["pair_reason"] = "child_bootstrap_exception"
        fixed["high_fidelity_error"] = f"{type(exc).__name__}: {exc}"
        fixed["high_fidelity_traceback"] = traceback.format_exc()
        fixed["score"] = float("inf")
        _atomic_write_json(str(args.result), fixed)
        trace.fail(reason="child_bootstrap_exception", exc=exc, exit_code=1)
        return 0
    finally:
        trace.close()


if __name__ == "__main__":
    raise SystemExit(main())
