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
    payload_raw = _load_json(str(args.payload))
    if not isinstance(payload_raw, dict):
        raise ValueError(f"Invalid payload JSON (expected dict): {args.payload}")

    payload = dict(payload_raw)
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

    _atomic_write_json(str(args.result), dict(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
