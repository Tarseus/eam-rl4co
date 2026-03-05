from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict, Mapping

import torch


if __package__ is None or __package__ == "":
    ptp_discovery_dir = os.path.dirname(os.path.abspath(__file__))
    ptp_root = os.path.abspath(os.path.join(ptp_discovery_dir, ".."))
    repo_root = os.path.abspath(os.path.join(ptp_root, ".."))
    for path in (repo_root, ptp_root):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

from fitness.free_loss_fidelity import extract_feature_cache
from ptp_discovery.free_loss_compiler import compile_free_loss
from ptp_discovery.free_loss_gates import run_joint_preference_gates, run_preference_builder_gates
from ptp_discovery.free_loss_ir import ir_from_json as free_loss_ir_from_json
from ptp_discovery.pref_builder_compiler import compile_preference_builder
from ptp_discovery.pref_builder_ir import ir_from_json as pref_builder_ir_from_json


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
    p = argparse.ArgumentParser(description="Run stage0 sandbox gate for one candidate pair.")
    p.add_argument("--payload", required=True, type=str)
    p.add_argument("--result", required=True, type=str)
    return p.parse_args(argv)


def _dummy_feature_cache(*, batch_size: int, k: int, variant: str, round_idx: int) -> Dict[str, torch.Tensor]:
    b = max(1, int(batch_size))
    kk = max(2, int(k))
    idx = torch.arange(kk, dtype=torch.float32)[None, :].repeat(b, 1)
    objective = idx + (torch.arange(b, dtype=torch.float32)[:, None] * 0.01)
    log_prob = -0.1 * idx
    if str(variant) == "hidden":
        objective = objective * 10.0
        log_prob = log_prob * 12.0
    if int(round_idx) > 0:
        scale = float(1.0 + 0.5 * int(round_idx))
        objective = objective * scale
        log_prob = log_prob * scale
    return extract_feature_cache(objective, log_prob)


def _fail(*, generation: int, pair_index: int, failure_kind: str, reason: str, trace: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "generation": int(generation),
        "pair_index": int(pair_index),
        "failure_kind": str(failure_kind),
        "reason": str(reason),
    }
    if isinstance(trace, Mapping):
        out["trace"] = dict(trace)
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Force sandbox to stay off CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    payload: Dict[str, Any] = {}
    try:
        raw = _load_json(str(args.payload))
        if not isinstance(raw, dict):
            raise ValueError("Payload must be a dict")
        payload = dict(raw)

        generation = int(payload.get("generation", -1))
        pair_index = int(payload.get("pair_index", -1))
        g_entry = payload.get("g_entry")
        f_entry = payload.get("f_entry")
        if not isinstance(g_entry, dict) or not isinstance(f_entry, dict):
            raise ValueError("Payload missing g_entry/f_entry")
        if not isinstance(g_entry.get("ir"), dict) or not isinstance(f_entry.get("ir"), dict):
            raise ValueError("Payload missing g_entry.ir/f_entry.ir")

        operator_whitelist_raw = payload.get("operator_whitelist", [])
        operator_whitelist = list(operator_whitelist_raw) if isinstance(operator_whitelist_raw, list) else []

        g_ir = pref_builder_ir_from_json(g_entry["ir"])
        f_ir = free_loss_ir_from_json(f_entry["ir"])
        compiled_g = compile_preference_builder(g_ir, operator_whitelist=operator_whitelist)
        compiled_f = compile_free_loss(f_ir, operator_whitelist=operator_whitelist)

        builder_cfg = payload.get("builder_gate", {}) if isinstance(payload.get("builder_gate"), dict) else {}
        joint_cfg = payload.get("joint_gate", {}) if isinstance(payload.get("joint_gate"), dict) else {}

        batch_size = max(1, int(payload.get("batch_size", 8) or 8))
        k = max(2, int(payload.get("k", 16) or 16))
        rounds = max(1, int(payload.get("rounds", 1) or 1))
        variants_raw = payload.get("variants", ["visible", "hidden"])
        if isinstance(variants_raw, (list, tuple)):
            variants = [str(v) for v in variants_raw if str(v).strip()]
        else:
            variants = [str(variants_raw)]
        if not variants:
            variants = ["visible", "hidden"]
        hard_failure_kinds_raw = payload.get("hard_failure_kinds", [])
        if isinstance(hard_failure_kinds_raw, (list, tuple, set)):
            hard_failure_kinds = {str(x) for x in hard_failure_kinds_raw if str(x).strip()}
        else:
            hard_failure_kinds = set()

        checks: list[Dict[str, Any]] = []
        for variant in variants:
            variant_s = str(variant).strip().lower() or "visible"
            for ridx in range(rounds):
                feature_cache = _dummy_feature_cache(
                    batch_size=int(batch_size),
                    k=int(k),
                    variant=variant_s,
                    round_idx=int(ridx),
                )
                pref_batch = compiled_g.build_fn(feature_cache, {"stage": "sandbox_gate"})
                bg = run_preference_builder_gates(
                    pref_batch,
                    feature_cache=feature_cache,
                    min_pairs=int(builder_cfg.get("min_pairs", 1) or 1),
                    min_coverage=float(builder_cfg.get("min_coverage", 0.0) or 0.0),
                    max_pairs_per_instance=int(builder_cfg.get("max_pairs_per_instance", 4096) or 4096),
                    weight_nonneg=bool(builder_cfg.get("weight_nonneg", True)),
                    semantic_tolerance=float(builder_cfg.get("semantic_tolerance", 0.0) or 0.0),
                    semantic_min_pass_rate=float(builder_cfg.get("semantic_min_pass_rate", 1.0) or 1.0),
                )
                if not bool(bg.ok):
                    out = _fail(
                        generation=generation,
                        pair_index=pair_index,
                        failure_kind="sandbox_builder_gate_failed",
                        reason=f"builder gate failed: {bg.reason}",
                        trace=bg.trace,
                    )
                    _atomic_write_json(str(args.result), out)
                    return 0

                jg = run_joint_preference_gates(
                    compiled_f,
                    pref_batch=pref_batch,
                    feature_cache=feature_cache,
                    min_pass_rate=float(joint_cfg.get("min_pass_rate", 0.8) or 0.8),
                    swap_tolerance=float(joint_cfg.get("swap_tolerance", 1e-3) or 1e-3),
                    swap_check_mode=str(joint_cfg.get("swap_check_mode", "data") or "data"),
                    swap_test_margin=float(joint_cfg.get("swap_test_margin", 1.0) or 1.0),
                    grad_eps=float(joint_cfg.get("grad_eps", 1e-8) or 1e-8),
                    min_effective_grad_ratio=float(joint_cfg.get("min_effective_grad_ratio", 0.1) or 0.1),
                    numeric_stress_enabled=bool(joint_cfg.get("numeric_stress_enabled", True)),
                    numeric_stress_margin=float(joint_cfg.get("numeric_stress_margin", 120.0) or 120.0),
                    numeric_stress_aux_scale=float(joint_cfg.get("numeric_stress_aux_scale", 32.0) or 32.0),
                    variant=variant_s,
                )
                checks.append(
                    {
                        "variant": str(variant_s),
                        "round": int(ridx),
                        "builder_gate_ok": bool(bg.ok),
                        "joint_gate_ok": bool(jg.ok),
                        "joint_failure_kind": None
                        if not isinstance(jg.trace, dict)
                        else jg.trace.get("failure_kind"),
                    }
                )
                if not bool(jg.ok):
                    failure_kind = None
                    if isinstance(jg.trace, dict):
                        failure_kind = jg.trace.get("failure_kind")
                    if not failure_kind:
                        failure_kind = "sandbox_joint_gate_failed"
                    if str(failure_kind) in hard_failure_kinds:
                        out = _fail(
                            generation=generation,
                            pair_index=pair_index,
                            failure_kind=str(failure_kind),
                            reason=str(jg.reason),
                            trace=jg.trace if isinstance(jg.trace, dict) else None,
                        )
                        out["checks"] = checks
                        _atomic_write_json(str(args.result), out)
                        return 0

        out_ok = {
            "ok": True,
            "generation": int(generation),
            "pair_index": int(pair_index),
            "failure_kind": None,
            "reason": "ok",
            "checks": checks,
        }
        _atomic_write_json(str(args.result), out_ok)
        return 0
    except Exception as exc:  # noqa: BLE001
        out = _fail(
            generation=int(payload.get("generation", -1)) if isinstance(payload, dict) else -1,
            pair_index=int(payload.get("pair_index", -1)) if isinstance(payload, dict) else -1,
            failure_kind="sandbox_runtime_error",
            reason=f"{type(exc).__name__}: {exc}",
            trace={
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        _atomic_write_json(str(args.result), out)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
