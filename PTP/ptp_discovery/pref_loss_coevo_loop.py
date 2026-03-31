from __future__ import annotations

"""Co-evolution loop for preference builders (g) and preference losses (f)."""

import base64
import ast
import collections
import gc
import io
import json
import logging
import math
import os
import pickle
import random
import re
import subprocess
import sys
import time
import traceback
import tokenize
from dataclasses import asdict
from hashlib import sha1
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import yaml

from fitness.free_loss_fidelity import (
    FreeLossFidelityConfig,
    PrefBatch,
    PrefBuilder,
    baseline_epoch_objectives_from_metrics_csv,
    extract_feature_cache,
    evaluate_free_loss_candidate,
    evaluate_po_baseline_rl4co,
)
from fitness.ptp_high_fidelity import (
    HighFidelityConfig,
    _set_seed,
    get_hf_epoch_plan,
    get_total_hf_train_steps,
    resolve_pomo_size,
)
from fitness.pref_loss_fidelity import (
    PrefLossEvalCaches,
    aggregate_proxy_metrics,
    build_or_get_rollout_feature_cache,
    eval_budget_signature,
    load_pair_cache_from_pairs_jsonl,
    micro_unroll_score_for_pair,
    proxy_metrics_for_pair_on_batch,
    seed_signature_for_proxy,
)
from ptp_discovery.free_loss_compiler import CompiledFreeLoss, CompileError, compile_free_loss
from ptp_discovery.free_loss_gates import (
    AffineInvarianceGateResult,
    JointPreferenceGateResult,
    ObjectiveSensitivityGateResult,
    StaticGateResult,
    run_affine_invariance_gate,
    run_joint_preference_gates,
    run_objective_sensitivity_gate,
    run_preference_builder_gates,
    run_preference_semantic_gates,
    run_static_gates,
)
from ptp_discovery.free_loss_ir import (
    FreeLossIR,
    FreeLossImplementationHint,
    ir_from_json as free_loss_ir_from_json,
)
from ptp_discovery.pref_builder_compiler import (
    CompiledPreferenceBuilder,
    PreferenceBuilderCompileError,
    compile_preference_builder,
)
from ptp_discovery.pref_builder_ir import (
    PreferenceBuilderIR,
    PreferenceBuilderImplementationHint,
    ir_from_json as pref_builder_ir_from_json,
)

import ptp_discovery.free_loss_llm_ops as loss_llm_ops
import ptp_discovery.pref_builder_llm_ops as builder_llm_ops
from ptp_discovery.runtime_trace import RuntimeTrace


LOGGER = logging.getLogger("ptp_discovery.pref_loss_coevo")


def _cache_brief(caches: PrefLossEvalCaches) -> str:
    try:
        return f"caches(rollout={len(caches.rollout_cache)} pref={len(caches.pref_cache)} pair={len(caches.pair_cache)})"
    except Exception:  # noqa: BLE001
        return "caches(?)"


def _cuda_mem_brief(devices: Sequence[str]) -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    out: list[str] = []
    for d in devices:
        ds = str(d)
        if not ds.startswith("cuda"):
            continue
        try:
            dev = torch.device(ds)
            alloc_gb = float(torch.cuda.memory_allocated(dev)) / (1024**3)
            reserv_gb = float(torch.cuda.memory_reserved(dev)) / (1024**3)
            out.append(f"{ds}(alloc={alloc_gb:.2f}G,resv={reserv_gb:.2f}G)")
        except Exception:  # noqa: BLE001
            continue
    return "cuda(" + " ".join(out) + ")" if out else "cuda=ok"


def _normalize_device_alias(device_str: str) -> str:
    ds = str(device_str or "").strip()
    if ds.lower() == "gpu":
        return "cuda"
    return ds


def _maybe_auto_flush_pref_cache(
    *,
    caches: PrefLossEvalCaches,
    cfg_yaml: Mapping[str, Any],
    default_device_str: str,
    generation: int,
    scope: str,
) -> int:
    if not bool(cfg_yaml.get("pref_cache_auto_flush_enabled", False)):
        return 0
    thr_gb = float(cfg_yaml.get("pref_cache_auto_flush_reserved_gb", 0.0) or 0.0)
    if thr_gb <= 0:
        return 0
    dev_s = _normalize_device_alias(str(cfg_yaml.get("pref_cache_auto_flush_device", default_device_str)))
    if not dev_s.startswith("cuda") or not torch.cuda.is_available():
        return 0

    try:
        dev = torch.device(dev_s)
        alloc_gb = float(torch.cuda.memory_allocated(dev)) / (1024**3)
        reserv_gb = float(torch.cuda.memory_reserved(dev)) / (1024**3)
        used_gb = max(alloc_gb, reserv_gb)
    except Exception:  # noqa: BLE001
        return 0

    if used_gb < float(thr_gb):
        return 0

    dropped_pref = caches.prune_pref_cache(keep_g_ids=[], keep_batch_ids=[])
    dropped_rollout = 0
    if bool(cfg_yaml.get("pref_cache_auto_flush_drop_rollout", False)):
        dropped_rollout = int(len(caches.rollout_cache))
        caches.rollout_cache.clear()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    LOGGER.warning(
        "Auto-flush pref cache gen=%d scope=%s device=%s used=%.2fG threshold=%.2fG dropped(pref=%d rollout=%d) %s %s",
        int(generation),
        str(scope),
        str(dev_s),
        float(used_gb),
        float(thr_gb),
        int(dropped_pref),
        int(dropped_rollout),
        _cache_brief(caches),
        _cuda_mem_brief([dev_s]),
    )
    return int(dropped_pref + dropped_rollout)


def _repo_root_dir() -> str:
    # This file lives at PTP/ptp_discovery/pref_loss_coevo_loop.py.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _abs_from_repo_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root_dir(), path))


_FILE_SHA1_CACHE: Dict[str, str] = {}
_BASELINE_MINI_EVAL_CACHE: Dict[str, Dict[str, Any]] = {}
_STAGE3_BASELINE_MULTI_SEED_CACHE: Dict[str, Dict[str, Any]] = {}


def _file_sha1_cached(path: str, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    p = _abs_from_repo_root(str(path))
    cached = _FILE_SHA1_CACHE.get(p)
    if cached is not None:
        return str(cached)
    h = sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(int(chunk_size))
            if not b:
                break
            h.update(b)
    out = h.hexdigest()
    _FILE_SHA1_CACHE[p] = out
    return str(out)


def _load_baseline_mini_eval(path: str) -> Dict[str, Any]:
    p = _abs_from_repo_root(str(path))
    cached = _BASELINE_MINI_EVAL_CACHE.get(p)
    if isinstance(cached, dict):
        return cached
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid baseline mini-eval JSON (expected dict): {path}")
    _BASELINE_MINI_EVAL_CACHE[p] = dict(payload)
    return dict(payload)


def _resolve_training_seed(cfg_yaml: Mapping[str, Any], *, default: int = 1234) -> int:
    raw = cfg_yaml.get("scratch_init_seed", None)
    if raw is not None:
        try:
            return int(raw)
        except Exception:  # noqa: BLE001
            pass
    raw = cfg_yaml.get("seed", default)
    try:
        return int(raw)
    except Exception:  # noqa: BLE001
        return int(default)


def _alpha_from_cfg(cfg_yaml: Mapping[str, Any], *, key: str = "alpha") -> float:
    env_name = str(cfg_yaml.get("env_name") or cfg_yaml.get("problem") or "tsp").strip().lower()
    default_alpha = 0.03 if env_name == "cvrp" else 0.05
    return float(cfg_yaml.get(key, default_alpha) or default_alpha)


def _po_impl_from_cfg(cfg_yaml: Mapping[str, Any], *, default: str = "bt") -> str:
    raw = str(cfg_yaml.get("po_impl", default) or default).strip().lower()
    if raw not in {"bt", "exponential"}:
        return str(default)
    return raw


def _stage3_multiseed_compare_cfg(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    seed0 = _resolve_training_seed(cfg_yaml)
    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    if not isinstance(baseline_cfg, Mapping):
        baseline_cfg = {}
    compare_raw = baseline_cfg.get("multiseed_compare", {}) or {}
    if not isinstance(compare_raw, Mapping):
        compare_raw = {}
    calib_raw = cfg_yaml.get("improve_eps_calibration", {}) or {}
    if not isinstance(calib_raw, Mapping):
        calib_raw = {}

    enabled = bool(compare_raw.get("enabled", baseline_cfg.get("multiseed_compare_enabled", False)))
    n_seeds = int(compare_raw.get("n_seeds", compare_raw.get("N", 0)) or 0)
    seed_stride = int(compare_raw.get("seed_stride", 997) or 997)
    seed0_out = int(compare_raw.get("seed0", seed0) or seed0)

    if bool(calib_raw.get("enabled", False)):
        enabled = True if not enabled else bool(enabled)
        if n_seeds <= 0:
            n_seeds = int(calib_raw.get("N", 0) or 0)
        if "seed0" not in compare_raw:
            seed0_out = int(seed0) + 999
        if "seed_stride" not in compare_raw:
            seed_stride = int(calib_raw.get("seed_stride", seed_stride) or seed_stride)

    if not enabled:
        return {
            "enabled": False,
            "n_seeds": 1,
            "seed0": int(seed0),
            "seed_stride": 0,
        }

    n_seeds = max(1, min(int(n_seeds or 1), 128))
    if int(seed_stride) == 0:
        seed_stride = 997
    return {
        "enabled": True,
        "n_seeds": int(n_seeds),
        "seed0": int(seed0_out),
        "seed_stride": int(seed_stride),
    }


def _stage3_baseline_multiseed_cache_root_dir() -> str:
    return os.path.join(_repo_root_dir(), "baseline", "stage3_multiseed")


def _stage3_baseline_multiseed_cache_key(
    cfg_yaml: Mapping[str, Any],
    *,
    include_scratch: bool,
    seed0: int,
    seed_stride: int,
) -> str:
    payload = {
        "eval_signature": _build_stage3_eval_signature(cfg_yaml),
        "include_scratch": bool(include_scratch),
        "seed0": int(seed0),
        "seed_stride": int(seed_stride),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = sha1(blob).hexdigest()[:16]
    return f"{_stage3_fidelity_key(cfg_yaml)}__{digest}"


def _stage3_baseline_multiseed_cache_path(
    cfg_yaml: Mapping[str, Any],
    *,
    include_scratch: bool,
    seed0: int,
    seed_stride: int,
) -> str:
    key = _stage3_baseline_multiseed_cache_key(
        cfg_yaml,
        include_scratch=bool(include_scratch),
        seed0=int(seed0),
        seed_stride=int(seed_stride),
    )
    return os.path.join(_stage3_baseline_multiseed_cache_root_dir(), str(key), "summary.json")


def _load_stage3_baseline_multiseed_cache(path: str) -> Dict[str, Any] | None:
    p = _abs_from_repo_root(str(path))
    cached = _STAGE3_BASELINE_MULTI_SEED_CACHE.get(p)
    if isinstance(cached, dict):
        return dict(cached)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    _STAGE3_BASELINE_MULTI_SEED_CACHE[p] = dict(payload)
    return dict(payload)


def _write_stage3_baseline_multiseed_cache(path: str, payload: Mapping[str, Any]) -> None:
    p = _abs_from_repo_root(str(path))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    _atomic_write_json(p, dict(payload))
    _STAGE3_BASELINE_MULTI_SEED_CACHE[p] = dict(payload)


def _aggregate_stage3_baseline_multiseed_records(
    *,
    per_init_base: Mapping[str, Any],
    per_seed: Mapping[str, Any],
) -> Dict[str, Any]:
    best_per_init: Dict[str, Any] = {}
    samples: List[float] = []

    for seed_key, seed_payload in per_seed.items():
        if not isinstance(seed_payload, Mapping):
            continue
        seed_per_init = seed_payload.get("per_init")
        if not isinstance(seed_per_init, Mapping):
            continue

        deltas: List[float] = []
        for init_name, base_entry in per_init_base.items():
            if not isinstance(base_entry, Mapping):
                continue
            seed_entry = seed_per_init.get(str(init_name))
            if not isinstance(seed_entry, Mapping):
                continue

            try:
                cand_agg = float(seed_entry.get("aggregated_objective"))
                base_agg = float(base_entry.get("aggregated_objective"))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(cand_agg) and math.isfinite(base_agg)):
                continue

            deltas.append(float(cand_agg - base_agg))

            current_best = best_per_init.get(str(init_name))
            if (
                not isinstance(current_best, Mapping)
                or float(current_best.get("aggregated_objective", float("inf"))) > float(cand_agg)
            ):
                best_per_init[str(init_name)] = {
                    "seed": int(seed_payload.get("seed", int(seed_key))),
                    "aggregated_objective": float(cand_agg),
                    "val_objective_by_size": dict(seed_entry.get("val_objective_by_size", {}) or {}),
                    "source": "multiseed_best",
                }

        if deltas:
            samples.append(float(sum(deltas) / float(len(deltas))))

    return {
        "best_per_init": best_per_init,
        "samples": samples,
    }


def _resolve_stage3_baseline_reference_entry(
    init_name: str,
    *,
    per_init_base: Mapping[str, Any],
    multiseed_cache: Mapping[str, Any] | None,
) -> Tuple[Dict[str, Any] | None, str]:
    best_map = None if not isinstance(multiseed_cache, Mapping) else multiseed_cache.get("best_per_init")
    if isinstance(best_map, Mapping):
        best_entry = best_map.get(str(init_name))
        if isinstance(best_entry, Mapping):
            try:
                best_agg = float(best_entry.get("aggregated_objective"))
            except (TypeError, ValueError):
                best_agg = float("inf")
            if math.isfinite(best_agg):
                return dict(best_entry), "multiseed_best"

    base_entry = per_init_base.get(str(init_name))
    if isinstance(base_entry, Mapping):
        return dict(base_entry), "mini_eval"
    return None, "missing"


def _ensure_stage3_baseline_multiseed_cache(
    *,
    cfg_yaml: Mapping[str, Any],
    operator_whitelist: Sequence[str],
    device_str: str,
    n_seeds: int | None = None,
    seed0: int | None = None,
    seed_stride: int | None = None,
) -> Dict[str, Any] | None:
    compare_cfg = _stage3_multiseed_compare_cfg(cfg_yaml)
    if n_seeds is None:
        n_seeds = int(compare_cfg.get("n_seeds", 0) or 0)
    if seed0 is None:
        seed0 = int(compare_cfg.get("seed0", 0) or 0)
    if seed_stride is None:
        seed_stride = int(compare_cfg.get("seed_stride", 997) or 997)

    n_seeds = max(0, min(int(n_seeds or 0), 128))
    if n_seeds < 2:
        return None
    if int(seed_stride) == 0:
        seed_stride = 997

    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    if not isinstance(baseline_cfg, Mapping):
        baseline_cfg = {}
    mini_eval_path = _resolve_stage3_baseline_mini_eval_path(cfg_yaml, baseline_cfg)
    if not mini_eval_path:
        LOGGER.warning(
            "stage3 multiseed baseline skipped: baseline mini_eval_path missing for fidelity=%s",
            _stage3_fidelity_key(cfg_yaml),
        )
        return None

    try:
        baseline_payload = _load_baseline_mini_eval(str(mini_eval_path))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "stage3 multiseed baseline skipped: failed to load baseline mini_eval_path=%s: %s",
            str(mini_eval_path),
            str(exc),
        )
        return None

    per_init_base = baseline_payload.get("per_init")
    if not isinstance(per_init_base, Mapping):
        LOGGER.warning(
            "stage3 multiseed baseline skipped: baseline JSON missing per_init dict: %s",
            str(mini_eval_path),
        )
        return None

    expected_sig = _build_stage3_eval_signature(cfg_yaml)
    got_sig = baseline_payload.get("eval_signature")
    if got_sig != expected_sig:
        LOGGER.warning(
            "stage3 multiseed baseline skipped: baseline eval_signature mismatch (fidelity=%s).",
            _stage3_fidelity_key(cfg_yaml),
        )
        return None

    init_specs = _stage3_init_specs_from_baseline_cfg(cfg_yaml)
    if not init_specs:
        LOGGER.warning(
            "stage3 multiseed baseline skipped: no init sources configured (need scratch and/or baseline.checkpoints)."
        )
        return None
    include_scratch = bool(any(init_ckpt is None for _, init_ckpt in init_specs))

    cache_path = _stage3_baseline_multiseed_cache_path(
        cfg_yaml,
        include_scratch=bool(include_scratch),
        seed0=int(seed0),
        seed_stride=int(seed_stride),
    )
    cached_payload = _load_stage3_baseline_multiseed_cache(cache_path)
    per_seed: Dict[str, Any] = {}
    if isinstance(cached_payload, Mapping):
        cached_sig = cached_payload.get("eval_signature")
        cached_include_scratch = bool(cached_payload.get("include_scratch", include_scratch))
        cached_seed0 = int(cached_payload.get("seed0", seed0) or seed0)
        cached_seed_stride = int(cached_payload.get("seed_stride", seed_stride) or seed_stride)
        cached_per_seed = cached_payload.get("per_seed")
        if (
            cached_sig == expected_sig
            and cached_include_scratch == bool(include_scratch)
            and cached_seed0 == int(seed0)
            and cached_seed_stride == int(seed_stride)
            and isinstance(cached_per_seed, Mapping)
        ):
            per_seed = {str(k): dict(v) for k, v in cached_per_seed.items() if isinstance(v, Mapping)}

    required_seeds = [int(seed0 + i * int(seed_stride)) for i in range(int(n_seeds))]
    missing_seeds = [int(s) for s in required_seeds if str(int(s)) not in per_seed]

    if missing_seeds:
        valid_sizes = [int(v) for v in cfg_yaml.get("valid_problem_sizes", [100])]
        if not valid_sizes:
            valid_sizes = [int(cfg_yaml.get("train_problem_size", 20) or 20)]
        valid_sizes = list(dict.fromkeys([int(v) for v in valid_sizes]))

        LOGGER.info(
            "Stage3 multiseed baseline cache miss fidelity=%s missing_seeds=%d path=%s",
            _stage3_fidelity_key(cfg_yaml),
            int(len(missing_seeds)),
            os.path.abspath(_abs_from_repo_root(cache_path)),
        )
        for seed_i in missing_seeds:
            hf_cfg = _build_hf_cfg(dict(cfg_yaml), seed=int(seed_i), device_str=str(device_str))
            seed_per_init: Dict[str, Any] = {}
            for init_name, init_ckpt in init_specs:
                try:
                    fit = _evaluate_stage3_reference_baseline(
                        cfg_yaml=cfg_yaml,
                        hf_cfg=hf_cfg,
                        operator_whitelist=operator_whitelist,
                        init_ckpt=init_ckpt,
                    )
                    by_size, cand_agg = _extract_stage3_size_objectives(fit, valid_sizes=valid_sizes)
                    cand_by_size = {str(int(sz)): float(by_size[int(sz)]) for sz in valid_sizes}
                    if not math.isfinite(cand_agg):
                        raise RuntimeError("Non-finite reference aggregated objective")
                    seed_per_init[str(init_name)] = {
                        "val_objective_by_size": cand_by_size,
                        "aggregated_objective": float(cand_agg),
                        "error": None,
                    }
                except Exception as exc:  # noqa: BLE001
                    seed_per_init[str(init_name)] = {
                        "val_objective_by_size": {},
                        "aggregated_objective": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            per_seed[str(int(seed_i))] = {
                "seed": int(seed_i),
                "per_init": seed_per_init,
            }

    aggregated = _aggregate_stage3_baseline_multiseed_records(
        per_init_base=per_init_base,
        per_seed=per_seed,
    )
    payload = {
        "schema_version": 1,
        "created_at": (
            str(cached_payload.get("created_at"))
            if isinstance(cached_payload, Mapping) and cached_payload.get("created_at")
            else time.strftime("%Y-%m-%d %H:%M:%S")
        ),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fidelity": _stage3_fidelity_key(cfg_yaml),
        "baseline_mini_eval_path": str(mini_eval_path),
        "cache_path": str(cache_path),
        "eval_signature": expected_sig,
        "include_scratch": bool(include_scratch),
        "seed0": int(seed0),
        "seed_stride": int(seed_stride),
        "n_seeds": int(n_seeds),
        "per_seed": per_seed,
        "best_per_init": aggregated.get("best_per_init", {}),
        "samples": list(aggregated.get("samples", [])),
    }
    _write_stage3_baseline_multiseed_cache(cache_path, payload)
    return dict(payload)


def _load_stage3_baseline_multiseed_cache_for_cfg(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any] | None:
    compare_cfg = _stage3_multiseed_compare_cfg(cfg_yaml)
    if not bool(compare_cfg.get("enabled", False)):
        return None

    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    if not isinstance(baseline_cfg, Mapping):
        baseline_cfg = {}
    cache_path = _stage3_baseline_multiseed_cache_path(
        cfg_yaml,
        include_scratch=bool(baseline_cfg.get("include_scratch", True)),
        seed0=int(compare_cfg.get("seed0", 0) or 0),
        seed_stride=int(compare_cfg.get("seed_stride", 997) or 997),
    )
    payload = _load_stage3_baseline_multiseed_cache(cache_path)
    if not isinstance(payload, Mapping):
        return None
    expected_sig = _build_stage3_eval_signature(cfg_yaml)
    if payload.get("eval_signature") != expected_sig:
        return None
    return dict(payload)


def _stage3_scenario_name_from_cfg(cfg_yaml: Mapping[str, Any]) -> str:
    explicit = cfg_yaml.get("stage3_scenario_name")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    problem = str(cfg_yaml.get("problem") or cfg_yaml.get("env_name") or "tsp").strip().lower()
    train_problem_size = int(cfg_yaml.get("train_problem_size", 20) or 20)
    return f"{problem}{int(train_problem_size)}"


def _stage3_slug(value: Any) -> str:
    raw = str(value or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return slug or "item"


def _stage3_init_specs_from_baseline_cfg(cfg_yaml: Mapping[str, Any]) -> List[Tuple[str, str | None]]:
    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    if not isinstance(baseline_cfg, Mapping):
        baseline_cfg = {}

    include_scratch = bool(baseline_cfg.get("include_scratch", True))
    ckpts = baseline_cfg.get("checkpoints") or []
    if not isinstance(ckpts, Sequence) or isinstance(ckpts, (str, bytes)):
        ckpts = []

    init_specs: List[Tuple[str, str | None]] = []
    if include_scratch:
        init_specs.append(("scratch", None))

    used_names = {str(name) for name, _ in init_specs}
    for idx, ckpt in enumerate(ckpts):
        if ckpt is None or not str(ckpt).strip():
            continue
        ckpt_s = str(ckpt)
        epoch = _infer_baseline_epoch_from_path(ckpt_s)
        if epoch is not None:
            init_name = f"ckpt_{int(epoch)}"
        else:
            stem = os.path.splitext(os.path.basename(ckpt_s))[0]
            init_name = f"ckpt_{_stage3_slug(stem)}"
        if init_name in used_names:
            stem = os.path.splitext(os.path.basename(ckpt_s))[0]
            init_name = f"ckpt_{int(idx):03d}_{_stage3_slug(stem)}"
        suffix = 2
        base_name = str(init_name)
        while init_name in used_names:
            init_name = f"{base_name}_{int(suffix)}"
            suffix += 1
        used_names.add(str(init_name))
        init_specs.append((str(init_name), ckpt_s))

    return init_specs


def _apply_stage3_scenario_overrides(cfg_yaml: Mapping[str, Any], scenario_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(cfg_yaml)

    baseline_out = dict(cfg_yaml.get("baseline", {}) or {}) if isinstance(cfg_yaml.get("baseline", {}) or {}, Mapping) else {}
    generator_out = dict(cfg_yaml.get("generator_params", {}) or {}) if isinstance(cfg_yaml.get("generator_params", {}) or {}, Mapping) else {}
    env_kwargs_out = dict(cfg_yaml.get("env_kwargs", {}) or {}) if isinstance(cfg_yaml.get("env_kwargs", {}) or {}, Mapping) else {}
    policy_kwargs_out = dict(cfg_yaml.get("policy_kwargs", {}) or {}) if isinstance(cfg_yaml.get("policy_kwargs", {}) or {}, Mapping) else {}

    baseline_shorthand_keys = {
        "checkpoints",
        "mini_eval_path",
        "mini_eval_paths",
        "include_scratch",
        "multiseed_compare_enabled",
        "multiseed_compare",
    }
    generator_shorthand_keys = {"offline_train_path", "offline_val_paths"}

    for key, value in dict(scenario_cfg).items():
        if key in {"name", "baseline", "generator_params", "env_kwargs", "policy_kwargs"}:
            continue
        if key in baseline_shorthand_keys:
            baseline_out[str(key)] = value
            continue
        if key in generator_shorthand_keys:
            generator_out[str(key)] = value
            continue
        out[str(key)] = value

    baseline_extra = scenario_cfg.get("baseline", {}) or {}
    if isinstance(baseline_extra, Mapping):
        baseline_out.update(dict(baseline_extra))
    baseline_out.pop("scenarios", None)

    generator_extra = scenario_cfg.get("generator_params", {}) or {}
    if isinstance(generator_extra, Mapping):
        generator_out.update(dict(generator_extra))

    env_kwargs_extra = scenario_cfg.get("env_kwargs", {}) or {}
    if isinstance(env_kwargs_extra, Mapping):
        env_kwargs_out.update(dict(env_kwargs_extra))

    policy_kwargs_extra = scenario_cfg.get("policy_kwargs", {}) or {}
    if isinstance(policy_kwargs_extra, Mapping):
        policy_kwargs_out.update(dict(policy_kwargs_extra))

    out["baseline"] = baseline_out
    out["generator_params"] = generator_out
    out["env_kwargs"] = env_kwargs_out
    out["policy_kwargs"] = policy_kwargs_out
    out["stage3_scenario_name"] = str(
        scenario_cfg.get("name") or _stage3_scenario_name_from_cfg(out)
    )
    return out


def _iter_stage3_scenario_cfgs(cfg_yaml: Mapping[str, Any]) -> List[Dict[str, Any]]:
    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    if not isinstance(baseline_cfg, Mapping):
        baseline_cfg = {}

    raw_scenarios = baseline_cfg.get("scenarios") or []
    if not isinstance(raw_scenarios, list):
        raw_scenarios = []

    if not raw_scenarios:
        single_cfg = dict(cfg_yaml)
        single_baseline = dict(baseline_cfg)
        single_baseline.pop("scenarios", None)
        single_cfg["baseline"] = single_baseline
        single_cfg["stage3_scenario_name"] = _stage3_scenario_name_from_cfg(single_cfg)
        return [{"name": str(single_cfg["stage3_scenario_name"]), "cfg": single_cfg}]

    scenarios: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_scenarios):
        if not isinstance(raw, Mapping):
            continue
        scenario_cfg = _apply_stage3_scenario_overrides(cfg_yaml, raw)
        name = str(raw.get("name") or scenario_cfg.get("stage3_scenario_name") or f"scenario_{int(idx)}")
        scenario_cfg["stage3_scenario_name"] = name
        scenarios.append({"name": name, "cfg": scenario_cfg})

    if scenarios:
        return scenarios

    single_cfg = dict(cfg_yaml)
    single_baseline = dict(baseline_cfg)
    single_baseline.pop("scenarios", None)
    single_cfg["baseline"] = single_baseline
    single_cfg["stage3_scenario_name"] = _stage3_scenario_name_from_cfg(single_cfg)
    return [{"name": str(single_cfg["stage3_scenario_name"]), "cfg": single_cfg}]


def _stage3_early_prune_cfg(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    raw = cfg_yaml.get("stage3_early_prune", {}) or {}
    if not isinstance(raw, Mapping):
        return {"enabled": False}

    try:
        max_init_ratio = float(raw.get("max_init_ratio_to_baseline", float("inf")))
    except (TypeError, ValueError):
        max_init_ratio = float("inf")
    try:
        max_mean_ratio = float(raw.get("max_mean_ratio_to_baseline", float("inf")))
    except (TypeError, ValueError):
        max_mean_ratio = float("inf")

    return {
        "enabled": bool(raw.get("enabled", False)),
        "scenario_name": str(raw.get("scenario_name") or "tsp50"),
        "max_init_ratio_to_baseline": float(max_init_ratio),
        "max_mean_ratio_to_baseline": float(max_mean_ratio),
    }


def _stage3_check_early_prune(
    *,
    cfg_yaml: Mapping[str, Any],
    scenario_name: str,
    scenario_per_init: Mapping[str, Any],
) -> Dict[str, Any] | None:
    prune_cfg = _stage3_early_prune_cfg(cfg_yaml)
    if not bool(prune_cfg.get("enabled", False)):
        return None
    if str(scenario_name) != str(prune_cfg.get("scenario_name") or ""):
        return None

    max_init_ratio = float(prune_cfg.get("max_init_ratio_to_baseline", float("inf")))
    max_mean_ratio = float(prune_cfg.get("max_mean_ratio_to_baseline", float("inf")))
    init_ratios: Dict[str, float] = {}
    triggered_inits: Dict[str, float] = {}

    for init_name, init_record in dict(scenario_per_init or {}).items():
        if not isinstance(init_record, Mapping):
            continue
        try:
            obj_cand = float(init_record.get("obj_cand"))
            obj_base = float(init_record.get("obj_base"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(obj_cand) and math.isfinite(obj_base) and obj_base > 0.0):
            continue
        ratio = float(obj_cand / obj_base)
        init_ratios[str(init_name)] = float(ratio)
        if math.isfinite(max_init_ratio) and ratio > max_init_ratio:
            triggered_inits[str(init_name)] = float(ratio)

    mean_ratio = float(sum(init_ratios.values()) / float(len(init_ratios))) if init_ratios else float("nan")
    trigger_mean = bool(math.isfinite(max_mean_ratio) and math.isfinite(mean_ratio) and mean_ratio > max_mean_ratio)
    if not triggered_inits and not trigger_mean:
        return None

    return {
        "scenario_name": str(scenario_name),
        "init_ratios": {str(k): float(v) for k, v in init_ratios.items()},
        "triggered_inits": {str(k): float(v) for k, v in triggered_inits.items()},
        "mean_ratio": float(mean_ratio) if math.isfinite(mean_ratio) else None,
        "max_init_ratio_to_baseline": float(max_init_ratio),
        "max_mean_ratio_to_baseline": float(max_mean_ratio) if math.isfinite(max_mean_ratio) else None,
    }


def _build_stage3_eval_signature(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    generator_params = cfg_yaml.get("generator_params", {}) or {}

    offline_train = generator_params.get("offline_train_path")
    offline_val_paths = generator_params.get("offline_val_paths") or {}
    init_specs = _stage3_init_specs_from_baseline_cfg(cfg_yaml)

    uses_offline_data = bool(offline_train or offline_val_paths)
    if uses_offline_data and (not offline_train or not offline_val_paths):
        raise ValueError(
            "stage3 offline mode requires both offline_train_path and offline_val_paths in generator_params"
        )
    if not init_specs:
        raise ValueError("stage3 requires at least one init source (scratch and/or baseline.checkpoints)")

    env_name = str(cfg_yaml.get("env_name") or cfg_yaml.get("problem") or "tsp")
    baseline_eval_mode = _stage3_baseline_eval_mode(cfg_yaml)
    policy_name = str(cfg_yaml.get("policy_name") or "")
    policy_kwargs = dict(cfg_yaml.get("policy_kwargs", {}) or {})
    env_kwargs = dict(cfg_yaml.get("env_kwargs", {}) or {})
    rollout_strategy = str(cfg_yaml.get("rollout_strategy", "auto") or "auto")
    objective_sign = str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward")

    pomo_size = cfg_yaml.get("pomo_size", 64)
    pomo_size_out = int(pomo_size) if pomo_size is not None else None

    validation_batch_size = int(cfg_yaml.get("validation_batch_size", 64) or 64)
    alpha = _alpha_from_cfg(cfg_yaml)
    lr = float(cfg_yaml.get("learning_rate", 3e-4) or 3e-4)
    wd = float(cfg_yaml.get("weight_decay", 1e-6) or 1e-6)
    size_aggregation = str(cfg_yaml.get("size_aggregation", "mean") or "mean")
    size_cvar_alpha = float(cfg_yaml.get("size_cvar_alpha", 0.2) or 0.2)

    # Budget signature:
    # - Legacy step-mode uses K=f1_steps and keeps the original signature keys for backward compatibility
    #   with existing baseline mini-eval JSONs.
    # - Optional epoch-mode (hf_epochs + hf_instances_per_epoch) adds additional keys.
    K = int(cfg_yaml.get("f1_steps", 32) or 32)
    hf_epochs = int(cfg_yaml.get("hf_epochs", 0) or 0)
    hf_instances_per_epoch = int(cfg_yaml.get("hf_instances_per_epoch", 0) or 0)
    train_problem_size = int(cfg_yaml.get("train_problem_size", 20) or 20)
    valid_problem_sizes = [int(v) for v in cfg_yaml.get("valid_problem_sizes", [100])]
    train_batch_size = int(cfg_yaml.get("train_batch_size", 64) or 64)
    num_validation_episodes = int(cfg_yaml.get("num_validation_episodes", 128) or 128)
    scratch_init_seed = int(_resolve_training_seed(cfg_yaml))

    data_sig: Dict[str, Any]
    protocol = "stage3_online_minitrain_v1"
    if uses_offline_data:
        offline_train_sha1 = _file_sha1_cached(str(offline_train))
        offline_val_sig: Dict[str, Any] = {}
        if isinstance(offline_val_paths, dict):
            for size_s, pth in sorted(((str(k), v) for k, v in offline_val_paths.items()), key=lambda kv: int(kv[0])):
                offline_val_sig[str(size_s)] = {
                    "path": str(pth),
                    "sha1": _file_sha1_cached(str(pth)),
                }
        data_sig = {
            "mode": "offline",
            "train": {"path": str(offline_train), "sha1": str(offline_train_sha1)},
            "val": offline_val_sig,
        }
        protocol = "stage3_offline_minitrain_v1"
    else:
        data_sig = {
            "mode": "online",
            "generator_params": dict(generator_params),
        }

    sig = {
        "protocol": str(protocol),
        "baseline_eval_impl_version": 2,
        "baseline_eval_mode": str(baseline_eval_mode),
        "scenario_name": str(_stage3_scenario_name_from_cfg(cfg_yaml)),
        "env_name": env_name,
        "policy_name": policy_name,
        "policy_kwargs": policy_kwargs,
        "env_kwargs": env_kwargs,
        "rollout_strategy": rollout_strategy,
        "objective_sign": objective_sign,
        "alpha": alpha,
        "po_impl": _po_impl_from_cfg(cfg_yaml),
        "K": int(K),
        "train_problem_size": int(train_problem_size),
        "valid_problem_sizes": [int(x) for x in valid_problem_sizes],
        "train_batch_size": int(train_batch_size),
        "num_validation_episodes": int(num_validation_episodes),
        "validation_batch_size": int(validation_batch_size),
        "pomo_size": pomo_size_out,
        "learning_rate": lr,
        "weight_decay": wd,
        "size_aggregation": size_aggregation,
        "size_cvar_alpha": size_cvar_alpha,
        "scratch_init_seed": int(scratch_init_seed),
        "data": data_sig,
        "include_scratch": bool(any(init_ckpt is None for _, init_ckpt in init_specs)),
        "checkpoints": [
            {
                "name": str(init_name),
                "path": str(init_ckpt),
                "sha1": _file_sha1_cached(str(init_ckpt)),
            }
            for init_name, init_ckpt in init_specs
            if init_ckpt is not None
        ],
    }
    if hf_epochs > 0 and hf_instances_per_epoch > 0:
        sig["budget_mode"] = "epochs"
        sig["hf_epochs"] = int(hf_epochs)
        sig["hf_instances_per_epoch"] = int(hf_instances_per_epoch)
    return sig


def _stage3_baseline_eval_mode(cfg_yaml: Mapping[str, Any]) -> str:
    env_name = str(cfg_yaml.get("env_name") or cfg_yaml.get("problem") or "tsp").strip().lower()
    # CVRP and FFSP compare against the native PO objective so the stage3 baseline
    # matches the paper training loss rather than the reference free-loss surrogate.
    return "native_po_loss" if env_name in {"cvrp", "ffsp"} else "ref_free_loss"


def _extract_stage3_size_objectives(
    fitness: Mapping[str, Any],
    *,
    valid_sizes: Sequence[int],
) -> Tuple[Dict[int, float], float]:
    size_objectives_raw = fitness.get("size_objectives", {})
    size_objectives: Dict[int, float] = {}
    if isinstance(size_objectives_raw, Mapping):
        for k, v in size_objectives_raw.items():
            try:
                size_objectives[int(k)] = float(v)
            except Exception:  # noqa: BLE001
                continue

    by_size: Dict[int, float] = {}
    for sz in valid_sizes:
        if int(sz) not in size_objectives:
            raise RuntimeError(f"Missing size_objectives[{int(sz)}] while generating stage3 baseline cache")
        by_size[int(sz)] = float(size_objectives[int(sz)])
    agg = float(sum(by_size[int(sz)] for sz in valid_sizes) / max(len(valid_sizes), 1))
    return by_size, float(agg)


def _evaluate_stage3_reference_baseline(
    *,
    cfg_yaml: Mapping[str, Any],
    hf_cfg: HighFidelityConfig,
    operator_whitelist: Sequence[str],
    init_ckpt: str | None,
) -> Dict[str, Any]:
    eval_mode = _stage3_baseline_eval_mode(cfg_yaml)
    init_ckpt_abs = _abs_from_repo_root(str(init_ckpt)) if init_ckpt else None

    if str(eval_mode) == "native_po_loss":
        return evaluate_po_baseline_rl4co(
            hf_cfg,
            init_checkpoint_path=init_ckpt_abs,
            init_checkpoint_epoch=None,
            scratch_hf_epochs=int(cfg_yaml.get("scratch_hf_epochs", 0) or 0),
            warmstart_hf_epochs=int(cfg_yaml.get("warmstart_hf_epochs", 0) or 0),
            baseline_epoch_compare_offset=int(cfg_yaml.get("baseline_epoch_compare_offset", 0) or 0),
            baseline_epoch_violation_weight=float(cfg_yaml.get("baseline_epoch_violation_weight", 1.0)),
            baseline_epoch_tail_frac=float(cfg_yaml.get("baseline_epoch_tail_frac", 1.0) or 1.0),
            baseline_epoch_window_k=int(cfg_yaml.get("baseline_epoch_window_k", 10) or 10),
            baseline_epoch_window_violation_weight=float(
                cfg_yaml.get("baseline_epoch_window_violation_weight", 1.0) or 1.0
            ),
        )

    compiled_builder = compile_preference_builder(
        _ref_builder_ir(),
        operator_whitelist=list(operator_whitelist),
    )
    ref_loss_ir = _ref_loss_ir()
    static_ref = run_static_gates(ref_loss_ir, operator_whitelist=list(operator_whitelist))
    if not static_ref.ok:
        raise RuntimeError(f"Reference loss failed static gates: {static_ref.reason}")
    compiled_loss = compile_free_loss(ref_loss_ir, operator_whitelist=list(operator_whitelist))
    adapter = _CompiledBuilderAdapter(compiled_builder)
    free_cfg = FreeLossFidelityConfig(
        hf=hf_cfg,
        f1_steps=int(cfg_yaml.get("f1_steps", 32) or 32),
        f2_steps=0,
        f3_enabled=False,
        init_checkpoint_path=init_ckpt_abs,
        init_checkpoint_epoch=None,
        scratch_hf_epochs=int(cfg_yaml.get("scratch_hf_epochs", 0) or 0),
        warmstart_hf_epochs=int(cfg_yaml.get("warmstart_hf_epochs", 0) or 0),
        baseline_epoch_compare_offset=int(cfg_yaml.get("baseline_epoch_compare_offset", 0) or 0),
        baseline_epoch_violation_weight=float(cfg_yaml.get("baseline_epoch_violation_weight", 1.0)),
        baseline_epoch_tail_frac=float(cfg_yaml.get("baseline_epoch_tail_frac", 1.0) or 1.0),
        baseline_epoch_window_k=int(cfg_yaml.get("baseline_epoch_window_k", 10) or 10),
        baseline_epoch_window_violation_weight=float(
            cfg_yaml.get("baseline_epoch_window_violation_weight", 1.0) or 1.0
        ),
    )
    return evaluate_free_loss_candidate(compiled_loss, free_cfg, pref_builder=adapter)


def _stage3_fidelity_key(cfg_yaml: Mapping[str, Any]) -> str:
    hf_epochs = int(cfg_yaml.get("hf_epochs", 0) or 0)
    hf_instances = int(cfg_yaml.get("hf_instances_per_epoch", 0) or 0)
    if hf_epochs > 0 and hf_instances > 0:
        return f"epoch{int(hf_epochs)}_inst{int(hf_instances)}"
    K = int(cfg_yaml.get("f1_steps", 32) or 32)
    return f"K{int(K)}"


def _default_stage3_baseline_mini_eval_path(cfg_yaml: Mapping[str, Any]) -> str:
    env_name = str(cfg_yaml.get("env_name") or cfg_yaml.get("problem") or "tsp").strip().lower()
    train_problem_size = int(cfg_yaml.get("train_problem_size", 20) or 20)
    fidelity = _stage3_fidelity_key(cfg_yaml)
    return os.path.join(
        "baseline",
        "mini_eval",
        f"baseline_minitrain_{env_name}{int(train_problem_size)}_{str(fidelity)}.json",
    ).replace("\\", "/")


def _stage3_baseline_cfg_dict(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg_yaml, dict):
        baseline_cfg = cfg_yaml.get("baseline", {}) or {}
        return dict(baseline_cfg) if isinstance(baseline_cfg, Mapping) else {}
    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    if not isinstance(baseline_cfg, dict):
        baseline_cfg = {}
        cfg_yaml["baseline"] = baseline_cfg
    return baseline_cfg


def _record_stage3_baseline_mini_eval_path(cfg_yaml: Mapping[str, Any], path: str) -> None:
    baseline_cfg = _stage3_baseline_cfg_dict(cfg_yaml)
    fidelity = _stage3_fidelity_key(cfg_yaml)

    raw_map = baseline_cfg.get("mini_eval_paths")
    if not isinstance(raw_map, dict):
        raw_map = {}
        baseline_cfg["mini_eval_paths"] = raw_map
    raw_map[str(fidelity)] = str(path)
    if str(fidelity).startswith("K"):
        try:
            raw_map[str(int(str(fidelity)[1:]))] = str(path)
        except Exception:  # noqa: BLE001
            pass
    baseline_cfg["mini_eval_path"] = str(path)


def _resolve_stage3_baseline_mini_eval_path(cfg_yaml: Mapping[str, Any], baseline_cfg: Mapping[str, Any]) -> str | None:
    raw = baseline_cfg.get("mini_eval_paths", None)
    if raw is None:
        raw = baseline_cfg.get("mini_eval_path", None)

    if isinstance(raw, Mapping):
        key = _stage3_fidelity_key(cfg_yaml)
        for cand in (
            key,
            str(key).replace("K", ""),
            str(key).lower(),
            str(key).upper(),
        ):
            if cand in raw and raw.get(cand):
                return str(raw.get(cand))
        # Also accept integer-ish keys for step-mode.
        if key.startswith("K"):
            try:
                k_int = int(key[1:])
            except Exception:  # noqa: BLE001
                k_int = None
            if k_int is not None:
                for cand in (k_int, str(k_int)):
                    if cand in raw and raw.get(cand):
                        return str(raw.get(cand))
        # Optional fallback entry.
        for cand in ("default", "DEFAULT", "_default_", "*"):
            if cand in raw and raw.get(cand):
                return str(raw.get(cand))
        return _default_stage3_baseline_mini_eval_path(cfg_yaml)

    if isinstance(raw, str) and raw.strip():
        return str(raw)
    return _default_stage3_baseline_mini_eval_path(cfg_yaml)


@torch.no_grad()
def _stage3_pre_minitrain_eval(
    *,
    cfg_yaml: Mapping[str, Any],
    init_checkpoint: str | None,
    train_problem_size: int,
    valid_problem_sizes: Sequence[int],
    num_validation_episodes: int,
    train_batch_size: int,
    scratch_init_seed: int,
    offline_train: str | None = None,
    offline_val_by_size: Mapping[int, str] | None = None,
) -> Tuple[Dict[int, float], float]:
    from fitness.free_loss_fidelity import (
        _evaluate_rl4co_model,
        _load_policy_weights_from_checkpoint,
        _rl4co_build_env,
        _rl4co_build_policy,
    )

    generator_params = dict(cfg_yaml.get("generator_params", {}) or {})
    if offline_train or offline_val_by_size:
        if not offline_train or not offline_val_by_size:
            raise ValueError(
                "stage3 pre-mini-train eval requires both offline_train and offline_val_by_size when using offline data"
            )
        generator_params["offline_train_path"] = str(offline_train)
        generator_params["offline_val_paths"] = {
            str(int(k)): str(v) for k, v in offline_val_by_size.items()
        }

    hf_cfg = HighFidelityConfig(
        problem=str(cfg_yaml.get("problem", "tsp")),
        backend=str(cfg_yaml.get("backend", "rl4co") or "rl4co"),
        env_name=str(cfg_yaml.get("env_name") or cfg_yaml.get("problem", "tsp")),
        env_kwargs=dict(cfg_yaml.get("env_kwargs", {}) or {}),
        generator_params=generator_params,
        policy_name=str(cfg_yaml.get("policy_name", "") or ""),
        policy_kwargs=dict(cfg_yaml.get("policy_kwargs", {}) or {}),
        rollout_strategy=str(cfg_yaml.get("rollout_strategy", "auto") or "auto"),
        objective_sign=str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward"),
        hf_steps=1,
        hf_epochs=0,
        hf_instances_per_epoch=0,
        train_problem_size=int(train_problem_size),
        valid_problem_sizes=tuple(int(x) for x in valid_problem_sizes),
        train_batch_size=int(train_batch_size),
        pomo_size=(
            int(cfg_yaml.get("pomo_size"))
            if cfg_yaml.get("pomo_size", None) is not None
            else None
        ),
        learning_rate=float(cfg_yaml.get("learning_rate", 3e-4) or 3e-4),
        weight_decay=float(cfg_yaml.get("weight_decay", 1e-6) or 1e-6),
        alpha=_alpha_from_cfg(cfg_yaml),
        device=str(cfg_yaml.get("device", "cuda") or "cuda"),
        seed=int(scratch_init_seed),
        num_validation_episodes=int(num_validation_episodes),
        validation_batch_size=int(cfg_yaml.get("validation_batch_size", 64) or 64),
        generalization_penalty_weight=float(
            cfg_yaml.get("generalization_penalty_weight", 1.0) or 1.0
        ),
        size_aggregation=str(cfg_yaml.get("size_aggregation", "mean") or "mean"),
        size_cvar_alpha=float(cfg_yaml.get("size_cvar_alpha", 0.2) or 0.2),
        pool_version=str(cfg_yaml.get("pool_version", "v0") or "v0"),
    )

    _set_seed(int(hf_cfg.seed))
    device_str = str(hf_cfg.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    env = _rl4co_build_env(hf_cfg, int(train_problem_size)).to(device)
    policy, rollout_strategy = _rl4co_build_policy(hf_cfg, env)
    if init_checkpoint:
        _load_policy_weights_from_checkpoint(policy, _abs_from_repo_root(str(init_checkpoint)))
    policy = policy.to(device)
    policy.eval()

    by_size: Dict[int, float] = {}
    for sz in valid_problem_sizes:
        obj = _evaluate_rl4co_model(
            policy=policy,
            cfg=hf_cfg,
            problem_size=int(sz),
            device=device,
            num_episodes=int(num_validation_episodes),
            batch_size=int(hf_cfg.validation_batch_size),
            rollout_strategy=str(rollout_strategy),
        )
        by_size[int(sz)] = float(obj)

    aggregated = float(sum(by_size[int(sz)] for sz in valid_problem_sizes) / max(len(valid_problem_sizes), 1))
    return by_size, float(aggregated)


def _ensure_stage3_baseline_mini_eval(
    *,
    cfg_yaml: Mapping[str, Any],
    operator_whitelist: Sequence[str],
    device_str: str,
) -> Dict[str, Any]:
    baseline_cfg = _stage3_baseline_cfg_dict(cfg_yaml)
    mini_eval_path = _resolve_stage3_baseline_mini_eval_path(cfg_yaml, baseline_cfg)
    if not mini_eval_path:
        raise ValueError("Failed to resolve stage3 baseline mini-eval path")
    _record_stage3_baseline_mini_eval_path(cfg_yaml, str(mini_eval_path))

    expected_sig = _build_stage3_eval_signature(cfg_yaml)
    existing = None
    if os.path.isfile(_abs_from_repo_root(str(mini_eval_path))):
        try:
            existing = _load_baseline_mini_eval(str(mini_eval_path))
        except Exception:  # noqa: BLE001
            existing = None
    if (
        isinstance(existing, Mapping)
        and existing.get("eval_signature") == expected_sig
        and isinstance(existing.get("per_init"), Mapping)
    ):
        return {
            "path": str(mini_eval_path),
            "cached": True,
            "regenerated": False,
            "eval_signature": expected_sig,
        }

    init_specs = _stage3_init_specs_from_baseline_cfg(cfg_yaml)
    if not init_specs:
        raise ValueError("stage3 baseline requires at least one init source (scratch and/or baseline.checkpoints)")

    scratch_init_seed = int(_resolve_training_seed(cfg_yaml))
    train_problem_size = int(cfg_yaml.get("train_problem_size", 20) or 20)
    valid_problem_sizes = [int(v) for v in cfg_yaml.get("valid_problem_sizes", [train_problem_size])]
    valid_problem_sizes = list(dict.fromkeys(valid_problem_sizes))
    num_validation_episodes = int(cfg_yaml.get("num_validation_episodes", 128) or 128)
    train_batch_size = int(cfg_yaml.get("train_batch_size", 64) or 64)
    K = int(cfg_yaml.get("f1_steps", 32) or 32)

    generator_params = dict(cfg_yaml.get("generator_params", {}) or {})
    offline_train = generator_params.get("offline_train_path")
    offline_val_paths = generator_params.get("offline_val_paths") or {}
    uses_offline_data = bool(offline_train or offline_val_paths)
    offline_val_by_size: Dict[int, str] | None = None
    if uses_offline_data:
        if not offline_train or not isinstance(offline_val_paths, Mapping):
            raise ValueError(
                "stage3 baseline auto-cache offline mode requires generator_params.offline_train_path and offline_val_paths"
            )
        offline_val_by_size = {}
        for sz in valid_problem_sizes:
            p = offline_val_paths.get(str(int(sz)), offline_val_paths.get(int(sz)))
            if not p:
                raise ValueError(f"Missing offline_val_paths[{int(sz)}] for stage3 baseline auto-cache")
            offline_val_by_size[int(sz)] = str(p)

    cfg_hf = dict(cfg_yaml)
    cfg_hf["f1_steps"] = int(K)
    if not (
        int(cfg_yaml.get("hf_epochs", 0) or 0) > 0
        and int(cfg_yaml.get("hf_instances_per_epoch", 0) or 0) > 0
    ):
        cfg_hf["hf_epochs"] = 0
        cfg_hf["hf_instances_per_epoch"] = 0
    hf_cfg = _build_hf_cfg(cfg_hf, seed=int(scratch_init_seed), device_str=str(device_str))

    objective_sign = str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward")
    per_init: Dict[str, Any] = {}
    for init_name, init_ckpt in init_specs:
        pre_by_size, pre_agg = _stage3_pre_minitrain_eval(
            cfg_yaml=cfg_yaml,
            init_checkpoint=init_ckpt,
            train_problem_size=int(train_problem_size),
            valid_problem_sizes=list(valid_problem_sizes),
            num_validation_episodes=int(num_validation_episodes),
            train_batch_size=int(train_batch_size),
            scratch_init_seed=int(scratch_init_seed),
            offline_train=(str(offline_train) if offline_train else None),
            offline_val_by_size=offline_val_by_size,
        )
        fitness = _evaluate_stage3_reference_baseline(
            cfg_yaml=cfg_yaml,
            hf_cfg=hf_cfg,
            operator_whitelist=operator_whitelist,
            init_ckpt=init_ckpt,
        )
        by_size, agg = _extract_stage3_size_objectives(fitness, valid_sizes=valid_problem_sizes)

        per_init[str(init_name)] = {
            "pre_val_objective_by_size": {str(int(k)): float(v) for k, v in pre_by_size.items()},
            "pre_val_reward_by_size": {
                str(int(k)): float((-float(v)) if objective_sign == "neg_reward" else float(v))
                for k, v in pre_by_size.items()
            },
            "pre_aggregated_objective": float(pre_agg),
            "pre_aggregated_reward": float((-float(pre_agg)) if objective_sign == "neg_reward" else float(pre_agg)),
            "val_objective_by_size": {str(int(k)): float(v) for k, v in by_size.items()},
            "val_reward_by_size": {
                str(int(k)): float((-float(v)) if objective_sign == "neg_reward" else float(v))
                for k, v in by_size.items()
            },
            "aggregated_objective": float(agg),
            "aggregated_reward": float((-float(agg)) if objective_sign == "neg_reward" else float(agg)),
            "delta_objective_post_minus_pre": float(float(agg) - float(pre_agg)),
            "delta_reward_post_minus_pre": float(
                ((-float(agg)) if objective_sign == "neg_reward" else float(agg))
                - ((-float(pre_agg)) if objective_sign == "neg_reward" else float(pre_agg))
            ),
            "init_checkpoint": str(init_ckpt) if init_ckpt else None,
        }

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": None,
        "eval_signature": expected_sig,
        "per_init": per_init,
        "reference": {
            "builder_ir": asdict(_ref_builder_ir()),
            "loss_ir": asdict(_ref_loss_ir()),
        },
    }
    _atomic_write_json(_abs_from_repo_root(str(mini_eval_path)), payload)
    _BASELINE_MINI_EVAL_CACHE[_abs_from_repo_root(str(mini_eval_path))] = dict(payload)
    return {
        "path": str(mini_eval_path),
        "cached": False,
        "regenerated": True,
        "eval_signature": expected_sig,
    }


def _infer_baseline_epoch_from_path(path: str) -> int | None:
    name = os.path.basename(str(path))
    m = re.search(r"(?:^|[._-])epoch_(\d+)(?:\D|$)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _timestamp_dir(root: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _append_jsonl(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")


def _truncate_jsonl_by_generation(path: str, gen_start: int) -> Tuple[int, int]:
    """Keep only JSONL records with generation < gen_start.

    Returns: (kept_count, dropped_count)
    """
    if not os.path.exists(path):
        return 0, 0

    keep: List[Dict[str, Any]] = []
    kept = 0
    dropped = 0
    threshold = int(gen_start)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:  # noqa: BLE001
                dropped += 1
                continue
            if not isinstance(obj, Mapping):
                dropped += 1
                continue
            try:
                g = int(obj.get("generation"))
            except Exception:  # noqa: BLE001
                dropped += 1
                continue
            if g < threshold:
                keep.append(dict(obj))
                kept += 1
            else:
                dropped += 1

    with open(path, "w", encoding="utf-8") as f:
        for obj in keep:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return kept, dropped


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _b64_pickle(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _unb64_pickle(data: str) -> Any:
    return pickle.loads(base64.b64decode(data.encode("ascii")))


def _checkpoint_path(run_dir: str) -> str:
    return os.path.join(run_dir, "checkpoint.json")


def _save_checkpoint(run_dir: str, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload["schema_version"] = 1
    payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(_checkpoint_path(run_dir), payload)


def _load_checkpoint(run_dir: str) -> Dict[str, Any]:
    with open(_checkpoint_path(run_dir), "r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint format: {_checkpoint_path(run_dir)}")
    return state


def _normalize_loss_transfer_seed_cfg(cfg_yaml: Mapping[str, Any]) -> Dict[str, Any]:
    raw = cfg_yaml.get("loss_transfer_seed", {}) or {}
    if not isinstance(raw, Mapping):
        return {"enabled": False}

    return {
        "enabled": bool(raw.get("enabled", False)),
        "source_loss_path": str(raw.get("source_loss_path", "") or "").strip(),
        "source_run_dir": str(raw.get("source_run_dir", "") or "").strip(),
        "source_checkpoint_path": str(raw.get("source_checkpoint_path", "") or "").strip(),
        "source_pool": str(raw.get("source_pool", "elites") or "elites").strip().lower(),
        "top_k": max(1, int(raw.get("top_k", 8) or 8)),
        "max_per_family": max(0, int(raw.get("max_per_family", 2) or 2)),
        "keep_source_fitness": bool(raw.get("keep_source_fitness", True)),
        "reset_history": bool(raw.get("reset_history", False)),
    }


def _resolve_loss_transfer_seed_checkpoint_path(seed_cfg: Mapping[str, Any]) -> str:
    ckpt_path = str(seed_cfg.get("source_checkpoint_path", "") or "").strip()
    if ckpt_path:
        return _abs_from_repo_root(ckpt_path)

    run_dir = str(seed_cfg.get("source_run_dir", "") or "").strip()
    if not run_dir:
        raise ValueError("loss_transfer_seed requires source_checkpoint_path or source_run_dir")

    return os.path.join(_abs_from_repo_root(run_dir), "checkpoint.json")


def _resolve_loss_transfer_seed_loss_path(seed_cfg: Mapping[str, Any]) -> str:
    loss_path = str(seed_cfg.get("source_loss_path", "") or "").strip()
    if not loss_path:
        raise ValueError("loss_transfer_seed requires source_loss_path")
    return _abs_from_repo_root(loss_path)


def _coerce_numeric_transfer_seed_fitness(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("delta_mean", "score", "final_score", "latest", "mean"):
            nested = value.get(key)
            try:
                return float(nested)
            except (TypeError, ValueError):
                continue
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_transfer_seed_entry(
    raw: Mapping[str, Any],
    *,
    index: int,
    source_ref: str,
    source_kind: str,
    source_pool: str,
    keep_source_fitness: bool,
    reset_history: bool,
) -> Dict[str, Any] | None:
    ir_raw = raw.get("ir")
    if not isinstance(ir_raw, dict):
        return None

    try:
        ir = free_loss_ir_from_json(ir_raw)
    except Exception:  # noqa: BLE001
        return None

    sig = str(raw.get("signature") or _sig_free_loss(ir))
    family = str(raw.get("family") or _loss_family_id(ir))
    family_signature = str(raw.get("family_signature") or _loss_family_signature(ir))
    src_id = str(raw.get("id") or sig[:8])
    new_id = f"fseed_{int(index):03d}_{sig[:8]}"

    hist: List[Dict[str, Any]] = []
    if (not bool(reset_history)) and isinstance(raw.get("history"), list):
        hist.extend([dict(item) for item in list(raw.get("history") or []) if isinstance(item, Mapping)])
    hist.append(
        {
            "op": "TRANSFER_SEED",
            source_kind: os.path.abspath(source_ref),
            "source_pool": str(source_pool),
            "source_id": str(src_id),
        }
    )

    source_fitness = _coerce_numeric_transfer_seed_fitness(raw.get("fitness"))
    if source_fitness is None:
        source_fitness = _coerce_numeric_transfer_seed_fitness(raw.get("score"))
    if source_fitness is None:
        source_fitness = _coerce_numeric_transfer_seed_fitness(raw.get("final_score"))
    if source_fitness is None:
        source_fitness = _coerce_numeric_transfer_seed_fitness(raw.get("score_history_summary"))

    return {
        "generation": -1,
        "index": int(index),
        "id": str(new_id),
        "signature": str(sig),
        "family": str(family),
        "family_signature": str(family_signature),
        "origin": "TRANSFER_SEED",
        "origin_base": str(src_id),
        "op_type": "TRANSFER_SEED",
        "parents": [str(src_id)],
        "attempt": 0,
        "prompt_sha1": None,
        "prompt_path": None,
        "llm_seed": None,
        "history": hist,
        "novelty": None,
        "ir": asdict(ir),
        "static_ok": True,
        "static_reason": "transfer_seed",
        "static_trace": {},
        "compile_ok": True,
        "compile_reason": "transfer_seed",
        "fitness": (float(source_fitness) if bool(keep_source_fitness) and source_fitness is not None else 0.0),
        "descriptor": raw.get("descriptor"),
        source_kind: os.path.abspath(source_ref),
        "source_loss_id": str(src_id),
        "source_fitness": source_fitness,
    }


def _load_loss_transfer_seed_entries(
    checkpoint_path: str,
    *,
    source_pool: str = "elites",
    top_k: int = 8,
    max_per_family: int = 2,
    keep_source_fitness: bool = True,
    reset_history: bool = False,
) -> List[Dict[str, Any]]:
    payload = _load_json(checkpoint_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid transfer checkpoint: {checkpoint_path}")

    pool_key_map = {
        "elites": "elites_f",
        "resident": "resident_pop_f",
        "hof": "hof_f",
    }
    pool_key = pool_key_map.get(str(source_pool).strip().lower(), "elites_f")
    raw_entries = payload.get(pool_key, []) or []
    if not isinstance(raw_entries, list):
        raw_entries = []

    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            continue
        ir_raw = raw.get("ir")
        if not isinstance(ir_raw, dict):
            continue
        fit = _coerce_numeric_transfer_seed_fitness(raw.get("fitness"))
        if fit is None:
            fit = _coerce_numeric_transfer_seed_fitness(raw.get("score"))
        if fit is None:
            fit = float("inf")
        ranked.append((float(fit), dict(raw)))

    ranked.sort(key=lambda x: float(x[0]))

    out: List[Dict[str, Any]] = []
    family_counter: Dict[str, int] = {}

    for _, raw in ranked:
        entry = _build_transfer_seed_entry(
            raw,
            index=len(out),
            source_ref=checkpoint_path,
            source_kind="source_checkpoint",
            source_pool=str(source_pool),
            keep_source_fitness=keep_source_fitness,
            reset_history=reset_history,
        )
        if entry is None:
            continue
        family_signature = str(entry.get("family_signature") or "")

        if max_per_family > 0:
            used = int(family_counter.get(family_signature, 0))
            if used >= int(max_per_family):
                continue
            family_counter[family_signature] = used + 1

        out.append(entry)

        if len(out) >= int(top_k):
            break

    return out


def _load_loss_transfer_seed_entries_from_loss_path(
    loss_path: str,
    *,
    keep_source_fitness: bool = True,
    reset_history: bool = False,
) -> List[Dict[str, Any]]:
    payload = _load_json(loss_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid transfer loss artifact: {loss_path}")

    entry = _build_transfer_seed_entry(
        payload,
        index=0,
        source_ref=loss_path,
        source_kind="source_loss_path",
        source_pool="artifact",
        keep_source_fitness=keep_source_fitness,
        reset_history=reset_history,
    )
    return [entry] if entry is not None else []


def _hf_scheduler_mode(cfg_yaml: Mapping[str, Any]) -> str:
    raw = cfg_yaml.get("high_fidelity_scheduler", cfg_yaml.get("hf_scheduler_mode", "subprocess"))
    mode = str(raw or "subprocess").strip().lower()
    if mode in {"subprocess", "process"}:
        return "subprocess"
    if mode in {"multiprocessing", "mp"}:
        return "multiprocessing"
    return "subprocess"


def _hf_subprocess_script_path() -> str:
    return os.path.join(_repo_root_dir(), "PTP", "ptp_discovery", "run_hf_pair_eval.py")


def _stage0_sandbox_script_path() -> str:
    return os.path.join(_repo_root_dir(), "PTP", "ptp_discovery", "run_stage0_sandbox_gate.py")


def _run_stage0_sandbox_gate(
    *,
    run_dir: str,
    generation: int,
    pair_index: int,
    g_id: str,
    f_id: str,
    g_ir: PreferenceBuilderIR,
    f_ir: FreeLossIR,
    operator_whitelist: Sequence[str],
    cfg_yaml: Mapping[str, Any],
) -> Dict[str, Any]:
    script_path = _stage0_sandbox_script_path()
    if not os.path.isfile(script_path):
        return {
            "ok": False,
            "failure_kind": "sandbox_script_missing",
            "reason": f"Missing sandbox script: {script_path}",
        }

    run_dir_s = str(run_dir or "").strip()
    if not run_dir_s:
        run_dir_s = os.path.join(_repo_root_dir(), "runs", "_sandbox")
    task_dir = os.path.join(
        run_dir_s,
        "stage0_sandbox",
        f"gen{int(generation):03d}_pair{int(pair_index):03d}_{str(g_id)[:16]}_{str(f_id)[:16]}",
    )
    os.makedirs(task_dir, exist_ok=True)
    payload_path = os.path.join(task_dir, "payload.json")
    result_path = os.path.join(task_dir, "result.json")
    log_path = os.path.join(task_dir, "subprocess.log")

    variants_raw = cfg_yaml.get("stage0_sandbox_variants", ["visible", "hidden"])
    if isinstance(variants_raw, (list, tuple)):
        variants = [str(v) for v in variants_raw if str(v).strip()]
    elif variants_raw is None:
        variants = ["visible", "hidden"]
    else:
        variants = [str(variants_raw)]
    if not variants:
        variants = ["visible", "hidden"]

    payload = {
        "generation": int(generation),
        "pair_index": int(pair_index),
        "g_entry": {"id": str(g_id), "ir": asdict(g_ir)},
        "f_entry": {"id": str(f_id), "ir": asdict(f_ir)},
        "operator_whitelist": list(operator_whitelist),
        "batch_size": int(cfg_yaml.get("stage0_sandbox_batch_size", 8) or 8),
        "k": int(cfg_yaml.get("stage0_sandbox_k", 16) or 16),
        "rounds": int(cfg_yaml.get("stage0_sandbox_rounds", 1) or 1),
        "variants": variants,
        "hard_failure_kinds": list(
            cfg_yaml.get(
                "stage0_sandbox_hard_failure_kinds",
                [
                    "pref_batch_to_loss_batch_error",
                    "forward_error",
                    "backward_error",
                    "loss_not_finite",
                    "missing_grads",
                    "grad_not_finite",
                    "numeric_stress_forward_error",
                    "numeric_stress_backward_error",
                    "numeric_stress_loss_not_finite",
                    "numeric_stress_missing_grads",
                    "numeric_stress_grad_not_finite",
                ],
            )
            or []
        ),
        "builder_gate": {
            "min_pairs": int(cfg_yaml.get("builder_min_pairs", 1) or 1),
            "min_coverage": float(cfg_yaml.get("builder_min_coverage", 0.0) or 0.0),
            "max_pairs_per_instance": int(cfg_yaml.get("builder_max_pairs_per_instance", 4096) or 4096),
            "weight_nonneg": bool(cfg_yaml.get("builder_weight_nonneg", True)),
            "semantic_tolerance": float(cfg_yaml.get("builder_semantic_tolerance", 0.0) or 0.0),
            "semantic_min_pass_rate": float(cfg_yaml.get("builder_semantic_min_pass_rate", 1.0) or 1.0),
        },
        "joint_gate": {
            "min_pass_rate": float(cfg_yaml.get("joint_min_pass_rate", 0.8) or 0.8),
            "swap_tolerance": float(cfg_yaml.get("joint_swap_tolerance", 1e-3) or 1e-3),
            "swap_check_mode": str(cfg_yaml.get("joint_swap_check_mode", "data") or "data"),
            "swap_test_margin": float(cfg_yaml.get("joint_swap_test_margin", 1.0) or 1.0),
            "grad_eps": float(cfg_yaml.get("joint_grad_eps", 1e-8) or 1e-8),
            "min_effective_grad_ratio": float(cfg_yaml.get("joint_min_effective_grad_ratio", 0.1) or 0.1),
            "numeric_stress_enabled": bool(cfg_yaml.get("joint_numeric_stress_enabled", True)),
            "numeric_stress_margin": float(cfg_yaml.get("joint_numeric_stress_margin", 120.0) or 120.0),
            "numeric_stress_aux_scale": float(cfg_yaml.get("joint_numeric_stress_aux_scale", 32.0) or 32.0),
        },
    }
    _atomic_write_json(payload_path, payload)

    timeout_raw = cfg_yaml.get("stage0_sandbox_timeout_s", 25.0)
    try:
        timeout_s = max(1.0, float(timeout_raw))
    except (TypeError, ValueError):
        timeout_s = 25.0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    proc: subprocess.Popen[Any] | None = None
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    try:
        proc = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-u",
                script_path,
                "--payload",
                payload_path,
                "--result",
                result_path,
            ],
            cwd=_repo_root_dir(),
            env=env,
            stdout=log_fh,
            stderr=log_fh,
        )
        try:
            proc.wait(timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            return {
                "ok": False,
                "failure_kind": "sandbox_timeout",
                "reason": f"Sandbox gate timed out after {timeout_s:.1f}s",
                "exit_code": proc.poll(),
                "sandbox_log": os.path.relpath(log_path, start=run_dir_s),
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "failure_kind": "sandbox_runtime_error",
            "reason": f"Failed launching sandbox subprocess: {exc}",
            "exception_type": type(exc).__name__,
        }
    finally:
        try:
            log_fh.close()
        except Exception:  # noqa: BLE001
            pass

    if not os.path.isfile(result_path):
        return {
            "ok": False,
            "failure_kind": "sandbox_no_result",
            "reason": "Sandbox subprocess exited without a result payload",
            "exit_code": None if proc is None else proc.poll(),
            "sandbox_log": os.path.relpath(log_path, start=run_dir_s),
        }

    try:
        loaded = _load_json(result_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "failure_kind": "sandbox_result_invalid",
            "reason": f"Failed to parse sandbox result: {exc}",
            "exit_code": None if proc is None else proc.poll(),
            "sandbox_log": os.path.relpath(log_path, start=run_dir_s),
        }

    if not isinstance(loaded, Mapping):
        return {
            "ok": False,
            "failure_kind": "sandbox_result_invalid",
            "reason": "Sandbox result payload must be a dict",
            "exit_code": None if proc is None else proc.poll(),
            "sandbox_log": os.path.relpath(log_path, start=run_dir_s),
        }

    out = dict(loaded)
    out.setdefault("ok", False)
    out.setdefault("failure_kind", None if bool(out.get("ok")) else "sandbox_gate_failed")
    out.setdefault("reason", "ok" if bool(out.get("ok")) else "sandbox gate failed")
    out["exit_code"] = None if proc is None else proc.poll()
    out["sandbox_log"] = os.path.relpath(log_path, start=run_dir_s)
    out["sandbox_result"] = os.path.relpath(result_path, start=run_dir_s)
    return out


def _hf_subprocess_env_and_device(device_str: str) -> Tuple[Dict[str, str], str]:
    env = dict(os.environ)
    py_paths = [str(_repo_root_dir()), os.path.join(_repo_root_dir(), "PTP")]
    if env.get("PYTHONPATH"):
        py_paths.append(str(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    dev = str(device_str or "").strip()
    if dev.startswith("cuda:"):
        idx = dev.split(":", 1)[1].strip()
        if idx:
            env["CUDA_VISIBLE_DEVICES"] = str(idx)
            return env, "cuda:0"
    return env, dev


def _hf_subprocess_failure_record(
    task: Mapping[str, Any],
    *,
    reason: str,
    error: str,
    exitcode: int | None = None,
) -> Dict[str, Any]:
    fixed = dict(task) if isinstance(task, Mapping) else {}
    physical_device = str(fixed.get("device_physical_str") or fixed.get("device_str") or "")
    fixed["device"] = physical_device
    fixed["device_str"] = physical_device
    fixed["pair_ok"] = False
    fixed["pair_reason"] = str(reason)
    fixed["high_fidelity_error"] = str(error)
    if exitcode is not None:
        fixed["high_fidelity_exitcode"] = int(exitcode)
    fixed["score"] = float("inf")
    return fixed


def _resolve_hf_timeout_s(
    cfg_like: Mapping[str, Any] | None,
    *,
    default: float | None = None,
) -> float | None:
    """Return the configured HF task timeout in seconds, or None when disabled."""

    raw = cfg_like.get("high_fidelity_task_timeout_s", default) if isinstance(cfg_like, Mapping) else default
    if raw is None:
        return None
    try:
        timeout_s = float(raw)
    except (TypeError, ValueError):
        return default
    if (not math.isfinite(timeout_s)) or timeout_s <= 0.0:
        return None
    return max(1.0, timeout_s)


def _run_hf_tasks_via_subprocess(  # noqa: PLR0912
    *,
    hf_tasks: Sequence[Mapping[str, Any]],
    run_dir: str,
    device_list: Sequence[str],
    max_workers: int,
    runtime_trace: RuntimeTrace | None = None,
) -> List[Dict[str, Any]]:
    if not hf_tasks:
        return []

    task_root = os.path.join(run_dir, "hf_subprocess")
    os.makedirs(task_root, exist_ok=True)
    script_path = _hf_subprocess_script_path()
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Missing HF subprocess worker script: {script_path}")

    max_workers = max(1, int(max_workers))
    max_workers = min(int(max_workers), max(1, len(device_list)), max(1, len(hf_tasks)))

    pending: List[Dict[str, Any]] = [dict(t) for t in hf_tasks]
    active: List[Dict[str, Any]] = []
    results_by_key: Dict[Tuple[int, int, str, str], Dict[str, Any]] = {}

    def _task_key(task_like: Mapping[str, Any]) -> Tuple[int, int, str, str]:
        return (
            _safe_int(task_like.get("generation", -1), -1),
            _safe_int(task_like.get("pair_index", -1), -1),
            str(task_like.get("g_entry", {}).get("id", task_like.get("g_id", ""))),
            str(task_like.get("f_entry", {}).get("id", task_like.get("f_id", ""))),
        )

    def _launch_ready_tasks() -> bool:
        launched = False
        busy_devices = {
            str(meta.get("device_physical_str", meta.get("device_str", ""))) for meta in active
        }
        while len(active) < int(max_workers):
            next_idx = None
            for idx, task in enumerate(pending):
                dev = str(task.get("device_str", ""))
                if dev not in busy_devices:
                    next_idx = idx
                    break
            if next_idx is None:
                break

            task = dict(pending.pop(next_idx))
            physical_device = str(task.get("device_str", ""))
            env, worker_device = _hf_subprocess_env_and_device(physical_device)
            task["device_physical_str"] = physical_device
            task["device_str"] = worker_device

            gen = _safe_int(task.get("generation", -1), -1)
            pair_index = _safe_int(task.get("pair_index", -1), -1)
            task_dir = os.path.join(
                task_root,
                f"gen{int(gen):03d}_pair{int(pair_index):03d}_{str(physical_device).replace(':', '_')}",
            )
            os.makedirs(task_dir, exist_ok=True)
            payload_path = os.path.join(task_dir, "payload.json")
            result_path = os.path.join(task_dir, "result.json")
            log_path = os.path.join(task_dir, "subprocess.log")
            _atomic_write_json(payload_path, task)
            timeout_s = _resolve_hf_timeout_s(task.get("cfg_yaml"), default=None)

            log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-u",
                    script_path,
                    "--payload",
                    payload_path,
                    "--result",
                    result_path,
                ],
                cwd=_repo_root_dir(),
                env=env,
                stdout=log_fh,
                stderr=log_fh,
            )
            active.append(
                {
                    "task": task,
                    "proc": proc,
                    "payload_path": payload_path,
                    "result_path": result_path,
                    "log_path": log_path,
                    "log_fh": log_fh,
                    "deadline": (float(time.time() + timeout_s) if timeout_s is not None else None),
                    "timeout_s": timeout_s,
                    "device_physical_str": physical_device,
                    "key": _task_key(task),
                }
            )
            busy_devices.add(physical_device)
            launched = True
        return launched

    while pending or active:
        _launch_ready_tasks()
        if runtime_trace is not None:
            runtime_trace.heartbeat(
                extra={
                    "stage": "hf_subprocess_scheduler",
                    "pending_tasks": int(len(pending)),
                    "active_tasks": int(len(active)),
                    "completed_tasks": int(len(results_by_key)),
                },
                min_interval_s=30.0,
            )
        progressed = False
        now = float(time.time())
        for meta in list(active):
            proc = meta["proc"]
            task = meta["task"]
            result_path = str(meta["result_path"])
            key = meta["key"]
            exitcode = proc.poll()
            deadline_raw = meta.get("deadline")
            timed_out = (deadline_raw is not None) and (now > float(deadline_raw))

            if exitcode is None and not timed_out:
                continue

            if exitcode is None and timed_out:
                try:
                    proc.terminate()
                    proc.wait(timeout=10.0)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                rec = _hf_subprocess_failure_record(
                    task,
                    reason="child_timeout",
                    error=f"HF subprocess exceeded timeout before producing a result: {result_path}",
                    exitcode=proc.poll(),
                )
            elif os.path.isfile(result_path):
                try:
                    rec_raw = _load_json(result_path)
                    rec = dict(rec_raw) if isinstance(rec_raw, Mapping) else _hf_subprocess_failure_record(
                        task,
                        reason="child_exit_no_result",
                        error=f"HF subprocess wrote invalid result payload: {result_path}",
                        exitcode=exitcode,
                    )
                except Exception as exc:  # noqa: BLE001
                    rec = _hf_subprocess_failure_record(
                        task,
                        reason="child_exception",
                        error=f"Failed to load HF subprocess result ({result_path}): {exc}",
                        exitcode=exitcode,
                    )
            else:
                rec = _hf_subprocess_failure_record(
                    task,
                    reason="child_exit_no_result",
                    error=f"HF subprocess exited without result file: {result_path}",
                    exitcode=exitcode,
                )

            physical_device = str(meta.get("device_physical_str", ""))
            if physical_device:
                rec["device"] = physical_device
                rec["device_str"] = physical_device
                rec["device_physical_str"] = physical_device
            rec["hf_subprocess_log"] = os.path.relpath(str(meta["log_path"]), start=run_dir)
            rec["hf_subprocess_result"] = os.path.relpath(result_path, start=run_dir)
            results_by_key[key] = dict(rec)
            active.remove(meta)
            try:
                meta["log_fh"].close()
            except Exception:  # noqa: BLE001
                pass
            progressed = True

        if not progressed:
            time.sleep(0.5)

    out: List[Dict[str, Any]] = []
    for task in hf_tasks:
        key = _task_key(task)
        rec = results_by_key.get(key)
        if rec is None:
            rec = _hf_subprocess_failure_record(
                task,
                reason="child_exit_no_result",
                error="HF subprocess scheduler completed without a matching result",
            )
        out.append(dict(rec))
    return out


def _sig(obj: Mapping[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha1(blob).hexdigest()


def _sig_free_loss(ir: FreeLossIR) -> str:
    return _sig(asdict(ir))


def _free_loss_code_digest(ir: FreeLossIR, *, preview_chars: int = 200) -> Dict[str, Any]:
    code = str(getattr(ir, "code", "") or "")
    n = max(int(preview_chars), 0)
    prefix = code[:n]
    suffix = code[-n:] if n > 0 and len(code) > n else code
    return {
        "sha1": sha1(code.encode("utf-8")).hexdigest(),
        "length": int(len(code)),
        "prefix": prefix,
        "suffix": suffix,
    }


def _configure_loss_llm_for_worker(*, cfg: Mapping[str, Any], run_dir: str | None) -> None:
    loss_cfg = cfg.get("loss_llm", {}) if isinstance(cfg.get("loss_llm"), dict) else {}
    offline_mode = bool(loss_cfg.get("offline_mode", cfg.get("llm_offline_mode", False)))
    loss_llm_ops.configure_llm_run(run_dir=run_dir, offline_mode=offline_mode)


def _configure_builder_llm_for_worker(*, cfg: Mapping[str, Any], run_dir: str | None) -> None:
    builder_cfg = cfg.get("builder_llm", {}) if isinstance(cfg.get("builder_llm"), dict) else {}
    offline_mode = bool(builder_cfg.get("offline_mode", cfg.get("llm_offline_mode", False)))
    builder_llm_ops.configure_llm_run(run_dir=run_dir, offline_mode=offline_mode)


def _pop_joint_gate_repair_reports(record: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(record, dict):
        return []
    reports = record.pop("joint_gate_repair_reports", None)
    if not isinstance(reports, list):
        return []
    out: List[Dict[str, Any]] = []
    for rep in reports:
        if isinstance(rep, dict):
            out.append(dict(rep))
    return out


_LOSS_FINGERPRINT_CACHE: Dict[str, Dict[str, Any]] = {}

_BUILDER_FAMILY_KEYS = (
    "geometry_family",
    "cap_family",
    "constraint_family",
)
_LOSS_FAMILY_KEYS = (
    "paradigm_family",
    "signal_family",
    "agg_family",
    "constraint_family",
)


def _loss_fingerprint(ir: FreeLossIR) -> Dict[str, Any]:
    """Compute a compact structural fingerprint for novelty checks.

    Detect near-duplicates at the operator/structure level (AST first, tokenize fallback).
    """

    sig = _sig_free_loss(ir)
    cached = _LOSS_FINGERPRINT_CACHE.get(sig)
    if isinstance(cached, dict):
        return cached

    code = str(getattr(ir, "code", "") or "")
    tokens: List[str] = []
    call_names: List[str] = []

    def _call_name(expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Name):
            return str(expr.id)
        if isinstance(expr, ast.Attribute):
            parts: List[str] = []
            cur: ast.AST | None = expr
            while isinstance(cur, ast.Attribute):
                parts.append(str(cur.attr))
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(str(cur.id))
            if parts:
                return ".".join(reversed(parts))
        return None

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    tokens.append(f"call:{name}")
                    call_names.append(name)
            elif isinstance(node, ast.BinOp):
                tokens.append(f"binop:{type(node.op).__name__}")
            elif isinstance(node, ast.UnaryOp):
                tokens.append(f"unop:{type(node.op).__name__}")
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    tokens.append(f"cmp:{type(op).__name__}")
            elif isinstance(node, ast.BoolOp):
                tokens.append(f"bool:{type(node.op).__name__}")
            elif isinstance(node, (ast.IfExp, ast.If)):
                tokens.append("if")
            elif isinstance(node, (ast.For, ast.While)):
                tokens.append("loop")
            elif isinstance(node, ast.Return):
                tokens.append("return")
    except Exception:  # noqa: BLE001
        try:
            keep_ops = {"+", "-", "*", "/", "**", "<", ">", "<=", ">=", "==", "!=", "%"}
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type in {
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.ENDMARKER,
                }:
                    continue
                if tok.type in {tokenize.STRING, tokenize.NUMBER}:
                    continue
                if tok.type == tokenize.OP and tok.string not in keep_ops:
                    continue
                if tok.string:
                    tokens.append(tok.string)
        except Exception:  # noqa: BLE001
            tokens = []

    unigrams: set[str] = set(tokens)
    bigrams: set[str] = set()
    for a, b in zip(tokens, tokens[1:]):
        bigrams.add(f"{a}->{b}")

    call_set: set[str] = set(call_names)
    fp = {
        "sig": sig,
        "token_unigrams": unigrams,
        "token_bigrams": bigrams,
        "call_names_set": call_set,
        "call_names_top": sorted(set(call_names))[:32],
    }
    _LOSS_FINGERPRINT_CACHE[sig] = fp
    return fp


def _normalize_family_value(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _family_tags_from_hparams(hparams: Mapping[str, Any] | None, *, keys: Sequence[str]) -> Dict[str, str]:
    hp = hparams if isinstance(hparams, Mapping) else {}
    out: Dict[str, str] = {}
    for key in keys:
        raw = hp.get(str(key)) if isinstance(hp, Mapping) else None
        out[str(key)] = _normalize_family_value(raw)
    return out


def _builder_family_tags(ir: PreferenceBuilderIR) -> Dict[str, str]:
    return _family_tags_from_hparams(getattr(ir, "hyperparams", {}), keys=_BUILDER_FAMILY_KEYS)


def _loss_family_tags(ir: FreeLossIR) -> Dict[str, str]:
    return _family_tags_from_hparams(getattr(ir, "hyperparams", {}), keys=_LOSS_FAMILY_KEYS)


def _family_signature_from_tags(tags: Mapping[str, Any], *, ordered_keys: Sequence[str]) -> str:
    if not isinstance(tags, Mapping):
        return "unknown"
    return "|".join(f"{str(key).replace('_family', '')}={_normalize_family_value(tags.get(str(key)))}" for key in ordered_keys)


def _all_family_tags_unknown(tags: Mapping[str, Any], *, keys: Sequence[str]) -> bool:
    if not isinstance(tags, Mapping):
        return True
    for key in keys:
        if _normalize_family_value(tags.get(str(key))) != "unknown":
            return False
    return True


def _maybe_coarsen_family_signature_str(sig: Any, *, keep_axes: Sequence[str]) -> str:
    """Best-effort migration for older stored family_signature strings.

    Expected format: 'axis=value|axis=value|...'. If parsing fails, returns the original string (or 'unknown').
    """

    text = str(sig or "").strip()
    if not text:
        return "unknown"
    keep = {str(k).strip() for k in keep_axes if str(k).strip()}
    if not keep:
        return text

    out_parts: list[str] = []
    for part in text.split("|"):
        part = str(part or "").strip()
        if not part or "=" not in part:
            continue
        axis, value = part.split("=", 1)
        axis = str(axis or "").strip()
        if axis in keep:
            out_parts.append(f"{axis}={_normalize_family_value(value)}")
    return "|".join(out_parts) if out_parts else text


def _best_pair_artifact_entry(
    *,
    cid: str,
    best_pair: Mapping[str, Any] | None,
    cid_key: str,
    ir_key: str,
    candidate_map: Mapping[str, Any],
    compiled_map: Mapping[str, Any],
    ref_ir_fn: Any,
) -> Dict[str, Any] | None:
    """Resolve a stable artifact entry (with at least {id, ir}) for best_pair ids.

    Priority:
      1) Use embedded IR from best_pair record (g_ir / f_ir) when present.
      2) Use a full stored entry from candidate_map (g_map / f_map).
      3) Use compiled_map[cid].ir (most reliable during a live run).
      4) Fallback to reference IR for g_ref / f_ref when applicable.
    """

    cid_s = str(cid or "").strip()
    if not cid_s:
        return None

    if isinstance(best_pair, Mapping) and str(best_pair.get(cid_key, "")).strip() == cid_s:
        ir = best_pair.get(ir_key)
        if isinstance(ir, Mapping) and isinstance(ir.get("code"), str):
            return {"id": cid_s, "ir": dict(ir)}

    entry = candidate_map.get(cid_s) if isinstance(candidate_map, Mapping) else None
    if isinstance(entry, Mapping) and entry.get("id"):
        ir = entry.get("ir")
        if isinstance(ir, Mapping) and isinstance(ir.get("code"), str):
            out = dict(entry)
            out["id"] = str(out.get("id") or cid_s)
            return out

    comp = compiled_map.get(cid_s) if isinstance(compiled_map, Mapping) else None
    ir = getattr(comp, "ir", None)
    if ir is not None:
        try:
            irj = asdict(ir)
        except Exception:  # noqa: BLE001
            irj = None
        if isinstance(irj, Mapping) and isinstance(irj.get("code"), str):
            return {"id": cid_s, "ir": dict(irj)}

    try:
        ref_ir = ref_ir_fn() if callable(ref_ir_fn) else None
        if ref_ir is not None:
            irj = asdict(ref_ir)
            if isinstance(irj, Mapping) and isinstance(irj.get("code"), str):
                return {"id": cid_s, "ir": dict(irj)}
    except Exception:  # noqa: BLE001
        pass

    return None


def _best_pair_eval_metadata(
    best_pair: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(best_pair, Mapping):
        return {}

    out: Dict[str, Any] = {}
    top_level_aliases = (
        "stage",
        "stage_final",
        "score",
        "final_score",
        "stages_enabled",
        "stages_ran",
        "stages_skipped",
        "pair_ok",
        "pair_reason",
        "compare_target",
        "metric_mode",
        "improve_eps",
        "reference_score",
        "better_than_incumbent",
    )
    for key in top_level_aliases:
        if key not in best_pair:
            continue
        value = best_pair.get(key)
        if isinstance(value, dict):
            out[key] = dict(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value

    alias_map = {
        "generation": "best_pair_generation",
        "phase": "best_pair_phase",
        "last_phase_label": "best_pair_last_phase_label",
        "last_phase_reference_score": "best_pair_last_phase_reference_score",
    }
    for src, dst in alias_map.items():
        if src in best_pair:
            out[dst] = best_pair.get(src)
    return out


def _pair_record_effective_score(rec: Mapping[str, Any] | None) -> float | None:
    if not isinstance(rec, Mapping):
        return None
    for key in ("final_score", "score"):
        try:
            value = float(rec.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _stage3_negative_priority(rec: Mapping[str, Any] | None) -> Tuple[int, int, float] | None:
    if not isinstance(rec, Mapping):
        return None

    nonneg_raw = rec.get("stage3_nonnegative_scenario_count")
    worst_raw = rec.get("stage3_worst_scenario_delta")
    all_neg_raw = rec.get("stage3_all_scenarios_negative")
    if nonneg_raw is not None or worst_raw is not None or all_neg_raw is not None:
        try:
            nonneg = int(nonneg_raw) if nonneg_raw is not None else 0
        except (TypeError, ValueError):
            nonneg = 0
        try:
            worst = float(worst_raw) if worst_raw is not None else float("inf")
        except (TypeError, ValueError):
            worst = float("inf")
        all_neg = bool(all_neg_raw) if all_neg_raw is not None else bool(nonneg == 0)
        return (0 if all_neg else 1, max(0, nonneg), float(worst))

    fitness = rec.get("fitness")
    if not isinstance(fitness, Mapping):
        return None
    per_scenario = fitness.get("per_scenario")
    if not isinstance(per_scenario, Mapping):
        return None

    scenario_scores: List[float] = []
    for item in per_scenario.values():
        if not isinstance(item, Mapping):
            continue
        try:
            delta = float(item.get("delta_mean"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(delta):
            scenario_scores.append(float(delta))
    if not scenario_scores:
        return None

    nonneg = sum(1 for delta in scenario_scores if not (float(delta) < 0.0))
    worst = float(max(scenario_scores))
    return (0 if nonneg == 0 else 1, int(nonneg), float(worst))


def _pair_record_beats_reference_record(
    candidate: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    *,
    metric_mode: str,
    improve_eps: float,
    prefer_all_stage3_scenarios_negative: bool,
) -> bool:
    cand_score = _pair_record_effective_score(candidate)
    ref_score = _pair_record_effective_score(reference)
    if cand_score is None or not math.isfinite(cand_score):
        return False

    if prefer_all_stage3_scenarios_negative:
        cand_priority = _stage3_negative_priority(candidate)
        ref_priority = _stage3_negative_priority(reference)
        if cand_priority is not None and ref_priority is not None and cand_priority != ref_priority:
            return bool(cand_priority < ref_priority)

    return _is_better_than_reference(
        cand_score=float(cand_score),
        reference_score=(float(ref_score) if ref_score is not None else None),
        metric_mode=metric_mode,
        improve_eps=improve_eps,
    )


def _pair_record_sort_key(
    rec: Mapping[str, Any],
    *,
    metric_mode: str,
    prefer_all_stage3_scenarios_negative: bool,
) -> Tuple[Any, ...]:
    score = _pair_record_effective_score(rec)
    score_key = float("inf")
    if score is not None and math.isfinite(score):
        score_key = float(score) if str(metric_mode).strip().lower() == "minimize" else float(-score)

    if prefer_all_stage3_scenarios_negative:
        priority = _stage3_negative_priority(rec)
        if priority is None:
            priority = (2, 10**9, float("inf"))
    else:
        priority = (0, 0, float("-inf"))

    return (
        int(priority[0]),
        int(priority[1]),
        float(priority[2]),
        float(score_key),
    )


def _select_best_valid_pair_record(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    metric_mode: str,
    prefer_all_stage3_scenarios_negative: bool = False,
) -> Dict[str, Any] | None:
    if not records:
        return None

    candidates: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        if not bool(rec.get("pair_ok")):
            continue
        if str(rec.get("stage")) == "anchor":
            continue
        if str(rec.get("g_id")) == G_REF_ID and str(rec.get("f_id")) == F_REF_ID:
            continue
        score = _pair_record_effective_score(rec)
        if score is None or not math.isfinite(score):
            continue
        candidates.append(dict(rec))

    if not candidates:
        return None

    def _stage_rank(rec: Mapping[str, Any]) -> int:
        stage_final = str(rec.get("stage_final", rec.get("stage", ""))).strip().lower()
        return 0 if stage_final == "high_fidelity" else 1

    def _sort_key(rec: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            _stage_rank(rec),
            0 if isinstance(rec.get("fitness"), Mapping) else 1,
            *_pair_record_sort_key(
                rec,
                metric_mode=metric_mode,
                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
            ),
            -_safe_int(rec.get("generation", -1), -1),
            -_safe_int(rec.get("pair_index", -1), -1),
        )

    return dict(min(candidates, key=_sort_key))


def _resolve_best_pair_record(
    *,
    best_so_far: Mapping[str, Any] | None,
    pair_records: Sequence[Mapping[str, Any]] | None,
    pair_cache_records: Sequence[Mapping[str, Any]] | None = None,
    metric_mode: str = "minimize",
    prefer_all_stage3_scenarios_negative: bool = False,
) -> Dict[str, Any] | None:
    if not isinstance(best_so_far, Mapping):
        return None

    gid_best = str(best_so_far.get("builder_id", "")).strip()
    fid_best = str(best_so_far.get("loss_id", "")).strip()
    if not gid_best or not fid_best:
        return None

    target_score = _pair_record_effective_score({"score": best_so_far.get("score")})
    target_generation = _safe_int(best_so_far.get("generation", -1), -1)
    target_phase = str(best_so_far.get("phase", "")).strip()
    target_stage_final = str(best_so_far.get("stage_final", "")).strip()

    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for source in (pair_records or []), (pair_cache_records or []):
        for rec in source:
            if not isinstance(rec, Mapping):
                continue
            if str(rec.get("g_id", "")).strip() != gid_best or str(rec.get("f_id", "")).strip() != fid_best:
                continue
            sig = (
                _safe_int(rec.get("generation", -1), -1),
                _safe_int(rec.get("pair_index", -1), -1),
                str(rec.get("phase", "")),
                str(rec.get("stage", "")),
                str(rec.get("stage_final", "")),
                str(rec.get("eval_budget_signature", "")),
                _pair_record_effective_score(rec),
            )
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append(dict(rec))

    if not candidates:
        return None

    def _stage_rank(rec: Mapping[str, Any]) -> int:
        stage_final = str(rec.get("stage_final", rec.get("stage", ""))).strip().lower()
        return 0 if stage_final == "high_fidelity" else 1

    def _score_matches(rec: Mapping[str, Any]) -> bool:
        cand_score = _pair_record_effective_score(rec)
        if cand_score is None or target_score is None:
            return False
        return abs(cand_score - target_score) <= 1e-12

    def _meta_matches(rec: Mapping[str, Any]) -> bool:
        return (
            _safe_int(rec.get("generation", -1), -1) == target_generation
            and str(rec.get("phase", "")).strip() == target_phase
            and str(rec.get("stage_final", rec.get("stage", ""))).strip() == target_stage_final
        )

    matched = [rec for rec in candidates if _score_matches(rec) or _meta_matches(rec)]
    pool = matched or candidates

    def _sort_key(rec: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            0 if _score_matches(rec) else 1,
            0 if _meta_matches(rec) else 1,
            _stage_rank(rec),
            0 if isinstance(rec.get("fitness"), Mapping) else 1,
            *_pair_record_sort_key(
                rec,
                metric_mode=metric_mode,
                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
            ),
            -_safe_int(rec.get("generation", -1), -1),
            -_safe_int(rec.get("pair_index", -1), -1),
        )

    return dict(min(pool, key=_sort_key))


def _pair_history_key(g_id: Any, f_id: Any) -> str:
    return f"{str(g_id or '').strip()}::{str(f_id or '').strip()}"


def _pair_score_history_entry(rec: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not bool(rec.get("pair_ok")):
        return None
    try:
        final_score = float(rec.get("final_score"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(final_score):
        return None

    out: Dict[str, Any] = {
        "generation": _safe_int(rec.get("generation", -1), -1),
        "pair_index": _safe_int(rec.get("pair_index", -1), -1),
        "phase": str(rec.get("phase", "unknown")),
        "stage": str(rec.get("stage", "unknown")),
        "stage_final": str(rec.get("stage_final", rec.get("stage", "none"))),
        "score": float(final_score),
        "eval_budget_signature": rec.get("eval_budget_signature"),
    }
    try:
        reference_score = rec.get("reference_score")
        if reference_score is not None:
            out["reference_score"] = float(reference_score)
    except (TypeError, ValueError):
        pass
    return out


def _append_pair_score_history(
    pair_score_history_map: Dict[str, List[Dict[str, Any]]],
    rec: Mapping[str, Any],
) -> None:
    key = _pair_history_key(rec.get("g_id"), rec.get("f_id"))
    if key == "::":
        return
    entry = _pair_score_history_entry(rec)
    if entry is None:
        return
    history = pair_score_history_map.setdefault(key, [])
    entry_sig = (
        int(entry.get("generation", -1)),
        int(entry.get("pair_index", -1)),
        str(entry.get("eval_budget_signature")),
        str(entry.get("stage_final")),
        float(entry.get("score")),
    )
    if history:
        last = history[-1]
        last_sig = (
            _safe_int(last.get("generation", -1), -1),
            _safe_int(last.get("pair_index", -1), -1),
            str(last.get("eval_budget_signature")),
            str(last.get("stage_final")),
            float(last.get("score", float("nan"))),
        )
        if last_sig == entry_sig:
            return
    history.append(entry)


def _rebuild_pair_score_history_map(
    records: Sequence[Mapping[str, Any]] | None,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records or []:
        if not isinstance(rec, Mapping):
            continue
        _append_pair_score_history(out, rec)
    return out


def _score_history_summary(history: Sequence[Mapping[str, Any]] | None) -> Dict[str, Any]:
    items = list(history or [])
    scores: List[float] = []
    for item in items:
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(score):
            scores.append(score)
    if not scores:
        return {"count": 0, "scores": []}
    return {
        "count": int(len(scores)),
        "scores": list(scores),
        "best": float(min(scores)),
        "worst": float(max(scores)),
        "mean": float(sum(scores) / len(scores)),
        "latest": float(scores[-1]),
    }


def _builder_family_signature(ir_or_entry: Any) -> str:
    if isinstance(ir_or_entry, PreferenceBuilderIR):
        tags = _builder_family_tags(ir_or_entry)
    elif isinstance(ir_or_entry, Mapping):
        ir_raw = ir_or_entry.get("ir")
        if isinstance(ir_raw, dict):
            try:
                tags = _builder_family_tags(pref_builder_ir_from_json(ir_raw))
            except Exception:  # noqa: BLE001
                tags = _family_tags_from_hparams(ir_raw.get("hyperparams", {}), keys=_BUILDER_FAMILY_KEYS)
        else:
            tags = _family_tags_from_hparams(ir_or_entry.get("hyperparams", {}), keys=_BUILDER_FAMILY_KEYS)
    else:
        return "unknown"
    if isinstance(ir_or_entry, Mapping) and _all_family_tags_unknown(tags, keys=_BUILDER_FAMILY_KEYS):
        # Fallback for older checkpoints that only stored the signature string.
        return _maybe_coarsen_family_signature_str(
            ir_or_entry.get("family_signature"),
            keep_axes=("geometry", "cap", "constraint"),
        )
    return _family_signature_from_tags(tags, ordered_keys=_BUILDER_FAMILY_KEYS)


def _loss_family_signature(ir_or_entry: Any) -> str:
    if isinstance(ir_or_entry, FreeLossIR):
        tags = _loss_family_tags(ir_or_entry)
    elif isinstance(ir_or_entry, Mapping):
        ir_raw = ir_or_entry.get("ir")
        if isinstance(ir_raw, dict):
            try:
                tags = _loss_family_tags(free_loss_ir_from_json(ir_raw))
            except Exception:  # noqa: BLE001
                tags = _family_tags_from_hparams(ir_raw.get("hyperparams", {}), keys=_LOSS_FAMILY_KEYS)
        else:
            tags = _family_tags_from_hparams(ir_or_entry.get("hyperparams", {}), keys=_LOSS_FAMILY_KEYS)
    else:
        return "unknown"
    if isinstance(ir_or_entry, Mapping) and _all_family_tags_unknown(tags, keys=_LOSS_FAMILY_KEYS):
        return _maybe_coarsen_family_signature_str(
            ir_or_entry.get("family_signature"),
            keep_axes=("paradigm", "signal", "agg", "constraint"),
        )
    return _family_signature_from_tags(tags, ordered_keys=_LOSS_FAMILY_KEYS)


def _loss_family_id(ir: FreeLossIR) -> str:
    """Coarse 'family' label for diversity quotas and parent sampling."""

    tags = _loss_family_tags(ir)
    if any(tags.get(k) not in {"", "unknown"} for k in _LOSS_FAMILY_KEYS):
        return "|".join(
            [
                _normalize_family_value(tags.get("paradigm_family")),
                _normalize_family_value(tags.get("signal_family")),
                _normalize_family_value(tags.get("agg_family")),
                _normalize_family_value(tags.get("constraint_family")),
            ]
        )

    mode = "pairwise"
    expects: List[str] = []
    try:
        hint = getattr(ir, "implementation_hint", None)
        if hint is not None:
            mode = str(getattr(hint, "mode", mode) or mode)
            exp = getattr(hint, "expects", None)
            if isinstance(exp, (list, tuple)):
                expects = [str(x) for x in exp if str(x)]
    except Exception:  # noqa: BLE001
        expects = []

    fp = _loss_fingerprint(ir)
    calls = fp.get("call_names_set") or set()
    if not isinstance(calls, set):
        calls = set()

    ops_calls = {c[4:] for c in calls if isinstance(c, str) and c.startswith("ops.")}
    link = "other"
    for cand in ("logsigmoid", "softplus", "sigmoid", "tanh"):
        if cand in ops_calls:
            link = cand
            break
    if link == "other" and "relu" in ops_calls:
        link = "hinge"

    signal = "none"
    exp_set = set(expects)
    if any(k in exp_set for k in ("delta_z", "obj_z")):
        signal = "delta_z"
    elif "delta_rank" in exp_set or "rank" in exp_set:
        signal = "delta_rank"
    elif any(k in exp_set for k in ("delta_regret", "regret")):
        signal = "delta_regret"
    elif ("cost_a" in exp_set) or ("cost_b" in exp_set) or ("objective" in exp_set):
        signal = "cost_or_obj"

    norm = "raw"
    for cand in ("zscore", "normalize", "standardize"):
        if cand in ops_calls:
            norm = cand
            break

    return f"{mode}:{link}:{signal}:{norm}"


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    if u <= 0:
        return 0.0
    return float(len(a & b)) / float(u)


def _check_loss_novelty(
    *,
    candidate: FreeLossIR,
    bank: Sequence[Mapping[str, Any]],
    max_similarity: float,
    neighbors: int = 3,
) -> tuple[bool, Dict[str, Any]]:
    """Return (ok, meta_or_failure_payload)."""

    if not bank:
        return True, {"max_similarity": 0.0, "neighbors": []}

    cand_fp = _loss_fingerprint(candidate)
    cand_bigrams = cand_fp.get("token_bigrams") or set()
    cand_unigrams = cand_fp.get("token_unigrams") or set()
    cand_calls = cand_fp.get("call_names_set") or set()
    if not isinstance(cand_bigrams, set):
        cand_bigrams = set()
    if not isinstance(cand_unigrams, set):
        cand_unigrams = set()
    if not isinstance(cand_calls, set):
        cand_calls = set()

    scored: List[Dict[str, Any]] = []
    best = 0.0
    best_parts = {"bigram": 0.0, "unigram": 0.0, "call": 0.0}
    for e in bank:
        bt = e.get("token_bigrams")
        ut = e.get("token_unigrams")
        ct = e.get("call_names_set")
        if not isinstance(bt, set) or not isinstance(ut, set) or not isinstance(ct, set):
            continue
        sim_big = _jaccard(cand_bigrams, bt)
        sim_uni = _jaccard(cand_unigrams, ut)
        sim_call = _jaccard(cand_calls, ct)
        sim = float(max(sim_big, sim_uni, sim_call))
        if sim > best:
            best = float(sim)
            best_parts = {"bigram": float(sim_big), "unigram": float(sim_uni), "call": float(sim_call)}
        scored.append(
            {
                "sig": str(e.get("sig", "")),
                "name": str(e.get("name", "")),
                "similarity": float(sim),
                "similarity_parts": {"bigram": float(sim_big), "unigram": float(sim_uni), "call": float(sim_call)},
                "call_names_top": list(e.get("call_names_top") or [])[:16],
            }
        )

    scored.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)
    top = scored[: max(1, int(neighbors))]

    if float(best) >= float(max_similarity):
        return (
            False,
            {
                "stage": "novelty",
                "max_similarity": float(max_similarity),
                "best_similarity": float(best),
                "best_similarity_parts": dict(best_parts),
                "too_similar_to": top,
            },
        )

    return True, {"max_similarity": float(best), "best_similarity_parts": dict(best_parts), "neighbors": top}


def _select_elites_by_family(
    *,
    ranked: Sequence[Mapping[str, Any]],
    total: int,
    enabled: bool,
    elite_per_family: int,
    elite_max_per_family: int,
) -> List[Dict[str, Any]]:
    total = max(0, int(total))
    if total <= 0:
        return []
    if not bool(enabled):
        return [dict(x) for x in list(ranked)[:total]]
    return _select_elites_with_family_quota(
        ranked=ranked,
        elite_n=total,
        metric_mode="minimize",
        min_per_family=max(1, int(elite_per_family)),
        max_per_family=max(1, int(elite_max_per_family)),
        include_unknown=True,
    )


def _select_elites_with_family_quota(
    ranked: Sequence[Mapping[str, Any]],
    elite_n: int,
    *,
    metric_mode: str,
    min_per_family: int = 1,
    max_per_family: int | None = None,
    include_unknown: bool = False,
    prefer_selection_sort_key: bool = False,
) -> List[Dict[str, Any]]:
    elite_n = max(0, int(elite_n))
    if elite_n <= 0:
        return []
    min_per_family = max(1, int(min_per_family))

    fam_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for item in ranked:
        fam = str(item.get("family_signature") or item.get("family") or "unknown")
        fam_to_items.setdefault(fam, []).append(dict(item))

    protected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    fam_counts: collections.Counter[str] = collections.Counter()

    for fam, items in fam_to_items.items():
        if fam == "unknown" and not bool(include_unknown):
            continue
        for item in items[:min_per_family]:
            eid = str(item.get("id") or "")
            if eid and eid in selected_ids:
                continue
            protected.append(dict(item))
            if eid:
                selected_ids.add(eid)
            fam_counts[fam] += 1

    if bool(prefer_selection_sort_key):
        protected.sort(key=lambda x: _stored_selection_sort_key(x, fallback_key="fitness"))
    else:
        protected.sort(
            key=lambda x: float(x.get("fitness", float("-inf") if str(metric_mode) == "maximize" else float("inf"))),
            reverse=bool(str(metric_mode) == "maximize"),
        )

    selected: List[Dict[str, Any]] = []
    for item in protected:
        if len(selected) >= elite_n:
            break
        selected.append(dict(item))

    for item in ranked:
        if len(selected) >= elite_n:
            break
        eid = str(item.get("id") or "")
        if eid and eid in selected_ids:
            continue
        fam = str(item.get("family_signature") or item.get("family") or "unknown")
        if max_per_family is not None and int(fam_counts.get(fam, 0)) >= int(max_per_family):
            continue
        selected.append(dict(item))
        if eid:
            selected_ids.add(eid)
        fam_counts[fam] += 1

    return selected[:elite_n]


def _select_resident_population(
    ranked: Sequence[Mapping[str, Any]],
    population_n: int,
    *,
    metric_mode: str,
    family_diversity_cfg: Mapping[str, Any] | None = None,
    prefer_selection_sort_key: bool = False,
) -> List[Dict[str, Any]]:
    population_n = max(0, int(population_n))
    if population_n <= 0:
        return []
    cfg = family_diversity_cfg if isinstance(family_diversity_cfg, Mapping) else {}
    if bool(cfg.get("enabled", False)):
        return _select_elites_with_family_quota(
            ranked,
            population_n,
            metric_mode=metric_mode,
            min_per_family=int(cfg.get("min_per_family", 1) or 1),
            max_per_family=(
                int(cfg.get("elite_max_per_family", 0) or 0)
                if int(cfg.get("elite_max_per_family", 0) or 0) > 0
                else None
            ),
            include_unknown=bool(cfg.get("include_unknown", False)),
            prefer_selection_sort_key=bool(prefer_selection_sort_key),
        )
    return [dict(x) for x in list(ranked)[:population_n]]


def _sig_pref_builder(ir: PreferenceBuilderIR) -> str:
    return _sig(asdict(ir))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _feature_cache_device(feature_cache: Mapping[str, Any] | None) -> torch.device | None:
    if not isinstance(feature_cache, Mapping):
        return None
    for v in feature_cache.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return None


def _build_pref_batch_with_memory_trace(
    builder: Any,
    feature_cache: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> PrefBatch:
    dev = _feature_cache_device(feature_cache)
    trace: Dict[str, Any] = {
        "memory_metric_available": False,
        "memory_device": (str(dev) if dev is not None else None),
    }

    if dev is not None and str(dev).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(dev)
            alloc_before = int(torch.cuda.memory_allocated(dev))
            reserved_before = int(torch.cuda.memory_reserved(dev))
            torch.cuda.reset_peak_memory_stats(dev)
            pref = builder.build_fn(feature_cache, dict(extra or {}))
            torch.cuda.synchronize(dev)
            peak_allocated = int(torch.cuda.max_memory_allocated(dev))
            peak_reserved = int(torch.cuda.max_memory_reserved(dev))
            trace.update(
                {
                    "memory_metric_available": True,
                    "memory_allocated_before_bytes": int(alloc_before),
                    "memory_reserved_before_bytes": int(reserved_before),
                    "memory_peak_allocated_bytes": int(peak_allocated),
                    "memory_peak_reserved_bytes": int(peak_reserved),
                    "memory_peak_allocated_delta_bytes": int(max(0, peak_allocated - alloc_before)),
                    "memory_peak_reserved_delta_bytes": int(max(0, peak_reserved - reserved_before)),
                    "memory_peak_allocated_delta_mb": float(max(0, peak_allocated - alloc_before)) / float(1024**2),
                    "memory_peak_reserved_delta_mb": float(max(0, peak_reserved - reserved_before)) / float(1024**2),
                }
            )
        except Exception as exc:  # noqa: BLE001
            pref = builder.build_fn(feature_cache, dict(extra or {}))
            trace["memory_metric_error"] = str(exc)
    else:
        pref = builder.build_fn(feature_cache, dict(extra or {}))

    if isinstance(pref, PrefBatch):
        meta = dict(pref.meta or {})
        meta["builder_memory_trace"] = dict(trace)
        pref.meta = meta
    return pref


def _enrich_builder_gate_trace_with_memory(
    trace: Mapping[str, Any] | None,
    pref_batch: PrefBatch | None,
    *,
    cache_hit: bool | None = None,
) -> Dict[str, Any] | None:
    if trace is None and pref_batch is None and cache_hit is None:
        return None
    out: Dict[str, Any] = dict(trace or {})
    if cache_hit is not None:
        out["builder_pref_cache_hit"] = bool(cache_hit)
    if isinstance(pref_batch, PrefBatch) and isinstance(pref_batch.meta, dict):
        mem = pref_batch.meta.get("builder_memory_trace")
        if isinstance(mem, Mapping):
            out.update(dict(mem))
    return out


def _std(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / float(len(vals) - 1)
    return float(math.sqrt(max(var, 0.0)))


def _calibrate_improve_eps_from_baseline_noise(
    *,
    cfg_yaml: Mapping[str, Any],
    operator_whitelist: Sequence[str],
    device_str: str,
) -> Dict[str, Any] | None:
    """Estimate baseline mini-train noise and return a calibrated improve_eps.

    Multiseed calibration is disabled; keep the configured improve_eps and bind stage3 to one seed.
    """

    calib_raw = cfg_yaml.get("improve_eps_calibration", {}) or {}
    if not isinstance(calib_raw, dict):
        return None
    if not bool(calib_raw.get("enabled", False)):
        return None

    training_seed = int(_resolve_training_seed(cfg_yaml))
    eps = float(cfg_yaml.get("improve_eps", 0.0) or 0.0)
    return {
        "enabled": False,
        "mode": "fixed_single_seed",
        "fidelity": _stage3_fidelity_key(cfg_yaml),
        "training_seed": int(training_seed),
        "improve_eps": float(eps),
        "reason": "multiseed_disabled",
    }


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _run_co_alignment_gates_for_loss(
    compiled_f: CompiledFreeLoss,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    co_enabled = _safe_bool(cfg.get("co_gate_enabled", True), True)
    hidden_enabled = _safe_bool(cfg.get("co_gate_hidden_variant", False), False)

    out: Dict[str, Any] = {
        "co_enabled": bool(co_enabled),
        "co_ok": True,
        "co_reason": "disabled" if not co_enabled else "ok",
        "co_failed_gate": None,
        "co_failure_kind": None,
        "co_sensitivity_ok": None,
        "co_invariance_ok": None,
        "co_sensitivity_visible_ok": None,
        "co_sensitivity_visible_reason": None,
        "co_sensitivity_visible_abs_delta": None,
        "co_sensitivity_visible_rel_delta": None,
        "co_sensitivity_visible_trace": None,
        "co_sensitivity_hidden_ok": None,
        "co_sensitivity_hidden_reason": None,
        "co_sensitivity_hidden_abs_delta": None,
        "co_sensitivity_hidden_rel_delta": None,
        "co_sensitivity_hidden_trace": None,
        "co_invariance_visible_ok": None,
        "co_invariance_visible_reason": None,
        "co_invariance_visible_abs_delta": None,
        "co_invariance_visible_rel_delta": None,
        "co_invariance_visible_trace": None,
        "co_invariance_hidden_ok": None,
        "co_invariance_hidden_reason": None,
        "co_invariance_hidden_abs_delta": None,
        "co_invariance_hidden_rel_delta": None,
        "co_invariance_hidden_trace": None,
    }
    if not co_enabled:
        return out

    sens_vis: ObjectiveSensitivityGateResult = run_objective_sensitivity_gate(
        compiled_f,
        min_abs_delta=_safe_float(cfg.get("co_sensitivity_min_abs_delta", 1e-3), 1e-3),
        min_rel_delta=_safe_float(cfg.get("co_sensitivity_min_rel_delta", 1e-2), 1e-2),
        variant="visible",
    )
    inv_vis: AffineInvarianceGateResult = run_affine_invariance_gate(
        compiled_f,
        max_abs_delta=_safe_float(cfg.get("co_invariance_max_abs_delta", 1e-3), 1e-3),
        max_rel_delta=_safe_float(cfg.get("co_invariance_max_rel_delta", 1e-2), 1e-2),
        variant="visible",
    )

    sens_hid: ObjectiveSensitivityGateResult | None = None
    inv_hid: AffineInvarianceGateResult | None = None

    if hidden_enabled and sens_vis.ok and inv_vis.ok:
        sens_hid = run_objective_sensitivity_gate(
            compiled_f,
            min_abs_delta=_safe_float(cfg.get("co_sensitivity_min_abs_delta", 1e-3), 1e-3),
            min_rel_delta=_safe_float(cfg.get("co_sensitivity_min_rel_delta", 1e-2), 1e-2),
            variant="hidden",
        )
        inv_hid = run_affine_invariance_gate(
            compiled_f,
            max_abs_delta=_safe_float(cfg.get("co_invariance_max_abs_delta", 1e-3), 1e-3),
            max_rel_delta=_safe_float(cfg.get("co_invariance_max_rel_delta", 1e-2), 1e-2),
            variant="hidden",
        )

    out.update(
        {
            "co_enabled": True,
            "co_sensitivity_ok": bool(sens_vis.ok) and (True if sens_hid is None else bool(sens_hid.ok)),
            "co_invariance_ok": bool(inv_vis.ok) and (True if inv_hid is None else bool(inv_hid.ok)),
            "co_sensitivity_visible_ok": bool(sens_vis.ok),
            "co_sensitivity_visible_reason": str(sens_vis.reason),
            "co_sensitivity_visible_abs_delta": sens_vis.abs_delta,
            "co_sensitivity_visible_rel_delta": sens_vis.rel_delta,
            "co_sensitivity_visible_trace": sens_vis.trace,
            "co_sensitivity_hidden_ok": None if sens_hid is None else bool(sens_hid.ok),
            "co_sensitivity_hidden_reason": None if sens_hid is None else str(sens_hid.reason),
            "co_sensitivity_hidden_abs_delta": None if sens_hid is None else sens_hid.abs_delta,
            "co_sensitivity_hidden_rel_delta": None if sens_hid is None else sens_hid.rel_delta,
            "co_sensitivity_hidden_trace": None if sens_hid is None else sens_hid.trace,
            "co_invariance_visible_ok": bool(inv_vis.ok),
            "co_invariance_visible_reason": str(inv_vis.reason),
            "co_invariance_visible_abs_delta": inv_vis.abs_delta,
            "co_invariance_visible_rel_delta": inv_vis.rel_delta,
            "co_invariance_visible_trace": inv_vis.trace,
            "co_invariance_hidden_ok": None if inv_hid is None else bool(inv_hid.ok),
            "co_invariance_hidden_reason": None if inv_hid is None else str(inv_hid.reason),
            "co_invariance_hidden_abs_delta": None if inv_hid is None else inv_hid.abs_delta,
            "co_invariance_hidden_rel_delta": None if inv_hid is None else inv_hid.rel_delta,
            "co_invariance_hidden_trace": None if inv_hid is None else inv_hid.trace,
        }
    )

    co_ok = bool(out["co_sensitivity_ok"]) and bool(out["co_invariance_ok"])
    out["co_ok"] = co_ok
    if co_ok:
        out["co_reason"] = "ok"
        return out

    if not sens_vis.ok:
        failed_gate = "ObjectiveSensitivity"
        reason = str(sens_vis.reason)
        trace = sens_vis.trace if isinstance(sens_vis.trace, dict) else {}
    elif not inv_vis.ok:
        failed_gate = "AffineInvariance"
        reason = str(inv_vis.reason)
        trace = inv_vis.trace if isinstance(inv_vis.trace, dict) else {}
    elif sens_hid is not None and not sens_hid.ok:
        failed_gate = "ObjectiveSensitivity"
        reason = str(sens_hid.reason)
        trace = sens_hid.trace if isinstance(sens_hid.trace, dict) else {}
    else:
        failed_gate = "AffineInvariance"
        reason = str(inv_hid.reason if inv_hid is not None else "co_gate_failed")
        trace = inv_hid.trace if (inv_hid is not None and isinstance(inv_hid.trace, dict)) else {}

    out["co_reason"] = reason
    out["co_failed_gate"] = failed_gate
    out["co_failure_kind"] = trace.get("failure_kind") if isinstance(trace, dict) else None
    if out["co_failure_kind"] is None and reason:
        out["co_failure_kind"] = str(reason)
    return out


def _normalize_metric_mode(value: Any) -> str:
    mode = str(value or "minimize").strip().lower()
    if mode not in {"minimize", "maximize"}:
        return "minimize"
    return mode


def _normalize_search_mode(value: Any, *, default_mode: str) -> str:
    mode = str(value or default_mode).strip().lower()
    if mode not in {"alternating", "coevo", "loss_only"}:
        return str(default_mode)
    return mode


def _uses_fixed_side_search(search_mode: Any) -> bool:
    return str(search_mode or "").strip().lower() in {"alternating", "loss_only"}


def _normalize_operator_name(name: str, side: str) -> str:
    raw = str(name or "").strip().upper()
    mapping = {
        "GEN": "GEN",
        "GENERATE": "GEN",
        "E1_GENERATE": "GEN",
        "XOVER": "XOVER",
        "CROSSOVER": "XOVER",
        "E1": "XOVER",
        "MUTATE": "MUTATE",
        "MUTATION": "MUTATE",
        "M1": "MUTATE",
        "TUNE": "TUNE",
        "M2": "TUNE",
        "PARADIGM_SHIFT": "PARADIGM_SHIFT",
        "STRUCTURE_SHIFT": "STRUCTURE_SHIFT",
        "CONSTRAINT_INJECT": "CONSTRAINT_INJECT",
    }
    if str(side) == "loss" and raw == "AGG_SHIFT":
        return "STRUCTURE_SHIFT"
    if str(side) == "builder" and raw == "CAP_SHIFT":
        return "STRUCTURE_SHIFT"
    return mapping.get(raw, raw)


def _expand_operator_bank(side_cfg: Mapping[str, Any], generation: int, rng: random.Random, *, side: str) -> List[str]:
    bank = side_cfg.get("operator_bank", {}) if isinstance(side_cfg, Mapping) else {}
    if not isinstance(bank, Mapping):
        return []
    section = bank.get("init" if int(generation) <= 0 else "per_gen", [])
    if not isinstance(section, Sequence) or isinstance(section, (str, bytes)):
        return []
    plan: List[str] = []
    for item in section:
        if not isinstance(item, Mapping):
            continue
        op = _normalize_operator_name(str(item.get("name", "")), side=side)
        if op in {"", "ELITE", "M3", "REPAIR"}:
            continue
        count = max(0, _safe_int(item.get("count", 0), 0))
        if count <= 0:
            continue
        plan.extend([op] * count)
    rng.shuffle(plan)
    return plan


def _normalize_family_diversity_cfg(raw: Any) -> Dict[str, Any]:
    cfg = dict(raw) if isinstance(raw, dict) else {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "min_per_family": int(cfg.get("min_per_family", cfg.get("elite_per_family", 1)) or 1),
        "elite_max_per_family": int(cfg.get("elite_max_per_family", 0) or 0),
        "parent_max_per_family": int(cfg.get("parent_max_per_family", 0) or 0),
        "include_unknown": bool(cfg.get("include_unknown", False)),
    }


def _normalize_stage3_multifidelity_cfg(raw: Any) -> Dict[str, Any]:
    cfg = dict(raw) if isinstance(raw, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    rounds_raw = cfg.get("rounds", cfg.get("fidelities", [])) or []
    if not isinstance(rounds_raw, list):
        rounds_raw = []
    rounds: List[Dict[str, Any]] = []
    for r in rounds_raw:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        rounds.append(rr)
    if enabled and not rounds:
        # Default: quick filter then a stronger confirmation.
        rounds = [
            {
                "name": "K200",
                "f1_steps": 200,
                "promote_top_m": 32,
                "promote_if_better_than_incumbent": True,
                "promote_selection_mode": "union",
                "always_include_incumbent": True,
            },
            {"name": "K1000", "f1_steps": 1000, "promote_top_m": 0},
        ]
    return {"enabled": bool(enabled), "rounds": rounds}


def _apply_stage3_round_overrides(cfg_yaml: Mapping[str, Any], round_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(cfg_yaml)
    # Budget overrides (step-mode and/or epoch-mode).
    if "f1_steps" in round_cfg and round_cfg.get("f1_steps") is not None:
        out["f1_steps"] = int(round_cfg.get("f1_steps") or out.get("f1_steps", 32) or 32)
    if "hf_epochs" in round_cfg and round_cfg.get("hf_epochs") is not None:
        out["hf_epochs"] = int(round_cfg.get("hf_epochs") or 0)
    if "hf_instances_per_epoch" in round_cfg and round_cfg.get("hf_instances_per_epoch") is not None:
        out["hf_instances_per_epoch"] = int(round_cfg.get("hf_instances_per_epoch") or 0)

    # Convenience: allow a round to specify K via "K".
    if "K" in round_cfg and round_cfg.get("K") is not None:
        out["f1_steps"] = int(round_cfg.get("K") or out.get("f1_steps", 32) or 32)
        out["hf_epochs"] = 0
        out["hf_instances_per_epoch"] = 0

    return out


def _select_stage3_promotions(
    records: Sequence[Mapping[str, Any]],
    *,
    promote_top_m: int,
    promote_top_frac: float | None,
    promote_if_better_than_incumbent: bool,
    promote_selection_mode: str,
    promote_only_if_better_than_baseline: bool,
    promote_baseline_mode: str,
    incumbent_ref_score: float | None,
    incumbent_ref_record: Mapping[str, Any] | None,
    metric_mode: str,
    improve_eps: float,
    always_include_pair: Tuple[str, str] | None,
    prefer_all_stage3_scenarios_negative: bool = False,
) -> List[Tuple[str, str]]:
    scored: List[Dict[str, Any]] = []
    better: List[Tuple[str, str]] = []
    better_set: set[Tuple[str, str]] = set()
    baseline_gate_mode = str(promote_baseline_mode or "mean").strip().lower()
    selection_mode = str(promote_selection_mode or "union").strip().lower()
    if selection_mode not in {"union", "intersection"}:
        selection_mode = "union"
    eligible_pairs: set[Tuple[str, str]] = set()
    for r in records:
        if not bool(r.get("pair_ok")):
            continue
        gid = str(r.get("g_id", ""))
        fid = str(r.get("f_id", ""))
        try:
            s = float(r.get("final_score", r.get("score")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(s):
            continue
        baseline_ok = True
        if bool(promote_only_if_better_than_baseline):
            if baseline_gate_mode == "strict":
                baseline_ok = bool(r.get("better_than_baseline_strict"))
            else:
                baseline_ok = bool(r.get("better_than_baseline_mean"))
        if not baseline_ok:
            continue
        eligible_pairs.add((gid, fid))
        scored.append(dict(r))
        if promote_if_better_than_incumbent:
            ref_record: Mapping[str, Any] | None = incumbent_ref_record
            if not isinstance(ref_record, Mapping) and incumbent_ref_score is not None:
                ref_record = {"score": float(incumbent_ref_score)}
            if _pair_record_beats_reference_record(
                dict(r),
                ref_record,
                metric_mode=str(metric_mode),
                improve_eps=float(improve_eps),
                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
            ):
                pair_key = (gid, fid)
                better.append(pair_key)
                better_set.add(pair_key)

    scored.sort(
        key=lambda rec: _pair_record_sort_key(
            rec,
            metric_mode=metric_mode,
            prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
        )
    )
    promoted: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    if always_include_pair is not None:
        if always_include_pair in eligible_pairs and always_include_pair not in seen:
            promoted.append(always_include_pair)
            seen.add(always_include_pair)

    if selection_mode == "union":
        for gid, fid in better:
            k = (gid, fid)
            if k in seen:
                continue
            promoted.append(k)
            seen.add(k)

    m = _resolve_stage3_promotion_count(
        total_candidates=int(len(scored)),
        promote_top_m=int(promote_top_m),
        promote_top_frac=promote_top_frac,
    )
    if m > 0:
        top_candidates = scored[:m]
        if selection_mode == "intersection" and bool(promote_if_better_than_incumbent):
            top_candidates = [
                rec
                for rec in top_candidates
                if (str(rec.get("g_id", "")), str(rec.get("f_id", ""))) in better_set
            ]
        for rec in top_candidates:
            gid = str(rec.get("g_id", ""))
            fid = str(rec.get("f_id", ""))
            k = (gid, fid)
            if k in seen:
                continue
            promoted.append(k)
            seen.add(k)

    return promoted


def _select_stage3_builder_promotions(
    records: Sequence[Mapping[str, Any]],
    *,
    promote_top_m: int,
    promote_top_frac: float | None,
    metric_mode: str,
    slack: float,
    fixed_loss_id: str | None,
    always_include_pair: Tuple[str, str] | None,
) -> List[Tuple[str, str]]:
    filtered: List[Mapping[str, Any]] = []
    for r in records:
        if not bool(r.get("pair_ok")):
            continue
        if fixed_loss_id and str(r.get("f_id") or "") != str(fixed_loss_id):
            continue
        filtered.append(r)

    perf_by_builder: Dict[str, float] = {}
    for r in filtered:
        gid = str(r.get("g_id") or "")
        if not gid:
            continue
        try:
            score_f = float(r.get("final_score", r.get("score")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score_f):
            continue
        prev = perf_by_builder.get(gid)
        if prev is None or _is_better_than_reference(
            cand_score=float(score_f),
            reference_score=float(prev),
            metric_mode=metric_mode,
            improve_eps=0.0,
        ):
            perf_by_builder[gid] = float(score_f)

    state = _compute_builder_constraint_state(
        records=filtered,
        perf_by_builder=perf_by_builder,
        metric_mode=metric_mode,
        slack=slack,
    )
    promoted: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    if always_include_pair is not None:
        promoted.append(always_include_pair)
        seen.add(always_include_pair)

    m = _resolve_stage3_promotion_count(
        total_candidates=int(len(list(state.get("feasible") or []))),
        promote_top_m=int(promote_top_m),
        promote_top_frac=promote_top_frac,
    )
    for stat in list(state.get("feasible") or [])[:m]:
        ref = stat.get("perf_ref")
        if not isinstance(ref, Mapping):
            continue
        pair = (str(ref.get("g_id") or stat.get("builder_id") or ""), str(ref.get("f_id") or fixed_loss_id or ""))
        if not pair[0] or not pair[1] or pair in seen:
            continue
        promoted.append(pair)
        seen.add(pair)

    return promoted


def _resolve_stage3_promotion_count(
    *,
    total_candidates: int,
    promote_top_m: int,
    promote_top_frac: float | None,
) -> int:
    total = max(int(total_candidates), 0)
    if promote_top_frac is not None:
        try:
            frac = float(promote_top_frac)
        except (TypeError, ValueError):
            frac = 0.0
        if math.isfinite(frac):
            frac = min(max(float(frac), 0.0), 1.0)
            if frac <= 0.0 or total <= 0:
                return 0
            return max(1, int(math.ceil(float(total) * float(frac))))
    return max(int(promote_top_m), 0)


def _cap_parent_pool_by_family(
    ranked_parents: Sequence[Tuple[float, str, Any, Mapping[str, Any]]],
    *,
    side: str,
    max_per_family: int,
    include_unknown: bool,
) -> List[Tuple[float, str, Any, Mapping[str, Any]]]:
    if int(max_per_family) <= 0:
        return list(ranked_parents)
    counts: collections.Counter[str] = collections.Counter()
    out: List[Tuple[float, str, Any, Mapping[str, Any]]] = []
    for item in ranked_parents:
        fam = (
            _builder_family_signature(item[3])
            if str(side) == "builder"
            else _loss_family_signature(item[3])
        )
        if fam == "unknown" and not bool(include_unknown):
            continue
        if int(counts[fam]) >= int(max_per_family):
            continue
        out.append(item)
        counts[fam] += 1
    return out if out else list(ranked_parents)


def _sample_cross_family_parents(
    rng: random.Random,
    ranked_parents: Sequence[Tuple[float, str, Any, Mapping[str, Any]]],
    *,
    side: str,
    k: int,
) -> Tuple[List[Tuple[float, str, Any, Mapping[str, Any]]], bool]:
    if not ranked_parents:
        return [], False
    sig_fn = _builder_family_signature if str(side) == "builder" else _loss_family_signature
    by_sig: Dict[str, List[Tuple[float, str, Any, Mapping[str, Any]]]] = {}
    for item in ranked_parents:
        by_sig.setdefault(sig_fn(item[3]), []).append(item)
    if len(by_sig) >= 2:
        ordered_sigs = list(by_sig.keys())
        chosen_sigs = _rank_weighted_sample_without_replacement(rng, ordered_sigs, k=min(len(ordered_sigs), max(2, int(k))))
        chosen: List[Tuple[float, str, Any, Mapping[str, Any]]] = []
        for sig in chosen_sigs[:2]:
            chosen.append(by_sig[sig][0])
        remaining_pool = [it for it in ranked_parents if str(it[1]) not in {str(x[1]) for x in chosen}]
        if int(k) > len(chosen):
            chosen.extend(_rank_weighted_sample_without_replacement(rng, remaining_pool, k=min(int(k) - len(chosen), len(remaining_pool))))
        return chosen[: max(2, int(k))], False
    chosen = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=min(max(2, int(k)), len(ranked_parents)))
    return chosen, True


def _majority_parent_tags(
    parent_irs: Sequence[Any],
    *,
    keys: Sequence[str],
    kind: str,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in keys:
        counts: collections.Counter[str] = collections.Counter()
        ordered: List[str] = []
        for ir in parent_irs:
            if str(kind) == "builder":
                tags = _builder_family_tags(ir)
            else:
                tags = _loss_family_tags(ir)
            val = _normalize_family_value(tags.get(str(key)))
            if val == "unknown":
                continue
            counts[val] += 1
            ordered.append(val)
        if not counts:
            out[str(key)] = "unknown"
            continue
        top_n = max(counts.values())
        tied = {k for k, v in counts.items() if int(v) == int(top_n)}
        picked = next((v for v in ordered if v in tied), "unknown")
        out[str(key)] = picked
    return out


def _missing_required_family_tags(tags: Mapping[str, Any], *, keys: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for key in keys:
        if _normalize_family_value(tags.get(str(key))) == "unknown":
            missing.append(str(key))
    return missing


def _normalize_eval_stages(cfg: Mapping[str, Any]) -> Dict[str, bool]:
    raw = cfg.get("eval_stages", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    stage0_gate = _safe_bool(raw.get("stage0_gate", cfg.get("cheap_gate_on", True)), True)
    stage1_proxy = _safe_bool(raw.get("stage1_proxy", True), True)
    stage2_micro = _safe_bool(raw.get("stage2_micro_unroll", cfg.get("micro_unroll_enabled", False)), False)
    stage3_hf = _safe_bool(raw.get("stage3_high_fidelity", cfg.get("high_fidelity_on", True)), True)
    return {
        "stage0_gate": bool(stage0_gate),
        "stage1_proxy": bool(stage1_proxy),
        "stage2_micro_unroll": bool(stage2_micro),
        "stage3_high_fidelity": bool(stage3_hf),
    }


def _format_eval_stages_for_log(eval_stages: Mapping[str, bool]) -> str:
    return (
        f"stage0_gate={1 if bool(eval_stages.get('stage0_gate', False)) else 0} "
        f"stage1_proxy={1 if bool(eval_stages.get('stage1_proxy', False)) else 0} "
        f"stage2_micro_unroll={1 if bool(eval_stages.get('stage2_micro_unroll', False)) else 0} "
        f"stage3_high_fidelity={1 if bool(eval_stages.get('stage3_high_fidelity', False)) else 0}"
    )


def _score_delta(*, cand_score: float, ref_score: float | None, metric_mode: str) -> float | None:
    if ref_score is None:
        return None
    if str(metric_mode) == "maximize":
        return float(cand_score - ref_score)
    return float(ref_score - cand_score)


def _is_better_than_reference(
    *,
    cand_score: float,
    reference_score: float | None,
    metric_mode: str,
    improve_eps: float,
) -> bool:
    if reference_score is None:
        return True
    if str(metric_mode) == "maximize":
        return bool(cand_score > (float(reference_score) + float(improve_eps)))
    return bool(cand_score < (float(reference_score) - float(improve_eps)))


def _score_threshold(*, reference_score: float | None, metric_mode: str, improve_eps: float) -> float | None:
    if reference_score is None:
        return None
    if str(metric_mode) == "maximize":
        return float(reference_score) + float(improve_eps)
    return float(reference_score) - float(improve_eps)


def _builder_perf_tiebreak_key(*, perf: float | None, metric_mode: str) -> float:
    if perf is None:
        return float("inf")
    try:
        perf_f = float(perf)
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(perf_f):
        return float("inf")
    return float(-perf_f) if str(metric_mode) == "maximize" else float(perf_f)


def _builder_selection_sort_key(
    *,
    feasible: bool,
    cost: float | None,
    perf: float | None,
    metric_mode: str,
) -> Tuple[float, float, float]:
    try:
        cost_f = float(cost)
    except (TypeError, ValueError):
        cost_f = float("inf")
    if not math.isfinite(cost_f):
        cost_f = float("inf")
    return (
        0.0 if bool(feasible) else 1.0,
        float(cost_f),
        _builder_perf_tiebreak_key(perf=perf, metric_mode=metric_mode),
    )


def _stored_selection_sort_key(entry: Mapping[str, Any], *, fallback_key: str) -> Tuple[float, ...]:
    raw = entry.get("selection_sort_key")
    if isinstance(raw, (list, tuple)) and raw:
        out: List[float] = []
        ok = True
        for v in raw:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            return tuple(out)
    try:
        score = float(entry.get(fallback_key, float("inf")))
    except (TypeError, ValueError):
        score = float("inf")
    if not math.isfinite(score):
        score = float("inf")
    return (float(score),)


def _refresh_loss_population_scores_from_history(
    entries: Sequence[Mapping[str, Any]],
    pair_score_history_map: Mapping[str, Any] | None,
    *,
    metric_mode: str,
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    if not isinstance(pair_score_history_map, Mapping):
        return [dict(e) for e in entries]

    score_by_loss_id: Dict[str, float] = {}
    for pair_key, hist in pair_score_history_map.items():
        if not isinstance(pair_key, str) or "::" not in pair_key or not isinstance(hist, list):
            continue
        _, loss_id = pair_key.split("::", 1)
        best_score = score_by_loss_id.get(str(loss_id))
        for item in hist:
            if not isinstance(item, Mapping):
                continue
            try:
                score = float(item.get("final_score", item.get("score")))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                score = float("inf")
            if best_score is None or _is_better_than_reference(
                cand_score=score,
                reference_score=best_score,
                metric_mode=metric_mode,
                improve_eps=0.0,
            ):
                best_score = float(score)
        if best_score is not None:
            score_by_loss_id[str(loss_id)] = float(best_score)

    refreshed: List[Dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        loss_id = str(item.get("id") or "")
        if loss_id in score_by_loss_id:
            item["fitness"] = float(score_by_loss_id[loss_id])
        refreshed.append(item)

    refreshed.sort(
        key=lambda x: float(x.get("fitness", float("-inf") if str(metric_mode) == "maximize" else float("inf"))),
        reverse=bool(str(metric_mode) == "maximize"),
    )
    return refreshed


def _extract_builder_cost(rec: Mapping[str, Any]) -> float | None:
    def _gate_check_value(trace: Mapping[str, Any] | None, metric_name: str) -> float | None:
        if not isinstance(trace, Mapping):
            return None
        checks = trace.get("checks")
        if not isinstance(checks, list):
            return None
        target = str(metric_name)
        for item in checks:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("metric_name")) != target:
                continue
            try:
                value = float(item.get("observed_value"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return float(value)
        return None

    desc = rec.get("descriptor")
    if isinstance(desc, Mapping):
        g_desc = desc.get("g")
        if isinstance(g_desc, Mapping):
            try:
                pair_count = float(g_desc.get("pair_count"))
            except (TypeError, ValueError):
                pair_count = None
            if pair_count is not None and math.isfinite(pair_count):
                return float(pair_count)

    bg = rec.get("builder_gate_trace")
    if isinstance(bg, Mapping):
        pair_count = _gate_check_value(bg, "pair_count")
        if pair_count is None:
            try:
                pair_count = float(bg.get("pair_count"))
            except (TypeError, ValueError):
                pair_count = None
        if pair_count is None:
            observed = bg.get("observed")
            if isinstance(observed, Mapping):
                try:
                    pair_count = float(observed.get("pair_count"))
                except (TypeError, ValueError):
                    pair_count = None
        if pair_count is not None and math.isfinite(pair_count):
            return float(pair_count)

    return None


def _compact_builder_perf_ref(rec: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "g_id": rec.get("g_id"),
        "f_id": rec.get("f_id"),
        "score": rec.get("score"),
        "final_score": rec.get("final_score"),
        "stage": rec.get("stage"),
        "stage_final": rec.get("stage_final"),
        "phase": rec.get("phase"),
        "generation": rec.get("generation"),
        "pair_reason": rec.get("pair_reason"),
        "pair_ok": rec.get("pair_ok"),
        "g_ir": rec.get("g_ir"),
    }
    desc = rec.get("descriptor")
    if isinstance(desc, Mapping):
        g_desc = desc.get("g")
        if isinstance(g_desc, Mapping):
            out["descriptor"] = {"g": dict(g_desc)}
    cost = _extract_builder_cost(rec)
    if cost is not None:
        out["cost"] = float(cost)
    return out


def _compute_builder_constraint_state(
    *,
    records: Sequence[Mapping[str, Any]],
    perf_by_builder: Mapping[str, float],
    metric_mode: str,
    slack: float,
) -> Dict[str, Any]:
    by_builder: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if not bool(rec.get("pair_ok")):
            continue
        gid = str(rec.get("g_id") or "")
        if not gid:
            continue
        try:
            perf = float(perf_by_builder[gid])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(perf):
            continue
        cost = _extract_builder_cost(rec)
        if cost is None or not math.isfinite(float(cost)):
            continue
        slot = by_builder.setdefault(
            gid,
            {
                "builder_id": str(gid),
                "perf": float(perf),
                "cost_samples": [],
                "perf_ref": None,
                "num_records": 0,
            },
        )
        slot["cost_samples"].append(float(cost))
        slot["num_records"] = int(slot.get("num_records", 0) or 0) + 1
        ref = slot.get("perf_ref")
        ref_score = None
        if isinstance(ref, Mapping):
            try:
                ref_score = float(ref.get("final_score", ref.get("score")))
            except (TypeError, ValueError):
                ref_score = None
        try:
            rec_score = float(rec.get("final_score", rec.get("score")))
        except (TypeError, ValueError):
            rec_score = None
        if rec_score is not None and math.isfinite(rec_score):
            if ref_score is None or _is_better_than_reference(
                cand_score=float(rec_score),
                reference_score=ref_score,
                metric_mode=metric_mode,
                improve_eps=0.0,
            ):
                slot["perf_ref"] = _compact_builder_perf_ref(rec)
        elif slot.get("perf_ref") is None:
            slot["perf_ref"] = _compact_builder_perf_ref(rec)

    if not by_builder:
        return {
            "builders": {},
            "best_perf": None,
            "threshold": None,
            "slack": float(slack),
            "feasible": [],
            "infeasible": [],
            "selected": None,
        }

    best_perf = None
    for stat in by_builder.values():
        perf = float(stat["perf"])
        if best_perf is None or _is_better_than_reference(
            cand_score=float(perf),
            reference_score=best_perf,
            metric_mode=metric_mode,
            improve_eps=0.0,
        ):
            best_perf = float(perf)

    threshold = None
    if best_perf is not None:
        if str(metric_mode) == "maximize":
            threshold = float(best_perf - float(slack))
        else:
            threshold = float(best_perf + float(slack))

    feasible: List[Dict[str, Any]] = []
    infeasible: List[Dict[str, Any]] = []
    for stat in by_builder.values():
        costs = [float(x) for x in stat.get("cost_samples", []) if math.isfinite(float(x))]
        if not costs:
            continue
        cost_mean = float(sum(costs) / float(len(costs)))
        perf = float(stat["perf"])
        is_feasible = False
        if best_perf is not None:
            if str(metric_mode) == "maximize":
                is_feasible = bool(perf >= (float(best_perf) - float(slack)))
            else:
                is_feasible = bool(perf <= (float(best_perf) + float(slack)))
        enriched = dict(stat)
        enriched["cost"] = float(cost_mean)
        enriched["feasible"] = bool(is_feasible)
        enriched["best_perf"] = float(best_perf) if best_perf is not None else None
        enriched["threshold"] = float(threshold) if threshold is not None else None
        enriched["slack"] = float(slack)
        enriched["selection_sort_key"] = list(
            _builder_selection_sort_key(
                feasible=bool(is_feasible),
                cost=float(cost_mean),
                perf=float(perf),
                metric_mode=metric_mode,
            )
        )
        if bool(is_feasible):
            feasible.append(enriched)
        else:
            infeasible.append(enriched)
        by_builder[str(enriched["builder_id"])] = enriched

    feasible.sort(key=lambda x: tuple(x.get("selection_sort_key", [])))
    infeasible.sort(key=lambda x: tuple(x.get("selection_sort_key", [])))
    selected = feasible[0] if feasible else None
    return {
        "builders": by_builder,
        "best_perf": float(best_perf) if best_perf is not None else None,
        "threshold": float(threshold) if threshold is not None else None,
        "slack": float(slack),
        "feasible": feasible,
        "infeasible": infeasible,
        "selected": (dict(selected) if isinstance(selected, Mapping) else None),
    }


def _merge_builder_archive_entry(
    incumbent: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
    *,
    metric_mode: str,
) -> Dict[str, Any]:
    merged = dict(incumbent) if isinstance(incumbent, Mapping) else {}
    cand = dict(candidate)

    if not merged:
        return cand

    merged_cost = None
    cand_cost = None
    try:
        merged_cost = float(merged.get("cost"))
    except (TypeError, ValueError):
        merged_cost = None
    try:
        cand_cost = float(cand.get("cost"))
    except (TypeError, ValueError):
        cand_cost = None

    merged_perf = None
    cand_perf = None
    try:
        merged_perf = float(merged.get("perf"))
    except (TypeError, ValueError):
        merged_perf = None
    try:
        cand_perf = float(cand.get("perf"))
    except (TypeError, ValueError):
        cand_perf = None

    if cand_cost is not None and math.isfinite(cand_cost):
        if merged_cost is None or not math.isfinite(merged_cost) or cand_cost < merged_cost:
            merged["cost"] = float(cand_cost)

    if cand_perf is not None and math.isfinite(cand_perf):
        if merged_perf is None or not math.isfinite(merged_perf) or _is_better_than_reference(
            cand_score=float(cand_perf),
            reference_score=float(merged_perf),
            metric_mode=metric_mode,
            improve_eps=0.0,
        ):
            merged["perf"] = float(cand_perf)
            if cand.get("perf_ref") is not None:
                merged["perf_ref"] = dict(cand.get("perf_ref") or {})
            for key in ("generation", "phase"):
                if cand.get(key) is not None:
                    merged[key] = cand.get(key)

    merged["builder_id"] = str(cand.get("builder_id") or merged.get("builder_id") or "")
    merged["num_records"] = int(cand.get("num_records", merged.get("num_records", 0)) or 0)
    return merged


def _select_best_builder_cost_from_archive(
    *,
    builder_archive: Mapping[str, Mapping[str, Any]],
    metric_mode: str,
    slack: float,
) -> Dict[str, Any] | None:
    if not isinstance(builder_archive, Mapping):
        return None

    best_perf = None
    candidates: List[Dict[str, Any]] = []
    for raw in builder_archive.values():
        if not isinstance(raw, Mapping):
            continue
        try:
            perf = float(raw.get("perf"))
            cost = float(raw.get("cost"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(perf) or not math.isfinite(cost):
            continue
        if best_perf is None or _is_better_than_reference(
            cand_score=float(perf),
            reference_score=best_perf,
            metric_mode=metric_mode,
            improve_eps=0.0,
        ):
            best_perf = float(perf)
        candidates.append(dict(raw))

    if best_perf is None or not candidates:
        return None

    threshold = float(best_perf - float(slack)) if str(metric_mode) == "maximize" else float(best_perf + float(slack))
    feasible: List[Dict[str, Any]] = []
    infeasible: List[Dict[str, Any]] = []
    for cand in candidates:
        perf = float(cand.get("perf"))
        is_feasible = bool(perf >= (float(best_perf) - float(slack))) if str(metric_mode) == "maximize" else bool(
            perf <= (float(best_perf) + float(slack))
        )
        cand["best_perf_anchor"] = float(best_perf)
        cand["threshold"] = float(threshold)
        cand["slack"] = float(slack)
        cand["feasible"] = bool(is_feasible)
        cand["selection_sort_key"] = list(
            _builder_selection_sort_key(
                feasible=bool(is_feasible),
                cost=float(cand.get("cost")),
                perf=float(cand.get("perf")),
                metric_mode=metric_mode,
            )
        )
        if bool(is_feasible):
            feasible.append(cand)
        else:
            infeasible.append(cand)

    feasible.sort(key=lambda x: tuple(x.get("selection_sort_key", [])))
    infeasible.sort(key=lambda x: tuple(x.get("selection_sort_key", [])))
    selected = feasible[0] if feasible else None
    return {
        "best_perf": float(best_perf),
        "threshold": float(threshold),
        "slack": float(slack),
        "feasible": feasible,
        "infeasible": infeasible,
        "selected": (dict(selected) if isinstance(selected, Mapping) else None),
    }


def _resolve_runtime_config(cfg_yaml: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = dict(cfg_yaml)
    preset = str(cfg.get("preset", "advanced") or "advanced").strip().lower()
    if preset not in {"simple", "advanced"}:
        preset = "advanced"

    default_search_mode = "alternating" if preset == "simple" else "coevo"
    search_mode = _normalize_search_mode(cfg.get("search_mode"), default_mode=default_search_mode)
    metric_mode = _normalize_metric_mode(cfg.get("metric_mode", "minimize"))
    improve_eps = _safe_float(cfg.get("improve_eps", 0.0), 0.0)
    eval_stages = _normalize_eval_stages(cfg)

    ignored_advanced_keys: List[str] = []
    simple_forced_defaults: Dict[str, Any] = {}

    if preset == "simple":
        budgets = cfg.get("budgets", {}) or {}
        if not isinstance(budgets, dict):
            budgets = {}
        population = cfg.get("population", {}) or {}
        if not isinstance(population, dict):
            population = {}
        builder_pair_budget = cfg.get("builder_pair_budget", {}) or {}
        if not isinstance(builder_pair_budget, dict):
            builder_pair_budget = {}
        eval_stages_raw = cfg.get("eval_stages", {}) or {}
        if not isinstance(eval_stages_raw, dict):
            eval_stages_raw = {}
        eval_stages = {
            "stage0_gate": _safe_bool(eval_stages_raw.get("stage0_gate", True), True),
            "stage1_proxy": _safe_bool(eval_stages_raw.get("stage1_proxy", True), True),
            "stage2_micro_unroll": _safe_bool(eval_stages_raw.get("stage2_micro_unroll", False), False),
            "stage3_high_fidelity": _safe_bool(eval_stages_raw.get("stage3_high_fidelity", True), True),
        }

        keep_top_k = _safe_int(population.get("keep_top_k", cfg.get("elite_g", 8)), 8)
        keep_top_k = max(1, int(keep_top_k))
        pairing_budget_per_gen = _safe_int(
            budgets.get("pairing_budget_per_gen", cfg.get("pairing_budget_per_gen", 128)),
            128,
        )
        pairing_budget_per_gen = max(1, int(pairing_budget_per_gen))
        pairing_budget_loss = _safe_int(
            budgets.get("pairing_budget_loss", cfg.get("pairing_budget_loss", pairing_budget_per_gen // 2)),
            pairing_budget_per_gen // 2,
        )
        pairing_budget_loss = max(0, min(int(pairing_budget_loss), int(pairing_budget_per_gen)))
        pairing_budget_builder = int(pairing_budget_per_gen - pairing_budget_loss)

        hf_top_m_default = min(8, keep_top_k) if bool(eval_stages.get("stage3_high_fidelity", False)) else 0
        micro_steps_default = 8 if bool(eval_stages.get("stage2_micro_unroll", False)) else 0
        micro_max_pairs_default = 8192 if bool(eval_stages.get("stage2_micro_unroll", False)) else 0

        simple_forced_defaults = {
            "generations": _safe_int(budgets.get("generations", cfg.get("generations", 50)), 50),
            "pairing_budget_per_gen": int(pairing_budget_per_gen),
            "pairing_budget_loss": int(pairing_budget_loss),
            "pairing_budget_builder": int(pairing_budget_builder),
            "proxy_batches": _safe_int(budgets.get("proxy_batches", cfg.get("proxy_batches", 10)), 10),
            "proxy_batch_size": _safe_int(budgets.get("proxy_batch_size", cfg.get("proxy_batch_size", 64)), 64),
            "micro_unroll_steps": _safe_int(
                budgets.get("micro_unroll_steps", cfg.get("micro_unroll_steps", micro_steps_default)),
                micro_steps_default,
            ),
            "micro_unroll_max_pairs": _safe_int(
                budgets.get("micro_unroll_max_pairs", cfg.get("micro_unroll_max_pairs", micro_max_pairs_default)),
                micro_max_pairs_default,
            ),
            "high_fidelity_top_m": _safe_int(
                budgets.get("high_fidelity_top_m", cfg.get("high_fidelity_top_m", hf_top_m_default)),
                hf_top_m_default,
            ),
            "hf_epochs": _safe_int(budgets.get("hf_epochs", cfg.get("hf_epochs", 1)), 1),
            "pop_f": _safe_int(population.get("n_candidates_loss", cfg.get("pop_f", 64)), 64),
            "pop_g": _safe_int(population.get("n_candidates_builder", cfg.get("pop_g", 64)), 64),
            "elite_f": int(keep_top_k),
            "elite_g": int(keep_top_k),
            "builder_max_pairs_per_instance": _safe_int(
                builder_pair_budget.get("per_instance", 64),
                64,
            ),
            "builder_min_coverage": _safe_float(cfg.get("builder_min_coverage", 0.0), 0.0),
            "descriptor_pair_count_cap": _safe_int(
                builder_pair_budget.get("total", 8192),
                8192,
            ),
            # Disable advanced coevo controls in simple mode.
            "anchor_enabled": False,
            "recheck_enabled": False,
            "coverage_k_elite_start": 0,
            "coverage_k_elite_end": 0,
            "coverage_k_hof_start": 0,
            "coverage_k_hof_end": 0,
            "coverage_k_random_start": 0,
            "coverage_k_random_end": 0,
            "crossplay_k_hof_start": 0,
            "crossplay_k_hof_end": 0,
            "archive_bins": 8,
            "archive_per_cell": 1,
            "diverse_elites_from_archive_max": 0,
            # Hidden defaults so simple configs can run HF without extra setup.
            "backend": str(cfg.get("backend", "rl4co")),
            "env_name": str(cfg.get("env_name", "tsp")),
            "generator_params": dict(cfg.get("generator_params", {"num_loc": 100}) or {"num_loc": 100}),
            "policy_name": str(cfg.get("policy_name", "pomo")),
            "policy_kwargs": dict(cfg.get("policy_kwargs", {"po4cops_compat": True}) or {"po4cops_compat": True}),
            "rollout_strategy": str(cfg.get("rollout_strategy", "auto")),
            "objective_sign": str(cfg.get("objective_sign", "neg_reward")),
            "hf_instances_per_epoch": _safe_int(cfg.get("hf_instances_per_epoch", 100000), 100000),
            "train_problem_size": _safe_int(cfg.get("train_problem_size", 100), 100),
            "valid_problem_sizes": list(cfg.get("valid_problem_sizes", [100]) or [100]),
            "train_batch_size": _safe_int(cfg.get("train_batch_size", 64), 64),
            "num_validation_episodes": _safe_int(cfg.get("num_validation_episodes", 10000), 10000),
            "validation_batch_size": _safe_int(cfg.get("validation_batch_size", 64), 64),
        }

        ignored_prefixes = (
            "coverage_",
            "bandit_",
            "crossplay_",
            "hof_size_",
            "archive_",
            "descriptor_",
            "anchor_",
            "recheck_",
        )
        ignored_exact = {
            "credit_assignment",
            "credit_best_k",
            "diverse_elites_from_archive_max",
            "cheap_gate_on",
            "high_fidelity_on",
            "micro_unroll_enabled",
        }
        for k in cfg.keys():
            if any(str(k).startswith(p) for p in ignored_prefixes) or str(k) in ignored_exact:
                ignored_advanced_keys.append(str(k))

        default_side_llm = {
            "enabled": True,
            "offline_mode": False,
            "parent_p": 5,
            "seed_reserve": 2,
            "init_num_E1": 6,
            "init_num_E2": 4,
            "init_num_M1": 4,
            "init_num_M2": 4,
            "num_E1": 8,
            "num_E2": 8,
            "num_M1": 8,
            "num_M2": 8,
            "repair": {"enabled": True, "max_attempts": 1, "simplify_first": True},
        }
        for side in ("builder_llm", "loss_llm"):
            side_cfg = cfg.get(side)
            if not isinstance(side_cfg, dict):
                cfg[side] = dict(default_side_llm)
                continue
            merged = dict(default_side_llm)
            merged.update(dict(side_cfg))
            repair_in = side_cfg.get("repair")
            if isinstance(repair_in, dict):
                rep = dict(default_side_llm.get("repair", {}))
                rep.update(dict(repair_in))
                merged["repair"] = rep
            cfg[side] = merged

    cfg["preset"] = str(preset)
    cfg["search_mode"] = str(search_mode)
    cfg["metric_mode"] = str(metric_mode)
    cfg["improve_eps"] = float(improve_eps)
    cfg["eval_stages"] = dict(eval_stages)
    cfg["cheap_gate_on"] = bool(eval_stages.get("stage0_gate", True))
    cfg["high_fidelity_on"] = bool(eval_stages.get("stage3_high_fidelity", True))
    cfg["micro_unroll_enabled"] = bool(eval_stages.get("stage2_micro_unroll", False))
    if simple_forced_defaults:
        cfg.update(simple_forced_defaults)

    return cfg, {"ignored_advanced_keys": sorted(set(ignored_advanced_keys))}


def _resolve_final_score(
    rec: Mapping[str, Any],
    *,
    eval_stages: Mapping[str, bool],
) -> Tuple[str, float | None]:
    if not bool(rec.get("pair_ok")):
        return "none", None
    hf_enabled = bool(eval_stages.get("stage3_high_fidelity", False))
    micro_enabled = bool(eval_stages.get("stage2_micro_unroll", False))
    proxy_enabled = bool(eval_stages.get("stage1_proxy", False))

    if hf_enabled:
        hf_ran = bool(isinstance(rec.get("fitness"), dict) or str(rec.get("stage")) == "high_fidelity")
        if hf_ran:
            try:
                return "high_fidelity", float(rec.get("score"))
            except (TypeError, ValueError):
                return "none", None
    if micro_enabled:
        if rec.get("micro_score") is not None:
            try:
                return "micro_unroll", float(rec.get("micro_score"))
            except (TypeError, ValueError):
                return "none", None
    if proxy_enabled:
        if rec.get("proxy_score") is not None:
            try:
                return "proxy", float(rec.get("proxy_score"))
            except (TypeError, ValueError):
                return "none", None
        if rec.get("score") is not None and str(rec.get("stage")) in {"cheap", "cheap_recheck", "anchor"}:
            try:
                return "proxy", float(rec.get("score"))
            except (TypeError, ValueError):
                return "none", None
    return "none", None


def _annotate_stage_fields(
    rec: Mapping[str, Any],
    *,
    eval_stages: Mapping[str, bool],
) -> Tuple[List[str], Dict[str, str]]:
    ran: List[str] = []
    skipped: Dict[str, str] = {}

    gate_enabled = bool(eval_stages.get("stage0_gate", False))
    proxy_enabled = bool(eval_stages.get("stage1_proxy", False))
    micro_enabled = bool(eval_stages.get("stage2_micro_unroll", False))
    hf_enabled = bool(eval_stages.get("stage3_high_fidelity", False))

    has_gate_ctx = rec.get("builder_gate_ok") is not None or rec.get("joint_gate_ok") is not None
    has_proxy_ctx = rec.get("proxy_score") is not None or isinstance(rec.get("proxy_metrics"), dict)
    has_micro_ctx = rec.get("micro_score") is not None or isinstance(rec.get("micro_metrics"), dict)
    has_hf_ctx = isinstance(rec.get("fitness"), dict) or str(rec.get("stage")) == "high_fidelity"

    if gate_enabled and has_gate_ctx:
        ran.append("stage0_gate")
    if proxy_enabled and has_proxy_ctx:
        ran.append("stage1_proxy")
    if micro_enabled and has_micro_ctx:
        ran.append("stage2_micro_unroll")
    if hf_enabled and has_hf_ctx:
        ran.append("stage3_high_fidelity")

    for stage_name in ("stage0_gate", "stage1_proxy", "stage2_micro_unroll", "stage3_high_fidelity"):
        if not bool(eval_stages.get(stage_name, False)):
            skipped[stage_name] = "disabled"
            continue
        if stage_name in ran:
            continue
        if stage_name == "stage3_high_fidelity":
            skipped[stage_name] = "not_selected_for_high_fidelity"
        elif stage_name == "stage2_micro_unroll":
            skipped[stage_name] = "not_selected_for_micro_unroll"
        elif stage_name == "stage1_proxy":
            skipped[stage_name] = "proxy_not_executed"
        else:
            skipped[stage_name] = "gate_not_executed"
    return ran, skipped


def _build_alternating_pairs(
    *,
    rng: random.Random,
    g_id_pool: Sequence[str],
    f_id_pool: Sequence[str],
    fixed_builder_id: str,
    fixed_loss_id: str,
    budget_loss: int,
    budget_builder: int,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str], str]]:
    pairs: List[Tuple[str, str]] = []
    reasons: Dict[Tuple[str, str], List[str]] = {}
    phases: Dict[Tuple[str, str], str] = {}
    used: set[Tuple[str, str]] = set()

    def _add_pair(gid: str, fid: str, *, reason: str, phase: str) -> None:
        key = (str(gid), str(fid))
        if key in used:
            reasons.setdefault(key, []).append(str(reason))
            return
        used.add(key)
        pairs.append(key)
        reasons.setdefault(key, []).append(str(reason))
        phases[key] = str(phase)

    loss_pool = [str(fid) for fid in f_id_pool if str(fid) != F_REF_ID]
    builder_pool = [str(gid) for gid in g_id_pool if str(gid) != G_REF_ID]
    rng.shuffle(loss_pool)
    rng.shuffle(builder_pool)

    for fid in loss_pool:
        if len([1 for p in pairs if phases.get(p) == "loss"]) >= int(max(0, budget_loss)):
            break
        _add_pair(str(fixed_builder_id), str(fid), reason="alternating_loss_phase", phase="loss")

    for gid in builder_pool:
        if len([1 for p in pairs if phases.get(p) == "builder"]) >= int(max(0, budget_builder)):
            break
        _add_pair(str(gid), str(fixed_loss_id), reason="alternating_builder_phase", phase="builder")

    return pairs, reasons, phases


def _resolve_alternating_phase_and_budgets(
    *,
    search_mode: str,
    generation: int,
    pairing_budget: int,
    pairing_budget_loss: int,
    pairing_budget_builder: int,
    alternating_schedule_enabled: bool,
    alternating_loss_generations: int,
    alternating_builder_generations: int,
    alternating_rounds: int,
    alternating_final_loss_generations: int,
    alternating_start_phase: str,
) -> Tuple[str, int, int, int, int, int]:
    """Resolve phase and pair budgets for one generation.

    Returns:
        (phase, loss_budget_now, builder_budget_now, round_idx, cycle_pos, cycle_len)
    """

    mode = str(search_mode).strip().lower()
    if mode == "loss_only":
        return "loss", int(pairing_budget), 0, -1, -1, 0
    if mode != "alternating":
        return "coevo", int(pairing_budget_loss), int(pairing_budget_builder), -1, -1, 0

    loss_budget_now = int(pairing_budget_loss)
    builder_budget_now = int(pairing_budget_builder)
    round_idx = -1
    cycle_pos = -1
    cycle_len = 0

    if bool(alternating_schedule_enabled):
        start_phase = str(alternating_start_phase or "loss").strip().lower()
        if start_phase not in {"loss", "builder"}:
            start_phase = "loss"
        cycle_len = int(max(0, int(alternating_loss_generations)) + max(0, int(alternating_builder_generations)))
        cycle_len_eff = max(cycle_len, 1)
        cycle_pos = int(int(generation) % int(cycle_len_eff))
        round_idx = int(int(generation) // int(cycle_len_eff))
        planned_main = (
            int(alternating_rounds) * int(cycle_len)
            if int(alternating_rounds) > 0
            else None
        )
        final_loss_generations = max(0, int(alternating_final_loss_generations))
        in_final_loss_tail = (
            planned_main is not None
            and int(generation) >= int(planned_main)
            and int(generation) < int(planned_main + final_loss_generations)
        )
        if in_final_loss_tail:
            loss_budget_now = int(pairing_budget)
            builder_budget_now = 0
            round_idx = int(alternating_rounds)
            cycle_pos = int(generation) - int(planned_main)
            cycle_len = int(final_loss_generations)
        elif int(alternating_rounds) > 0 and round_idx >= int(alternating_rounds):
            loss_budget_now = 0
            builder_budget_now = 0
        elif (
            start_phase == "loss"
            and cycle_pos < int(max(0, int(alternating_loss_generations)))
        ) or (
            start_phase == "builder"
            and cycle_pos >= int(max(0, int(alternating_builder_generations)))
        ):
            loss_budget_now = int(pairing_budget)
            builder_budget_now = 0
        else:
            loss_budget_now = 0
            builder_budget_now = int(pairing_budget)

    phase = "none"
    if int(loss_budget_now) > 0 and int(builder_budget_now) == 0:
        phase = "loss"
    elif int(builder_budget_now) > 0 and int(loss_budget_now) == 0:
        phase = "builder"
    elif int(loss_budget_now) > 0 and int(builder_budget_now) > 0:
        phase = "mixed"
    return phase, int(loss_budget_now), int(builder_budget_now), int(round_idx), int(cycle_pos), int(cycle_len)


class _CompiledBuilderAdapter(PrefBuilder):
    def __init__(self, compiled: CompiledPreferenceBuilder) -> None:
        self._compiled = compiled

    def build(
        self,
        feature_cache: Mapping[str, torch.Tensor],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> PrefBatch:
        return self._compiled.build_fn(feature_cache, dict(meta or {}))


def _make_builtin_builder_irs(rng: random.Random, n: int) -> List[PreferenceBuilderIR]:
    """Rule-based initial/mutated builder pool (no LLM dependency).

    Output:
        list[PreferenceBuilderIR] of length >= 1, each defining `generated_builder(feature_cache, extra)`.
    """

    def _hint() -> PreferenceBuilderImplementationHint:
        return PreferenceBuilderImplementationHint(
            expects=["objective", "log_prob"],
            returns="PrefBatch",
            mode="pairwise",
        )

    base_code = (
        "def generated_builder(feature_cache, extra):\n"
        "    objective = feature_cache['objective']\n"
        "    mask = objective[:, :, None] < objective[:, None, :]\n"
        "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
        "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'all_pairs'})\n"
    )
    pool: List[PreferenceBuilderIR] = []
    for i in range(max(1, int(n))):
        # For n>1, include a stable baseline at i==0; for n==1, allow diversity.
        if i == 0 and int(n) > 1:
            choice = "all_pairs"
        else:
            choice = rng.choice(["all_pairs", "anchor_best", "gap_threshold", "sampled_pairs"])

        kind = str(choice)
        code = base_code
        if choice == "anchor_best":
            code = (
                "def generated_builder(feature_cache, extra):\n"
                "    objective = feature_cache['objective']\n"
                "    best = objective.argmin(dim=1, keepdim=True)\n"
                "    all_idx = torch.arange(objective.shape[1], device=objective.device)[None, :].expand_as(objective)\n"
                "    b_idx = torch.arange(objective.shape[0], device=objective.device)[:, None].expand_as(all_idx)\n"
                "    winner_idx = best.expand_as(all_idx)\n"
                "    loser_idx = all_idx\n"
                "    mask = winner_idx != loser_idx\n"
                "    b = b_idx[mask]\n"
                "    w = winner_idx[mask]\n"
                "    l = loser_idx[mask]\n"
                "    return PrefBatch(mode='pairwise', pair_idx=(b, w, l), weight=None, meta={'builder': 'anchor_best'})\n"
            )
        elif choice == "gap_threshold":
            thr = float(rng.uniform(0.0, 1.0))
            code = (
                "def generated_builder(feature_cache, extra):\n"
                "    objective = feature_cache['objective']\n"
                f"    thr = float(extra.get('gap_threshold', {thr}))\n"
                "    diff = objective[:, None, :] - objective[:, :, None]\n"
                "    mask = (diff > thr)\n"
                "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
                "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'gap_threshold', 'thr': thr})\n"
            )
        elif choice == "sampled_pairs":
            max_pairs = int(rng.randint(64, 512))
            code = (
                "def generated_builder(feature_cache, extra):\n"
                "    objective = feature_cache['objective']\n"
                "    gap = objective[:, None, :] - objective[:, :, None]\n"
                "    mask = gap > 0.0\n"
                "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
                f"    max_pairs = int(extra.get('max_pairs', {max_pairs}))\n"
                "    if b_idx.numel() > max_pairs:\n"
                "        scores = gap[b_idx, winner_idx, loser_idx]\n"
                "        keep = torch.topk(scores, k=max_pairs, largest=True, sorted=False).indices\n"
                "        b_idx = b_idx[keep]\n"
                "        winner_idx = winner_idx[keep]\n"
                "        loser_idx = loser_idx[keep]\n"
                "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'sampled_pairs', 'max_pairs': max_pairs})\n"
            )

        pool.append(
            PreferenceBuilderIR(
                name=f"builder_{kind}_{i:03d}",
                intuition=f"rule_based:{kind}",
                implementation_hint=_hint(),
                hyperparams={
                    "geometry_family": (
                        "dense_all_pairs"
                        if kind == "all_pairs"
                        else ("anchor_star" if kind == "anchor_best" else ("threshold_pairs" if kind == "gap_threshold" else "sampled_pairs"))
                    ),
                    "cap_family": (
                        "uncapped_full"
                        if kind == "all_pairs"
                        else ("anchor_single" if kind == "anchor_best" else ("threshold_keep" if kind == "gap_threshold" else "budget_sample"))
                    ),
                    "weight_family": "uniform_none",
                    "constraint_family": (
                        "none"
                        if kind == "all_pairs"
                        else ("dedup_anchor" if kind == "anchor_best" else ("gap_filter" if kind == "gap_threshold" else "budget_clamp"))
                    ),
                },
                operators_used=[kind],
                code=code,
            )
        )
    return pool


def _rank_weighted_sample_without_replacement(
    rng: random.Random,
    items: Sequence[Any],
    *,
    k: int,
) -> List[Any]:
    k = max(0, min(int(k), len(items)))
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    # Use rank weights 1/(rank+1) assuming items are already sorted by quality.
    weights = [1.0 / (i + 1.0) for i in range(len(items))]
    chosen: List[Any] = []
    pool = list(items)
    w = list(weights)
    for _ in range(k):
        s = float(sum(w))
        if s <= 0:
            # Fallback to uniform.
            idx = rng.randrange(len(pool))
        else:
            r = rng.random() * s
            acc = 0.0
            idx = 0
            for j, ww in enumerate(w):
                acc += float(ww)
                if acc >= r:
                    idx = j
                    break
        chosen.append(pool.pop(idx))
        w.pop(idx)
    return chosen


def _rank_weighted_sample_without_replacement_diverse(
    rng: random.Random,
    items: Sequence[Any],
    *,
    k: int,
    key_fn,
    max_per_key: int,
) -> List[Any]:
    """Rank-weighted sampling while limiting repeats per key (best-effort)."""

    k = max(0, min(int(k), len(items)))
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)

    max_per_key = max(1, int(max_per_key))

    weights = [1.0 / (i + 1.0) for i in range(len(items))]
    chosen: List[Any] = []
    pool = list(items)
    w = list(weights)
    key_counts: collections.Counter[str] = collections.Counter()

    for _ in range(k):
        eligible = [j for j, it in enumerate(pool) if int(key_counts[str(key_fn(it))]) < int(max_per_key)]
        if not eligible:
            eligible = list(range(len(pool)))

        s = float(sum(w[j] for j in eligible))
        if s <= 0:
            idx = rng.choice(eligible)
        else:
            r = rng.random() * s
            acc = 0.0
            idx = eligible[-1]
            for j in eligible:
                acc += float(w[j])
                if acc >= r:
                    idx = j
                    break

        it = pool.pop(idx)
        ww = w.pop(idx)
        chosen.append(it)
        _ = ww
        try:
            key_counts[str(key_fn(it))] += 1
        except Exception:  # noqa: BLE001
            key_counts["?"] += 1

    return chosen


def _llm_op_choice(rng: random.Random, *, gen: int, parent_pool_size: int) -> str:
    if parent_pool_size <= 0:
        return "E1_GENERATE"
    if gen <= 0:
        return "E1_GENERATE" if rng.random() < 0.7 else "E2"
    if parent_pool_size < 2:
        return "M1"
    r = rng.random()
    if r < 0.22:
        return "E2"
    if r < 0.50:
        return "M1"
    if r < 0.70:
        return "E1"
    if r < 0.90:
        return "M2"
    return "E1_GENERATE"


def _truncate_code(s: str, *, max_chars: int = 1600) -> str:
    s2 = str(s or "")
    if len(s2) <= max_chars:
        return s2
    return s2[: max_chars - 12] + "\n# ... truncated"


def summarize_best_builder(
    entry: Mapping[str, Any] | None,
    *,
    max_code_chars: int = 500,
) -> Dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    ir = entry.get("ir")
    if not isinstance(ir, Mapping):
        return None
    code = ir.get("code", "")
    if isinstance(code, str):
        code = _truncate_code(code, max_chars=int(max_code_chars))
    impl = ir.get("implementation_hint") if isinstance(ir.get("implementation_hint"), Mapping) else {}
    return {
        "id": entry.get("id"),
        "fitness": entry.get("fitness"),
        "signature": entry.get("signature"),
        "name": ir.get("name"),
        "mode": (impl or {}).get("mode"),
        "expects": (impl or {}).get("expects"),
        "intuition": ir.get("intuition"),
        "code": code,
        "descriptor": entry.get("descriptor"),
    }


def summarize_best_loss(
    entry: Mapping[str, Any] | None,
    *,
    max_code_chars: int = 500,
) -> Dict[str, Any] | None:
    if not isinstance(entry, Mapping):
        return None
    ir = entry.get("ir")
    if not isinstance(ir, Mapping):
        return None
    code = ir.get("code", "")
    if isinstance(code, str):
        code = _truncate_code(code, max_chars=int(max_code_chars))
    hint = ir.get("implementation_hint") if isinstance(ir.get("implementation_hint"), Mapping) else {}
    return {
        "id": entry.get("id"),
        "fitness": entry.get("fitness"),
        "signature": entry.get("signature"),
        "name": ir.get("name"),
        "operators_used": ir.get("operators_used"),
        "hyperparams": ir.get("hyperparams"),
        "mode": (hint or {}).get("mode"),
        "expects": (hint or {}).get("expects"),
        "intuition": ir.get("intuition"),
        "pseudocode": ir.get("pseudocode"),
        "code": code,
        "descriptor": entry.get("descriptor"),
    }


def _make_builtin_loss_irs(rng: random.Random, n: int) -> List[FreeLossIR]:
    """Rule-based initial/mutated loss pool (no LLM dependency).

    Output:
        list[FreeLossIR] of length >= 1, each defining `generated_loss(batch, model_output, extra)`.
    """

    pool: List[FreeLossIR] = []
    for i in range(max(1, int(n))):
        scale = float(rng.uniform(0.5, 2.0))
        use_cost = bool(rng.choice([False, True]))
        name = f"loss_logsigmoid_{i:03d}" + ("_cost" if use_cost else "")
        expects = ["log_prob_w", "log_prob_l", "delta_z", "weight"]
        if use_cost:
            expects = ["log_prob_w", "log_prob_l", "cost_a", "cost_b", "weight"]

        code = (
            "def generated_loss(batch, model_output, extra):\n"
            "    lpw = batch['log_prob_w']\n"
            "    lpl = batch['log_prob_l']\n"
            "    weight = batch.get('weight', None)\n"
            "    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))\n"
            f"    scale = float(extra.get('hyperparams', {{}}).get('scale', {scale}))\n"
        )
        if use_cost:
            code += (
                "    cost_a = batch['cost_a']\n"
                "    cost_b = batch['cost_b']\n"
                "    gap = (cost_b - cost_a).detach()\n"
                "    x = alpha * scale * (lpw - lpl) - 0.1 * gap\n"
            )
        else:
            code += "    x = alpha * scale * (lpw - lpl)\n"
        code += (
            "    x = ops.clamp(x, -20.0, 20.0)\n"
            "    loss = -ops.logsigmoid(x)\n"
            "    if weight is not None:\n"
            "        loss = loss * weight\n"
            "    return loss.mean()\n"
        )

        ir_obj = {
            "name": name,
            "intuition": "rule_based: pairwise logsigmoid loss",
            "pseudocode": "loss = -logsigmoid(clamp(alpha*scale*(lpw-lpl), -20, 20))",
            "hyperparams": {
                "scale": scale,
                "paradigm_family": "pairwise_margin",
                "signal_family": ("cost_gap" if use_cost else "logprob_gap"),
                "link_family": "logsigmoid",
                "agg_family": "mean",
                "constraint_family": "clamp_stabilized",
            },
            "operators_used": ["logsigmoid", "clamp"],
            "implementation_hint": {"expects": expects, "returns": "scalar", "mode": "pairwise"},
            "code": code,
        }
        pool.append(free_loss_ir_from_json(ir_obj))
    return pool


def _build_hf_cfg(cfg: Mapping[str, Any], *, seed: int, device_str: str) -> HighFidelityConfig:
    backend = str(cfg.get("backend", "rl4co") or "rl4co").strip().lower()
    env_name = cfg.get("env_name") or cfg.get("problem", "tsp")
    generator_params = cfg.get("generator_params", {}) or {}
    env_kwargs = cfg.get("env_kwargs", {}) or {}
    policy_name = cfg.get("policy_name", "") or ""
    policy_kwargs = cfg.get("policy_kwargs", {}) or {}
    rollout_strategy = cfg.get("rollout_strategy", "auto")
    objective_sign = cfg.get("objective_sign", "neg_reward")

    return HighFidelityConfig(
        problem=str(cfg.get("problem", "tsp")),
        backend=backend,
        env_name=str(env_name),
        env_kwargs=dict(env_kwargs),
        generator_params=dict(generator_params),
        policy_name=str(policy_name),
        policy_kwargs=dict(policy_kwargs),
        rollout_strategy=str(rollout_strategy),
        objective_sign=str(objective_sign),
        hf_steps=int(cfg.get("f1_steps", 32) or 32),
        hf_epochs=int(cfg.get("hf_epochs", 0) or 0),
        hf_instances_per_epoch=int(cfg.get("hf_instances_per_epoch", 0) or 0),
        train_problem_size=int(cfg.get("train_problem_size", 20)),
        valid_problem_sizes=tuple(int(v) for v in cfg.get("valid_problem_sizes", [100])),
        train_batch_size=int(cfg.get("train_batch_size", 64)),
        pomo_size=int(cfg.get("pomo_size", 64)) if cfg.get("pomo_size", None) is not None else None,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-6)),
        alpha=float(cfg.get("alpha", 0.05)),
        po_impl=_po_impl_from_cfg(cfg),
        precision=str(cfg.get("precision", "32-true") or "32-true"),
        device=str(device_str),
        seed=int(seed),
        num_validation_episodes=int(cfg.get("num_validation_episodes", 128)),
        validation_batch_size=int(cfg.get("validation_batch_size", 64)),
        generalization_penalty_weight=float(cfg.get("generalization_penalty_weight", 1.0)),
        size_aggregation=str(cfg.get("size_aggregation", "cvar")),
        size_cvar_alpha=float(cfg.get("size_cvar_alpha", 0.2)),
        pool_version=str(cfg.get("pool_version", "v0")),
        loss_observables=tuple(str(v) for v in cfg.get("loss_observables", []) if str(v).strip()),
    )


def _compute_early_eval_steps(cfg_yaml: Mapping[str, Any], hf_cfg: HighFidelityConfig) -> int:
    """Compute early-eval steps, aligned with free_loss_discovery_eoh behavior.

    Early-eval is used only for warm-start phase early stopping inside
    `fitness.free_loss_fidelity.evaluate_free_loss_candidate`.
    """

    total_steps = get_total_hf_train_steps(hf_cfg)
    early_eval_epochs = int(cfg_yaml.get("early_eval_epochs", 0) or 0)
    early_eval_instances_per_epoch = int(cfg_yaml.get("early_eval_instances_per_epoch", 0) or 0)
    early_eval_steps_cfg = cfg_yaml.get("early_eval_steps")

    if early_eval_epochs > 0:
        instances_per_epoch = early_eval_instances_per_epoch
        if instances_per_epoch <= 0:
            instances_per_epoch = int(getattr(hf_cfg, "hf_instances_per_epoch", 0) or 0)
        if instances_per_epoch > 0:
            batch_size = max(int(getattr(hf_cfg, "train_batch_size", 64) or 64), 1)
            steps_per_epoch = math.ceil(instances_per_epoch / batch_size)
            steps = early_eval_epochs * steps_per_epoch
        else:
            steps = 0
    elif early_eval_steps_cfg is not None:
        steps = int(early_eval_steps_cfg or 0)
    else:
        steps = min(100, int(total_steps))

    if steps <= 0:
        return 0
    return min(int(steps), int(total_steps))


def _build_free_cfg(cfg: Mapping[str, Any], *, hf_cfg: HighFidelityConfig) -> FreeLossFidelityConfig:
    baseline_cfg = cfg.get("baseline", {}) or {}
    ckpt = (
        baseline_cfg.get("checkpoint")
        or cfg.get("baseline_checkpoint")
        or cfg.get("baseline_ckpt")
        or cfg.get("init_checkpoint_path")
    )
    ckpt_epoch = baseline_cfg.get("checkpoint_epoch", cfg.get("baseline_checkpoint_epoch"))
    if ckpt_epoch is None and ckpt:
        ckpt_epoch = _infer_baseline_epoch_from_path(str(ckpt))

    return FreeLossFidelityConfig(
        hf=hf_cfg,
        f1_steps=int(cfg.get("f1_steps", 32) or 32),
        f2_steps=int(cfg.get("f2_steps", 0) or 0),
        f3_enabled=bool(cfg.get("f3_enabled", False)),
        init_checkpoint_path=_abs_from_repo_root(str(ckpt)) if ckpt else None,
        init_checkpoint_epoch=(int(ckpt_epoch) if ckpt_epoch is not None else None),
        scratch_hf_epochs=int(cfg.get("scratch_hf_epochs", 0) or 0),
        warmstart_hf_epochs=int(cfg.get("warmstart_hf_epochs", 0) or 0),
        baseline_epoch_compare_offset=int(cfg.get("baseline_epoch_compare_offset", 0) or 0),
        baseline_epoch_violation_weight=float(cfg.get("baseline_epoch_violation_weight", 1.0)),
        baseline_epoch_tail_frac=float(cfg.get("baseline_epoch_tail_frac", 1.0) or 1.0),
        baseline_epoch_window_k=int(cfg.get("baseline_epoch_window_k", 10) or 10),
        baseline_epoch_window_violation_weight=float(
            cfg.get("baseline_epoch_window_violation_weight", 1.0) or 1.0
        ),
    )


def _dummy_feature_cache(*, batch_size: int, k: int, variant: str) -> Dict[str, torch.Tensor]:
    # Deterministic dummy cache for gates/compilation smoke checks.
    b = max(1, int(batch_size))
    kk = max(2, int(k))
    idx = torch.arange(kk, dtype=torch.float32)[None, :].repeat(b, 1)
    objective = idx + (torch.arange(b, dtype=torch.float32)[:, None] * 0.01)
    log_prob = -0.1 * idx
    if variant == "hidden":
        objective = objective * 10.0
        log_prob = log_prob * 12.0
    return extract_feature_cache(objective, log_prob)


def _builder_failure_report(
    *,
    stage: str,
    reason: str,
    trace: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stage": str(stage), "reason": str(reason)}
    if error is not None:
        out["error"] = str(error)
    if trace is not None:
        try:
            out["trace"] = dict(trace)
        except Exception:  # noqa: BLE001
            out["trace"] = {"_unserializable_trace": True}
    return out


def _prompt_sha1(text: str) -> str:
    return sha1(str(text).encode("utf-8")).hexdigest()


def _read_prompt_best_effort(path: str) -> str:
    # Match free_loss_llm_ops prompt reading (including its fallback) when possible.
    try:
        fn = getattr(loss_llm_ops, "_read_prompt", None)
        if callable(fn):
            return str(fn(path))
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _append_global_feedback_block(prompt: str, global_feedback: Mapping[str, Any] | None) -> str:
    if global_feedback is None:
        return prompt
    return prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)


def _append_runtime_context_block(prompt: str, prompt_context: Mapping[str, Any] | None) -> str:
    try:
        fn = getattr(loss_llm_ops, "_append_prompt_context_block", None)
        if callable(fn):
            return str(fn(prompt, prompt_context))
    except Exception:  # noqa: BLE001
        pass
    return prompt


def _build_free_loss_generation_prompt(
    prompt_path: str,
    *,
    global_feedback: Mapping[str, Any] | None,
    prompt_context: Mapping[str, Any] | None = None,
) -> Tuple[str, str]:
    prompt = _read_prompt_best_effort(prompt_path)
    prompt = _append_runtime_context_block(prompt, prompt_context)
    prompt = _append_global_feedback_block(prompt, global_feedback)
    return prompt, _prompt_sha1(prompt)


def _build_free_loss_parents_prompt(
    prompt_path: str,
    *,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None,
    global_feedback: Mapping[str, Any] | None,
    parent_block_name: str,
    prompt_context: Mapping[str, Any] | None = None,
) -> Tuple[str, str]:
    prompt = _read_prompt_best_effort(prompt_path)
    prompt = _append_runtime_context_block(prompt, prompt_context)
    blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blobs.append(
            {
                "index": idx,
                "name": parent.name,
                "intuition": parent.intuition,
                "pseudocode": parent.pseudocode,
                "hyperparams": parent.hyperparams,
                "operators_used": parent.operators_used,
                "code": parent.code,
                "theoretical_basis": getattr(parent, "theoretical_basis", ""),
                "metrics": {
                    "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
                    "validation_objective": float(metrics.get("validation_objective", float("inf")))
                    if metrics
                    else None,
                    "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
                    if metrics
                    else None,
                    "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
                    "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
                },
            }
        )
    prompt = prompt + f"\n\n{str(parent_block_name)}:\n" + json.dumps(blobs, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback_block(prompt, global_feedback)
    return prompt, _prompt_sha1(prompt)


def _build_free_loss_parent_prompt(
    prompt_path: str,
    *,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None,
    global_feedback: Mapping[str, Any] | None,
    parent_block_name: str,
    prompt_context: Mapping[str, Any] | None = None,
) -> Tuple[str, str]:
    prompt = _read_prompt_best_effort(prompt_path)
    prompt = _append_runtime_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf"))) if metrics else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0)) if metrics else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
        },
    }
    prompt = prompt + f"\n\n{str(parent_block_name)}:\n" + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback_block(prompt, global_feedback)
    return prompt, _prompt_sha1(prompt)


def _build_free_loss_failure_prompt(
    prompt_path: str,
    *,
    candidate: FreeLossIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None,
    block_name: str,
    ensure_ascii: bool,
) -> Tuple[str, str]:
    prompt = _read_prompt_best_effort(prompt_path)
    payload = {
        "candidate": asdict(candidate),
        "failure_reason": dict(failure_reason),
    }
    prompt = prompt + f"\n\n{str(block_name)}:\n" + json.dumps(payload, indent=2, ensure_ascii=bool(ensure_ascii))
    prompt = _append_global_feedback_block(prompt, global_feedback)
    return prompt, _prompt_sha1(prompt)


def _operator_contract_failure(
    *,
    op_type: str,
    reason: str,
    parent_tags: Any,
    cand_tags: Mapping[str, Any],
    required_change: Any,
) -> Dict[str, Any]:
    return {
        "stage": "operator_contract",
        "reason": str(reason),
        "op_type": str(op_type),
        "parent_tags": parent_tags,
        "cand_tags": dict(cand_tags),
        "required_change": required_change,
        "trace": {
            "failed_gate": "OperatorContract",
            "failure_kind": str(reason),
        },
    }


def _validate_builder_operator_contract(
    ir: PreferenceBuilderIR,
    op_type: str,
    parent_irs: Sequence[PreferenceBuilderIR],
    parent_entries: Sequence[Mapping[str, Any]] | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    del parent_entries
    op = str(op_type or "").strip().upper()
    if op not in {"BUILDER_PARADIGM_SHIFT", "BUILDER_STRUCTURE_SHIFT", "BUILDER_CONSTRAINT_INJECT"}:
        return True, {}
    cand_tags = _builder_family_tags(ir)
    missing = _missing_required_family_tags(cand_tags, keys=_BUILDER_FAMILY_KEYS)
    if missing:
        return False, _operator_contract_failure(
            op_type=op,
            reason="missing_family_tags",
            parent_tags=[_builder_family_tags(p) for p in parent_irs],
            cand_tags=cand_tags,
            required_change={"required_keys": list(_BUILDER_FAMILY_KEYS), "missing": missing},
        )
    if op == "BUILDER_PARADIGM_SHIFT":
        maj = _majority_parent_tags(parent_irs, keys=_BUILDER_FAMILY_KEYS, kind="builder")
        if _normalize_family_value(cand_tags.get("geometry_family")) == _normalize_family_value(maj.get("geometry_family")):
            return False, _operator_contract_failure(
                op_type=op,
                reason="geometry_family_not_changed",
                parent_tags=maj,
                cand_tags=cand_tags,
                required_change={"must_change": ["geometry_family"], "must_also_change_one_of": ["cap_family", "constraint_family"]},
            )
        if (
            _normalize_family_value(cand_tags.get("cap_family")) == _normalize_family_value(maj.get("cap_family"))
            and _normalize_family_value(cand_tags.get("constraint_family")) == _normalize_family_value(maj.get("constraint_family"))
        ):
            return False, _operator_contract_failure(
                op_type=op,
                reason="secondary_family_not_changed",
                parent_tags=maj,
                cand_tags=cand_tags,
                required_change={"must_change": ["geometry_family"], "must_also_change_one_of": ["cap_family", "constraint_family"]},
            )
    elif op == "BUILDER_STRUCTURE_SHIFT":
        parent = parent_irs[0] if parent_irs else None
        p_tags = _builder_family_tags(parent) if parent is not None else {}
        if _normalize_family_value(cand_tags.get("cap_family")) == _normalize_family_value(p_tags.get("cap_family")):
            return False, _operator_contract_failure(
                op_type=op,
                reason="cap_family_not_changed",
                parent_tags=p_tags,
                cand_tags=cand_tags,
                required_change={"must_change": ["cap_family"]},
            )
    elif op == "BUILDER_CONSTRAINT_INJECT":
        parent = parent_irs[0] if parent_irs else None
        p_tags = _builder_family_tags(parent) if parent is not None else {}
        if _normalize_family_value(cand_tags.get("constraint_family")) == _normalize_family_value(p_tags.get("constraint_family")):
            return False, _operator_contract_failure(
                op_type=op,
                reason="constraint_family_not_changed",
                parent_tags=p_tags,
                cand_tags=cand_tags,
                required_change={"must_change": ["constraint_family"]},
            )
    return True, {}


def _validate_loss_operator_contract(
    ir: FreeLossIR,
    op_type: str,
    parent_irs: Sequence[FreeLossIR],
    parent_entries: Sequence[Mapping[str, Any]] | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    del parent_entries
    op = str(op_type or "").strip().upper()
    if op not in {"LOSS_PARADIGM_SHIFT", "LOSS_STRUCTURE_SHIFT", "LOSS_CONSTRAINT_INJECT"}:
        return True, {}
    cand_tags = _loss_family_tags(ir)
    missing = _missing_required_family_tags(cand_tags, keys=_LOSS_FAMILY_KEYS)
    if missing:
        return False, _operator_contract_failure(
            op_type=op,
            reason="missing_family_tags",
            parent_tags=[_loss_family_tags(p) for p in parent_irs],
            cand_tags=cand_tags,
            required_change={"required_keys": list(_LOSS_FAMILY_KEYS), "missing": missing},
        )
    if op == "LOSS_PARADIGM_SHIFT":
        maj = _majority_parent_tags(parent_irs, keys=_LOSS_FAMILY_KEYS, kind="loss")
        if _normalize_family_value(cand_tags.get("paradigm_family")) == _normalize_family_value(maj.get("paradigm_family")):
            return False, _operator_contract_failure(
                op_type=op,
                reason="paradigm_family_not_changed",
                parent_tags=maj,
                cand_tags=cand_tags,
                required_change={"must_change": ["paradigm_family"], "must_also_change_one_of": ["signal_family", "agg_family", "constraint_family"]},
            )
        if all(
            _normalize_family_value(cand_tags.get(key)) == _normalize_family_value(maj.get(key))
            for key in ("signal_family", "agg_family", "constraint_family")
        ):
            return False, _operator_contract_failure(
                op_type=op,
                reason="secondary_family_not_changed",
                parent_tags=maj,
                cand_tags=cand_tags,
                required_change={"must_change": ["paradigm_family"], "must_also_change_one_of": ["signal_family", "agg_family", "constraint_family"]},
            )
    elif op == "LOSS_STRUCTURE_SHIFT":
        parent = parent_irs[0] if parent_irs else None
        p_tags = _loss_family_tags(parent) if parent is not None else {}
        if _normalize_family_value(cand_tags.get("agg_family")) == _normalize_family_value(p_tags.get("agg_family")):
            return False, _operator_contract_failure(
                op_type=op,
                reason="agg_family_not_changed",
                parent_tags=p_tags,
                cand_tags=cand_tags,
                required_change={"must_change": ["agg_family"]},
            )
    elif op == "LOSS_CONSTRAINT_INJECT":
        parent = parent_irs[0] if parent_irs else None
        p_tags = _loss_family_tags(parent) if parent is not None else {}
        if _normalize_family_value(cand_tags.get("constraint_family")) == _normalize_family_value(p_tags.get("constraint_family")):
            return False, _operator_contract_failure(
                op_type=op,
                reason="constraint_family_not_changed",
                parent_tags=p_tags,
                cand_tags=cand_tags,
                required_change={"must_change": ["constraint_family"]},
            )
    return True, {}


def validate_builder_candidate(
    ir: PreferenceBuilderIR,
    *,
    operator_whitelist: Sequence[str],
    gate_cfg: Mapping[str, Any],
    op_type: str | None = None,
    parent_irs: Sequence[PreferenceBuilderIR] | None = None,
    parent_entries: Sequence[Mapping[str, Any]] | None = None,
    dummy_variant: str = "visible",
) -> Tuple[bool, Dict[str, Any]]:
    """Compile + smoke-run + gate a builder on a synthetic feature_cache.

    Returns:
      (ok, failure_report). failure_report is JSON-serializable and suitable for LLM repair prompts.
    """

    if not str(getattr(ir, "intuition", "") or "").strip():
        return False, _builder_failure_report(
            stage="interpretability",
            reason="missing_intuition",
            trace={"failed_gate": "Interpretability", "failure_kind": "missing_intuition"},
        )

    try:
        compiled = compile_preference_builder(ir, operator_whitelist=operator_whitelist)
    except Exception as exc:  # noqa: BLE001
        return False, _builder_failure_report(stage="compile", reason="compile_failed", error=str(exc))

    try:
        fc = _dummy_feature_cache(batch_size=8, k=16, variant=str(dummy_variant))
        pb = _build_pref_batch_with_memory_trace(compiled, fc, {"stage": "builder_validate", "seed": 0})
    except Exception as exc:  # noqa: BLE001
        return False, _builder_failure_report(stage="runtime", reason="runtime_failed", error=str(exc))

    try:
        bg = run_preference_builder_gates(
            pb,
            feature_cache=fc,
            min_pairs=int(gate_cfg.get("min_pairs", 1) or 1),
            min_coverage=float(gate_cfg.get("min_coverage", 0.0) or 0.0),
            max_pairs_per_instance=int(gate_cfg.get("max_pairs_per_instance", 4096) or 4096),
            weight_nonneg=bool(gate_cfg.get("weight_nonneg", True)),
            semantic_tolerance=float(gate_cfg.get("semantic_tolerance", 0.0) or 0.0),
            semantic_min_pass_rate=float(gate_cfg.get("semantic_min_pass_rate", 1.0) or 1.0),
        )
        bg.trace = _enrich_builder_gate_trace_with_memory(bg.trace, pb, cache_hit=False)
    except Exception as exc:  # noqa: BLE001
        return False, _builder_failure_report(stage="gate", reason="builder_gate_exception", error=str(exc))

    if not bool(bg.ok):
        return False, _builder_failure_report(
            stage="gate",
            reason=str(bg.reason),
            trace=bg.trace,
        )

    contract_ok, contract_fail = _validate_builder_operator_contract(
        ir,
        str(op_type or ""),
        list(parent_irs or []),
        list(parent_entries or []),
    )
    if not bool(contract_ok):
        return False, dict(contract_fail)

    return True, {}


def _repair_builder_candidate_loop(
    ir: PreferenceBuilderIR,
    *,
    failure_report: Mapping[str, Any],
    operator_whitelist: Sequence[str],
    gate_cfg: Mapping[str, Any],
    op_type: str | None = None,
    parent_irs: Sequence[PreferenceBuilderIR] | None = None,
    parent_entries: Sequence[Mapping[str, Any]] | None = None,
    llm_prompts: Mapping[str, str],
    global_feedback: Mapping[str, Any] | None,
    max_attempts: int,
    simplify_first: bool = True,
) -> Tuple[PreferenceBuilderIR | None, Dict[str, Any]]:
    """Attempt to repair a failing builder candidate via LLM, re-validating each attempt.

    Returns:
      (repaired_ir or None, meta)
    """

    p_m3 = str(llm_prompts.get("builder_m3", "") or "")
    p_rep = str(llm_prompts.get("builder_repair", "") or "")
    if not p_rep:
        return None, {"repair_skipped": True, "reason": "missing_builder_repair_prompt"}

    last_fail = dict(failure_report)
    attempts: List[Dict[str, Any]] = []
    current = ir
    for attempt in range(max(0, int(max_attempts))):
        try:
            fb = dict(global_feedback or {})
            fb.update({"repair_attempt": int(attempt), "repair_stage": str(last_fail.get("stage", ""))})
        except Exception:  # noqa: BLE001
            fb = dict(global_feedback or {})

        candidate = current
        if simplify_first and attempt == 0 and p_m3:
            try:
                simplified, m3_meta = builder_llm_ops.m3_simplify_builder_with_meta(
                    p_m3,
                    candidate=candidate,
                    failure_reason=last_fail,
                    global_feedback=fb,
                )
                candidate = simplified
                attempts.append({"attempt": int(attempt), "op": "M3", **dict(m3_meta)})
            except Exception as exc:  # noqa: BLE001
                attempts.append({"attempt": int(attempt), "op": "M3", "error": str(exc)})

        try:
            repaired, rep_meta = builder_llm_ops.repair_pref_builder_with_meta(
                p_rep,
                failed_ir=candidate,
                failure_reason=last_fail,
                global_feedback=fb,
            )
            attempts.append({"attempt": int(attempt), "op": "REPAIR", **dict(rep_meta)})
        except Exception as exc:  # noqa: BLE001
            attempts.append({"attempt": int(attempt), "op": "REPAIR", "error": str(exc)})
            return None, {"attempts": attempts, "last_fail": last_fail}

        ok, fail2 = validate_builder_candidate(
            repaired,
            operator_whitelist=operator_whitelist,
            gate_cfg=gate_cfg,
            op_type=op_type,
            parent_irs=parent_irs,
            parent_entries=parent_entries,
        )
        if ok:
            return repaired, {"attempts": attempts, "repaired": True}
        last_fail = dict(fail2)
        current = repaired

    return None, {"attempts": attempts, "last_fail": last_fail}


def _propose_builders_for_generation(
    *,
    generation: int,
    pop_g: int,
    elites_g: Sequence[Mapping[str, Any]],
    diverse_elites_g: Sequence[Mapping[str, Any]],
    rng: random.Random,
    llm_cfg: Mapping[str, Any] | None = None,
    operator_whitelist: Sequence[str] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    llm_init_only: bool = False,
    carry_elites: bool = True,
) -> List[Dict[str, Any]]:
    """Propose builder candidates with elitism + mutation/crossover."""

    pop_g = max(int(pop_g), 1)
    out: List[Dict[str, Any]] = []

    parent_pool: List[Mapping[str, Any]] = []
    for src in (elites_g, diverse_elites_g):
        for item in src:
            if isinstance(item, dict) and isinstance(item.get("ir"), dict) and item.get("id"):
                parent_pool.append(item)

    if bool(carry_elites):
        # Elitism: carry over a small subset verbatim.
        elite_carry = min(len(elites_g), max(1, pop_g // 4))
        for item in list(elites_g)[:elite_carry]:
            if not isinstance(item, dict) or not isinstance(item.get("ir"), dict):
                continue
            ir = pref_builder_ir_from_json(item["ir"])
            out.append(
                {
                    "ir": ir,
                    "origin": "ELITE",
                    "op_type": "ELITE",
                    "parents": [str(item.get("id", ""))],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                    "novelty": None,
                }
            )

    # LLM candidates (Double-EoH: g-side).
    llm_root: Mapping[str, Any] = llm_cfg or {}
    builder_cfg: Mapping[str, Any] = llm_root
    if isinstance(llm_root.get("builder"), Mapping):
        builder_cfg = llm_root.get("builder")  # type: ignore[assignment]

    llm_enabled = bool(builder_cfg and bool(builder_cfg.get("enabled", False)))
    seed_reserve_raw = None if not isinstance(builder_cfg, Mapping) else builder_cfg.get("seed_reserve")
    if seed_reserve_raw is None:
        seed_reserve_raw = 2 if llm_enabled else 0
    try:
        seed_reserve = max(0, min(int(seed_reserve_raw or 0), int(pop_g)))
    except (TypeError, ValueError):
        seed_reserve = 2 if llm_enabled else 0
    if bool(llm_init_only):
        seed_reserve = 0

    if llm_enabled:
        if operator_whitelist is None:
            operator_whitelist = []
        parent_p = int(builder_cfg.get("parent_p", 5) or 5)
        repair_cfg = builder_cfg.get("repair", {}) or {}
        if not isinstance(repair_cfg, dict):
            repair_cfg = {}
        repair_on_fail = bool(repair_cfg.get("enabled", builder_cfg.get("repair_on_failure", True)))
        repair_attempts = int(repair_cfg.get("max_attempts", builder_cfg.get("repair_attempts", 1)) or 1)
        family_div_cfg = _normalize_family_diversity_cfg(builder_cfg.get("family_diversity", {}))

        prompts = llm_root.get("prompts", builder_cfg.get("prompts", {})) or {}
        if not isinstance(prompts, dict):
            prompts = {}
        p_gen = str(prompts.get("builder_generation", "") or "")
        p_x = str(prompts.get("builder_crossover", "") or "")
        p_m = str(prompts.get("builder_mutation", "") or "")
        p_e2 = str(prompts.get("builder_e2", "") or "")
        p_shift = str(prompts.get("builder_paradigm_shift", "") or "")
        p_structure = str(prompts.get("builder_structure_shift", "") or "")
        p_constraint = str(prompts.get("builder_constraint_inject", "") or "")
        p_m2 = str(prompts.get("builder_m2", "") or "")
        p_m3 = str(prompts.get("builder_m3", "") or "")
        p_rep = str(prompts.get("builder_repair", "") or "")

        gate_cfg = llm_root.get("builder_gate", builder_cfg.get("builder_gate", {})) or {}
        if not isinstance(gate_cfg, dict):
            gate_cfg = {}
        min_pairs = int(gate_cfg.get("min_pairs", 1) or 1)
        min_cov = float(gate_cfg.get("min_coverage", 0.0) or 0.0)
        max_pairs_pi = int(gate_cfg.get("max_pairs_per_instance", 4096) or 4096)
        weight_nonneg = bool(gate_cfg.get("weight_nonneg", True))
        sem_tol = float(gate_cfg.get("semantic_tolerance", 0.0) or 0.0)
        sem_min_pass = float(gate_cfg.get("semantic_min_pass_rate", 1.0) or 1.0)

        # Build ranked parent pool for LLM operations.
        ranked_parents: List[Tuple[float, str, PreferenceBuilderIR, Mapping[str, Any]]] = []
        for item in parent_pool:
            try:
                ir = pref_builder_ir_from_json(item["ir"])
            except Exception:  # noqa: BLE001
                continue
            try:
                fit = float(item.get("fitness", float("inf")))
            except (TypeError, ValueError):
                fit = float("inf")
            item2 = dict(item)
            item2["family_signature"] = _builder_family_signature(item2)
            ranked_parents.append((fit, str(item.get("id", "")), ir, item2))
        ranked_parents.sort(key=lambda x: _stored_selection_sort_key(x[3], fallback_key="fitness"))
        if not ranked_parents:
            # Bootstrap parents so E2/M1/M2 are usable at gen0 (aligns with free_loss EoH behavior).
            bootstrap = _make_builtin_builder_irs(rng, max(2, int(parent_p)))
            for i, ir0 in enumerate(bootstrap):
                ranked_parents.append(
                    (
                        0.0,
                        f"bootstrap_g_{i:03d}",
                        ir0,
                        {
                            "id": f"bootstrap_g_{i:03d}",
                            "fitness": 0.0,
                            "ir": asdict(ir0),
                            "family_signature": _builder_family_signature(ir0),
                        },
                    )
                )
        if bool(family_div_cfg.get("enabled", False)):
            ranked_parents = _cap_parent_pool_by_family(
                ranked_parents,
                side="builder",
                max_per_family=int(family_div_cfg.get("parent_max_per_family", 0) or 0),
                include_unknown=bool(family_div_cfg.get("include_unknown", False)),
            )
        existing_sigs = {str((item[3] or {}).get("family_signature") or _builder_family_signature(item[2])) for item in ranked_parents}
        if len(existing_sigs) < 2:
            supplements = [_ref_builder_ir()] + _make_builtin_builder_irs(rng, max(2, int(parent_p)))
            for sup_idx, ir0 in enumerate(supplements):
                sig0 = _builder_family_signature(ir0)
                if sig0 in existing_sigs:
                    continue
                ranked_parents.append(
                    (
                        0.0,
                        f"builder_sup_{sup_idx:03d}_{sig0[:12]}",
                        ir0,
                        {"id": f"builder_sup_{sup_idx:03d}_{sig0[:12]}", "fitness": 0.0, "ir": asdict(ir0), "family_signature": sig0},
                    )
                )
                existing_sigs.add(sig0)
                if len(existing_sigs) >= 2:
                    break
        ranked_parents.sort(key=lambda x: _stored_selection_sort_key(x[3], fallback_key="fitness"))

        # Operator plan: either explicit counts (preferred) or legacy budget+random choice.
        def _op_plan() -> List[str]:
            init = int(generation) <= 0
            plan = _expand_operator_bank(builder_cfg, int(generation), rng, side="builder")
            if plan:
                return plan
            keys = ("num_E1", "num_E2", "num_M1", "num_M2", "init_num_E1", "init_num_E2", "init_num_M1", "init_num_M2")
            if any(builder_cfg.get(k) is not None for k in keys):
                nE1 = int(builder_cfg.get("init_num_E1" if init else "num_E1", builder_cfg.get("num_E1", 0)) or 0)
                nE2 = int(builder_cfg.get("init_num_E2" if init else "num_E2", builder_cfg.get("num_E2", 0)) or 0)
                nM1 = int(builder_cfg.get("init_num_M1" if init else "num_M1", builder_cfg.get("num_M1", 0)) or 0)
                nM2 = int(builder_cfg.get("init_num_M2" if init else "num_M2", builder_cfg.get("num_M2", 0)) or 0)
                plan = (["E1"] * max(0, nE1)) + (["E2"] * max(0, nE2)) + (["M1"] * max(0, nM1)) + (["M2"] * max(0, nM2))
                rng.shuffle(plan)
                return plan

            llm_budget = int(builder_cfg.get("init_llm_g", 0) or 0) if init else int(builder_cfg.get("llm_per_gen_g", 0) or 0)
            return [_llm_op_choice(rng, gen=int(generation), parent_pool_size=len(ranked_parents)) for _ in range(max(0, llm_budget))]

        for op in _op_plan():
            if len(out) >= max(0, int(pop_g) - int(seed_reserve)):
                break
            parents_ir: List[PreferenceBuilderIR] = []
            parents_fit: List[Mapping[str, Any]] = []
            parents_ids: List[str] = []
            parent_entries_used: List[Mapping[str, Any]] = []

            # Map high-level EoH ops to concrete LLM ops.
            raw_op = _normalize_operator_name(str(op).strip().upper(), side="builder")
            llm_op = str(raw_op)
            if llm_op in {"XOVER", "E1"}:
                if len(ranked_parents) >= 2 and p_x:
                    llm_op = "E1"
                else:
                    llm_op = "E1_GENERATE"
            elif llm_op in {"GEN", "E2"}:
                llm_op = "E2" if str(raw_op) == "E2" else "E1_GENERATE"
            elif llm_op in {"TUNE", "M2"}:
                llm_op = "M2"
            elif llm_op in {"PARADIGM_SHIFT", "STRUCTURE_SHIFT", "CONSTRAINT_INJECT"}:
                llm_op = str(raw_op)
            else:
                llm_op = "M1"

            if llm_op in {"E1", "E2"} and len(ranked_parents) >= 2:
                chosen = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=max(2, min(parent_p, len(ranked_parents))))
                parents_ir = [c[2] for c in chosen]
                parents_fit = [{"fitness": float(c[0])} for c in chosen]
                parents_ids = [str(c[1]) for c in chosen]
                parent_entries_used = [c[3] for c in chosen]
            elif llm_op in {"M1", "M2", "STRUCTURE_SHIFT", "CONSTRAINT_INJECT"} and len(ranked_parents) >= 1:
                chosen1 = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=1)[0]
                parents_ir = [chosen1[2]]
                parents_fit = [{"fitness": float(chosen1[0])}]
                parents_ids = [str(chosen1[1])]
                parent_entries_used = [chosen1[3]]
            elif llm_op == "PARADIGM_SHIFT" and len(ranked_parents) >= 2:
                chosen, shortage = _sample_cross_family_parents(
                    rng,
                    ranked_parents,
                    side="builder",
                    k=max(2, min(parent_p, len(ranked_parents))),
                )
                parents_ir = [c[2] for c in chosen]
                parents_fit = [{"fitness": float(c[0])} for c in chosen]
                parents_ids = [str(c[1]) for c in chosen]
                parent_entries_used = [c[3] for c in chosen]
            else:
                llm_op = "E1_GENERATE"
                shortage = False

            llm_seed = int(rng.randint(0, 2**31 - 1))
            call_feedback = dict(global_feedback or {})
            call_feedback["llm_call"] = {"side": "builder", "op_type": str(llm_op), "seed": llm_seed}
            if llm_op == "PARADIGM_SHIFT":
                call_feedback["llm_call"]["parent_family_shortage"] = bool(locals().get("shortage", False))

            history: List[Dict[str, Any]] = []
            try:
                if llm_op == "E1_GENERATE":
                    ir, meta = builder_llm_ops.generate_pref_builder_candidate_with_meta(
                        p_gen,
                        operator_whitelist=operator_whitelist,
                        global_feedback=call_feedback,
                    )
                    base_origin = "E1"
                    op_type = "E1_GENERATE"
                    parent_ids = []
                    parent_entries_used = []
                elif llm_op == "E1":
                    ir, meta = builder_llm_ops.crossover_pref_builder_with_meta(
                        p_x,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                    )
                    base_origin = "E1"
                    op_type = "E1"
                    parent_ids = parents_ids
                elif llm_op == "E2":
                    ir, meta = builder_llm_ops.e2_pref_builder_with_meta(
                        p_e2,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                    )
                    base_origin = "E2"
                    op_type = "E2"
                    parent_ids = parents_ids
                elif llm_op == "PARADIGM_SHIFT":
                    ir, meta = builder_llm_ops.paradigm_shift_builder_with_meta(
                        p_shift,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                    )
                    base_origin = "PARADIGM_SHIFT"
                    op_type = "BUILDER_PARADIGM_SHIFT"
                    parent_ids = parents_ids
                elif llm_op == "STRUCTURE_SHIFT":
                    ir, meta = builder_llm_ops.structure_shift_builder_with_meta(
                        p_structure,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                    )
                    base_origin = "STRUCTURE_SHIFT"
                    op_type = "BUILDER_STRUCTURE_SHIFT"
                    parent_ids = parents_ids
                elif llm_op == "CONSTRAINT_INJECT":
                    ir, meta = builder_llm_ops.constraint_inject_builder_with_meta(
                        p_constraint,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                    )
                    base_origin = "CONSTRAINT_INJECT"
                    op_type = "BUILDER_CONSTRAINT_INJECT"
                    parent_ids = parents_ids
                elif llm_op == "M2":
                    ir, meta = builder_llm_ops.m2_tune_builder_with_meta(
                        p_m2,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                    )
                    base_origin = "M2"
                    op_type = "M2"
                    parent_ids = parents_ids
                else:
                    ir, meta = builder_llm_ops.mutate_pref_builder_with_meta(
                        p_m,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                    )
                    base_origin = "M1"
                    op_type = "M1"
                    parent_ids = parents_ids
                history.append({"attempt": 0, "side": "builder", **dict(meta)})
            except Exception:  # noqa: BLE001
                continue

            ok, fail_reason = validate_builder_candidate(
                ir,
                operator_whitelist=operator_whitelist,
                gate_cfg={
                    "min_pairs": min_pairs,
                    "min_coverage": min_cov,
                    "max_pairs_per_instance": max_pairs_pi,
                    "weight_nonneg": weight_nonneg,
                    "semantic_tolerance": sem_tol,
                    "semantic_min_pass_rate": sem_min_pass,
                },
                op_type=str(op_type),
                parent_irs=parents_ir,
                parent_entries=parent_entries_used,
            )

            origin = base_origin
            prompt_sha1 = dict(meta).get("prompt_sha1") if isinstance(meta, Mapping) else None
            prompt_path = dict(meta).get("prompt_path") if isinstance(meta, Mapping) else None

            if (not ok) and repair_on_fail:
                repaired, rep_meta = _repair_builder_candidate_loop(
                    ir,
                    failure_report=fail_reason,
                    operator_whitelist=operator_whitelist,
                    gate_cfg={
                        "min_pairs": min_pairs,
                        "min_coverage": min_cov,
                        "max_pairs_per_instance": max_pairs_pi,
                        "weight_nonneg": weight_nonneg,
                        "semantic_tolerance": sem_tol,
                        "semantic_min_pass_rate": sem_min_pass,
                    },
                    op_type=str(op_type),
                    parent_irs=parents_ir,
                    parent_entries=parent_entries_used,
                    llm_prompts={"builder_m3": p_m3, "builder_repair": p_rep},
                    global_feedback=call_feedback,
                    max_attempts=max(0, int(repair_attempts)),
                    simplify_first=bool(repair_cfg.get("simplify_first", True)),
                )
                if repaired is not None:
                    ir = repaired
                    origin = "REPAIR"
                    op_type = "REPAIR"
                    ok = True
                    if isinstance(rep_meta, dict) and isinstance(rep_meta.get("attempts"), list) and rep_meta["attempts"]:
                        history.extend(list(rep_meta["attempts"]))
                        last = rep_meta["attempts"][-1]
                        prompt_sha1 = last.get("prompt_sha1", prompt_sha1)
                        prompt_path = last.get("prompt_path", prompt_path)

            if ok:
                out.append(
                    {
                        "ir": ir,
                        "origin": str(origin),
                        "origin_base": str(base_origin),
                        "op_type": str(op_type),
                        "parents": list(parent_ids),
                        "attempt": 0,
                        "prompt_sha1": prompt_sha1,
                        "prompt_path": prompt_path,
                        "history": history,
                        "llm_seed": llm_seed,
                    }
                )

    # Mutations/crossover and fresh seeds.
    if bool(llm_init_only):
        return out[:pop_g]
    while len(out) < pop_g:
        op = rng.choice(["mutate", "crossover", "seed"]) if parent_pool else "seed"
        if op == "seed":
            ir = _make_builtin_builder_irs(rng, 1)[0]
            out.append(
                {
                    "ir": ir,
                    "origin": "SEED",
                    "op_type": "SEED",
                    "parents": [],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                    "novelty": None,
                }
            )
            continue

        if op == "mutate":
            parent = rng.choice(parent_pool)
            # Mutation: resample a rule-based variant; keep parent id as provenance.
            ir = _make_builtin_builder_irs(rng, 1)[0]
            ir.name = f"{ir.name}_m_from_{str(parent.get('id',''))[:12]}"
            out.append(
                {
                    "ir": ir,
                    "origin": "SEED",
                    "op_type": "SEED_MUTATE",
                    "parents": [str(parent.get("id", ""))],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                    "novelty": None,
                }
            )
            continue

        # crossover
        p1 = rng.choice(parent_pool)
        p2 = rng.choice(parent_pool)
        ir = _make_builtin_builder_irs(rng, 1)[0]
        ir.name = f"{ir.name}_x_{str(p1.get('id',''))[:8]}_{str(p2.get('id',''))[:8]}"
        out.append(
            {
                "ir": ir,
                "origin": "SEED",
                "op_type": "SEED_CROSSOVER",
                "parents": [str(p1.get("id", "")), str(p2.get("id", ""))],
                "attempt": 0,
                "prompt_sha1": None,
                "prompt_path": None,
                "history": [],
                "novelty": None,
            }
        )

    return out[:pop_g]


def _propose_losses_for_generation(
    *,
    generation: int,
    pop_f: int,
    elites_f: Sequence[Mapping[str, Any]],
    diverse_elites_f: Sequence[Mapping[str, Any]],
    rng: random.Random,
    llm_cfg: Mapping[str, Any] | None = None,
    operator_whitelist: Sequence[str] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    llm_init_only: bool = False,
    carry_elites: bool = True,
) -> List[Dict[str, Any]]:
    """Propose loss candidates with elitism + mutation/crossover."""

    pop_f = max(int(pop_f), 1)
    out: List[Dict[str, Any]] = []
    loss_prompt_context = loss_llm_ops.build_runtime_prompt_context(
        loss_observables=tuple(str(v) for v in cfg_yaml.get("loss_observables", []) if str(v).strip()),
        mode="pairwise",
    )

    parent_pool: List[Mapping[str, Any]] = []
    for src in (elites_f, diverse_elites_f):
        for item in src:
            if isinstance(item, dict) and isinstance(item.get("ir"), dict) and item.get("id"):
                parent_pool.append(item)

    if bool(carry_elites):
        elite_carry = min(len(elites_f), max(1, pop_f // 4))
        for item in list(elites_f)[:elite_carry]:
            if not isinstance(item, dict) or not isinstance(item.get("ir"), dict):
                continue
            ir = free_loss_ir_from_json(item["ir"])
            out.append(
                {
                    "ir": ir,
                    "origin": "ELITE",
                    "op_type": "ELITE",
                    "parents": [str(item.get("id", ""))],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                }
            )

    # LLM candidates (Double-EoH: f-side).
    llm_root: Mapping[str, Any] = llm_cfg or {}
    loss_cfg: Mapping[str, Any] = llm_root
    if isinstance(llm_root.get("loss"), Mapping):
        loss_cfg = llm_root.get("loss")  # type: ignore[assignment]

    llm_enabled = bool(loss_cfg and bool(loss_cfg.get("enabled", False)))
    seed_reserve_raw = None if not isinstance(loss_cfg, Mapping) else loss_cfg.get("seed_reserve")
    if seed_reserve_raw is None:
        seed_reserve_raw = 2 if llm_enabled else 0
    try:
        seed_reserve = max(0, min(int(seed_reserve_raw or 0), int(pop_f)))
    except (TypeError, ValueError):
        seed_reserve = 2 if llm_enabled else 0
    if bool(llm_init_only):
        seed_reserve = 0

    if llm_enabled:
        if operator_whitelist is None:
            operator_whitelist = []
        parent_p = int(loss_cfg.get("parent_p", 5) or 5)
        repair_cfg = loss_cfg.get("repair", {}) or {}
        if not isinstance(repair_cfg, dict):
            repair_cfg = {}
        repair_on_fail = bool(repair_cfg.get("enabled", loss_cfg.get("repair_on_failure", True)))
        repair_attempts = int(repair_cfg.get("max_attempts", loss_cfg.get("repair_attempts", 1)) or 1)

        prompts = llm_root.get("prompts", loss_cfg.get("prompts", {})) or {}
        if not isinstance(prompts, dict):
            prompts = {}
        p_gen = str(prompts.get("loss_generation", "") or "")
        p_x = str(prompts.get("loss_crossover", "") or "")
        p_m = str(prompts.get("loss_mutation", "") or "")
        p_e2 = str(prompts.get("loss_e2", "") or "")
        p_shift = str(prompts.get("loss_paradigm_shift", "") or "")
        p_structure = str(prompts.get("loss_structure_shift", "") or "")
        p_constraint = str(prompts.get("loss_constraint_inject", "") or "")
        p_m2 = str(prompts.get("loss_m2", "") or "")
        p_m3 = str(prompts.get("loss_m3", "") or "")
        p_rep = str(prompts.get("loss_repair", "") or "")

        novelty_cfg = loss_cfg.get("novelty", {}) or {}
        if not isinstance(novelty_cfg, dict):
            novelty_cfg = {}
        novelty_enabled = bool(novelty_cfg.get("enabled", False))
        novelty_max_sim = float(novelty_cfg.get("max_similarity", 0.92) or 0.92)
        novelty_neighbors = int(novelty_cfg.get("neighbors", 3) or 3)
        novelty_apply_to_elites = bool(novelty_cfg.get("apply_to_elites", False))

        family_div = _normalize_family_diversity_cfg(loss_cfg.get("family_diversity", {}))
        family_parent_div_enabled = bool(family_div.get("enabled", False))
        family_parent_max_per = int(family_div.get("parent_max_per_family", 1) or 1)

        explore_cfg = loss_cfg.get("exploration", {}) or {}
        if not isinstance(explore_cfg, dict):
            explore_cfg = {}
        explore_enabled = bool(explore_cfg.get("enabled", False))
        explore_replace_p = float(explore_cfg.get("replace_p", 0.7) or 0.7)
        explore_mode = bool((global_feedback or {}).get("loss_search", {}).get("explore_mode", False))

        ranked_parents: List[Tuple[float, str, FreeLossIR, Mapping[str, Any]]] = []
        for item in parent_pool:
            try:
                ir0 = free_loss_ir_from_json(item["ir"])
            except Exception:  # noqa: BLE001
                continue
            try:
                fit = float(item.get("fitness", float("inf")))
            except (TypeError, ValueError):
                fit = float("inf")
            item2 = dict(item)
            item2["family_signature"] = _loss_family_signature(item2)
            ranked_parents.append((fit, str(item.get("id", "")), ir0, item2))
        ranked_parents.sort(key=lambda x: float(x[0]))
        if not ranked_parents:
            bootstrap = _make_builtin_loss_irs(rng, max(2, int(parent_p)))
            for i, ir0 in enumerate(bootstrap):
                ranked_parents.append(
                    (
                        0.0,
                        f"bootstrap_f_{i:03d}",
                        ir0,
                        {
                            "id": f"bootstrap_f_{i:03d}",
                            "fitness": 0.0,
                            "ir": asdict(ir0),
                            "family_signature": _loss_family_signature(ir0),
                        },
                    )
                )
        if bool(family_parent_div_enabled):
            ranked_parents = _cap_parent_pool_by_family(
                ranked_parents,
                side="loss",
                max_per_family=int(family_parent_max_per),
                include_unknown=bool(family_div.get("include_unknown", False)),
            )
        existing_sigs = {str((item[3] or {}).get("family_signature") or _loss_family_signature(item[2])) for item in ranked_parents}
        if len(existing_sigs) < 2:
            supplements = [_ref_loss_ir()] + _make_builtin_loss_irs(rng, max(2, int(parent_p)))
            for sup_idx, ir0 in enumerate(supplements):
                sig0 = _loss_family_signature(ir0)
                if sig0 in existing_sigs:
                    continue
                ranked_parents.append(
                    (
                        0.0,
                        f"loss_sup_{sup_idx:03d}_{sig0[:12]}",
                        ir0,
                        {"id": f"loss_sup_{sup_idx:03d}_{sig0[:12]}", "fitness": 0.0, "ir": asdict(ir0), "family_signature": sig0},
                    )
                )
                existing_sigs.add(sig0)
                if len(existing_sigs) >= 2:
                    break
        ranked_parents.sort(key=lambda x: float(x[0]))

        novelty_bank: List[Dict[str, Any]] = []
        novelty_seen: set[str] = set()

        def _bank_add(ir_in: FreeLossIR) -> None:
            try:
                fp = _loss_fingerprint(ir_in)
                sig0 = str(fp.get("sig", "")) or _sig_free_loss(ir_in)
                if sig0 in novelty_seen:
                    return
                novelty_seen.add(sig0)
                novelty_bank.append({"sig": sig0, "name": str(getattr(ir_in, "name", "") or ""), **fp})
            except Exception:  # noqa: BLE001
                return

        # Compare novelty against carried elites + parent pool, so new children cannot be
        # near-duplicates of incumbent structures.
        for carried in out:
            if isinstance(carried, dict) and isinstance(carried.get("ir"), FreeLossIR):
                _bank_add(carried["ir"])
        for _, _, pir, _ in ranked_parents:
            _bank_add(pir)

        def _op_plan() -> List[str]:
            init = int(generation) <= 0
            plan = _expand_operator_bank(loss_cfg, int(generation), rng, side="loss")
            if plan:
                return plan
            keys = ("num_E1", "num_E2", "num_M1", "num_M2", "init_num_E1", "init_num_E2", "init_num_M1", "init_num_M2")
            if any(loss_cfg.get(k) is not None for k in keys):
                nE1 = int(loss_cfg.get("init_num_E1" if init else "num_E1", loss_cfg.get("num_E1", 0)) or 0)
                nE2 = int(loss_cfg.get("init_num_E2" if init else "num_E2", loss_cfg.get("num_E2", 0)) or 0)
                nM1 = int(loss_cfg.get("init_num_M1" if init else "num_M1", loss_cfg.get("num_M1", 0)) or 0)
                nM2 = int(loss_cfg.get("init_num_M2" if init else "num_M2", loss_cfg.get("num_M2", 0)) or 0)
                plan = (["E1"] * max(0, nE1)) + (["E2"] * max(0, nE2)) + (["M1"] * max(0, nM1)) + (["M2"] * max(0, nM2))
                rng.shuffle(plan)
                if explore_enabled and explore_mode:
                    # Bias away from consensus (E2) and hyperparam-tuning (M2) when stagnating.
                    plan2: List[str] = []
                    for op in plan:
                        if op in {"E2", "M2", "GEN", "TUNE"} and rng.random() < float(explore_replace_p):
                            plan2.append("E1")
                        else:
                            plan2.append(op)
                    plan = plan2
                return plan

            llm_budget = int(loss_cfg.get("init_llm_f", 0) or 0) if init else int(loss_cfg.get("llm_per_gen_f", 0) or 0)
            return [_llm_op_choice(rng, gen=int(generation), parent_pool_size=len(ranked_parents)) for _ in range(max(0, llm_budget))]

        for op in _op_plan():
            if len(out) >= max(0, int(pop_f) - int(seed_reserve)):
                break
            parents_ir: List[FreeLossIR] = []
            parents_fit: List[Mapping[str, Any]] = []
            parents_ids: List[str] = []
            parent_entries_used: List[Mapping[str, Any]] = []
            raw_op = _normalize_operator_name(str(op).strip().upper(), side="loss")
            llm_op = str(raw_op)
            if llm_op in {"XOVER", "E1"}:
                if len(ranked_parents) >= 2 and p_x:
                    llm_op = "E1"
                else:
                    llm_op = "E1_GENERATE"
            elif llm_op in {"GEN", "E2"}:
                llm_op = "E2" if str(raw_op) == "E2" else "E1_GENERATE"
            elif llm_op in {"TUNE", "M2"}:
                llm_op = "M2"
            elif llm_op in {"PARADIGM_SHIFT", "STRUCTURE_SHIFT", "CONSTRAINT_INJECT"}:
                llm_op = str(raw_op)
            else:
                llm_op = "M1"

            if llm_op in {"E1", "E2"} and len(ranked_parents) >= 2:
                k = max(2, min(parent_p, len(ranked_parents)))
                chosen = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=k)
                parents_ir = [c[2] for c in chosen]
                parents_fit = [{"fitness": float(c[0])} for c in chosen]
                parents_ids = [str(c[1]) for c in chosen]
                parent_entries_used = [c[3] for c in chosen]
            elif llm_op in {"M1", "M2", "STRUCTURE_SHIFT", "CONSTRAINT_INJECT"} and len(ranked_parents) >= 1:
                chosen1 = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=1)[0]
                parents_ir = [chosen1[2]]
                parents_fit = [{"fitness": float(chosen1[0])}]
                parents_ids = [str(chosen1[1])]
                parent_entries_used = [chosen1[3]]
            elif llm_op == "PARADIGM_SHIFT" and len(ranked_parents) >= 2:
                chosen, shortage = _sample_cross_family_parents(
                    rng,
                    ranked_parents,
                    side="loss",
                    k=max(2, min(parent_p, len(ranked_parents))),
                )
                parents_ir = [c[2] for c in chosen]
                parents_fit = [{"fitness": float(c[0])} for c in chosen]
                parents_ids = [str(c[1]) for c in chosen]
                parent_entries_used = [c[3] for c in chosen]
            else:
                llm_op = "E1_GENERATE"
                shortage = False

            llm_seed = int(rng.randint(0, 2**31 - 1))
            call_feedback = dict(global_feedback or {})
            call_feedback["llm_call"] = {"side": "loss", "op_type": str(llm_op), "seed": llm_seed}
            if llm_op == "PARADIGM_SHIFT":
                call_feedback["llm_call"]["parent_family_shortage"] = bool(locals().get("shortage", False))
            if explore_enabled:
                call_feedback["loss_search"] = dict(call_feedback.get("loss_search") or {})
                call_feedback["loss_search"]["explore_mode"] = bool(explore_mode)
                call_feedback["loss_search"]["stagnation_generations"] = int(
                    (global_feedback or {}).get("loss_search", {}).get("stagnation_generations", 0) or 0
                )
                if bool(explore_mode):
                    # Provide lightweight guidance without changing prompts.
                    try:
                        avoid_fams = list((global_feedback or {}).get("loss_search", {}).get("avoid_families", []) or [])[:16]
                    except Exception:  # noqa: BLE001
                        avoid_fams = []
                    call_feedback["loss_search"]["avoid_families"] = avoid_fams
                    call_feedback["loss_search"]["exploration_instructions"] = [
                        "Avoid the dominant family patterns from avoid_families.",
                        "Change the objective decomposition/statistic/normalization/contrast (not just variable order).",
                        "Prefer switching signal source (delta_z vs delta_rank vs delta_regret vs cost/objective) when possible.",
                    ]

            history: List[Dict[str, Any]] = []
            base_origin = "E1"
            op_type = str(op)
            parent_ids = list(parents_ids)
            prompt_sha1 = None
            prompt_path = None

            try:
                if llm_op == "E1_GENERATE":
                    _, sha = _build_free_loss_generation_prompt(
                        p_gen,
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_gen)
                    ir = loss_llm_ops.generate_free_loss_candidate(
                        p_gen,
                        operator_whitelist=operator_whitelist,
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    base_origin = "E1"
                    op_type = "E1_GENERATE"
                    parent_ids = []
                    parent_entries_used = []
                elif llm_op == "E1":
                    _, sha = _build_free_loss_parents_prompt(
                        p_x,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        parent_block_name="PARENTS_JSON",
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_x)
                    ir = loss_llm_ops.crossover_free_loss(
                        p_x,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    base_origin = "E1"
                    op_type = "E1"
                elif llm_op == "E2":
                    _, sha = _build_free_loss_parents_prompt(
                        p_e2,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        parent_block_name="PARENTS_JSON",
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_e2)
                    ir = loss_llm_ops.e2_free_loss(
                        p_e2,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    base_origin = "E2"
                    op_type = "E2"
                elif llm_op == "PARADIGM_SHIFT":
                    _, sha = _build_free_loss_parents_prompt(
                        p_shift,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        parent_block_name="PARENTS_JSON",
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_shift)
                    ir, meta = loss_llm_ops.paradigm_shift_free_loss_with_meta(
                        p_shift,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = str(meta.get("prompt_sha1", prompt_sha1))
                    prompt_path = str(meta.get("prompt_path", prompt_path))
                    base_origin = "PARADIGM_SHIFT"
                    op_type = "LOSS_PARADIGM_SHIFT"
                elif llm_op == "STRUCTURE_SHIFT":
                    _, sha = _build_free_loss_parent_prompt(
                        p_structure,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        parent_block_name="PARENT_JSON",
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_structure)
                    ir, meta = loss_llm_ops.structure_shift_free_loss_with_meta(
                        p_structure,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = str(meta.get("prompt_sha1", prompt_sha1))
                    prompt_path = str(meta.get("prompt_path", prompt_path))
                    base_origin = "STRUCTURE_SHIFT"
                    op_type = "LOSS_STRUCTURE_SHIFT"
                elif llm_op == "CONSTRAINT_INJECT":
                    _, sha = _build_free_loss_parent_prompt(
                        p_constraint,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        parent_block_name="PARENT_JSON",
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_constraint)
                    ir, meta = loss_llm_ops.constraint_inject_free_loss_with_meta(
                        p_constraint,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = str(meta.get("prompt_sha1", prompt_sha1))
                    prompt_path = str(meta.get("prompt_path", prompt_path))
                    base_origin = "CONSTRAINT_INJECT"
                    op_type = "LOSS_CONSTRAINT_INJECT"
                elif llm_op == "M2":
                    _, sha = _build_free_loss_parent_prompt(
                        p_m2,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        parent_block_name="PARENT_JSON",
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_m2)
                    ir = loss_llm_ops.m2_tune_hparams(
                        p_m2,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    base_origin = "M2"
                    op_type = "M2"
                else:
                    _, sha = _build_free_loss_parent_prompt(
                        p_m,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        parent_block_name="PARENT_JSON",
                        prompt_context=loss_prompt_context,
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_m)
                    ir = loss_llm_ops.mutate_free_loss(
                        p_m,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        prompt_context=loss_prompt_context,
                    )
                    base_origin = "M1"
                    op_type = "M1"
                history.append(
                    {
                        "attempt": 0,
                        "side": "loss",
                        "llm_op": str(op_type),
                        "prompt_path": str(prompt_path),
                        "prompt_sha1": str(prompt_sha1),
                    }
                )
            except Exception:  # noqa: BLE001
                continue

            ok = False
            fail_reason: Dict[str, Any] = {}
            novelty_meta: Dict[str, Any] | None = None
            try:
                static_res = run_static_gates(ir, operator_whitelist=operator_whitelist)
                if not bool(static_res.ok):
                    ok = False
                    fail_reason = {"stage": "static_gate", "reason": str(static_res.reason), "trace": static_res.trace}
                else:
                    _ = compile_free_loss(ir, operator_whitelist=operator_whitelist)
                    ok = True
                    contract_ok, contract_fail = _validate_loss_operator_contract(
                        ir,
                        str(op_type),
                        parents_ir,
                        parent_entries_used,
                    )
                    if not bool(contract_ok):
                        ok = False
                        fail_reason = dict(contract_fail)
                    elif novelty_enabled and (novelty_apply_to_elites or str(base_origin) != "ELITE"):
                        nov_ok, nov_payload = _check_loss_novelty(
                            candidate=ir,
                            bank=novelty_bank,
                            max_similarity=float(novelty_max_sim),
                            neighbors=int(novelty_neighbors),
                        )
                        if not bool(nov_ok):
                            ok = False
                            fail_reason = dict(nov_payload)
                        else:
                            novelty_meta = dict(nov_payload)
            except Exception as exc:  # noqa: BLE001
                ok = False
                fail_reason = {"stage": "compile", "error": str(exc)}

            if (not ok) and repair_on_fail and p_rep:
                repaired = None
                for _ra in range(max(0, repair_attempts)):
                    # Inject call metadata into the failure payload so the cache key (prompt hash)
                    # captures op_type + seed, even for prompts that do not accept GLOBAL_FEEDBACK_JSON.
                    try:
                        fail_reason = dict(fail_reason)
                        fail_reason["llm_call"] = dict(call_feedback.get("llm_call") or {})
                        fail_reason["repair_attempt"] = int(_ra)
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        if (
                            bool(repair_cfg.get("simplify_first", True))
                            and p_m3
                            and fail_reason.get("stage") in {"static_gate", "compile"}
                        ):
                            history.append(
                                {
                                    "attempt": int(_ra),
                                    "side": "loss",
                                    "llm_op": "M3",
                                    "prompt_path": str(p_m3),
                                    "prompt_sha1": _build_free_loss_failure_prompt(
                                        p_m3,
                                        candidate=ir,
                                        failure_reason=fail_reason,
                                        global_feedback=call_feedback,
                                        block_name="CANDIDATE_AND_FAILURE_JSON",
                                        ensure_ascii=False,
                                    )[1],
                                }
                            )
                            repaired = loss_llm_ops.m3_simplify_loss(
                                p_m3,
                                candidate=ir,
                                failure_reason=fail_reason,
                                global_feedback=call_feedback,
                                prompt_context=loss_prompt_context,
                            )
                        else:
                            repaired = None
                    except Exception:  # noqa: BLE001
                        repaired = None
                    if repaired is None:
                        try:
                            history.append(
                                {
                                    "attempt": int(_ra),
                                    "side": "loss",
                                    "llm_op": "REPAIR",
                                    "prompt_path": str(p_rep),
                                    "prompt_sha1": _build_free_loss_failure_prompt(
                                        p_rep,
                                        candidate=ir,
                                        failure_reason=fail_reason,
                                        global_feedback=None,
                                        block_name="CANDIDATE_AND_FAILURE_JSON",
                                        ensure_ascii=True,
                                    )[1],
                                }
                            )
                            repaired = loss_llm_ops.repair_free_loss(
                                p_rep,
                                failed_ir=ir,
                                failure_reason=fail_reason,
                                prompt_context=loss_prompt_context,
                            )
                        except Exception:  # noqa: BLE001
                            repaired = None
                            break
                    try:
                        static_res = run_static_gates(repaired, operator_whitelist=operator_whitelist)
                        if not bool(static_res.ok):
                            fail_reason = {"stage": "static_gate", "reason": str(static_res.reason), "trace": static_res.trace}
                            continue
                        _ = compile_free_loss(repaired, operator_whitelist=operator_whitelist)
                        contract_ok, contract_fail = _validate_loss_operator_contract(
                            repaired,
                            str(op_type if str(op_type).startswith("LOSS_") else base_origin),
                            parents_ir,
                            parent_entries_used,
                        )
                        if not bool(contract_ok):
                            fail_reason = dict(contract_fail)
                            ir = repaired
                            continue
                        if novelty_enabled and (novelty_apply_to_elites or str(base_origin) != "ELITE"):
                            nov_ok, nov_payload = _check_loss_novelty(
                                candidate=repaired,
                                bank=novelty_bank,
                                max_similarity=float(novelty_max_sim),
                                neighbors=int(novelty_neighbors),
                            )
                            if not bool(nov_ok):
                                fail_reason = dict(nov_payload)
                                ir = repaired
                                continue
                            novelty_meta = dict(nov_payload)
                        ir = repaired
                        ok = True
                        op_type = "REPAIR"
                        break
                    except Exception as exc:  # noqa: BLE001
                        fail_reason = {"stage": "compile", "error": str(exc)}
                        continue

            if ok:
                if novelty_enabled and (novelty_apply_to_elites or str(base_origin) != "ELITE"):
                    _bank_add(ir)
                out.append(
                    {
                        "ir": ir,
                        "origin": "REPAIR" if str(op_type) == "REPAIR" else str(base_origin),
                        "origin_base": str(base_origin),
                        "op_type": str(op_type),
                        "parents": list(parent_ids),
                        "attempt": 0,
                        "prompt_sha1": prompt_sha1,
                        "prompt_path": prompt_path,
                        "history": history,
                        "llm_seed": llm_seed,
                        "novelty": novelty_meta,
                    }
                )

    if bool(llm_init_only):
        return out[:pop_f]
    while len(out) < pop_f:
        op = rng.choice(["mutate", "crossover", "seed"]) if parent_pool else "seed"
        if op == "seed":
            ir = _make_builtin_loss_irs(rng, 1)[0]
            out.append(
                {
                    "ir": ir,
                    "origin": "SEED",
                    "op_type": "SEED",
                    "parents": [],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                }
            )
            continue

        if op == "mutate":
            parent = rng.choice(parent_pool)
            ir = _make_builtin_loss_irs(rng, 1)[0]
            ir.name = f"{ir.name}_m_from_{str(parent.get('id',''))[:12]}"
            out.append(
                {
                    "ir": ir,
                    "origin": "SEED",
                    "op_type": "SEED_MUTATE",
                    "parents": [str(parent.get("id", ""))],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                }
            )
            continue

        p1 = rng.choice(parent_pool)
        p2 = rng.choice(parent_pool)
        ir = _make_builtin_loss_irs(rng, 1)[0]
        ir.name = f"{ir.name}_x_{str(p1.get('id',''))[:8]}_{str(p2.get('id',''))[:8]}"
        out.append(
            {
                "ir": ir,
                "origin": "SEED",
                "op_type": "SEED_CROSSOVER",
                "parents": [str(p1.get("id", "")), str(p2.get("id", ""))],
                "attempt": 0,
                "prompt_sha1": None,
                "prompt_path": None,
                "history": [],
            }
        )

    return out[:pop_f]


def _sample_pairs(
    *,
    g_ids: Sequence[str],
    f_ids: Sequence[str],
    budget: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """Sample up to `budget` pairs, prioritizing coverage and uniqueness.

    Inputs:
        g_ids: candidate builder ids
        f_ids: candidate loss ids
        budget: maximum number of pairs to return
        rng: random generator
    Output:
        list of (g_id, f_id) pairs with no duplicates
    """

    budget = max(int(budget), 0)
    if budget <= 0 or not g_ids or not f_ids:
        return []

    used: set[Tuple[str, str]] = set()
    pairs: List[Tuple[str, str]] = []

    gi = list(g_ids)
    fi = list(f_ids)
    rng.shuffle(gi)
    rng.shuffle(fi)

    # Coverage-first: spread across G while cycling F.
    ptr = 0
    for g in gi:
        if len(pairs) >= budget:
            break
        f = fi[ptr % len(fi)]
        ptr += 1
        if (g, f) in used:
            continue
        used.add((g, f))
        pairs.append((g, f))

    # Fill remaining budget uniformly at random.
    all_pairs = [(g, f) for g in g_ids for f in f_ids]
    rng.shuffle(all_pairs)
    for g, f in all_pairs:
        if len(pairs) >= budget:
            break
        if (g, f) in used:
            continue
        used.add((g, f))
        pairs.append((g, f))

    return pairs[:budget]


def _credit_assignment(
    *,
    pair_records: Sequence[Mapping[str, Any]],
    credit_mode: str,
    best_k: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Assign utilities to g and f based on evaluated pair records.

    Inputs:
        pair_records: list of pair evaluation records (from `_evaluate_pair_worker`)
        credit_mode: "mean" | "best-k" | "shapley_approx"
        best_k: used when credit_mode == "best-k"
    Output:
        (utility_by_g, utility_by_f) where larger is better.
    """

    def _pair_utility(rec: Mapping[str, Any]) -> float | None:
        if not rec.get("pair_ok"):
            return None
        fitness = rec.get("fitness")
        if isinstance(fitness, dict):
            score = fitness.get("fitness_score", fitness.get("validation_objective"))
            if score is None:
                return None
            try:
                return -float(score)
            except (TypeError, ValueError):
                return None
        if rec.get("pair_reason") == "ok_gate_only":
            # Gate-only runs can still be useful for screening; treat as neutral utility.
            return 0.0
        return None

    u_by_g: Dict[str, List[float]] = {}
    u_by_f: Dict[str, List[float]] = {}
    u_by_pair: List[Tuple[str, str, float]] = []
    for rec in pair_records:
        u = _pair_utility(rec)
        if u is None:
            continue
        g_id = str(rec.get("g_id"))
        f_id = str(rec.get("f_id"))
        u_by_g.setdefault(g_id, []).append(u)
        u_by_f.setdefault(f_id, []).append(u)
        u_by_pair.append((g_id, f_id, u))

    def _mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("-inf")

    def _bestk(xs: Sequence[float], k: int) -> float:
        if not xs:
            return float("-inf")
        kk = max(1, min(int(k), len(xs)))
        return _mean(sorted(xs, reverse=True)[:kk])

    cm = str(credit_mode or "mean").strip().lower()
    if cm == "mean":
        return {k: _mean(v) for k, v in u_by_g.items()}, {k: _mean(v) for k, v in u_by_f.items()}
    if cm in {"best-k", "best_k"}:
        return {k: _bestk(v, best_k) for k, v in u_by_g.items()}, {k: _bestk(v, best_k) for k, v in u_by_f.items()}
    if cm != "shapley_approx":
        return {k: _mean(v) for k, v in u_by_g.items()}, {k: _mean(v) for k, v in u_by_f.items()}

    # Shapley approximation: subtract counterpart's average utility as baseline.
    base_f = {f: _mean(us) for f, us in u_by_f.items()}
    base_g = {g: _mean(us) for g, us in u_by_g.items()}
    g_vals: Dict[str, List[float]] = {}
    f_vals: Dict[str, List[float]] = {}
    for g_id, f_id, u in u_by_pair:
        g_vals.setdefault(g_id, []).append(u - base_f.get(f_id, float("-inf")))
        f_vals.setdefault(f_id, []).append(u - base_g.get(g_id, float("-inf")))
    return {k: _mean(v) for k, v in g_vals.items()}, {k: _mean(v) for k, v in f_vals.items()}


def _trimmed_mean(values: Sequence[float], *, trim: float) -> float:
    xs = [float(v) for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if not xs:
        return float("inf")
    xs.sort()
    t = float(trim)
    if not (0.0 <= t < 0.5):
        t = 0.0
    k = int(t * len(xs))
    core = xs[k : len(xs) - k] if len(xs) - 2 * k > 0 else xs
    return float(sum(core) / len(core)) if core else float("inf")


def _best_k_mean(values: Sequence[float], *, k: int) -> float:
    xs = [float(v) for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if not xs:
        return float("inf")
    kk = max(1, min(int(k), len(xs)))
    xs.sort()
    best = xs[:kk]
    return float(sum(best) / len(best))


def _std(values: Sequence[float]) -> float:
    xs = [float(v) for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if len(xs) <= 1:
        return 0.0
    m = float(sum(xs) / len(xs))
    v = float(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return float(v ** 0.5)


def _credit_assignment_v2(
    *,
    pair_records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-candidate fitness from this generation's pair scores (lower is better).

    fitness = 0.6*trimmed_mean(S, trim=0.2) + 0.4*best_k_mean(S, k=floor(|S|/3)) + 0.2*std(S)/sqrt(|S|)
    """

    def _score(rec: Mapping[str, Any]) -> float:
        v = rec.get("score")
        if v is None and isinstance(rec.get("fitness"), dict):
            v = rec["fitness"].get("fitness_score", rec["fitness"].get("validation_objective"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    scores_by_g: Dict[str, List[float]] = {}
    scores_by_f: Dict[str, List[float]] = {}
    for rec in pair_records:
        g_id = str(rec.get("g_id"))
        f_id = str(rec.get("f_id"))
        s = _score(rec)
        scores_by_g.setdefault(g_id, []).append(s)
        scores_by_f.setdefault(f_id, []).append(s)

    def _fitness(scores: Sequence[float]) -> float:
        n = len(scores)
        if n <= 0:
            return float("inf")
        tm = _trimmed_mean(scores, trim=0.2)
        bk = _best_k_mean(scores, k=max(1, n // 3))
        se = _std(scores) / float(max(n, 1) ** 0.5)
        return float(0.6 * tm + 0.4 * bk + 0.2 * se)

    return {k: _fitness(v) for k, v in scores_by_g.items()}, {k: _fitness(v) for k, v in scores_by_f.items()}


def _anneal_float(*, start: float, end: float, gen: int, horizon: int) -> float:
    h = max(int(horizon), 1)
    t = min(max(int(gen), 0), h)
    if h <= 0:
        return float(end)
    w = float(t) / float(h)
    return float(start + (end - start) * w)


def _anneal_int(*, start: int, end: int, gen: int, horizon: int) -> int:
    return int(round(_anneal_float(start=float(start), end=float(end), gen=gen, horizon=horizon)))


def _descriptor_cell(x: float, y: float, *, bins: int) -> Tuple[int, int]:
    b = max(int(bins), 2)
    xx = min(max(float(x), 0.0), 1.0)
    yy = min(max(float(y), 0.0), 1.0)
    i = min(b - 1, max(0, int(xx * b)))
    j = min(b - 1, max(0, int(yy * b)))
    return int(i), int(j)


def _build_pair_descriptor(
    *,
    builder_gate_trace: Mapping[str, Any] | None,
    proxy_agg: Mapping[str, Any] | None,
    pair_count_cap: int,
    loss_scale: float,
    bins: int,
) -> Dict[str, Any]:
    bg = dict(builder_gate_trace or {})
    pa = dict(proxy_agg or {})

    def _gate_check_value(metric_name: str) -> float | None:
        checks = bg.get("checks")
        if not isinstance(checks, list):
            return None
        target = str(metric_name)
        for item in checks:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("metric_name")) != target:
                continue
            try:
                value = float(item.get("observed_value"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return float(value)
        return None

    coverage = _gate_check_value("coverage")
    if coverage is None and isinstance(bg.get("metric"), Mapping) and bg["metric"].get("metric_name") == "coverage":
        coverage = bg["metric"].get("observed_value")
    if coverage is None:
        observed = bg.get("observed")
        observed_cov = observed.get("coverage") if isinstance(observed, Mapping) else None
        coverage = bg.get("coverage", observed_cov)
    try:
        coverage_f = float(coverage)
    except (TypeError, ValueError):
        coverage_f = 0.0

    pair_count = _gate_check_value("pair_count")
    if pair_count is None:
        pair_count = bg.get("pair_count")
    try:
        pair_count_i = int(pair_count)
    except (TypeError, ValueError):
        observed = bg.get("observed")
        observed_pc = observed.get("pair_count", 0) if isinstance(observed, Mapping) else 0
        pair_count_i = int(observed_pc or 0)

    semantic = _gate_check_value("semantic_pass_rate")
    if semantic is None:
        semantic = bg.get("semantic_pass_rate")
    try:
        sem_f = float(semantic)
    except (TypeError, ValueError):
        observed = bg.get("observed")
        observed_sem = observed.get("semantic_pass_rate", 0.0) if isinstance(observed, Mapping) else 0.0
        sem_f = float(observed_sem or 0.0)

    mem_alloc_mb = bg.get("memory_peak_allocated_delta_mb")
    try:
        mem_alloc_mb_f = float(mem_alloc_mb)
    except (TypeError, ValueError):
        mem_alloc_mb_f = None

    mem_reserved_mb = bg.get("memory_peak_reserved_delta_mb")
    try:
        mem_reserved_mb_f = float(mem_reserved_mb)
    except (TypeError, ValueError):
        mem_reserved_mb_f = None

    eff = pa.get("proxy_effective_grad_ratio_mean", pa.get("effective_grad_ratio"))
    ess = pa.get("proxy_ess_ratio_mean", pa.get("ess_ratio"))
    loss = pa.get("proxy_loss_mean", pa.get("loss"))
    try:
        eff_f = float(eff)
    except (TypeError, ValueError):
        eff_f = 0.0
    try:
        ess_f = float(ess)
    except (TypeError, ValueError):
        ess_f = 0.0
    try:
        loss_f = float(loss)
    except (TypeError, ValueError):
        loss_f = float("inf")

    # Builder descriptor in [0,1]^2: (coverage, normalized pair_count)
    cap = max(int(pair_count_cap), 1)
    g_x = min(max(coverage_f, 0.0), 1.0)
    g_y = min(max(float(pair_count_i) / float(cap), 0.0), 1.0)
    g_cell = _descriptor_cell(g_x, g_y, bins=bins)

    # Loss descriptor in [0,1]^2: (effective_grad_ratio, normalized loss)
    ls = max(float(loss_scale), 1e-6)
    f_x = min(max(eff_f, 0.0), 1.0)
    f_y = min(max(loss_f / ls, 0.0), 1.0) if loss_f != float("inf") else 1.0
    f_cell = _descriptor_cell(f_x, f_y, bins=bins)

    return {
        "g": {
            "coverage": float(coverage_f),
            "pair_count": int(pair_count_i),
            "semantic_pass_rate": float(sem_f),
            "memory_peak_allocated_mb": mem_alloc_mb_f,
            "memory_peak_reserved_mb": mem_reserved_mb_f,
            "x": float(g_x),
            "y": float(g_y),
            "cell": [int(g_cell[0]), int(g_cell[1])],
        },
        "f": {
            "effective_grad_ratio": float(eff_f),
            "ess_ratio": float(ess_f),
            "proxy_loss_mean": float(loss_f) if loss_f != float("inf") else None,
            "x": float(f_x),
            "y": float(f_y),
            "cell": [int(f_cell[0]), int(f_cell[1])],
        },
    }


def _archive_key(cell: Tuple[int, int]) -> str:
    return f"{int(cell[0])},{int(cell[1])}"


def _archive_add(
    archive: Dict[str, List[Dict[str, Any]]],
    *,
    cell: Tuple[int, int],
    entry: Mapping[str, Any],
    score: float,
    per_cell: int,
) -> None:
    k = _archive_key(cell)
    lst = list(archive.get(k, []))
    e = dict(entry)
    e["archive_cell"] = [int(cell[0]), int(cell[1])]
    e["archive_score"] = float(score)
    lst.append(e)
    lst.sort(key=lambda x: _stored_selection_sort_key(x, fallback_key="archive_score"))
    seen: set[str] = set()
    dedup: List[Dict[str, Any]] = []
    for it in lst:
        sid = str(it.get("signature", it.get("id", "")))
        if sid in seen:
            continue
        seen.add(sid)
        dedup.append(it)
        if len(dedup) >= int(per_cell):
            break
    archive[k] = dedup


def _archive_flatten(
    archive: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_items: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for _, lst in archive.items():
        for it in lst:
            items.append(dict(it))
    items.sort(key=lambda x: _stored_selection_sort_key(x, fallback_key="archive_score"))
    return items[: max(0, int(max_items))]


def _update_hof(
    hof: List[Dict[str, Any]],
    *,
    candidates: Sequence[Mapping[str, Any]],
    max_size: int,
) -> List[Dict[str, Any]]:
    """Update Hall-of-Fame with new candidates (dedup by signature), keep lowest scores."""

    out = list(hof)
    seen = {str(x.get("signature", "")) for x in out if isinstance(x, dict)}
    for c in candidates:
        sig = str(c.get("signature", ""))
        if not sig or sig in seen:
            continue
        out.append(dict(c))
        seen.add(sig)
    out.sort(key=lambda x: _stored_selection_sort_key(x, fallback_key="fitness"))
    return out[: max(0, int(max_size))]


G_REF_ID = "g_ref"
F_REF_ID = "f_ref"


def _ref_builder_ir() -> PreferenceBuilderIR:
    code = (
        "def generated_builder(feature_cache, extra):\n"
        "    objective = feature_cache['objective']\n"
        "    mask = objective[:, :, None] < objective[:, None, :]\n"
        "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
        "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'ref_all_pairs'})\n"
    )
    return PreferenceBuilderIR(
        name="ref_all_pairs_builder",
        intuition="Reference builder: all winner/loser pairs by objective ordering.",
        implementation_hint=PreferenceBuilderImplementationHint(
            expects=["objective", "log_prob"],
            returns="PrefBatch",
            mode="pairwise",
        ),
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "uniform_none",
            "constraint_family": "none",
        },
        operators_used=["ref_all_pairs"],
        code=code,
    )


def _ref_loss_ir() -> FreeLossIR:
    code = (
        "def generated_loss(batch, model_output, extra):\n"
        "    alpha = float(extra.get('alpha', 1.0))\n"
        "    logit = alpha * (batch['log_prob_w'] - batch['log_prob_l'])\n"
        "    w = batch.get('weight')\n"
        "    loss = -ops.logsigmoid(logit)\n"
        "    if w is not None:\n"
        "        loss = loss * w\n"
        "    return loss.mean()\n"
    )
    return FreeLossIR(
        name="ref_logsigmoid",
        intuition="Reference loss: negative logsigmoid on log-prob winner-loser gap.",
        pseudocode="loss = -logsigmoid(alpha*(log_prob_w-log_prob_l))",
        hyperparams={
            "alpha": 1.0,
            "paradigm_family": "pairwise_margin",
            "signal_family": "logprob_gap",
            "link_family": "logsigmoid",
            "agg_family": "mean",
            "constraint_family": "weight_optional",
        },
        operators_used=["logsigmoid"],
        implementation_hint=FreeLossImplementationHint(
            expects=["cost_a", "cost_b", "log_prob_w", "log_prob_l", "weight"],
            returns="scalar",
            mode="pairwise",
        ),
        code=code,
        theoretical_basis="",
    )


def _ensure_reference_compiled(
    *,
    compiled_g: Dict[str, CompiledPreferenceBuilder],
    compiled_f: Dict[str, CompiledFreeLoss],
    operator_whitelist: Sequence[str],
) -> None:
    if G_REF_ID not in compiled_g:
        compiled_g[G_REF_ID] = compile_preference_builder(_ref_builder_ir(), operator_whitelist=operator_whitelist)
    if F_REF_ID not in compiled_f:
        ref_ir = _ref_loss_ir()
        static_ref = run_static_gates(ref_ir, operator_whitelist=operator_whitelist)
        if not static_ref.ok:
            raise RuntimeError(f"Reference loss failed static gates: {static_ref.reason}")
        compiled_f[F_REF_ID] = compile_free_loss(ref_ir, operator_whitelist=operator_whitelist)


def _compile_hof_candidates(
    *,
    hof_g: Sequence[Mapping[str, Any]],
    hof_f: Sequence[Mapping[str, Any]],
    compiled_g: Dict[str, CompiledPreferenceBuilder],
    compiled_f: Dict[str, CompiledFreeLoss],
    operator_whitelist: Sequence[str],
) -> Tuple[List[str], List[str]]:
    hof_g_ids: List[str] = []
    for e in hof_g:
        gid = e.get("id")
        ir = e.get("ir")
        if not gid or not isinstance(ir, dict):
            continue
        gid_s = str(gid)
        hof_g_ids.append(gid_s)
        if gid_s in compiled_g:
            continue
        try:
            compiled_g[gid_s] = compile_preference_builder(pref_builder_ir_from_json(ir), operator_whitelist=operator_whitelist)
        except Exception:  # noqa: BLE001
            continue

    hof_f_ids: List[str] = []
    for e in hof_f:
        fid = e.get("id")
        ir = e.get("ir")
        if not fid or not isinstance(ir, dict):
            continue
        fid_s = str(fid)
        hof_f_ids.append(fid_s)
        if fid_s in compiled_f:
            continue
        try:
            compiled_f[fid_s] = compile_free_loss(free_loss_ir_from_json(ir), operator_whitelist=operator_whitelist)
        except Exception:  # noqa: BLE001
            continue

    return hof_g_ids, hof_f_ids


def _cheap_eval_pair_cached(
    *,
    caches: PrefLossEvalCaches,
    compiled_g: Mapping[str, CompiledPreferenceBuilder],
    compiled_f: Mapping[str, CompiledFreeLoss],
    rollout_feature_caches: Sequence[Mapping[str, torch.Tensor]],
    cfg_yaml: Mapping[str, Any],
    eval_sig: str,
    gid: str,
    fid: str,
    generation: int,
    pair_index: int,
    seed_used: int,
    seed_sig: str,
    pref_batch_id_offset: int,
    stage: str,
    reasons: Sequence[str],
    proxy_device_str: str,
    joint_gate_kwargs: Mapping[str, Any],
    proxy_weights: Mapping[str, float],
    bins: int,
    pair_count_cap: int,
    loss_scale: float,
    cheap_gate_on: bool,
) -> Dict[str, Any]:
    """Cheap proxy evaluation (no training), with caching and required JSONL fields.

    Output record always contains:
      generation, g_id, f_id, score, proxy_metrics, seed_signature, descriptor
    """

    cache_key = (str(gid), str(fid), str(eval_sig))
    cached = caches.get_pair(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("seed_records"), dict) and str(seed_sig) in cached["seed_records"]:
        out = dict(cached)
        out["generation"] = int(generation)
        out["pair_index"] = int(pair_index)
        out["cached"] = True
        out["stage"] = str(stage)
        return out

    g_comp = compiled_g.get(str(gid))
    f_comp = compiled_f.get(str(fid))
    base: Dict[str, Any] = {
        "generation": int(generation),
        "pair_index": int(pair_index),
        "g_id": str(gid),
        "f_id": str(fid),
        "eval_budget_signature": str(eval_sig),
        "seed_signature": str(seed_sig),
        "seed_used": int(seed_used),
        "device": str(proxy_device_str),
        "stage": str(stage),
        "reasons": list(reasons),
        "cached": False,
    }
    if g_comp is None or f_comp is None:
        base["pair_ok"] = False
        base["pair_reason"] = "compile_missing"
        base["score"] = float("inf")
        base["proxy_metrics"] = {}
        base["descriptor"] = {}
        caches.set_pair(cache_key, base)
        return dict(caches.get_pair(cache_key) or base)

    batch_metrics: List[Dict[str, Any]] = []
    builder_gate_first: Dict[str, Any] | None = None
    builder_ok = True
    joint_ok_count = 0
    joint_total_count = 0
    first_joint_fail_reason: str | None = None
    first_joint_fail_trace: Dict[str, Any] | None = None
    joint_failure_kinds: collections.Counter[str] = collections.Counter()

    total_batches = int(len(rollout_feature_caches))
    progress_every_s = float(cfg_yaml.get("progress_log_every_s_proxy_batch", 0) or 0)
    t_last_progress = time.time()
    pref_cache_enabled = bool(cfg_yaml.get("pref_cache_enabled", True))

    for local_batch_id, fc in enumerate(rollout_feature_caches):
        if progress_every_s > 0 and (time.time() - t_last_progress) >= progress_every_s and (local_batch_id + 1) < total_batches:
            LOGGER.info(
                "Proxy progress gen=%d stage=%s pair_index=%d g_id=%s f_id=%s batch=%d/%d %s",
                int(generation),
                str(stage),
                int(pair_index),
                str(gid),
                str(fid),
                int(local_batch_id + 1),
                int(total_batches),
                _cache_brief(caches),
            )
            t_last_progress = time.time()

        pref_cache_hit = False
        if pref_cache_enabled:
            pref_key = (str(gid), int(pref_batch_id_offset + local_batch_id))
            cached_pref = caches.get_pref(pref_key)
            if cached_pref is not None:
                pref = cached_pref
                pref_cache_hit = True
            else:
                pref = _build_pref_batch_with_memory_trace(
                    g_comp,
                    fc,
                    {"stage": "proxy", "seed_signature": str(seed_sig)},
                )
                caches.set_pref(pref_key, pref)
        else:
            # For VRAM stability: avoid storing PrefBatch tensors in a long-lived cache.
            pref = _build_pref_batch_with_memory_trace(
                g_comp,
                fc,
                {"stage": "proxy", "seed_signature": str(seed_sig)},
            )
        if builder_gate_first is None:
            bg = run_preference_builder_gates(
                pref,
                feature_cache=fc,
                min_pairs=int(cfg_yaml.get("builder_min_pairs", 1) or 1),
                min_coverage=float(cfg_yaml.get("builder_min_coverage", 0.0) or 0.0),
                max_pairs_per_instance=int(cfg_yaml.get("builder_max_pairs_per_instance", 4096) or 4096),
                weight_nonneg=bool(cfg_yaml.get("builder_weight_nonneg", True)),
                semantic_tolerance=float(cfg_yaml.get("builder_semantic_tolerance", 0.0) or 0.0),
                semantic_min_pass_rate=float(cfg_yaml.get("builder_semantic_min_pass_rate", 1.0) or 1.0),
            )
            bg.trace = _enrich_builder_gate_trace_with_memory(bg.trace, pref, cache_hit=pref_cache_hit)
            builder_ok = bool(bg.ok)
            builder_gate_first = {
                "builder_gate_ok": bool(bg.ok),
                "builder_gate_reason": str(bg.reason),
                "builder_gate_trace": bg.trace,
                "pair_count": bg.pair_count,
                "coverage": bg.coverage,
                "semantic_pass_rate": bg.semantic_pass_rate,
                "memory_peak_allocated_delta_mb": (
                    None if not isinstance(bg.trace, dict) else bg.trace.get("memory_peak_allocated_delta_mb")
                ),
                "memory_peak_reserved_delta_mb": (
                    None if not isinstance(bg.trace, dict) else bg.trace.get("memory_peak_reserved_delta_mb")
                ),
            }

        m = proxy_metrics_for_pair_on_batch(
            g=g_comp,
            f=f_comp,
            feature_cache=fc,
            pref_batch=pref,
            joint_gate_kwargs=joint_gate_kwargs,
        )
        joint_batch_ok = bool(m.get("joint_ok", False))
        joint_total_count += 1
        if joint_batch_ok:
            joint_ok_count += 1
        else:
            jr = str(m.get("joint_reason", "") or "")
            jt = m.get("joint_trace")
            kind = None
            if isinstance(jt, dict):
                kind = str(jt.get("failure_kind") or jt.get("failed_gate") or "unknown")
            if kind:
                joint_failure_kinds[kind] += 1
            if first_joint_fail_reason is None and jr:
                first_joint_fail_reason = jr
            if first_joint_fail_trace is None and isinstance(jt, dict):
                first_joint_fail_trace = dict(jt)
        batch_metrics.append(m)

    joint_batch_pass_rate_threshold = _safe_float(
        cfg_yaml.get("proxy_joint_batch_pass_rate_threshold", 1.0),
        1.0,
    )
    if not math.isfinite(joint_batch_pass_rate_threshold):
        joint_batch_pass_rate_threshold = 1.0
    joint_batch_pass_rate_threshold = max(0.0, min(1.0, float(joint_batch_pass_rate_threshold)))
    joint_pass_rate = (
        float(joint_ok_count) / float(joint_total_count)
        if joint_total_count > 0
        else 1.0
    )
    joint_ok = bool(joint_pass_rate >= joint_batch_pass_rate_threshold)
    if (not joint_ok) and first_joint_fail_reason is None:
        first_joint_fail_reason = "joint_batch_pass_rate_below_threshold"
    joint_gate_trace = None
    if not joint_ok:
        joint_gate_trace = {
            "failed_gate": "JointPreference",
            "failure_kind": "joint_batch_pass_rate_below_threshold",
            "observed": {
                "joint_ok_batches": int(joint_ok_count),
                "joint_total_batches": int(joint_total_count),
                "joint_pass_rate": float(joint_pass_rate),
            },
            "threshold": {
                "min_joint_pass_rate": float(joint_batch_pass_rate_threshold),
            },
            "first_failure_reason": first_joint_fail_reason,
            "first_failure_trace": dict(first_joint_fail_trace) if isinstance(first_joint_fail_trace, dict) else None,
            "joint_failure_kind_counts": dict(joint_failure_kinds),
        }

    proxy_score, proxy_agg = aggregate_proxy_metrics(batch_metrics, proxy_weights=dict(proxy_weights))
    per_batch_summary = [
        {
            "loss": float(m.get("loss", float("inf"))),
            "effective_grad_ratio": float(m.get("effective_grad_ratio", 0.0)),
            "ess_ratio": float(m.get("ess_ratio", 0.0)),
            "pair_count": int(m.get("pair_count", 0)),
            "joint_ok": bool(m.get("joint_ok", False)),
            "joint_reason": str(m.get("joint_reason", "") or ""),
            "joint_failure_kind": (
                str(m.get("joint_trace", {}).get("failure_kind"))
                if isinstance(m.get("joint_trace"), dict) and m["joint_trace"].get("failure_kind") is not None
                else None
            ),
        }
        for m in batch_metrics
    ]
    proxy_metrics: Dict[str, Any] = dict(proxy_agg)
    proxy_metrics["batches"] = per_batch_summary
    proxy_metrics["joint_ok_batches"] = int(joint_ok_count)
    proxy_metrics["joint_total_batches"] = int(joint_total_count)
    proxy_metrics["joint_pass_rate"] = float(joint_pass_rate)
    proxy_metrics["joint_batch_pass_rate_threshold"] = float(joint_batch_pass_rate_threshold)
    if joint_failure_kinds:
        proxy_metrics["joint_failure_kinds"] = dict(joint_failure_kinds)

    descriptor = _build_pair_descriptor(
        builder_gate_trace=(builder_gate_first or {}).get("builder_gate_trace") if builder_gate_first else None,
        proxy_agg=proxy_agg,
        pair_count_cap=int(pair_count_cap),
        loss_scale=float(loss_scale),
        bins=int(bins),
    )

    base.update(
        {
            "proxy_score": float(proxy_score),
            "score": float(proxy_score),
            "proxy_metrics": proxy_metrics,
            "descriptor": descriptor,
            "builder_gate_ok": None if builder_gate_first is None else builder_gate_first.get("builder_gate_ok"),
            "builder_gate_reason": None if builder_gate_first is None else builder_gate_first.get("builder_gate_reason"),
            "builder_gate_trace": None if builder_gate_first is None else builder_gate_first.get("builder_gate_trace"),
            "joint_gate_ok": bool(joint_ok),
            "joint_gate_reason": (
                "ok"
                if joint_ok
                else "joint_failed:joint_batch_pass_rate_below_threshold"
            ),
            "joint_gate_trace": joint_gate_trace,
        }
    )
    base.update(_run_co_alignment_gates_for_loss(f_comp, cfg_yaml))

    co_ok = bool(base.get("co_ok", True))
    if cheap_gate_on and (not builder_ok or not joint_ok):
        base["pair_ok"] = False
        base["pair_reason"] = "cheap_proxy_gate_failed"
        base["score"] = float("inf")
    elif cheap_gate_on and (not co_ok):
        base["pair_ok"] = False
        base["pair_reason"] = "co_gate_failed"
        base["score"] = float("inf")
    else:
        base["pair_ok"] = True
        base["pair_reason"] = "ok_proxy"

    caches.set_pair(cache_key, base)
    out = dict(caches.get_pair(cache_key) or base)
    out["generation"] = int(generation)
    out["pair_index"] = int(pair_index)
    out["stage"] = str(stage)
    return out


def _build_coverage_plus_bandit_pairs(
    *,
    cfg_yaml: Mapping[str, Any],
    gen: int,
    generations: int,
    pairing_budget: int,
    rng: random.Random,
    new_g_ids: Sequence[str],
    new_f_ids: Sequence[str],
    elite_g_ids: Sequence[str],
    elite_f_ids: Sequence[str],
    hof_g_ids: Sequence[str],
    hof_f_ids: Sequence[str],
    g_id_pool: Sequence[str],
    f_id_pool: Sequence[str],
    caches: PrefLossEvalCaches,
    eval_sig: str,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], List[str]]]:
    """Pairing: coverage first, then bandit acquisition.

    Output:
        (pairs, reasons_by_pair)
    """

    top_e = int(cfg_yaml.get("coverage_top_e_elites", 4) or 4)
    elite_g_top = list(elite_g_ids)[: max(0, top_e)]
    elite_f_top = list(elite_f_ids)[: max(0, top_e)]

    horizon = int(cfg_yaml.get("coverage_anneal_generations", 50) or 50)
    k_elite = _anneal_int(
        start=int(cfg_yaml.get("coverage_k_elite_start", 1) or 1),
        end=int(cfg_yaml.get("coverage_k_elite_end", 2) or 2),
        gen=int(gen),
        horizon=horizon,
    )
    k_hof_cov = _anneal_int(
        start=int(cfg_yaml.get("coverage_k_hof_start", 1) or 1),
        end=int(cfg_yaml.get("coverage_k_hof_end", 1) or 1),
        gen=int(gen),
        horizon=horizon,
    )
    k_rand = _anneal_int(
        start=int(cfg_yaml.get("coverage_k_random_start", 1) or 1),
        end=int(cfg_yaml.get("coverage_k_random_end", 0) or 0),
        gen=int(gen),
        horizon=horizon,
    )
    k_hof_force = _anneal_int(
        start=int(cfg_yaml.get("crossplay_k_hof_start", 1) or 1),
        end=int(cfg_yaml.get("crossplay_k_hof_end", 2) or 2),
        gen=int(gen),
        horizon=horizon,
    )
    k_hof = max(int(k_hof_cov), int(k_hof_force))

    k_total_target = int(k_elite + k_hof + k_rand)
    k_total_max = max(1, int(pairing_budget) // max(1, max(len(new_g_ids), len(new_f_ids), 1)))
    k_total = min(k_total_target, k_total_max)

    def _scaled_counts() -> Tuple[int, int, int]:
        if k_total_target <= 0:
            return 0, 0, 0
        if k_total >= k_total_target:
            return int(k_elite), int(k_hof), int(k_rand)
        parts = [("elite", float(k_elite)), ("hof", float(k_hof)), ("rand", float(k_rand))]
        s = sum(p for _, p in parts) or 1.0
        raw = {n: (p / s) * float(k_total) for n, p in parts}
        out = {n: int(math.floor(v)) for n, v in raw.items()}
        rem = k_total - sum(out.values())
        for n, _ in sorted(parts, key=lambda x: raw[x[0]] - math.floor(raw[x[0]]), reverse=True):
            if rem <= 0:
                break
            out[n] += 1
            rem -= 1
        return int(out["elite"]), int(out["hof"]), int(out["rand"])

    k_elite_s, k_hof_s, k_rand_s = _scaled_counts()

    reasons_by_pair: Dict[Tuple[str, str], List[str]] = {}
    pairs: List[Tuple[str, str]] = []
    used: set[Tuple[str, str]] = set()

    def _choice(xs: Sequence[str]) -> str | None:
        ys = [str(x) for x in xs if x]
        if not ys:
            return None
        return str(rng.choice(ys))

    def _add(gid: str, fid: str, reason: str) -> None:
        key = (str(gid), str(fid))
        if key in used:
            reasons_by_pair.setdefault(key, []).append(str(reason))
            return
        if len(pairs) >= int(pairing_budget):
            return
        used.add(key)
        pairs.append(key)
        reasons_by_pair.setdefault(key, []).append(str(reason))

    for gid in list(new_g_ids):
        for _ in range(int(k_elite_s)):
            opp = _choice(elite_f_top or f_id_pool)
            if opp is not None:
                _add(str(gid), str(opp), "coverage_g_vs_elite_f")
        for _ in range(int(k_hof_s)):
            opp = _choice(hof_f_ids or f_id_pool)
            if opp is not None:
                _add(str(gid), str(opp), "coverage_g_vs_hof_f")
        for _ in range(int(k_rand_s)):
            opp = _choice(f_id_pool)
            if opp is not None:
                _add(str(gid), str(opp), "coverage_g_vs_random_f")

    for fid in list(new_f_ids):
        for _ in range(int(k_elite_s)):
            opp = _choice(elite_g_top or g_id_pool)
            if opp is not None:
                _add(str(opp), str(fid), "coverage_f_vs_elite_g")
        for _ in range(int(k_hof_s)):
            opp = _choice(hof_g_ids or g_id_pool)
            if opp is not None:
                _add(str(opp), str(fid), "coverage_f_vs_hof_g")
        for _ in range(int(k_rand_s)):
            opp = _choice(g_id_pool)
            if opp is not None:
                _add(str(opp), str(fid), "coverage_f_vs_random_g")

    remaining = int(pairing_budget) - len(pairs)
    if remaining <= 0:
        return pairs[: int(pairing_budget)], reasons_by_pair

    prior_mu = float(cfg_yaml.get("bandit_prior_mu", 1.0) or 1.0)
    c_start = float(cfg_yaml.get("bandit_c_start", 1.0) or 1.0)
    c_end = float(cfg_yaml.get("bandit_c_end", 0.3) or 0.3)
    c_now = _anneal_float(start=c_start, end=c_end, gen=int(gen), horizon=max(1, generations - 1))

    T = 1.0
    for rec in caches.pair_cache.values():
        if not isinstance(rec, dict):
            continue
        seeds = rec.get("seed_signature")
        if isinstance(seeds, (list, tuple)):
            T += float(len(seeds))
        elif seeds:
            T += 1.0

    cand_pairs: List[Tuple[str, str]] = [(str(g), str(f)) for g in g_id_pool for f in f_id_pool]
    rng.shuffle(cand_pairs)

    def _mu_n(gid: str, fid: str) -> Tuple[float, float]:
        rec = caches.get_pair((gid, fid, str(eval_sig)))
        if isinstance(rec, dict):
            try:
                mu = float(rec.get("score", prior_mu))
            except (TypeError, ValueError):
                mu = float(prior_mu)
            seeds = rec.get("seed_signature")
            if isinstance(seeds, (list, tuple)):
                n = float(len(seeds))
            elif seeds:
                n = 1.0
            else:
                n = 0.0
            return float(mu), float(n)
        return float(prior_mu), 0.0

    cand_pairs.sort(key=lambda p: _mu_n(p[0], p[1])[0] - c_now * math.sqrt(max(0.0, math.log(T)) / (_mu_n(p[0], p[1])[1] + 1.0)))
    for gid, fid in cand_pairs:
        if remaining <= 0:
            break
        if (gid, fid) in used:
            continue
        _add(gid, fid, "bandit")
        remaining -= 1

    return pairs[: int(pairing_budget)], reasons_by_pair


def _evaluate_pair_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate a single (g,f) pair, optionally including high-fidelity training.

    Inputs:
        payload: dict with keys:
            - generation: int
            - pair_index: int
            - g_entry: dict with keys {id, ir}
            - f_entry: dict with keys {id, ir}
            - cfg_yaml: dict (YAML config)
            - device_str: str
            - operator_whitelist: list[str]
            - cheap_gate_on: bool
            - high_fidelity_on: bool
    Output:
        record dict suitable for `pairs.jsonl`.
    """

    t0 = time.time()
    generation = int(payload["generation"])
    pair_index = int(payload["pair_index"])
    g_entry = dict(payload["g_entry"])
    f_entry = dict(payload["f_entry"])
    cfg = dict(payload["cfg_yaml"])
    device_str = str(payload["device_str"])
    run_dir = payload.get("run_dir")
    run_dir_s = str(run_dir) if isinstance(run_dir, (str, os.PathLike)) and run_dir else None
    operator_whitelist = list(payload.get("operator_whitelist", []))
    cheap_gate_on = bool(payload.get("cheap_gate_on", True))
    high_fidelity_on = bool(payload.get("high_fidelity_on", True))
    baseline_epoch_objectives = payload.get("baseline_epoch_objectives")
    if not isinstance(baseline_epoch_objectives, list) or not baseline_epoch_objectives:
        baseline_epoch_objectives = None
    baseline_early_valid = payload.get("baseline_early_valid")
    try:
        baseline_early_valid_f = float(baseline_early_valid) if baseline_early_valid is not None else None
    except (TypeError, ValueError):
        baseline_early_valid_f = None
    early_eval_steps = payload.get("early_eval_steps", 0)
    try:
        early_eval_steps_i = int(early_eval_steps or 0)
    except (TypeError, ValueError):
        early_eval_steps_i = 0
    eval_sig = str(payload.get("eval_budget_signature", ""))
    proxy_record = payload.get("proxy_record")
    if not isinstance(proxy_record, dict):
        proxy_record = None

    g_ir = pref_builder_ir_from_json(g_entry["ir"])
    f_ir = free_loss_ir_from_json(f_entry["ir"])

    record: Dict[str, Any] = {
        "generation": generation,
        "pair_index": pair_index,
        "g_id": str(g_entry["id"]),
        "f_id": str(f_entry["id"]),
        "g_ir": dict(g_entry["ir"]),
        "f_ir": dict(f_entry["ir"]),
        "device": device_str,
        "cheap_gate_on": cheap_gate_on,
        "high_fidelity_on": high_fidelity_on,
        "eval_budget_signature": eval_sig,
        # Required JSONL fields (filled from proxy stage if provided).
        "score": None,
        "proxy_metrics": None,
        "seed_signature": None,
        "descriptor": None,
    }
    if proxy_record is not None:
        record["proxy_metrics"] = proxy_record.get("proxy_metrics")
        record["seed_signature"] = proxy_record.get("seed_signature")
        record["descriptor"] = proxy_record.get("descriptor")
        record["proxy_score"] = proxy_record.get("score", proxy_record.get("proxy_score"))
    record["joint_gate_repaired"] = False
    record["joint_gate_repair_attempts"] = 0
    record["f_id_before_repair"] = None
    record["f_id_after_repair"] = None
    record["joint_gate_repair_reports"] = []
    record["builder_gate_repaired"] = False
    record["builder_gate_repair_attempts"] = 0
    record["g_id_before_repair"] = None
    record["g_id_after_repair"] = None
    record["builder_gate_repair_reports"] = []

    try:
        compiled_g = compile_preference_builder(g_ir, operator_whitelist=operator_whitelist)
        record["g_compile_ok"] = True
        record["g_compile_reason"] = "ok"
    except PreferenceBuilderCompileError as exc:
        record["pair_ok"] = False
        record["pair_reason"] = "g_compile_failed"
        record["g_compile_ok"] = False
        record["g_compile_reason"] = str(exc)
        record["score"] = float("inf")
        record["elapsed_s"] = float(time.time() - t0)
        return record

    try:
        static_res: StaticGateResult = run_static_gates(f_ir, operator_whitelist=operator_whitelist)
        record["f_static_ok"] = bool(static_res.ok)
        record["f_static_reason"] = str(static_res.reason)
        if not static_res.ok:
            raise CompileError(f"static_gate_failed: {static_res.reason}")
        compiled_f = compile_free_loss(f_ir, operator_whitelist=operator_whitelist)
        record["f_compile_ok"] = True
        record["f_compile_reason"] = "ok"
    except Exception as exc:  # noqa: BLE001
        record["pair_ok"] = False
        record["pair_reason"] = "f_compile_failed"
        record["f_compile_ok"] = False
        record["f_compile_reason"] = str(exc)
        record["score"] = float("inf")
        record["elapsed_s"] = float(time.time() - t0)
        return record

    variant = "hidden" if bool(cfg.get("hidden_dynamic_gates_enabled", False)) else "visible"
    feature_cache = _dummy_feature_cache(
        batch_size=int(cfg.get("cheap_gate_batch_size", 8) or 8),
        k=int(cfg.get("cheap_gate_k", 16) or 16),
        variant=variant,
    )
    pref_batch = _build_pref_batch_with_memory_trace(compiled_g, feature_cache, {"stage": "cheap_gate"})

    builder_gate = run_preference_builder_gates(
        pref_batch,
        feature_cache=feature_cache,
        min_pairs=int(cfg.get("builder_min_pairs", 1) or 1),
        min_coverage=float(cfg.get("builder_min_coverage", 0.0) or 0.0),
        max_pairs_per_instance=int(cfg.get("builder_max_pairs_per_instance", 4096) or 4096),
        weight_nonneg=bool(cfg.get("builder_weight_nonneg", True)),
        semantic_tolerance=float(cfg.get("builder_semantic_tolerance", 0.0) or 0.0),
        semantic_min_pass_rate=float(cfg.get("builder_semantic_min_pass_rate", 1.0) or 1.0),
    )
    builder_gate.trace = _enrich_builder_gate_trace_with_memory(builder_gate.trace, pref_batch, cache_hit=False)
    builder_gate_cfg = {
        "min_pairs": int(cfg.get("builder_min_pairs", 1) or 1),
        "min_coverage": float(cfg.get("builder_min_coverage", 0.0) or 0.0),
        "max_pairs_per_instance": int(cfg.get("builder_max_pairs_per_instance", 4096) or 4096),
        "weight_nonneg": bool(cfg.get("builder_weight_nonneg", True)),
        "semantic_tolerance": float(cfg.get("builder_semantic_tolerance", 0.0) or 0.0),
        "semantic_min_pass_rate": float(cfg.get("builder_semantic_min_pass_rate", 1.0) or 1.0),
    }
    builder_gate_repair_reports: List[Dict[str, Any]] = []
    builder_llm_cfg = cfg.get("builder_llm", {}) if isinstance(cfg.get("builder_llm"), dict) else {}
    builder_repair_cfg = (
        builder_llm_cfg.get("repair", {}) if isinstance(builder_llm_cfg.get("repair"), dict) else {}
    )
    builder_gate_repair_enabled = bool(
        cfg.get("builder_gate_repair_enabled", builder_repair_cfg.get("enabled", False))
    )
    try:
        builder_gate_repair_max_attempts = max(
            int(
                cfg.get(
                    "builder_gate_repair_max_attempts",
                    builder_repair_cfg.get("max_attempts", 1),
                )
                or 0
            ),
            0,
        )
    except (TypeError, ValueError):
        builder_gate_repair_max_attempts = 0
    builder_gate_repair_simplify_first = bool(
        cfg.get(
            "builder_gate_repair_simplify_first",
            builder_repair_cfg.get("simplify_first", True),
        )
    )
    llm_prompts_cfg = cfg.get("llm_prompts", {}) if isinstance(cfg.get("llm_prompts"), dict) else {}
    builder_gate_repair_prompt_path = _abs_from_repo_root(
        str(
            cfg.get(
                "builder_gate_repair_prompt_path",
                llm_prompts_cfg.get("builder_repair", "PTP/prompts/pref_builder_repair.txt"),
            )
            or "PTP/prompts/pref_builder_repair.txt"
        )
    )
    builder_gate_repair_m3_prompt_path = _abs_from_repo_root(
        str(
            cfg.get(
                "builder_gate_repair_m3_prompt_path",
                llm_prompts_cfg.get("builder_m3", "PTP/prompts/pref_builder_m3.txt"),
            )
            or "PTP/prompts/pref_builder_m3.txt"
        )
    )
    if (
        builder_gate_repair_enabled
        and (not builder_gate.ok)
        and str(g_entry["id"]) != G_REF_ID
        and builder_gate_repair_max_attempts > 0
    ):
        try:
            _configure_builder_llm_for_worker(cfg=cfg, run_dir=run_dir_s)
        except Exception:  # noqa: BLE001
            pass

        failure_report = _builder_failure_report(
            stage="gate",
            reason=str(builder_gate.reason),
            trace=builder_gate.trace,
        )
        repaired_g_ir, repair_meta = _repair_builder_candidate_loop(
            g_ir,
            failure_report=failure_report,
            operator_whitelist=operator_whitelist,
            gate_cfg=builder_gate_cfg,
            llm_prompts={
                "builder_m3": builder_gate_repair_m3_prompt_path,
                "builder_repair": builder_gate_repair_prompt_path,
            },
            global_feedback=None,
            max_attempts=int(builder_gate_repair_max_attempts),
            simplify_first=bool(builder_gate_repair_simplify_first),
        )
        repair_record: Dict[str, Any] = {
            "record_type": "builder_gate_repair_attempt",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generation": int(generation),
            "pair_index": int(pair_index),
            "stage": "builder_gate",
            "g_id_before": str(record["g_id"]),
            "builder_gate_reason_before": str(builder_gate.reason),
            "builder_gate_trace_before": builder_gate.trace,
            "repair_prompt_path": builder_gate_repair_prompt_path,
            "m3_prompt_path": builder_gate_repair_m3_prompt_path,
            "max_attempts": int(builder_gate_repair_max_attempts),
            "meta": dict(repair_meta) if isinstance(repair_meta, dict) else {},
            "final_ok": False,
        }
        repair_attempt_list = (repair_meta or {}).get("attempts", []) if isinstance(repair_meta, dict) else []
        record["builder_gate_repair_attempts"] = int(len(repair_attempt_list))
        if repaired_g_ir is not None:
            try:
                compiled_g_repaired = compile_preference_builder(
                    repaired_g_ir,
                    operator_whitelist=operator_whitelist,
                )
                pref_batch_repaired = _build_pref_batch_with_memory_trace(
                    compiled_g_repaired,
                    feature_cache,
                    {"stage": "cheap_gate"},
                )
                builder_gate_repaired = run_preference_builder_gates(
                    pref_batch_repaired,
                    feature_cache=feature_cache,
                    min_pairs=int(builder_gate_cfg["min_pairs"]),
                    min_coverage=float(builder_gate_cfg["min_coverage"]),
                    max_pairs_per_instance=int(builder_gate_cfg["max_pairs_per_instance"]),
                    weight_nonneg=bool(builder_gate_cfg["weight_nonneg"]),
                    semantic_tolerance=float(builder_gate_cfg["semantic_tolerance"]),
                    semantic_min_pass_rate=float(builder_gate_cfg["semantic_min_pass_rate"]),
                )
                builder_gate_repaired.trace = _enrich_builder_gate_trace_with_memory(
                    builder_gate_repaired.trace,
                    pref_batch_repaired,
                    cache_hit=False,
                )
                repair_record["g_id_after"] = _sig_pref_builder(repaired_g_ir)
                repair_record["builder_gate_reason_after"] = str(builder_gate_repaired.reason)
                repair_record["builder_gate_trace_after"] = builder_gate_repaired.trace
                repair_record["final_ok"] = bool(builder_gate_repaired.ok)
                if builder_gate_repaired.ok:
                    compiled_g = compiled_g_repaired
                    g_ir = repaired_g_ir
                    pref_batch = pref_batch_repaired
                    builder_gate = builder_gate_repaired
                    record["g_ir"] = asdict(g_ir)
                    record["builder_gate_repaired"] = True
                    record["g_id_before_repair"] = str(record["g_id"])
                    record["g_id_after_repair"] = str(repair_record["g_id_after"])
            except Exception as exc:  # noqa: BLE001
                repair_record["repair_error"] = str(exc)
                repair_record["repair_error_type"] = type(exc).__name__
        builder_gate_repair_reports.append(repair_record)

    joint_min_pass_rate = float(cfg.get("joint_min_pass_rate", 0.8) or 0.8)
    joint_swap_tolerance = float(cfg.get("joint_swap_tolerance", 1e-3) or 1e-3)
    joint_swap_check_mode = str(cfg.get("joint_swap_check_mode", "data") or "data")
    joint_swap_test_margin = float(cfg.get("joint_swap_test_margin", 1.0) or 1.0)
    joint_grad_eps = float(cfg.get("joint_grad_eps", 1e-8) or 1e-8)
    joint_min_effective_grad_ratio = float(
        cfg.get("joint_min_effective_grad_ratio", 0.1) or 0.1
    )
    joint_numeric_stress_enabled = bool(cfg.get("joint_numeric_stress_enabled", False))
    try:
        joint_numeric_stress_margin = abs(float(cfg.get("joint_numeric_stress_margin", 120.0) or 120.0))
    except (TypeError, ValueError):
        joint_numeric_stress_margin = 120.0
    if joint_numeric_stress_margin < 1e-6:
        joint_numeric_stress_margin = 120.0
    try:
        joint_numeric_stress_aux_scale = abs(float(cfg.get("joint_numeric_stress_aux_scale", 32.0) or 32.0))
    except (TypeError, ValueError):
        joint_numeric_stress_aux_scale = 32.0
    if joint_numeric_stress_aux_scale < 1.0:
        joint_numeric_stress_aux_scale = 1.0
    joint_gate = run_joint_preference_gates(
        compiled_f,
        pref_batch=pref_batch,
        feature_cache=feature_cache,
        min_pass_rate=joint_min_pass_rate,
        swap_tolerance=joint_swap_tolerance,
        swap_check_mode=joint_swap_check_mode,
        swap_test_margin=joint_swap_test_margin,
        grad_eps=joint_grad_eps,
        min_effective_grad_ratio=joint_min_effective_grad_ratio,
        numeric_stress_enabled=joint_numeric_stress_enabled,
        numeric_stress_margin=joint_numeric_stress_margin,
        numeric_stress_aux_scale=joint_numeric_stress_aux_scale,
        variant="visible",
    )
    sandbox_gate_enabled = bool(cfg.get("stage0_sandbox_gate_enabled", False))
    sandbox_gate_only_when_hf = bool(cfg.get("stage0_sandbox_gate_only_when_hf", True))
    sandbox_gate_hard_block_hf = bool(cfg.get("stage0_sandbox_gate_hard_block_hf", True))
    sandbox_should_run = bool(sandbox_gate_enabled and (high_fidelity_on or (not sandbox_gate_only_when_hf)))
    sandbox_gate_result: Dict[str, Any] | None = None

    def _joint_gate_from_sandbox_failure(sandbox_res: Mapping[str, Any]) -> JointPreferenceGateResult:
        failure_kind = str(sandbox_res.get("failure_kind") or "sandbox_gate_failed")
        reason = str(sandbox_res.get("reason") or failure_kind)
        trace = dict(sandbox_res.get("trace") or {}) if isinstance(sandbox_res.get("trace"), dict) else {}
        trace.setdefault("failed_gate", "Stage0Sandbox")
        trace["failure_kind"] = str(failure_kind)
        trace["sandbox"] = {
            "reason": str(reason),
            "result": str(sandbox_res.get("sandbox_result", "")),
            "log": str(sandbox_res.get("sandbox_log", "")),
            "exit_code": sandbox_res.get("exit_code"),
        }
        return JointPreferenceGateResult(
            ok=False,
            reason=f"sandbox_gate_failed: {reason}",
            trace=trace,
        )

    if sandbox_should_run:
        sandbox_gate_result = _run_stage0_sandbox_gate(
            run_dir=str(run_dir_s),
            generation=int(generation),
            pair_index=int(pair_index),
            g_id=str(record.get("g_id", "")),
            f_id=str(record.get("f_id", "")),
            g_ir=g_ir,
            f_ir=f_ir,
            operator_whitelist=list(operator_whitelist),
            cfg_yaml=cfg,
        )
        if not bool(sandbox_gate_result.get("ok")):
            joint_gate = _joint_gate_from_sandbox_failure(sandbox_gate_result)

    joint_gate_repair_reports: List[Dict[str, Any]] = []
    repair_enabled = bool(cfg.get("joint_gate_repair_enabled", False))
    try:
        repair_max_attempts = max(int(cfg.get("joint_gate_repair_max_attempts", 2)), 0)
    except (TypeError, ValueError):
        repair_max_attempts = 2
    repair_prompt_path = _abs_from_repo_root(
        str(
            cfg.get(
                "joint_gate_repair_prompt_path",
                "PTP/prompts/free_loss_forward_error_repair.txt",
            )
            or "PTP/prompts/free_loss_forward_error_repair.txt"
        )
    )
    general_repair_prompt_path = _abs_from_repo_root(
        str(
            cfg.get(
                "loss_gate_repair_prompt_path",
                llm_prompts_cfg.get("loss_repair", "PTP/prompts/free_loss_repair.txt"),
            )
            or "PTP/prompts/free_loss_repair.txt"
        )
    )
    raw_repair_failure_kinds = cfg.get(
        "joint_gate_repair_only_failure_kinds",
        [
            "forward_error",
            "backward_error",
            "pref_batch_to_loss_batch_error",
            "loss_not_finite",
            "grad_not_finite",
            "numeric_stress_forward_error",
            "numeric_stress_backward_error",
            "numeric_stress_loss_not_finite",
            "numeric_stress_grad_not_finite",
            "numeric_stress_missing_grads",
            "sandbox_builder_gate_failed",
            "sandbox_runtime_error",
            "sandbox_timeout",
            "sandbox_no_result",
            "sandbox_result_invalid",
            "sandbox_gate_failed",
        ],
    )
    if not isinstance(raw_repair_failure_kinds, (list, tuple, set)):
        raw_repair_failure_kinds = [
            "forward_error",
            "backward_error",
            "pref_batch_to_loss_batch_error",
            "loss_not_finite",
            "grad_not_finite",
            "numeric_stress_forward_error",
            "numeric_stress_backward_error",
            "numeric_stress_loss_not_finite",
            "numeric_stress_grad_not_finite",
            "numeric_stress_missing_grads",
            "sandbox_builder_gate_failed",
            "sandbox_runtime_error",
            "sandbox_timeout",
            "sandbox_no_result",
            "sandbox_result_invalid",
            "sandbox_gate_failed",
        ]
    repair_only_failure_kinds = {str(x) for x in raw_repair_failure_kinds if str(x).strip()}
    expects_repair_prompt_path = _abs_from_repo_root("PTP/prompts/free_loss_expects_repair.txt")
    runtime_repair_failure_kinds = {
        "forward_error",
        "backward_error",
        "pref_batch_to_loss_batch_error",
        "loss_not_finite",
        "grad_not_finite",
        "missing_grads",
        "numeric_stress_forward_error",
        "numeric_stress_backward_error",
        "numeric_stress_loss_not_finite",
        "numeric_stress_grad_not_finite",
        "numeric_stress_missing_grads",
        "sandbox_builder_gate_failed",
        "sandbox_runtime_error",
        "sandbox_timeout",
        "sandbox_no_result",
        "sandbox_result_invalid",
        "sandbox_gate_failed",
        "compile_error",
    }

    def _co_failure_trace_from_record(rec: Mapping[str, Any]) -> Dict[str, Any] | None:
        for key in (
            "co_sensitivity_visible_trace",
            "co_sensitivity_hidden_trace",
            "co_invariance_visible_trace",
            "co_invariance_hidden_trace",
        ):
            trace = rec.get(key)
            if isinstance(trace, dict) and trace:
                return dict(trace)
        failure_kind = rec.get("co_failure_kind") or rec.get("co_reason")
        if failure_kind is None:
            return None
        return {
            "failed_gate": "COAlignment",
            "failure_kind": str(failure_kind),
            "message": str(rec.get("co_reason") or failure_kind),
        }

    def _select_loss_gate_repair_prompt_path(
        *,
        failure_stage: str,
        failure_trace: Mapping[str, Any] | None,
        pair_reason: str | None = None,
    ) -> str:
        failure_kind = (
            str(failure_trace.get("failure_kind"))
            if isinstance(failure_trace, Mapping) and failure_trace.get("failure_kind") is not None
            else ""
        )
        if failure_kind in runtime_repair_failure_kinds:
            return str(repair_prompt_path)
        if str(pair_reason or "") in {"cheap_gate_failed", "f_compile_failed"}:
            return str(repair_prompt_path)
        if str(failure_stage) in {"joint_gate", "static_gate", "compile"}:
            return str(repair_prompt_path)
        return str(general_repair_prompt_path)

    def _run_loss_gate_validation(
        compiled_f_candidate: CompiledFreeLoss,
        repaired_ir: FreeLossIR,
    ) -> Dict[str, Any]:
        joint_gate_candidate = run_joint_preference_gates(
            compiled_f_candidate,
            pref_batch=pref_batch,
            feature_cache=feature_cache,
            min_pass_rate=joint_min_pass_rate,
            swap_tolerance=joint_swap_tolerance,
            swap_check_mode=joint_swap_check_mode,
            swap_test_margin=joint_swap_test_margin,
            grad_eps=joint_grad_eps,
            min_effective_grad_ratio=joint_min_effective_grad_ratio,
            numeric_stress_enabled=joint_numeric_stress_enabled,
            numeric_stress_margin=joint_numeric_stress_margin,
            numeric_stress_aux_scale=joint_numeric_stress_aux_scale,
            variant="visible",
        )
        sandbox_candidate = sandbox_gate_result
        if sandbox_should_run:
            sandbox_candidate = _run_stage0_sandbox_gate(
                run_dir=str(run_dir_s),
                generation=int(generation),
                pair_index=int(pair_index),
                g_id=str(record.get("g_id", "")),
                f_id=str(record.get("f_id", "")),
                g_ir=g_ir,
                f_ir=repaired_ir,
                operator_whitelist=list(operator_whitelist),
                cfg_yaml=cfg,
            )
        if high_fidelity_on and sandbox_should_run and sandbox_gate_hard_block_hf and (not bool((sandbox_candidate or {}).get("ok", False))):
            sandbox_trace = dict((sandbox_candidate or {}).get("trace") or {}) if isinstance((sandbox_candidate or {}).get("trace"), dict) else {}
            sandbox_trace.setdefault("failed_gate", "Stage0Sandbox")
            sandbox_trace["failure_kind"] = str((sandbox_candidate or {}).get("failure_kind") or "sandbox_gate_failed")
            return {
                "ok": False,
                "failure_stage": "stage0_sandbox",
                "pair_reason": "stage0_sandbox_failed",
                "failure_reason": str((sandbox_candidate or {}).get("reason") or "sandbox_gate_failed"),
                "failure_trace": sandbox_trace,
                "joint_gate": joint_gate_candidate,
                "sandbox_gate_result": sandbox_candidate,
                "pref_sem": None,
                "co_updates": None,
            }
        if cheap_gate_on and (not builder_gate.ok or not joint_gate_candidate.ok):
            return {
                "ok": False,
                "failure_stage": "joint_gate",
                "pair_reason": "cheap_gate_failed",
                "failure_reason": str(joint_gate_candidate.reason),
                "failure_trace": (
                    dict(joint_gate_candidate.trace)
                    if isinstance(joint_gate_candidate.trace, dict)
                    else None
                ),
                "joint_gate": joint_gate_candidate,
                "sandbox_gate_result": sandbox_candidate,
                "pref_sem": None,
                "co_updates": None,
            }
        pref_sem_candidate = None
        if bool(cfg.get("pref_semantic_gate_enabled", False)):
            pref_sem_candidate = run_preference_semantic_gates(
                compiled_f_candidate,
                trials=int(cfg.get("pref_semantic_trials", 6) or 6),
                batch_size=int(cfg.get("pref_semantic_batch_size", 128) or 128),
                min_pass_rate=float(cfg.get("pref_semantic_min_pass_rate", 0.8) or 0.8),
                swap_tolerance=float(cfg.get("pref_semantic_swap_tolerance", 1e-3) or 1e-3),
                gap_min_ratio=float(cfg.get("pref_semantic_gap_min_ratio", 0.9) or 0.9),
                variant="visible",
            )
            if cheap_gate_on and (not bool(pref_sem_candidate.ok)):
                return {
                    "ok": False,
                    "failure_stage": "pref_semantic",
                    "pair_reason": "pref_semantic_failed",
                    "failure_reason": str(pref_sem_candidate.reason),
                    "failure_trace": (
                        dict(pref_sem_candidate.trace)
                        if isinstance(pref_sem_candidate.trace, dict)
                        else None
                    ),
                    "joint_gate": joint_gate_candidate,
                    "sandbox_gate_result": sandbox_candidate,
                    "pref_sem": pref_sem_candidate,
                    "co_updates": None,
                }
        co_updates_candidate = _run_co_alignment_gates_for_loss(compiled_f_candidate, cfg)
        if cheap_gate_on and (not bool(co_updates_candidate.get("co_ok", True))):
            co_trace = _co_failure_trace_from_record(co_updates_candidate)
            return {
                "ok": False,
                "failure_stage": "co_gate",
                "pair_reason": "co_gate_failed",
                "failure_reason": str(co_updates_candidate.get("co_reason") or "co_gate_failed"),
                "failure_trace": co_trace,
                "joint_gate": joint_gate_candidate,
                "sandbox_gate_result": sandbox_candidate,
                "pref_sem": pref_sem_candidate,
                "co_updates": co_updates_candidate,
            }
        return {
            "ok": True,
            "failure_stage": None,
            "pair_reason": "ok_gate_only" if not high_fidelity_on else "ok_after_repair",
            "failure_reason": None,
            "failure_trace": None,
            "joint_gate": joint_gate_candidate,
            "sandbox_gate_result": sandbox_candidate,
            "pref_sem": pref_sem_candidate,
            "co_updates": co_updates_candidate,
        }

    def _attempt_loss_gate_repair(
        *,
        failure_stage: str,
        failure_reason: str,
        failure_trace: Mapping[str, Any] | None,
        pair_reason: str,
    ) -> Dict[str, Any] | None:
        if not repair_enabled or repair_max_attempts <= 0 or str(record.get("f_id")) == F_REF_ID:
            return None
        try:
            _configure_loss_llm_for_worker(cfg=cfg, run_dir=run_dir_s)
        except Exception:  # noqa: BLE001
            pass

        current_ir = f_ir
        current_failure_reason = str(failure_reason)
        current_failure_trace = dict(failure_trace) if isinstance(failure_trace, Mapping) else None
        for attempt_idx in range(repair_max_attempts):
            attempt_no = int(attempt_idx + 1)
            selected_prompt_path = _select_loss_gate_repair_prompt_path(
                failure_stage=str(failure_stage),
                failure_trace=current_failure_trace,
                pair_reason=str(pair_reason),
            )
            attempt_record: Dict[str, Any] = {
                "record_type": "joint_gate_repair_attempt",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation": int(generation),
                "pair_index": int(pair_index),
                "stage": str(failure_stage),
                "g_id": str(g_entry["id"]),
                "f_id_before": str(record["f_id"]),
                "attempt": attempt_no,
                "failure_kind_before": (
                    str(current_failure_trace.get("failure_kind"))
                    if isinstance(current_failure_trace, dict)
                    and current_failure_trace.get("failure_kind") is not None
                    else None
                ),
                "joint_gate_trace_before": current_failure_trace,
                "builder_gate_trace": builder_gate.trace,
                "code_before": _free_loss_code_digest(current_ir),
                "repair_prompt_path": str(selected_prompt_path),
                "static_ok": False,
                "compile_ok": False,
                "final_ok": False,
            }
            failure_payload = {
                "stage": str(failure_stage),
                "pair_reason": str(pair_reason),
                "pair": {"g_id": str(g_entry["id"]), "f_id": str(record["f_id"])},
                "failure_reason": str(current_failure_reason),
                "failure_trace": current_failure_trace,
                "builder_gate_trace": builder_gate.trace,
                "operator_whitelist": list(operator_whitelist),
            }
            try:
                repaired_ir = loss_llm_ops.repair_free_loss(
                    selected_prompt_path,
                    failed_ir=current_ir,
                    failure_reason=failure_payload,
                    prompt_context=loss_prompt_context,
                )
                attempt_record["repair_call_ok"] = True
            except Exception as exc:  # noqa: BLE001
                attempt_record.update(
                    {
                        "repair_call_ok": False,
                        "repair_error": str(exc),
                        "repair_error_type": type(exc).__name__,
                    }
                )
                joint_gate_repair_reports.append(attempt_record)
                continue
            try:
                repaired_ir = loss_llm_ops.repair_expects_with_prompt(
                    expects_repair_prompt_path,
                    repaired_ir,
                    prompt_context=loss_prompt_context,
                )
                attempt_record["expects_repair_ok"] = True
            except Exception as exc:  # noqa: BLE001
                attempt_record["expects_repair_ok"] = False
                attempt_record["expects_repair_error"] = str(exc)
                attempt_record["expects_repair_error_type"] = type(exc).__name__
            attempt_record["f_id_after"] = _sig_free_loss(repaired_ir)
            attempt_record["code_after"] = _free_loss_code_digest(repaired_ir)
            static_res = run_static_gates(repaired_ir, operator_whitelist=operator_whitelist)
            attempt_record["static_ok"] = bool(static_res.ok)
            attempt_record["static_reason"] = str(static_res.reason)
            attempt_record["static_trace"] = static_res.trace
            if not static_res.ok:
                current_ir = repaired_ir
                current_failure_reason = str(static_res.reason)
                current_failure_trace = dict(static_res.trace) if isinstance(static_res.trace, dict) else None
                attempt_record["joint_gate_trace_after"] = current_failure_trace
                joint_gate_repair_reports.append(attempt_record)
                continue
            try:
                compiled_f_repaired = compile_free_loss(
                    repaired_ir,
                    operator_whitelist=operator_whitelist,
                )
                attempt_record["compile_ok"] = True
            except Exception as exc:  # noqa: BLE001
                current_ir = repaired_ir
                current_failure_reason = f"compile_error: {exc}"
                current_failure_trace = {
                    "failed_gate": "Compile",
                    "failure_kind": "compile_error",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                }
                attempt_record.update(
                    {
                        "compile_ok": False,
                        "compile_error": str(exc),
                        "compile_error_type": type(exc).__name__,
                        "joint_gate_trace_after": current_failure_trace,
                    }
                )
                joint_gate_repair_reports.append(attempt_record)
                continue
            validation = _run_loss_gate_validation(compiled_f_repaired, repaired_ir)
            attempt_record["joint_gate_trace_after"] = validation.get("failure_trace")
            attempt_record["final_ok"] = bool(validation.get("ok"))
            if isinstance(validation.get("co_updates"), Mapping):
                attempt_record["co_failure_kind_after"] = validation["co_updates"].get("co_failure_kind")
                attempt_record["co_reason_after"] = validation["co_updates"].get("co_reason")
            joint_gate_repair_reports.append(attempt_record)
            current_ir = repaired_ir
            current_failure_reason = str(validation.get("failure_reason") or "")
            current_failure_trace = (
                dict(validation["failure_trace"])
                if isinstance(validation.get("failure_trace"), Mapping)
                else None
            )
            if bool(validation.get("ok")):
                return {
                    "compiled_f": compiled_f_repaired,
                    "f_ir": repaired_ir,
                    "validation": validation,
                    "attempt_no": attempt_no,
                }
        return None

    joint_failure_kind = (
        str(joint_gate.trace.get("failure_kind"))
        if isinstance(joint_gate.trace, dict) and joint_gate.trace.get("failure_kind") is not None
        else None
    )
    if (
        repair_enabled
        and (not joint_gate.ok)
        and repair_max_attempts > 0
    ):
        try:
            _configure_loss_llm_for_worker(cfg=cfg, run_dir=run_dir_s)
        except Exception:  # noqa: BLE001
            pass
        current_ir = f_ir
        current_failure_reason = str(joint_gate.reason)
        current_failure_trace = (
            dict(joint_gate.trace) if isinstance(joint_gate.trace, dict) else None
        )
        repaired_ok = False
        for attempt_idx in range(repair_max_attempts):
            attempt_no = int(attempt_idx + 1)
            attempt_record: Dict[str, Any] = {
                "record_type": "joint_gate_repair_attempt",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation": int(generation),
                "pair_index": int(pair_index),
                "stage": "joint_gate",
                "g_id": str(g_entry["id"]),
                "f_id_before": str(record["f_id"]),
                "attempt": attempt_no,
                "failure_kind_before": (
                    str(current_failure_trace.get("failure_kind"))
                    if isinstance(current_failure_trace, dict)
                    and current_failure_trace.get("failure_kind") is not None
                    else None
                ),
                "joint_gate_trace_before": current_failure_trace,
                "builder_gate_trace": builder_gate.trace,
                "code_before": _free_loss_code_digest(current_ir),
                "static_ok": False,
                "compile_ok": False,
                "final_ok": False,
            }
            failure_reason = {
                "stage": "joint_gate",
                "pair": {"g_id": str(g_entry["id"]), "f_id": str(record["f_id"])},
                "joint_gate_reason": current_failure_reason,
                "joint_gate_trace": current_failure_trace,
                "builder_gate_trace": builder_gate.trace,
                "operator_whitelist": list(operator_whitelist),
            }
            selected_repair_prompt_path = _select_loss_gate_repair_prompt_path(
                failure_stage="joint_gate",
                failure_trace=current_failure_trace,
                pair_reason="cheap_gate_failed",
            )
            try:
                repaired_ir = loss_llm_ops.repair_free_loss(
                    selected_repair_prompt_path,
                    failed_ir=current_ir,
                    failure_reason=failure_reason,
                    prompt_context=loss_prompt_context,
                )
                attempt_record["repair_call_ok"] = True
                attempt_record["repair_prompt_path"] = str(selected_repair_prompt_path)
            except Exception as exc:  # noqa: BLE001
                attempt_record.update(
                    {
                        "repair_call_ok": False,
                        "repair_error": str(exc),
                        "repair_error_type": type(exc).__name__,
                    }
                )
                joint_gate_repair_reports.append(attempt_record)
                continue

            try:
                repaired_ir = loss_llm_ops.repair_expects_with_prompt(
                    expects_repair_prompt_path,
                    repaired_ir,
                    prompt_context=loss_prompt_context,
                )
                attempt_record["expects_repair_ok"] = True
            except Exception as exc:  # noqa: BLE001
                attempt_record["expects_repair_ok"] = False
                attempt_record["expects_repair_error"] = str(exc)
                attempt_record["expects_repair_error_type"] = type(exc).__name__

            attempt_record["f_id_after"] = _sig_free_loss(repaired_ir)
            attempt_record["code_after"] = _free_loss_code_digest(repaired_ir)

            static_res = run_static_gates(repaired_ir, operator_whitelist=operator_whitelist)
            attempt_record["static_ok"] = bool(static_res.ok)
            attempt_record["static_reason"] = str(static_res.reason)
            attempt_record["static_trace"] = static_res.trace
            if not static_res.ok:
                current_ir = repaired_ir
                current_failure_reason = str(static_res.reason)
                current_failure_trace = (
                    dict(static_res.trace) if isinstance(static_res.trace, dict) else None
                )
                attempt_record["joint_gate_trace_after"] = current_failure_trace
                joint_gate_repair_reports.append(attempt_record)
                continue

            try:
                compiled_f_repaired = compile_free_loss(
                    repaired_ir,
                    operator_whitelist=operator_whitelist,
                )
                attempt_record["compile_ok"] = True
            except Exception as exc:  # noqa: BLE001
                current_ir = repaired_ir
                current_failure_reason = f"compile_error: {exc}"
                current_failure_trace = {
                    "failed_gate": "Compile",
                    "failure_kind": "compile_error",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                }
                attempt_record.update(
                    {
                        "compile_ok": False,
                        "compile_error": str(exc),
                        "compile_error_type": type(exc).__name__,
                        "joint_gate_trace_after": current_failure_trace,
                    }
                )
                joint_gate_repair_reports.append(attempt_record)
                continue

            joint_gate_repaired = run_joint_preference_gates(
                compiled_f_repaired,
                pref_batch=pref_batch,
                feature_cache=feature_cache,
                min_pass_rate=joint_min_pass_rate,
                swap_tolerance=joint_swap_tolerance,
                swap_check_mode=joint_swap_check_mode,
                swap_test_margin=joint_swap_test_margin,
                grad_eps=joint_grad_eps,
                min_effective_grad_ratio=joint_min_effective_grad_ratio,
                numeric_stress_enabled=joint_numeric_stress_enabled,
                numeric_stress_margin=joint_numeric_stress_margin,
                numeric_stress_aux_scale=joint_numeric_stress_aux_scale,
                variant="visible",
            )
            attempt_record["joint_gate_trace_after"] = joint_gate_repaired.trace
            attempt_record["final_ok"] = bool(joint_gate_repaired.ok)
            joint_gate_repair_reports.append(attempt_record)

            current_ir = repaired_ir
            current_failure_reason = str(joint_gate_repaired.reason)
            current_failure_trace = (
                dict(joint_gate_repaired.trace)
                if isinstance(joint_gate_repaired.trace, dict)
                else None
            )
            if joint_gate_repaired.ok:
                compiled_f = compiled_f_repaired
                f_ir = repaired_ir
                record["f_ir"] = asdict(f_ir)
                joint_gate = joint_gate_repaired
                record["joint_gate_repaired"] = True
                record["joint_gate_repair_attempts"] = attempt_no
                record["f_id_before_repair"] = str(record["f_id"])
                record["f_id_after_repair"] = str(attempt_record["f_id_after"])
                repaired_ok = True
                break
        if not repaired_ok:
            record["joint_gate_repair_attempts"] = int(len(joint_gate_repair_reports))

    if sandbox_should_run:
        need_rerun_sandbox = bool(record.get("joint_gate_repaired", False)) or (not bool((sandbox_gate_result or {}).get("ok", False)))
        if need_rerun_sandbox:
            sandbox_gate_result = _run_stage0_sandbox_gate(
                run_dir=str(run_dir_s),
                generation=int(generation),
                pair_index=int(pair_index),
                g_id=str(record.get("g_id", "")),
                f_id=str(record.get("f_id", "")),
                g_ir=g_ir,
                f_ir=f_ir,
                operator_whitelist=list(operator_whitelist),
                cfg_yaml=cfg,
            )
        if sandbox_gate_result is not None and (not bool(sandbox_gate_result.get("ok"))):
            joint_gate = _joint_gate_from_sandbox_failure(sandbox_gate_result)

    record["sandbox_gate"] = dict(sandbox_gate_result) if isinstance(sandbox_gate_result, dict) else None
    record["sandbox_gate_ok"] = None if sandbox_gate_result is None else bool(sandbox_gate_result.get("ok"))
    record["sandbox_gate_reason"] = None if sandbox_gate_result is None else str(sandbox_gate_result.get("reason", ""))

    record["joint_gate_repair_reports"] = list(joint_gate_repair_reports)
    record["builder_gate_repair_reports"] = list(builder_gate_repair_reports)
    record.update(
        {
            "builder_gate_ok": bool(builder_gate.ok),
            "builder_gate_reason": str(builder_gate.reason),
            "builder_gate_trace": builder_gate.trace,
            "joint_gate_ok": bool(joint_gate.ok),
            "joint_gate_reason": str(joint_gate.reason),
            "joint_gate_trace": joint_gate.trace,
        }
    )

    if high_fidelity_on and sandbox_should_run and sandbox_gate_hard_block_hf and (not bool(record.get("sandbox_gate_ok"))):
        repair_out = _attempt_loss_gate_repair(
            failure_stage="stage0_sandbox",
            failure_reason=str(record.get("sandbox_gate_reason") or "sandbox_gate_failed"),
            failure_trace=(record.get("sandbox_gate") if isinstance(record.get("sandbox_gate"), dict) else None),
            pair_reason="stage0_sandbox_failed",
        )
        if repair_out is not None:
            compiled_f = repair_out["compiled_f"]
            f_ir = repair_out["f_ir"]
            record["f_ir"] = asdict(f_ir)
            joint_gate = repair_out["validation"]["joint_gate"]
            sandbox_gate_result = repair_out["validation"]["sandbox_gate_result"]
            record["joint_gate_repaired"] = True
            record["joint_gate_repair_attempts"] = int(repair_out["attempt_no"])
            record["f_id_before_repair"] = str(record["f_id"])
            record["f_id_after_repair"] = str(_sig_free_loss(f_ir))
            record["sandbox_gate"] = dict(sandbox_gate_result) if isinstance(sandbox_gate_result, dict) else None
            record["sandbox_gate_ok"] = None if sandbox_gate_result is None else bool(sandbox_gate_result.get("ok"))
            record["sandbox_gate_reason"] = None if sandbox_gate_result is None else str(sandbox_gate_result.get("reason", ""))
        else:
            record["pair_ok"] = False
            record["pair_reason"] = "stage0_sandbox_failed"
            record["score"] = float("inf")
            record["elapsed_s"] = float(time.time() - t0)
            return record

    if cheap_gate_on and (not builder_gate.ok or not joint_gate.ok):
        record["pair_ok"] = False
        record["pair_reason"] = "cheap_gate_failed"
        record["score"] = float("inf")
        record["elapsed_s"] = float(time.time() - t0)
        return record

    if bool(cfg.get("pref_semantic_gate_enabled", False)):
        pref_sem = run_preference_semantic_gates(
            compiled_f,
            trials=int(cfg.get("pref_semantic_trials", 6) or 6),
            batch_size=int(cfg.get("pref_semantic_batch_size", 128) or 128),
            min_pass_rate=float(cfg.get("pref_semantic_min_pass_rate", 0.8) or 0.8),
            swap_tolerance=float(cfg.get("pref_semantic_swap_tolerance", 1e-3) or 1e-3),
            gap_min_ratio=float(cfg.get("pref_semantic_gap_min_ratio", 0.9) or 0.9),
            variant="visible",
        )
        record.update(
            {
                "pref_semantic_ok": bool(pref_sem.ok),
                "pref_semantic_reason": str(pref_sem.reason),
                "pref_semantic_trace": pref_sem.trace,
            }
        )
        if cheap_gate_on and not pref_sem.ok:
            repair_out = _attempt_loss_gate_repair(
                failure_stage="pref_semantic",
                failure_reason=str(pref_sem.reason),
                failure_trace=(pref_sem.trace if isinstance(pref_sem.trace, dict) else None),
                pair_reason="pref_semantic_failed",
            )
            if repair_out is not None:
                compiled_f = repair_out["compiled_f"]
                f_ir = repair_out["f_ir"]
                record["f_ir"] = asdict(f_ir)
                joint_gate = repair_out["validation"]["joint_gate"]
                sandbox_gate_result = repair_out["validation"]["sandbox_gate_result"]
                pref_sem = repair_out["validation"]["pref_sem"]
                record["joint_gate_repaired"] = True
                record["joint_gate_repair_attempts"] = int(repair_out["attempt_no"])
                record["f_id_before_repair"] = str(record["f_id"])
                record["f_id_after_repair"] = str(_sig_free_loss(f_ir))
                record["pref_semantic_ok"] = bool(pref_sem.ok) if pref_sem is not None else None
                record["pref_semantic_reason"] = str(pref_sem.reason) if pref_sem is not None else None
                record["pref_semantic_trace"] = pref_sem.trace if pref_sem is not None else None
            else:
                record["pair_ok"] = False
                record["pair_reason"] = "pref_semantic_failed"
                record["score"] = float("inf")
                record["elapsed_s"] = float(time.time() - t0)
                return record

    record.update(_run_co_alignment_gates_for_loss(compiled_f, cfg))
    if cheap_gate_on and (not bool(record.get("co_ok", True))):
        repair_out = _attempt_loss_gate_repair(
            failure_stage="co_gate",
            failure_reason=str(record.get("co_reason") or "co_gate_failed"),
            failure_trace=_co_failure_trace_from_record(record),
            pair_reason="co_gate_failed",
        )
        if repair_out is not None:
            compiled_f = repair_out["compiled_f"]
            f_ir = repair_out["f_ir"]
            record["f_ir"] = asdict(f_ir)
            joint_gate = repair_out["validation"]["joint_gate"]
            sandbox_gate_result = repair_out["validation"]["sandbox_gate_result"]
            record["joint_gate_repaired"] = True
            record["joint_gate_repair_attempts"] = int(repair_out["attempt_no"])
            record["f_id_before_repair"] = str(record["f_id"])
            record["f_id_after_repair"] = str(_sig_free_loss(f_ir))
            record.update(_run_co_alignment_gates_for_loss(compiled_f, cfg))
        else:
            record["pair_ok"] = False
            record["pair_reason"] = "co_gate_failed"
            record["score"] = float("inf")
            record["elapsed_s"] = float(time.time() - t0)
            return record

    if not high_fidelity_on:
        eff = float(joint_gate.effective_grad_ratio or 0.0)
        sem = float(builder_gate.semantic_pass_rate or 0.0)
        cheap_score = float(2.0 - eff - sem)
        record["pair_ok"] = True
        record["pair_reason"] = "ok_gate_only"
        record["score"] = float(cheap_score)
        record["proxy_metrics"] = record.get("proxy_metrics") or {
            "cheap_effective_grad_ratio": eff,
            "cheap_semantic_pass_rate": sem,
        }
        record["elapsed_s"] = float(time.time() - t0)
        return record

    # Stage3: discovery-style offline mini-train fitness, compared against one or more
    # precomputed baseline JSONs. Multi-scenario configs aggregate deltas across all
    # configured scenarios and init checkpoints.
    scenario_entries = _iter_stage3_scenario_cfgs(cfg)
    try:
        # Route stage3 mini-train logs to a per-pair file (like free_loss_discovery).
        file_handler: logging.Handler | None = None
        fl_logger = logging.getLogger("fitness.free_loss_fidelity")
        if run_dir_s:
            safe_gid = str(record.get("g_id", "g")).replace(os.sep, "_").replace(":", "_")[:24]
            safe_fid = str(record.get("f_id", "f")).replace(os.sep, "_").replace(":", "_")[:24]
            safe_dev = str(device_str).replace(os.sep, "_").replace(":", "_")
            log_path = os.path.join(
                run_dir_s,
                f"gen{generation:03d}_pair{pair_index:03d}_{safe_dev}_{safe_gid}_{safe_fid}.log",
            )
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")

            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
            root_logger.setLevel(logging.INFO)

            for handler in list(fl_logger.handlers):
                try:
                    fl_logger.removeHandler(handler)
                finally:
                    try:
                        handler.close()
                    except Exception:  # noqa: BLE001
                        pass
            fl_logger.setLevel(logging.INFO)

            try:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
                file_handler.setFormatter(fmt)
                # Attach the handler to the root logger only. Since
                # fitness.free_loss_fidelity propagates to root by default, attaching
                # to both root and the module logger would duplicate every line.
                fl_logger.propagate = True
                root_logger.addHandler(file_handler)
                record["hf_log_file"] = os.path.basename(log_path)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[pref_loss_coevo][worker] failed to open stage3 log file: {log_path}: {exc}",
                    flush=True,
                )
                file_handler = None

        adapter = _CompiledBuilderAdapter(compiled_g)
        per_scenario: Dict[str, Any] = {}
        all_deltas: List[float] = []
        all_per_init: Dict[str, Any] = {}
        any_error = False
        baseline_mini_eval_paths: Dict[str, str] = {}
        baseline_multiseed_cache_paths: Dict[str, str | None] = {}
        baseline_compare_modes: Dict[str, str] = {}
        eval_signatures: Dict[str, Any] = {}
        early_prune_report: Dict[str, Any] | None = None

        fl_logger.info(
            "Stage3 offline mini-train start gen=%d pair_index=%d scenarios=%s",
            int(generation),
            int(pair_index),
            [str(entry.get("name")) for entry in scenario_entries],
        )

        for scenario_entry in scenario_entries:
            scenario_name = str(scenario_entry.get("name") or "scenario")
            scenario_cfg = dict(scenario_entry.get("cfg") or {})
            baseline_cfg = scenario_cfg.get("baseline", {}) or {}
            mini_eval_path = _resolve_stage3_baseline_mini_eval_path(scenario_cfg, baseline_cfg)
            if not mini_eval_path:
                raise ValueError(
                    "baseline mini_eval_path is required for stage3 "
                    f"(scenario={scenario_name}, fidelity={_stage3_fidelity_key(scenario_cfg)})"
                )

            baseline_payload = _load_baseline_mini_eval(str(mini_eval_path))
            expected_sig = _build_stage3_eval_signature(scenario_cfg)
            got_sig = baseline_payload.get("eval_signature")
            if got_sig != expected_sig:
                raise ValueError(
                    f"baseline eval_signature mismatch for scenario={scenario_name} "
                    "(re-run scripts/eval_baseline_minitrain.py)"
                )

            per_init_base = baseline_payload.get("per_init")
            if not isinstance(per_init_base, dict):
                raise ValueError(f"baseline JSON missing per_init dict for scenario={scenario_name}")
            multiseed_baseline = _load_stage3_baseline_multiseed_cache_for_cfg(scenario_cfg)

            init_specs = _stage3_init_specs_from_baseline_cfg(scenario_cfg)
            if not init_specs:
                raise ValueError(f"scenario={scenario_name} has no init sources configured")

            scratch_init_seed = int(_resolve_training_seed(scenario_cfg))
            hf_epochs_cfg = int(scenario_cfg.get("hf_epochs", 0) or 0)
            hf_inst_cfg = int(scenario_cfg.get("hf_instances_per_epoch", 0) or 0)

            # Budget: step-mode uses K=f1_steps; epoch-mode uses hf_epochs/hf_instances_per_epoch.
            K = int(scenario_cfg.get("f1_steps", 32) or 32)
            cfg_hf = dict(scenario_cfg)
            cfg_hf["f1_steps"] = int(K)
            if not (hf_epochs_cfg > 0 and hf_inst_cfg > 0):
                cfg_hf["hf_epochs"] = 0
                cfg_hf["hf_instances_per_epoch"] = 0
            hf_cfg = _build_hf_cfg(cfg_hf, seed=int(scratch_init_seed), device_str=device_str)

            valid_sizes = [int(v) for v in cfg_hf.get("valid_problem_sizes", list(hf_cfg.valid_problem_sizes))]
            if not valid_sizes:
                valid_sizes = [int(hf_cfg.train_problem_size)]
            valid_sizes = list(dict.fromkeys([int(v) for v in valid_sizes]))

            scenario_per_init: Dict[str, Any] = {}
            scenario_deltas: List[float] = []
            scenario_any_error = False

            fl_logger.info(
                "Stage3 scenario start gen=%d pair_index=%d scenario=%s K=%d seed=%d valid_sizes=%s baseline_json=%s",
                int(generation),
                int(pair_index),
                str(scenario_name),
                int(K),
                int(scratch_init_seed),
                list(valid_sizes),
                str(mini_eval_path),
            )

            for init_name, init_ckpt in init_specs:
                base_entry, base_source = _resolve_stage3_baseline_reference_entry(
                    str(init_name),
                    per_init_base=per_init_base,
                    multiseed_cache=multiseed_baseline,
                )
                if not isinstance(base_entry, dict):
                    raise ValueError(f"baseline JSON missing per_init[{init_name}] for scenario={scenario_name}")
                try:
                    base_agg = float(base_entry.get("aggregated_objective"))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"baseline per_init[{init_name}].aggregated_objective is invalid for scenario={scenario_name}"
                    )
                base_by_size_raw = base_entry.get("val_objective_by_size", {})
                base_by_size: Dict[int, float] = {}
                if isinstance(base_by_size_raw, dict):
                    for k, v in base_by_size_raw.items():
                        try:
                            base_by_size[int(k)] = float(v)
                        except Exception:  # noqa: BLE001
                            continue

                cand_by_size: Dict[int, float] = {}
                cand_agg: float
                error: str | None = None
                try:
                    free_cfg = FreeLossFidelityConfig(
                        hf=hf_cfg,
                        f1_steps=int(K),
                        f2_steps=0,
                        f3_enabled=False,
                        init_checkpoint_path=_abs_from_repo_root(str(init_ckpt)) if init_ckpt else None,
                        init_checkpoint_epoch=None,
                        scratch_hf_epochs=int(scenario_cfg.get("scratch_hf_epochs", 0) or 0),
                        warmstart_hf_epochs=int(scenario_cfg.get("warmstart_hf_epochs", 0) or 0),
                        baseline_epoch_compare_offset=int(scenario_cfg.get("baseline_epoch_compare_offset", 0) or 0),
                        baseline_epoch_violation_weight=float(scenario_cfg.get("baseline_epoch_violation_weight", 1.0)),
                        baseline_epoch_tail_frac=float(scenario_cfg.get("baseline_epoch_tail_frac", 1.0) or 1.0),
                        baseline_epoch_window_k=int(scenario_cfg.get("baseline_epoch_window_k", 10) or 10),
                        baseline_epoch_window_violation_weight=float(
                            scenario_cfg.get("baseline_epoch_window_violation_weight", 1.0) or 1.0
                        ),
                    )
                    fitness = evaluate_free_loss_candidate(compiled_f, free_cfg, pref_builder=adapter)
                    size_objectives_raw = fitness.get("size_objectives", {})
                    size_objectives: Dict[int, float] = {}
                    if isinstance(size_objectives_raw, dict):
                        for k, v in size_objectives_raw.items():
                            try:
                                size_objectives[int(k)] = float(v)
                            except Exception:  # noqa: BLE001
                                continue
                    for sz in valid_sizes:
                        if int(sz) not in size_objectives:
                            raise RuntimeError(f"Missing fitness.size_objectives[{int(sz)}]")
                        cand_by_size[int(sz)] = float(size_objectives[int(sz)])
                    cand_agg = float(sum(float(cand_by_size[int(sz)]) for sz in valid_sizes) / float(len(valid_sizes)))
                    if not math.isfinite(cand_agg):
                        raise RuntimeError("Non-finite candidate aggregated objective")
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    try:
                        fl_logger.exception(
                            "Stage3 mini-train FAILED scenario=%s init=%s g_id=%s f_id=%s device=%s",
                            str(scenario_name),
                            str(init_name),
                            str(record.get("g_id")),
                            str(record.get("f_id")),
                            str(device_str),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    cand_by_size = {int(sz): 1.0e9 for sz in valid_sizes}
                    cand_agg = 1.0e9

                delta = float(cand_agg) - float(base_agg)
                scenario_deltas.append(float(delta))
                scenario_any_error = scenario_any_error or bool(error)
                init_record = {
                    "obj_cand_by_size": {str(int(k)): float(v) for k, v in cand_by_size.items()},
                    "obj_base_by_size": {str(int(k)): float(base_by_size.get(int(k), float("nan"))) for k in valid_sizes},
                    "obj_cand": float(cand_agg),
                    "obj_base": float(base_agg),
                    "delta": float(delta),
                    "baseline_source": str(base_source),
                    "baseline_seed": (
                        int(base_entry.get("seed"))
                        if str(base_source) == "multiseed_best" and base_entry.get("seed") is not None
                        else None
                    ),
                    "init_checkpoint": str(init_ckpt) if init_ckpt else None,
                    "error": error,
                }
                scenario_per_init[str(init_name)] = init_record
                flat_init_name = (
                    str(init_name)
                    if len(scenario_entries) == 1
                    else f"{scenario_name}:{str(init_name)}"
                )
                all_per_init[str(flat_init_name)] = dict(init_record)

            scenario_delta_mean = (
                float(sum(scenario_deltas) / float(len(scenario_deltas))) if scenario_deltas else float("inf")
            )
            scenario_delta_worst = float(max(scenario_deltas)) if scenario_deltas else float("inf")

            any_error = any_error or bool(scenario_any_error)
            all_deltas.extend(float(d) for d in scenario_deltas)
            baseline_mini_eval_paths[str(scenario_name)] = str(mini_eval_path)
            baseline_multiseed_cache_paths[str(scenario_name)] = (
                str(multiseed_baseline.get("cache_path"))
                if isinstance(multiseed_baseline, dict) and multiseed_baseline.get("cache_path")
                else None
            )
            baseline_compare_modes[str(scenario_name)] = (
                "multiseed_best"
                if isinstance(multiseed_baseline, dict) and multiseed_baseline.get("best_per_init")
                else "mini_eval"
            )
            eval_signatures[str(scenario_name)] = expected_sig
            per_scenario[str(scenario_name)] = {
                "scenario_name": str(scenario_name),
                "problem": str(scenario_cfg.get("problem") or scenario_cfg.get("env_name") or "tsp"),
                "env_name": str(scenario_cfg.get("env_name") or scenario_cfg.get("problem") or "tsp"),
                "train_problem_size": int(scenario_cfg.get("train_problem_size", 20) or 20),
                "valid_problem_sizes": list(valid_sizes),
                "baseline_mini_eval_path": str(mini_eval_path),
                "baseline_multiseed_cache_path": baseline_multiseed_cache_paths[str(scenario_name)],
                "baseline_compare_mode": baseline_compare_modes[str(scenario_name)],
                "eval_signature": expected_sig,
                "K": int(K),
                "per_init": scenario_per_init,
                "delta_mean": float(scenario_delta_mean),
                "delta_worst": float(scenario_delta_worst),
            }

            if early_prune_report is None:
                early_prune_report = _stage3_check_early_prune(
                    cfg_yaml=cfg,
                    scenario_name=str(scenario_name),
                    scenario_per_init=scenario_per_init,
                )
                if early_prune_report is not None:
                    any_error = True
                    fl_logger.warning(
                        "Stage3 early prune triggered gen=%d pair_index=%d scenario=%s report=%s",
                        int(generation),
                        int(pair_index),
                        str(scenario_name),
                        dict(early_prune_report),
                    )
                    break

        delta_mean = float(sum(all_deltas) / float(len(all_deltas))) if all_deltas else float("inf")
        delta_worst = float(max(all_deltas)) if all_deltas else float("inf")
        scenario_delta_means: List[float] = []
        for scenario_record in per_scenario.values():
            if not isinstance(scenario_record, Mapping):
                continue
            try:
                scenario_delta = float(scenario_record.get("delta_mean"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(scenario_delta):
                scenario_delta_means.append(float(scenario_delta))
        scenario_nonnegative_count = sum(1 for delta in scenario_delta_means if not (float(delta) < 0.0))
        all_scenarios_negative = bool(scenario_delta_means) and bool(scenario_nonnegative_count == 0)
        worst_scenario_delta = float(max(scenario_delta_means)) if scenario_delta_means else float("inf")

        if early_prune_report is not None:
            record["pair_ok"] = False
            record["pair_reason"] = "stage3_early_pruned"
            record["stage3_early_prune"] = dict(early_prune_report)
        else:
            record["pair_ok"] = not any_error
            record["pair_reason"] = "ok_stage3_offline_minitrain" if not any_error else "stage3_runtime_error"
        record["fitness"] = {
            "per_init": all_per_init,
            "per_scenario": per_scenario,
            "scenario_names": list(per_scenario.keys()),
            "delta_mean": float(delta_mean),
            "delta_worst": float(delta_worst),
            "all_scenarios_negative": bool(all_scenarios_negative),
            "nonnegative_scenario_count": int(scenario_nonnegative_count),
            "worst_scenario_delta": float(worst_scenario_delta),
            "baseline_mini_eval_path": (
                next(iter(baseline_mini_eval_paths.values())) if len(baseline_mini_eval_paths) == 1 else None
            ),
            "baseline_mini_eval_paths": baseline_mini_eval_paths,
            "baseline_multiseed_cache_path": (
                next(iter(baseline_multiseed_cache_paths.values()))
                if len(baseline_multiseed_cache_paths) == 1
                else None
            ),
            "baseline_multiseed_cache_paths": baseline_multiseed_cache_paths,
            "baseline_compare_mode": (
                next(iter(baseline_compare_modes.values()))
                if len(set(baseline_compare_modes.values())) == 1 and baseline_compare_modes
                else "mixed"
            ),
            "baseline_compare_modes": baseline_compare_modes,
            "eval_signature": (
                next(iter(eval_signatures.values())) if len(eval_signatures) == 1 else eval_signatures
            ),
        }
        if early_prune_report is not None:
            record["fitness"]["early_prune"] = dict(early_prune_report)
        record["score"] = float("inf") if any_error else float(delta_mean)
        record["better_than_baseline_mean"] = bool(delta_mean < 0.0)
        record["better_than_baseline_strict"] = bool(all_deltas and all(float(d) < 0.0 for d in all_deltas))
        record["better_than_baseline_all_scenarios"] = bool(all_scenarios_negative)
        record["stage3_all_scenarios_negative"] = bool(all_scenarios_negative)
        record["stage3_nonnegative_scenario_count"] = int(scenario_nonnegative_count)
        record["stage3_worst_scenario_delta"] = float(worst_scenario_delta)
        fl_logger.info(
            "Stage3 offline mini-train DONE gen=%d pair_index=%d score=%s any_error=%s",
            int(generation),
            int(pair_index),
            str(record.get("score")),
            str(any_error),
        )
        record["elapsed_s"] = float(time.time() - t0)
        return record
    except Exception as exc:  # noqa: BLE001
        record["pair_ok"] = False
        record["pair_reason"] = "stage3_fatal"
        record["fatal_error"] = str(exc)
        record["score"] = float("inf")
        record["elapsed_s"] = float(time.time() - t0)
        return record
    finally:
        if "file_handler" in locals() and file_handler is not None:
            try:
                logging.getLogger("fitness.free_loss_fidelity").removeHandler(file_handler)
            except Exception:  # noqa: BLE001
                pass
            try:
                logging.getLogger().removeHandler(file_handler)
            except Exception:  # noqa: BLE001
                pass
            try:
                file_handler.close()
            except Exception:  # noqa: BLE001
                pass


def _hf_eval_task_child(payload: Mapping[str, Any], conn: Any) -> None:
    """Run one HF task in an isolated child process."""

    try:
        rec = _evaluate_pair_worker(payload)
    except Exception as exc:  # noqa: BLE001
        fixed = dict(payload) if isinstance(payload, Mapping) else {}
        fixed["pair_ok"] = False
        fixed["pair_reason"] = "child_exception"
        fixed["high_fidelity_error"] = f"{type(exc).__name__}: {exc}"
        fixed["high_fidelity_traceback"] = traceback.format_exc()
        fixed["score"] = float("inf")
        rec = fixed
    try:
        conn.send(dict(rec))
    finally:
        conn.close()


def _hf_pinned_device_worker(  # noqa: PLR0912
    device_str: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    """Run HF evaluations on a single pinned device.

    This ensures we never oversubscribe a GPU with multiple concurrent HF trainings.
    """

    while True:
        payload = task_queue.get()
        if payload is None:
            return
        try:
            fixed = dict(payload)
            fixed["device_str"] = str(device_str)
            cfg_yaml = fixed.get("cfg_yaml")
            hf_timeout_s = _resolve_hf_timeout_s(cfg_yaml, default=None)

            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            child = ctx.Process(target=_hf_eval_task_child, args=(fixed, child_conn))
            child.daemon = False
            try:
                child.start()
                child_conn.close()
                deadline = (time.time() + float(hf_timeout_s)) if hf_timeout_s is not None else None
                rec = None
                while True:
                    if deadline is None:
                        remaining = None
                        poll_s = 1.0
                    else:
                        remaining = float(deadline - time.time())
                        poll_s = max(0.0, min(1.0, remaining))
                    if parent_conn.poll(poll_s):
                        rec = dict(parent_conn.recv())
                        break
                    if not child.is_alive():
                        break
                    if remaining is not None and remaining <= 0.0:
                        break

                if rec is None:
                    fixed["pair_ok"] = False
                    fixed["score"] = float("inf")
                    if child.is_alive():
                        fixed["pair_reason"] = "child_timeout"
                        fixed["high_fidelity_error"] = f"HF task exceeded timeout_s={hf_timeout_s:.1f}"
                        child.terminate()
                    else:
                        fixed["pair_reason"] = "child_exit_no_result"
                        fixed["high_fidelity_error"] = f"HF child exited without result (exitcode={child.exitcode})"
                    rec = fixed
            finally:
                try:
                    child_conn.close()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    if child.is_alive():
                        child.join(timeout=5.0)
                    else:
                        child.join()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    parent_conn.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            fixed = dict(payload) if isinstance(payload, dict) else {}
            fixed["device_str"] = str(device_str)
            fixed["pair_ok"] = False
            fixed["pair_reason"] = "high_fidelity_failed"
            fixed["high_fidelity_error"] = str(exc)
            fixed["score"] = float("inf")
            rec = fixed
        result_queue.put(dict(rec))


def run_pref_loss_coevo(
    config_path: str,
    *,
    resume_dir: str | None = None,
    **overrides: Any,
) -> None:
    """Run the co-evolution search loop for builder population G and loss population F.

    Inputs:
        config_path: YAML path with at least the keys:
            - generations, pop_g, pop_f, elite_g, elite_f
            - pairing_budget_per_gen
            - cheap_gate_on, high_fidelity_on
            - credit_assignment ("mean" | "best-k" | "shapley_approx")
            - devices (list[str]) and mp worker config (mp.enabled/processes/start_method)
        resume_dir: if set, load `checkpoint.json` under this directory and resume from `next_generation`.
        overrides: flat key overrides (e.g. device="cpu") applied after loading YAML.
    Outputs:
        Writes artifacts under a run directory (timestamped under output_root or resume_dir):
            - builders.jsonl, losses.jsonl, pairs.jsonl, gate_reports.jsonl
            - checkpoint.json, summary.json, eval_protocol.json
            - best_builder.json, best_loss.json, best_pair.json
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f) or {}
    if not isinstance(cfg_yaml, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    cfg_yaml.update({k: v for k, v in overrides.items() if v is not None})
    cfg_yaml, runtime_meta = _resolve_runtime_config(cfg_yaml)

    seed = int(cfg_yaml.get("seed", 0))
    _set_seed(seed)
    rng = random.Random(seed)

    generations = int(cfg_yaml.get("generations", 1))
    pop_g = int(cfg_yaml.get("pop_g", 8))
    pop_f = int(cfg_yaml.get("pop_f", 8))
    elite_g = int(cfg_yaml.get("elite_g", 4))
    elite_f = int(cfg_yaml.get("elite_f", 4))
    pairing_budget = int(cfg_yaml.get("pairing_budget_per_gen", 16))
    pairing_budget_loss = int(cfg_yaml.get("pairing_budget_loss", pairing_budget // 2) or (pairing_budget // 2))
    pairing_budget_loss = max(0, min(int(pairing_budget_loss), int(pairing_budget)))
    pairing_budget_builder = int(cfg_yaml.get("pairing_budget_builder", pairing_budget - pairing_budget_loss) or 0)
    if pairing_budget_builder < 0:
        pairing_budget_builder = 0
    if (pairing_budget_loss + pairing_budget_builder) <= 0:
        pairing_budget_loss = int(pairing_budget // 2)
        pairing_budget_builder = int(pairing_budget - pairing_budget_loss)

    alternating_schedule_raw = cfg_yaml.get("alternating_schedule", {}) or {}
    if not isinstance(alternating_schedule_raw, dict):
        alternating_schedule_raw = {}
    alternating_loss_generations = max(
        0,
        _safe_int(
            alternating_schedule_raw.get("loss_generations", alternating_schedule_raw.get("loss_phase_generations", 0)),
            0,
        ),
    )
    alternating_builder_generations = max(
        0,
        _safe_int(
            alternating_schedule_raw.get(
                "builder_generations",
                alternating_schedule_raw.get("pair_generations", alternating_schedule_raw.get("builder_phase_generations", 0)),
            ),
            0,
        ),
    )
    alternating_final_loss_generations = max(
        0,
        _safe_int(
            alternating_schedule_raw.get(
                "final_loss_generations",
                alternating_schedule_raw.get("tail_loss_generations", 0),
            ),
            0,
        ),
    )
    alternating_start_phase = str(alternating_schedule_raw.get("start_phase", "loss") or "loss").strip().lower()
    if alternating_start_phase not in {"loss", "builder"}:
        LOGGER.warning(
            "Invalid alternating_schedule.start_phase=%r; falling back to 'loss'.",
            alternating_schedule_raw.get("start_phase"),
        )
        alternating_start_phase = "loss"
    alternating_rounds = max(0, _safe_int(alternating_schedule_raw.get("rounds", 0), 0))
    alternating_schedule_enabled = bool(alternating_loss_generations > 0 and alternating_builder_generations > 0)

    preset = str(cfg_yaml.get("preset", "advanced") or "advanced").strip().lower()
    search_mode = str(cfg_yaml.get("search_mode", "coevo") or "coevo").strip().lower()
    metric_mode = _normalize_metric_mode(cfg_yaml.get("metric_mode", "minimize"))
    improve_eps = float(cfg_yaml.get("improve_eps", 0.0) or 0.0)
    eval_stages = _normalize_eval_stages(cfg_yaml)
    stage3_multifidelity_cfg = _normalize_stage3_multifidelity_cfg(cfg_yaml.get("stage3_multifidelity", {}))
    prefer_all_stage3_scenarios_negative = bool(cfg_yaml.get("stage3_prefer_all_scenarios_negative", False))

    cheap_gate_on = bool(cfg_yaml.get("cheap_gate_on", True))
    high_fidelity_on = bool(cfg_yaml.get("high_fidelity_on", True))

    credit_mode = str(cfg_yaml.get("credit_assignment", "mean") or "mean")
    credit_best_k = int(cfg_yaml.get("credit_best_k", 3) or 3)

    operator_whitelist = list(cfg_yaml.get("operator_whitelist", []))

    devices = cfg_yaml.get("devices")
    if isinstance(devices, (list, tuple)) and devices:
        device_list = [_normalize_device_alias(str(d)) for d in devices]
    else:
        device_list = [_normalize_device_alias(str(cfg_yaml.get("device", "cuda")))]

    mp_cfg = cfg_yaml.get("mp", {}) or {}
    if not isinstance(mp_cfg, dict):
        mp_cfg = {}
    mp_enabled = bool(mp_cfg.get("enabled", False))
    mp_processes = int(mp_cfg.get("processes", len(device_list)) or len(device_list))
    mp_start_method = str(mp_cfg.get("start_method", "spawn") or "spawn")

    out_root = str(cfg_yaml.get("output_root", "runs/pref_loss_coevo"))
    os.makedirs(out_root, exist_ok=True)

    resume_state: Dict[str, Any] | None = None
    if resume_dir:
        run_dir = os.path.abspath(str(resume_dir))
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"resume_dir does not exist: {run_dir}")
        resume_state = _load_checkpoint(run_dir)
        for k, default_v in (
            ("preset", preset),
            ("search_mode", search_mode),
            ("metric_mode", metric_mode),
            ("improve_eps", improve_eps),
        ):
            if k in resume_state and resume_state.get(k) != default_v:
                LOGGER.warning(
                    "Resume override: checkpoint %s=%r (config requested %r); using checkpoint value.",
                    str(k),
                    resume_state.get(k),
                    default_v,
                )
        ckpt_eval_stages = resume_state.get("eval_stages")
        if isinstance(ckpt_eval_stages, dict) and ckpt_eval_stages != eval_stages:
            LOGGER.warning(
                "Resume override: checkpoint eval_stages=%s (config requested %s); using checkpoint value.",
                dict(ckpt_eval_stages),
                dict(eval_stages),
            )

        seed_from_ckpt = resume_state.get("seed")
        if seed_from_ckpt is not None:
            seed = int(seed_from_ckpt)
            _set_seed(seed)
            rng = random.Random(seed)
        rng_state = resume_state.get("rng_state_b64")
        if isinstance(rng_state, str) and rng_state:
            try:
                rng.setstate(_unb64_pickle(rng_state))
            except Exception:  # noqa: BLE001
                pass
        preset = str(resume_state.get("preset", preset) or preset).strip().lower()
        search_mode = _normalize_search_mode(resume_state.get("search_mode", search_mode), default_mode=search_mode)
        metric_mode = _normalize_metric_mode(resume_state.get("metric_mode", metric_mode))
        improve_eps = float(resume_state.get("improve_eps", improve_eps) or 0.0)
        ckpt_eval_stages = resume_state.get("eval_stages")
        if isinstance(ckpt_eval_stages, dict):
            eval_stages = _normalize_eval_stages({"eval_stages": ckpt_eval_stages})
        cfg_yaml["preset"] = str(preset)
        cfg_yaml["search_mode"] = str(search_mode)
        cfg_yaml["metric_mode"] = str(metric_mode)
        cfg_yaml["improve_eps"] = float(improve_eps)
        cfg_yaml["eval_stages"] = dict(eval_stages)
        cfg_yaml["cheap_gate_on"] = bool(eval_stages.get("stage0_gate", True))
        cfg_yaml["high_fidelity_on"] = bool(eval_stages.get("stage3_high_fidelity", True))
        cfg_yaml["micro_unroll_enabled"] = bool(eval_stages.get("stage2_micro_unroll", False))
        cheap_gate_on = bool(cfg_yaml.get("cheap_gate_on", True))
        high_fidelity_on = bool(cfg_yaml.get("high_fidelity_on", True))
        LOGGER.info("Resuming run_dir=%s next_generation=%s", run_dir, resume_state.get("next_generation"))
    else:
        run_dir = _timestamp_dir(out_root)

    LOGGER.info("Run directory: %s", os.path.abspath(run_dir))
    LOGGER.info("EVAL_STAGES enabled: %s", _format_eval_stages_for_log(eval_stages))
    try:
        runtime_trace_hb_s = max(5.0, float(cfg_yaml.get("runtime_trace_heartbeat_s", 30.0) or 30.0))
    except (TypeError, ValueError):
        runtime_trace_hb_s = 30.0
    runtime_trace = RuntimeTrace(
        os.path.join(run_dir, "runtime_status.json"),
        role="pref_loss_coevo_main",
        heartbeat_interval_s=runtime_trace_hb_s,
    )
    runtime_trace.start(
        extra={
            "config_path": os.path.abspath(str(config_path)),
            "resume_dir": (os.path.abspath(str(resume_dir)) if resume_dir else None),
            "run_dir": os.path.abspath(str(run_dir)),
            "search_mode": str(search_mode),
            "preset": str(preset),
            "generations": int(generations),
            "seed": int(seed),
            "devices": list(device_list),
            "mp_enabled": bool(mp_enabled),
            "mp_processes": int(mp_processes),
            "hf_scheduler_mode": _hf_scheduler_mode(cfg_yaml),
        },
    )
    runtime_trace.install_signal_handlers()
    runtime_trace.heartbeat(
        extra={
            "stage": "initialized",
            "next_generation": int(resume_state.get("next_generation", 0)) if isinstance(resume_state, dict) else 0,
        },
        force=True,
    )

    improve_eps_calibration: Dict[str, Any] | None = None
    if resume_state is not None:
        raw_calib = resume_state.get("improve_eps_calibration")
        if isinstance(raw_calib, dict):
            improve_eps_calibration = dict(raw_calib)
    else:
        # Optional: calibrate `improve_eps` from baseline mini-train noise.
        calib_cfg_yaml: Mapping[str, Any] = cfg_yaml
        try:
            if bool(stage3_multifidelity_cfg.get("enabled")) and stage3_multifidelity_cfg.get("rounds"):
                calib_cfg_yaml = _apply_stage3_round_overrides(cfg_yaml, stage3_multifidelity_cfg["rounds"][-1])
        except Exception:  # noqa: BLE001
            calib_cfg_yaml = cfg_yaml
        try:
            improve_eps_calibration = _calibrate_improve_eps_from_baseline_noise(
                cfg_yaml=calib_cfg_yaml,
                operator_whitelist=operator_whitelist,
                device_str=str(device_list[0] if device_list else "cuda"),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("improve_eps calibration failed (ignored): %s", str(exc))
            improve_eps_calibration = None
        if isinstance(improve_eps_calibration, dict) and improve_eps_calibration.get("improve_eps") is not None:
            try:
                improve_eps = float(improve_eps_calibration["improve_eps"])
                cfg_yaml["improve_eps"] = float(improve_eps)
                LOGGER.info(
                    "improve_eps calibrated: sigma_delta=%s improve_eps=%s fidelity=%s N=%s",
                    improve_eps_calibration.get("sigma_delta"),
                    float(improve_eps),
                    improve_eps_calibration.get("fidelity"),
                    improve_eps_calibration.get("N"),
                )
                _atomic_write_json(os.path.join(run_dir, "improve_eps_calibration.json"), improve_eps_calibration)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to apply calibrated improve_eps (ignored): %s", str(exc))

    if bool(eval_stages.get("stage3_high_fidelity", True)):
        baseline_prepare_cfgs: List[Tuple[str, Mapping[str, Any]]] = []
        scenario_entries = _iter_stage3_scenario_cfgs(cfg_yaml)
        if bool(stage3_multifidelity_cfg.get("enabled")) and stage3_multifidelity_cfg.get("rounds"):
            rounds_raw = stage3_multifidelity_cfg.get("rounds") or []
            rounds = [dict(r) for r in rounds_raw if isinstance(r, dict)]
            for scenario_entry in scenario_entries:
                scenario_name = str(scenario_entry.get("name") or "scenario")
                scenario_cfg = dict(scenario_entry.get("cfg") or {})
                for rc in rounds:
                    baseline_prepare_cfgs.append((scenario_name, _apply_stage3_round_overrides(scenario_cfg, rc)))
        else:
            baseline_prepare_cfgs = [
                (str(scenario_entry.get("name") or "scenario"), dict(scenario_entry.get("cfg") or {}))
                for scenario_entry in scenario_entries
            ]

        baseline_prepare_summaries: List[Dict[str, Any]] = []
        seen_stage3_keys: set[str] = set()
        for scenario_name, cfg_stage3 in baseline_prepare_cfgs:
            fidelity_key = _stage3_fidelity_key(cfg_stage3)
            cache_key = f"{scenario_name}::{str(fidelity_key)}"
            if cache_key in seen_stage3_keys:
                continue
            seen_stage3_keys.add(cache_key)
            prepared = _ensure_stage3_baseline_mini_eval(
                cfg_yaml=cfg_stage3,
                operator_whitelist=operator_whitelist,
                device_str=str(device_list[0] if device_list else "cuda"),
            )
            baseline_prepare_summaries.append(
                {
                    "scenario": str(scenario_name),
                    "fidelity": str(fidelity_key),
                    "path": str(prepared.get("path")),
                    "cached": bool(prepared.get("cached", False)),
                    "regenerated": bool(prepared.get("regenerated", False)),
                }
            )
            LOGGER.info(
                "Stage3 baseline mini-eval prepared scenario=%s fidelity=%s path=%s cached=%s regenerated=%s",
                str(scenario_name),
                str(fidelity_key),
                str(prepared.get("path")),
                str(bool(prepared.get("cached", False))),
                str(bool(prepared.get("regenerated", False))),
            )
        if baseline_prepare_summaries:
            try:
                _atomic_write_json(
                    os.path.join(run_dir, "stage3_baseline_mini_eval_caches.json"),
                    {"entries": baseline_prepare_summaries},
                )
            except Exception:  # noqa: BLE001
                pass
    stage3_multiseed_cfg = _stage3_multiseed_compare_cfg(cfg_yaml)
    if bool(eval_stages.get("stage3_high_fidelity", True)) and bool(stage3_multiseed_cfg.get("enabled", False)):
        preload_cfgs: List[Tuple[str, Mapping[str, Any]]] = []
        scenario_entries = _iter_stage3_scenario_cfgs(cfg_yaml)
        if bool(stage3_multifidelity_cfg.get("enabled")) and stage3_multifidelity_cfg.get("rounds"):
            rounds_raw = stage3_multifidelity_cfg.get("rounds") or []
            rounds = [dict(r) for r in rounds_raw if isinstance(r, dict)]
            for scenario_entry in scenario_entries:
                scenario_name = str(scenario_entry.get("name") or "scenario")
                scenario_cfg = dict(scenario_entry.get("cfg") or {})
                for rc in rounds:
                    preload_cfgs.append((scenario_name, _apply_stage3_round_overrides(scenario_cfg, rc)))
        else:
            preload_cfgs = [
                (str(scenario_entry.get("name") or "scenario"), dict(scenario_entry.get("cfg") or {}))
                for scenario_entry in scenario_entries
            ]

        seen_preload_keys: set[str] = set()
        preload_summaries: List[Dict[str, Any]] = []
        for scenario_name, cfg_stage3 in preload_cfgs:
            try:
                fidelity_key = _stage3_fidelity_key(cfg_stage3)
            except Exception:  # noqa: BLE001
                fidelity_key = "unknown"
            cache_key = f"{scenario_name}::{str(fidelity_key)}"
            if cache_key in seen_preload_keys:
                continue
            seen_preload_keys.add(cache_key)
            try:
                preload_payload = _ensure_stage3_baseline_multiseed_cache(
                    cfg_yaml=cfg_stage3,
                    operator_whitelist=operator_whitelist,
                    device_str=str(device_list[0] if device_list else "cuda"),
                    n_seeds=int(stage3_multiseed_cfg.get("n_seeds", 0) or 0),
                    seed0=int(stage3_multiseed_cfg.get("seed0", 0) or 0),
                    seed_stride=int(stage3_multiseed_cfg.get("seed_stride", 997) or 997),
                )
                if isinstance(preload_payload, dict):
                    preload_summaries.append(
                        {
                            "scenario": str(scenario_name),
                            "fidelity": str(preload_payload.get("fidelity", fidelity_key)),
                            "cache_path": str(preload_payload.get("cache_path")),
                            "baseline_mini_eval_path": str(preload_payload.get("baseline_mini_eval_path")),
                            "n_seeds": int(preload_payload.get("n_seeds", 0) or 0),
                            "num_cached_seeds": int(len(preload_payload.get("per_seed", {}) or {})),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to prepare stage3 multiseed baseline cache for scenario=%s fidelity=%s: %s",
                    str(scenario_name),
                    str(fidelity_key),
                    str(exc),
                )
        if preload_summaries:
            try:
                _atomic_write_json(
                    os.path.join(run_dir, "stage3_baseline_multiseed_caches.json"),
                    {"entries": preload_summaries},
                )
            except Exception:  # noqa: BLE001
                pass
    if preset == "simple":
        ignored_keys = list(runtime_meta.get("ignored_advanced_keys", []))
        if ignored_keys:
            LOGGER.warning("preset=simple: advanced keys are ignored: %s", ignored_keys)
    if search_mode == "coevo":
        LOGGER.warning("search_mode=coevo is supported but not encouraged; prefer search_mode=alternating.")
    elif search_mode == "loss_only":
        LOGGER.info("search_mode=loss_only: builder population and selection are frozen; only losses are searched.")
    if search_mode == "alternating" and alternating_schedule_enabled:
        LOGGER.info(
            "Alternating schedule enabled: start_phase=%s loss_generations=%d builder_generations=%d final_loss_generations=%d rounds=%s",
            str(alternating_start_phase),
            int(alternating_loss_generations),
            int(alternating_builder_generations),
            int(alternating_final_loss_generations),
            (int(alternating_rounds) if alternating_rounds > 0 else "unbounded"),
        )
        if alternating_rounds > 0:
            planned = (
                int(alternating_rounds) * int(alternating_loss_generations + alternating_builder_generations)
                + int(alternating_final_loss_generations)
            )
            if int(generations) != int(planned):
                LOGGER.warning(
                    "alternating_schedule rounds imply %d generations, but configured generations=%d.",
                    int(planned),
                    int(generations),
                )

    # ----------------------
    # Double-EoH LLM wiring
    # ----------------------
    legacy_llm_enabled = bool(cfg_yaml.get("llm_enabled", False))
    legacy_llm_offline_mode = bool(cfg_yaml.get("llm_offline_mode", False))

    builder_llm_raw = cfg_yaml.get("builder_llm", {}) or {}
    loss_llm_raw = cfg_yaml.get("loss_llm", {}) or {}
    if not isinstance(builder_llm_raw, dict):
        builder_llm_raw = {}
    if not isinstance(loss_llm_raw, dict):
        loss_llm_raw = {}

    builder_llm_enabled = bool(builder_llm_raw.get("enabled", legacy_llm_enabled))
    loss_llm_enabled = bool(loss_llm_raw.get("enabled", legacy_llm_enabled))
    llm_enabled = bool(builder_llm_enabled or loss_llm_enabled)

    builder_offline = bool(builder_llm_raw.get("offline_mode", legacy_llm_offline_mode))
    loss_offline = bool(loss_llm_raw.get("offline_mode", legacy_llm_offline_mode))
    llm_offline_mode = bool(builder_offline or loss_offline or legacy_llm_offline_mode)

    llm_prompts_raw: Dict[str, Any] = {}
    legacy_prompts = cfg_yaml.get("llm_prompts", {}) or {}
    if isinstance(legacy_prompts, dict):
        llm_prompts_raw.update(dict(legacy_prompts))
    if isinstance(builder_llm_raw.get("prompts"), dict):
        llm_prompts_raw.update(dict(builder_llm_raw.get("prompts") or {}))
    if isinstance(loss_llm_raw.get("prompts"), dict):
        llm_prompts_raw.update(dict(loss_llm_raw.get("prompts") or {}))

    loss_prompt_context = loss_llm_ops.build_runtime_prompt_context(
        loss_observables=tuple(str(v) for v in cfg_yaml.get("loss_observables", []) if str(v).strip()),
        mode="pairwise",
    )
    llm_prompts_defaults = {
        "builder_generation": "PTP/prompts/pref_builder_generation.txt",
        "builder_crossover": "PTP/prompts/pref_builder_crossover.txt",
        "builder_mutation": "PTP/prompts/pref_builder_mutation.txt",
        "builder_e2": "PTP/prompts/pref_builder_e2.txt",
        "builder_paradigm_shift": "PTP/prompts/pref_builder_paradigm_shift.txt",
        "builder_structure_shift": "PTP/prompts/pref_builder_structure_shift.txt",
        "builder_constraint_inject": "PTP/prompts/pref_builder_constraint_inject.txt",
        "builder_m2": "PTP/prompts/pref_builder_m2.txt",
        "builder_m3": "PTP/prompts/pref_builder_m3.txt",
        "builder_repair": "PTP/prompts/pref_builder_repair.txt",
        "loss_generation": "PTP/prompts/free_loss_generation.txt",
        "loss_crossover": "PTP/prompts/free_loss_crossover.txt",
        "loss_mutation": "PTP/prompts/free_loss_mutation.txt",
        "loss_e2": "PTP/prompts/free_loss_e2.txt",
        "loss_paradigm_shift": "PTP/prompts/free_loss_paradigm_shift.txt",
        "loss_structure_shift": "PTP/prompts/free_loss_structure_shift.txt",
        "loss_constraint_inject": "PTP/prompts/free_loss_constraint_inject.txt",
        "loss_m2": "PTP/prompts/free_loss_m2.txt",
        "loss_m3": "PTP/prompts/free_loss_m3.txt",
        "loss_repair": "PTP/prompts/free_loss_repair.txt",
    }
    llm_prompts: Dict[str, str] = {}
    for k, v in llm_prompts_defaults.items():
        val = llm_prompts_raw.get(k, v)
        llm_prompts[k] = _abs_from_repo_root(str(val))

    default_parent_p = int(cfg_yaml.get("llm_parent_p", 5) or 5)
    default_repair_attempts = int(cfg_yaml.get("llm_repair_attempts", 1) or 1)
    default_repair_enabled = bool(cfg_yaml.get("llm_repair_on_failure", True))

    builder_repair_raw = builder_llm_raw.get("repair", {}) or {}
    if not isinstance(builder_repair_raw, dict):
        builder_repair_raw = {}
    loss_repair_raw = loss_llm_raw.get("repair", {}) or {}
    if not isinstance(loss_repair_raw, dict):
        loss_repair_raw = {}

    builder_cfg: Dict[str, Any] = {
        "enabled": bool(builder_llm_enabled),
        "parent_p": int(builder_llm_raw.get("parent_p", default_parent_p) or default_parent_p),
        "seed_reserve": int(builder_llm_raw.get("seed_reserve", cfg_yaml.get("builder_seed_reserve", 2)) or 2),
        "operator_bank": dict(builder_llm_raw.get("operator_bank") or {}) if isinstance(builder_llm_raw.get("operator_bank"), dict) else (
            dict((cfg_yaml.get("builder", {}) or {}).get("operator_bank") or {})
            if isinstance((cfg_yaml.get("builder", {}) or {}).get("operator_bank"), dict)
            else {}
        ),
        # New-style per-op counts (optional).
        "init_num_E1": builder_llm_raw.get("init_num_E1"),
        "init_num_E2": builder_llm_raw.get("init_num_E2"),
        "init_num_M1": builder_llm_raw.get("init_num_M1"),
        "init_num_M2": builder_llm_raw.get("init_num_M2"),
        "num_E1": builder_llm_raw.get("num_E1"),
        "num_E2": builder_llm_raw.get("num_E2"),
        "num_M1": builder_llm_raw.get("num_M1"),
        "num_M2": builder_llm_raw.get("num_M2"),
        # Legacy total budgets (fallback).
        "init_llm_g": int(builder_llm_raw.get("init_llm_g", cfg_yaml.get("init_llm_g", 0)) or 0),
        "llm_per_gen_g": int(builder_llm_raw.get("llm_per_gen_g", cfg_yaml.get("llm_per_gen_g", 0)) or 0),
        "repair": {
            "enabled": bool(builder_repair_raw.get("enabled", builder_llm_raw.get("repair_enabled", default_repair_enabled))),
            "max_attempts": int(
                builder_repair_raw.get(
                    "max_attempts",
                    builder_llm_raw.get("repair_max_attempts", default_repair_attempts),
                )
                or default_repair_attempts
            ),
            "simplify_first": bool(builder_repair_raw.get("simplify_first", True)),
        },
        "family_diversity": _normalize_family_diversity_cfg(
            builder_llm_raw.get("family_diversity", (cfg_yaml.get("builder", {}) or {}).get("family_diversity", {}))
        ),
    }
    loss_cfg: Dict[str, Any] = {
        "enabled": bool(loss_llm_enabled),
        "parent_p": int(loss_llm_raw.get("parent_p", default_parent_p) or default_parent_p),
        "seed_reserve": int(loss_llm_raw.get("seed_reserve", cfg_yaml.get("loss_seed_reserve", 2)) or 2),
        "operator_bank": dict(loss_llm_raw.get("operator_bank") or {}) if isinstance(loss_llm_raw.get("operator_bank"), dict) else (
            dict((cfg_yaml.get("loss", {}) or {}).get("operator_bank") or {})
            if isinstance((cfg_yaml.get("loss", {}) or {}).get("operator_bank"), dict)
            else {}
        ),
        "init_num_E1": loss_llm_raw.get("init_num_E1"),
        "init_num_E2": loss_llm_raw.get("init_num_E2"),
        "init_num_M1": loss_llm_raw.get("init_num_M1"),
        "init_num_M2": loss_llm_raw.get("init_num_M2"),
        "num_E1": loss_llm_raw.get("num_E1"),
        "num_E2": loss_llm_raw.get("num_E2"),
        "num_M1": loss_llm_raw.get("num_M1"),
        "num_M2": loss_llm_raw.get("num_M2"),
        "init_llm_f": int(loss_llm_raw.get("init_llm_f", cfg_yaml.get("init_llm_f", 0)) or 0),
        "llm_per_gen_f": int(loss_llm_raw.get("llm_per_gen_f", cfg_yaml.get("llm_per_gen_f", 0)) or 0),
        "repair": {
            "enabled": bool(loss_repair_raw.get("enabled", loss_llm_raw.get("repair_enabled", default_repair_enabled))),
            "max_attempts": int(
                loss_repair_raw.get(
                    "max_attempts",
                    loss_llm_raw.get("repair_max_attempts", default_repair_attempts),
                )
                or default_repair_attempts
            ),
            "simplify_first": bool(loss_repair_raw.get("simplify_first", True)),
        },
        "novelty": dict(loss_llm_raw.get("novelty") or {}) if isinstance(loss_llm_raw.get("novelty"), dict) else {},
        "exploration": dict(loss_llm_raw.get("exploration") or {}) if isinstance(loss_llm_raw.get("exploration"), dict) else {},
        "family_diversity": _normalize_family_diversity_cfg(
            loss_llm_raw.get("family_diversity", (cfg_yaml.get("loss", {}) or {}).get("family_diversity", {}))
        ),
    }

    llm_cfg: Dict[str, Any] = {
        "enabled": bool(llm_enabled),
        "offline_mode": bool(llm_offline_mode),
        "prompts": dict(llm_prompts),
        "builder_gate": {
            "min_pairs": int(cfg_yaml.get("builder_min_pairs", 1) or 1),
            "min_coverage": float(cfg_yaml.get("builder_min_coverage", 0.0) or 0.0),
            "max_pairs_per_instance": int(cfg_yaml.get("builder_max_pairs_per_instance", 4096) or 4096),
            "weight_nonneg": bool(cfg_yaml.get("builder_weight_nonneg", True)),
            "semantic_tolerance": float(cfg_yaml.get("builder_semantic_tolerance", 0.0) or 0.0),
            "semantic_min_pass_rate": float(cfg_yaml.get("builder_semantic_min_pass_rate", 1.0) or 1.0),
        },
        "builder": builder_cfg,
        "loss": loss_cfg,
    }

    if llm_enabled:
        loss_llm_ops.configure_llm_run(run_dir=run_dir, offline_mode=llm_offline_mode)
        builder_llm_ops.configure_llm_run(run_dir=run_dir, offline_mode=llm_offline_mode)
        try:
            LOGGER.info("LLM cache stats: %s", dict(loss_llm_ops.llm_cache_stats()))
        except Exception:  # noqa: BLE001
            pass

    builders_jsonl = os.path.join(run_dir, "builders.jsonl")
    losses_jsonl = os.path.join(run_dir, "losses.jsonl")
    pairs_jsonl = os.path.join(run_dir, "pairs.jsonl")
    gate_jsonl = os.path.join(run_dir, "gate_reports.jsonl")
    gate_repair_jsonl = os.path.join(run_dir, "gate_repair_reports.jsonl")
    summary_json = os.path.join(run_dir, "summary.json")
    eval_protocol_json = os.path.join(run_dir, "eval_protocol.json")

    if resume_state is None:
        for path in (builders_jsonl, losses_jsonl, pairs_jsonl, gate_jsonl, gate_repair_jsonl):
            with open(path, "w", encoding="utf-8"):
                pass

    gen_start = 0 if resume_state is None else int(resume_state.get("next_generation", 0) or 0)
    if resume_state is not None:
        for path in (builders_jsonl, losses_jsonl, pairs_jsonl, gate_jsonl, gate_repair_jsonl):
            kept, dropped = _truncate_jsonl_by_generation(path, gen_start)
            if dropped > 0:
                LOGGER.info(
                    "Resume JSONL truncate: path=%s gen_start=%d kept=%d dropped=%d",
                    path,
                    int(gen_start),
                    int(kept),
                    int(dropped),
                )
    seen_g = set(resume_state.get("seen_g", [])) if resume_state else set()
    seen_f = set(resume_state.get("seen_f", [])) if resume_state else set()
    resident_pop_g: List[Dict[str, Any]] = list(resume_state.get("resident_pop_g", [])) if resume_state else []
    resident_pop_f: List[Dict[str, Any]] = list(resume_state.get("resident_pop_f", [])) if resume_state else []
    elites_g: List[Dict[str, Any]] = list(resume_state.get("elites_g", [])) if resume_state else []
    elites_f: List[Dict[str, Any]] = list(resume_state.get("elites_f", [])) if resume_state else []
    if not resident_pop_g and elites_g:
        resident_pop_g = list(elites_g)
    if not resident_pop_f and elites_f:
        resident_pop_f = list(elites_f)
    diverse_elites_g: List[Dict[str, Any]] = list(resume_state.get("diverse_elites_g", [])) if resume_state else []
    diverse_elites_f: List[Dict[str, Any]] = list(resume_state.get("diverse_elites_f", [])) if resume_state else []
    hof_g: List[Dict[str, Any]] = list(resume_state.get("hof_g", [])) if resume_state else []
    hof_f: List[Dict[str, Any]] = list(resume_state.get("hof_f", [])) if resume_state else []
    archive_g: Dict[str, List[Dict[str, Any]]] = dict(resume_state.get("archive_g", {})) if resume_state else {}
    archive_f: Dict[str, List[Dict[str, Any]]] = dict(resume_state.get("archive_f", {})) if resume_state else {}
    if resume_state:
        pair_score_history_map_resume = resume_state.get("pair_score_history_map", {})
        resident_pop_f = _refresh_loss_population_scores_from_history(
            resident_pop_f,
            pair_score_history_map_resume,
            metric_mode=metric_mode,
        )
        if elites_f:
            resident_lookup_f = {str(item.get("id") or ""): dict(item) for item in resident_pop_f}
            refreshed_elites_f: List[Dict[str, Any]] = []
            for item in elites_f:
                loss_id = str((item or {}).get("id") or "")
                if loss_id and loss_id in resident_lookup_f:
                    refreshed_elites_f.append(dict(resident_lookup_f[loss_id]))
                elif isinstance(item, Mapping):
                    refreshed_elites_f.append(dict(item))
            refreshed_elites_f = _refresh_loss_population_scores_from_history(
                refreshed_elites_f,
                pair_score_history_map_resume,
                metric_mode=metric_mode,
            )
            elites_f = list(refreshed_elites_f[: max(0, min(len(refreshed_elites_f), len(elites_f)))])
    if resume_state is None:
        loss_transfer_seed_cfg = _normalize_loss_transfer_seed_cfg(cfg_yaml)
        if bool(loss_transfer_seed_cfg.get("enabled", False)) and (not resident_pop_f):
            transfer_loss_path = str(loss_transfer_seed_cfg.get("source_loss_path", "") or "").strip()
            if transfer_loss_path:
                resolved_loss_path = _resolve_loss_transfer_seed_loss_path(loss_transfer_seed_cfg)
                imported_losses = _load_loss_transfer_seed_entries_from_loss_path(
                    resolved_loss_path,
                    keep_source_fitness=bool(loss_transfer_seed_cfg.get("keep_source_fitness", True)),
                    reset_history=bool(loss_transfer_seed_cfg.get("reset_history", False)),
                )
            else:
                transfer_ckpt = _resolve_loss_transfer_seed_checkpoint_path(loss_transfer_seed_cfg)
                imported_losses = _load_loss_transfer_seed_entries(
                    transfer_ckpt,
                    source_pool=str(loss_transfer_seed_cfg.get("source_pool", "elites")),
                    top_k=min(int(pop_f), int(loss_transfer_seed_cfg.get("top_k", 8) or 8)),
                    max_per_family=int(loss_transfer_seed_cfg.get("max_per_family", 2) or 2),
                    keep_source_fitness=bool(loss_transfer_seed_cfg.get("keep_source_fitness", True)),
                    reset_history=bool(loss_transfer_seed_cfg.get("reset_history", False)),
                )
            if imported_losses:
                resident_pop_f = list(imported_losses)
                elites_f = list(imported_losses[: max(0, int(elite_f))])
                for item in imported_losses:
                    sig = str(item.get("signature") or "").strip()
                    if sig:
                        seen_f.add(sig)
                if transfer_loss_path:
                    LOGGER.info(
                        "Initialized loss population from transfer seed artifact: loss=%s imported=%d",
                        str(resolved_loss_path),
                        int(len(imported_losses)),
                    )
                else:
                    LOGGER.info(
                        "Initialized loss population from transfer seeds: checkpoint=%s pool=%s imported=%d",
                        str(transfer_ckpt),
                        str(loss_transfer_seed_cfg.get("source_pool", "elites")),
                        int(len(imported_losses)),
                    )
    best_so_far: Dict[str, Any] | None = None
    if resume_state and isinstance(resume_state.get("best_so_far"), dict):
        best_so_far = dict(resume_state.get("best_so_far", {}))
    elif resume_state:
        legacy_best_score = resume_state.get("best_score")
        legacy_g = resume_state.get("best_builder_id")
        legacy_f = resume_state.get("best_loss_id")
        if legacy_best_score is not None and legacy_g and legacy_f:
            try:
                best_so_far = {
                    "score": float(legacy_best_score),
                    "builder_id": str(legacy_g),
                    "loss_id": str(legacy_f),
                    "stage_final": str(resume_state.get("best_stage_final", "unknown")),
                    "generation": int(resume_state.get("best_gen", -1) or -1),
                    "phase": str(resume_state.get("best_phase", "unknown")),
                }
            except (TypeError, ValueError):
                best_so_far = None
    best_builder_cost: Dict[str, Any] | None = None
    if resume_state and isinstance(resume_state.get("best_builder_cost"), dict):
        best_builder_cost = dict(resume_state.get("best_builder_cost", {}))
    builder_cost_archive: Dict[str, Dict[str, Any]] = {}
    if resume_state and isinstance(resume_state.get("builder_cost_archive"), Mapping):
        for gid, raw in dict(resume_state.get("builder_cost_archive") or {}).items():
            if isinstance(raw, Mapping) and str(gid):
                builder_cost_archive[str(gid)] = dict(raw)
    pair_score_history_map: Dict[str, List[Dict[str, Any]]] = {}
    if resume_state and isinstance(resume_state.get("pair_score_history_map"), Mapping):
        for key, value in dict(resume_state.get("pair_score_history_map", {})).items():
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                continue
            pair_score_history_map[str(key)] = [dict(item) for item in value if isinstance(item, Mapping)]

    # Pair-level cached evaluation results (in-memory by default).
    cache_dir = cfg_yaml.get("cache_persist_dir", None)
    persist_dir: str | None = None
    if isinstance(cache_dir, str) and cache_dir.strip():
        persist_dir = os.path.join(run_dir, cache_dir)
    caches = PrefLossEvalCaches(persist_dir=persist_dir)

    # Stable budget signature for caching across resume.
    sig_hf_cfg = _build_hf_cfg(cfg_yaml, seed=seed, device_str="cpu")
    proxy_problem_size = int(cfg_yaml.get("proxy_problem_size", sig_hf_cfg.train_problem_size))
    proxy_batch_size = int(cfg_yaml.get("proxy_batch_size", 64) or 64)
    proxy_batches = int(cfg_yaml.get("proxy_batches", 2) or 2)
    proxy_weights = cfg_yaml.get("proxy_weights", {"effective_grad_ratio": 1.0, "ess_ratio": 0.1}) or {}
    if not isinstance(proxy_weights, dict):
        proxy_weights = {"effective_grad_ratio": 1.0, "ess_ratio": 0.1}
    micro_budget = {
        "micro_unroll_enabled": bool(cfg_yaml.get("micro_unroll_enabled", False)),
        "micro_unroll_top_k": int(cfg_yaml.get("micro_unroll_top_k", 0) or 0),
        "micro_unroll_steps": int(cfg_yaml.get("micro_unroll_steps", 0) or 0),
        "micro_unroll_lr": float(cfg_yaml.get("micro_unroll_lr", 0.0) or 0.0),
        "eval_stages": dict(eval_stages),
        "metric_mode": str(metric_mode),
        "improve_eps": float(improve_eps),
        "search_mode": str(search_mode),
    }
    eval_sig = eval_budget_signature(
        cfg=sig_hf_cfg,
        proxy_problem_size=proxy_problem_size,
        proxy_batch_size=proxy_batch_size,
        proxy_batches=proxy_batches,
        proxy_weights={str(k): float(v) for k, v in dict(proxy_weights).items()},
        extra_budget=micro_budget,
    )

    eval_protocol_payload = {
        "preset": str(preset),
        "search_mode": str(search_mode),
        "alternating_schedule": {
            "enabled": bool(alternating_schedule_enabled),
            "start_phase": str(alternating_start_phase),
            "loss_generations": int(alternating_loss_generations),
            "builder_generations": int(alternating_builder_generations),
            "final_loss_generations": int(alternating_final_loss_generations),
            "rounds": int(alternating_rounds),
        },
        "metric_mode": str(metric_mode),
        "improve_eps": float(improve_eps),
        "eval_stages": dict(eval_stages),
        "budgets": {
            "generations": int(generations),
            "pairing_budget_per_gen": int(pairing_budget),
            "pairing_budget_loss": int(pairing_budget_loss),
            "pairing_budget_builder": int(pairing_budget_builder),
            "proxy_batches": int(proxy_batches),
            "proxy_batch_size": int(proxy_batch_size),
            "micro_unroll_steps": int(cfg_yaml.get("micro_unroll_steps", 0) or 0),
            "micro_unroll_max_pairs": int(cfg_yaml.get("micro_unroll_max_pairs", 0) or 0),
            "high_fidelity_top_m": int(cfg_yaml.get("high_fidelity_top_m", 0) or 0),
            "hf_epochs": int(cfg_yaml.get("hf_epochs", 0) or 0),
        },
        "builder_pair_budget": {
            "per_instance": int(cfg_yaml.get("builder_max_pairs_per_instance", 0) or 0),
            "total": int(cfg_yaml.get("descriptor_pair_count_cap", 0) or 0),
        },
        "pair_score_selection_policy": "HF > micro > proxy > none",
        "eval_budget_signature": str(eval_sig),
    }
    _atomic_write_json(eval_protocol_json, eval_protocol_payload)

    # Sanity: proxy rollouts (pomo_size) control per-instance pair count for all_pairs (~K*(K-1)/2).
    # If this exceeds builder_max_pairs_per_instance, cheap gates will reject most/all builders,
    # resulting in no `cheap` / `high_fidelity` stages.
    try:
        proxy_rollouts = resolve_pomo_size(sig_hf_cfg.pomo_size, int(proxy_problem_size))
        max_pairs_per_instance = int(cfg_yaml.get("builder_max_pairs_per_instance", 4096) or 4096)
        expected_all_pairs = int(proxy_rollouts * (proxy_rollouts - 1) // 2)
        if expected_all_pairs > max_pairs_per_instance:
            LOGGER.warning(
                "Proxy rollouts K=%d implies all_pairs has ~%d pairs/instance, which exceeds builder_max_pairs_per_instance=%d. "
                "Fix by setting `pomo_size: null` (align to problem size) or increasing builder_max_pairs_per_instance.",
                int(proxy_rollouts),
                int(expected_all_pairs),
                int(max_pairs_per_instance),
            )
    except Exception:  # noqa: BLE001
        pass

    # Optional external baseline (metrics.csv) for epoch-by-epoch comparisons during HF.
    baseline_epoch_objectives: List[float] | None = None
    baseline_early_valid: float | None = None
    early_eval_steps: int = _compute_early_eval_steps(cfg_yaml, sig_hf_cfg)
    baseline_cfg = cfg_yaml.get("baseline", {}) or {}
    baseline_metrics_csv = (
        baseline_cfg.get("metrics_csv")
        or cfg_yaml.get("baseline_metrics_csv")
        or cfg_yaml.get("baseline_metrics_path")
    )
    baseline_ckpt_epoch = baseline_cfg.get(
        "checkpoint_epoch", cfg_yaml.get("baseline_checkpoint_epoch")
    )
    if baseline_ckpt_epoch is None:
        ckpt = baseline_cfg.get("checkpoint") or cfg_yaml.get("baseline_checkpoint") or cfg_yaml.get("baseline_ckpt")
        if ckpt:
            baseline_ckpt_epoch = _infer_baseline_epoch_from_path(str(ckpt))
    baseline_val_column = str(
        baseline_cfg.get("val_column", cfg_yaml.get("baseline_val_column", "val/reward"))
        or "val/reward"
    )

    scratch_hf_epochs_cfg = int(cfg_yaml.get("scratch_hf_epochs", 0) or 0)
    warmstart_hf_epochs_cfg = int(cfg_yaml.get("warmstart_hf_epochs", 0) or 0)
    split_hf_epoch_eval = bool(scratch_hf_epochs_cfg > 0 and warmstart_hf_epochs_cfg > 0)

    if baseline_metrics_csv and baseline_ckpt_epoch is not None and int(getattr(sig_hf_cfg, "hf_epochs", 0) or 0) > 0:
        metrics_path = _abs_from_repo_root(str(baseline_metrics_csv))
        start_epoch = int(baseline_ckpt_epoch) + 1
        baseline_scratch_epoch_objectives: List[float] | None = None
        baseline_warmstart_epoch_objectives: List[float] | None = None
        scratch_start_epoch_used: int | None = None
        try:
            if split_hf_epoch_eval:
                scratch_start_epoch_cfg = (
                    (baseline_cfg or {}).get("scratch_start_epoch")
                    or cfg_yaml.get("baseline_scratch_start_epoch")
                )
                if scratch_start_epoch_cfg is None:
                    try:
                        baseline_scratch_epoch_objectives = baseline_epoch_objectives_from_metrics_csv(
                            metrics_path,
                            value_col=baseline_val_column,
                            start_epoch=0,
                            num_epochs=int(scratch_hf_epochs_cfg),
                            objective_sign=str(sig_hf_cfg.objective_sign),
                        )
                        scratch_start_epoch_used = 0
                    except Exception:  # noqa: BLE001
                        baseline_scratch_epoch_objectives = baseline_epoch_objectives_from_metrics_csv(
                            metrics_path,
                            value_col=baseline_val_column,
                            start_epoch=1,
                            num_epochs=int(scratch_hf_epochs_cfg),
                            objective_sign=str(sig_hf_cfg.objective_sign),
                        )
                        scratch_start_epoch_used = 1
                else:
                    scratch_start_epoch_used = int(scratch_start_epoch_cfg)
                    baseline_scratch_epoch_objectives = baseline_epoch_objectives_from_metrics_csv(
                        metrics_path,
                        value_col=baseline_val_column,
                        start_epoch=int(scratch_start_epoch_used),
                        num_epochs=int(scratch_hf_epochs_cfg),
                        objective_sign=str(sig_hf_cfg.objective_sign),
                    )

                baseline_warmstart_epoch_objectives = baseline_epoch_objectives_from_metrics_csv(
                    metrics_path,
                    value_col=baseline_val_column,
                    start_epoch=int(start_epoch),
                    num_epochs=int(warmstart_hf_epochs_cfg),
                    objective_sign=str(sig_hf_cfg.objective_sign),
                )

                baseline_epoch_objectives = list(baseline_scratch_epoch_objectives or []) + list(
                    baseline_warmstart_epoch_objectives or []
                )
            else:
                baseline_epoch_objectives = baseline_epoch_objectives_from_metrics_csv(
                    metrics_path,
                    value_col=baseline_val_column,
                    start_epoch=start_epoch,
                    num_epochs=int(sig_hf_cfg.hf_epochs),
                    objective_sign=str(sig_hf_cfg.objective_sign),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load external baseline epoch objectives (%s): %s", metrics_path, exc)
            baseline_epoch_objectives = None
            baseline_early_valid = None
        else:
            steps_per_epoch, _ = get_hf_epoch_plan(sig_hf_cfg)
            baseline_early_valid = None
            if (
                steps_per_epoch > 0
                and int(early_eval_steps) > 0
                and int(early_eval_steps) % int(steps_per_epoch) == 0
            ):
                early_epochs = int(int(early_eval_steps) // int(steps_per_epoch))
                early_source = (
                    baseline_warmstart_epoch_objectives
                    if split_hf_epoch_eval and baseline_warmstart_epoch_objectives
                    else baseline_epoch_objectives
                )
                if early_source and 1 <= early_epochs <= len(early_source):
                    baseline_early_valid = float(early_source[early_epochs - 1])

            LOGGER.info(
                "External baseline loaded for HF comparisons: metrics=%s warmstart_epoch_start=%d epochs=%d val_column=%s split=%s scratch_start_epoch=%s early_eval_steps=%d",
                os.path.abspath(metrics_path),
                int(baseline_ckpt_epoch) + 1,
                int(len(baseline_epoch_objectives or [])),
                str(baseline_val_column),
                bool(split_hf_epoch_eval),
                int(scratch_start_epoch_used) if scratch_start_epoch_used is not None else None,
                int(early_eval_steps),
            )
    if resume_state is not None:
        eval_sigs_to_load: List[str] = [str(eval_sig)]
        if bool(stage3_multifidelity_cfg.get("enabled")) and stage3_multifidelity_cfg.get("rounds"):
            rounds_raw = stage3_multifidelity_cfg.get("rounds") or []
            rounds = [dict(r) for r in rounds_raw if isinstance(r, dict)]
            for rc in rounds:
                cfg_r = _apply_stage3_round_overrides(cfg_yaml, rc)
                sig_hf_r = _build_hf_cfg(cfg_r, seed=int(seed), device_str="cpu")
                sig_r = eval_budget_signature(
                    cfg=sig_hf_r,
                    proxy_problem_size=proxy_problem_size,
                    proxy_batch_size=proxy_batch_size,
                    proxy_batches=proxy_batches,
                    proxy_weights={str(k): float(v) for k, v in dict(proxy_weights).items()},
                    extra_budget=micro_budget,
                )
                eval_sigs_to_load.append(str(sig_r))
        eval_sigs_to_load = list(dict.fromkeys([str(s) for s in eval_sigs_to_load if str(s)]))
        loaded_total = 0
        for sig in eval_sigs_to_load:
            loaded_total += int(load_pair_cache_from_pairs_jsonl(caches=caches, pairs_jsonl_path=pairs_jsonl, eval_sig=sig))
        LOGGER.info("Loaded %d cached pair records from pairs.jsonl (eval_sigs=%s)", loaded_total, eval_sigs_to_load)
        rebuilt_pair_score_history_map = _rebuild_pair_score_history_map(list(caches.pair_cache.values()))
        if rebuilt_pair_score_history_map:
            pair_score_history_map = dict(rebuilt_pair_score_history_map)
            resident_pop_f = _refresh_loss_population_scores_from_history(
                resident_pop_f,
                pair_score_history_map,
                metric_mode=metric_mode,
            )
            if elites_f:
                resident_lookup_f = {str(item.get("id") or ""): dict(item) for item in resident_pop_f}
                refreshed_elites_f: List[Dict[str, Any]] = []
                for item in elites_f:
                    loss_id = str((item or {}).get("id") or "")
                    if loss_id and loss_id in resident_lookup_f:
                        refreshed_elites_f.append(dict(resident_lookup_f[loss_id]))
                    elif isinstance(item, Mapping):
                        refreshed_elites_f.append(dict(item))
                elites_f = list(
                    _refresh_loss_population_scores_from_history(
                        refreshed_elites_f,
                        pair_score_history_map,
                        metric_mode=metric_mode,
                    )[: max(0, min(len(refreshed_elites_f), len(elites_f)))]
                )
        resolved_resume_best = _resolve_best_pair_record(
            best_so_far=best_so_far,
            pair_records=None,
            pair_cache_records=list(caches.pair_cache.values()),
            metric_mode=metric_mode,
            prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
        )
        if (best_so_far is None) or (not isinstance(resolved_resume_best, Mapping)) or (not bool(resolved_resume_best.get("pair_ok"))):
            rebuilt_best_rec = _select_best_valid_pair_record(
                list(caches.pair_cache.values()),
                metric_mode=metric_mode,
                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
            )
            if rebuilt_best_rec is not None:
                rebuilt_best_score = _pair_record_effective_score(rebuilt_best_rec)
                if rebuilt_best_score is not None:
                    LOGGER.warning(
                        "Rebuilt incumbent from persisted valid pair records: old_best=%s new_best=%s pair=(%s,%s) stage=%s gen=%s",
                        (dict(best_so_far) if isinstance(best_so_far, dict) else None),
                        float(rebuilt_best_score),
                        rebuilt_best_rec.get("g_id"),
                        rebuilt_best_rec.get("f_id"),
                        rebuilt_best_rec.get("stage_final", rebuilt_best_rec.get("stage")),
                        rebuilt_best_rec.get("generation"),
                    )
                    best_so_far = {
                        "score": float(rebuilt_best_score),
                        "builder_id": str(rebuilt_best_rec.get("g_id")),
                        "loss_id": str(rebuilt_best_rec.get("f_id")),
                        "stage_final": str(rebuilt_best_rec.get("stage_final", rebuilt_best_rec.get("stage", "none"))),
                        "generation": int(_safe_int(rebuilt_best_rec.get("generation", -1), -1)),
                        "phase": str(rebuilt_best_rec.get("phase", "coevo")),
                        "stage3_all_scenarios_negative": rebuilt_best_rec.get("stage3_all_scenarios_negative"),
                        "stage3_nonnegative_scenario_count": rebuilt_best_rec.get("stage3_nonnegative_scenario_count"),
                        "stage3_worst_scenario_delta": rebuilt_best_rec.get("stage3_worst_scenario_delta"),
                    }

    if best_so_far is None:
        if baseline_early_valid is not None:
            incumbent_score = float(baseline_early_valid)
        else:
            incumbent_score = float("inf") if str(metric_mode) == "minimize" else float("-inf")
        best_so_far = {
            "score": float(incumbent_score),
            "builder_id": str(G_REF_ID),
            "loss_id": str(F_REF_ID),
            "stage_final": "baseline",
            "generation": -1,
            "phase": "baseline",
        }
        LOGGER.info(
            "Initialized incumbent from baseline: score=%s pair=(%s,%s) stage=%s gen=%d",
            best_so_far.get("score"),
            best_so_far.get("builder_id"),
            best_so_far.get("loss_id"),
            best_so_far.get("stage_final"),
            int(best_so_far.get("generation", -1)),
        )
        if baseline_early_valid is None:
            LOGGER.info(
                "Baseline incumbent uses sentinel score (no external baseline_early_valid); "
                "will calibrate with evaluated (g_ref,f_ref) when rollout caches are available."
            )

    def _current_best_pair_score_history() -> List[Dict[str, Any]]:
        if not isinstance(best_so_far, dict):
            return []
        key = _pair_history_key(best_so_far.get("builder_id"), best_so_far.get("loss_id"))
        history = pair_score_history_map.get(key, [])
        return [dict(item) for item in history if isinstance(item, Mapping)]

    def _summary_state(last_generation: int) -> Dict[str, Any]:
        best_pair_score_history = _current_best_pair_score_history()
        return {
            "config_path": os.path.abspath(config_path),
            "run_dir": os.path.abspath(run_dir),
            "preset": str(preset),
            "search_mode": str(search_mode),
            "alternating_schedule": {
                "enabled": bool(alternating_schedule_enabled),
                "start_phase": str(alternating_start_phase),
                "loss_generations": int(alternating_loss_generations),
                "builder_generations": int(alternating_builder_generations),
                "final_loss_generations": int(alternating_final_loss_generations),
                "rounds": int(alternating_rounds),
            },
            "metric_mode": str(metric_mode),
            "improve_eps": float(improve_eps),
            "improve_eps_calibration": (dict(improve_eps_calibration) if isinstance(improve_eps_calibration, dict) else None),
            "eval_stages": dict(eval_stages),
            "last_generation": int(last_generation),
            "best_so_far": dict(best_so_far) if isinstance(best_so_far, dict) else None,
            "best_builder_cost": dict(best_builder_cost) if isinstance(best_builder_cost, dict) else None,
            "builder_cost_archive": dict(builder_cost_archive),
            "best_pair_ids": (
                {
                    "builder_id": str(best_so_far.get("builder_id")),
                    "loss_id": str(best_so_far.get("loss_id")),
                }
                if isinstance(best_so_far, dict)
                else None
            ),
            "best_score": (float(best_so_far.get("score")) if isinstance(best_so_far, dict) else None),
            "best_stage_final": (best_so_far.get("stage_final") if isinstance(best_so_far, dict) else None),
            "best_gen": (best_so_far.get("generation") if isinstance(best_so_far, dict) else None),
            "best_phase": (best_so_far.get("phase") if isinstance(best_so_far, dict) else None),
            "best_pair_score_history": best_pair_score_history,
            "best_pair_score_history_summary": _score_history_summary(best_pair_score_history),
        }

    stagnation_generations = int(resume_state.get("stagnation_generations", 0) or 0) if resume_state else 0

    def _checkpoint_state(next_generation: int) -> Dict[str, Any]:
        best_pair_score_history = _current_best_pair_score_history()
        return {
            "config_path": os.path.abspath(config_path),
            "seed": int(seed),
            "next_generation": int(next_generation),
            "preset": str(preset),
            "search_mode": str(search_mode),
            "alternating_schedule": {
                "enabled": bool(alternating_schedule_enabled),
                "start_phase": str(alternating_start_phase),
                "loss_generations": int(alternating_loss_generations),
                "builder_generations": int(alternating_builder_generations),
                "final_loss_generations": int(alternating_final_loss_generations),
                "rounds": int(alternating_rounds),
            },
            "metric_mode": str(metric_mode),
            "improve_eps": float(improve_eps),
            "improve_eps_calibration": (dict(improve_eps_calibration) if isinstance(improve_eps_calibration, dict) else None),
            "eval_stages": dict(eval_stages),
            "best_so_far": dict(best_so_far) if isinstance(best_so_far, dict) else None,
            "best_builder_cost": dict(best_builder_cost) if isinstance(best_builder_cost, dict) else None,
            "builder_cost_archive": dict(builder_cost_archive),
            "best_score": (float(best_so_far.get("score")) if isinstance(best_so_far, dict) else None),
            "best_pair_ids": (
                {
                    "builder_id": str(best_so_far.get("builder_id")),
                    "loss_id": str(best_so_far.get("loss_id")),
                }
                if isinstance(best_so_far, dict)
                else None
            ),
            "best_stage_final": (best_so_far.get("stage_final") if isinstance(best_so_far, dict) else None),
            "best_gen": (best_so_far.get("generation") if isinstance(best_so_far, dict) else None),
            "best_phase": (best_so_far.get("phase") if isinstance(best_so_far, dict) else None),
            "best_pair_score_history": best_pair_score_history,
            "best_pair_score_history_summary": _score_history_summary(best_pair_score_history),
            "pair_score_history_map": dict(pair_score_history_map),
            "rng_state_b64": _b64_pickle(rng.getstate()),
            "seen_g": sorted(seen_g),
            "seen_f": sorted(seen_f),
            "resident_pop_g": list(resident_pop_g),
            "resident_pop_f": list(resident_pop_f),
            "elites_g": list(elites_g),
            "elites_f": list(elites_f),
            "diverse_elites_g": list(diverse_elites_g),
            "diverse_elites_f": list(diverse_elites_f),
            "hof_g": list(hof_g),
            "hof_f": list(hof_f),
            "archive_g": dict(archive_g),
            "archive_f": dict(archive_f),
            "stagnation_generations": int(stagnation_generations),
        }

    _save_checkpoint(run_dir, _checkpoint_state(gen_start))
    _atomic_write_json(summary_json, _summary_state(gen_start - 1))

    llm_feedback_state: Dict[str, Any] = {}
    phase_block_label: str | None = None
    phase_block_best_score: float | None = None
    # Treat baseline as the "last phase" before generation-0 search.
    last_phase_block_label: str | None = None
    last_phase_block_best_score: float | None = None
    if isinstance(best_so_far, dict):
        try:
            last_phase_block_best_score = float(best_so_far.get("score"))
        except (TypeError, ValueError):
            last_phase_block_best_score = None
        last_phase_block_label = str(best_so_far.get("phase") or "baseline")
    baseline_incumbent_calibrated = False

    for gen in range(gen_start, generations):
        (
            alternating_phase_hint,
            alternating_loss_budget_now,
            alternating_builder_budget_now,
            alternating_round_idx,
            alternating_cycle_pos,
            alternating_cycle_len,
        ) = _resolve_alternating_phase_and_budgets(
            search_mode=str(search_mode),
            generation=int(gen),
            pairing_budget=int(pairing_budget),
            pairing_budget_loss=int(pairing_budget_loss),
            pairing_budget_builder=int(pairing_budget_builder),
            alternating_schedule_enabled=bool(alternating_schedule_enabled),
            alternating_loss_generations=int(alternating_loss_generations),
            alternating_builder_generations=int(alternating_builder_generations),
            alternating_rounds=int(alternating_rounds),
            alternating_final_loss_generations=int(alternating_final_loss_generations),
            alternating_start_phase=str(alternating_start_phase),
        )
        generation_phase_label = str(alternating_phase_hint) if _uses_fixed_side_search(search_mode) else "coevo"
        runtime_trace.heartbeat(
            extra={
                "stage": "generation_loop",
                "generation": int(gen),
                "phase": str(generation_phase_label),
                "next_generation": int(gen + 1),
            },
            force=True,
        )
        if phase_block_label is None:
            phase_block_label = str(generation_phase_label)
        elif str(generation_phase_label) != str(phase_block_label):
            last_phase_block_label = str(phase_block_label)
            last_phase_block_best_score = phase_block_best_score
            LOGGER.info(
                "Phase transition detected: prev_phase=%s prev_phase_best=%s -> current_phase=%s",
                str(last_phase_block_label),
                last_phase_block_best_score,
                str(generation_phase_label),
            )
            phase_block_label = str(generation_phase_label)
            phase_block_best_score = None

        builder_llm_enabled_this_gen = bool(builder_cfg.get("enabled", False))
        loss_llm_enabled_this_gen = bool(loss_cfg.get("enabled", False))
        if str(search_mode) == "loss_only":
            builder_llm_enabled_this_gen = False
        elif str(search_mode) == "alternating":
            builder_llm_enabled_this_gen = bool(builder_llm_enabled_this_gen and alternating_phase_hint in {"builder", "mixed"})
            loss_llm_enabled_this_gen = bool(loss_llm_enabled_this_gen and alternating_phase_hint in {"loss", "mixed"})

        llm_cfg_for_gen: Dict[str, Any] | None = None
        if llm_enabled:
            llm_cfg_for_gen = dict(llm_cfg)
            builder_cfg_for_gen = dict(builder_cfg)
            loss_cfg_for_gen = dict(loss_cfg)
            builder_cfg_for_gen["enabled"] = bool(builder_llm_enabled_this_gen)
            loss_cfg_for_gen["enabled"] = bool(loss_llm_enabled_this_gen)
            llm_cfg_for_gen["builder"] = builder_cfg_for_gen
            llm_cfg_for_gen["loss"] = loss_cfg_for_gen
            llm_cfg_for_gen["enabled"] = bool(builder_llm_enabled_this_gen or loss_llm_enabled_this_gen)
            if _uses_fixed_side_search(search_mode):
                LOGGER.info(
                    "%s LLM gating gen=%d phase=%s llm_enabled(builder=%s,loss=%s)",
                    str(search_mode),
                    int(gen),
                    str(alternating_phase_hint),
                    str(builder_llm_enabled_this_gen),
                    str(loss_llm_enabled_this_gen),
                )

        LOGGER.info("=== %s generation %d/%d ===", str(search_mode), gen, generations - 1)
        LOGGER.info("EVAL_STAGES enabled: %s", _format_eval_stages_for_log(eval_stages))
        if isinstance(best_so_far, dict):
            LOGGER.info(
                "INCUMBENT best_score=%s best_pair=(%s,%s) best_stage=%s metric_mode=%s improve_eps=%s",
                best_so_far.get("score"),
                best_so_far.get("builder_id"),
                best_so_far.get("loss_id"),
                best_so_far.get("stage_final"),
                str(metric_mode),
                float(improve_eps),
            )
        else:
            LOGGER.info(
                "INCUMBENT best_score=None best_pair=(None,None) best_stage=none metric_mode=%s improve_eps=%s",
                str(metric_mode),
                float(improve_eps),
            )
        LOGGER.info("Gen %d start: %s %s", int(gen), _cache_brief(caches), _cuda_mem_brief(device_list))

        best_builder_ir = dict(elites_g[0].get("ir")) if elites_g else None
        if isinstance(best_builder_ir, dict) and isinstance(best_builder_ir.get("code"), str):
            best_builder_ir["code"] = _truncate_code(best_builder_ir.get("code", ""))
        best_loss_ir = dict(elites_f[0].get("ir")) if elites_f else None
        if isinstance(best_loss_ir, dict) and isinstance(best_loss_ir.get("code"), str):
            best_loss_ir["code"] = _truncate_code(best_loss_ir.get("code", ""))

        best_builder_summary = summarize_best_builder(elites_g[0]) if elites_g else None
        best_loss_summary = summarize_best_loss(elites_f[0]) if elites_f else None
        if llm_enabled:
            try:
                b_len = len(str((best_builder_summary or {}).get("code", "")))
                f_len = len(str((best_loss_summary or {}).get("code", "")))
            except Exception:  # noqa: BLE001
                b_len = -1
                f_len = -1
            LOGGER.info(
                "Gen %d prompt context sizes: best_builder_summary.code=%d chars best_loss_summary.code=%d chars",
                int(gen),
                int(b_len),
                int(f_len),
            )

        global_feedback: Dict[str, Any] = dict(llm_feedback_state)
        loss_llm_cfg_raw = cfg_yaml.get("loss_llm", {}) or {}
        if not isinstance(loss_llm_cfg_raw, dict):
            loss_llm_cfg_raw = {}
        explore_cfg = loss_llm_cfg_raw.get("exploration", {}) or {}
        if not isinstance(explore_cfg, dict):
            explore_cfg = {}
        explore_enabled = bool(explore_cfg.get("enabled", False))
        stagnation_trigger = int(explore_cfg.get("stagnation_generations", 0) or 0)
        explore_mode = bool(explore_enabled and stagnation_trigger > 0 and int(stagnation_generations) >= int(stagnation_trigger))
        avoid_families: List[str] = []
        try:
            fam_ctr = collections.Counter(str(e.get("family") or "unknown") for e in (elites_f or []) if isinstance(e, dict))
            avoid_families = [f for f, _ in fam_ctr.most_common(8)]
        except Exception:  # noqa: BLE001
            avoid_families = []
        if bool(explore_enabled) and int(stagnation_trigger) > 0:
            LOGGER.info(
                "Loss exploration schedule gen=%d: stagnation=%d trigger=%d explore_mode=%s avoid_families=%s",
                int(gen),
                int(stagnation_generations),
                int(stagnation_trigger),
                str(bool(explore_mode)),
                list(avoid_families),
            )
        global_feedback.update(
            {
                "generation": int(gen),
                "objective": ("lower_is_better" if str(metric_mode) == "minimize" else "higher_is_better"),
                "proxy_problem_size": int(proxy_problem_size),
                "proxy_batch_size": int(proxy_batch_size),
                "proxy_batches": int(proxy_batches),
                "pairing_budget_per_gen": int(pairing_budget),
                "generations": int(generations),
                "operator_whitelist": list(operator_whitelist),
                "builder_gate": dict(llm_cfg.get("builder_gate", {})),
                "best_builder": best_builder_ir,
                "best_loss": best_loss_ir,
                "best_builder_summary": best_builder_summary,
                "best_loss_summary": best_loss_summary,
                "loss_search": {
                    "explore_mode": bool(explore_mode),
                    "stagnation_generations": int(stagnation_generations),
                    "avoid_families": list(avoid_families),
                },
            }
        )

        builder_population_active = not (_uses_fixed_side_search(search_mode) and str(alternating_phase_hint) == "loss")
        loss_population_active = not (_uses_fixed_side_search(search_mode) and str(alternating_phase_hint) == "builder")
        builder_offspring_target = 0
        loss_offspring_target = 0
        if bool(builder_population_active):
            raw = builder_cfg.get("init_llm_g", 0) if int(gen) <= 0 else builder_cfg.get("llm_per_gen_g", 0)
            builder_offspring_target = max(1, int(raw or pop_g))
        if bool(loss_population_active):
            raw = loss_cfg.get("init_llm_f", 0) if int(gen) <= 0 else loss_cfg.get("llm_per_gen_f", 0)
            loss_offspring_target = max(1, int(raw or pop_f))

        llm_init_only = bool(cfg_yaml.get("llm_init_only", False))
        proposed_g = _propose_builders_for_generation(
            generation=int(gen),
            pop_g=int(max(builder_offspring_target, 1)),
            elites_g=resident_pop_g,
            diverse_elites_g=[],
            rng=rng,
            llm_cfg=llm_cfg_for_gen if llm_enabled else None,
            operator_whitelist=operator_whitelist,
            global_feedback=global_feedback if llm_enabled else None,
            llm_init_only=bool(llm_init_only),
            carry_elites=False,
        )
        proposed_f = _propose_losses_for_generation(
            generation=int(gen),
            pop_f=int(max(loss_offspring_target, 1)),
            elites_f=resident_pop_f,
            diverse_elites_f=[],
            rng=rng,
            llm_cfg=llm_cfg_for_gen if llm_enabled else None,
            operator_whitelist=operator_whitelist,
            global_feedback=global_feedback if llm_enabled else None,
            llm_init_only=bool(llm_init_only),
            carry_elites=False,
        )

        # Ensure generation-0 default pair/loss match PO4COPs-style baseline
        # before search-driven variants are considered.
        if int(gen) == 0 and bool(cfg_yaml.get("seed_with_po4cops_default", True)) and (not bool(llm_init_only)):
            proposed_g = [
                {
                    "ir": _ref_builder_ir(),
                    "origin": "SEED_PO4COPS_DEFAULT",
                    "op_type": "SEED_PO4COPS_DEFAULT",
                    "parents": [],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                }
            ] + list(proposed_g)
            proposed_f = [
                {
                    "ir": _ref_loss_ir(),
                    "origin": "SEED_PO4COPS_DEFAULT",
                    "op_type": "SEED_PO4COPS_DEFAULT",
                    "parents": [],
                    "attempt": 0,
                    "prompt_sha1": None,
                    "prompt_path": None,
                    "history": [],
                }
            ] + list(proposed_f)
            LOGGER.info("Gen %d injected PO4COPs-compatible default builder/loss seeds.", int(gen))

        # Dedupe for novelty across resume + prior generations.
        def _fill_unique_builders(
            proposals: Sequence[Mapping[str, Any]],
            *,
            target_size: int,
            resident_entries: Sequence[Mapping[str, Any]],
        ) -> List[Dict[str, Any]]:
            unique: List[Dict[str, Any]] = []
            resident_sigs = {
                str(item.get("signature"))
                for item in resident_entries
                if isinstance(item, Mapping) and item.get("signature")
            }
            current_sigs = set(resident_sigs) | set(seen_g)
            attempts = 0
            for p in proposals:
                if not isinstance(p, dict) or not isinstance(p.get("ir"), PreferenceBuilderIR):
                    continue
                ir = p["ir"]
                sig = _sig_pref_builder(ir)
                if sig in current_sigs:
                    continue
                unique.append(dict(p))
                current_sigs.add(sig)
            while (not bool(llm_init_only)) and len(unique) < int(target_size) and attempts < int(max(target_size, 1)) * 20:
                attempts += 1
                ir = _make_builtin_builder_irs(rng, 1)[0]
                sig = _sig_pref_builder(ir)
                if sig in current_sigs:
                    continue
                unique.append(
                    {
                        "ir": ir,
                        "origin": "SEED",
                        "op_type": "SEED_FILL",
                        "parents": [],
                        "attempt": 0,
                        "prompt_sha1": None,
                        "prompt_path": None,
                        "history": [],
                    }
                )
                current_sigs.add(sig)
            return unique[: int(target_size)]

        def _fill_unique_losses(
            proposals: Sequence[Mapping[str, Any]],
            *,
            target_size: int,
            resident_entries: Sequence[Mapping[str, Any]],
        ) -> List[Dict[str, Any]]:
            unique2: List[Dict[str, Any]] = []
            resident_sigs = {
                str(item.get("signature"))
                for item in resident_entries
                if isinstance(item, Mapping) and item.get("signature")
            }
            current_sigs = set(resident_sigs) | set(seen_f)
            attempts2 = 0
            for p in proposals:
                if not isinstance(p, dict) or not isinstance(p.get("ir"), FreeLossIR):
                    continue
                ir = p["ir"]
                sig = _sig_free_loss(ir)
                if sig in current_sigs:
                    continue
                unique2.append(dict(p))
                current_sigs.add(sig)
            while (not bool(llm_init_only)) and len(unique2) < int(target_size) and attempts2 < int(max(target_size, 1)) * 20:
                attempts2 += 1
                ir = _make_builtin_loss_irs(rng, 1)[0]
                sig = _sig_free_loss(ir)
                if sig in current_sigs:
                    continue
                unique2.append(
                    {
                        "ir": ir,
                        "origin": "SEED",
                        "op_type": "SEED_FILL",
                        "parents": [],
                        "attempt": 0,
                        "prompt_sha1": None,
                        "prompt_path": None,
                        "history": [],
                    }
                )
                current_sigs.add(sig)
            return unique2[: int(target_size)]

        proposed_g = _fill_unique_builders(
            proposed_g,
            target_size=int(builder_offspring_target),
            resident_entries=resident_pop_g,
        )
        proposed_f = _fill_unique_losses(
            proposed_f,
            target_size=int(loss_offspring_target),
            resident_entries=resident_pop_f,
        )
        if len(proposed_g) < int(builder_offspring_target) or len(proposed_f) < int(loss_offspring_target):
            LOGGER.warning(
                "Offspring fill shortfall at gen=%d: proposed_g=%d/%d proposed_f=%d/%d resident_g=%d resident_f=%d",
                int(gen),
                int(len(proposed_g)),
                int(builder_offspring_target),
                int(len(proposed_f)),
                int(loss_offspring_target),
                int(len(resident_pop_g)),
                int(len(resident_pop_f)),
            )

        g_entries: List[Dict[str, Any]] = []
        for idx, proposal in enumerate(proposed_g):
            ir: PreferenceBuilderIR = proposal["ir"]
            sig = _sig_pref_builder(ir)
            family_signature = _builder_family_signature(ir)
            entry: Dict[str, Any] = {
                "generation": int(gen),
                "index": int(idx),
                "id": f"g{gen:03d}_{idx:03d}_{sig[:8]}",
                "signature": sig,
                "family_signature": str(family_signature),
                "origin": str(proposal.get("origin", "unknown")),
                "origin_base": proposal.get("origin_base"),
                "op_type": proposal.get("op_type"),
                "parents": list(proposal.get("parents", [])),
                "attempt": proposal.get("attempt", 0),
                "prompt_sha1": proposal.get("prompt_sha1"),
                "prompt_path": proposal.get("prompt_path"),
                "llm_seed": proposal.get("llm_seed"),
                "history": list(proposal.get("history", [])) if isinstance(proposal.get("history", []), list) else [],
                "ir": asdict(ir),
            }
            try:
                compiled = compile_preference_builder(ir, operator_whitelist=operator_whitelist)
                entry["compile_ok"] = True
                entry["compile_reason"] = "ok"

                # Builder "static" gates (cheap): run on a deterministic dummy feature_cache.
                fc = _dummy_feature_cache(batch_size=8, k=16, variant="visible")
                pb = _build_pref_batch_with_memory_trace(compiled, fc, {"stage": "builder_static"})
                bg = run_preference_builder_gates(
                    pb,
                    feature_cache=fc,
                    min_pairs=int(cfg_yaml.get("builder_min_pairs", 1) or 1),
                    min_coverage=float(cfg_yaml.get("builder_min_coverage", 0.0) or 0.0),
                    max_pairs_per_instance=int(cfg_yaml.get("builder_max_pairs_per_instance", 4096) or 4096),
                    weight_nonneg=bool(cfg_yaml.get("builder_weight_nonneg", True)),
                    semantic_tolerance=float(cfg_yaml.get("builder_semantic_tolerance", 0.0) or 0.0),
                    semantic_min_pass_rate=float(cfg_yaml.get("builder_semantic_min_pass_rate", 1.0) or 1.0),
                )
                bg.trace = _enrich_builder_gate_trace_with_memory(bg.trace, pb, cache_hit=False)
                entry["builder_static_ok"] = bool(bg.ok)
                entry["builder_static_reason"] = str(bg.reason)
                entry["builder_static_trace"] = bg.trace
            except Exception as exc:  # noqa: BLE001
                entry["compile_ok"] = False
                entry["compile_reason"] = str(exc)
                entry["builder_static_ok"] = False
                entry["builder_static_reason"] = "compile_or_static_failed"
            g_entries.append(entry)
        _append_jsonl(builders_jsonl, g_entries)

        f_entries: List[Dict[str, Any]] = []
        for idx, proposal in enumerate(proposed_f):
            ir: FreeLossIR = proposal["ir"]
            sig = _sig_free_loss(ir)
            static_res = run_static_gates(ir, operator_whitelist=operator_whitelist)
            family = _loss_family_id(ir)
            family_signature = _loss_family_signature(ir)
            entry: Dict[str, Any] = {
                "generation": int(gen),
                "index": int(idx),
                "id": f"f{gen:03d}_{idx:03d}_{sig[:8]}",
                "signature": sig,
                "family": str(family),
                "family_signature": str(family_signature),
                "origin": str(proposal.get("origin", "unknown")),
                "origin_base": proposal.get("origin_base"),
                "op_type": proposal.get("op_type"),
                "parents": list(proposal.get("parents", [])),
                "attempt": proposal.get("attempt", 0),
                "prompt_sha1": proposal.get("prompt_sha1"),
                "prompt_path": proposal.get("prompt_path"),
                "llm_seed": proposal.get("llm_seed"),
                "history": list(proposal.get("history", [])) if isinstance(proposal.get("history", []), list) else [],
                "novelty": proposal.get("novelty"),
                "ir": asdict(ir),
                "static_ok": bool(static_res.ok),
                "static_reason": str(static_res.reason),
                "static_trace": static_res.trace,
            }
            if static_res.ok:
                try:
                    _ = compile_free_loss(ir, operator_whitelist=operator_whitelist)
                    entry["compile_ok"] = True
                    entry["compile_reason"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    entry["compile_ok"] = False
                    entry["compile_reason"] = str(exc)
            else:
                entry["compile_ok"] = False
                entry["compile_reason"] = "static_gate_failed"
            f_entries.append(entry)
        _append_jsonl(losses_jsonl, f_entries)
        for entry in g_entries:
            sig = entry.get("signature")
            if sig:
                seen_g.add(str(sig))
        for entry in f_entries:
            sig = entry.get("signature")
            if sig:
                seen_f.add(str(sig))

        if llm_enabled:
            g_llm_ops = collections.Counter()
            f_llm_ops = collections.Counter()
            g_repairs = 0
            f_repairs = 0
            for e in g_entries:
                if str(e.get("origin")) == "REPAIR":
                    g_repairs += 1
                hist = e.get("history")
                if isinstance(hist, list):
                    for h in hist:
                        if not isinstance(h, dict):
                            continue
                        op = h.get("llm_op") or h.get("op")
                        if op:
                            g_llm_ops[str(op)] += 1
            for e in f_entries:
                if str(e.get("origin")) == "REPAIR":
                    f_repairs += 1
                hist = e.get("history")
                if isinstance(hist, list):
                    for h in hist:
                        if not isinstance(h, dict):
                            continue
                        op = h.get("llm_op") or h.get("op")
                        if op:
                            f_llm_ops[str(op)] += 1
            try:
                cache_stats = dict(loss_llm_ops.llm_cache_stats())
            except Exception:  # noqa: BLE001
                cache_stats = {}
            LOGGER.info(
                "Gen %d LLM ops: builders=%s (repairs=%d) losses=%s (repairs=%d) cache=%s",
                int(gen),
                dict(g_llm_ops),
                int(g_repairs),
                dict(f_llm_ops),
                int(f_repairs),
                cache_stats,
            )

        g_pool = [e for e in g_entries if bool(e.get("compile_ok")) and bool(e.get("builder_static_ok", True))]
        f_pool = [e for e in f_entries if bool(e.get("compile_ok"))]

        resident_g_ids = [str(e["id"]) for e in resident_pop_g if isinstance(e, dict) and "id" in e]
        resident_f_ids = [str(e["id"]) for e in resident_pop_f if isinstance(e, dict) and "id" in e]
        g_id_pool = list(dict.fromkeys(resident_g_ids + [str(e["id"]) for e in g_pool]))
        f_id_pool = list(dict.fromkeys(resident_f_ids + [str(e["id"]) for e in f_pool]))

        g_population_map = {
            str(e["id"]): e
            for e in [e for e in resident_pop_g if isinstance(e, dict) and "id" in e] + g_pool
        }
        f_population_map = {
            str(e["id"]): e
            for e in [e for e in resident_pop_f if isinstance(e, dict) and "id" in e] + f_pool
        }
        g_map = dict(g_population_map)
        f_map = dict(f_population_map)

        # "New" candidates for this generation (subject to anchor filtering below).
        new_g_ids = [str(e["id"]) for e in g_pool]
        new_f_ids = [str(e["id"]) for e in f_pool]
        pairs: List[Tuple[str, str]] = []

        # === Multi-fidelity evaluation with caching ===
        # Cheap stage: reuse cached rollout feature_cache across all pairs; build PrefBatch once per (g_id, batch_id).
        stage0_gate_enabled = bool(eval_stages.get("stage0_gate", True))
        stage1_proxy_enabled = bool(eval_stages.get("stage1_proxy", True))
        stage2_micro_enabled = bool(eval_stages.get("stage2_micro_unroll", False))
        stage3_hf_enabled = bool(eval_stages.get("stage3_high_fidelity", True))
        need_rollout_caches = bool(stage1_proxy_enabled or stage2_micro_enabled)

        proxy_device_str = _normalize_device_alias(str(cfg_yaml.get("proxy_device", device_list[0])))
        if proxy_device_str == "cuda" and not torch.cuda.is_available():
            proxy_device_str = "cpu"
        proxy_device = torch.device(proxy_device_str)

        # Pre-build (or reuse) rollout feature caches when proxy/micro stages are enabled.
        progress_every_proxy_batches = int(cfg_yaml.get("progress_log_every_proxy_batches", 10) or 10)
        rollout_feature_caches: List[Dict[str, torch.Tensor]] = []
        if need_rollout_caches:
            t_rollouts0 = time.time()
            LOGGER.info(
                "Gen %d proxy rollouts: device=%s batches=%d batch_size=%d problem_size=%d",
                int(gen),
                str(proxy_device_str),
                int(proxy_batches),
                int(proxy_batch_size),
                int(proxy_problem_size),
            )
            for batch_id in range(int(proxy_batches)):
                key = (int(seed), int(proxy_problem_size), int(batch_id))
                hit = caches.get_rollout(key) is not None
                fc = build_or_get_rollout_feature_cache(
                    caches=caches,
                    cfg=_build_hf_cfg(cfg_yaml, seed=seed, device_str=proxy_device_str),
                    seed=int(seed),
                    problem_size=int(proxy_problem_size),
                    batch_id=int(batch_id),
                    batch_size=int(proxy_batch_size),
                    device=proxy_device,
                )
                rollout_feature_caches.append(fc)
                if (batch_id + 1) in (1, int(proxy_batches)) or (
                    progress_every_proxy_batches > 0 and ((batch_id + 1) % progress_every_proxy_batches == 0)
                ):
                    LOGGER.info(
                        "Gen %d proxy rollouts progress: %d/%d cache_hit=%s %s %s",
                        int(gen),
                        int(batch_id + 1),
                        int(proxy_batches),
                        str(hit),
                        _cache_brief(caches),
                        _cuda_mem_brief([proxy_device_str]),
                    )
            LOGGER.info(
                "Gen %d proxy rollouts done: elapsed_s=%.1f %s %s",
                int(gen),
                float(time.time() - t_rollouts0),
                _cache_brief(caches),
                _cuda_mem_brief([proxy_device_str]),
            )
            _maybe_auto_flush_pref_cache(
                caches=caches,
                cfg_yaml=cfg_yaml,
                default_device_str=proxy_device_str,
                generation=int(gen),
                scope="after_proxy_rollouts",
            )
        else:
            LOGGER.info("Gen %d proxy rollouts skipped: stage1_proxy=0 and stage2_micro_unroll=0", int(gen))

        # Compile pools locally for proxy evaluation (avoids mp pickling issues).
        compiled_g: Dict[str, CompiledPreferenceBuilder] = {}
        compiled_f: Dict[str, CompiledFreeLoss] = {}
        for gid in g_id_pool:
            if gid not in g_map:
                continue
            try:
                compiled_g[gid] = compile_preference_builder(
                    pref_builder_ir_from_json(g_map[gid]["ir"]),
                    operator_whitelist=operator_whitelist,
                )
            except Exception:  # noqa: BLE001
                continue
        for fid in f_id_pool:
            if fid not in f_map:
                continue
            try:
                compiled_f[fid] = compile_free_loss(
                    free_loss_ir_from_json(f_map[fid]["ir"]),
                    operator_whitelist=operator_whitelist,
                )
            except Exception:  # noqa: BLE001
                continue

        joint_gate_kwargs = {
            "min_pass_rate": float(cfg_yaml.get("proxy_joint_min_pass_rate", 0.8) or 0.8),
            "swap_tolerance": float(cfg_yaml.get("proxy_joint_swap_tolerance", 1e-3) or 1e-3),
            "swap_check_mode": str(cfg_yaml.get("proxy_joint_swap_check_mode", "data") or "data"),
            "swap_test_margin": float(cfg_yaml.get("proxy_joint_swap_test_margin", 1.0) or 1.0),
            "grad_eps": float(cfg_yaml.get("proxy_joint_grad_eps", 1e-8) or 1e-8),
            "min_effective_grad_ratio": float(cfg_yaml.get("proxy_joint_min_effective_grad_ratio", 0.1) or 0.1),
            "variant": "visible",
        }

        gate_repair_cfg_raw = cfg_yaml.get("proxy_gate_repair", {})
        gate_repair_cfg = gate_repair_cfg_raw if isinstance(gate_repair_cfg_raw, dict) else {}
        gate_repair_enabled = bool(
            (builder_llm_enabled_this_gen or loss_llm_enabled_this_gen)
            and stage1_proxy_enabled
            and bool(gate_repair_cfg.get("enabled", False))
        )
        gate_repair_remaining = max(0, int(gate_repair_cfg.get("max_repairs_per_gen", 0) or 0))
        gate_repair_attempts = max(0, int(gate_repair_cfg.get("max_attempts_per_pair", 1) or 1))
        gate_repair_builder_on_fail = bool(
            gate_repair_cfg.get("repair_builder_on_builder_gate_fail", True) and builder_llm_enabled_this_gen
        )
        gate_repair_loss_on_fail = bool(gate_repair_cfg.get("repair_loss_on_joint_gate_fail", True) and loss_llm_enabled_this_gen)
        gate_repair_attempted_pairs: set[tuple[str, str]] = set()
        gate_repair_attempted_builders: set[str] = set()
        gate_repair_attempted_losses: set[str] = set()

        llm_prompts_cfg = llm_cfg.get("prompts", {}) if isinstance(llm_cfg.get("prompts"), dict) else {}
        p_builder_rep = str(llm_prompts_cfg.get("builder_repair", "") or "")
        p_builder_m3 = str(llm_prompts_cfg.get("builder_m3", "") or "")
        p_loss_rep = str(llm_prompts_cfg.get("loss_repair", "") or "")
        p_loss_runtime_rep = str(
            cfg_yaml.get("joint_gate_repair_prompt_path", "PTP/prompts/free_loss_forward_error_repair.txt")
            or "PTP/prompts/free_loss_forward_error_repair.txt"
        )
        p_loss_m3 = str(llm_prompts_cfg.get("loss_m3", "") or "")
        builder_repair_live_cfg = builder_cfg.get("repair", {}) if isinstance(builder_cfg.get("repair"), dict) else {}
        loss_repair_live_cfg = loss_cfg.get("repair", {}) if isinstance(loss_cfg.get("repair"), dict) else {}
        builder_gate_live_cfg = llm_cfg.get("builder_gate", {}) if isinstance(llm_cfg.get("builder_gate"), dict) else {}
        runtime_repair_failure_kinds = {
            "forward_error",
            "backward_error",
            "pref_batch_to_loss_batch_error",
            "loss_not_finite",
            "grad_not_finite",
            "missing_grads",
            "numeric_stress_forward_error",
            "numeric_stress_backward_error",
            "numeric_stress_loss_not_finite",
            "numeric_stress_grad_not_finite",
            "numeric_stress_missing_grads",
            "sandbox_builder_gate_failed",
            "sandbox_runtime_error",
            "sandbox_timeout",
            "sandbox_no_result",
            "sandbox_result_invalid",
            "sandbox_gate_failed",
            "compile_error",
        }

        if gate_repair_enabled:
            LOGGER.info(
                "Proxy gate-repair enabled: max_repairs_per_gen=%d max_attempts_per_pair=%d builder_on_fail=%s loss_on_fail=%s",
                int(gate_repair_remaining),
                int(gate_repair_attempts),
                str(gate_repair_builder_on_fail),
                str(gate_repair_loss_on_fail),
            )

        bins = int(cfg_yaml.get("archive_bins", 8) or 8)
        pair_count_cap = int(cfg_yaml.get("descriptor_pair_count_cap", 4096) or 4096)
        loss_scale = float(cfg_yaml.get("descriptor_loss_scale", 5.0) or 5.0)

        if bool(stage1_proxy_enabled) and bool(cfg_yaml.get("anchor_enabled", True)):
            _ensure_reference_compiled(compiled_g=compiled_g, compiled_f=compiled_f, operator_whitelist=operator_whitelist)
        hof_g_ids, hof_f_ids = _compile_hof_candidates(
            hof_g=hof_g,
            hof_f=hof_f,
            compiled_g=compiled_g,
            compiled_f=compiled_f,
            operator_whitelist=operator_whitelist,
        )

        base_seed = int(seed)
        base_seed_sig = seed_signature_for_proxy(
            seed=base_seed,
            problem_size=int(proxy_problem_size),
            batch_ids=list(range(int(proxy_batches))),
            batch_size=int(proxy_batch_size),
        )

        if not baseline_incumbent_calibrated:
            need_calibrate = False
            if isinstance(best_so_far, dict) and str(best_so_far.get("stage_final")) == "baseline":
                try:
                    bs = float(best_so_far.get("score"))
                except (TypeError, ValueError):
                    bs = float("inf") if str(metric_mode) == "minimize" else float("-inf")
                need_calibrate = not math.isfinite(bs)
            if need_calibrate and need_rollout_caches and rollout_feature_caches:
                try:
                    _ensure_reference_compiled(
                        compiled_g=compiled_g,
                        compiled_f=compiled_f,
                        operator_whitelist=operator_whitelist,
                    )
                    rec_base = _cheap_eval_pair_cached(
                        caches=caches,
                        compiled_g=compiled_g,
                        compiled_f=compiled_f,
                        rollout_feature_caches=rollout_feature_caches,
                        cfg_yaml=cfg_yaml,
                        eval_sig=str(eval_sig),
                        gid=str(G_REF_ID),
                        fid=str(F_REF_ID),
                        generation=-1,
                        pair_index=-1,
                        seed_used=int(base_seed),
                        seed_sig=str(base_seed_sig),
                        pref_batch_id_offset=0,
                        stage="baseline",
                        reasons=["baseline_ref_pair"],
                        proxy_device_str=proxy_device_str,
                        joint_gate_kwargs=joint_gate_kwargs,
                        proxy_weights=dict(proxy_weights),
                        bins=bins,
                        pair_count_cap=pair_count_cap,
                        loss_scale=loss_scale,
                        cheap_gate_on=bool(stage0_gate_enabled),
                    )
                    base_score = float(rec_base.get("score", float("inf")))
                    base_stage = "proxy"
                    if bool(stage2_micro_enabled):
                        g_ref_comp = compiled_g.get(str(G_REF_ID))
                        f_ref_comp = compiled_f.get(str(F_REF_ID))
                        if g_ref_comp is not None and f_ref_comp is not None:
                            micro_steps = int(cfg_yaml.get("micro_unroll_steps", 3) or 3)
                            micro_lr = float(cfg_yaml.get("micro_unroll_lr", 5e-2) or 5e-2)
                            micro_alpha = _alpha_from_cfg(cfg_yaml, key="micro_unroll_alpha")
                            micro_weight_decay = float(cfg_yaml.get("micro_unroll_weight_decay", 0.0) or 0.0)
                            micro_reuse_pref = bool(cfg_yaml.get("micro_unroll_reuse_pref_batch_when_safe", True))
                            micro_max_pairs = cfg_yaml.get("micro_unroll_max_pairs", None)
                            try:
                                micro_max_pairs_i = int(micro_max_pairs) if micro_max_pairs is not None else None
                            except (TypeError, ValueError):
                                micro_max_pairs_i = None
                            micro_timeout_s = cfg_yaml.get("micro_unroll_timeout_s", None)
                            try:
                                micro_timeout_s_f = float(micro_timeout_s) if micro_timeout_s is not None else None
                            except (TypeError, ValueError):
                                micro_timeout_s_f = None
                            mu_score, _mu_metrics = micro_unroll_score_for_pair(
                                g=g_ref_comp,
                                f=f_ref_comp,
                                rollout_feature_caches=rollout_feature_caches,
                                steps=int(micro_steps),
                                lr=float(micro_lr),
                                alpha=float(micro_alpha),
                                weight_decay=float(micro_weight_decay),
                                reuse_pref_batch_when_safe=bool(micro_reuse_pref),
                                max_pairs=micro_max_pairs_i,
                                timeout_s=micro_timeout_s_f,
                            )
                            base_score = float(mu_score)
                            base_stage = "micro_unroll"
                    if math.isfinite(base_score):
                        best_so_far = {
                            "score": float(base_score),
                            "builder_id": str(G_REF_ID),
                            "loss_id": str(F_REF_ID),
                            "stage_final": str(base_stage),
                            "generation": -1,
                            "phase": "baseline",
                        }
                        LOGGER.info(
                            "Calibrated incumbent from evaluated baseline pair: score=%s stage=%s pair=(%s,%s)",
                            float(base_score),
                            str(base_stage),
                            str(G_REF_ID),
                            str(F_REF_ID),
                        )
                        # Keep gen0 last-phase reference aligned with calibrated baseline.
                        if str(last_phase_block_label) == "baseline":
                            last_phase_block_best_score = float(base_score)
                        baseline_incumbent_calibrated = True
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to calibrate baseline incumbent from ref pair: %s", str(exc))

        # Anchor evaluation for anti-collapse (only coevo + proxy stage).
        anchor_enabled = bool(cfg_yaml.get("anchor_enabled", True)) and bool(stage1_proxy_enabled) and str(search_mode) == "coevo"
        anchor_max_score = float(cfg_yaml.get("anchor_proxy_max_score", 10.0) or 10.0)
        eliminated_g: set[str] = set()
        eliminated_f: set[str] = set()
        pair_phase_by_pair: Dict[Tuple[str, str], str] = {}

        anchor_pairs: List[Tuple[str, str]] = []
        anchor_reasons: Dict[Tuple[str, str], List[str]] = {}
        if anchor_enabled:
            for gid in list(new_g_ids):
                anchor_pairs.append((str(gid), F_REF_ID))
                anchor_reasons[(str(gid), F_REF_ID)] = ["anchor_builder_vs_f_ref"]
                pair_phase_by_pair[(str(gid), F_REF_ID)] = "coevo"
            for fid in list(new_f_ids):
                anchor_pairs.append((G_REF_ID, str(fid)))
                anchor_reasons[(G_REF_ID, str(fid))] = ["anchor_loss_vs_g_ref"]
                pair_phase_by_pair[(G_REF_ID, str(fid))] = "coevo"

        anchor_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        t_anchor0 = time.time()
        progress_every_pairs = int(cfg_yaml.get("progress_log_every_pairs", 10) or 10)
        for p_idx, (gid, fid) in enumerate(anchor_pairs):
            rec = _cheap_eval_pair_cached(
                caches=caches,
                compiled_g=compiled_g,
                compiled_f=compiled_f,
                rollout_feature_caches=rollout_feature_caches,
                cfg_yaml=cfg_yaml,
                eval_sig=str(eval_sig),
                gid=str(gid),
                fid=str(fid),
                generation=int(gen),
                pair_index=int(p_idx),
                seed_used=int(base_seed),
                seed_sig=str(base_seed_sig),
                pref_batch_id_offset=0,
                stage="anchor",
                reasons=anchor_reasons.get((gid, fid), ["anchor"]),
                proxy_device_str=proxy_device_str,
                joint_gate_kwargs=joint_gate_kwargs,
                proxy_weights=dict(proxy_weights),
                bins=bins,
                pair_count_cap=pair_count_cap,
                loss_scale=loss_scale,
                cheap_gate_on=bool(stage0_gate_enabled),
            )
            anchor_records[(str(gid), str(fid))] = rec
            if progress_every_pairs > 0 and ((p_idx + 1) in (1, int(len(anchor_pairs))) or ((p_idx + 1) % progress_every_pairs == 0)):
                LOGGER.info(
                    "Anchor eval progress gen=%d: %d/%d %s %s",
                    int(gen),
                    int(p_idx + 1),
                    int(len(anchor_pairs)),
                    _cache_brief(caches),
                    _cuda_mem_brief([proxy_device_str]),
                )
            if str(fid) == F_REF_ID and str(gid) in new_g_ids:
                if (not bool(rec.get("pair_ok"))) or float(rec.get("score", float("inf"))) > anchor_max_score:
                    eliminated_g.add(str(gid))
            if str(gid) == G_REF_ID and str(fid) in new_f_ids:
                if (not bool(rec.get("pair_ok"))) or float(rec.get("score", float("inf"))) > anchor_max_score:
                    eliminated_f.add(str(fid))
        if anchor_pairs:
            LOGGER.info(
                "Anchor eval done gen=%d: pairs=%d elapsed_s=%.1f %s %s",
                int(gen),
                int(len(anchor_pairs)),
                float(time.time() - t_anchor0),
                _cache_brief(caches),
                _cuda_mem_brief([proxy_device_str]),
            )
            _maybe_auto_flush_pref_cache(
                caches=caches,
                cfg_yaml=cfg_yaml,
                default_device_str=proxy_device_str,
                generation=int(gen),
                scope="after_anchor",
            )

        if eliminated_g:
            new_g_ids = [x for x in new_g_ids if x not in eliminated_g]
            g_pool = [e for e in g_pool if str(e.get("id")) not in eliminated_g]
        if eliminated_f:
            new_f_ids = [x for x in new_f_ids if x not in eliminated_f]
            f_pool = [e for e in f_pool if str(e.get("id")) not in eliminated_f]
        if anchor_enabled:
            LOGGER.info(
                "Anchor filter gen=%d: eliminated_g=%d eliminated_f=%d (anchor_max_score=%.3f)",
                int(gen),
                int(len(eliminated_g)),
                int(len(eliminated_f)),
                float(anchor_max_score),
            )

        # Rebuild candidate pools with HoF after anchor filtering.
        resident_g_ids = [str(e["id"]) for e in resident_pop_g if isinstance(e, dict) and "id" in e]
        resident_f_ids = [str(e["id"]) for e in resident_pop_f if isinstance(e, dict) and "id" in e]
        g_id_pool = list(dict.fromkeys(resident_g_ids + [str(e["id"]) for e in g_pool] + list(hof_g_ids)))
        f_id_pool = list(dict.fromkeys(resident_f_ids + [str(e["id"]) for e in f_pool] + list(hof_f_ids)))

        # Rebuild maps to include HoF entries for high-fidelity cross-play.
        g_population_map = {
            str(e["id"]): e
            for e in [e for e in resident_pop_g if isinstance(e, dict) and "id" in e] + g_pool
        }
        f_population_map = {
            str(e["id"]): e
            for e in [e for e in resident_pop_f if isinstance(e, dict) and "id" in e] + f_pool
        }
        g_map = dict(g_population_map)
        f_map = dict(f_population_map)
        for e in hof_g:
            if isinstance(e, dict) and e.get("id") and isinstance(e.get("ir"), dict):
                g_map.setdefault(str(e["id"]), dict(e))
        for e in hof_f:
            if isinstance(e, dict) and e.get("id") and isinstance(e.get("ir"), dict):
                f_map.setdefault(str(e["id"]), dict(e))

        pairs: List[Tuple[str, str]] = []
        reasons_by_pair: Dict[Tuple[str, str], List[str]] = {}
        alternating_active_phase = "none"
        alternating_fixed_builder_id: str | None = None
        alternating_fixed_loss_id: str | None = None
        if _uses_fixed_side_search(search_mode):
            loss_budget_now = int(alternating_loss_budget_now)
            builder_budget_now = int(alternating_builder_budget_now)
            if str(search_mode) == "alternating" and alternating_schedule_enabled:
                LOGGER.info(
                    "Alternating block status: round=%d cycle_pos=%d/%d budgets(loss=%d,builder=%d)",
                    int(alternating_round_idx),
                    int(alternating_cycle_pos),
                    int(alternating_cycle_len),
                    int(loss_budget_now),
                    int(builder_budget_now),
                )

            fixed_builder_id = None
            fixed_loss_id = None

            try:
                _ensure_reference_compiled(
                    compiled_g=compiled_g,
                    compiled_f=compiled_f,
                    operator_whitelist=operator_whitelist,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Failed to ensure reference g_ref/f_ref for alternating fallback: %s",
                    str(exc),
                )

            # Fixed builder for loss-search: prefer best_builder_cost, then resident-pop leader, then global incumbent, then g_ref.
            if (not fixed_builder_id) and isinstance(best_builder_cost, dict):
                cand_g = str(best_builder_cost.get("builder_id") or "")
                if cand_g and cand_g in compiled_g:
                    fixed_builder_id = cand_g
            if (not fixed_builder_id) and resident_pop_g:
                cand_g = str(resident_pop_g[0].get("id") or "")
                if cand_g and cand_g in compiled_g:
                    fixed_builder_id = cand_g
            if (not fixed_builder_id) and isinstance(best_so_far, dict):
                cand_g = str(best_so_far.get("builder_id") or "")
                if cand_g and cand_g in compiled_g:
                    fixed_builder_id = cand_g
            if not fixed_builder_id and G_REF_ID in compiled_g:
                fixed_builder_id = str(G_REF_ID)
            if (not fixed_builder_id) and g_id_pool:
                for gid0 in g_id_pool:
                    if str(gid0) in compiled_g:
                        fixed_builder_id = str(gid0)
                        break

            # Fixed loss for builder-search: prefer current best loss incumbent; fallback to resident-pop leader then f_ref.
            if (not fixed_loss_id) and isinstance(best_so_far, dict):
                cand_f = str(best_so_far.get("loss_id") or "")
                if cand_f and cand_f in compiled_f:
                    fixed_loss_id = cand_f
            if (not fixed_loss_id) and resident_pop_f:
                cand_f = str(resident_pop_f[0].get("id") or "")
                if cand_f and cand_f in compiled_f:
                    fixed_loss_id = cand_f
            if not fixed_loss_id and F_REF_ID in compiled_f:
                fixed_loss_id = str(F_REF_ID)
            if (not fixed_loss_id) and f_id_pool:
                for fid0 in f_id_pool:
                    if str(fid0) in compiled_f:
                        fixed_loss_id = str(fid0)
                        break

            LOGGER.info(
                "Alternating fixed candidates resolved: fixed_g=%s fixed_f=%s",
                str(fixed_builder_id),
                str(fixed_loss_id),
            )
            if int(loss_budget_now) > 0 and int(builder_budget_now) == 0:
                alternating_active_phase = "loss"
                alternating_fixed_builder_id = str(fixed_builder_id) if fixed_builder_id else None
            elif int(builder_budget_now) > 0 and int(loss_budget_now) == 0:
                alternating_active_phase = "builder"
                alternating_fixed_loss_id = str(fixed_loss_id) if fixed_loss_id else None
            elif int(loss_budget_now) > 0 and int(builder_budget_now) > 0:
                alternating_active_phase = "mixed"

            if fixed_builder_id and fixed_loss_id:
                LOGGER.info(
                    "Alternating phase=loss fixed_g=%s budget=%d",
                    str(fixed_builder_id),
                    int(loss_budget_now),
                )
                LOGGER.info("EVAL_STAGES enabled: %s", _format_eval_stages_for_log(eval_stages))
                LOGGER.info(
                    "Alternating phase=builder fixed_f=%s budget=%d",
                    str(fixed_loss_id),
                    int(builder_budget_now),
                )
                LOGGER.info("EVAL_STAGES enabled: %s", _format_eval_stages_for_log(eval_stages))
                pairs, reasons_by_pair, pair_phase_by_pair = _build_alternating_pairs(
                    rng=rng,
                    g_id_pool=list(g_id_pool),
                    f_id_pool=list(f_id_pool),
                    fixed_builder_id=str(fixed_builder_id),
                    fixed_loss_id=str(fixed_loss_id),
                    budget_loss=int(loss_budget_now),
                    budget_builder=int(builder_budget_now),
                )
                if alternating_active_phase == "loss" and alternating_fixed_builder_id:
                    before_n = len(pairs)
                    pairs = [p for p in pairs if str(p[0]) == str(alternating_fixed_builder_id)]
                    if len(pairs) != before_n:
                        LOGGER.warning(
                            "Alternating loss-phase pair cleanup gen=%d: dropped %d pairs not using fixed_g=%s",
                            int(gen),
                            int(before_n - len(pairs)),
                            str(alternating_fixed_builder_id),
                        )
                if alternating_active_phase == "builder" and alternating_fixed_loss_id:
                    before_n = len(pairs)
                    pairs = [p for p in pairs if str(p[1]) == str(alternating_fixed_loss_id)]
                    if len(pairs) != before_n:
                        LOGGER.warning(
                            "Alternating builder-phase pair cleanup gen=%d: dropped %d pairs not using fixed_f=%s",
                            int(gen),
                            int(before_n - len(pairs)),
                            str(alternating_fixed_loss_id),
                        )
                if pairs:
                    pair_set = set((str(p[0]), str(p[1])) for p in pairs)
                    reasons_by_pair = {
                        k: v for k, v in reasons_by_pair.items() if (str(k[0]), str(k[1])) in pair_set
                    }
                    pair_phase_by_pair = {
                        k: v for k, v in pair_phase_by_pair.items() if (str(k[0]), str(k[1])) in pair_set
                    }
            else:
                LOGGER.warning(
                    "Alternating skipped this generation: missing fixed incumbent ids (fixed_builder=%s fixed_loss=%s).",
                    str(fixed_builder_id),
                    str(fixed_loss_id),
                )
        else:
            LOGGER.info("EVAL_STAGES enabled: %s", _format_eval_stages_for_log(eval_stages))
            # Core pairing budget excludes mandatory anchor pairs.
            core_budget = max(0, int(pairing_budget) - len(anchor_pairs))
            core_pairs, core_reasons = _build_coverage_plus_bandit_pairs(
                cfg_yaml=cfg_yaml,
                gen=int(gen),
                generations=int(generations),
                pairing_budget=int(core_budget),
                rng=rng,
                new_g_ids=list(new_g_ids),
                new_f_ids=list(new_f_ids),
                elite_g_ids=list(resident_g_ids),
                elite_f_ids=list(resident_f_ids),
                hof_g_ids=list(hof_g_ids),
                hof_f_ids=list(hof_f_ids),
                g_id_pool=list(g_id_pool),
                f_id_pool=list(f_id_pool),
                caches=caches,
                eval_sig=str(eval_sig),
            )
            pairs = list(anchor_pairs) + list(core_pairs)
            reasons_by_pair = dict(core_reasons)
            for k, v in anchor_reasons.items():
                reasons_by_pair.setdefault(k, []).extend(list(v))
            for p in pairs:
                pair_phase_by_pair.setdefault((str(p[0]), str(p[1])), "coevo")

        active_g_ids = list(dict.fromkeys([str(gid) for gid, _ in pairs] + ([G_REF_ID] if anchor_enabled else [])))
        keep_base_batch_ids = [int(i) for i in range(int(proxy_batches))]
        dropped = caches.prune_pref_cache(keep_g_ids=active_g_ids, keep_batch_ids=keep_base_batch_ids)
        if dropped > 0:
            LOGGER.info(
                "Gen %d pref_cache pruned: dropped=%d kept=%d keep_g_ids=%d keep_batch_ids=%d %s %s",
                int(gen),
                int(dropped),
                int(len(caches.pref_cache)),
                int(len(set(active_g_ids))),
                int(len(set(keep_base_batch_ids))),
                _cache_brief(caches),
                _cuda_mem_brief([proxy_device_str]),
            )
            if bool(cfg_yaml.get("cuda_empty_cache_on_pref_prune", False)):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

        pair_records_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        gate_repair_records_gen: List[Dict[str, Any]] = []
        joint_gate_repair_attempt_records_gen: List[Dict[str, Any]] = []
        # Carry over anchor records (already computed).
        pair_records_map.update(anchor_records)

        # Cheap stage evaluation for scheduled pairs using Common Random Numbers (base_seed).
        t_cheap0 = time.time()
        if stage1_proxy_enabled:
            for p_idx, (gid, fid) in enumerate(pairs):
                if (gid, fid) in pair_records_map and pair_records_map[(gid, fid)].get("stage") == "anchor":
                    pair_records_map[(gid, fid)]["phase"] = pair_phase_by_pair.get((gid, fid), "coevo")
                    continue
                rec = _cheap_eval_pair_cached(
                    caches=caches,
                    compiled_g=compiled_g,
                    compiled_f=compiled_f,
                    rollout_feature_caches=rollout_feature_caches,
                    cfg_yaml=cfg_yaml,
                    eval_sig=str(eval_sig),
                    gid=str(gid),
                    fid=str(fid),
                    generation=int(gen),
                    pair_index=int(p_idx),
                    seed_used=int(base_seed),
                    seed_sig=str(base_seed_sig),
                    pref_batch_id_offset=0,
                    stage="cheap",
                    reasons=reasons_by_pair.get((gid, fid), ["scheduled"]),
                    proxy_device_str=proxy_device_str,
                    joint_gate_kwargs=joint_gate_kwargs,
                    proxy_weights=dict(proxy_weights),
                    bins=bins,
                    pair_count_cap=pair_count_cap,
                    loss_scale=loss_scale,
                    cheap_gate_on=bool(stage0_gate_enabled),
                )
                pair_key = (str(gid), str(fid))
                gate_repair_events: List[Dict[str, Any]] = []
                gate_repair_applied = False
                gate_repair_attempted = False
                t_gate_repair0 = time.time()
                pre_gate_state = {
                    "pair_ok": bool(rec.get("pair_ok")),
                    "pair_reason": str(rec.get("pair_reason", "")),
                    "builder_gate_ok": rec.get("builder_gate_ok"),
                    "builder_gate_reason": rec.get("builder_gate_reason"),
                    "builder_gate_trace": rec.get("builder_gate_trace"),
                    "joint_gate_ok": rec.get("joint_gate_ok"),
                    "joint_gate_reason": rec.get("joint_gate_reason"),
                    "joint_gate_trace": rec.get("joint_gate_trace"),
                }
                if (
                    gate_repair_enabled
                    and gate_repair_remaining > 0
                    and pair_key not in gate_repair_attempted_pairs
                    and not bool(rec.get("pair_ok"))
                    and str(rec.get("pair_reason", "")) in {"cheap_proxy_gate_failed", "co_gate_failed"}
                ):
                    gate_repair_attempted = True
                    gate_repair_attempted_pairs.add(pair_key)
                    gate_repair_remaining -= 1
                    pair_fail_context = {
                        "stage": "proxy_gate",
                        "pair_reason": str(rec.get("pair_reason", "")),
                        "builder_gate_ok": rec.get("builder_gate_ok"),
                        "builder_gate_reason": rec.get("builder_gate_reason"),
                        "builder_gate_trace": rec.get("builder_gate_trace"),
                        "joint_gate_ok": rec.get("joint_gate_ok"),
                        "joint_gate_reason": rec.get("joint_gate_reason"),
                        "joint_gate_trace": rec.get("joint_gate_trace"),
                        "pair_context": {
                            "generation": int(gen),
                            "pair_index": int(p_idx),
                            "g_id": str(gid),
                            "f_id": str(fid),
                            "phase": str(pair_phase_by_pair.get((str(gid), str(fid)), "coevo")),
                        },
                    }
                    call_feedback = dict(global_feedback or {})
                    call_feedback["llm_call"] = {
                        "side": "pair_gate_repair",
                        "op_type": "PAIR_GATE_REPAIR",
                        "seed": int(rng.randint(0, 2**31 - 1)),
                    }

                    if (
                        gate_repair_builder_on_fail
                        and (not bool(rec.get("builder_gate_ok")))
                        and str(gid) != G_REF_ID
                        and str(gid) not in gate_repair_attempted_builders
                        and str(gid) in g_map
                        and isinstance(g_map[str(gid)].get("ir"), dict)
                        and p_builder_rep
                    ):
                        gate_repair_attempted_builders.add(str(gid))
                        try:
                            g_ir = pref_builder_ir_from_json(g_map[str(gid)]["ir"])
                            before_sig = _sig_pref_builder(g_ir)
                            builder_fail_report = _builder_failure_report(
                                stage="proxy_gate",
                                reason=str(rec.get("builder_gate_reason") or rec.get("pair_reason") or "builder_gate_failed"),
                                trace={
                                    "builder_gate_trace": rec.get("builder_gate_trace"),
                                    "joint_gate_trace": rec.get("joint_gate_trace"),
                                    "pair_context": pair_fail_context.get("pair_context"),
                                },
                            )
                            repaired_g_ir, rep_meta = _repair_builder_candidate_loop(
                                g_ir,
                                failure_report=builder_fail_report,
                                operator_whitelist=operator_whitelist,
                                gate_cfg={
                                    "min_pairs": int(builder_gate_live_cfg.get("min_pairs", 1) or 1),
                                    "min_coverage": float(builder_gate_live_cfg.get("min_coverage", 0.0) or 0.0),
                                    "max_pairs_per_instance": int(
                                        builder_gate_live_cfg.get("max_pairs_per_instance", 4096) or 4096
                                    ),
                                    "weight_nonneg": bool(builder_gate_live_cfg.get("weight_nonneg", True)),
                                    "semantic_tolerance": float(builder_gate_live_cfg.get("semantic_tolerance", 0.0) or 0.0),
                                    "semantic_min_pass_rate": float(
                                        builder_gate_live_cfg.get("semantic_min_pass_rate", 1.0) or 1.0
                                    ),
                                },
                                llm_prompts={"builder_m3": p_builder_m3, "builder_repair": p_builder_rep},
                                global_feedback=call_feedback,
                                max_attempts=max(0, int(gate_repair_attempts)),
                                simplify_first=bool(builder_repair_live_cfg.get("simplify_first", True)),
                            )
                            if repaired_g_ir is not None:
                                after_sig = _sig_pref_builder(repaired_g_ir)
                                g_map[str(gid)]["ir"] = asdict(repaired_g_ir)
                                g_map[str(gid)]["signature"] = after_sig
                                g_map[str(gid)]["origin"] = "REPAIR_GATE"
                                hist = g_map[str(gid)].get("history")
                                if not isinstance(hist, list):
                                    hist = []
                                    g_map[str(gid)]["history"] = hist
                                if isinstance(rep_meta, dict) and isinstance(rep_meta.get("attempts"), list):
                                    hist.extend(list(rep_meta.get("attempts") or []))
                                compiled_g[str(gid)] = compile_preference_builder(
                                    repaired_g_ir,
                                    operator_whitelist=operator_whitelist,
                                )
                                dropped_pref = 0
                                for pref_key in list(caches.pref_cache.keys()):
                                    if str(pref_key[0]) == str(gid):
                                        dropped_pref += 1
                                        del caches.pref_cache[pref_key]
                                dropped_pair = 0
                                for ck in list(caches.pair_cache.keys()):
                                    if str(ck[2]) == str(eval_sig) and str(ck[0]) == str(gid):
                                        dropped_pair += 1
                                        del caches.pair_cache[ck]
                                gate_repair_events.append(
                                    {
                                        "side": "builder",
                                        "status": "repaired",
                                        "before_signature": str(before_sig),
                                        "after_signature": str(after_sig),
                                        "dropped_pref_cache": int(dropped_pref),
                                        "dropped_pair_cache": int(dropped_pair),
                                        "meta": rep_meta,
                                    }
                                )
                                gate_repair_applied = True
                            else:
                                gate_repair_events.append(
                                    {
                                        "side": "builder",
                                        "status": "failed",
                                        "meta": rep_meta,
                                    }
                                )
                        except Exception as exc:  # noqa: BLE001
                            gate_repair_events.append({"side": "builder", "status": "error", "error": str(exc)})

                    if (
                        gate_repair_loss_on_fail
                        and (not bool(rec.get("joint_gate_ok")))
                        and str(fid) not in gate_repair_attempted_losses
                        and str(fid) in f_map
                        and isinstance(f_map[str(fid)].get("ir"), dict)
                        and p_loss_rep
                    ):
                        gate_repair_attempted_losses.add(str(fid))
                        try:
                            f_ir = free_loss_ir_from_json(f_map[str(fid)]["ir"])
                            before_sig = _sig_free_loss(f_ir)
                            last_fail: Dict[str, Any] = dict(pair_fail_context)
                            repaired_f_ir: FreeLossIR | None = None
                            rep_hist: List[Dict[str, Any]] = []
                            loss_attempt_reports: List[Dict[str, Any]] = []
                            for _ra in range(max(0, int(gate_repair_attempts))):
                                fail_payload = dict(last_fail)
                                fail_payload["repair_attempt"] = int(_ra)
                                fail_payload["llm_call"] = dict(call_feedback.get("llm_call") or {})
                                attempt_report: Dict[str, Any] = {
                                    "attempt": int(_ra),
                                    "input_failure_stage": str(fail_payload.get("stage", "")),
                                    "input_failure_reason": str(fail_payload.get("reason", fail_payload.get("pair_reason", "")) or ""),
                                }
                                loss_failure_trace = fail_payload.get("joint_gate_trace")
                                loss_failure_kind = (
                                    str(loss_failure_trace.get("failure_kind"))
                                    if isinstance(loss_failure_trace, dict) and loss_failure_trace.get("failure_kind") is not None
                                    else ""
                                )
                                selected_loss_prompt = str(p_loss_rep)
                                if str(fail_payload.get("pair_reason", "")) != "co_gate_failed" and loss_failure_kind in runtime_repair_failure_kinds:
                                    selected_loss_prompt = str(p_loss_runtime_rep)
                                prompt_sha = _build_free_loss_failure_prompt(
                                    selected_loss_prompt,
                                    candidate=f_ir,
                                    failure_reason=fail_payload,
                                    global_feedback=None,
                                    block_name="CANDIDATE_AND_FAILURE_JSON",
                                    ensure_ascii=True,
                                )[1]
                                rep_hist.append(
                                    {
                                        "attempt": int(_ra),
                                        "side": "loss",
                                        "llm_op": "REPAIR",
                                        "prompt_path": str(selected_loss_prompt),
                                        "prompt_sha1": str(prompt_sha),
                                    }
                                )
                                candidate = loss_llm_ops.repair_free_loss(
                                    selected_loss_prompt,
                                    failed_ir=f_ir,
                                    failure_reason=fail_payload,
                                    prompt_context=loss_prompt_context,
                                )
                                static_res = run_static_gates(candidate, operator_whitelist=operator_whitelist)
                                if not bool(static_res.ok):
                                    last_fail = {
                                        "stage": "static_gate",
                                        "reason": str(static_res.reason),
                                        "trace": static_res.trace,
                                    }
                                    attempt_report["result"] = "static_gate_failed"
                                    attempt_report["static_reason"] = str(static_res.reason)
                                    if bool(loss_repair_live_cfg.get("simplify_first", True)) and p_loss_m3:
                                        try:
                                            prompt_sha_m3 = _build_free_loss_failure_prompt(
                                                p_loss_m3,
                                                candidate=candidate,
                                                failure_reason=last_fail,
                                                global_feedback=call_feedback,
                                                block_name="CANDIDATE_AND_FAILURE_JSON",
                                                ensure_ascii=False,
                                            )[1]
                                            rep_hist.append(
                                                {
                                                    "attempt": int(_ra),
                                                    "side": "loss",
                                                    "llm_op": "M3",
                                                    "prompt_path": str(p_loss_m3),
                                                    "prompt_sha1": str(prompt_sha_m3),
                                                }
                                            )
                                            candidate = loss_llm_ops.m3_simplify_loss(
                                                p_loss_m3,
                                                candidate=candidate,
                                                failure_reason=last_fail,
                                                global_feedback=call_feedback,
                                                prompt_context=loss_prompt_context,
                                            )
                                            attempt_report["m3_attempted"] = True
                                        except Exception:
                                            attempt_report["m3_attempted"] = False
                                    else:
                                        attempt_report["m3_attempted"] = False
                                    loss_attempt_reports.append(attempt_report)
                                    f_ir = candidate
                                    continue
                                try:
                                    _ = compile_free_loss(candidate, operator_whitelist=operator_whitelist)
                                except Exception as exc:  # noqa: BLE001
                                    last_fail = {"stage": "compile", "error": str(exc)}
                                    attempt_report["result"] = "compile_failed"
                                    attempt_report["compile_error"] = str(exc)
                                    loss_attempt_reports.append(attempt_report)
                                    f_ir = candidate
                                    continue
                                repaired_f_ir = candidate
                                attempt_report["result"] = "ok"
                                loss_attempt_reports.append(attempt_report)
                                break

                            if repaired_f_ir is not None:
                                after_sig = _sig_free_loss(repaired_f_ir)
                                f_map[str(fid)]["ir"] = asdict(repaired_f_ir)
                                f_map[str(fid)]["signature"] = after_sig
                                f_map[str(fid)]["origin"] = "REPAIR_GATE"
                                hist = f_map[str(fid)].get("history")
                                if not isinstance(hist, list):
                                    hist = []
                                    f_map[str(fid)]["history"] = hist
                                hist.extend(rep_hist)
                                compiled_f[str(fid)] = compile_free_loss(
                                    repaired_f_ir,
                                    operator_whitelist=operator_whitelist,
                                )
                                dropped_pair = 0
                                for ck in list(caches.pair_cache.keys()):
                                    if str(ck[2]) == str(eval_sig) and str(ck[1]) == str(fid):
                                        dropped_pair += 1
                                        del caches.pair_cache[ck]
                                gate_repair_events.append(
                                    {
                                        "side": "loss",
                                        "status": "repaired",
                                        "before_signature": str(before_sig),
                                        "after_signature": str(after_sig),
                                        "attempts": int(len(rep_hist)),
                                        "dropped_pair_cache": int(dropped_pair),
                                        "attempt_reports": loss_attempt_reports,
                                    }
                                )
                                gate_repair_applied = True
                            else:
                                gate_repair_events.append(
                                    {
                                        "side": "loss",
                                        "status": "failed",
                                        "attempts": int(gate_repair_attempts),
                                        "attempt_reports": loss_attempt_reports,
                                    }
                                )
                        except Exception as exc:  # noqa: BLE001
                            gate_repair_events.append({"side": "loss", "status": "error", "error": str(exc)})

                    if gate_repair_applied:
                        rec = _cheap_eval_pair_cached(
                            caches=caches,
                            compiled_g=compiled_g,
                            compiled_f=compiled_f,
                            rollout_feature_caches=rollout_feature_caches,
                            cfg_yaml=cfg_yaml,
                            eval_sig=str(eval_sig),
                            gid=str(gid),
                            fid=str(fid),
                            generation=int(gen),
                            pair_index=int(p_idx),
                            seed_used=int(base_seed),
                            seed_sig=str(base_seed_sig),
                            pref_batch_id_offset=0,
                            stage="cheap",
                            reasons=list(reasons_by_pair.get((gid, fid), ["scheduled"])) + ["gate_repair_retry"],
                            proxy_device_str=proxy_device_str,
                            joint_gate_kwargs=joint_gate_kwargs,
                            proxy_weights=dict(proxy_weights),
                            bins=bins,
                            pair_count_cap=pair_count_cap,
                            loss_scale=loss_scale,
                            cheap_gate_on=bool(stage0_gate_enabled),
                        )
                        rec["gate_repair"] = {
                            "enabled": True,
                            "events": gate_repair_events,
                        }
                if gate_repair_attempted:
                    post_gate_state = {
                        "pair_ok": bool(rec.get("pair_ok")),
                        "pair_reason": str(rec.get("pair_reason", "")),
                        "builder_gate_ok": rec.get("builder_gate_ok"),
                        "builder_gate_reason": rec.get("builder_gate_reason"),
                        "builder_gate_trace": rec.get("builder_gate_trace"),
                        "joint_gate_ok": rec.get("joint_gate_ok"),
                        "joint_gate_reason": rec.get("joint_gate_reason"),
                        "joint_gate_trace": rec.get("joint_gate_trace"),
                    }
                    repair_record = {
                        "generation": int(gen),
                        "pair_index": int(p_idx),
                        "g_id": str(gid),
                        "f_id": str(fid),
                        "phase": str(pair_phase_by_pair.get((str(gid), str(fid)), "coevo")),
                        "repair_applied": bool(gate_repair_applied),
                        "repair_elapsed_s": float(time.time() - t_gate_repair0),
                        "pre": pre_gate_state,
                        "post": post_gate_state,
                        "events": gate_repair_events,
                    }
                    rec["gate_repair"] = dict(repair_record)
                    gate_repair_records_gen.append(dict(repair_record))
                    LOGGER.info(
                        "Gate-repair retry gen=%d pair=(%s,%s): pre(pair=%s,bg=%s,jg=%s) post(pair=%s,bg=%s,jg=%s) events=%d elapsed_s=%.1f",
                        int(gen),
                        str(gid),
                        str(fid),
                        str(pre_gate_state.get("pair_reason")),
                        str(pre_gate_state.get("builder_gate_reason")),
                        str(pre_gate_state.get("joint_gate_reason")),
                        str(post_gate_state.get("pair_reason")),
                        str(post_gate_state.get("builder_gate_reason")),
                        str(post_gate_state.get("joint_gate_reason")),
                        int(len(gate_repair_events)),
                        float(repair_record["repair_elapsed_s"]),
                    )
                rec["phase"] = pair_phase_by_pair.get((str(gid), str(fid)), "coevo")
                pair_records_map[(str(gid), str(fid))] = rec
                if progress_every_pairs > 0 and (
                    (p_idx + 1) in (1, int(len(pairs))) or ((p_idx + 1) % progress_every_pairs == 0)
                ):
                    LOGGER.info(
                        "Cheap eval progress gen=%d: %d/%d %s %s",
                        int(gen),
                        int(p_idx + 1),
                        int(len(pairs)),
                        _cache_brief(caches),
                        _cuda_mem_brief([proxy_device_str]),
                    )
            if pairs:
                LOGGER.info(
                    "Cheap eval done gen=%d: pairs=%d elapsed_s=%.1f %s %s",
                    int(gen),
                    int(len(pairs)),
                    float(time.time() - t_cheap0),
                    _cache_brief(caches),
                    _cuda_mem_brief([proxy_device_str]),
                )
                _maybe_auto_flush_pref_cache(
                    caches=caches,
                    cfg_yaml=cfg_yaml,
                    default_device_str=proxy_device_str,
                    generation=int(gen),
                    scope="after_cheap",
                )
        else:
            # Proxy is disabled: still run stage0_gate (dummy-feature gates) so stage3 can
            # filter strictly by gate outcomes without relying on proxy/micro metrics.
            t_gate0 = time.time()
            LOGGER.info(
                "Gate-only eval (proxy disabled) gen=%d: pairs=%d cheap_gate_on=%s",
                int(gen),
                int(len(pairs)),
                str(bool(stage0_gate_enabled)),
            )
            for p_idx, (gid, fid) in enumerate(pairs):
                if (gid, fid) in pair_records_map and pair_records_map[(gid, fid)].get("stage") == "anchor":
                    pair_records_map[(gid, fid)]["phase"] = pair_phase_by_pair.get((gid, fid), "coevo")
                    continue

                cache_key = (str(gid), str(fid), str(eval_sig))
                cached = caches.get_pair(cache_key)
                if isinstance(cached, dict) and str(cached.get("stage")) == "gate":
                    rec = dict(cached)
                    rec["generation"] = int(gen)
                    rec["pair_index"] = int(p_idx)
                else:
                    g_entry = g_map.get(str(gid))
                    if not isinstance(g_entry, dict) and str(gid) == G_REF_ID:
                        g_entry = {"id": str(G_REF_ID), "ir": asdict(_ref_builder_ir())}
                    f_entry = f_map.get(str(fid))
                    if not isinstance(f_entry, dict) and str(fid) == F_REF_ID:
                        f_entry = {"id": str(F_REF_ID), "ir": asdict(_ref_loss_ir())}
                    if not isinstance(g_entry, dict) or not isinstance(f_entry, dict):
                        LOGGER.warning(
                            "Gate-only skip gen=%d pair_index=%d missing entry for pair (%s,%s): g=%s f=%s",
                            int(gen),
                            int(p_idx),
                            str(gid),
                            str(fid),
                            str(isinstance(g_entry, dict)),
                            str(isinstance(f_entry, dict)),
                        )
                        continue

                    rec = _evaluate_pair_worker(
                        {
                            "generation": int(gen),
                            "pair_index": int(p_idx),
                            "g_entry": dict(g_entry),
                            "f_entry": dict(f_entry),
                            "cfg_yaml": dict(cfg_yaml),
                            "device_str": str(proxy_device_str),
                            "operator_whitelist": list(operator_whitelist),
                            "run_dir": str(run_dir),
                            "cheap_gate_on": bool(stage0_gate_enabled),
                            "high_fidelity_on": False,
                            "eval_budget_signature": str(eval_sig),
                        }
                    )
                joint_gate_repair_attempt_records_gen.extend(_pop_joint_gate_repair_reports(rec))
                rec["stage"] = "gate"
                rec["phase"] = pair_phase_by_pair.get((str(gid), str(fid)), "coevo")
                caches.set_pair(cache_key, dict(rec))
                pair_records_map[(str(gid), str(fid))] = rec

                if progress_every_pairs > 0 and (
                    (p_idx + 1) in (1, int(len(pairs))) or ((p_idx + 1) % progress_every_pairs == 0)
                ):
                    LOGGER.info(
                        "Gate-only progress gen=%d: %d/%d %s",
                        int(gen),
                        int(p_idx + 1),
                        int(len(pairs)),
                        _cache_brief(caches),
                    )
            if pairs:
                LOGGER.info(
                    "Gate-only eval done gen=%d: pairs=%d elapsed_s=%.1f %s",
                    int(gen),
                    int(len(pairs)),
                    float(time.time() - t_gate0),
                    _cache_brief(caches),
                )

        # 2-seed recheck: best_pair + elite-boundary pairs (cheap stage only).
        recheck_enabled = bool(cfg_yaml.get("recheck_enabled", True))
        recheck_num_seeds = int(cfg_yaml.get("recheck_num_seeds", 2) or 2)
        if stage1_proxy_enabled and recheck_enabled and recheck_num_seeds >= 2 and pairs:
            recheck_offset = int(cfg_yaml.get("recheck_seed_offset", 10007) or 10007)
            recheck_elite_margin = int(cfg_yaml.get("recheck_elite_margin", 1) or 1)
            recheck_max_pairs = int(cfg_yaml.get("recheck_max_pairs", 16) or 16)

            cand_pairs = [
                p
                for p in pairs
                if p[0] != G_REF_ID
                and p[1] != F_REF_ID
                and str(pair_records_map.get((p[0], p[1]), {}).get("stage")) != "anchor"
            ]
            if not cand_pairs:
                cand_pairs = list(pairs)
            best_pair = min(cand_pairs, key=lambda p: float(pair_records_map.get((p[0], p[1]), {}).get("score", float("inf"))))
            g_fit, f_fit = _credit_assignment_v2(pair_records=list(pair_records_map.values()))
            g_rank = sorted([(gid, float(v)) for gid, v in g_fit.items()], key=lambda x: x[1])
            f_rank = sorted([(fid, float(v)) for fid, v in f_fit.items()], key=lambda x: x[1])
            g_boundary: set[str] = set()
            f_boundary: set[str] = set()
            for idx in range(max(0, elite_g - 1 - recheck_elite_margin), min(len(g_rank), elite_g + recheck_elite_margin)):
                g_boundary.add(str(g_rank[idx][0]))
            for idx in range(max(0, elite_f - 1 - recheck_elite_margin), min(len(f_rank), elite_f + recheck_elite_margin)):
                f_boundary.add(str(f_rank[idx][0]))

            recheck_pairs: List[Tuple[str, str]] = [tuple(best_pair)]
            for gid, fid in pairs:
                if len(recheck_pairs) >= recheck_max_pairs:
                    break
                if (gid, fid) == tuple(best_pair):
                    continue
                if gid in g_boundary or fid in f_boundary:
                    recheck_pairs.append((gid, fid))

            t_recheck0 = time.time()
            seed2 = int(base_seed + recheck_offset)
            seed2_sig = seed_signature_for_proxy(
                seed=seed2,
                problem_size=int(proxy_problem_size),
                batch_ids=list(range(int(proxy_batches))),
                batch_size=int(proxy_batch_size),
            )
            LOGGER.info(
                "Recheck seed2 gen=%d: pairs=%d seed=%d device=%s proxy_batches=%d",
                int(gen),
                int(len(recheck_pairs)),
                int(seed2),
                str(proxy_device_str),
                int(proxy_batches),
            )
            rollout_feature_caches_2: List[Dict[str, torch.Tensor]] = []
            for batch_id in range(int(proxy_batches)):
                key = (int(seed2), int(proxy_problem_size), int(batch_id))
                hit = caches.get_rollout(key) is not None
                fc = build_or_get_rollout_feature_cache(
                    caches=caches,
                    cfg=_build_hf_cfg(cfg_yaml, seed=seed2, device_str=proxy_device_str),
                    seed=int(seed2),
                    problem_size=int(proxy_problem_size),
                    batch_id=int(batch_id),
                    batch_size=int(proxy_batch_size),
                    device=proxy_device,
                )
                rollout_feature_caches_2.append(fc)
                if (batch_id + 1) in (1, int(proxy_batches)) or (
                    progress_every_proxy_batches > 0 and ((batch_id + 1) % progress_every_proxy_batches == 0)
                ):
                    LOGGER.info(
                        "Recheck seed2 rollouts progress gen=%d: %d/%d cache_hit=%s %s %s",
                        int(gen),
                        int(batch_id + 1),
                        int(proxy_batches),
                        str(hit),
                        _cache_brief(caches),
                        _cuda_mem_brief([proxy_device_str]),
                    )

            for gid, fid in recheck_pairs:
                rec2 = _cheap_eval_pair_cached(
                    caches=caches,
                    compiled_g=compiled_g,
                    compiled_f=compiled_f,
                    rollout_feature_caches=rollout_feature_caches_2,
                    cfg_yaml=cfg_yaml,
                    eval_sig=str(eval_sig),
                    gid=str(gid),
                    fid=str(fid),
                    generation=int(gen),
                    pair_index=_safe_int(pair_records_map.get((gid, fid), {}).get("pair_index", 0), 0),
                    seed_used=int(seed2),
                    seed_sig=str(seed2_sig),
                    pref_batch_id_offset=100000,
                    stage="cheap_recheck",
                    reasons=["recheck_seed2"],
                    proxy_device_str=proxy_device_str,
                    joint_gate_kwargs=joint_gate_kwargs,
                    proxy_weights=dict(proxy_weights),
                    bins=bins,
                    pair_count_cap=pair_count_cap,
                    loss_scale=loss_scale,
                    cheap_gate_on=bool(stage0_gate_enabled),
                )
                pair_records_map[(str(gid), str(fid))] = rec2
            if recheck_pairs:
                LOGGER.info(
                    "Recheck seed2 done gen=%d: pairs=%d elapsed_s=%.1f %s %s",
                    int(gen),
                    int(len(recheck_pairs)),
                    float(time.time() - t_recheck0),
                    _cache_brief(caches),
                    _cuda_mem_brief([proxy_device_str]),
                )
                _maybe_auto_flush_pref_cache(
                    caches=caches,
                    cfg_yaml=cfg_yaml,
                    default_device_str=proxy_device_str,
                    generation=int(gen),
                    scope="after_recheck",
                )
                # Drop recheck-only pref batches (batch_id offset 100000) to keep VRAM stable during
                # subsequent micro-unroll / HF stages and across generations.
                dropped_recheck = caches.prune_pref_cache(
                    keep_g_ids=active_g_ids,
                    keep_batch_ids=keep_base_batch_ids,
                )
                if dropped_recheck > 0:
                    LOGGER.info(
                        "Gen %d pref_cache drop recheck: dropped=%d kept=%d %s %s",
                        int(gen),
                        int(dropped_recheck),
                        int(len(caches.pref_cache)),
                        _cache_brief(caches),
                        _cuda_mem_brief([proxy_device_str]),
                    )
                    if bool(cfg_yaml.get("cuda_empty_cache_on_pref_prune", False)):
                        try:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass

        pair_records: List[Dict[str, Any]] = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]
        for rec in pair_records:
            k = (str(rec.get("g_id")), str(rec.get("f_id")))
            rec["phase"] = pair_phase_by_pair.get(k, rec.get("phase", "coevo"))

        if stage2_micro_enabled and bool(cfg_yaml.get("drop_pref_cache_before_micro_unroll", True)):
            dropped_pre_mu = caches.prune_pref_cache(keep_g_ids=[], keep_batch_ids=[])
            if dropped_pre_mu > 0:
                LOGGER.info(
                    "Gen %d pref_cache drop before micro-unroll: dropped=%d kept=%d %s %s",
                    int(gen),
                    int(dropped_pre_mu),
                    int(len(caches.pref_cache)),
                    _cache_brief(caches),
                    _cuda_mem_brief([proxy_device_str]),
                )
                if bool(cfg_yaml.get("cuda_empty_cache_on_pref_prune", False)):
                    try:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass

        # Optional Stage B: offline micro-unroll on cached rollouts (no new rollouts).
        # This improves selection signal at a fraction of HF cost by taking a few
        # gradient steps on cached log_prob tensors.
        micro_enabled = bool(stage2_micro_enabled)
        if micro_enabled:
            default_top_m = int(
                cfg_yaml.get(
                    "high_fidelity_top_m",
                    max(1, min(len(pair_records), pairing_budget // 4)),
                )
                or 1
            )
            mu_candidates = [
                r
                for r in pair_records
                if (bool(r.get("pair_ok")) if stage1_proxy_enabled else True)
                and str(r.get("stage")) != "anchor"
            ]
            if _uses_fixed_side_search(search_mode):
                if alternating_active_phase == "loss" and alternating_fixed_builder_id:
                    mu_candidates = [
                        r for r in mu_candidates if str(r.get("g_id")) == str(alternating_fixed_builder_id)
                    ]
                if alternating_active_phase == "builder" and alternating_fixed_loss_id:
                    mu_candidates = [
                        r for r in mu_candidates if str(r.get("f_id")) == str(alternating_fixed_loss_id)
                    ]
            if stage1_proxy_enabled:
                mu_candidates.sort(key=lambda r: float(r.get("score", float("inf"))))
            else:
                mu_candidates.sort(
                    key=lambda r: (
                        _safe_int(r.get("pair_index", 10**9), 10**9),
                        str(r.get("g_id", "")),
                        str(r.get("f_id", "")),
                    )
                )
            if mu_candidates:
                micro_top_k = int(
                    cfg_yaml.get(
                        "micro_unroll_top_k",
                        max(int(default_top_m), min(len(mu_candidates), int(default_top_m) * 2)),
                    )
                    or 0
                )
                micro_top_k = max(int(default_top_m), min(int(micro_top_k), int(len(mu_candidates))))
                micro_steps = int(cfg_yaml.get("micro_unroll_steps", 3) or 3)
                micro_lr = float(cfg_yaml.get("micro_unroll_lr", 5e-2) or 5e-2)
                micro_alpha = _alpha_from_cfg(cfg_yaml, key="micro_unroll_alpha")
                micro_weight_decay = float(cfg_yaml.get("micro_unroll_weight_decay", 0.0) or 0.0)
                micro_reuse_pref = bool(cfg_yaml.get("micro_unroll_reuse_pref_batch_when_safe", True))
                micro_max_pairs = cfg_yaml.get("micro_unroll_max_pairs", None)
                try:
                    micro_max_pairs_i = int(micro_max_pairs) if micro_max_pairs is not None else None
                except (TypeError, ValueError):
                    micro_max_pairs_i = None
                micro_timeout_s = cfg_yaml.get("micro_unroll_timeout_s", None)
                try:
                    micro_timeout_s_f = float(micro_timeout_s) if micro_timeout_s is not None else None
                except (TypeError, ValueError):
                    micro_timeout_s_f = None

                mu_eval = mu_candidates[: int(micro_top_k)]
                LOGGER.info(
                    "Micro-unroll gen=%d: tasks=%d steps=%d lr=%.3g alpha=%.3g (proxy_batches=%d)",
                    int(gen),
                    int(len(mu_eval)),
                    int(micro_steps),
                    float(micro_lr),
                    float(micro_alpha),
                    int(len(rollout_feature_caches)),
                )
                t_mu0 = time.time()
                progress_every_mu = int(cfg_yaml.get("progress_log_every_micro_unroll_tasks", 8) or 8)
                for mu_idx, r in enumerate(mu_eval):
                    gid = str(r.get("g_id"))
                    fid = str(r.get("f_id"))
                    cache_key = (gid, fid, str(eval_sig))
                    cached = caches.get_pair(cache_key)
                    if isinstance(cached, dict) and str(cached.get("stage")) == "micro_unroll" and cached.get("micro_metrics"):
                        continue
                    g_comp = compiled_g.get(gid)
                    f_comp = compiled_f.get(fid)
                    if g_comp is None or f_comp is None:
                        rec_mu = dict(pair_records_map.get((gid, fid), dict(r)))
                        rec_mu["pair_ok"] = False
                        rec_mu["pair_reason"] = "micro_unroll_compile_missing"
                        rec_mu["stage"] = "micro_unroll"
                        rec_mu["score"] = float("inf")
                        caches.set_pair(cache_key, rec_mu)
                        pair_records_map[(gid, fid)] = rec_mu
                        continue

                    try:
                        t_pair0 = time.time()
                        LOGGER.info(
                            "Micro-unroll start gen=%d: %d/%d g_id=%s f_id=%s proxy_score=%.4g",
                            int(gen),
                            int(mu_idx + 1),
                            int(len(mu_eval)),
                            str(gid),
                            str(fid),
                            float(r.get("proxy_score", r.get("score", float("inf")))),
                        )
                        mu_score, mu_metrics = micro_unroll_score_for_pair(
                            g=g_comp,
                            f=f_comp,
                            rollout_feature_caches=rollout_feature_caches,
                            steps=int(micro_steps),
                            lr=float(micro_lr),
                            alpha=float(micro_alpha),
                            weight_decay=float(micro_weight_decay),
                            reuse_pref_batch_when_safe=bool(micro_reuse_pref),
                            max_pairs=micro_max_pairs_i,
                            timeout_s=micro_timeout_s_f,
                        )
                        rec_mu = dict(pair_records_map.get((gid, fid), dict(r)))
                        rec_mu["pair_ok"] = True
                        rec_mu["pair_reason"] = "ok_micro_unroll"
                        rec_mu["stage"] = "micro_unroll"
                        rec_mu["micro_score"] = float(mu_score)
                        rec_mu["micro_metrics"] = dict(mu_metrics)
                        rec_mu["score"] = float(mu_score)
                        caches.set_pair(cache_key, rec_mu)
                        pair_records_map[(gid, fid)] = rec_mu
                        LOGGER.info(
                            "Micro-unroll done gen=%d: %d/%d g_id=%s f_id=%s score=%.4g elapsed_s=%.1f",
                            int(gen),
                            int(mu_idx + 1),
                            int(len(mu_eval)),
                            str(gid),
                            str(fid),
                            float(mu_score),
                            float(time.time() - t_pair0),
                        )
                    except Exception as exc:  # noqa: BLE001
                        rec_mu = dict(pair_records_map.get((gid, fid), dict(r)))
                        rec_mu["pair_ok"] = False
                        rec_mu["pair_reason"] = "micro_unroll_failed"
                        rec_mu["micro_error"] = str(exc)
                        rec_mu["stage"] = "micro_unroll"
                        rec_mu["score"] = float("inf")
                        caches.set_pair(cache_key, rec_mu)
                        pair_records_map[(gid, fid)] = rec_mu
                        LOGGER.exception(
                            "Micro-unroll failed gen=%d: %d/%d g_id=%s f_id=%s",
                            int(gen),
                            int(mu_idx + 1),
                            int(len(mu_eval)),
                            str(gid),
                            str(fid),
                        )
                    if progress_every_mu > 0 and ((mu_idx + 1) in (1, int(len(mu_eval))) or ((mu_idx + 1) % progress_every_mu == 0)):
                        LOGGER.info(
                            "Micro-unroll progress gen=%d: %d/%d elapsed_s=%.1f %s %s",
                            int(gen),
                            int(mu_idx + 1),
                            int(len(mu_eval)),
                            float(time.time() - t_mu0),
                            _cache_brief(caches),
                            _cuda_mem_brief([proxy_device_str]),
                        )
                        _maybe_auto_flush_pref_cache(
                            caches=caches,
                            cfg_yaml=cfg_yaml,
                            default_device_str=proxy_device_str,
                            generation=int(gen),
                            scope="micro_unroll_progress",
                        )

                # Rebuild after micro-unroll overwrites `score` for some pairs.
                pair_records = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]
                for rec in pair_records:
                    k = (str(rec.get("g_id")), str(rec.get("f_id")))
                    rec["phase"] = pair_phase_by_pair.get(k, rec.get("phase", "coevo"))

        # High-fidelity stage: offline mini-train.
        if stage3_hf_enabled:
            top_m = int(cfg_yaml.get("high_fidelity_top_m", max(1, min(len(pair_records), pairing_budget // 4))) or 1)
            # Stage3 must include *all* gate-passed pairs (no top-m truncation).
            candidates = [r for r in pair_records if bool(r.get("pair_ok")) and str(r.get("stage")) != "anchor"]
            if _uses_fixed_side_search(search_mode):
                if alternating_active_phase == "loss" and alternating_fixed_builder_id:
                    candidates = [
                        r for r in candidates if str(r.get("g_id")) == str(alternating_fixed_builder_id)
                    ]
                if alternating_active_phase == "builder" and alternating_fixed_loss_id:
                    candidates = [
                        r for r in candidates if str(r.get("f_id")) == str(alternating_fixed_loss_id)
                    ]
            candidates.sort(
                key=lambda r: (
                    _safe_int(r.get("pair_index", 10**9), 10**9),
                    str(r.get("g_id", "")),
                    str(r.get("f_id", "")),
                )
            )

            selected = list(candidates)
            LOGGER.info(
                "HF selection gen=%d: eligible=%d selected=%d (top_m=%d ignored)",
                int(gen),
                int(len(candidates)),
                int(len(selected)),
                int(top_m),
            )

            hf_tasks: List[Dict[str, Any]] = []
            for r in selected:
                gid = str(r["g_id"])
                fid = str(r["f_id"])
                cache_key = (gid, fid, str(eval_sig))
                cached = caches.get_pair(cache_key)
                if isinstance(cached, dict) and str(cached.get("stage")) == "high_fidelity" and cached.get("fitness"):
                    continue
                g_entry = g_map.get(gid)
                if not isinstance(g_entry, dict) and gid == G_REF_ID:
                    g_entry = {"id": str(G_REF_ID), "ir": asdict(_ref_builder_ir())}
                f_entry = f_map.get(fid)
                if not isinstance(f_entry, dict) and fid == F_REF_ID:
                    f_entry = {"id": str(F_REF_ID), "ir": asdict(_ref_loss_ir())}
                if isinstance(r, dict) and isinstance(r.get("g_ir"), dict):
                    g_entry = {"id": str(gid), "ir": dict(r.get("g_ir") or {})}
                if isinstance(r, dict) and isinstance(r.get("f_ir"), dict):
                    f_entry = {"id": str(fid), "ir": dict(r.get("f_ir") or {})}
                if not isinstance(g_entry, dict) or not isinstance(f_entry, dict):
                    LOGGER.warning(
                        "HF skip gen=%d pair_index=%s missing entry for pair (%s,%s): g_entry=%s f_entry=%s",
                        int(gen),
                        str(r.get("pair_index", -1)),
                        str(gid),
                        str(fid),
                        str(isinstance(g_entry, dict)),
                        str(isinstance(f_entry, dict)),
                    )
                    continue
                # Distribute high-fidelity tasks round-robin across the configured devices.
                # Do not use the global pair_index here because it includes anchors and
                # other non-HF stages, which can skew GPU assignment and leave devices idle.
                device_str = device_list[int(len(hf_tasks)) % len(device_list)]
                hf_tasks.append(
                    {
                        "generation": int(gen),
                        "pair_index": _safe_int(r.get("pair_index", -1), -1),
                        "g_entry": dict(g_entry),
                        "f_entry": dict(f_entry),
                        "cfg_yaml": dict(cfg_yaml),
                        "device_str": device_str,
                        "operator_whitelist": list(operator_whitelist),
                        "run_dir": str(run_dir),
                        # Proxy stage already computed joint metrics; do not re-block HF on dummy-gate mismatch.
                        "cheap_gate_on": False,
                        "high_fidelity_on": True,
                        "eval_budget_signature": str(eval_sig),
                        "proxy_record": dict(r),
                        "baseline_epoch_objectives": list(baseline_epoch_objectives)
                        if baseline_epoch_objectives
                        else None,
                        "baseline_early_valid": baseline_early_valid,
                        "early_eval_steps": int(early_eval_steps),
                    }
                )

            hf_results: List[Dict[str, Any]] = []
            if hf_tasks:
                LOGGER.info(
                    "HF device assignment gen=%d: %s",
                    int(gen),
                    dict(collections.Counter(str(t.get("device_str", "")) for t in hf_tasks)),
                )
                LOGGER.info(
                    "HF eval gen=%d: tasks=%d mp=%s procs=%d",
                    int(gen),
                    int(len(hf_tasks)),
                    str(mp_enabled),
                    int(mp_processes),
                )
                hf_scheduler_mode = _hf_scheduler_mode(cfg_yaml)
                if mp_enabled and mp_processes > 0 and len(hf_tasks) > 1:
                    procs = min(int(mp_processes), max(1, len(hf_tasks)), max(1, len(device_list)))
                    if hf_scheduler_mode == "subprocess":
                        LOGGER.info(
                            "High-fidelity via external subprocess scheduler: processes=%d tasks=%d",
                            procs,
                            len(hf_tasks),
                        )
                        hf_results.extend(
                            _run_hf_tasks_via_subprocess(
                                hf_tasks=hf_tasks,
                                run_dir=str(run_dir),
                                device_list=device_list,
                                max_workers=int(procs),
                                runtime_trace=runtime_trace,
                            )
                        )
                    else:
                        import multiprocessing as mp

                        ctx = mp.get_context(mp_start_method)
                        LOGGER.info(
                            "High-fidelity via pinned mp: start_method=%s processes=%d tasks=%d",
                            mp_start_method,
                            procs,
                            len(hf_tasks),
                        )

                        task_queue: Any = ctx.Queue()
                        result_queue: Any = ctx.Queue()
                        workers: List[Any] = []
                        try:
                            for task in hf_tasks:
                                task_queue.put(dict(task))
                            for _ in range(int(procs)):
                                task_queue.put(None)

                            for w_idx in range(int(procs)):
                                dev = device_list[int(w_idx) % len(device_list)]
                                p = ctx.Process(
                                    target=_hf_pinned_device_worker,
                                    args=(str(dev), task_queue, result_queue),
                                )
                                p.daemon = False
                                p.start()
                                workers.append(p)

                            for _ in range(int(len(hf_tasks))):
                                hf_results.append(dict(result_queue.get()))
                        finally:
                            for p in workers:
                                p.join()
                else:
                    for task in hf_tasks:
                        hf_results.append(_evaluate_pair_worker(task))
                for hf_rec in hf_results:
                    joint_gate_repair_attempt_records_gen.extend(
                        _pop_joint_gate_repair_reports(hf_rec)
                    )

                fatal = [r for r in hf_results if isinstance(r, dict) and r.get("fatal_error")]
                if fatal:
                    first = dict(fatal[0])
                    raise RuntimeError(
                        f"Stage3 fatal error (pair={first.get('g_id')},{first.get('f_id')}): {first.get('fatal_error')}"
                    )
                try:
                    incumbent_ref_for_hf = None
                    if isinstance(best_so_far, dict):
                        try:
                            incumbent_ref_for_hf = float(best_so_far.get("score"))
                        except (TypeError, ValueError):
                            incumbent_ref_for_hf = None
                    ok = sum(1 for r in hf_results if bool(r.get("pair_ok")))
                    failed = int(len(hf_results) - ok)
                    better = 0
                    worse = 0
                    unknown = 0
                    for r in hf_results:
                        score_raw = r.get("score")
                        try:
                            score_f = float(score_raw)
                        except (TypeError, ValueError):
                            r["better_than_incumbent"] = None
                            unknown += 1
                            continue
                        if not math.isfinite(score_f):
                            r["better_than_incumbent"] = None
                            unknown += 1
                            continue
                        better_i = _is_better_than_reference(
                            cand_score=score_f,
                            reference_score=incumbent_ref_for_hf,
                            metric_mode=metric_mode,
                            improve_eps=improve_eps,
                        )
                        r["better_than_incumbent"] = bool(better_i)
                        if bool(better_i):
                            better += 1
                        else:
                            worse += 1
                    LOGGER.info(
                        "HF vs incumbent gen=%d: better=%d worse=%d unknown=%d (ok=%d failed=%d) ref=%s threshold=%s",
                        int(gen),
                        int(better),
                        int(worse),
                        int(unknown),
                        int(ok),
                        int(failed),
                        incumbent_ref_for_hf,
                        _score_threshold(
                            reference_score=incumbent_ref_for_hf,
                            metric_mode=metric_mode,
                            improve_eps=improve_eps,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    pass

            # Merge high-fidelity results back into cached records.
            for hf_rec in hf_results:
                gid = str(hf_rec.get("g_id"))
                fid = str(hf_rec.get("f_id"))
                cache_key = (gid, fid, str(eval_sig))
                merged = dict(pair_records_map.get((gid, fid), {}))
                merged.update(dict(hf_rec))
                merged["eval_budget_signature"] = str(eval_sig)
                merged["stage"] = "high_fidelity"
                merged["phase"] = pair_phase_by_pair.get((gid, fid), merged.get("phase", "coevo"))
                if isinstance(merged.get("fitness"), dict):
                    merged["fitness"]["cheap_only"] = False
                    merged["fitness"]["proxy_score"] = float(merged.get("proxy_score", merged["fitness"].get("fitness_score", float("inf"))))
                caches.set_pair(cache_key, merged)
                pair_records_map[(gid, fid)] = merged

            pair_records = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]
            for rec in pair_records:
                k = (str(rec.get("g_id")), str(rec.get("f_id")))
                rec["phase"] = pair_phase_by_pair.get(k, rec.get("phase", "coevo"))

            # Stage3 multi-fidelity promotion: re-run HF on a smaller pool with higher budgets.
            if bool(stage3_multifidelity_cfg.get("enabled")):
                rounds_raw = stage3_multifidelity_cfg.get("rounds") or []
                rounds = [dict(r) for r in rounds_raw if isinstance(r, dict)]
                if len(rounds) >= 2:
                    base_fidelity = _stage3_fidelity_key(cfg_yaml)
                    base_idx: int | None = None
                    for i, rc in enumerate(rounds):
                        cfg_i = _apply_stage3_round_overrides(cfg_yaml, rc)
                        if _stage3_fidelity_key(cfg_i) == base_fidelity:
                            base_idx = int(i)
                            break
                    if base_idx is None:
                        LOGGER.warning(
                            "stage3_multifidelity enabled but current fidelity=%s not found in rounds; assuming rounds[0] is the filter round.",
                            str(base_fidelity),
                        )
                        base_idx = 0

                    missing: List[str] = []
                    for scenario_entry in _iter_stage3_scenario_cfgs(cfg_yaml):
                        scenario_name = str(scenario_entry.get("name") or "scenario")
                        scenario_cfg = dict(scenario_entry.get("cfg") or {})
                        for rc in rounds[base_idx:]:
                            cfg_r = _apply_stage3_round_overrides(scenario_cfg, rc)
                            baseline_cfg_r = cfg_r.get("baseline", {}) or {}
                            if not _resolve_stage3_baseline_mini_eval_path(cfg_r, baseline_cfg_r):
                                missing.append(
                                    f"{scenario_name}:{str(rc.get('name') or _stage3_fidelity_key(cfg_r))}"
                                )
                    if missing:
                        raise RuntimeError(
                            "stage3_multifidelity enabled but missing baseline.mini_eval_paths entries for: "
                            + ", ".join(missing)
                        )

                    pool_pairs: List[Tuple[str, str]] = [(str(r.get("g_id")), str(r.get("f_id"))) for r in selected]
                    pool_pairs = [p for p in pool_pairs if p[0] and p[1]]

                    for i in range(int(base_idx), int(len(rounds) - 1)):
                        curr_round = dict(rounds[i])
                        next_round = dict(rounds[i + 1])

                        pool_records: List[Dict[str, Any]] = []
                        for gid, fid in pool_pairs:
                            rec = pair_records_map.get((str(gid), str(fid)))
                            if isinstance(rec, dict):
                                pool_records.append(rec)

                        incumbent_ref_score = None
                        if isinstance(best_so_far, dict):
                            try:
                                incumbent_ref_score = float(best_so_far.get("score"))
                            except (TypeError, ValueError):
                                incumbent_ref_score = None

                        promote_top_m = int(curr_round.get("promote_top_m", 0) or 0)
                        promote_top_frac_raw = curr_round.get("promote_top_frac")
                        if promote_top_frac_raw is None:
                            promote_top_frac = None
                        else:
                            try:
                                promote_top_frac = float(promote_top_frac_raw)
                            except (TypeError, ValueError):
                                promote_top_frac = None
                        promote_if_better = bool(curr_round.get("promote_if_better_than_incumbent", False))
                        promote_only_if_better_than_baseline = bool(
                            curr_round.get("promote_only_if_better_than_baseline", False)
                        )
                        promote_baseline_mode = str(curr_round.get("promote_baseline_mode", "mean") or "mean")
                        always_include_inc = bool(curr_round.get("always_include_incumbent", True))
                        always_pair = None
                        if always_include_inc and isinstance(best_so_far, dict):
                            inc_pair = (str(best_so_far.get("builder_id")), str(best_so_far.get("loss_id")))
                            if inc_pair in set(pool_pairs):
                                always_pair = inc_pair

                        if str(alternating_active_phase) == "builder" and alternating_fixed_loss_id:
                            promoted = _select_stage3_builder_promotions(
                                pool_records,
                                promote_top_m=int(promote_top_m),
                                promote_top_frac=promote_top_frac,
                                metric_mode=str(metric_mode),
                                slack=float(improve_eps),
                                fixed_loss_id=str(alternating_fixed_loss_id),
                                always_include_pair=always_pair,
                            )
                        else:
                            promoted = _select_stage3_promotions(
                                pool_records,
                                promote_top_m=int(promote_top_m),
                                promote_top_frac=promote_top_frac,
                                promote_if_better_than_incumbent=bool(promote_if_better),
                                promote_selection_mode=str(curr_round.get("promote_selection_mode", "union") or "union"),
                                promote_only_if_better_than_baseline=bool(promote_only_if_better_than_baseline),
                                promote_baseline_mode=str(promote_baseline_mode),
                                incumbent_ref_score=incumbent_ref_score,
                                incumbent_ref_record=(best_so_far if isinstance(best_so_far, Mapping) else None),
                                metric_mode=str(metric_mode),
                                improve_eps=float(improve_eps),
                                always_include_pair=always_pair,
                                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
                            )
                        always_promote_best = True if curr_round.get("always_promote_best") is None else bool(
                            curr_round.get("always_promote_best")
                        )
                        if always_promote_best and pool_records:
                            best_rec = None
                            for rec in pool_records:
                                if not bool(rec.get("pair_ok")):
                                    continue
                                if best_rec is None or _pair_record_beats_reference_record(
                                    rec,
                                    best_rec,
                                    metric_mode=str(metric_mode),
                                    improve_eps=0.0,
                                    prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
                                ):
                                    best_rec = rec
                            if best_rec is not None:
                                best_pair = (str(best_rec.get("g_id")), str(best_rec.get("f_id")))
                                if best_pair not in promoted:
                                    promoted.insert(0, best_pair)

                        uniq: List[Tuple[str, str]] = []
                        seen_prom: set[Tuple[str, str]] = set()
                        for p in promoted:
                            if p in seen_prom:
                                continue
                            uniq.append(p)
                            seen_prom.add(p)
                        promoted = uniq

                        LOGGER.info(
                            "HF MF promote gen=%d: %s -> %s next_pool=%d (from=%d) top_m=%d top_frac=%s promote_if_better=%s selection_mode=%s promote_if_better_than_baseline=%s baseline_mode=%s include_incumbent=%s",
                            int(gen),
                            str(curr_round.get("name") or _stage3_fidelity_key(_apply_stage3_round_overrides(cfg_yaml, curr_round))),
                            str(next_round.get("name") or _stage3_fidelity_key(_apply_stage3_round_overrides(cfg_yaml, next_round))),
                            int(len(promoted)),
                            int(len(pool_pairs)),
                            int(promote_top_m),
                            (str(promote_top_frac) if promote_top_frac is not None else "None"),
                            str(bool(promote_if_better)),
                            str(curr_round.get("promote_selection_mode", "union") or "union"),
                            str(bool(promote_only_if_better_than_baseline)),
                            str(promote_baseline_mode),
                            str(bool(always_include_inc)),
                        )
                        if not promoted:
                            break

                        cfg_round = _apply_stage3_round_overrides(cfg_yaml, next_round)
                        fidelity_key = _stage3_fidelity_key(cfg_round)
                        round_name = str(next_round.get("name") or fidelity_key)
                        sig_hf_round = _build_hf_cfg(cfg_round, seed=int(seed), device_str="cpu")
                        eval_sig_round = eval_budget_signature(
                            cfg=sig_hf_round,
                            proxy_problem_size=proxy_problem_size,
                            proxy_batch_size=proxy_batch_size,
                            proxy_batches=proxy_batches,
                            proxy_weights={str(k): float(v) for k, v in dict(proxy_weights).items()},
                            extra_budget=micro_budget,
                        )

                        hf_tasks2: List[Dict[str, Any]] = []
                        for gid, fid in promoted:
                            cache_key = (str(gid), str(fid), str(eval_sig_round))
                            cached = caches.get_pair(cache_key)
                            if isinstance(cached, dict) and str(cached.get("stage")) == "high_fidelity" and cached.get("fitness"):
                                continue
                            g_entry = g_map.get(str(gid))
                            if not isinstance(g_entry, dict) and str(gid) == G_REF_ID:
                                g_entry = {"id": str(G_REF_ID), "ir": asdict(_ref_builder_ir())}
                            f_entry = f_map.get(str(fid))
                            if not isinstance(f_entry, dict) and str(fid) == F_REF_ID:
                                f_entry = {"id": str(F_REF_ID), "ir": asdict(_ref_loss_ir())}
                            proxy_rec = pair_records_map.get((str(gid), str(fid)))
                            if isinstance(proxy_rec, dict) and isinstance(proxy_rec.get("g_ir"), dict):
                                g_entry = {"id": str(gid), "ir": dict(proxy_rec.get("g_ir") or {})}
                            if isinstance(proxy_rec, dict) and isinstance(proxy_rec.get("f_ir"), dict):
                                f_entry = {"id": str(fid), "ir": dict(proxy_rec.get("f_ir") or {})}
                            if not isinstance(g_entry, dict) or not isinstance(f_entry, dict):
                                continue
                            device_str = device_list[int(len(hf_tasks2)) % len(device_list)]
                            hf_tasks2.append(
                                {
                                    "generation": int(gen),
                                    "pair_index": _safe_int((proxy_rec or {}).get("pair_index", -1), -1),
                                    "g_entry": dict(g_entry),
                                    "f_entry": dict(f_entry),
                                    "cfg_yaml": dict(cfg_round),
                                    "device_str": device_str,
                                    "operator_whitelist": list(operator_whitelist),
                                    "run_dir": str(run_dir),
                                    "cheap_gate_on": False,
                                    "high_fidelity_on": True,
                                    "eval_budget_signature": str(eval_sig_round),
                                    "proxy_record": dict(proxy_rec) if isinstance(proxy_rec, dict) else None,
                                    "baseline_epoch_objectives": list(baseline_epoch_objectives) if baseline_epoch_objectives else None,
                                    "baseline_early_valid": baseline_early_valid,
                                    "early_eval_steps": int(early_eval_steps),
                                }
                            )

                        hf_results2: List[Dict[str, Any]] = []
                        if hf_tasks2:
                            LOGGER.info(
                                "HF MF round gen=%d: %s fidelity=%s pool=%d tasks=%d mp=%s procs=%d devices=%s",
                                int(gen),
                                str(round_name),
                                str(fidelity_key),
                                int(len(promoted)),
                                int(len(hf_tasks2)),
                                str(mp_enabled),
                                int(mp_processes),
                                dict(collections.Counter(str(t.get("device_str", "")) for t in hf_tasks2)),
                            )
                            hf_scheduler_mode = _hf_scheduler_mode(cfg_round)
                            if mp_enabled and mp_processes > 0 and len(hf_tasks2) > 1:
                                procs = min(int(mp_processes), max(1, len(hf_tasks2)), max(1, len(device_list)))
                                if hf_scheduler_mode == "subprocess":
                                    hf_results2.extend(
                                        _run_hf_tasks_via_subprocess(
                                            hf_tasks=hf_tasks2,
                                            run_dir=str(run_dir),
                                            device_list=device_list,
                                            max_workers=int(procs),
                                            runtime_trace=runtime_trace,
                                        )
                                    )
                                else:
                                    import multiprocessing as mp

                                    ctx = mp.get_context(mp_start_method)
                                    task_queue: Any = ctx.Queue()
                                    result_queue: Any = ctx.Queue()
                                    workers: List[Any] = []
                                    try:
                                        for task in hf_tasks2:
                                            task_queue.put(dict(task))
                                        for _ in range(int(procs)):
                                            task_queue.put(None)
                                        for w_idx in range(int(procs)):
                                            dev = device_list[int(w_idx) % len(device_list)]
                                            p = ctx.Process(
                                                target=_hf_pinned_device_worker,
                                                args=(str(dev), task_queue, result_queue),
                                            )
                                            p.daemon = False
                                            p.start()
                                            workers.append(p)
                                        for _ in range(int(len(hf_tasks2))):
                                            hf_results2.append(dict(result_queue.get()))
                                    finally:
                                        for p in workers:
                                            p.join()
                            else:
                                for task in hf_tasks2:
                                    hf_results2.append(_evaluate_pair_worker(task))
                            for hf_rec2 in hf_results2:
                                joint_gate_repair_attempt_records_gen.extend(
                                    _pop_joint_gate_repair_reports(hf_rec2)
                                )

                            fatal2 = [r for r in hf_results2 if isinstance(r, dict) and r.get("fatal_error")]
                            if fatal2:
                                first = dict(fatal2[0])
                                raise RuntimeError(
                                    f"Stage3 fatal error (round={round_name} pair={first.get('g_id')},{first.get('f_id')}): {first.get('fatal_error')}"
                                )

                        for hf_rec in hf_results2:
                            gid = str(hf_rec.get("g_id"))
                            fid = str(hf_rec.get("f_id"))
                            cache_key = (gid, fid, str(eval_sig_round))
                            merged = dict(pair_records_map.get((gid, fid), {}))
                            merged.update(dict(hf_rec))
                            merged["eval_budget_signature"] = str(eval_sig_round)
                            merged["stage"] = "high_fidelity"
                            merged["hf_round_name"] = str(round_name)
                            merged["hf_fidelity_key"] = str(fidelity_key)
                            merged["phase"] = pair_phase_by_pair.get((gid, fid), merged.get("phase", "coevo"))
                            if isinstance(merged.get("fitness"), dict):
                                merged["fitness"]["cheap_only"] = False
                                merged["fitness"]["proxy_score"] = float(
                                    merged.get("proxy_score", merged["fitness"].get("fitness_score", float("inf")))
                                )
                            caches.set_pair(cache_key, merged)
                            pair_records_map[(gid, fid)] = merged

                        pool_pairs = list(promoted)

                        pair_records = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]
                        for rec in pair_records:
                            k = (str(rec.get("g_id")), str(rec.get("f_id")))
                            rec["phase"] = pair_phase_by_pair.get(k, rec.get("phase", "coevo"))

        # Stage annotations + incumbent-gating (better_than_incumbent).
        for rec in pair_records:
            rec["stages_enabled"] = dict(eval_stages)
            ran, skipped = _annotate_stage_fields(rec, eval_stages=eval_stages)
            rec["stages_ran"] = list(ran)
            rec["stages_skipped"] = dict(skipped)
            stage_final, final_score = _resolve_final_score(rec, eval_stages=eval_stages)
            rec["stage_final"] = str(stage_final)
            rec["final_score"] = final_score
            rec["metric_mode"] = str(metric_mode)
            rec["compare_target"] = "incumbent"
            rec["improve_eps"] = float(improve_eps)
            rec["last_phase_label"] = str(last_phase_block_label) if last_phase_block_label is not None else None
            rec["last_phase_reference_score"] = (
                float(last_phase_block_best_score) if last_phase_block_best_score is not None else None
            )
            if str(stage_final) == "micro_unroll":
                rec["better_than_incumbent_note"] = (
                    "stage2_micro_unroll compares final_score (micro_score) against incumbent score."
                )
            elif str(stage_final) == "high_fidelity":
                rec["better_than_incumbent_note"] = (
                    "stage3_high_fidelity compares final_score (HF score) against incumbent score."
                )
            else:
                rec["better_than_incumbent_note"] = None

        last_phase_reference_score = None
        if last_phase_block_best_score is not None:
            try:
                last_phase_reference_score = float(last_phase_block_best_score)
            except (TypeError, ValueError):
                last_phase_reference_score = None

        current_ref = None
        if isinstance(best_so_far, dict):
            try:
                current_ref = float(best_so_far.get("score"))
            except (TypeError, ValueError):
                current_ref = None
        LOGGER.info(
            "COMPARE target=incumbent reference_score=%s threshold=%s metric_mode=%s improve_eps=%s",
            current_ref,
            _score_threshold(reference_score=current_ref, metric_mode=metric_mode, improve_eps=improve_eps),
            str(metric_mode),
            float(improve_eps),
        )
        LOGGER.info(
            "COMPARE target=last_phase last_phase=%s reference_score=%s threshold=%s metric_mode=%s improve_eps=%s",
            (str(last_phase_block_label) if last_phase_block_label is not None else None),
            last_phase_reference_score,
            _score_threshold(
                reference_score=last_phase_reference_score,
                metric_mode=metric_mode,
                improve_eps=improve_eps,
            ),
            str(metric_mode),
            float(improve_eps),
        )

        for rec in sorted(
            pair_records,
            key=lambda r: (
                _safe_int(r.get("pair_index", 10**9), 10**9),
                str(r.get("g_id", "")),
                str(r.get("f_id", "")),
            ),
        ):
            reference_score = None
            if isinstance(best_so_far, dict):
                try:
                    reference_score = float(best_so_far.get("score"))
                except (TypeError, ValueError):
                    reference_score = None
            rec["reference_score"] = reference_score
            if (
                (str(rec.get("g_id")) == G_REF_ID and str(rec.get("f_id")) == F_REF_ID)
                or str(rec.get("stage")) == "anchor"
            ):
                rec["better_than_incumbent"] = False
                rec["delta_vs_incumbent"] = None
                rec["better_than_last_phase"] = (False if last_phase_reference_score is not None else None)
                rec["delta_vs_last_phase"] = None
                continue
            if not bool(rec.get("pair_ok")):
                rec["better_than_incumbent"] = False
                rec["delta_vs_incumbent"] = None
                rec["better_than_last_phase"] = (False if last_phase_reference_score is not None else None)
                rec["delta_vs_last_phase"] = None
                rec["final_score"] = None
                rec["stage_final"] = "none"
                continue

            final_score = rec.get("final_score")
            if final_score is None:
                rec["better_than_incumbent"] = False
                rec["delta_vs_incumbent"] = None
                rec["better_than_last_phase"] = (False if last_phase_reference_score is not None else None)
                rec["delta_vs_last_phase"] = None
                continue
            try:
                cand_score_f = float(final_score)
            except (TypeError, ValueError):
                rec["better_than_incumbent"] = False
                rec["delta_vs_incumbent"] = None
                rec["better_than_last_phase"] = (False if last_phase_reference_score is not None else None)
                rec["delta_vs_last_phase"] = None
                rec["final_score"] = None
                rec["stage_final"] = "none"
                continue
            if not math.isfinite(cand_score_f):
                rec["better_than_incumbent"] = False
                rec["delta_vs_incumbent"] = None
                rec["better_than_last_phase"] = (False if last_phase_reference_score is not None else None)
                rec["delta_vs_last_phase"] = None
                continue

            delta = _score_delta(cand_score=cand_score_f, ref_score=reference_score, metric_mode=metric_mode)
            better = _pair_record_beats_reference_record(
                rec,
                (best_so_far if isinstance(best_so_far, Mapping) else None),
                metric_mode=metric_mode,
                improve_eps=improve_eps,
                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
            )
            rec["better_than_incumbent"] = bool(better)
            rec["delta_vs_incumbent"] = delta
            if last_phase_reference_score is None:
                rec["better_than_last_phase"] = None
                rec["delta_vs_last_phase"] = None
            else:
                rec["delta_vs_last_phase"] = _score_delta(
                    cand_score=cand_score_f,
                    ref_score=last_phase_reference_score,
                    metric_mode=metric_mode,
                )
                rec["better_than_last_phase"] = bool(
                    _is_better_than_reference(
                        cand_score=cand_score_f,
                        reference_score=last_phase_reference_score,
                        metric_mode=metric_mode,
                        improve_eps=improve_eps,
                    )
                )
            rec["score"] = float(cand_score_f)
            if bool(better):
                threshold = _score_threshold(reference_score=reference_score, metric_mode=metric_mode, improve_eps=improve_eps)
                best_so_far = {
                    "score": float(cand_score_f),
                    "builder_id": str(rec.get("g_id")),
                    "loss_id": str(rec.get("f_id")),
                    "stage_final": str(rec.get("stage_final", "none")),
                    "generation": int(gen),
                    "phase": str(rec.get("phase", "coevo")),
                    "stage3_all_scenarios_negative": rec.get("stage3_all_scenarios_negative"),
                    "stage3_nonnegative_scenario_count": rec.get("stage3_nonnegative_scenario_count"),
                    "stage3_worst_scenario_delta": rec.get("stage3_worst_scenario_delta"),
                }
                LOGGER.info(
                    "NEW BEST: score=%s ref=%s delta=%s pair=(%s,%s) stage=%s gen=%d phase=%s threshold=%s compare_target=incumbent improve_eps=%s",
                    float(cand_score_f),
                    reference_score,
                    delta,
                    rec.get("g_id"),
                    rec.get("f_id"),
                    rec.get("stage_final"),
                    int(gen),
                    rec.get("phase", "coevo"),
                    threshold,
                    float(improve_eps),
                )

        for rec in pair_records:
            _append_pair_score_history(pair_score_history_map, rec)

        improved_this_gen = False
        if isinstance(best_so_far, dict):
            try:
                improved_this_gen = int(best_so_far.get("generation", -999)) == int(gen)
            except (TypeError, ValueError):
                improved_this_gen = False
        stagnation_generations = 0 if improved_this_gen else int(stagnation_generations) + 1

        gen_phase_best_score: float | None = None
        for rec in pair_records:
            if (
                (str(rec.get("g_id")) == G_REF_ID and str(rec.get("f_id")) == F_REF_ID)
                or str(rec.get("stage")) == "anchor"
            ):
                continue
            if not bool(rec.get("pair_ok")):
                continue
            try:
                score_f = float(rec.get("final_score"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_f):
                continue
            if _is_better_than_reference(
                cand_score=score_f,
                reference_score=gen_phase_best_score,
                metric_mode=metric_mode,
                improve_eps=0.0,
            ):
                gen_phase_best_score = score_f

        if gen_phase_best_score is not None and _is_better_than_reference(
            cand_score=float(gen_phase_best_score),
            reference_score=phase_block_best_score,
            metric_mode=metric_mode,
            improve_eps=0.0,
        ):
            phase_block_best_score = float(gen_phase_best_score)

        # Per-generation summary for troubleshooting.
        stage_ctr = collections.Counter(str(r.get("stage", "")) for r in pair_records)
        ok_ctr = sum(1 for r in pair_records if bool(r.get("pair_ok")))
        reason_ctr = collections.Counter(str(r.get("pair_reason", "")) for r in pair_records)
        LOGGER.info(
            "Gen %d summary: g_pool=%d f_pool=%d pairs=%d ok=%d stages=%s top_reasons=%s",
            int(gen),
            int(len(g_pool)),
            int(len(f_pool)),
            int(len(pair_records)),
            int(ok_ctr),
            dict(stage_ctr),
            reason_ctr.most_common(3),
        )
        if gate_repair_records_gen:
            repair_applied = sum(1 for r in gate_repair_records_gen if bool(r.get("repair_applied")))
            repair_pair_ok = sum(1 for r in gate_repair_records_gen if bool((r.get("post") or {}).get("pair_ok")))
            repair_event_ctr: collections.Counter[str] = collections.Counter()
            for rr in gate_repair_records_gen:
                for ev in rr.get("events", []) or []:
                    if not isinstance(ev, dict):
                        continue
                    side = str(ev.get("side", "unknown"))
                    status = str(ev.get("status", "unknown"))
                    repair_event_ctr[f"{side}:{status}"] += 1
            LOGGER.info(
                "Gen %d gate-repair summary: attempted_pairs=%d applied=%d post_pair_ok=%d events=%s",
                int(gen),
                int(len(gate_repair_records_gen)),
                int(repair_applied),
                int(repair_pair_ok),
                dict(repair_event_ctr),
            )

        # Candidate-level failures (useful for repair prompts).
        g_fail_compile = sum(1 for e in g_entries if not bool(e.get("compile_ok")))
        g_fail_gate = sum(1 for e in g_entries if bool(e.get("compile_ok")) and not bool(e.get("builder_static_ok", True)))
        f_fail_static = sum(1 for e in f_entries if not bool(e.get("static_ok", True)))
        f_fail_compile = sum(1 for e in f_entries if bool(e.get("static_ok", True)) and not bool(e.get("compile_ok", False)))

        # Pair-level gate failure kinds.
        gate_kind_ctr: collections.Counter[str] = collections.Counter()
        for rec in pair_records:
            for k in (
                "builder_gate_trace",
                "joint_gate_trace",
                "pref_semantic_trace",
                "co_sensitivity_visible_trace",
                "co_sensitivity_hidden_trace",
                "co_invariance_visible_trace",
                "co_invariance_hidden_trace",
            ):
                t = rec.get(k)
                if not isinstance(t, dict):
                    continue
                kind = t.get("failure_kind") or t.get("failed_gate")
                if kind is None:
                    continue
                gate_kind_ctr[str(kind)] += 1

        # Best pair preview (for coevolution guidance).
        best_pair_preview: Dict[str, Any] | None = None
        for rec in pair_records:
            if str(rec.get("g_id")) == G_REF_ID and str(rec.get("f_id")) == F_REF_ID:
                continue
            if str(rec.get("stage")) == "anchor":
                continue
            if not bool(rec.get("pair_ok")):
                continue
            if best_pair_preview is None or _pair_record_beats_reference_record(
                rec,
                best_pair_preview,
                metric_mode=metric_mode,
                improve_eps=0.0,
                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
            ):
                score_f = _pair_record_effective_score(rec)
                if score_f is None:
                    continue
                best_pair_preview = {
                    "g_id": rec.get("g_id"),
                    "f_id": rec.get("f_id"),
                    "score": score_f,
                    "stage": rec.get("stage"),
                    "stage3_all_scenarios_negative": rec.get("stage3_all_scenarios_negative"),
                    "stage3_nonnegative_scenario_count": rec.get("stage3_nonnegative_scenario_count"),
                    "stage3_worst_scenario_delta": rec.get("stage3_worst_scenario_delta"),
                    "proxy_score": rec.get("proxy_score"),
                    "pair_reason": rec.get("pair_reason"),
                    "builder_gate_reason": rec.get("builder_gate_reason"),
                    "joint_gate_reason": rec.get("joint_gate_reason"),
                }

        # Feed back coarse failure modes to LLM in the next generation.
        llm_feedback_state = {
            "prev_gen_summary": {
                "stage_counts": dict(stage_ctr),
                "ok_pairs": int(ok_ctr),
                "top_pair_reasons": list(reason_ctr.most_common(8)),
                "top_gate_failure_kinds": list(gate_kind_ctr.most_common(8)),
            }
        }
        llm_feedback_state["stagnation_generations"] = int(stagnation_generations)
        try:
            fam_ctr = collections.Counter(str(e.get("family") or "unknown") for e in elites_f if isinstance(e, dict))
            llm_feedback_state["prev_gen_loss_families"] = list(fam_ctr.most_common(10))
        except Exception:  # noqa: BLE001
            llm_feedback_state["prev_gen_loss_families"] = []
        llm_feedback_state["prev_gen_candidates"] = {
            "g_fail_compile": int(g_fail_compile),
            "g_fail_gate": int(g_fail_gate),
            "f_fail_static": int(f_fail_static),
            "f_fail_compile": int(f_fail_compile),
            "pop_g": int(len(g_entries)),
            "pop_f": int(len(f_entries)),
        }
        if gate_repair_records_gen:
            llm_feedback_state["prev_gen_gate_repair"] = {
                "attempted_pairs": int(len(gate_repair_records_gen)),
                "applied_pairs": int(sum(1 for r in gate_repair_records_gen if bool(r.get("repair_applied")))),
                "post_pair_ok_pairs": int(sum(1 for r in gate_repair_records_gen if bool((r.get("post") or {}).get("pair_ok")))),
                "top_pre_pair_reasons": list(
                    collections.Counter(
                        str((r.get("pre") or {}).get("pair_reason", "")) for r in gate_repair_records_gen
                    ).most_common(6)
                ),
                "top_post_pair_reasons": list(
                    collections.Counter(
                        str((r.get("post") or {}).get("pair_reason", "")) for r in gate_repair_records_gen
                    ).most_common(6)
                ),
            }
        llm_feedback_state["prev_gen_best_pair_preview"] = best_pair_preview
        try:
            llm_feedback_state["prev_gen_llm_cache"] = dict(loss_llm_ops.llm_cache_stats())
        except Exception:  # noqa: BLE001
            pass
        if stage1_proxy_enabled and "cheap" not in stage_ctr and "cheap_recheck" not in stage_ctr:
            LOGGER.warning(
                "Gen %d produced no core cheap evaluations (stage='cheap'). "
                "This usually means candidate pools collapsed or pairing_budget_per_gen is too small after anchors.",
                int(gen),
            )
        if stage3_hf_enabled and "high_fidelity" not in stage_ctr:
            LOGGER.warning(
                "Gen %d produced no high-fidelity evaluations (stage='high_fidelity'). "
                "Check proxy gates/thresholds: if all pairs are pair_ok=False, HF won't run.",
                int(gen),
            )

        gate_records: List[Dict[str, Any]] = []
        for rec in pair_records:
            gate_records.append(
                {
                    "generation": int(rec.get("generation", gen)),
                    "pair_index": int(rec.get("pair_index", -1)),
                    "g_id": rec.get("g_id"),
                    "f_id": rec.get("f_id"),
                    "stage": rec.get("stage"),
                    "phase": rec.get("phase"),
                    "score": rec.get("score"),
                    "proxy_metrics": rec.get("proxy_metrics"),
                    "seed_signature": rec.get("seed_signature"),
                    "descriptor": rec.get("descriptor"),
                    "stages_enabled": rec.get("stages_enabled"),
                    "stages_ran": rec.get("stages_ran"),
                    "stages_skipped": rec.get("stages_skipped"),
                    "stage_final": rec.get("stage_final"),
                    "final_score": rec.get("final_score"),
                    "reference_score": rec.get("reference_score"),
                    "improve_eps": rec.get("improve_eps"),
                    "better_than_incumbent": rec.get("better_than_incumbent"),
                    "delta_vs_incumbent": rec.get("delta_vs_incumbent"),
                    "better_than_last_phase": rec.get("better_than_last_phase"),
                    "delta_vs_last_phase": rec.get("delta_vs_last_phase"),
                    "last_phase_label": rec.get("last_phase_label"),
                    "last_phase_reference_score": rec.get("last_phase_reference_score"),
                    "better_than_incumbent_note": rec.get("better_than_incumbent_note"),
                    "compare_target": rec.get("compare_target"),
                    "static_ok": rec.get("f_static_ok"),
                    "static_reason": rec.get("f_static_reason"),
                    "builder_gate_ok": rec.get("builder_gate_ok"),
                    "builder_gate_reason": rec.get("builder_gate_reason"),
                    "builder_gate_trace": rec.get("builder_gate_trace"),
                    "joint_gate_ok": rec.get("joint_gate_ok"),
                    "joint_gate_reason": rec.get("joint_gate_reason"),
                    "joint_gate_trace": rec.get("joint_gate_trace"),
                    "co_ok": rec.get("co_ok"),
                    "co_reason": rec.get("co_reason"),
                    "co_failed_gate": rec.get("co_failed_gate"),
                    "co_failure_kind": rec.get("co_failure_kind"),
                    "co_sensitivity_ok": rec.get("co_sensitivity_ok"),
                    "co_invariance_ok": rec.get("co_invariance_ok"),
                    "co_sensitivity_visible_ok": rec.get("co_sensitivity_visible_ok"),
                    "co_sensitivity_visible_reason": rec.get("co_sensitivity_visible_reason"),
                    "co_sensitivity_visible_abs_delta": rec.get("co_sensitivity_visible_abs_delta"),
                    "co_sensitivity_visible_rel_delta": rec.get("co_sensitivity_visible_rel_delta"),
                    "co_sensitivity_visible_trace": rec.get("co_sensitivity_visible_trace"),
                    "co_sensitivity_hidden_ok": rec.get("co_sensitivity_hidden_ok"),
                    "co_sensitivity_hidden_reason": rec.get("co_sensitivity_hidden_reason"),
                    "co_sensitivity_hidden_abs_delta": rec.get("co_sensitivity_hidden_abs_delta"),
                    "co_sensitivity_hidden_rel_delta": rec.get("co_sensitivity_hidden_rel_delta"),
                    "co_sensitivity_hidden_trace": rec.get("co_sensitivity_hidden_trace"),
                    "co_invariance_visible_ok": rec.get("co_invariance_visible_ok"),
                    "co_invariance_visible_reason": rec.get("co_invariance_visible_reason"),
                    "co_invariance_visible_abs_delta": rec.get("co_invariance_visible_abs_delta"),
                    "co_invariance_visible_rel_delta": rec.get("co_invariance_visible_rel_delta"),
                    "co_invariance_visible_trace": rec.get("co_invariance_visible_trace"),
                    "co_invariance_hidden_ok": rec.get("co_invariance_hidden_ok"),
                    "co_invariance_hidden_reason": rec.get("co_invariance_hidden_reason"),
                    "co_invariance_hidden_abs_delta": rec.get("co_invariance_hidden_abs_delta"),
                    "co_invariance_hidden_rel_delta": rec.get("co_invariance_hidden_rel_delta"),
                    "co_invariance_hidden_trace": rec.get("co_invariance_hidden_trace"),
                    "pref_semantic_ok": rec.get("pref_semantic_ok"),
                    "pref_semantic_reason": rec.get("pref_semantic_reason"),
                    "pref_semantic_trace": rec.get("pref_semantic_trace"),
                    "dynamic_ok": None,
                    "dynamic_visible_trace": None,
                    "dynamic_hidden_trace": None,
                    "pair_ok": rec.get("pair_ok"),
                    "pair_reason": rec.get("pair_reason"),
                    "gate_repair": rec.get("gate_repair"),
                }
            )

        _append_jsonl(pairs_jsonl, pair_records)
        _append_jsonl(gate_jsonl, gate_records)
        _append_jsonl(gate_repair_jsonl, gate_repair_records_gen)
        _append_jsonl(gate_repair_jsonl, joint_gate_repair_attempt_records_gen)

        fitness_g, fitness_f = _credit_assignment_v2(pair_records=pair_records)
        builder_selection_active = not (_uses_fixed_side_search(search_mode) and str(alternating_active_phase) == "loss")
        loss_selection_active = not (_uses_fixed_side_search(search_mode) and str(alternating_active_phase) == "builder")
        builder_perf_map: Dict[str, float] = {}
        builder_constraint_state: Dict[str, Any] | None = None
        builder_ok_records = [
            r
            for r in pair_records
            if bool(r.get("pair_ok")) and str(r.get("stage")) != "anchor"
        ]
        if builder_selection_active and builder_ok_records:
            builder_perf_map, _ = _credit_assignment_v2(pair_records=builder_ok_records)
            builder_constraint_state = _compute_builder_constraint_state(
                records=builder_ok_records,
                perf_by_builder=builder_perf_map,
                metric_mode=metric_mode,
                slack=improve_eps,
            )
            if isinstance(builder_constraint_state, Mapping):
                for gid, stat in dict(builder_constraint_state.get("builders") or {}).items():
                    if not isinstance(stat, Mapping) or not str(gid):
                        continue
                    stat_copy = dict(stat)
                    stat_copy["generation"] = int(gen)
                    stat_copy["phase"] = str(generation_phase_label)
                    builder_cost_archive[str(gid)] = _merge_builder_archive_entry(
                        builder_cost_archive.get(str(gid)),
                        stat_copy,
                        metric_mode=metric_mode,
                    )

                global_builder_selection = _select_best_builder_cost_from_archive(
                    builder_archive=builder_cost_archive,
                    metric_mode=metric_mode,
                    slack=improve_eps,
                )
                builder_candidate = (
                    dict(global_builder_selection.get("selected"))
                    if isinstance(global_builder_selection, Mapping) and isinstance(global_builder_selection.get("selected"), Mapping)
                    else None
                )
                if builder_candidate is not None:
                    incumbent_id = str(best_builder_cost.get("builder_id")) if isinstance(best_builder_cost, dict) else None
                    candidate_id = str(builder_candidate.get("builder_id"))
                    incumbent_cost = None
                    candidate_cost = None
                    if isinstance(best_builder_cost, dict):
                        try:
                            incumbent_cost = float(best_builder_cost.get("cost"))
                        except (TypeError, ValueError):
                            incumbent_cost = None
                    try:
                        candidate_cost = float(builder_candidate.get("cost"))
                    except (TypeError, ValueError):
                        candidate_cost = None
                    should_update_builder_cost = (
                        incumbent_id != candidate_id
                        or incumbent_cost is None
                        or candidate_cost is None
                        or abs(float(candidate_cost) - float(incumbent_cost)) > 1e-12
                    )
                    if should_update_builder_cost:
                        best_builder_cost = {
                            "builder_id": str(builder_candidate.get("builder_id")),
                            "cost": float(builder_candidate.get("cost")),
                            "perf": float(builder_candidate.get("perf")),
                            "metric_mode": str(metric_mode),
                            "slack": float(improve_eps),
                            "best_perf": global_builder_selection.get("best_perf"),
                            "best_perf_anchor": global_builder_selection.get("best_perf"),
                            "threshold": global_builder_selection.get("threshold"),
                            "generation": int(builder_candidate.get("generation", gen) or gen),
                            "phase": str(builder_candidate.get("phase", generation_phase_label)),
                            "perf_ref": dict(builder_candidate.get("perf_ref") or {}),
                            "feasible_count": int(len(global_builder_selection.get("feasible") or [])),
                        }
                        LOGGER.info(
                            "NEW BEST_BUILDER_COST: builder=%s cost=%s perf=%s best_perf_anchor=%s slack=%s gen=%d phase=%s",
                            best_builder_cost.get("builder_id"),
                            best_builder_cost.get("cost"),
                            best_builder_cost.get("perf"),
                            best_builder_cost.get("best_perf_anchor"),
                            best_builder_cost.get("slack"),
                            int(best_builder_cost.get("generation", gen) or gen),
                            str(best_builder_cost.get("phase", generation_phase_label)),
                        )

        def _candidate_descriptor(cid: str, *, kind: str) -> Dict[str, Any]:
            xs: List[float] = []
            ys: List[float] = []
            cell = None
            for rec in pair_records:
                if kind == "g" and str(rec.get("g_id")) != str(cid):
                    continue
                if kind == "f" and str(rec.get("f_id")) != str(cid):
                    continue
                desc = rec.get("descriptor")
                if not isinstance(desc, dict):
                    continue
                d = desc.get("g" if kind == "g" else "f")
                if not isinstance(d, dict):
                    continue
                try:
                    xs.append(float(d.get("x", 0.0)))
                    ys.append(float(d.get("y", 0.0)))
                except (TypeError, ValueError):
                    continue
                if cell is None and isinstance(d.get("cell"), (list, tuple)) and len(d["cell"]) == 2:
                    cell = (int(d["cell"][0]), int(d["cell"][1]))
            if not xs:
                return {"x": 0.0, "y": 0.0, "cell": [0, 0]}
            mx = float(sum(xs) / len(xs))
            my = float(sum(ys) / len(ys))
            if cell is None:
                cell = _descriptor_cell(mx, my, bins=int(cfg_yaml.get("archive_bins", 8) or 8))
            return {"x": mx, "y": my, "cell": [int(cell[0]), int(cell[1])]}

        def _rank_entries(entries: Sequence[Mapping[str, Any]], fit_map: Mapping[str, float], *, kind: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for e in entries:
                eid = str(e.get("id"))
                if kind == "g" and eid == G_REF_ID:
                    continue
                if kind == "f" and eid == F_REF_ID:
                    continue
                e2 = dict(e)
                e2["fitness"] = float(fit_map.get(eid, float("inf")))
                e2["descriptor"] = _candidate_descriptor(eid, kind=kind)
                if kind == "g" and isinstance(builder_constraint_state, Mapping):
                    builder_stat = (builder_constraint_state.get("builders") or {}).get(eid)
                    if isinstance(builder_stat, Mapping):
                        e2["builder_perf"] = builder_stat.get("perf")
                        e2["builder_cost"] = builder_stat.get("cost")
                        e2["builder_feasible"] = builder_stat.get("feasible")
                        e2["builder_threshold"] = builder_stat.get("threshold")
                        e2["builder_best_perf"] = builder_stat.get("best_perf")
                        e2["selection_sort_key"] = list(builder_stat.get("selection_sort_key", []))
                if kind == "g" and "family_signature" not in e2:
                    try:
                        irj = e2.get("ir")
                        if isinstance(irj, dict):
                            e2["family_signature"] = str(_builder_family_signature(pref_builder_ir_from_json(irj)))
                    except Exception:  # noqa: BLE001
                        e2["family_signature"] = "unknown"
                if kind == "f" and "family_signature" not in e2:
                    try:
                        irj = e2.get("ir")
                        if isinstance(irj, dict):
                            e2["family_signature"] = str(_loss_family_signature(free_loss_ir_from_json(irj)))
                    except Exception:  # noqa: BLE001
                        e2["family_signature"] = "unknown"
                if kind == "f" and "family" not in e2:
                    try:
                        irj = e2.get("ir")
                        if isinstance(irj, dict):
                            e2["family"] = str(_loss_family_id(free_loss_ir_from_json(irj)))
                    except Exception:  # noqa: BLE001
                        e2["family"] = "unknown"
                out.append(e2)
            if kind == "g" and isinstance(builder_constraint_state, Mapping):
                out.sort(key=lambda x: _stored_selection_sort_key(x, fallback_key="fitness"))
            else:
                out.sort(
                    key=lambda x: float(x.get("fitness", float("-inf") if str(metric_mode) == "maximize" else float("inf"))),
                    reverse=bool(str(metric_mode) == "maximize"),
                )
            return out

        builder_family_div = _normalize_family_diversity_cfg((cfg_yaml.get("builder_llm", {}) or {}).get("family_diversity", {}))  # type: ignore[union-attr]
        loss_family_div = _normalize_family_diversity_cfg((cfg_yaml.get("loss_llm", {}) or {}).get("family_diversity", {}))  # type: ignore[union-attr]
        if builder_selection_active:
            ranked_g = _rank_entries(list(g_population_map.values()), fitness_g, kind="g")
            resident_pop_g = _select_resident_population(
                ranked_g,
                max(0, pop_g),
                metric_mode=metric_mode,
                family_diversity_cfg=builder_family_div,
                prefer_selection_sort_key=True,
            )
            elites_g = resident_pop_g[: max(0, elite_g)]
        else:
            ranked_g = [dict(e) for e in resident_pop_g]
            elites_g = resident_pop_g[: max(0, elite_g)]
            LOGGER.info(
                "Builder selection frozen gen=%d phase=%s; keeping previous builder resident population/archives.",
                int(gen),
                str(alternating_active_phase),
            )

        if loss_selection_active:
            ranked_f = _rank_entries(list(f_population_map.values()), fitness_f, kind="f")
            resident_pop_f = _select_resident_population(
                ranked_f,
                max(0, pop_f),
                metric_mode=metric_mode,
                family_diversity_cfg=loss_family_div,
            )
            elites_f = resident_pop_f[: max(0, elite_f)]
        else:
            ranked_f = [dict(e) for e in resident_pop_f]
            elites_f = resident_pop_f[: max(0, elite_f)]
            LOGGER.info(
                "Loss selection frozen gen=%d phase=%s; keeping previous loss resident population/archives.",
                int(gen),
                str(alternating_active_phase),
            )

        # MAP-Elites archive update (8x8 default, top2 per cell).
        archive_bins = int(cfg_yaml.get("archive_bins", 8) or 8)
        archive_per_cell = int(cfg_yaml.get("archive_per_cell", 2) or 2)
        if builder_selection_active:
            for e in ranked_g:
                cell = tuple(e.get("descriptor", {}).get("cell", [0, 0]))  # type: ignore[assignment]
                try:
                    cell_t = (int(cell[0]), int(cell[1]))
                except Exception:  # noqa: BLE001
                    cell_t = (0, 0)
                _archive_add(
                    archive_g,
                    cell=cell_t,
                    entry={
                        "id": e.get("id"),
                        "signature": e.get("signature"),
                        "ir": e.get("ir"),
                        "descriptor": e.get("descriptor"),
                        "fitness": e.get("fitness"),
                        "selection_sort_key": e.get("selection_sort_key"),
                        "builder_perf": e.get("builder_perf"),
                        "builder_cost": e.get("builder_cost"),
                        "builder_feasible": e.get("builder_feasible"),
                    },
                    score=float(e.get("fitness", float("inf"))),
                    per_cell=archive_per_cell,
                )
        if loss_selection_active:
            for e in ranked_f:
                cell = tuple(e.get("descriptor", {}).get("cell", [0, 0]))  # type: ignore[assignment]
                try:
                    cell_t = (int(cell[0]), int(cell[1]))
                except Exception:  # noqa: BLE001
                    cell_t = (0, 0)
                _archive_add(
                    archive_f,
                    cell=cell_t,
                    entry={
                        "id": e.get("id"),
                        "signature": e.get("signature"),
                        "family": e.get("family"),
                        "ir": e.get("ir"),
                        "descriptor": e.get("descriptor"),
                        "fitness": e.get("fitness"),
                    },
                    score=float(e.get("fitness", float("inf"))),
                    per_cell=archive_per_cell,
                )

        diverse_max = int(cfg_yaml.get("diverse_elites_from_archive_max", max(elite_g * 2, 1)) or max(elite_g * 2, 1))
        if builder_selection_active:
            diverse_elites_g = _archive_flatten(archive_g, max_items=diverse_max)
        if loss_selection_active:
            diverse_elites_f = _archive_flatten(archive_f, max_items=diverse_max)

        # Hall-of-Fame update.
        if builder_selection_active:
            hof_g = _update_hof(hof_g, candidates=elites_g, max_size=int(cfg_yaml.get("hof_size_g", 64) or 64))
        if loss_selection_active:
            hof_f = _update_hof(hof_f, candidates=elites_f, max_size=int(cfg_yaml.get("hof_size_f", 64) or 64))

        if builder_selection_active and elites_g:
            _atomic_write_json(os.path.join(run_dir, "best_elite_builder.json"), dict(elites_g[0]))
        if elites_f:
            _atomic_write_json(os.path.join(run_dir, "best_elite_loss.json"), dict(elites_f[0]))

        best_pair: Dict[str, Any] | None = None
        if isinstance(best_so_far, dict):
            gid_best = str(best_so_far.get("builder_id"))
            fid_best = str(best_so_far.get("loss_id"))
            best_pair = _resolve_best_pair_record(
                best_so_far=best_so_far,
                pair_records=pair_records,
                pair_cache_records=list(caches.pair_cache.values()),
                metric_mode=metric_mode,
                prefer_all_stage3_scenarios_negative=prefer_all_stage3_scenarios_negative,
            )
            if best_pair is None:
                best_pair = {
                    "g_id": gid_best,
                    "f_id": fid_best,
                    "score": best_so_far.get("score"),
                    "final_score": best_so_far.get("score"),
                    "stage_final": best_so_far.get("stage_final"),
                    "generation": best_so_far.get("generation"),
                    "phase": best_so_far.get("phase"),
                    "compare_target": "incumbent",
                    "metric_mode": str(metric_mode),
                    "improve_eps": float(improve_eps),
                    "reference_score": None,
                    "better_than_incumbent": True,
                }
        if best_pair is not None:
            best_pair_score_history = _current_best_pair_score_history()
            best_pair["score_history"] = list(best_pair_score_history)
            best_pair["score_history_summary"] = _score_history_summary(best_pair_score_history)
            _atomic_write_json(os.path.join(run_dir, "best_pair.json"), best_pair)
            gid_best = str(best_pair.get("g_id", "")).strip()
            fid_best = str(best_pair.get("f_id", "")).strip()
            best_pair_eval_meta = _best_pair_eval_metadata(best_pair)

            best_builder = _best_pair_artifact_entry(
                cid=gid_best,
                best_pair=best_pair,
                cid_key="g_id",
                ir_key="g_ir",
                candidate_map=g_map,
                compiled_map=compiled_g,
                ref_ir_fn=_ref_builder_ir if gid_best == G_REF_ID else None,
            )
            if best_builder is not None:
                best_builder.update(best_pair_eval_meta)
                best_builder["score_history"] = list(best_pair_score_history)
                best_builder["score_history_summary"] = _score_history_summary(best_pair_score_history)
                _atomic_write_json(os.path.join(run_dir, "best_builder.json"), dict(best_builder))
            else:
                LOGGER.warning("Failed to resolve best_builder.json for best_pair g_id=%s", gid_best)

            best_loss = _best_pair_artifact_entry(
                cid=fid_best,
                best_pair=best_pair,
                cid_key="f_id",
                ir_key="f_ir",
                candidate_map=f_map,
                compiled_map=compiled_f,
                ref_ir_fn=_ref_loss_ir if fid_best == F_REF_ID else None,
            )
            if best_loss is not None:
                best_loss.update(best_pair_eval_meta)
                best_loss["score_history"] = list(best_pair_score_history)
                best_loss["score_history_summary"] = _score_history_summary(best_pair_score_history)
                _atomic_write_json(os.path.join(run_dir, "best_loss.json"), dict(best_loss))
            else:
                LOGGER.warning("Failed to resolve best_loss.json for best_pair f_id=%s", fid_best)

            _atomic_write_json(
                os.path.join(run_dir, "best_pair_score_history.json"),
                {
                    "g_id": gid_best,
                    "f_id": fid_best,
                    "history": list(best_pair_score_history),
                    "summary": _score_history_summary(best_pair_score_history),
                },
            )

        if isinstance(best_builder_cost, dict):
            best_builder_cost_artifact: Dict[str, Any] = dict(best_builder_cost)
            gid_cost = str(best_builder_cost.get("builder_id", "")).strip()
            perf_ref = best_builder_cost.get("perf_ref")
            best_builder_cost_entry = _best_pair_artifact_entry(
                cid=gid_cost,
                best_pair=(perf_ref if isinstance(perf_ref, Mapping) else None),
                cid_key="g_id",
                ir_key="g_ir",
                candidate_map=g_map,
                compiled_map=compiled_g,
                ref_ir_fn=_ref_builder_ir if gid_cost == G_REF_ID else None,
            )
            if best_builder_cost_entry is not None:
                best_builder_cost_artifact["ir"] = dict(best_builder_cost_entry.get("ir") or {})
                if best_builder_cost_entry.get("signature") is not None:
                    best_builder_cost_artifact["signature"] = best_builder_cost_entry.get("signature")
            _atomic_write_json(os.path.join(run_dir, "best_builder_cost.json"), best_builder_cost_artifact)

        if bool(cfg_yaml.get("drop_pref_cache_after_generation", False)):
            dropped_end = caches.prune_pref_cache(keep_g_ids=[], keep_batch_ids=[])
            if dropped_end > 0:
                LOGGER.info(
                    "Gen %d pref_cache drop after generation: dropped=%d kept=%d %s %s",
                    int(gen),
                    int(dropped_end),
                    int(len(caches.pref_cache)),
                    _cache_brief(caches),
                    _cuda_mem_brief([proxy_device_str]),
                )
                if bool(cfg_yaml.get("cuda_empty_cache_on_pref_prune", False)):
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass

        _save_checkpoint(run_dir, _checkpoint_state(gen + 1))
        _atomic_write_json(summary_json, _summary_state(gen))

    if generations <= gen_start:
        _atomic_write_json(summary_json, _summary_state(gen_start - 1))
    runtime_trace.finish(
        reason="completed",
        exit_code=0,
        extra={
            "last_generation": int(generations - 1),
            "next_generation": int(generations),
            "best_score": (best_so_far.get("score") if isinstance(best_so_far, dict) else None),
            "best_builder_id": (best_so_far.get("builder_id") if isinstance(best_so_far, dict) else None),
            "best_loss_id": (best_so_far.get("loss_id") if isinstance(best_so_far, dict) else None),
        },
    )
    runtime_trace.close()
    LOGGER.info("Co-evolution complete. Artifacts saved under: %s", os.path.abspath(run_dir))
