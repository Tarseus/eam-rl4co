from __future__ import annotations

"""Co-evolution loop for preference builders (g) and preference losses (f)."""

import base64
import collections
import gc
import json
import logging
import math
import os
import pickle
import random
import re
import time
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
    build_or_get_pref_batch,
    build_or_get_rollout_feature_cache,
    eval_budget_signature,
    load_pair_cache_from_pairs_jsonl,
    micro_unroll_score_for_pair,
    proxy_metrics_for_pair_on_batch,
    seed_signature_for_proxy,
)
from ptp_discovery.free_loss_compiler import CompiledFreeLoss, CompileError, compile_free_loss
from ptp_discovery.free_loss_gates import (
    StaticGateResult,
    run_joint_preference_gates,
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


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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


def _sig(obj: Mapping[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha1(blob).hexdigest()


def _sig_free_loss(ir: FreeLossIR) -> str:
    return _sig(asdict(ir))


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


def _normalize_metric_mode(value: Any) -> str:
    mode = str(value or "minimize").strip().lower()
    if mode not in {"minimize", "maximize"}:
        return "minimize"
    return mode


def _normalize_search_mode(value: Any, *, default_mode: str) -> str:
    mode = str(value or default_mode).strip().lower()
    if mode not in {"alternating", "coevo"}:
        return str(default_mode)
    return mode


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

    has_gate_ctx = rec.get("builder_gate_ok") is not None or rec.get("joint_gate_ok") is not None
    has_proxy_ctx = rec.get("proxy_score") is not None or isinstance(rec.get("proxy_metrics"), dict)
    has_micro_ctx = rec.get("micro_score") is not None or isinstance(rec.get("micro_metrics"), dict)
    has_hf_ctx = isinstance(rec.get("fitness"), dict) or str(rec.get("stage")) == "high_fidelity"

    if has_gate_ctx:
        ran.append("stage0_gate")
    if has_proxy_ctx:
        ran.append("stage1_proxy")
    if has_micro_ctx:
        ran.append("stage2_micro_unroll")
    if has_hf_ctx:
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
                "    mask = objective[:, :, None] < objective[:, None, :]\n"
                "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
                f"    max_pairs = int(extra.get('max_pairs', {max_pairs}))\n"
                "    if b_idx.numel() > max_pairs:\n"
                "        perm = torch.randperm(b_idx.numel(), device=b_idx.device)[:max_pairs]\n"
                "        b_idx = b_idx[perm]\n"
                "        winner_idx = winner_idx[perm]\n"
                "        loser_idx = loser_idx[perm]\n"
                "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'sampled_pairs', 'max_pairs': max_pairs})\n"
            )

        pool.append(
            PreferenceBuilderIR(
                name=f"builder_{kind}_{i:03d}",
                intuition=f"rule_based:{kind}",
                implementation_hint=_hint(),
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
            "hyperparams": {"scale": scale},
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
        device=str(device_str),
        seed=int(seed),
        num_validation_episodes=int(cfg.get("num_validation_episodes", 128)),
        validation_batch_size=int(cfg.get("validation_batch_size", 64)),
        generalization_penalty_weight=float(cfg.get("generalization_penalty_weight", 1.0)),
        size_aggregation=str(cfg.get("size_aggregation", "cvar")),
        size_cvar_alpha=float(cfg.get("size_cvar_alpha", 0.2)),
        pool_version=str(cfg.get("pool_version", "v0")),
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


def _build_free_loss_generation_prompt(
    prompt_path: str,
    *,
    global_feedback: Mapping[str, Any] | None,
) -> Tuple[str, str]:
    prompt = _read_prompt_best_effort(prompt_path)
    prompt = _append_global_feedback_block(prompt, global_feedback)
    return prompt, _prompt_sha1(prompt)


def _build_free_loss_parents_prompt(
    prompt_path: str,
    *,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None,
    global_feedback: Mapping[str, Any] | None,
    parent_block_name: str,
) -> Tuple[str, str]:
    prompt = _read_prompt_best_effort(prompt_path)
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
) -> Tuple[str, str]:
    prompt = _read_prompt_best_effort(prompt_path)
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


def validate_builder_candidate(
    ir: PreferenceBuilderIR,
    *,
    operator_whitelist: Sequence[str],
    gate_cfg: Mapping[str, Any],
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
        pb = compiled.build_fn(fc, {"stage": "builder_validate", "seed": 0})
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
    except Exception as exc:  # noqa: BLE001
        return False, _builder_failure_report(stage="gate", reason="builder_gate_exception", error=str(exc))

    if not bool(bg.ok):
        return False, _builder_failure_report(
            stage="gate",
            reason=str(bg.reason),
            trace=bg.trace,
        )

    return True, {}


def _repair_builder_candidate_loop(
    ir: PreferenceBuilderIR,
    *,
    failure_report: Mapping[str, Any],
    operator_whitelist: Sequence[str],
    gate_cfg: Mapping[str, Any],
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
) -> List[Dict[str, Any]]:
    """Propose builder candidates with elitism + mutation/crossover."""

    pop_g = max(int(pop_g), 1)
    out: List[Dict[str, Any]] = []

    parent_pool: List[Mapping[str, Any]] = []
    for src in (elites_g, diverse_elites_g):
        for item in src:
            if isinstance(item, dict) and isinstance(item.get("ir"), dict) and item.get("id"):
                parent_pool.append(item)

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

    if llm_enabled:
        if operator_whitelist is None:
            operator_whitelist = []
        parent_p = int(builder_cfg.get("parent_p", 5) or 5)
        repair_cfg = builder_cfg.get("repair", {}) or {}
        if not isinstance(repair_cfg, dict):
            repair_cfg = {}
        repair_on_fail = bool(repair_cfg.get("enabled", builder_cfg.get("repair_on_failure", True)))
        repair_attempts = int(repair_cfg.get("max_attempts", builder_cfg.get("repair_attempts", 1)) or 1)

        prompts = llm_root.get("prompts", builder_cfg.get("prompts", {})) or {}
        if not isinstance(prompts, dict):
            prompts = {}
        p_gen = str(prompts.get("builder_generation", "") or "")
        p_x = str(prompts.get("builder_crossover", "") or "")
        p_m = str(prompts.get("builder_mutation", "") or "")
        p_e2 = str(prompts.get("builder_e2", "") or "")
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
            ranked_parents.append((fit, str(item.get("id", "")), ir, item))
        ranked_parents.sort(key=lambda x: float(x[0]))
        if not ranked_parents:
            # Bootstrap parents so E2/M1/M2 are usable at gen0 (aligns with free_loss EoH behavior).
            bootstrap = _make_builtin_builder_irs(rng, max(2, int(parent_p)))
            for i, ir0 in enumerate(bootstrap):
                ranked_parents.append((0.0, f"bootstrap_g_{i:03d}", ir0, {"id": f"bootstrap_g_{i:03d}", "fitness": 0.0, "ir": asdict(ir0)}))

        # Operator plan: either explicit counts (preferred) or legacy budget+random choice.
        def _op_plan() -> List[str]:
            init = int(generation) <= 0
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

            # Map high-level EoH ops to concrete LLM ops.
            llm_op = str(op).strip().upper()
            if llm_op == "E1":
                # E1 uses generation when parent pool is small; otherwise crossover or generation.
                if len(ranked_parents) >= 2 and p_x:
                    llm_op = "E1"
                else:
                    llm_op = "E1_GENERATE"
            elif llm_op == "E2":
                llm_op = "E2"
            elif llm_op == "M2":
                llm_op = "M2"
            else:
                llm_op = "M1"

            if llm_op in {"E1", "E2"} and len(ranked_parents) >= 2:
                chosen = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=max(2, min(parent_p, len(ranked_parents))))
                parents_ir = [c[2] for c in chosen]
                parents_fit = [{"fitness": float(c[0])} for c in chosen]
                parents_ids = [str(c[1]) for c in chosen]
            elif llm_op in {"M1", "M2"} and len(ranked_parents) >= 1:
                chosen1 = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=1)[0]
                parents_ir = [chosen1[2]]
                parents_fit = [{"fitness": float(chosen1[0])}]
                parents_ids = [str(chosen1[1])]
            else:
                llm_op = "E1_GENERATE"

            llm_seed = int(rng.randint(0, 2**31 - 1))
            call_feedback = dict(global_feedback or {})
            call_feedback["llm_call"] = {"side": "builder", "op_type": str(llm_op), "seed": llm_seed}

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
) -> List[Dict[str, Any]]:
    """Propose loss candidates with elitism + mutation/crossover."""

    pop_f = max(int(pop_f), 1)
    out: List[Dict[str, Any]] = []

    parent_pool: List[Mapping[str, Any]] = []
    for src in (elites_f, diverse_elites_f):
        for item in src:
            if isinstance(item, dict) and isinstance(item.get("ir"), dict) and item.get("id"):
                parent_pool.append(item)

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
        p_m2 = str(prompts.get("loss_m2", "") or "")
        p_m3 = str(prompts.get("loss_m3", "") or "")
        p_rep = str(prompts.get("loss_repair", "") or "")

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
            ranked_parents.append((fit, str(item.get("id", "")), ir0, item))
        ranked_parents.sort(key=lambda x: float(x[0]))
        if not ranked_parents:
            bootstrap = _make_builtin_loss_irs(rng, max(2, int(parent_p)))
            for i, ir0 in enumerate(bootstrap):
                ranked_parents.append((0.0, f"bootstrap_f_{i:03d}", ir0, {"id": f"bootstrap_f_{i:03d}", "fitness": 0.0, "ir": asdict(ir0)}))

        def _op_plan() -> List[str]:
            init = int(generation) <= 0
            keys = ("num_E1", "num_E2", "num_M1", "num_M2", "init_num_E1", "init_num_E2", "init_num_M1", "init_num_M2")
            if any(loss_cfg.get(k) is not None for k in keys):
                nE1 = int(loss_cfg.get("init_num_E1" if init else "num_E1", loss_cfg.get("num_E1", 0)) or 0)
                nE2 = int(loss_cfg.get("init_num_E2" if init else "num_E2", loss_cfg.get("num_E2", 0)) or 0)
                nM1 = int(loss_cfg.get("init_num_M1" if init else "num_M1", loss_cfg.get("num_M1", 0)) or 0)
                nM2 = int(loss_cfg.get("init_num_M2" if init else "num_M2", loss_cfg.get("num_M2", 0)) or 0)
                plan = (["E1"] * max(0, nE1)) + (["E2"] * max(0, nE2)) + (["M1"] * max(0, nM1)) + (["M2"] * max(0, nM2))
                rng.shuffle(plan)
                return plan

            llm_budget = int(loss_cfg.get("init_llm_f", 0) or 0) if init else int(loss_cfg.get("llm_per_gen_f", 0) or 0)
            return [_llm_op_choice(rng, gen=int(generation), parent_pool_size=len(ranked_parents)) for _ in range(max(0, llm_budget))]

        for op in _op_plan():
            if len(out) >= max(0, int(pop_f) - int(seed_reserve)):
                break
            parents_ir: List[FreeLossIR] = []
            parents_fit: List[Mapping[str, Any]] = []
            parents_ids: List[str] = []

            llm_op = str(op).strip().upper()
            if llm_op == "E1":
                if len(ranked_parents) >= 2 and p_x:
                    llm_op = "E1"
                else:
                    llm_op = "E1_GENERATE"
            elif llm_op == "E2":
                llm_op = "E2"
            elif llm_op == "M2":
                llm_op = "M2"
            else:
                llm_op = "M1"

            if llm_op in {"E1", "E2"} and len(ranked_parents) >= 2:
                chosen = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=max(2, min(parent_p, len(ranked_parents))))
                parents_ir = [c[2] for c in chosen]
                parents_fit = [{"fitness": float(c[0])} for c in chosen]
                parents_ids = [str(c[1]) for c in chosen]
            elif llm_op in {"M1", "M2"} and len(ranked_parents) >= 1:
                chosen1 = _rank_weighted_sample_without_replacement(rng, ranked_parents, k=1)[0]
                parents_ir = [chosen1[2]]
                parents_fit = [{"fitness": float(chosen1[0])}]
                parents_ids = [str(chosen1[1])]
            else:
                llm_op = "E1_GENERATE"

            llm_seed = int(rng.randint(0, 2**31 - 1))
            call_feedback = dict(global_feedback or {})
            call_feedback["llm_call"] = {"side": "loss", "op_type": str(llm_op), "seed": llm_seed}

            history: List[Dict[str, Any]] = []
            base_origin = "E1"
            op_type = str(op)
            parent_ids = list(parents_ids)
            prompt_sha1 = None
            prompt_path = None

            try:
                if llm_op == "E1_GENERATE":
                    _, sha = _build_free_loss_generation_prompt(p_gen, global_feedback=call_feedback)
                    prompt_sha1 = sha
                    prompt_path = str(p_gen)
                    ir = loss_llm_ops.generate_free_loss_candidate(
                        p_gen,
                        operator_whitelist=operator_whitelist,
                        global_feedback=call_feedback,
                    )
                    base_origin = "E1"
                    op_type = "E1_GENERATE"
                    parent_ids = []
                elif llm_op == "E1":
                    _, sha = _build_free_loss_parents_prompt(
                        p_x,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        parent_block_name="PARENTS_JSON",
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_x)
                    ir = loss_llm_ops.crossover_free_loss(p_x, parents=parents_ir, parents_fitness=parents_fit, global_feedback=call_feedback)
                    base_origin = "E1"
                    op_type = "E1"
                elif llm_op == "E2":
                    _, sha = _build_free_loss_parents_prompt(
                        p_e2,
                        parents=parents_ir,
                        parents_fitness=parents_fit,
                        global_feedback=call_feedback,
                        parent_block_name="PARENTS_JSON",
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_e2)
                    ir = loss_llm_ops.e2_free_loss(p_e2, parents=parents_ir, parents_fitness=parents_fit, global_feedback=call_feedback)
                    base_origin = "E2"
                    op_type = "E2"
                elif llm_op == "M2":
                    _, sha = _build_free_loss_parent_prompt(
                        p_m2,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
                        parent_block_name="PARENT_JSON",
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_m2)
                    ir = loss_llm_ops.m2_tune_hparams(
                        p_m2,
                        parent=parents_ir[0],
                        parent_fitness=parents_fit[0],
                        global_feedback=call_feedback,
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
                    )
                    prompt_sha1 = sha
                    prompt_path = str(p_m)
                    ir = loss_llm_ops.mutate_free_loss(p_m, parent=parents_ir[0], parent_fitness=parents_fit[0], global_feedback=call_feedback)
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
            try:
                static_res = run_static_gates(ir, operator_whitelist=operator_whitelist)
                if not bool(static_res.ok):
                    ok = False
                    fail_reason = {"stage": "static_gate", "reason": str(static_res.reason), "trace": static_res.trace}
                else:
                    _ = compile_free_loss(ir, operator_whitelist=operator_whitelist)
                    ok = True
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
                        if bool(repair_cfg.get("simplify_first", True)) and p_m3 and fail_reason.get("stage") in {"static_gate", "compile"}:
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
                            repaired = loss_llm_ops.repair_free_loss(p_rep, failed_ir=ir, failure_reason=fail_reason)
                        except Exception:  # noqa: BLE001
                            repaired = None
                            break
                    try:
                        static_res = run_static_gates(repaired, operator_whitelist=operator_whitelist)
                        if not bool(static_res.ok):
                            fail_reason = {"stage": "static_gate", "reason": str(static_res.reason), "trace": static_res.trace}
                            continue
                        _ = compile_free_loss(repaired, operator_whitelist=operator_whitelist)
                        ir = repaired
                        ok = True
                        op_type = "REPAIR"
                        break
                    except Exception as exc:  # noqa: BLE001
                        fail_reason = {"stage": "compile", "error": str(exc)}
                        continue

            if ok:
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
                    }
                )

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

    coverage = None
    if isinstance(bg.get("metric"), dict) and bg["metric"].get("metric_name") == "coverage":
        coverage = bg["metric"].get("observed_value")
    if coverage is None:
        coverage = bg.get("coverage", bg.get("observed", {}).get("coverage"))
    try:
        coverage_f = float(coverage)
    except (TypeError, ValueError):
        coverage_f = 0.0

    pair_count = bg.get("pair_count")
    try:
        pair_count_i = int(pair_count)
    except (TypeError, ValueError):
        pair_count_i = int(bg.get("observed", {}).get("pair_count", 0) or 0)

    semantic = bg.get("semantic_pass_rate")
    try:
        sem_f = float(semantic)
    except (TypeError, ValueError):
        sem_f = float(bg.get("observed", {}).get("semantic_pass_rate", 0.0) or 0.0)

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
    lst.sort(key=lambda x: float(x.get("archive_score", float("inf"))))
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
    items.sort(key=lambda x: float(x.get("archive_score", float("inf"))))
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
    out.sort(key=lambda x: float(x.get("fitness", float("inf"))))
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
        hyperparams={"alpha": 1.0},
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

        if pref_cache_enabled:
            pref = build_or_get_pref_batch(
                caches=caches,
                g_id=str(gid),
                batch_id=int(pref_batch_id_offset + local_batch_id),
                builder=g_comp,
                feature_cache=fc,
                extra={"stage": "proxy", "seed_signature": str(seed_sig)},
            )
        else:
            # For VRAM stability: avoid storing PrefBatch tensors in a long-lived cache.
            pref = g_comp.build_fn(fc, {"stage": "proxy", "seed_signature": str(seed_sig)})
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
            builder_ok = bool(bg.ok)
            builder_gate_first = {
                "builder_gate_ok": bool(bg.ok),
                "builder_gate_reason": str(bg.reason),
                "builder_gate_trace": bg.trace,
                "pair_count": bg.pair_count,
                "coverage": bg.coverage,
                "semantic_pass_rate": bg.semantic_pass_rate,
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

    if cheap_gate_on and (not builder_ok or not joint_ok):
        base["pair_ok"] = False
        base["pair_reason"] = "cheap_proxy_gate_failed"
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
    pref_batch = compiled_g.build_fn(feature_cache, {"stage": "cheap_gate"})

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
    joint_gate = run_joint_preference_gates(
        compiled_f,
        pref_batch=pref_batch,
        feature_cache=feature_cache,
        min_pass_rate=float(cfg.get("joint_min_pass_rate", 0.8) or 0.8),
        swap_tolerance=float(cfg.get("joint_swap_tolerance", 1e-3) or 1e-3),
        grad_eps=float(cfg.get("joint_grad_eps", 1e-8) or 1e-8),
        min_effective_grad_ratio=float(cfg.get("joint_min_effective_grad_ratio", 0.1) or 0.1),
        variant="visible",
    )
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
            record["pair_ok"] = False
            record["pair_reason"] = "pref_semantic_failed"
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

    seed = int(cfg.get("seed", 0))
    hf_cfg = _build_hf_cfg(cfg, seed=seed, device_str=device_str)
    free_cfg = _build_free_cfg(cfg, hf_cfg=hf_cfg)
    adapter = _CompiledBuilderAdapter(compiled_g)
    try:
        # Route high-fidelity training logs to a per-pair file (like free_loss_discovery).
        file_handler: logging.Handler | None = None
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

            fl_logger = logging.getLogger("fitness.free_loss_fidelity")
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
                fl_logger.addHandler(file_handler)
                root_logger.addHandler(file_handler)
                record["hf_log_file"] = os.path.basename(log_path)
                fl_logger.info(
                    "HF start gen=%d pair_index=%d device=%s g_id=%s f_id=%s",
                    int(generation),
                    int(pair_index),
                    str(device_str),
                    str(record.get("g_id")),
                    str(record.get("f_id")),
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[pref_loss_coevo][worker] failed to open HF log file: {log_path}: {exc}",
                    flush=True,
                )
                file_handler = None

        fitness = evaluate_free_loss_candidate(
            compiled_f,
            free_cfg,
            pref_builder=adapter,
            baseline_early_valid=baseline_early_valid_f,
            early_eval_steps=early_eval_steps_i,
            baseline_epoch_objectives=baseline_epoch_objectives,
        )
    except Exception as exc:  # noqa: BLE001
        record["pair_ok"] = False
        record["pair_reason"] = "high_fidelity_failed"
        record["high_fidelity_error"] = str(exc)
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

    record["pair_ok"] = True
    record["pair_reason"] = "ok"
    record["fitness"] = dict(fitness)
    # If baseline epoch objectives are provided, `epoch_better_than_baseline` is
    # True only when all HF epochs are better than the baseline (smaller objective).
    if isinstance(fitness, dict):
        tail_better = fitness.get("epoch_tail_better_than_baseline")
        if tail_better is None:
            tail_better = fitness.get("epoch_better_than_baseline")
        if tail_better is not None:
            record["better_than_baseline"] = bool(tail_better)
    try:
        record["score"] = float(
            fitness.get(
                "hf_like_score",
                fitness.get("fitness_score", fitness.get("validation_objective", float("inf"))),
            )
        )
    except (TypeError, ValueError):
        record["score"] = float("inf")
    record["elapsed_s"] = float(time.time() - t0)
    return record


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
            rec = _evaluate_pair_worker(fixed)
        except Exception as exc:  # noqa: BLE001
            fixed = dict(payload) if isinstance(payload, dict) else {}
            fixed["device"] = str(device_str)
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
    alternating_rounds = max(0, _safe_int(alternating_schedule_raw.get("rounds", 0), 0))
    alternating_schedule_enabled = bool(alternating_loss_generations > 0 and alternating_builder_generations > 0)

    preset = str(cfg_yaml.get("preset", "advanced") or "advanced").strip().lower()
    search_mode = str(cfg_yaml.get("search_mode", "coevo") or "coevo").strip().lower()
    metric_mode = _normalize_metric_mode(cfg_yaml.get("metric_mode", "minimize"))
    improve_eps = float(cfg_yaml.get("improve_eps", 0.0) or 0.0)
    eval_stages = _normalize_eval_stages(cfg_yaml)

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
    if preset == "simple":
        ignored_keys = list(runtime_meta.get("ignored_advanced_keys", []))
        if ignored_keys:
            LOGGER.warning("preset=simple: advanced keys are ignored: %s", ignored_keys)
    if search_mode == "coevo":
        LOGGER.warning("search_mode=coevo is supported but not encouraged; prefer search_mode=alternating.")
    if search_mode == "alternating" and alternating_schedule_enabled:
        LOGGER.info(
            "Alternating schedule enabled: loss_generations=%d builder_generations=%d rounds=%s",
            int(alternating_loss_generations),
            int(alternating_builder_generations),
            (int(alternating_rounds) if alternating_rounds > 0 else "unbounded"),
        )
        if alternating_rounds > 0:
            planned = int(alternating_rounds) * int(alternating_loss_generations + alternating_builder_generations)
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
    llm_prompts_defaults = {
        "builder_generation": "PTP/prompts/pref_builder_generation.txt",
        "builder_crossover": "PTP/prompts/pref_builder_crossover.txt",
        "builder_mutation": "PTP/prompts/pref_builder_mutation.txt",
        "builder_e2": "PTP/prompts/pref_builder_e2.txt",
        "builder_m2": "PTP/prompts/pref_builder_m2.txt",
        "builder_m3": "PTP/prompts/pref_builder_m3.txt",
        "builder_repair": "PTP/prompts/pref_builder_repair.txt",
        "loss_generation": "PTP/prompts/free_loss_generation.txt",
        "loss_crossover": "PTP/prompts/free_loss_crossover.txt",
        "loss_mutation": "PTP/prompts/free_loss_mutation.txt",
        "loss_e2": "PTP/prompts/free_loss_e2.txt",
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
    }
    loss_cfg: Dict[str, Any] = {
        "enabled": bool(loss_llm_enabled),
        "parent_p": int(loss_llm_raw.get("parent_p", default_parent_p) or default_parent_p),
        "seed_reserve": int(loss_llm_raw.get("seed_reserve", cfg_yaml.get("loss_seed_reserve", 2)) or 2),
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
    seen_g = set(resume_state.get("seen_g", [])) if resume_state else set()
    seen_f = set(resume_state.get("seen_f", [])) if resume_state else set()
    elites_g: List[Dict[str, Any]] = list(resume_state.get("elites_g", [])) if resume_state else []
    elites_f: List[Dict[str, Any]] = list(resume_state.get("elites_f", [])) if resume_state else []
    diverse_elites_g: List[Dict[str, Any]] = list(resume_state.get("diverse_elites_g", [])) if resume_state else []
    diverse_elites_f: List[Dict[str, Any]] = list(resume_state.get("diverse_elites_f", [])) if resume_state else []
    hof_g: List[Dict[str, Any]] = list(resume_state.get("hof_g", [])) if resume_state else []
    hof_f: List[Dict[str, Any]] = list(resume_state.get("hof_f", [])) if resume_state else []
    archive_g: Dict[str, List[Dict[str, Any]]] = dict(resume_state.get("archive_g", {})) if resume_state else {}
    archive_f: Dict[str, List[Dict[str, Any]]] = dict(resume_state.get("archive_f", {})) if resume_state else {}
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
            "loss_generations": int(alternating_loss_generations),
            "builder_generations": int(alternating_builder_generations),
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
        loaded = load_pair_cache_from_pairs_jsonl(caches=caches, pairs_jsonl_path=pairs_jsonl, eval_sig=eval_sig)
        LOGGER.info("Loaded %d cached pair records from pairs.jsonl (eval_sig=%s)", loaded, eval_sig)

    def _summary_state(last_generation: int) -> Dict[str, Any]:
        return {
            "config_path": os.path.abspath(config_path),
            "run_dir": os.path.abspath(run_dir),
            "preset": str(preset),
            "search_mode": str(search_mode),
            "alternating_schedule": {
                "enabled": bool(alternating_schedule_enabled),
                "loss_generations": int(alternating_loss_generations),
                "builder_generations": int(alternating_builder_generations),
                "rounds": int(alternating_rounds),
            },
            "metric_mode": str(metric_mode),
            "improve_eps": float(improve_eps),
            "eval_stages": dict(eval_stages),
            "last_generation": int(last_generation),
            "best_so_far": dict(best_so_far) if isinstance(best_so_far, dict) else None,
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
        }

    def _checkpoint_state(next_generation: int) -> Dict[str, Any]:
        return {
            "config_path": os.path.abspath(config_path),
            "seed": int(seed),
            "next_generation": int(next_generation),
            "preset": str(preset),
            "search_mode": str(search_mode),
            "alternating_schedule": {
                "enabled": bool(alternating_schedule_enabled),
                "loss_generations": int(alternating_loss_generations),
                "builder_generations": int(alternating_builder_generations),
                "rounds": int(alternating_rounds),
            },
            "metric_mode": str(metric_mode),
            "improve_eps": float(improve_eps),
            "eval_stages": dict(eval_stages),
            "best_so_far": dict(best_so_far) if isinstance(best_so_far, dict) else None,
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
            "rng_state_b64": _b64_pickle(rng.getstate()),
            "seen_g": sorted(seen_g),
            "seen_f": sorted(seen_f),
            "elites_g": list(elites_g),
            "elites_f": list(elites_f),
            "diverse_elites_g": list(diverse_elites_g),
            "diverse_elites_f": list(diverse_elites_f),
            "hof_g": list(hof_g),
            "hof_f": list(hof_f),
            "archive_g": dict(archive_g),
            "archive_f": dict(archive_f),
        }

    _save_checkpoint(run_dir, _checkpoint_state(gen_start))
    _atomic_write_json(summary_json, _summary_state(gen_start - 1))

    llm_feedback_state: Dict[str, Any] = {}

    for gen in range(gen_start, generations):
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
            }
        )

        proposed_g = _propose_builders_for_generation(
            generation=int(gen),
            pop_g=int(pop_g),
            elites_g=elites_g,
            diverse_elites_g=diverse_elites_g,
            rng=rng,
            llm_cfg=llm_cfg if llm_enabled else None,
            operator_whitelist=operator_whitelist,
            global_feedback=global_feedback if llm_enabled else None,
        )
        proposed_f = _propose_losses_for_generation(
            generation=int(gen),
            pop_f=int(pop_f),
            elites_f=elites_f,
            diverse_elites_f=diverse_elites_f,
            rng=rng,
            llm_cfg=llm_cfg if llm_enabled else None,
            operator_whitelist=operator_whitelist,
            global_feedback=global_feedback if llm_enabled else None,
        )

        # Ensure generation-0 default pair/loss match PO4COPs-style baseline
        # before search-driven variants are considered.
        if int(gen) == 0 and bool(cfg_yaml.get("seed_with_po4cops_default", True)):
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
        def _fill_unique_builders(proposals: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
            unique: List[Dict[str, Any]] = []
            attempts = 0
            for p in proposals:
                if not isinstance(p, dict) or not isinstance(p.get("ir"), PreferenceBuilderIR):
                    continue
                ir = p["ir"]
                sig = _sig_pref_builder(ir)
                if sig in seen_g:
                    continue
                unique.append(dict(p))
                seen_g.add(sig)
            while len(unique) < pop_g and attempts < pop_g * 20:
                attempts += 1
                ir = _make_builtin_builder_irs(rng, 1)[0]
                sig = _sig_pref_builder(ir)
                if sig in seen_g:
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
                seen_g.add(sig)
            return unique[:pop_g]

        def _fill_unique_losses(proposals: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
            unique2: List[Dict[str, Any]] = []
            attempts2 = 0
            for p in proposals:
                if not isinstance(p, dict) or not isinstance(p.get("ir"), FreeLossIR):
                    continue
                ir = p["ir"]
                sig = _sig_free_loss(ir)
                if sig in seen_f:
                    continue
                unique2.append(dict(p))
                seen_f.add(sig)
            while len(unique2) < pop_f and attempts2 < pop_f * 20:
                attempts2 += 1
                ir = _make_builtin_loss_irs(rng, 1)[0]
                sig = _sig_free_loss(ir)
                if sig in seen_f:
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
                seen_f.add(sig)
            return unique2[:pop_f]

        proposed_g = _fill_unique_builders(proposed_g)
        proposed_f = _fill_unique_losses(proposed_f)
        if len(proposed_g) < int(pop_g) or len(proposed_f) < int(pop_f):
            LOGGER.warning(
                "Population fill shortfall at gen=%d: proposed_g=%d/%d proposed_f=%d/%d (seen_g=%d seen_f=%d)",
                int(gen),
                int(len(proposed_g)),
                int(pop_g),
                int(len(proposed_f)),
                int(pop_f),
                int(len(seen_g)),
                int(len(seen_f)),
            )

        g_entries: List[Dict[str, Any]] = []
        for idx, proposal in enumerate(proposed_g):
            ir: PreferenceBuilderIR = proposal["ir"]
            sig = _sig_pref_builder(ir)
            entry: Dict[str, Any] = {
                "generation": int(gen),
                "index": int(idx),
                "id": f"g{gen:03d}_{idx:03d}_{sig[:8]}",
                "signature": sig,
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
                pb = compiled.build_fn(fc, {"stage": "builder_static"})
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
            entry: Dict[str, Any] = {
                "generation": int(gen),
                "index": int(idx),
                "id": f"f{gen:03d}_{idx:03d}_{sig[:8]}",
                "signature": sig,
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

        elite_g_ids = [str(e["id"]) for e in elites_g if isinstance(e, dict) and "id" in e]
        elite_f_ids = [str(e["id"]) for e in elites_f if isinstance(e, dict) and "id" in e]
        g_id_pool = list(dict.fromkeys(elite_g_ids + [str(e["id"]) for e in g_pool]))
        f_id_pool = list(dict.fromkeys(elite_f_ids + [str(e["id"]) for e in f_pool]))

        g_map = {str(e["id"]): e for e in g_pool + [e for e in elites_g if isinstance(e, dict) and "id" in e]}
        f_map = {str(e["id"]): e for e in f_pool + [e for e in elites_f if isinstance(e, dict) and "id" in e]}

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
        gate_repair_enabled = bool(llm_enabled and stage1_proxy_enabled and bool(gate_repair_cfg.get("enabled", False)))
        gate_repair_remaining = max(0, int(gate_repair_cfg.get("max_repairs_per_gen", 0) or 0))
        gate_repair_attempts = max(0, int(gate_repair_cfg.get("max_attempts_per_pair", 1) or 1))
        gate_repair_builder_on_fail = bool(gate_repair_cfg.get("repair_builder_on_builder_gate_fail", True))
        gate_repair_loss_on_fail = bool(gate_repair_cfg.get("repair_loss_on_joint_gate_fail", True))
        gate_repair_attempted_pairs: set[tuple[str, str]] = set()
        gate_repair_attempted_builders: set[str] = set()
        gate_repair_attempted_losses: set[str] = set()

        llm_prompts_cfg = llm_cfg.get("prompts", {}) if isinstance(llm_cfg.get("prompts"), dict) else {}
        p_builder_rep = str(llm_prompts_cfg.get("builder_repair", "") or "")
        p_builder_m3 = str(llm_prompts_cfg.get("builder_m3", "") or "")
        p_loss_rep = str(llm_prompts_cfg.get("loss_repair", "") or "")
        p_loss_m3 = str(llm_prompts_cfg.get("loss_m3", "") or "")
        builder_repair_live_cfg = builder_cfg.get("repair", {}) if isinstance(builder_cfg.get("repair"), dict) else {}
        loss_repair_live_cfg = loss_cfg.get("repair", {}) if isinstance(loss_cfg.get("repair"), dict) else {}
        builder_gate_live_cfg = llm_cfg.get("builder_gate", {}) if isinstance(llm_cfg.get("builder_gate"), dict) else {}

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
        elite_g_ids = [str(e["id"]) for e in elites_g if isinstance(e, dict) and "id" in e]
        elite_f_ids = [str(e["id"]) for e in elites_f if isinstance(e, dict) and "id" in e]
        g_id_pool = list(dict.fromkeys(elite_g_ids + [str(e["id"]) for e in g_pool] + list(hof_g_ids)))
        f_id_pool = list(dict.fromkeys(elite_f_ids + [str(e["id"]) for e in f_pool] + list(hof_f_ids)))

        # Rebuild maps to include HoF entries for high-fidelity cross-play.
        g_map = {str(e["id"]): e for e in g_pool + [e for e in elites_g if isinstance(e, dict) and "id" in e]}
        f_map = {str(e["id"]): e for e in f_pool + [e for e in elites_f if isinstance(e, dict) and "id" in e]}
        for e in hof_g:
            if isinstance(e, dict) and e.get("id") and isinstance(e.get("ir"), dict):
                g_map.setdefault(str(e["id"]), dict(e))
        for e in hof_f:
            if isinstance(e, dict) and e.get("id") and isinstance(e.get("ir"), dict):
                f_map.setdefault(str(e["id"]), dict(e))

        pairs: List[Tuple[str, str]] = []
        reasons_by_pair: Dict[Tuple[str, str], List[str]] = {}
        if str(search_mode) == "alternating":
            loss_budget_now = int(pairing_budget_loss)
            builder_budget_now = int(pairing_budget_builder)
            if alternating_schedule_enabled:
                cycle_len = int(alternating_loss_generations + alternating_builder_generations)
                cycle_pos = int(gen % max(cycle_len, 1))
                round_idx = int(gen // max(cycle_len, 1))
                if alternating_rounds > 0 and round_idx >= int(alternating_rounds):
                    loss_budget_now = 0
                    builder_budget_now = 0
                elif cycle_pos < int(alternating_loss_generations):
                    loss_budget_now = int(pairing_budget)
                    builder_budget_now = 0
                else:
                    loss_budget_now = 0
                    builder_budget_now = int(pairing_budget)
                LOGGER.info(
                    "Alternating block status: round=%d cycle_pos=%d/%d budgets(loss=%d,builder=%d)",
                    int(round_idx),
                    int(cycle_pos),
                    int(cycle_len),
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

            # Fixed builder for loss-search: prefer current best builder; fallback to g_ref.
            if elites_g:
                cand_g = str(elites_g[0].get("id") or "")
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

            # Fixed loss for pair-search: prefer current best loss; fallback to f_ref.
            if elites_f:
                cand_f = str(elites_f[0].get("id") or "")
                if cand_f and cand_f in compiled_f:
                    fixed_loss_id = cand_f
            if (not fixed_loss_id) and isinstance(best_so_far, dict):
                cand_f = str(best_so_far.get("loss_id") or "")
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
                elite_g_ids=list(elite_g_ids),
                elite_f_ids=list(elite_f_ids),
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
                    and str(rec.get("pair_reason", "")) == "cheap_proxy_gate_failed"
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
                                prompt_sha = _build_free_loss_failure_prompt(
                                    p_loss_rep,
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
                                        "prompt_path": str(p_loss_rep),
                                        "prompt_sha1": str(prompt_sha),
                                    }
                                )
                                candidate = loss_llm_ops.repair_free_loss(
                                    p_loss_rep,
                                    failed_ir=f_ir,
                                    failure_reason=fail_payload,
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
            for p_idx, (gid, fid) in enumerate(pairs):
                base = dict(pair_records_map.get((str(gid), str(fid)), {}))
                base.update(
                    {
                        "generation": int(gen),
                        "pair_index": int(p_idx),
                        "g_id": str(gid),
                        "f_id": str(fid),
                        "eval_budget_signature": str(eval_sig),
                        "seed_signature": str(base_seed_sig),
                        "seed_used": int(base_seed),
                        "device": str(proxy_device_str),
                        "stage": str(base.get("stage", "scheduled")),
                        "reasons": list(reasons_by_pair.get((gid, fid), ["scheduled"])),
                        "pair_ok": True,
                        "pair_reason": "proxy_disabled",
                        "score": base.get("score"),
                        "proxy_score": base.get("proxy_score"),
                        "proxy_metrics": base.get("proxy_metrics"),
                        "descriptor": base.get("descriptor"),
                        "phase": pair_phase_by_pair.get((str(gid), str(fid)), "coevo"),
                    }
                )
                pair_records_map[(str(gid), str(fid))] = base

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
                micro_alpha = float(cfg_yaml.get("micro_unroll_alpha", cfg_yaml.get("alpha", 0.05)) or 0.05)
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

        # High-fidelity stage: evaluate only top-m by cheap proxy `score`.
        if stage3_hf_enabled:
            top_m = int(cfg_yaml.get("high_fidelity_top_m", max(1, min(len(pair_records), pairing_budget // 4))) or 1)
            candidates = [
                r
                for r in pair_records
                if (bool(r.get("pair_ok")) if stage1_proxy_enabled else True)
                and str(r.get("stage")) != "anchor"
            ]
            if micro_enabled:
                candidates = [r for r in candidates if isinstance(r.get("micro_metrics"), dict)]
            if micro_enabled or stage1_proxy_enabled:
                candidates.sort(key=lambda r: float(r.get("score", float("inf"))))
            else:
                candidates.sort(
                    key=lambda r: (
                        _safe_int(r.get("pair_index", 10**9), 10**9),
                        str(r.get("g_id", "")),
                        str(r.get("f_id", "")),
                    )
                )

            selected = candidates[: max(0, top_m)]
            LOGGER.info(
                "HF selection gen=%d: eligible=%d selected=%d (top_m=%d)",
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
                # Distribute high-fidelity tasks round-robin across the configured devices.
                # Do not use the global pair_index here because it includes anchors and
                # other non-HF stages, which can skew GPU assignment and leave devices idle.
                device_str = device_list[int(len(hf_tasks)) % len(device_list)]
                hf_tasks.append(
                    {
                        "generation": int(gen),
                        "pair_index": _safe_int(r.get("pair_index", -1), -1),
                        "g_entry": g_map[gid],
                        "f_entry": f_map[fid],
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
                if mp_enabled and mp_processes > 0 and len(hf_tasks) > 1:
                    import multiprocessing as mp

                    ctx = mp.get_context(mp_start_method)
                    # Prefer one process per device to avoid oversubscribing GPUs.
                    procs = min(int(mp_processes), max(1, len(hf_tasks)), max(1, len(device_list)))
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
                try:
                    ok = sum(1 for r in hf_results if bool(r.get("pair_ok")))
                    failed = int(len(hf_results) - ok)
                    better = sum(1 for r in hf_results if r.get("better_than_baseline") is True)
                    worse = sum(1 for r in hf_results if r.get("better_than_baseline") is False)
                    unknown = int(len(hf_results) - better - worse)
                    LOGGER.info(
                        "HF vs baseline gen=%d: better=%d worse=%d unknown=%d (ok=%d failed=%d)",
                        int(gen),
                        int(better),
                        int(worse),
                        int(unknown),
                        int(ok),
                        int(failed),
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

        # Stage annotations + incumbent-gating (better_than_prev_best).
        for rec in pair_records:
            rec["stages_enabled"] = dict(eval_stages)
            ran, skipped = _annotate_stage_fields(rec, eval_stages=eval_stages)
            rec["stages_ran"] = list(ran)
            rec["stages_skipped"] = dict(skipped)
            stage_final, final_score = _resolve_final_score(rec, eval_stages=eval_stages)
            rec["stage_final"] = str(stage_final)
            rec["final_score"] = final_score
            rec["metric_mode"] = str(metric_mode)
            rec["compare_target"] = "prev_best"
            rec["improve_eps"] = float(improve_eps)

        current_ref = None
        if isinstance(best_so_far, dict):
            try:
                current_ref = float(best_so_far.get("score"))
            except (TypeError, ValueError):
                current_ref = None
        LOGGER.info(
            "COMPARE target=prev_best reference_score=%s threshold=%s metric_mode=%s improve_eps=%s",
            current_ref,
            _score_threshold(reference_score=current_ref, metric_mode=metric_mode, improve_eps=improve_eps),
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
            if str(rec.get("g_id")) == G_REF_ID or str(rec.get("f_id")) == F_REF_ID or str(rec.get("stage")) == "anchor":
                rec["better_than_prev_best"] = False
                rec["delta_vs_prev_best"] = None
                continue

            final_score = rec.get("final_score")
            if final_score is None:
                rec["better_than_prev_best"] = False
                rec["delta_vs_prev_best"] = None
                continue
            try:
                cand_score_f = float(final_score)
            except (TypeError, ValueError):
                rec["better_than_prev_best"] = False
                rec["delta_vs_prev_best"] = None
                rec["final_score"] = None
                rec["stage_final"] = "none"
                continue
            if not math.isfinite(cand_score_f):
                rec["better_than_prev_best"] = False
                rec["delta_vs_prev_best"] = None
                continue

            delta = _score_delta(cand_score=cand_score_f, ref_score=reference_score, metric_mode=metric_mode)
            better = _is_better_than_reference(
                cand_score=cand_score_f,
                reference_score=reference_score,
                metric_mode=metric_mode,
                improve_eps=improve_eps,
            )
            rec["better_than_prev_best"] = bool(better)
            rec["delta_vs_prev_best"] = delta
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
                }
                LOGGER.info(
                    "NEW BEST: score=%s ref=%s delta=%s pair=(%s,%s) stage=%s gen=%d phase=%s threshold=%s compare_target=prev_best improve_eps=%s",
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
            for k in ("builder_gate_trace", "joint_gate_trace", "pref_semantic_trace"):
                t = rec.get(k)
                if not isinstance(t, dict):
                    continue
                kind = t.get("failure_kind") or t.get("failed_gate")
                if kind is None:
                    continue
                gate_kind_ctr[str(kind)] += 1

        # Best pair preview (for coevolution guidance).
        best_pair_preview: Dict[str, Any] | None = None
        best_score_preview: float | None = None
        for rec in pair_records:
            if str(rec.get("g_id")) == G_REF_ID or str(rec.get("f_id")) == F_REF_ID:
                continue
            if str(rec.get("stage")) == "anchor":
                continue
            if not bool(rec.get("pair_ok")):
                continue
            try:
                score_f = float(rec.get("final_score", rec.get("score", float("inf"))))
            except (TypeError, ValueError):
                continue
            if _is_better_than_reference(
                cand_score=score_f,
                reference_score=best_score_preview,
                metric_mode=metric_mode,
                improve_eps=0.0,
            ):
                best_score_preview = score_f
                best_pair_preview = {
                    "g_id": rec.get("g_id"),
                    "f_id": rec.get("f_id"),
                    "score": score_f,
                    "stage": rec.get("stage"),
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
                    "better_than_prev_best": rec.get("better_than_prev_best"),
                    "delta_vs_prev_best": rec.get("delta_vs_prev_best"),
                    "compare_target": rec.get("compare_target"),
                    "static_ok": rec.get("f_static_ok"),
                    "static_reason": rec.get("f_static_reason"),
                    "builder_gate_ok": rec.get("builder_gate_ok"),
                    "builder_gate_reason": rec.get("builder_gate_reason"),
                    "builder_gate_trace": rec.get("builder_gate_trace"),
                    "joint_gate_ok": rec.get("joint_gate_ok"),
                    "joint_gate_reason": rec.get("joint_gate_reason"),
                    "joint_gate_trace": rec.get("joint_gate_trace"),
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

        fitness_g, fitness_f = _credit_assignment_v2(pair_records=pair_records)

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
                out.append(e2)
            out.sort(
                key=lambda x: float(x.get("fitness", float("-inf") if str(metric_mode) == "maximize" else float("inf"))),
                reverse=bool(str(metric_mode) == "maximize"),
            )
            return out

        ranked_g = _rank_entries(list(g_map.values()), fitness_g, kind="g")
        ranked_f = _rank_entries(list(f_map.values()), fitness_f, kind="f")
        elites_g = ranked_g[: max(0, elite_g)]
        elites_f = ranked_f[: max(0, elite_f)]

        # MAP-Elites archive update (8x8 default, top2 per cell).
        archive_bins = int(cfg_yaml.get("archive_bins", 8) or 8)
        archive_per_cell = int(cfg_yaml.get("archive_per_cell", 2) or 2)
        for e in ranked_g:
            cell = tuple(e.get("descriptor", {}).get("cell", [0, 0]))  # type: ignore[assignment]
            try:
                cell_t = (int(cell[0]), int(cell[1]))
            except Exception:  # noqa: BLE001
                cell_t = (0, 0)
            _archive_add(
                archive_g,
                cell=cell_t,
                entry={"id": e.get("id"), "signature": e.get("signature"), "ir": e.get("ir"), "descriptor": e.get("descriptor"), "fitness": e.get("fitness")},
                score=float(e.get("fitness", float("inf"))),
                per_cell=archive_per_cell,
            )
        for e in ranked_f:
            cell = tuple(e.get("descriptor", {}).get("cell", [0, 0]))  # type: ignore[assignment]
            try:
                cell_t = (int(cell[0]), int(cell[1]))
            except Exception:  # noqa: BLE001
                cell_t = (0, 0)
            _archive_add(
                archive_f,
                cell=cell_t,
                entry={"id": e.get("id"), "signature": e.get("signature"), "ir": e.get("ir"), "descriptor": e.get("descriptor"), "fitness": e.get("fitness")},
                score=float(e.get("fitness", float("inf"))),
                per_cell=archive_per_cell,
            )

        diverse_max = int(cfg_yaml.get("diverse_elites_from_archive_max", max(elite_g * 2, 1)) or max(elite_g * 2, 1))
        diverse_elites_g = _archive_flatten(archive_g, max_items=diverse_max)
        diverse_elites_f = _archive_flatten(archive_f, max_items=diverse_max)

        # Hall-of-Fame update.
        hof_g = _update_hof(hof_g, candidates=elites_g, max_size=int(cfg_yaml.get("hof_size_g", 64) or 64))
        hof_f = _update_hof(hof_f, candidates=elites_f, max_size=int(cfg_yaml.get("hof_size_f", 64) or 64))

        if elites_g:
            _atomic_write_json(os.path.join(run_dir, "best_builder.json"), dict(elites_g[0]))
        if elites_f:
            _atomic_write_json(os.path.join(run_dir, "best_loss.json"), dict(elites_f[0]))

        best_pair: Dict[str, Any] | None = None
        if isinstance(best_so_far, dict):
            gid_best = str(best_so_far.get("builder_id"))
            fid_best = str(best_so_far.get("loss_id"))
            for rec in pair_records:
                if str(rec.get("g_id")) == gid_best and str(rec.get("f_id")) == fid_best:
                    best_pair = dict(rec)
                    break
            if best_pair is None:
                best_pair = {
                    "g_id": gid_best,
                    "f_id": fid_best,
                    "score": best_so_far.get("score"),
                    "final_score": best_so_far.get("score"),
                    "stage_final": best_so_far.get("stage_final"),
                    "generation": best_so_far.get("generation"),
                    "phase": best_so_far.get("phase"),
                    "compare_target": "prev_best",
                    "metric_mode": str(metric_mode),
                    "improve_eps": float(improve_eps),
                    "reference_score": None,
                    "better_than_prev_best": True,
                }
        if best_pair is not None:
            _atomic_write_json(os.path.join(run_dir, "best_pair.json"), best_pair)

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
    LOGGER.info("Co-evolution complete. Artifacts saved under: %s", os.path.abspath(run_dir))
