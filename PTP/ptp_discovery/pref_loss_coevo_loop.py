from __future__ import annotations

"""Co-evolution loop for preference builders (g) and preference losses (f)."""

import base64
import collections
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
            min_coverage=float(gate_cfg.get("min_coverage", 1.0) or 1.0),
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
        min_cov = float(gate_cfg.get("min_coverage", 1.0) or 1.0)
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
    joint_ok_all = True
    first_joint_fail_reason: str | None = None
    first_joint_fail_trace: Dict[str, Any] | None = None
    joint_failure_kinds: collections.Counter[str] = collections.Counter()

    for local_batch_id, fc in enumerate(rollout_feature_caches):
        pref = build_or_get_pref_batch(
            caches=caches,
            g_id=str(gid),
            batch_id=int(pref_batch_id_offset + local_batch_id),
            builder=g_comp,
            feature_cache=fc,
            extra={"stage": "proxy", "seed_signature": str(seed_sig)},
        )
        if builder_gate_first is None:
            bg = run_preference_builder_gates(
                pref,
                feature_cache=fc,
                min_pairs=int(cfg_yaml.get("builder_min_pairs", 1) or 1),
                min_coverage=float(cfg_yaml.get("builder_min_coverage", 1.0) or 1.0),
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
        joint_ok_all = joint_ok_all and bool(m.get("joint_ok", False))
        if not bool(m.get("joint_ok", False)):
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
            "joint_gate_ok": bool(joint_ok_all),
            "joint_gate_reason": (
                "ok"
                if joint_ok_all
                else (
                    f"joint_failed:{joint_failure_kinds.most_common(1)[0][0]}"
                    if joint_failure_kinds
                    else (first_joint_fail_reason or "joint_failed_on_some_batch")
                )
            ),
            "joint_gate_trace": None if joint_ok_all else first_joint_fail_trace,
        }
    )

    if cheap_gate_on and (not builder_ok or not joint_ok_all):
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
        min_coverage=float(cfg.get("builder_min_coverage", 1.0) or 1.0),
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
            - checkpoint.json, best_builder.json, best_loss.json, best_pair.json
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f) or {}
    if not isinstance(cfg_yaml, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    cfg_yaml.update({k: v for k, v in overrides.items() if v is not None})

    seed = int(cfg_yaml.get("seed", 0))
    _set_seed(seed)
    rng = random.Random(seed)

    generations = int(cfg_yaml.get("generations", 1))
    pop_g = int(cfg_yaml.get("pop_g", 8))
    pop_f = int(cfg_yaml.get("pop_f", 8))
    elite_g = int(cfg_yaml.get("elite_g", 4))
    elite_f = int(cfg_yaml.get("elite_f", 4))
    pairing_budget = int(cfg_yaml.get("pairing_budget_per_gen", 16))

    cheap_gate_on = bool(cfg_yaml.get("cheap_gate_on", True))
    high_fidelity_on = bool(cfg_yaml.get("high_fidelity_on", True))

    credit_mode = str(cfg_yaml.get("credit_assignment", "mean") or "mean")
    credit_best_k = int(cfg_yaml.get("credit_best_k", 3) or 3)

    operator_whitelist = list(cfg_yaml.get("operator_whitelist", []))

    devices = cfg_yaml.get("devices")
    if isinstance(devices, (list, tuple)) and devices:
        device_list = [str(d) for d in devices]
    else:
        device_list = [str(cfg_yaml.get("device", "cuda"))]

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
        LOGGER.info("Resuming run_dir=%s next_generation=%s", run_dir, resume_state.get("next_generation"))
    else:
        run_dir = _timestamp_dir(out_root)

    LOGGER.info("Run directory: %s", os.path.abspath(run_dir))

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
            "min_coverage": float(cfg_yaml.get("builder_min_coverage", 1.0) or 1.0),
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

    if resume_state is None:
        for path in (builders_jsonl, losses_jsonl, pairs_jsonl, gate_jsonl):
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
    }
    eval_sig = eval_budget_signature(
        cfg=sig_hf_cfg,
        proxy_problem_size=proxy_problem_size,
        proxy_batch_size=proxy_batch_size,
        proxy_batches=proxy_batches,
        proxy_weights={str(k): float(v) for k, v in dict(proxy_weights).items()},
        extra_budget=micro_budget,
    )

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

    def _checkpoint_state(next_generation: int) -> Dict[str, Any]:
        return {
            "config_path": os.path.abspath(config_path),
            "seed": int(seed),
            "next_generation": int(next_generation),
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

    llm_feedback_state: Dict[str, Any] = {}

    for gen in range(gen_start, generations):
        LOGGER.info("=== coevo generation %d/%d ===", gen, generations - 1)

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
                "objective": "lower_is_better",
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
                    min_coverage=float(cfg_yaml.get("builder_min_coverage", 1.0) or 1.0),
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
        proxy_device_str = str(cfg_yaml.get("proxy_device", device_list[0]))
        if proxy_device_str == "cuda" and not torch.cuda.is_available():
            proxy_device_str = "cpu"
        proxy_device = torch.device(proxy_device_str)

        # Pre-build (or reuse) rollout feature caches for the cheap stage.
        rollout_feature_caches: List[Dict[str, torch.Tensor]] = []
        for batch_id in range(int(proxy_batches)):
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

        bins = int(cfg_yaml.get("archive_bins", 8) or 8)
        pair_count_cap = int(cfg_yaml.get("descriptor_pair_count_cap", 4096) or 4096)
        loss_scale = float(cfg_yaml.get("descriptor_loss_scale", 5.0) or 5.0)

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

        # Anchor evaluation for anti-collapse.
        anchor_enabled = bool(cfg_yaml.get("anchor_enabled", True))
        anchor_max_score = float(cfg_yaml.get("anchor_proxy_max_score", 10.0) or 10.0)
        eliminated_g: set[str] = set()
        eliminated_f: set[str] = set()

        anchor_pairs: List[Tuple[str, str]] = []
        anchor_reasons: Dict[Tuple[str, str], List[str]] = {}
        if anchor_enabled:
            for gid in list(new_g_ids):
                anchor_pairs.append((str(gid), F_REF_ID))
                anchor_reasons[(str(gid), F_REF_ID)] = ["anchor_builder_vs_f_ref"]
            for fid in list(new_f_ids):
                anchor_pairs.append((G_REF_ID, str(fid)))
                anchor_reasons[(G_REF_ID, str(fid))] = ["anchor_loss_vs_g_ref"]

        anchor_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
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
                cheap_gate_on=True,
            )
            anchor_records[(str(gid), str(fid))] = rec
            if str(fid) == F_REF_ID and str(gid) in new_g_ids:
                if (not bool(rec.get("pair_ok"))) or float(rec.get("score", float("inf"))) > anchor_max_score:
                    eliminated_g.add(str(gid))
            if str(gid) == G_REF_ID and str(fid) in new_f_ids:
                if (not bool(rec.get("pair_ok"))) or float(rec.get("score", float("inf"))) > anchor_max_score:
                    eliminated_f.add(str(fid))

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
        reasons_by_pair: Dict[Tuple[str, str], List[str]] = dict(core_reasons)
        for k, v in anchor_reasons.items():
            reasons_by_pair.setdefault(k, []).extend(list(v))

        pair_records_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Carry over anchor records (already computed).
        pair_records_map.update(anchor_records)

        # Cheap stage evaluation for scheduled pairs using Common Random Numbers (base_seed).
        for p_idx, (gid, fid) in enumerate(pairs):
            if (gid, fid) in pair_records_map and pair_records_map[(gid, fid)].get("stage") == "anchor":
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
                cheap_gate_on=bool(cheap_gate_on),
            )
            pair_records_map[(str(gid), str(fid))] = rec

        # 2-seed recheck: best_pair + elite-boundary pairs (cheap stage only).
        recheck_enabled = bool(cfg_yaml.get("recheck_enabled", True))
        recheck_num_seeds = int(cfg_yaml.get("recheck_num_seeds", 2) or 2)
        if recheck_enabled and recheck_num_seeds >= 2 and pairs:
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

            seed2 = int(base_seed + recheck_offset)
            seed2_sig = seed_signature_for_proxy(
                seed=seed2,
                problem_size=int(proxy_problem_size),
                batch_ids=list(range(int(proxy_batches))),
                batch_size=int(proxy_batch_size),
            )
            rollout_feature_caches_2: List[Dict[str, torch.Tensor]] = []
            for batch_id in range(int(proxy_batches)):
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
                    pair_index=int(pair_records_map.get((gid, fid), {}).get("pair_index", 0)),
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
                    cheap_gate_on=bool(cheap_gate_on),
                )
                pair_records_map[(str(gid), str(fid))] = rec2

        pair_records: List[Dict[str, Any]] = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]

        # Optional Stage B: offline micro-unroll on cached rollouts (no new rollouts).
        # This improves selection signal at a fraction of HF cost by taking a few
        # gradient steps on cached log_prob tensors.
        micro_enabled = bool(cfg_yaml.get("micro_unroll_enabled", False))
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
                if bool(r.get("pair_ok"))
                and str(r.get("g_id")) != G_REF_ID
                and str(r.get("f_id")) != F_REF_ID
                and str(r.get("stage")) != "anchor"
            ]
            mu_candidates.sort(key=lambda r: float(r.get("score", float("inf"))))
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
                for r in mu_eval:
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
                        mu_score, mu_metrics = micro_unroll_score_for_pair(
                            g=g_comp,
                            f=f_comp,
                            rollout_feature_caches=rollout_feature_caches,
                            steps=int(micro_steps),
                            lr=float(micro_lr),
                            alpha=float(micro_alpha),
                            weight_decay=float(micro_weight_decay),
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
                    except Exception as exc:  # noqa: BLE001
                        rec_mu = dict(pair_records_map.get((gid, fid), dict(r)))
                        rec_mu["pair_ok"] = False
                        rec_mu["pair_reason"] = "micro_unroll_failed"
                        rec_mu["micro_error"] = str(exc)
                        rec_mu["stage"] = "micro_unroll"
                        rec_mu["score"] = float("inf")
                        caches.set_pair(cache_key, rec_mu)
                        pair_records_map[(gid, fid)] = rec_mu

                # Rebuild after micro-unroll overwrites `score` for some pairs.
                pair_records = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]

        # High-fidelity stage: evaluate only top-m by cheap proxy `score`.
        if high_fidelity_on:
            top_m = int(cfg_yaml.get("high_fidelity_top_m", max(1, min(len(pair_records), pairing_budget // 4))) or 1)
            candidates = [
                r
                for r in pair_records
                if bool(r.get("pair_ok"))
                and str(r.get("g_id")) != G_REF_ID
                and str(r.get("f_id")) != F_REF_ID
                and str(r.get("stage")) != "anchor"
            ]
            if micro_enabled:
                candidates = [r for r in candidates if isinstance(r.get("micro_metrics"), dict)]
            candidates.sort(key=lambda r: float(r.get("score", float("inf"))))

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
                        "pair_index": int(r.get("pair_index", -1)),
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

            # Merge high-fidelity results back into cached records.
            for hf_rec in hf_results:
                gid = str(hf_rec.get("g_id"))
                fid = str(hf_rec.get("f_id"))
                cache_key = (gid, fid, str(eval_sig))
                merged = dict(pair_records_map.get((gid, fid), {}))
                merged.update(dict(hf_rec))
                merged["eval_budget_signature"] = str(eval_sig)
                merged["stage"] = "high_fidelity"
                if isinstance(merged.get("fitness"), dict):
                    merged["fitness"]["cheap_only"] = False
                    merged["fitness"]["proxy_score"] = float(merged.get("proxy_score", merged["fitness"].get("fitness_score", float("inf"))))
                caches.set_pair(cache_key, merged)
                pair_records_map[(gid, fid)] = merged

            pair_records = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]

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
        best_score_preview = float("inf")
        for rec in pair_records:
            if str(rec.get("g_id")) == G_REF_ID or str(rec.get("f_id")) == F_REF_ID:
                continue
            if str(rec.get("stage")) == "anchor":
                continue
            if not bool(rec.get("pair_ok")):
                continue
            try:
                score_f = float(rec.get("score", float("inf")))
            except (TypeError, ValueError):
                continue
            if score_f < best_score_preview:
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
        llm_feedback_state["prev_gen_best_pair_preview"] = best_pair_preview
        try:
            llm_feedback_state["prev_gen_llm_cache"] = dict(loss_llm_ops.llm_cache_stats())
        except Exception:  # noqa: BLE001
            pass
        if "cheap" not in stage_ctr and "cheap_recheck" not in stage_ctr:
            LOGGER.warning(
                "Gen %d produced no core cheap evaluations (stage='cheap'). "
                "This usually means candidate pools collapsed or pairing_budget_per_gen is too small after anchors.",
                int(gen),
            )
        if high_fidelity_on and "high_fidelity" not in stage_ctr:
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
                }
            )

        _append_jsonl(pairs_jsonl, pair_records)
        _append_jsonl(gate_jsonl, gate_records)

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
            out.sort(key=lambda x: float(x.get("fitness", float("inf"))))
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
        best_score = float("inf")
        for rec in pair_records:
            if str(rec.get("g_id")) == G_REF_ID or str(rec.get("f_id")) == F_REF_ID:
                continue
            if str(rec.get("stage")) == "anchor":
                continue
            try:
                score_f = float(rec.get("score", float("inf")))
            except (TypeError, ValueError):
                continue
            if score_f < best_score:
                best_score = score_f
                best_pair = dict(rec)
        if best_pair is not None:
            _atomic_write_json(os.path.join(run_dir, "best_pair.json"), best_pair)

        _save_checkpoint(run_dir, _checkpoint_state(gen + 1))

    LOGGER.info("Co-evolution complete. Artifacts saved under: %s", os.path.abspath(run_dir))
