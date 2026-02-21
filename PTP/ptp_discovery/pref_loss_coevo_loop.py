from __future__ import annotations

"""Co-evolution loop for preference builders (g) and preference losses (f)."""

import base64
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
from fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed
from fitness.pref_loss_fidelity import (
    PrefLossEvalCaches,
    aggregate_proxy_metrics,
    build_or_get_pref_batch,
    build_or_get_rollout_feature_cache,
    eval_budget_signature,
    load_pair_cache_from_pairs_jsonl,
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
from ptp_discovery.free_loss_ir import FreeLossIR, ir_from_json as free_loss_ir_from_json
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
        kind = "all_pairs"
        code = base_code
        if i > 0:
            choice = rng.choice(["anchor_best", "gap_threshold", "sampled_pairs"])
            if choice == "anchor_best":
                kind = "anchor_best"
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
                kind = "gap_threshold"
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
            else:
                kind = "sampled_pairs"
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
            f"    scale = float(extra.get('hyperparams', {{}}).get('scale', {scale}))\n"
        )
        if use_cost:
            code += (
                "    cost_a = batch['cost_a']\n"
                "    cost_b = batch['cost_b']\n"
                "    gap = (cost_b - cost_a).detach()\n"
                "    x = scale * (lpw - lpl) - 0.1 * gap\n"
            )
        else:
            code += "    x = scale * (lpw - lpl)\n"
        code += (
            "    loss = -ops.logsigmoid(x)\n"
            "    if weight is not None:\n"
            "        loss = loss * weight\n"
            "    return loss.mean()\n"
        )

        ir_obj = {
            "name": name,
            "intuition": "rule_based: pairwise logsigmoid loss",
            "pseudocode": "loss = -log(sigmoid(scale*(lpw-lpl)))",
            "hyperparams": {"scale": scale},
            "operators_used": ["logsigmoid"],
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


def _propose_builders_for_generation(
    *,
    generation: int,
    pop_g: int,
    elites_g: Sequence[Mapping[str, Any]],
    diverse_elites_g: Sequence[Mapping[str, Any]],
    rng: random.Random,
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
        out.append({"ir": ir, "origin": "elite_copy", "parents": [str(item.get("id", ""))]})

    # Mutations/crossover and fresh seeds.
    while len(out) < pop_g:
        op = rng.choice(["mutate", "crossover", "seed"]) if parent_pool else "seed"
        if op == "seed":
            ir = _make_builtin_builder_irs(rng, 1)[0]
            out.append({"ir": ir, "origin": "seed", "parents": []})
            continue

        if op == "mutate":
            parent = rng.choice(parent_pool)
            # Mutation: resample a rule-based variant; keep parent id as provenance.
            ir = _make_builtin_builder_irs(rng, 1)[0]
            ir.name = f"{ir.name}_m_from_{str(parent.get('id',''))[:12]}"
            out.append({"ir": ir, "origin": "mutate", "parents": [str(parent.get("id", ""))]})
            continue

        # crossover
        p1 = rng.choice(parent_pool)
        p2 = rng.choice(parent_pool)
        ir = _make_builtin_builder_irs(rng, 1)[0]
        ir.name = f"{ir.name}_x_{str(p1.get('id',''))[:8]}_{str(p2.get('id',''))[:8]}"
        out.append(
            {
                "ir": ir,
                "origin": "crossover",
                "parents": [str(p1.get("id", "")), str(p2.get("id", ""))],
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
        out.append({"ir": ir, "origin": "elite_copy", "parents": [str(item.get("id", ""))]})

    while len(out) < pop_f:
        op = rng.choice(["mutate", "crossover", "seed"]) if parent_pool else "seed"
        if op == "seed":
            ir = _make_builtin_loss_irs(rng, 1)[0]
            out.append({"ir": ir, "origin": "seed", "parents": []})
            continue

        if op == "mutate":
            parent = rng.choice(parent_pool)
            ir = _make_builtin_loss_irs(rng, 1)[0]
            ir.name = f"{ir.name}_m_from_{str(parent.get('id',''))[:12]}"
            out.append({"ir": ir, "origin": "mutate", "parents": [str(parent.get("id", ""))]})
            continue

        p1 = rng.choice(parent_pool)
        p2 = rng.choice(parent_pool)
        ir = _make_builtin_loss_irs(rng, 1)[0]
        ir.name = f"{ir.name}_x_{str(p1.get('id',''))[:8]}_{str(p2.get('id',''))[:8]}"
        out.append({"ir": ir, "origin": "crossover", "parents": [str(p1.get("id", "")), str(p2.get("id", ""))]})

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
        implementation_hint=dict(  # type: ignore[arg-type]
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
        batch_metrics.append(m)

    proxy_score, proxy_agg = aggregate_proxy_metrics(batch_metrics, proxy_weights=dict(proxy_weights))
    per_batch_summary = [
        {
            "loss": float(m.get("loss", float("inf"))),
            "effective_grad_ratio": float(m.get("effective_grad_ratio", 0.0)),
            "ess_ratio": float(m.get("ess_ratio", 0.0)),
            "pair_count": int(m.get("pair_count", 0)),
            "joint_ok": bool(m.get("joint_ok", False)),
        }
        for m in batch_metrics
    ]
    proxy_metrics: Dict[str, Any] = dict(proxy_agg)
    proxy_metrics["batches"] = per_batch_summary

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
            "joint_gate_reason": "ok" if joint_ok_all else "joint_failed_on_some_batch",
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
    eval_sig = eval_budget_signature(
        cfg=sig_hf_cfg,
        proxy_problem_size=proxy_problem_size,
        proxy_batch_size=proxy_batch_size,
        proxy_batches=proxy_batches,
        proxy_weights={str(k): float(v) for k, v in dict(proxy_weights).items()},
    )

    # Optional external baseline (metrics.csv) for epoch-by-epoch comparisons during HF.
    baseline_epoch_objectives: List[float] | None = None
    baseline_early_valid: float | None = None
    early_eval_steps: int = 0
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
    if baseline_metrics_csv and baseline_ckpt_epoch is not None and int(getattr(sig_hf_cfg, "hf_epochs", 0) or 0) > 0:
        metrics_path = _abs_from_repo_root(str(baseline_metrics_csv))
        start_epoch = int(baseline_ckpt_epoch) + 1
        try:
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
            early_eval_steps = 0
        else:
            LOGGER.info(
                "External baseline loaded for HF comparisons: metrics=%s epoch_start=%d epochs=%d val_column=%s",
                os.path.abspath(metrics_path),
                int(baseline_ckpt_epoch) + 1,
                int(sig_hf_cfg.hf_epochs),
                str(baseline_val_column),
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

    for gen in range(gen_start, generations):
        LOGGER.info("=== coevo generation %d/%d ===", gen, generations - 1)

        proposed_g = _propose_builders_for_generation(
            generation=int(gen),
            pop_g=int(pop_g),
            elites_g=elites_g,
            diverse_elites_g=diverse_elites_g,
            rng=rng,
        )
        proposed_f = _propose_losses_for_generation(
            generation=int(gen),
            pop_f=int(pop_f),
            elites_f=elites_f,
            diverse_elites_f=diverse_elites_f,
            rng=rng,
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
                unique.append({"ir": ir, "origin": "seed_fill", "parents": []})
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
                unique2.append({"ir": ir, "origin": "seed_fill", "parents": []})
                seen_f.add(sig)
            return unique2[:pop_f]

        proposed_g = _fill_unique_builders(proposed_g)
        proposed_f = _fill_unique_losses(proposed_f)

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
                "parents": list(proposal.get("parents", [])),
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
                "parents": list(proposal.get("parents", [])),
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
            candidates.sort(key=lambda r: float(r.get("score", float("inf"))))
            selected = candidates[: max(0, top_m)]

            hf_tasks: List[Dict[str, Any]] = []
            for r in selected:
                gid = str(r["g_id"])
                fid = str(r["f_id"])
                cache_key = (gid, fid, str(eval_sig))
                cached = caches.get_pair(cache_key)
                if isinstance(cached, dict) and str(cached.get("stage")) == "high_fidelity" and cached.get("fitness"):
                    continue
                hf_tasks.append(
                    {
                        "generation": int(gen),
                        "pair_index": int(r.get("pair_index", -1)),
                        "g_entry": g_map[gid],
                        "f_entry": f_map[fid],
                        "cfg_yaml": dict(cfg_yaml),
                        "device_str": device_list[int(r.get("pair_index", 0)) % len(device_list)],
                        "operator_whitelist": list(operator_whitelist),
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
                if mp_enabled and mp_processes > 0 and len(hf_tasks) > 1:
                    import multiprocessing as mp

                    ctx = mp.get_context(mp_start_method)
                    procs = min(int(mp_processes), max(1, len(hf_tasks)))
                    LOGGER.info(
                        "High-fidelity via mp: start_method=%s processes=%d tasks=%d",
                        mp_start_method,
                        procs,
                        len(hf_tasks),
                    )
                    with ctx.Pool(processes=procs) as pool:
                        for rec in pool.imap_unordered(_evaluate_pair_worker, hf_tasks):
                            hf_results.append(dict(rec))
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
