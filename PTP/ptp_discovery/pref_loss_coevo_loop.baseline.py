from __future__ import annotations

"""Co-evolution loop for preference builders (g) and preference losses (f)."""

import base64
import json
import logging
import os
import pickle
import random
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
)
from ptp_discovery.free_loss_compiler import CompileError, compile_free_loss
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
        loss_observables=tuple(str(v) for v in cfg.get("loss_observables", []) if str(v).strip()),
    )


def _build_free_cfg(cfg: Mapping[str, Any], *, hf_cfg: HighFidelityConfig) -> FreeLossFidelityConfig:
    return FreeLossFidelityConfig(
        hf=hf_cfg,
        f1_steps=int(cfg.get("f1_steps", 32) or 32),
        f2_steps=int(cfg.get("f2_steps", 0) or 0),
        f3_enabled=bool(cfg.get("f3_enabled", False)),
        baseline_epoch_violation_weight=float(cfg.get("baseline_epoch_violation_weight", 1.0)),
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
    }

    try:
        compiled_g = compile_preference_builder(g_ir, operator_whitelist=operator_whitelist)
        record["g_compile_ok"] = True
        record["g_compile_reason"] = "ok"
    except PreferenceBuilderCompileError as exc:
        record["pair_ok"] = False
        record["pair_reason"] = "g_compile_failed"
        record["g_compile_ok"] = False
        record["g_compile_reason"] = str(exc)
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
            record["elapsed_s"] = float(time.time() - t0)
            return record

    if not high_fidelity_on:
        eff = float(joint_gate.effective_grad_ratio or 0.0)
        sem = float(builder_gate.semantic_pass_rate or 0.0)
        cheap_score = float(2.0 - eff - sem)
        record["pair_ok"] = True
        record["pair_reason"] = "ok_gate_only"
        record["fitness"] = {
            "fitness_score": cheap_score,
            "validation_objective": cheap_score,
            "cheap_only": True,
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
        fitness = evaluate_free_loss_candidate(compiled_f, free_cfg, pref_builder=adapter)
    except Exception as exc:  # noqa: BLE001
        record["pair_ok"] = False
        record["pair_reason"] = "high_fidelity_failed"
        record["high_fidelity_error"] = str(exc)
        record["elapsed_s"] = float(time.time() - t0)
        return record

    record["pair_ok"] = True
    record["pair_reason"] = "ok"
    record["fitness"] = dict(fitness)
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

        # Pairing plan: novelty + elites + random; prioritize coverage of distinct combinations.
        new_g_ids = [str(e["id"]) for e in g_pool]
        new_f_ids = [str(e["id"]) for e in f_pool]

        def _pairing_plan(
            *,
            elite_g_ids_: Sequence[str],
            elite_f_ids_: Sequence[str],
            novel_g_ids_: Sequence[str],
            novel_f_ids_: Sequence[str],
            all_g_ids_: Sequence[str],
            all_f_ids_: Sequence[str],
            budget: int,
            rng: random.Random,
        ) -> List[Tuple[str, str]]:
            used: set[Tuple[str, str]] = set()
            out: List[Tuple[str, str]] = []

            def _add(g: str, f: str) -> None:
                if len(out) >= budget:
                    return
                key = (g, f)
                if key in used:
                    return
                used.add(key)
                out.append(key)

            all_g = list(all_g_ids_) or list(elite_g_ids_) or list(novel_g_ids_)
            all_f = list(all_f_ids_) or list(elite_f_ids_) or list(novel_f_ids_)
            elite_g = list(elite_g_ids_) or []
            elite_f = list(elite_f_ids_) or []
            novel_g = list(novel_g_ids_) or []
            novel_f = list(novel_f_ids_) or []

            # Novelty coverage: each new candidate paired at least once.
            for g in novel_g:
                _add(g, rng.choice(elite_f or all_f))
            for f in novel_f:
                _add(rng.choice(elite_g or all_g), f)

            # Elite cross (high-value dense evaluation).
            for g in elite_g:
                for f in elite_f:
                    _add(g, f)
                    if len(out) >= budget:
                        return out

            # Coverage-first remainder, then random fill.
            remaining = budget - len(out)
            if remaining > 0:
                cov_pairs = _sample_pairs(g_ids=all_g, f_ids=all_f, budget=remaining, rng=rng)
                for g, f in cov_pairs:
                    _add(g, f)
                    if len(out) >= budget:
                        break

            if len(out) < budget:
                all_pairs = [(g, f) for g in all_g for f in all_f]
                rng.shuffle(all_pairs)
                for g, f in all_pairs:
                    _add(g, f)
                    if len(out) >= budget:
                        break
            return out[:budget]

        pairs = _pairing_plan(
            elite_g_ids_=elite_g_ids,
            elite_f_ids_=elite_f_ids,
            novel_g_ids_=new_g_ids,
            novel_f_ids_=new_f_ids,
            all_g_ids_=g_id_pool,
            all_f_ids_=f_id_pool,
            budget=int(pairing_budget),
            rng=rng,
        )

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
        compiled_f: Dict[str, Any] = {}
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

        pair_records_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for p_idx, (gid, fid) in enumerate(pairs):
            gid = str(gid)
            fid = str(fid)
            cache_key = (gid, fid, str(eval_sig))

            cached = caches.get_pair(cache_key)
            if cached is not None:
                rec = dict(cached)
                rec["generation"] = int(gen)
                rec["pair_index"] = int(p_idx)
                rec["cached"] = True
                pair_records_map[(gid, fid)] = rec
                continue

            rec: Dict[str, Any] = {
                "generation": int(gen),
                "pair_index": int(p_idx),
                "g_id": gid,
                "f_id": fid,
                "eval_budget_signature": str(eval_sig),
                "device": proxy_device_str,
                "stage": "cheap",
                "cheap_gate_on": bool(cheap_gate_on),
                "high_fidelity_on": bool(high_fidelity_on),
                "cached": False,
            }

            g_comp = compiled_g.get(gid)
            f_comp = compiled_f.get(fid)
            if g_comp is None or f_comp is None:
                rec["pair_ok"] = False
                rec["pair_reason"] = "compile_missing"
                rec["fitness"] = {"fitness_score": float("inf"), "validation_objective": float("inf"), "cheap_only": True}
                caches.set_pair(cache_key, rec)
                pair_records_map[(gid, fid)] = rec
                continue

            batch_metrics: List[Dict[str, Any]] = []
            builder_gate_first: Dict[str, Any] | None = None
            builder_ok = True
            joint_ok_all = True

            for batch_id, fc in enumerate(rollout_feature_caches):
                pref = build_or_get_pref_batch(
                    caches=caches,
                    g_id=gid,
                    batch_id=int(batch_id),
                    builder=g_comp,
                    feature_cache=fc,
                    extra={"stage": "proxy"},
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
            rec["proxy_score"] = float(proxy_score)
            rec["proxy_agg"] = dict(proxy_agg)
            rec["proxy_batches"] = batch_metrics
            rec["builder_gate_ok"] = None if builder_gate_first is None else builder_gate_first.get("builder_gate_ok")
            rec["builder_gate_reason"] = None if builder_gate_first is None else builder_gate_first.get("builder_gate_reason")
            rec["builder_gate_trace"] = None if builder_gate_first is None else builder_gate_first.get("builder_gate_trace")
            rec["joint_gate_ok"] = bool(joint_ok_all)
            rec["joint_gate_reason"] = "ok" if joint_ok_all else "joint_failed_on_some_batch"

            if cheap_gate_on and (not builder_ok or not joint_ok_all):
                rec["pair_ok"] = False
                rec["pair_reason"] = "cheap_proxy_gate_failed"
                rec["fitness"] = {
                    "fitness_score": float("inf"),
                    "validation_objective": float("inf"),
                    "cheap_only": True,
                    "proxy_score": float(proxy_score),
                }
            else:
                rec["pair_ok"] = True
                rec["pair_reason"] = "ok_proxy"
                rec["fitness"] = {
                    "fitness_score": float(proxy_score),
                    "validation_objective": float(proxy_score),
                    "cheap_only": True,
                    "proxy_score": float(proxy_score),
                    **dict(proxy_agg),
                }

            caches.set_pair(cache_key, rec)
            pair_records_map[(gid, fid)] = rec

        pair_records: List[Dict[str, Any]] = [pair_records_map[(str(g), str(f))] for (g, f) in pairs]

        # High-fidelity stage: evaluate only top-m by proxy_score.
        if high_fidelity_on:
            top_m = int(cfg_yaml.get("high_fidelity_top_m", max(1, min(len(pair_records), pairing_budget // 4))) or 1)
            candidates = [r for r in pair_records if bool(r.get("pair_ok"))]
            candidates.sort(key=lambda r: float(r.get("proxy_score", r.get("fitness", {}).get("fitness_score", float("inf")))))
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

        util_g, util_f = _credit_assignment(
            pair_records=pair_records,
            credit_mode=credit_mode,
            best_k=credit_best_k,
        )

        def _rank_entries(entries: Sequence[Mapping[str, Any]], util_map: Mapping[str, float]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for e in entries:
                e2 = dict(e)
                e2["utility"] = float(util_map.get(str(e2.get("id")), float("-inf")))
                out.append(e2)
            out.sort(key=lambda x: float(x.get("utility", float("-inf"))), reverse=True)
            return out

        ranked_g = _rank_entries(list(g_map.values()), util_g)
        ranked_f = _rank_entries(list(f_map.values()), util_f)
        elites_g = ranked_g[: max(0, elite_g)]
        elites_f = ranked_f[: max(0, elite_f)]
        diverse_elites_g = ranked_g[: max(elite_g * 2, elite_g)]
        diverse_elites_f = ranked_f[: max(elite_f * 2, elite_f)]

        if elites_g:
            _atomic_write_json(os.path.join(run_dir, "best_builder.json"), dict(elites_g[0]))
        if elites_f:
            _atomic_write_json(os.path.join(run_dir, "best_loss.json"), dict(elites_f[0]))

        best_pair: Dict[str, Any] | None = None
        best_score = float("inf")
        for rec in pair_records:
            fitness = rec.get("fitness")
            if not isinstance(fitness, dict):
                continue
            score = fitness.get("fitness_score", fitness.get("validation_objective"))
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if score_f < best_score:
                best_score = score_f
                best_pair = dict(rec)
        if best_pair is not None:
            _atomic_write_json(os.path.join(run_dir, "best_pair.json"), best_pair)

        _save_checkpoint(run_dir, _checkpoint_state(gen + 1))

    LOGGER.info("Co-evolution complete. Artifacts saved under: %s", os.path.abspath(run_dir))
