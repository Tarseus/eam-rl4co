from __future__ import annotations

"""Cached, multi-fidelity evaluation for (PreferenceBuilder g, FreeLoss f) pairs.

This module is intended for co-evolution (Route B):
  - Cheap stage: reuse a fixed set of rollouts (feature_cache) across many (g,f) pairs,
    without training the policy. Produce joint proxy metrics + a scalar proxy_score.
  - High-fidelity stage: run short training + validation objective only for top-m pairs.

Default caching is in-memory. The cache classes expose a `persist_dir` hook so the
backend can be extended to on-disk caching without changing call sites.
"""

import json
import os
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import torch

from fitness.free_loss_fidelity import (
    PrefBatch,
    extract_feature_cache,
    resolve_pomo_size,
    _rl4co_build_env,
    _rl4co_build_policy,
    _rl4co_objective_from_reward,
    _rl4co_rollout,
)
from fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed
from ptp_discovery.free_loss_compiler import CompiledFreeLoss
from ptp_discovery.free_loss_gates import JointPreferenceGateResult, run_joint_preference_gates
from ptp_discovery.pref_builder_compiler import CompiledPreferenceBuilder


RolloutKey = Tuple[int, int, int]  # (seed, problem_size, batch_id)
PrefKey = Tuple[str, int]  # (g_id, batch_id)
PairKey = Tuple[str, str, str]  # (g_id, f_id, eval_budget_signature)


@dataclass
class PrefLossEvalCaches:
    """In-memory caches for rollout/pref/pair evaluations.

    - rollout_cache: feature_cache keyed by (seed, problem_size, batch_id)
    - pref_cache: PrefBatch keyed by (g_id, batch_id)
    - pair_cache: JSON-serializable evaluation result keyed by (g_id, f_id, eval_sig)
    """

    rollout_cache: MutableMapping[RolloutKey, Dict[str, torch.Tensor]] = field(default_factory=dict)
    pref_cache: MutableMapping[PrefKey, PrefBatch] = field(default_factory=dict)
    # Stored records are JSON-serializable and may represent an aggregation over multiple seed_signatures.
    pair_cache: MutableMapping[PairKey, Dict[str, Any]] = field(default_factory=dict)
    persist_dir: str | None = None

    def get_rollout(self, key: RolloutKey) -> Dict[str, torch.Tensor] | None:
        return self.rollout_cache.get(key)

    def set_rollout(self, key: RolloutKey, feature_cache: Dict[str, torch.Tensor]) -> None:
        self.rollout_cache[key] = feature_cache

    def get_pref(self, key: PrefKey) -> PrefBatch | None:
        return self.pref_cache.get(key)

    def set_pref(self, key: PrefKey, pref_batch: PrefBatch) -> None:
        self.pref_cache[key] = pref_batch

    def prune_pref_cache(
        self,
        *,
        keep_g_ids: Sequence[str] | None = None,
        keep_batch_ids: Sequence[int] | None = None,
    ) -> int:
        """Drop pref_cache entries outside the provided keep-sets.

        This cache can grow unbounded in long runs because new builder ids are
        created every generation. Pruning keeps memory bounded while preserving
        the main speedups for active candidates (elites/HoF/current pools).
        """

        keep_g = set(str(x) for x in keep_g_ids) if keep_g_ids is not None else None
        keep_b = set(int(x) for x in keep_batch_ids) if keep_batch_ids is not None else None
        if keep_g is None and keep_b is None:
            return 0

        to_delete: list[PrefKey] = []
        for (g_id, batch_id) in self.pref_cache.keys():
            if keep_g is not None and str(g_id) not in keep_g:
                to_delete.append((g_id, batch_id))
                continue
            if keep_b is not None and int(batch_id) not in keep_b:
                to_delete.append((g_id, batch_id))
                continue

        for k in to_delete:
            try:
                del self.pref_cache[k]
            except KeyError:
                pass
        return int(len(to_delete))

    def get_pair(self, key: PairKey) -> Dict[str, Any] | None:
        return self.pair_cache.get(key)

    def set_pair(self, key: PairKey, record: Mapping[str, Any]) -> None:
        """Insert or merge a pair evaluation record.

        Records may include repeated evaluations under different `seed_signature`s.
        This method merges them into one aggregated record keyed by (g_id, f_id, eval_budget_signature).
        """

        new_rec = dict(record)
        existing = self.pair_cache.get(key)
        if existing is None:
            agg = _normalize_pair_record(new_rec)
        else:
            agg = _merge_pair_records(existing, new_rec)
        self.pair_cache[key] = agg
        if self.persist_dir:
            self._persist_pair_record(key, self.pair_cache[key])

    def _persist_pair_record(self, key: PairKey, record: Mapping[str, Any]) -> None:
        """Optional persistence hook (default: write one JSON per key)."""
        if not self.persist_dir:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        g_id, f_id, sig = key
        safe_name = f"{g_id}__{f_id}__{sig}.json".replace(os.sep, "_")
        path = os.path.join(self.persist_dir, safe_name)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(record), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def seed_signature_for_proxy(
    *,
    seed: int,
    problem_size: int,
    batch_ids: Sequence[int],
    batch_size: int,
) -> str:
    payload = {
        "seed": int(seed),
        "problem_size": int(problem_size),
        "batch_ids": [int(b) for b in batch_ids],
        "batch_size": int(batch_size),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha1(blob).hexdigest()[:16]


def _normalize_pair_record(rec: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    if str(out.get("stage", "")).strip().lower() == "high_fidelity":
        # High-fidelity records are treated as final scores; do not add a new seed record.
        out.setdefault("seed_records", {})
        seed_sig = out.get("seed_signature")
        if isinstance(seed_sig, (list, tuple)):
            out["seed_signature"] = [str(x) for x in seed_sig]
        elif seed_sig is None:
            out["seed_signature"] = []
        else:
            out["seed_signature"] = [str(seed_sig)]
        try:
            out["score"] = float(out.get("score", float("inf")))
        except (TypeError, ValueError):
            out["score"] = float("inf")
        if not isinstance(out.get("proxy_metrics"), dict):
            out["proxy_metrics"] = dict(out.get("proxy_metrics") or {})
        return out
    seed_sig = out.get("seed_signature")
    if isinstance(seed_sig, (list, tuple)):
        seed_list = [str(x) for x in seed_sig]
    elif seed_sig is None:
        seed_list = []
    else:
        seed_list = [str(seed_sig)]

    score = out.get("score")
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = float("inf")

    proxy = out.get("proxy_metrics")
    if not isinstance(proxy, dict):
        proxy = {}

    seed_records: Dict[str, Any] = {}
    if seed_list:
        seed_records[seed_list[0]] = {
            "score": score_f,
            "proxy_metrics": dict(proxy),
            "descriptor": out.get("descriptor"),
        }
    out["seed_records"] = seed_records
    out["seed_signature"] = sorted(seed_records.keys())
    out["score"] = float(score_f)
    out["proxy_metrics"] = dict(proxy)
    return out


def _merge_pair_records(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> Dict[str, Any]:
    ex = dict(existing)
    inc = dict(incoming)

    if str(inc.get("stage", "")).strip().lower() == "high_fidelity":
        # Preserve existing seed_records (proxy stage) and overwrite final score/fitness.
        ex.setdefault("seed_records", ex.get("seed_records") if isinstance(ex.get("seed_records"), dict) else {})
        ex["stage"] = "high_fidelity"
        ex["pair_ok"] = bool(inc.get("pair_ok", ex.get("pair_ok", False)))
        ex["pair_reason"] = inc.get("pair_reason", ex.get("pair_reason", ""))
        ex["fitness"] = inc.get("fitness", ex.get("fitness"))
        ex["elapsed_s"] = inc.get("elapsed_s", ex.get("elapsed_s"))
        ex["generation"] = inc.get("generation", ex.get("generation"))
        ex["pair_index"] = inc.get("pair_index", ex.get("pair_index"))
        ex["cached"] = inc.get("cached", ex.get("cached", False))
        ex["eval_budget_signature"] = inc.get("eval_budget_signature", ex.get("eval_budget_signature"))
        # Keep proxy fields from incoming if present; otherwise retain existing.
        if inc.get("proxy_metrics") is not None:
            ex["proxy_metrics"] = inc.get("proxy_metrics")
        if inc.get("descriptor") is not None:
            ex["descriptor"] = inc.get("descriptor")
        if inc.get("seed_signature") is not None:
            ex["seed_signature"] = inc.get("seed_signature")
        try:
            ex["score"] = float(inc.get("score", ex.get("score", float("inf"))))
        except (TypeError, ValueError):
            ex["score"] = float("inf")
        return ex

    ex_seed_records = ex.get("seed_records")
    if not isinstance(ex_seed_records, dict):
        ex_seed_records = {}

    inc_norm = _normalize_pair_record(inc)
    inc_seed_records = inc_norm.get("seed_records")
    if isinstance(inc_seed_records, dict):
        ex_seed_records.update(inc_seed_records)

    ex["seed_records"] = ex_seed_records
    ex["seed_signature"] = sorted(str(k) for k in ex_seed_records.keys())

    # Aggregate score: mean over seeds (lower is better).
    scores: list[float] = []
    for v in ex_seed_records.values():
        if not isinstance(v, dict):
            continue
        try:
            scores.append(float(v.get("score")))
        except (TypeError, ValueError):
            continue
    ex["score"] = float(sum(scores) / len(scores)) if scores else float("inf")

    # Aggregate proxy metrics (mean for a few scalar keys; keep other fields from the latest record).
    def _mean_key(key: str) -> float | None:
        vals: list[float] = []
        for v in ex_seed_records.values():
            if not isinstance(v, dict):
                continue
            pm = v.get("proxy_metrics")
            if not isinstance(pm, dict):
                continue
            try:
                vals.append(float(pm.get(key)))
            except (TypeError, ValueError):
                continue
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    proxy_out = dict(ex.get("proxy_metrics") or {})
    for k in ("proxy_loss_mean", "proxy_effective_grad_ratio_mean", "proxy_ess_ratio_mean"):
        mv = _mean_key(k)
        if mv is not None:
            proxy_out[k] = float(mv)
    if isinstance(inc_norm.get("proxy_metrics"), dict):
        proxy_out.update({k: v for k, v in inc_norm["proxy_metrics"].items() if k not in proxy_out})
    ex["proxy_metrics"] = proxy_out

    # Descriptor: keep last incoming if provided, else keep existing.
    if inc.get("descriptor") is not None:
        ex["descriptor"] = inc.get("descriptor")

    # Keep a few latest bookkeeping fields.
    for k in ("stage", "pair_ok", "pair_reason", "fitness", "elapsed_s", "generation", "pair_index", "cached"):
        if k in inc:
            ex[k] = inc[k]
    return ex


def eval_budget_signature(
    *,
    cfg: HighFidelityConfig,
    proxy_problem_size: int,
    proxy_batch_size: int,
    proxy_batches: int,
    proxy_weights: Mapping[str, float],
    extra_budget: Mapping[str, Any] | None = None,
    version: str = "v1",
) -> str:
    """Return a stable signature describing evaluation budget/settings."""

    payload = {
        "version": str(version),
        "seed": int(cfg.seed),
        "problem": str(cfg.problem),
        "backend": str(cfg.backend),
        "policy_name": str(cfg.policy_name),
        "rollout_strategy": str(cfg.rollout_strategy),
        "objective_sign": str(cfg.objective_sign),
        "proxy_problem_size": int(proxy_problem_size),
        "proxy_batch_size": int(proxy_batch_size),
        "proxy_batches": int(proxy_batches),
        "proxy_weights": {str(k): float(v) for k, v in dict(proxy_weights).items()},
        "hf_steps": int(cfg.hf_steps),
        "hf_epochs": int(cfg.hf_epochs),
        "hf_instances_per_epoch": int(cfg.hf_instances_per_epoch),
        "train_batch_size": int(cfg.train_batch_size),
        "num_validation_episodes": int(cfg.num_validation_episodes),
        "validation_batch_size": int(cfg.validation_batch_size),
        "valid_problem_sizes": [int(x) for x in cfg.valid_problem_sizes],
    }
    if extra_budget:
        payload["extra_budget"] = dict(extra_budget)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha1(blob).hexdigest()[:16]


@torch.no_grad()
def build_or_get_rollout_feature_cache(
    *,
    caches: PrefLossEvalCaches,
    cfg: HighFidelityConfig,
    seed: int,
    problem_size: int,
    batch_id: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create or reuse rollout feature_cache keyed by (seed, problem_size, batch_id).

    Output:
        feature_cache dict of tensors (typically includes objective/log_prob and CO features).
    """

    key: RolloutKey = (int(seed), int(problem_size), int(batch_id))
    cached = caches.get_rollout(key)
    if cached is not None:
        return cached

    # Make rollout deterministic per batch_id.
    _set_seed(int(seed) + int(batch_id))

    env = _rl4co_build_env(cfg, int(problem_size)).to(device)
    policy, rollout_strategy = _rl4co_build_policy(cfg, env)
    policy = policy.to(device)
    policy.eval()

    num_rollouts = resolve_pomo_size(cfg.pomo_size, int(problem_size))
    reward, log_likelihood = _rl4co_rollout(
        env,
        policy,
        int(batch_size),
        int(num_rollouts),
        phase="test",
        rollout_strategy=str(rollout_strategy),
        device=device,
    )
    objective = _rl4co_objective_from_reward(reward, cfg)
    feature_cache = extract_feature_cache(objective, log_likelihood)
    caches.set_rollout(key, feature_cache)
    return feature_cache


def build_or_get_pref_batch(
    *,
    caches: PrefLossEvalCaches,
    g_id: str,
    batch_id: int,
    builder: CompiledPreferenceBuilder,
    feature_cache: Mapping[str, torch.Tensor],
    extra: Mapping[str, Any] | None = None,
) -> PrefBatch:
    """Create or reuse PrefBatch keyed by (g_id, batch_id)."""

    key: PrefKey = (str(g_id), int(batch_id))
    cached = caches.get_pref(key)
    if cached is not None:
        return cached
    pref = builder.build_fn(feature_cache, dict(extra or {}))
    caches.set_pref(key, pref)
    return pref


def _safe_ess(weights: torch.Tensor, *, eps: float = 1e-12) -> float:
    w = weights.detach()
    if w.numel() == 0:
        return 0.0
    w = torch.clamp(w, min=0.0)
    if not torch.isfinite(w).all().item():
        w = w[torch.isfinite(w)]
    if w.numel() == 0:
        return 0.0
    s1 = float(w.sum().item())
    s2 = float((w * w).sum().item())
    if s2 <= eps:
        return 0.0
    ess = (s1 * s1) / s2
    # Normalize to [0,1] by dividing by N.
    return float(ess / float(max(int(w.numel()), 1)))


def proxy_metrics_for_pair_on_batch(
    *,
    g: CompiledPreferenceBuilder,
    f: CompiledFreeLoss,
    feature_cache: Mapping[str, torch.Tensor],
    pref_batch: PrefBatch,
    joint_gate_kwargs: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute proxy metrics for one (g,f) on a single cached rollout batch.

    Outputs (JSON-serializable):
        {
          "loss": float,
          "loss_swap": float|None,
          "effective_grad_ratio": float,
          "grad_w_pass_rate": float,
          "grad_l_pass_rate": float,
          "swap_ok": bool|None,
          "ess_ratio": float,
          "pair_count": int,
        }
    """

    joint_gate_kwargs = dict(joint_gate_kwargs or {})
    gate: JointPreferenceGateResult = run_joint_preference_gates(
        f,
        pref_batch=pref_batch,
        feature_cache=feature_cache,
        **joint_gate_kwargs,
    )
    observed = {}
    if isinstance(gate.trace, dict):
        observed = dict(gate.trace.get("observed") or {})

    # ESS proxy: importance weights from pairwise log-prob gaps.
    ess_ratio = 0.0
    try:
        batch = pref_batch.to_pairwise_loss_batch(feature_cache)
        lpw = batch.get("log_prob_w")
        lpl = batch.get("log_prob_l")
        if isinstance(lpw, torch.Tensor) and isinstance(lpl, torch.Tensor):
            w = torch.exp(torch.clamp((lpw - lpl).detach(), min=-20.0, max=20.0))
            ess_ratio = _safe_ess(w)
    except Exception:  # noqa: BLE001
        ess_ratio = 0.0

    return {
        "loss": float(observed.get("loss", float("inf"))),
        "loss_swap": observed.get("loss_swap"),
        "effective_grad_ratio": float(observed.get("effective_grad_ratio", 0.0)),
        "grad_w_pass_rate": float(observed.get("grad_w_pass_rate", 0.0)),
        "grad_l_pass_rate": float(observed.get("grad_l_pass_rate", 0.0)),
        "swap_ok": observed.get("swap_ok"),
        "ess_ratio": float(ess_ratio),
        "pair_count": int(pref_batch.num_examples()),
        "joint_ok": bool(gate.ok),
        "joint_reason": str(gate.reason),
        "joint_trace": gate.trace,
    }


def aggregate_proxy_metrics(
    batch_metrics: Sequence[Mapping[str, Any]],
    *,
    proxy_weights: Mapping[str, float],
) -> Tuple[float, Dict[str, Any]]:
    """Aggregate per-batch proxy metrics into one proxy_score (lower is better)."""

    if not batch_metrics:
        return float("inf"), {"reason": "no_batches"}

    def _mean(key: str, default: float) -> float:
        vals: list[float] = []
        for m in batch_metrics:
            v = m.get(key)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        return float(sum(vals) / len(vals)) if vals else float(default)

    loss = _mean("loss", float("inf"))
    eff = _mean("effective_grad_ratio", 0.0)
    ess = _mean("ess_ratio", 0.0)
    w_eff = float(proxy_weights.get("effective_grad_ratio", 1.0))
    w_ess = float(proxy_weights.get("ess_ratio", 0.1))

    # Penalize low effective gradient / low ESS, keep primary term as loss.
    proxy_score = float(loss + w_eff * (1.0 - eff) + w_ess * (1.0 - ess))
    agg = {
        "proxy_loss_mean": float(loss),
        "proxy_effective_grad_ratio_mean": float(eff),
        "proxy_ess_ratio_mean": float(ess),
        "proxy_weights": {str(k): float(v) for k, v in dict(proxy_weights).items()},
    }
    return proxy_score, agg


def micro_unroll_score_for_pair(
    *,
    g: CompiledPreferenceBuilder,
    f: CompiledFreeLoss,
    rollout_feature_caches: Sequence[Mapping[str, torch.Tensor]],
    steps: int,
    lr: float,
    alpha: float,
    weight_decay: float = 0.0,
    reuse_pref_batch_when_safe: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Offline micro-unroll on cached rollouts (no new rollouts).

    We treat cached log_prob tensors as optimizable variables and take a few
    gradient steps to estimate "trainability" / optimization dynamics without
    running policy rollouts. Lower is better.
    """

    steps_i = max(int(steps), 0)
    if steps_i <= 0:
        return float("inf"), {"reason": "steps<=0"}
    if not rollout_feature_caches:
        return float("inf"), {"reason": "no_rollout_feature_caches"}

    # We optimize one log_prob tensor per cached rollout batch.
    objectives: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    device: torch.device | None = None
    for fc in rollout_feature_caches:
        obj = fc.get("objective")
        lp = fc.get("log_prob")
        if not isinstance(obj, torch.Tensor) or not isinstance(lp, torch.Tensor):
            return float("inf"), {"reason": "missing_objective_or_log_prob"}
        if obj.shape != lp.shape:
            return float("inf"), {"reason": f"shape_mismatch objective={tuple(obj.shape)} log_prob={tuple(lp.shape)}"}
        if device is None:
            device = obj.device
        objectives.append(obj.detach())
        log_probs.append(lp.detach().clone().requires_grad_(True))

    optimizer = torch.optim.Adam(log_probs, lr=float(lr), weight_decay=float(weight_decay))

    mode = str(getattr(f.ir.implementation_hint, "mode", "pairwise") or "pairwise").strip().lower()
    expects = [str(x) for x in (getattr(f.ir.implementation_hint, "expects", None) or [])]
    builder_expects = [str(x).strip().lower() for x in (getattr(g.ir.implementation_hint, "expects", None) or [])]
    reuse_pref_templates = bool(reuse_pref_batch_when_safe) and mode != "setwise" and ("log_prob" not in builder_expects)

    def _detach_pref(pref: PrefBatch) -> PrefBatch:
        pair_idx = pref.pair_idx
        if isinstance(pair_idx, tuple) and len(pair_idx) == 3:
            pair_idx = tuple(t.detach() for t in pair_idx)  # type: ignore[assignment]
        weight = pref.weight.detach() if isinstance(pref.weight, torch.Tensor) else pref.weight
        return PrefBatch(
            mode=str(pref.mode),
            pair_idx=pair_idx,
            list_idx=pref.list_idx.detach() if isinstance(pref.list_idx, torch.Tensor) else pref.list_idx,
            weight=weight,
            meta=dict(pref.meta or {}),
        )

    pref_templates: list[PrefBatch] | None = None
    if reuse_pref_templates:
        pref_templates = []
        with torch.no_grad():
            for obj, lp in zip(objectives, log_probs, strict=True):
                fc0 = extract_feature_cache(obj, lp)
                pref_templates.append(_detach_pref(g.build_fn(fc0, {"stage": "micro_unroll_pref_template"})))

    def _forward_mean_loss() -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for i, (obj, lp) in enumerate(zip(objectives, log_probs, strict=True)):
            fc = extract_feature_cache(obj, lp)
            if mode == "setwise":
                loss_t = f.loss_fn(batch={}, model_output=fc, extra={"alpha": float(alpha)})
            else:
                if pref_templates is not None:
                    pref = pref_templates[int(i)]
                else:
                    pref = g.build_fn(fc, {"stage": "micro_unroll"})
                batch = pref.to_pairwise_loss_batch(fc)
                if expects:
                    batch = {k: batch[k] for k in expects if k in batch}
                loss_t = f.loss_fn(batch=batch, model_output={}, extra={"alpha": float(alpha)})
            if not isinstance(loss_t, torch.Tensor):
                raise TypeError(f"loss_fn returned non-tensor: {type(loss_t)}")
            if loss_t.numel() != 1:
                raise ValueError(f"loss_fn returned non-scalar tensor: shape={tuple(loss_t.shape)}")
            losses.append(loss_t)
        return torch.stack(losses).mean()

    # Initial loss (before any update).
    with torch.no_grad():
        init_loss_t = _forward_mean_loss()
        init_loss = float(init_loss_t.item())

    last_loss = init_loss
    for _ in range(int(steps_i)):
        optimizer.zero_grad(set_to_none=True)
        loss_t = _forward_mean_loss()
        if not torch.isfinite(loss_t).all().item():
            return float("inf"), {"reason": "loss_not_finite"}
        loss_t.backward()
        optimizer.step()
        last_loss = float(loss_t.detach().item())

    # Final loss (after updates).
    with torch.no_grad():
        final_loss_t = _forward_mean_loss()
        final_loss = float(final_loss_t.item())

    metrics = {
        "micro_unroll_steps": int(steps_i),
        "micro_unroll_lr": float(lr),
        "micro_unroll_alpha": float(alpha),
        "micro_unroll_init_loss": float(init_loss),
        "micro_unroll_final_loss": float(final_loss),
        "micro_unroll_delta_loss": float(final_loss - init_loss),
        "micro_unroll_reuse_pref_batch": bool(pref_templates is not None),
        "mode": mode,
        "batches": int(len(rollout_feature_caches)),
        "device": str(device) if device is not None else None,
    }
    return float(final_loss), metrics


def load_pair_cache_from_pairs_jsonl(
    *,
    caches: PrefLossEvalCaches,
    pairs_jsonl_path: str,
    eval_sig: str,
) -> int:
    """Populate pair_cache from an existing pairs.jsonl (resume support).

    Returns:
        number of cached entries loaded.
    """

    if not os.path.isfile(pairs_jsonl_path):
        return 0
    loaded = 0
    with open(pairs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(rec, dict):
                continue
            if str(rec.get("eval_budget_signature", "")) != str(eval_sig):
                continue
            g_id = rec.get("g_id")
            f_id = rec.get("f_id")
            if not g_id or not f_id:
                continue
            key: PairKey = (str(g_id), str(f_id), str(eval_sig))
            # Merge using the same logic as `set_pair`, so multi-seed records keep their aggregation.
            caches.set_pair(key, rec)
            loaded += 1
    return loaded
