from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PTP_ROOT = REPO_ROOT / "PTP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PTP_ROOT) not in sys.path:
    sys.path.insert(0, str(PTP_ROOT))

_mplconfigdir = REPO_ROOT / ".cache" / "matplotlib"
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

from fitness.free_loss_fidelity import PrefBatch  # noqa: E402
from scripts.final_gradient_behavior_analysis import build_problem_specs, rollout_feature_caches  # noqa: E402
from scripts.plot_scale_generalization_loss_weighting import (  # noqa: E402
    LOSS_ONLY,
    WEIGHTING,
    PROBLEM_LABELS,
    _ensure_jssp_data,
    _load_pair,
    _method_loss,
    _replace_objective,
    _state_log_prob,
    _target_spec,
)


METHODS = ["Loss-only", "Loss+Weighting"]
COLORS = {"Loss-only": "#0072B2", "Loss+Weighting": "#D55E00"}
SCALE_PLAN = {
    "tsp100": ("TSP100->50", 1.0, 0.5),
    "cvrp100": ("CVRP100->50", 1.0, 0.5),
    "ffsp100": ("FFSP100->50", 1.0, 0.5),
    "jssp10x10": ("JSSP10x10->15x15", 1.0, 1.5),
}
FEATURES = [
    "objective_gap",
    "relative_gap",
    "rank_diff",
    "winner_percentile",
    "loser_percentile",
    "logp_diff",
    "normalized_advantage_gap",
    "seq_len_mean",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy().reshape(-1)


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    vals = np.sort(np.unique(np.concatenate([a, b])))
    ca = np.searchsorted(np.sort(a), vals, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), vals, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _wasserstein_1d(a: np.ndarray, b: np.ndarray, max_points: int = 2048) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, min(max_points, max(len(a), len(b))))
    aq = np.quantile(a, q)
    bq = np.quantile(b, q)
    return float(np.mean(np.abs(aq - bq)))


def _gini(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    arr = np.sort(arr)
    n = len(arr)
    return float((2.0 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - (n + 1.0) / n)


def _entropy_ratio(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    p = arr / arr.sum()
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / math.log(max(len(p), 2)))


def _top_mass(x: np.ndarray, frac: float = 0.10) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    k = max(int(math.ceil(len(arr) * frac)), 1)
    return float(np.sort(arr)[-k:].sum() / arr.sum())


def _eff_ratio(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    return float((arr.sum() ** 2) / (np.square(arr).sum() * len(arr)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    n = int(mask.sum())
    if n <= 1:
        return out
    order = np.argsort(arr[mask], kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / float(n - 1)
    out[mask] = ranks
    return out


def _informativeness_alignment(features: Mapping[str, np.ndarray], weights: np.ndarray) -> dict[str, float]:
    weight = np.asarray(weights, dtype=np.float64)
    rel_gap = np.asarray(features["relative_gap"], dtype=np.float64)
    margin = np.asarray(features["logp_diff"], dtype=np.float64)
    n = min(len(weight), len(rel_gap), len(margin))
    if n <= 2:
        return {
            "weight_informativeness_corr": float("nan"),
            "top10_weight_informative_coverage": float("nan"),
            "informative_base_rate": float("nan"),
            "top10_weight_informative_enrichment": float("nan"),
        }
    weight = weight[:n]
    gap_q = _rank01(rel_gap[:n])
    hard_q = 1.0 - _rank01(margin[:n])
    info = gap_q * hard_q
    finite = np.isfinite(weight) & np.isfinite(info)
    if int(finite.sum()) <= 2:
        return {
            "weight_informativeness_corr": float("nan"),
            "top10_weight_informative_coverage": float("nan"),
            "informative_base_rate": float("nan"),
            "top10_weight_informative_enrichment": float("nan"),
        }
    weight_f = weight[finite]
    info_f = info[finite]
    informative = info_f >= np.nanquantile(info_f, 0.80)
    k = max(int(math.ceil(0.10 * len(weight_f))), 1)
    top_idx = np.argsort(weight_f)[-k:]
    coverage = float(np.mean(informative[top_idx])) if k > 0 else float("nan")
    base_rate = float(np.mean(informative))
    return {
        "weight_informativeness_corr": _corr(weight_f, info_f),
        "top10_weight_informative_coverage": coverage,
        "informative_base_rate": base_rate,
        "top10_weight_informative_enrichment": coverage / max(base_rate, 1e-12),
    }


def _signal_alignment(weights: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    weight = np.asarray(weights, dtype=np.float64)
    score = np.asarray(signal, dtype=np.float64)
    n = min(len(weight), len(score))
    if n <= 2:
        return {
            "weight_loss_signal_corr": float("nan"),
            "top10_weight_loss_signal_coverage": float("nan"),
            "loss_signal_base_rate": float("nan"),
            "top10_weight_loss_signal_enrichment": float("nan"),
        }
    weight = weight[:n]
    score = score[:n]
    finite = np.isfinite(weight) & np.isfinite(score)
    if int(finite.sum()) <= 2:
        return {
            "weight_loss_signal_corr": float("nan"),
            "top10_weight_loss_signal_coverage": float("nan"),
            "loss_signal_base_rate": float("nan"),
            "top10_weight_loss_signal_enrichment": float("nan"),
        }
    weight_f = weight[finite]
    score_f = score[finite]
    informative = score_f >= np.nanquantile(score_f, 0.80)
    k = max(int(math.ceil(0.10 * len(weight_f))), 1)
    top_idx = np.argsort(weight_f)[-k:]
    coverage = float(np.mean(informative[top_idx])) if k > 0 else float("nan")
    base_rate = float(np.mean(informative))
    return {
        "weight_loss_signal_corr": _corr(weight_f, score_f),
        "top10_weight_loss_signal_coverage": coverage,
        "loss_signal_base_rate": base_rate,
        "top10_weight_loss_signal_enrichment": coverage / max(base_rate, 1e-12),
    }


def _subsample(idx: torch.Tensor, max_pairs: int, seed: int) -> torch.Tensor:
    n = int(idx.numel())
    if n <= max_pairs:
        return idx
    gen = torch.Generator(device=idx.device)
    gen.manual_seed(int(seed))
    chosen = torch.randperm(n, generator=gen, device=idx.device)[:max_pairs]
    return idx[chosen]


def _all_pair_features(fc: Mapping[str, torch.Tensor], log_prob: torch.Tensor, *, max_pairs: int, seed: int) -> dict[str, np.ndarray]:
    objective = fc["objective"].detach()
    batch, k = objective.shape
    mask = objective[:, :, None] < objective[:, None, :]
    b, w, l = mask.nonzero(as_tuple=True)
    pick = _subsample(torch.arange(b.numel(), device=objective.device), max_pairs, seed)
    b = b[pick]
    w = w[pick]
    l = l[pick]

    sorted_idx = objective.argsort(dim=1, descending=False)
    ranks = torch.empty_like(sorted_idx)
    rank_values = torch.arange(k, device=objective.device)[None, :].expand(batch, k)
    ranks.scatter_(1, sorted_idx, rank_values)
    obj_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
    obj_std = objective.std(dim=1).clamp_min(1e-12)
    seq_len = fc.get("seq_len")
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.full_like(objective, float(k))
    gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
    rank_diff = (ranks[b, l] - ranks[b, w]).float()
    return {
        "objective_gap": _as_np(gap),
        "relative_gap": _as_np(gap / obj_range[b]),
        "rank_diff": _as_np(rank_diff / max(k - 1, 1)),
        "winner_percentile": _as_np(ranks[b, w].float() / max(k - 1, 1)),
        "loser_percentile": _as_np(ranks[b, l].float() / max(k - 1, 1)),
        "logp_diff": _as_np(log_prob[b, w] - log_prob[b, l]),
        "normalized_advantage_gap": _as_np(gap / obj_std[b]),
        "seq_len_mean": _as_np(0.5 * (seq_len[b, w].float() + seq_len[b, l].float())),
    }


def _pref_pair_features(
    pref: PrefBatch,
    fc: Mapping[str, torch.Tensor],
    log_prob: torch.Tensor,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if pref.pair_idx is None or pref.num_examples() <= 0:
        raise ValueError("Expected non-empty pairwise preference batch")
    objective = fc["objective"].detach()
    batch, k = objective.shape
    b, w, l = pref.pair_idx
    pick = _subsample(torch.arange(b.numel(), device=objective.device), max_pairs, seed)
    b = b[pick]
    w = w[pick]
    l = l[pick]
    if isinstance(pref.weight, torch.Tensor):
        weight = pref.weight.detach().float().reshape(-1)[pick]
    else:
        weight = torch.ones_like(b, dtype=objective.dtype)

    sorted_idx = objective.argsort(dim=1, descending=False)
    ranks = torch.empty_like(sorted_idx)
    rank_values = torch.arange(k, device=objective.device)[None, :].expand(batch, k)
    ranks.scatter_(1, sorted_idx, rank_values)
    obj_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
    obj_std = objective.std(dim=1).clamp_min(1e-12)
    seq_len = fc.get("seq_len")
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.full_like(objective, float(k))
    gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
    rank_diff = (ranks[b, l] - ranks[b, w]).float()
    features = {
        "objective_gap": _as_np(gap),
        "relative_gap": _as_np(gap / obj_range[b]),
        "rank_diff": _as_np(rank_diff / max(k - 1, 1)),
        "winner_percentile": _as_np(ranks[b, w].float() / max(k - 1, 1)),
        "loser_percentile": _as_np(ranks[b, l].float() / max(k - 1, 1)),
        "logp_diff": _as_np(log_prob[b, w] - log_prob[b, l]),
        "normalized_advantage_gap": _as_np(gap / obj_std[b]),
        "seq_len_mean": _as_np(0.5 * (seq_len[b, w].float() + seq_len[b, l].float())),
    }
    return features, _as_np(weight), (b, w, l)


def _build_pref(method: str, fc: Mapping[str, torch.Tensor], pairs: Mapping[str, Any]) -> PrefBatch:
    pref = pairs[method][0](fc)
    if method == "Loss-only" and not isinstance(pref.weight, torch.Tensor):
        b, _, _ = pref.pair_idx
        pref = PrefBatch(mode=pref.mode, pair_idx=pref.pair_idx, list_idx=pref.list_idx, weight=torch.ones_like(b, dtype=fc["objective"].dtype), meta=dict(pref.meta or {}))
    return pref


def _coefficient_vector(method: str, fc: Mapping[str, torch.Tensor], pref: PrefBatch, pairs: Mapping[str, Any]) -> np.ndarray:
    live_fc = _replace_objective(fc, fc["objective"].detach(), fc["log_prob"].detach())
    batch = pref.to_pairwise_loss_batch(live_fc)
    live_batch: dict[str, Any] = {}
    for key, value in batch.items():
        live_batch[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
    for key in ("log_prob_w", "log_prob_l"):
        live_batch[key].requires_grad_(True)
    loss = pairs[method][1](live_batch)
    loss.backward()
    gw = live_batch["log_prob_w"].grad
    gl = live_batch["log_prob_l"].grad
    if gw is None or gl is None:
        return np.zeros((pref.num_examples(),), dtype=np.float64)
    return _as_np(gw.abs() + gl.abs())


def _one_step_consistency(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
    log_prob: torch.Tensor,
    step_size: float,
) -> tuple[float, dict[str, float]]:
    lp = log_prob.detach().clone().requires_grad_(True)
    loss = _method_loss(method, problem, {**dict(fc), "log_prob": lp}, lp, pairs)
    loss.backward()
    update = -lp.grad.detach()
    update = update - update.mean(dim=1, keepdim=True)
    update = update / update.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    after = lp.detach() + float(step_size) * update
    objective = fc["objective"].detach()
    mask = objective[:, :, None] < objective[:, None, :]
    b, w, l = mask.nonzero(as_tuple=True)
    gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
    obj_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
    rel_gap = gap / obj_range[b]
    good = (after[b, w] > after[b, l]).float()
    by_bin = {
        "small": float(good[rel_gap <= 0.15].mean().item()) if (rel_gap <= 0.15).any() else float("nan"),
        "medium": float(good[(rel_gap > 0.15) & (rel_gap <= 0.50)].mean().item()) if ((rel_gap > 0.15) & (rel_gap <= 0.50)).any() else float("nan"),
        "large": float(good[rel_gap > 0.50].mean().item()) if (rel_gap > 0.50).any() else float("nan"),
    }
    return float(good.mean().item()), by_bin


def collect(
    *,
    problems: list[str],
    batches: int,
    seed: int,
    device: str,
    state: str,
    sharpness: float,
    max_pairs: int,
    step_size: float,
) -> dict[str, list[dict[str, Any]]]:
    if "jssp10x10" in problems:
        _ensure_jssp_data(REPO_ROOT)
    specs = build_problem_specs(device, batches)
    pairs = {
        problem: {
            "Loss-only": _load_pair(LOSS_ONLY[problem]),
            "Loss+Weighting": _load_pair(WEIGHTING[problem]),
        }
        for problem in problems
    }
    feature_samples: dict[tuple[str, str, str], list[np.ndarray]] = {}
    weight_samples: dict[tuple[str, str, str], list[np.ndarray]] = {}
    feature_shift_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    for problem in problems:
        transfer_label, source_scale, transfer_scale = SCALE_PLAN[problem]
        for scale_name, scale in [("search", source_scale), ("transfer", transfer_scale)]:
            target_spec = _target_spec(specs[problem], problem, scale)
            target_size = int(target_spec.hf.train_problem_size)
            print(f"[replay] {problem} {scale_name} target_size={target_size}", flush=True)
            caches = rollout_feature_caches(target_spec, seed=seed + int(round(scale * 1000)), device=torch.device(device))
            for cache_id, raw_fc in enumerate(caches):
                log_prob = _state_log_prob(raw_fc, state, sharpness).detach()
                fc = {**dict(raw_fc), "log_prob": log_prob}
                all_features = _all_pair_features(fc, log_prob, max_pairs=max_pairs, seed=seed + cache_id)
                for feat, values in all_features.items():
                    feature_samples.setdefault((problem, scale_name, feat), []).append(values)

                for method in METHODS:
                    pref = _build_pref(method, fc, pairs[problem])
                    pref_features, weights, sampled_pair_idx = _pref_pair_features(pref, fc, log_prob, max_pairs=max_pairs, seed=seed + 17 * cache_id)
                    weight_samples.setdefault((problem, scale_name, method), []).append(weights)
                    alignment = _informativeness_alignment(pref_features, weights)
                    signal_alignment = {
                        "weight_loss_signal_corr": float("nan"),
                        "top10_weight_loss_signal_coverage": float("nan"),
                        "loss_signal_base_rate": float("nan"),
                        "top10_weight_loss_signal_enrichment": float("nan"),
                    }
                    if method == "Loss+Weighting" and pref.pair_idx is not None:
                        b, _, _ = sampled_pair_idx
                        ref_pref = PrefBatch(
                            mode=pref.mode,
                            pair_idx=sampled_pair_idx,
                            list_idx=None,
                            weight=torch.ones_like(b, dtype=fc["objective"].dtype),
                            meta=dict(pref.meta or {}),
                        )
                        ref_signal = _coefficient_vector("Loss-only", fc, ref_pref, pairs[problem])
                        signal_alignment = _signal_alignment(weights, ref_signal)
                    alignment_rows.append(
                        {
                            "problem": problem,
                            "transfer": transfer_label,
                            "scale_name": scale_name,
                            "target_size": target_size,
                            "method": method,
                            "cache_id": cache_id,
                            **alignment,
                            **signal_alignment,
                        }
                    )
                    coeff = _coefficient_vector(method, fc, pref, pairs[problem])
                    if coeff.size > max_pairs:
                        rng = np.random.default_rng(seed + 999 + cache_id)
                        coeff = coeff[rng.choice(coeff.size, size=max_pairs, replace=False)]

                    consistency, by_gap = _one_step_consistency(
                        method=method,
                        problem=problem,
                        fc=fc,
                        pairs=pairs[problem],
                        log_prob=log_prob,
                        step_size=step_size,
                    )
                    for gap_bin, val in {"all": consistency, **by_gap}.items():
                        consistency_rows.append(
                            {
                                "problem": problem,
                                "transfer": transfer_label,
                                "scale_name": scale_name,
                                "target_size": target_size,
                                "method": method,
                                "cache_id": cache_id,
                                "gap_bin": gap_bin,
                                "consistency": val,
                            }
                        )

                    margin = pref_features["logp_diff"]
                    confidence_left = 1.0 / (1.0 + np.exp(np.clip(margin, -30, 30)))
                    concentration_rows.append(
                        {
                            "problem": problem,
                            "transfer": transfer_label,
                            "scale_name": scale_name,
                            "target_size": target_size,
                            "method": method,
                            "cache_id": cache_id,
                            "effective_pair_ratio": _eff_ratio(weights),
                            "weight_entropy": _entropy_ratio(weights),
                            "top10_weight_mass": _top_mass(weights),
                            "max_over_mean_weight": float(np.nanmax(weights) / max(np.nanmean(weights), 1e-12)),
                            "weight_gini": _gini(weights),
                            "weight_cv": float(np.nanstd(weights) / max(np.nanmean(weights), 1e-12)),
                        }
                    )
                    coeff_rows.append(
                        {
                            "problem": problem,
                            "transfer": transfer_label,
                            "scale_name": scale_name,
                            "target_size": target_size,
                            "method": method,
                            "cache_id": cache_id,
                            "coefficient_eff_ratio": _eff_ratio(coeff),
                            "coefficient_entropy": _entropy_ratio(coeff),
                            "top10_coefficient_mass": _top_mass(coeff),
                            "saturated_pair_ratio": float(np.mean(confidence_left < 0.05)),
                            "over_amplified_pair_ratio": float(np.mean(coeff > (np.nanmean(coeff) + 3.0 * np.nanstd(coeff)))) if coeff.size else float("nan"),
                            "coefficient_gap_corr": _corr(coeff[: len(pref_features["relative_gap"])], pref_features["relative_gap"][: len(coeff)]),
                            "coefficient_rank_corr": _corr(coeff[: len(pref_features["rank_diff"])], pref_features["rank_diff"][: len(coeff)]),
                        }
                    )

    for problem in problems:
        transfer_label, _, _ = SCALE_PLAN[problem]
        for feat in FEATURES:
            src = np.concatenate(feature_samples.get((problem, "search", feat), [np.asarray([], dtype=float)]))
            tr = np.concatenate(feature_samples.get((problem, "transfer", feat), [np.asarray([], dtype=float)]))
            feature_shift_rows.append(
                {
                    "problem": problem,
                    "transfer": transfer_label,
                    "feature": feat,
                    "ks": _ks_stat(src, tr),
                    "wasserstein": _wasserstein_1d(src, tr),
                    "search_mean": float(np.nanmean(src)) if src.size else float("nan"),
                    "transfer_mean": float(np.nanmean(tr)) if tr.size else float("nan"),
                }
            )
        for method in METHODS:
            src_w = np.concatenate(weight_samples.get((problem, "search", method), [np.asarray([], dtype=float)]))
            tr_w = np.concatenate(weight_samples.get((problem, "transfer", method), [np.asarray([], dtype=float)]))
            feature_shift_rows.append(
                {
                    "problem": problem,
                    "transfer": transfer_label,
                    "feature": f"{method}:weight",
                    "ks": _ks_stat(src_w, tr_w),
                    "wasserstein": _wasserstein_1d(src_w, tr_w),
                    "search_mean": float(np.nanmean(src_w)) if src_w.size else float("nan"),
                    "transfer_mean": float(np.nanmean(tr_w)) if tr_w.size else float("nan"),
                }
            )

    return {
        "feature_shift": feature_shift_rows,
        "concentration": concentration_rows,
        "consistency": consistency_rows,
        "coefficients": coeff_rows,
        "alignment": alignment_rows,
    }


def _mean(rows: list[dict[str, Any]], metric: str, **conds: Any) -> float:
    vals = []
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            val = row.get(metric)
            if val is not None and np.isfinite(float(val)):
                vals.append(float(val))
    return float(np.mean(vals)) if vals else float("nan")


def plot(results: dict[str, list[dict[str, Any]]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    problems = list(SCALE_PLAN)
    labels = [SCALE_PLAN[p][0] for p in problems]
    feature_shift = results["feature_shift"]
    concentration = results["concentration"]
    consistency = results["consistency"]
    coeffs = results["coefficients"]
    alignment = results.get("alignment", [])

    # Figure B: pair/weight distribution shift.
    rows = []
    for p in problems:
        feat_vals = [
            float(r["ks"])
            for r in feature_shift
            if r["problem"] == p and not str(r["feature"]).endswith(":weight") and np.isfinite(float(r["ks"]))
        ]
        weight_ks = _mean(feature_shift, "ks", problem=p, feature="Loss+Weighting:weight")
        tr_eff = _mean(concentration, "effective_pair_ratio", problem=p, scale_name="transfer", method="Loss+Weighting")
        rows.append([float(np.mean(feat_vals)), weight_ks, 1.0 - tr_eff])
    mat = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 3.2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(3), ["pair feature shift\nmean KS", "weight shift\nKS", "transfer pair\nconcentration"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5)
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle("Pair and weight statistics shift under scale transfer", fontweight="bold")
    fig.savefig(out_dir / "01_pair_weight_distribution_shift.png", bbox_inches="tight")
    fig.savefig(out_dir / "01_pair_weight_distribution_shift.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure C: one-step replay preference consistency by gap bin.
    gap_bins = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, len(problems), figsize=(3.25 * len(problems), 3.35), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    conds = [
        ("Loss-only", "search", "LO search"),
        ("Loss-only", "transfer", "LO transfer"),
        ("Loss+Weighting", "search", "LW search"),
        ("Loss+Weighting", "transfer", "LW transfer"),
    ]
    cond_colors = ["#7DB7D9", "#0072B2", "#F0A45D", "#D55E00"]
    width = 0.18
    for ax, p, label in zip(axes, problems, labels):
        x = np.arange(len(gap_bins))
        for ci, (method, scale_name, cond_label) in enumerate(conds):
            vals = [_mean(consistency, "consistency", problem=p, method=method, scale_name=scale_name, gap_bin=g) for g in gap_bins]
            ax.bar(x + (ci - 1.5) * width, vals, width=width, color=cond_colors[ci], label=cond_label)
        ax.set_title(label)
        ax.set_xticks(x, gap_bins)
        ax.set_xlabel("objective-gap bin")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("one-step preference consistency")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Replay consistency: does the loss update preserve fine-grained preference ordering?", fontweight="bold")
    fig.savefig(out_dir / "02_policy_preference_consistency.png", bbox_inches="tight")
    fig.savefig(out_dir / "02_policy_preference_consistency.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure D: pair allocation concentration at transfer scale.
    metrics = [
        ("effective_pair_ratio", concentration, "effective pair ratio ↑"),
        ("top10_weight_mass", concentration, "top-10% weight mass ↓"),
        ("weight_gini", concentration, "weight Gini ↓"),
        ("weight_cv", concentration, "weight CV ↓"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), constrained_layout=True)
    for ax, (metric, source_rows, title) in zip(axes.reshape(-1), metrics):
        x = np.arange(len(problems))
        width = 0.34
        for offset, method in [(-width / 2, "Loss-only"), (width / 2, "Loss+Weighting")]:
            vals = [_mean(source_rows, metric, problem=p, method=method, scale_name="transfer") for p in problems]
            ax.bar(x + offset, vals, width=width, color=COLORS[method], label=method, alpha=0.92)
            for xi, val in zip(x + offset, vals):
                ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes.reshape(-1)[0].legend(frameon=False, loc="best")
    fig.suptitle("Weight allocation concentration at the transfer scale", fontweight="bold")
    fig.savefig(out_dir / "03_effective_signal_concentration.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_effective_signal_concentration.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for p in problems:
        for method in METHODS:
            summary_rows.append(
                {
                    "Problem": PROBLEM_LABELS[p].replace("JSSP10x10", "JSSP"),
                    "Transfer": SCALE_PLAN[p][0],
                    "Method": method,
                    "Consistency ↑": _mean(consistency, "consistency", problem=p, method=method, scale_name="transfer", gap_bin="all"),
                    "Effective pair ratio ↑": _mean(concentration, "effective_pair_ratio", problem=p, method=method, scale_name="transfer"),
                    "Top-10% weight mass ↓": _mean(concentration, "top10_weight_mass", problem=p, method=method, scale_name="transfer"),
                    "Weight Gini ↓": _mean(concentration, "weight_gini", problem=p, method=method, scale_name="transfer"),
                }
            )
    _write_csv(out_dir / "04_replay_diagnosis_summary_table.csv", summary_rows)
    fig, ax = plt.subplots(figsize=(11.8, 3.8))
    ax.axis("off")
    display_rows = []
    for row in summary_rows:
        display_rows.append(
            [
                row["Problem"],
                row["Transfer"],
                row["Method"],
                f"{row['Consistency ↑']:.3f}",
                f"{row['Effective pair ratio ↑']:.3f}",
                f"{row['Top-10% weight mass ↓']:.3f}",
                f"{row['Weight Gini ↓']:.3f}",
            ]
        )
    columns = ["Problem", "Transfer", "Method", "Consistency ↑", "Eff. pair ratio ↑", "Top-10% weight ↓", "Weight Gini ↓"]
    table = ax.table(cellText=display_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d0c4")
        cell.set_linewidth(0.65)
        if r == 0:
            cell.set_facecolor("#ece7df")
            cell.set_text_props(weight="bold")
        elif c == 2 and "Weighting" in cell.get_text().get_text():
            cell.set_facecolor("#fff0df")
        elif c == 2:
            cell.set_facecolor("#e8f1fb")
    ax.set_title("Replay diagnosis summary at transfer scale", fontweight="bold", pad=12)
    fig.savefig(out_dir / "04_replay_diagnosis_summary_table.png", bbox_inches="tight")
    fig.savefig(out_dir / "04_replay_diagnosis_summary_table.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure E: whether learned weights still target informative pairs.
    align_rows = []
    for p in problems:
        search_corr = _mean(alignment, "weight_loss_signal_corr", problem=p, scale_name="search", method="Loss+Weighting")
        transfer_corr = _mean(alignment, "weight_loss_signal_corr", problem=p, scale_name="transfer", method="Loss+Weighting")
        search_cov = _mean(alignment, "top10_weight_loss_signal_coverage", problem=p, scale_name="search", method="Loss+Weighting")
        transfer_cov = _mean(alignment, "top10_weight_loss_signal_coverage", problem=p, scale_name="transfer", method="Loss+Weighting")
        search_enrich = _mean(alignment, "top10_weight_loss_signal_enrichment", problem=p, scale_name="search", method="Loss+Weighting")
        transfer_enrich = _mean(alignment, "top10_weight_loss_signal_enrichment", problem=p, scale_name="transfer", method="Loss+Weighting")
        align_rows.append([search_corr, transfer_corr, search_corr - transfer_corr, search_cov, transfer_cov, search_enrich, transfer_enrich])
    align_mat = np.asarray(align_rows, dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 3.3), constrained_layout=True)
    im = ax.imshow(align_mat, aspect="auto", cmap="RdYlBu_r")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(
        np.arange(7),
        [
            "search\ncorr(w, info)",
            "transfer\ncorr(w, info)",
            "alignment\ndrop",
            "search top-10%\ncoverage",
            "transfer top-10%\ncoverage",
            "search\nenrichment",
            "transfer\nenrichment",
        ],
    )
    for i in range(align_mat.shape[0]):
        for j in range(align_mat.shape[1]):
            ax.text(j, i, f"{align_mat[i, j]:.2f}", ha="center", va="center", fontsize=8.0)
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle("Does learned weighting still target the pairs emphasized by loss-only?", fontweight="bold")
    fig.savefig(out_dir / "05_weight_informativeness_alignment.png", bbox_inches="tight")
    fig.savefig(out_dir / "05_weight_informativeness_alignment.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Scale-transfer replay diagnosis\n\n"
            "This is a CPU replay/diagnosis analysis, not a training experiment.\n\n"
            "Transfer directions: TSP/CVRP/FFSP use 100->50 down-scale transfer; JSSP uses 10x10->15x15 larger-shape transfer.\n\n"
            "Main claim: loss-only learns a more transferable comparison rule, while loss+weighting learns scale-sensitive pair allocation.\n\n"
            "Outputs:\n"
            "- `01_pair_weight_distribution_shift.*`: pair feature shift, weight shift, and effective pair ratio drop.\n"
            "- `02_policy_preference_consistency.*`: one-step replay preference consistency by objective-gap bin.\n"
            "- `03_effective_signal_concentration.*`: effective pair/gradient-coefficient concentration at transfer scale.\n"
            "- `04_replay_diagnosis_summary_table.*`: compact transfer-scale summary.\n"
            "- `05_weight_informativeness_alignment.*`: whether learned weights remain aligned with high-gap, low-margin informative pairs.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--state", default="aligned", choices=["sampled", "aligned", "misaligned"])
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "scale_transfer_replay_diagnosis" / stamp))
    results = collect(
        problems=problems,
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        state=str(args.state),
        sharpness=float(args.sharpness),
        max_pairs=max(int(args.max_pairs), 1024),
        step_size=float(args.step_size),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "pair_feature_shift.csv", results["feature_shift"])
    _write_csv(out_dir / "weight_concentration.csv", results["concentration"])
    _write_csv(out_dir / "policy_consistency.csv", results["consistency"])
    _write_csv(out_dir / "coefficient_diagnosis.csv", results["coefficients"])
    _write_csv(out_dir / "weight_informativeness_alignment.csv", results["alignment"])
    plot(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
