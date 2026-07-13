from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


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
    _replace_objective,
    _state_log_prob,
    _target_spec,
)


PROBLEMS = ["tsp100", "cvrp100", "ffsp100", "jssp10x10"]
SOURCE_SCALE = {"tsp100": 1.0, "cvrp100": 1.0, "ffsp100": 1.0, "jssp10x10": 1.0}
TRANSFER_SCALE = {"tsp100": 0.5, "cvrp100": 0.5, "ffsp100": 0.5, "jssp10x10": 1.5}
TRANSFER_LABEL = {
    "tsp100": "TSP100->50",
    "cvrp100": "CVRP100->50",
    "ffsp100": "FFSP100->50",
    "jssp10x10": "JSSP10x10->15x15",
}

SIGNAL_METHODS = ["RL", "PO/BT", "BOPO-style", "SimPO-style", "Loss-only"]
CONSISTENCY_METHODS = SIGNAL_METHODS + ["Loss+Weighting"]
COLORS = {
    "RL": "#8C8C8C",
    "PO/BT": "#0072B2",
    "BOPO-style": "#D55E00",
    "SimPO-style": "#CC79A7",
    "Loss-only": "#009E73",
    "Loss+Weighting": "#E69F00",
}
MARKERS = {"RL": "o", "PO/BT": "s", "BOPO-style": "^", "SimPO-style": "D", "Loss-only": "P", "Loss+Weighting": "X"}


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


def _sigmoid_neg_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(x, -50.0, 50.0)))


def _top_mass(x: np.ndarray, frac: float = 0.10) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    k = max(int(math.ceil(float(frac) * len(arr))), 1)
    return float(np.sort(arr)[-k:].sum() / arr.sum())


def _entropy_ratio(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    p = arr / arr.sum()
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / math.log(max(len(p), 2)))


def _eff_ratio(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    return float((arr.sum() ** 2) / (np.square(arr).sum() * len(arr)))


def _subsample_indices(n: int, max_items: int, seed: int, device: torch.device) -> torch.Tensor:
    if n <= max_items:
        return torch.arange(n, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randperm(n, generator=gen, device=device)[:max_items]


def _with_uniform_weight(pref: PrefBatch, fc: Mapping[str, torch.Tensor]) -> PrefBatch:
    if pref.pair_idx is None:
        raise ValueError("Expected pairwise preference batch")
    if isinstance(pref.weight, torch.Tensor):
        return pref
    b, _, _ = pref.pair_idx
    return PrefBatch(
        mode=pref.mode,
        pair_idx=pref.pair_idx,
        list_idx=pref.list_idx,
        weight=torch.ones_like(b, dtype=fc["objective"].dtype),
        meta=dict(pref.meta or {}),
    )


def _subsample_pref(pref: PrefBatch, max_pairs: int, seed: int) -> PrefBatch:
    if pref.pair_idx is None:
        raise ValueError("Expected pairwise preference batch")
    b, w, l = pref.pair_idx
    pick = _subsample_indices(int(b.numel()), max_pairs, seed, b.device)
    weight = pref.weight[pick] if isinstance(pref.weight, torch.Tensor) else pref.weight
    return PrefBatch(
        mode=pref.mode,
        pair_idx=(b[pick], w[pick], l[pick]),
        list_idx=None,
        weight=weight,
        meta=dict(pref.meta or {}),
    )


def _pair_features(pref: PrefBatch, fc: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if pref.pair_idx is None:
        raise ValueError("Expected pairwise preference batch")
    objective = fc["objective"].detach()
    log_prob = fc["log_prob"].detach()
    b, w, l = pref.pair_idx
    batch_size, k = objective.shape
    sorted_idx = objective.argsort(dim=1, descending=False)
    ranks = torch.empty_like(sorted_idx)
    rank_values = torch.arange(k, device=objective.device)[None, :].expand(batch_size, k)
    ranks.scatter_(1, sorted_idx, rank_values)
    obj_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
    obj_std = objective.std(dim=1).clamp_min(1e-12)
    reward = -objective
    adv = reward - reward.mean(dim=1, keepdim=True)
    adv_z = adv / reward.std(dim=1, keepdim=True).clamp_min(1e-12)
    seq_len = fc.get("seq_len")
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.full_like(objective, float(k))
    gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
    rank_diff = (ranks[b, l] - ranks[b, w]).float()
    return {
        "b": b,
        "w": w,
        "l": l,
        "objective_gap": gap,
        "relative_gap": gap / obj_range[b],
        "normalized_gap": gap / obj_std[b],
        "rank_diff": rank_diff,
        "rank_diff_norm": rank_diff / max(k - 1, 1),
        "winner_rank_norm": ranks[b, w].float() / max(k - 1, 1),
        "loser_rank_norm": ranks[b, l].float() / max(k - 1, 1),
        "margin": log_prob[b, w] - log_prob[b, l],
        "seq_len_mean": 0.5 * (seq_len[b, w].float() + seq_len[b, l].float()),
        "adv_w": adv_z[b, w],
        "adv_l": adv_z[b, l],
    }


def _manual_coefficients(method: str, features: Mapping[str, torch.Tensor], *, alpha: float) -> np.ndarray:
    margin = _as_np(features["margin"])
    rel_gap = _as_np(features["relative_gap"])
    norm_gap = _as_np(features["normalized_gap"])
    seq_len = np.maximum(_as_np(features["seq_len_mean"]), 1.0)
    if method == "RL":
        adv_w = _as_np(features["adv_w"])
        adv_l = _as_np(features["adv_l"])
        return 0.5 * (np.abs(adv_w) + np.abs(adv_l)) / seq_len
    if method == "PO/BT":
        logit = float(alpha) * margin
        return float(alpha) * _sigmoid_neg_np(logit)
    if method == "BOPO-style":
        scale = np.clip(1.0 + norm_gap, 1.0, 5.0)
        logit = float(alpha) * scale * margin
        return float(alpha) * scale * _sigmoid_neg_np(logit)
    if method == "SimPO-style":
        beta = 2.0
        gamma = 0.10
        mean_margin = margin / seq_len
        logit = beta * (mean_margin - gamma)
        return (beta / seq_len) * _sigmoid_neg_np(logit)
    raise KeyError(method)


def _autograd_pair_coefficients(
    pref: PrefBatch,
    fc: Mapping[str, torch.Tensor],
    loss_fn: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
) -> np.ndarray:
    batch = pref.to_pairwise_loss_batch(fc)
    live_batch: dict[str, Any] = {}
    for key, value in batch.items():
        live_batch[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
    for key in ("log_prob_w", "log_prob_l"):
        live_batch[key].requires_grad_(True)
    loss = loss_fn(live_batch)
    loss.backward()
    gw = live_batch["log_prob_w"].grad
    gl = live_batch["log_prob_l"].grad
    if gw is None or gl is None:
        return np.zeros((pref.num_examples(),), dtype=np.float64)
    # Convert mean-reduced losses back to a local per-pair derivative scale.
    n = max(int(pref.num_examples()), 1)
    return _as_np(0.5 * float(n) * (gw.abs() + gl.abs()))


def _coefficients_for_method(
    method: str,
    problem: str,
    pref: PrefBatch,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]],
    *,
    alpha: float,
) -> np.ndarray:
    features = _pair_features(pref, fc)
    if method in {"RL", "PO/BT", "BOPO-style", "SimPO-style"}:
        return _manual_coefficients(method, features, alpha=alpha)
    if method == "Loss-only":
        return _autograd_pair_coefficients(pref, fc, pairs["Loss-only"][1])
    if method == "Loss+Weighting":
        weighted_pref = _with_uniform_weight(pairs["Loss+Weighting"][0](fc), fc)
        weighted_pref = _subsample_pref(weighted_pref, int(pref.num_examples()), seed=9357)
        return _autograd_pair_coefficients(weighted_pref, fc, pairs["Loss+Weighting"][1])
    raise KeyError(method)


def _net_signal_by_rank(pref: PrefBatch, fc: Mapping[str, torch.Tensor], coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pref.pair_idx is None:
        raise ValueError("Expected pairwise preference batch")
    objective = fc["objective"].detach()
    b, w, l = pref.pair_idx
    coeff_t = torch.as_tensor(coeff, device=objective.device, dtype=objective.dtype)
    net = torch.zeros_like(objective)
    net.index_put_((b, w), coeff_t, accumulate=True)
    net.index_put_((b, l), -coeff_t, accumulate=True)
    denom = net.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    net = net / denom
    sorted_idx = objective.argsort(dim=1, descending=False)
    ranked = net.gather(1, sorted_idx)
    k = objective.shape[1]
    rank_x = torch.arange(k, device=objective.device).float() / max(k - 1, 1)
    return _as_np(rank_x[None, :].expand_as(ranked)), _as_np(ranked)


def _one_step_consistency_from_update(objective: torch.Tensor, log_prob: torch.Tensor, update: torch.Tensor, *, step_size: float) -> dict[str, float]:
    upd = update - update.mean(dim=1, keepdim=True)
    upd = upd / upd.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    after = log_prob.detach() + float(step_size) * upd
    mask = objective[:, :, None] < objective[:, None, :]
    b, w, l = mask.nonzero(as_tuple=True)
    gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
    obj_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
    rel_gap = gap / obj_range[b]
    good = (after[b, w] > after[b, l]).float()
    bins = {
        "all": torch.ones_like(good, dtype=torch.bool),
        "small": rel_gap <= 0.15,
        "medium": (rel_gap > 0.15) & (rel_gap <= 0.50),
        "large": rel_gap > 0.50,
    }
    out: dict[str, float] = {}
    for name, m in bins.items():
        out[name] = float(good[m].mean().item()) if bool(m.any().item()) else float("nan")
    return out


def _update_for_method(
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]],
    *,
    alpha: float,
    max_pairs: int,
    seed: int,
) -> torch.Tensor:
    objective = fc["objective"].detach()
    if method == "RL":
        reward = -objective
        adv = reward - reward.mean(dim=1, keepdim=True)
        return adv / reward.std(dim=1, keepdim=True).clamp_min(1e-12)
    if method in {"PO/BT", "BOPO-style", "SimPO-style", "Loss-only"}:
        pref = _with_uniform_weight(pairs["Loss-only"][0](fc), fc)
        pref = _subsample_pref(pref, max_pairs, seed)
        coeff = _coefficients_for_method(method, problem, pref, fc, pairs, alpha=alpha)
    elif method == "Loss+Weighting":
        pref = _with_uniform_weight(pairs["Loss+Weighting"][0](fc), fc)
        pref = _subsample_pref(pref, max_pairs, seed)
        coeff = _autograd_pair_coefficients(pref, fc, pairs["Loss+Weighting"][1])
    else:
        raise KeyError(method)
    if pref.pair_idx is None:
        raise ValueError("Expected pairwise preference batch")
    b, w, l = pref.pair_idx
    coeff_t = torch.as_tensor(coeff, device=objective.device, dtype=objective.dtype)
    update = torch.zeros_like(objective)
    update.index_put_((b, w), coeff_t, accumulate=True)
    update.index_put_((b, l), -coeff_t, accumulate=True)
    return update


def _bin_means(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> list[float]:
    vals: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi if hi < bins[-1] else x <= hi)
        vals.append(float(np.nanmean(y[mask])) if np.any(mask) else float("nan"))
    return vals


def _normalize_coeff(coeff: np.ndarray) -> np.ndarray:
    arr = np.asarray(coeff, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    denom = float(np.nanmean(finite)) if finite.size else 1.0
    return arr / max(denom, 1e-12)


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
    pair_cache = {
        problem: {
            "Loss-only": _load_pair(LOSS_ONLY[problem]),
            "Loss+Weighting": _load_pair(WEIGHTING[problem]),
        }
        for problem in problems
    }
    signal_rows: list[dict[str, Any]] = []
    net_rows: list[dict[str, Any]] = []
    saturation_rows: list[dict[str, Any]] = []
    gap_allocation_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []

    for problem in problems:
        source_spec = _target_spec(specs[problem], problem, SOURCE_SCALE[problem])
        print(f"[signal] {problem} source-scale replay", flush=True)
        caches = rollout_feature_caches(source_spec, seed=seed + 11, device=torch.device(device))
        alpha = float(source_spec.hf.alpha)
        for cache_id, raw_fc in enumerate(caches):
            log_prob = _state_log_prob(raw_fc, state, sharpness).detach()
            fc = {**dict(raw_fc), "log_prob": log_prob}
            pref = _with_uniform_weight(pair_cache[problem]["Loss-only"][0](fc), fc)
            pref = _subsample_pref(pref, max_pairs, seed + 1009 * cache_id)
            features = _pair_features(pref, fc)
            rel_gap = _as_np(features["relative_gap"])
            rank_diff = _as_np(features["rank_diff_norm"])
            for method in SIGNAL_METHODS:
                coeff = _coefficients_for_method(method, problem, pref, fc, pair_cache[problem], alpha=alpha)
                coeff_norm = _normalize_coeff(coeff)
                for x_name, x_vals in [("relative_gap", rel_gap), ("rank_diff", rank_diff)]:
                    bins = np.linspace(0.0, 1.0, 11)
                    means = _bin_means(x_vals, coeff_norm, bins)
                    for bi, val in enumerate(means):
                        signal_rows.append(
                            {
                                "problem": problem,
                                "cache_id": cache_id,
                                "method": method,
                                "x_name": x_name,
                                "bin_lo": float(bins[bi]),
                                "bin_hi": float(bins[bi + 1]),
                                "bin_mid": float(0.5 * (bins[bi] + bins[bi + 1])),
                                "mean_normalized_coefficient": val,
                            }
                        )
                rank_x, net_y = _net_signal_by_rank(pref, fc, coeff)
                bins = np.linspace(0.0, 1.0, 11)
                means = _bin_means(rank_x, net_y, bins)
                for bi, val in enumerate(means):
                    net_rows.append(
                        {
                            "problem": problem,
                            "cache_id": cache_id,
                            "method": method,
                            "rank_bin_lo": float(bins[bi]),
                            "rank_bin_hi": float(bins[bi + 1]),
                            "rank_bin_mid": float(0.5 * (bins[bi] + bins[bi + 1])),
                            "net_signal": val,
                        }
                    )
                coeff_pos = np.asarray(coeff, dtype=np.float64)
                coeff_pos = coeff_pos[np.isfinite(coeff_pos)]
                rel_gap_f = np.asarray(rel_gap, dtype=np.float64)[: len(coeff)]
                coeff_all = np.asarray(coeff, dtype=np.float64)[: len(rel_gap_f)]
                finite_mask = np.isfinite(rel_gap_f) & np.isfinite(coeff_all) & (coeff_all >= 0.0)
                if np.any(finite_mask):
                    q1, q2 = np.nanquantile(rel_gap_f[finite_mask], [0.33, 0.67])
                    groups = {
                        "small": finite_mask & (rel_gap_f <= q1),
                        "medium": finite_mask & (rel_gap_f > q1) & (rel_gap_f <= q2),
                        "large": finite_mask & (rel_gap_f > q2),
                    }
                    total_mass = float(np.nansum(coeff_all[finite_mask]))
                    total_pairs = int(np.sum(finite_mask))
                    for gap_group, gm in groups.items():
                        pair_share = float(np.sum(gm) / max(total_pairs, 1))
                        mass_share = float(np.nansum(coeff_all[gm]) / max(total_mass, 1e-12))
                        gap_allocation_rows.append(
                            {
                                "problem": problem,
                                "cache_id": cache_id,
                                "method": method,
                                "gap_group": gap_group,
                                "pair_share": pair_share,
                                "coefficient_mass_share": mass_share,
                                "mass_over_pair_share": mass_share / max(pair_share, 1e-12),
                            }
                        )
                mean_c = float(np.nanmean(coeff_pos)) if coeff_pos.size else float("nan")
                std_c = float(np.nanstd(coeff_pos)) if coeff_pos.size else float("nan")
                saturation_rows.append(
                    {
                        "problem": problem,
                        "cache_id": cache_id,
                        "method": method,
                        "active_pair_ratio": float(np.mean(coeff_pos > 0.05 * max(mean_c, 1e-12))) if coeff_pos.size else float("nan"),
                        "saturated_pair_ratio": float(np.mean(coeff_pos < 0.05 * max(mean_c, 1e-12))) if coeff_pos.size else float("nan"),
                        "over_amplified_pair_ratio": float(np.mean(coeff_pos > mean_c + 3.0 * std_c)) if coeff_pos.size else float("nan"),
                        "top10_coefficient_mass": _top_mass(coeff_pos),
                        "coefficient_entropy": _entropy_ratio(coeff_pos),
                        "effective_coefficient_ratio": _eff_ratio(coeff_pos),
                    }
                )

            if cache_id == 0:
                shape_rows.extend(_loss_shape_rows(problem, pref, fc, pair_cache[problem], alpha=alpha, seed=seed))

        for scale_name, scale in [("search", SOURCE_SCALE[problem]), ("transfer", TRANSFER_SCALE[problem])]:
            target_spec = _target_spec(specs[problem], problem, scale)
            print(f"[consistency] {problem} {scale_name}", flush=True)
            caches_scale = rollout_feature_caches(target_spec, seed=seed + int(1000 * scale), device=torch.device(device))
            alpha_scale = float(target_spec.hf.alpha)
            for cache_id, raw_fc in enumerate(caches_scale):
                log_prob = _state_log_prob(raw_fc, state, sharpness).detach()
                fc = {**dict(raw_fc), "log_prob": log_prob}
                for method in CONSISTENCY_METHODS:
                    update = _update_for_method(
                        method,
                        problem,
                        fc,
                        pair_cache[problem],
                        alpha=alpha_scale,
                        max_pairs=max_pairs,
                        seed=seed + 2003 * cache_id,
                    )
                    vals = _one_step_consistency_from_update(fc["objective"].detach(), log_prob, update, step_size=step_size)
                    for gap_bin, val in vals.items():
                        consistency_rows.append(
                            {
                                "problem": problem,
                                "transfer": TRANSFER_LABEL[problem],
                                "scale_name": scale_name,
                                "method": method,
                                "cache_id": cache_id,
                                "gap_bin": gap_bin,
                                "consistency": val,
                            }
                        )

    return {
        "signal": signal_rows,
        "net_signal": net_rows,
        "saturation": saturation_rows,
        "gap_allocation": gap_allocation_rows,
        "consistency": consistency_rows,
        "shape": shape_rows,
    }


def _loss_shape_rows(
    problem: str,
    pref: PrefBatch,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]],
    *,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    features = _pair_features(pref, fc)
    rel_gap = _as_np(features["relative_gap"])
    groups = {
        "small": rel_gap <= np.nanquantile(rel_gap, 0.33),
        "medium": (rel_gap > np.nanquantile(rel_gap, 0.33)) & (rel_gap <= np.nanquantile(rel_gap, 0.67)),
        "large": rel_gap > np.nanquantile(rel_gap, 0.67),
    }
    base_batch = pref.to_pairwise_loss_batch(fc)
    margins = np.linspace(-8.0, 8.0, 81)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 427)
    for gap_group, mask in groups.items():
        idx_np = np.flatnonzero(mask)
        if idx_np.size == 0:
            continue
        if idx_np.size > 2048:
            idx_np = rng.choice(idx_np, size=2048, replace=False)
        idx = torch.as_tensor(idx_np, device=fc["objective"].device, dtype=torch.long)
        seq_len = np.maximum(_as_np(features["seq_len_mean"])[idx_np], 1.0)
        norm_gap = np.clip(1.0 + _as_np(features["normalized_gap"])[idx_np], 1.0, 5.0)
        for margin in margins:
            hand = {
                "PO/BT": -np.log(np.clip(1.0 / (1.0 + np.exp(-float(alpha) * margin)), 1e-12, 1.0)),
                "BOPO-style": float(np.nanmean(-np.log(np.clip(1.0 / (1.0 + np.exp(-float(alpha) * norm_gap * margin)), 1e-12, 1.0)))),
                "SimPO-style": float(np.nanmean(-np.log(np.clip(1.0 / (1.0 + np.exp(-2.0 * (margin / seq_len - 0.10))), 1e-12, 1.0)))),
            }
            hand_grad = {
                "PO/BT": float(alpha) / (1.0 + math.exp(np.clip(float(alpha) * margin, -50.0, 50.0))),
                "BOPO-style": float(np.nanmean(float(alpha) * norm_gap / (1.0 + np.exp(np.clip(float(alpha) * norm_gap * margin, -50.0, 50.0))))),
                "SimPO-style": float(np.nanmean((2.0 / seq_len) / (1.0 + np.exp(np.clip(2.0 * (margin / seq_len - 0.10), -50.0, 50.0))))),
            }
            for method, value in hand.items():
                rows.append(
                    {
                        "problem": problem,
                        "gap_group": gap_group,
                        "method": method,
                        "margin": float(margin),
                        "loss_value": float(value),
                        "gradient_magnitude": float(hand_grad[method]),
                    }
                )

        loss_values: list[float] = []
        for margin in margins:
            live_batch: dict[str, Any] = {}
            for key, value in base_batch.items():
                if isinstance(value, torch.Tensor):
                    live_batch[key] = value.detach().clone()[idx]
                else:
                    live_batch[key] = value
            center = 0.5 * (live_batch["log_prob_w"] + live_batch["log_prob_l"])
            m_t = torch.full_like(center, float(margin))
            live_batch["log_prob_w"] = center + 0.5 * m_t
            live_batch["log_prob_l"] = center - 0.5 * m_t
            if "seq_len_w" in live_batch and "seq_len_l" in live_batch:
                live_batch["log_prob_w_mean"] = live_batch["log_prob_w"] / live_batch["seq_len_w"].clamp_min(1.0)
                live_batch["log_prob_l_mean"] = live_batch["log_prob_l"] / live_batch["seq_len_l"].clamp_min(1.0)
                live_batch["log_prob_mean_gap"] = live_batch["log_prob_w_mean"] - live_batch["log_prob_l_mean"]
            loss = pairs["Loss-only"][1](live_batch)
            loss_values.append(float(loss.detach().item()))
        loss_arr = np.asarray(loss_values, dtype=np.float64)
        grad_arr = np.abs(np.gradient(loss_arr, margins))
        for margin, value, grad in zip(margins, loss_arr, grad_arr):
            rows.append(
                {
                    "problem": problem,
                    "gap_group": gap_group,
                    "method": "Loss-only",
                    "margin": float(margin),
                    "loss_value": float(value),
                    "gradient_magnitude": float(grad),
                }
            )
    return rows


def _mean(rows: list[dict[str, Any]], metric: str, **conds: Any) -> float:
    vals = []
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            val = row.get(metric)
            if val is not None and np.isfinite(float(val)):
                vals.append(float(val))
    return float(np.nanmean(vals)) if vals else float("nan")


def plot(results: dict[str, list[dict[str, Any]]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.7,
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    problems = [p for p in PROBLEMS if any(r["problem"] == p for r in results["signal"])]
    labels = [PROBLEM_LABELS[p].replace("JSSP10x10", "JSSP") for p in problems]

    fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 3.1), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, label in zip(axes, problems, labels):
        for method in SIGNAL_METHODS:
            xs = sorted({float(r["bin_mid"]) for r in results["signal"] if r["problem"] == problem and r["method"] == method and r["x_name"] == "relative_gap"})
            ys = [
                _mean(results["signal"], "mean_normalized_coefficient", problem=problem, method=method, x_name="relative_gap", bin_mid=x)
                for x in xs
            ]
            ax.plot(xs, ys, color=COLORS[method], marker=MARKERS[method], ms=3.0, lw=1.45, label=method)
        ax.set_title(label)
        ax.set_xlabel("relative objective gap")
        ax.axhline(1.0, color="#444444", lw=0.8, ls="--", alpha=0.5)
    axes[0].set_ylabel("mean-normalized coefficient")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Loss signal separation on the same preference pairs", fontweight="bold")
    fig.savefig(out_dir / "01_loss_signal_separation.png", bbox_inches="tight")
    fig.savefig(out_dir / "01_loss_signal_separation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 3.1), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, label in zip(axes, problems, labels):
        for method in SIGNAL_METHODS:
            xs = sorted({float(r["rank_bin_mid"]) for r in results["net_signal"] if r["problem"] == problem and r["method"] == method})
            ys = [_mean(results["net_signal"], "net_signal", problem=problem, method=method, rank_bin_mid=x) for x in xs]
            ax.plot(xs, ys, color=COLORS[method], marker=MARKERS[method], ms=3.0, lw=1.45, label=method)
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(label)
        ax.set_xlabel("solution rank percentile")
    axes[0].set_ylabel("net training signal")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Advantage separation: which ranks are encouraged or suppressed?", fontweight="bold")
    fig.savefig(out_dir / "02_advantage_separation_by_rank.png", bbox_inches="tight")
    fig.savefig(out_dir / "02_advantage_separation_by_rank.pdf", bbox_inches="tight")
    plt.close(fig)

    gap_bins = ["small", "medium", "large"]
    conds = [("search", "solid"), ("transfer", "dashed")]
    fig, axes = plt.subplots(1, len(problems), figsize=(3.3 * len(problems), 3.3), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, label in zip(axes, problems, labels):
        x = np.arange(len(gap_bins))
        width = 0.12
        offsets = np.linspace(-0.30, 0.30, len(CONSISTENCY_METHODS))
        for mi, method in enumerate(CONSISTENCY_METHODS):
            vals = [_mean(results["consistency"], "consistency", problem=problem, method=method, scale_name="transfer", gap_bin=g) for g in gap_bins]
            ax.bar(x + offsets[mi], vals, width=width, color=COLORS[method], label=method)
        ax.set_title(label)
        ax.set_xticks(x, gap_bins)
        ax.set_xlabel("gap bin at transfer scale")
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("one-step preference consistency")
    axes[-1].legend(frameon=False, loc="lower right", ncol=1)
    fig.suptitle("Preference consistency improvement under replay updates", fontweight="bold")
    fig.savefig(out_dir / "03_preference_consistency_transfer.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_preference_consistency_transfer.pdf", bbox_inches="tight")
    plt.close(fig)

    metrics = [
        ("active_pair_ratio", "active pair ratio"),
        ("saturated_pair_ratio", "saturated pair ratio"),
        ("top10_coefficient_mass", "top-10% coeff. mass"),
        ("coefficient_entropy", "coefficient entropy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes.reshape(-1), metrics):
        x = np.arange(len(problems))
        width = 0.14
        offsets = np.linspace(-0.28, 0.28, len(SIGNAL_METHODS))
        for mi, method in enumerate(SIGNAL_METHODS):
            vals = [_mean(results["saturation"], metric, problem=p, method=method) for p in problems]
            ax.bar(x + offsets[mi], vals, width=width, color=COLORS[method], label=method)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title)
    axes.reshape(-1)[0].legend(frameon=False, loc="best", ncol=2)
    fig.suptitle("Saturation and concentration of effective training coefficients", fontweight="bold")
    fig.savefig(out_dir / "04_saturation_analysis.png", bbox_inches="tight")
    fig.savefig(out_dir / "04_saturation_analysis.pdf", bbox_inches="tight")
    plt.close(fig)

    gap_groups = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, len(problems), figsize=(3.3 * len(problems), 3.2), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, label in zip(axes, problems, labels):
        x = np.arange(len(gap_groups))
        width = 0.14
        offsets = np.linspace(-0.28, 0.28, len(SIGNAL_METHODS))
        for mi, method in enumerate(SIGNAL_METHODS):
            vals = [
                _mean(results["gap_allocation"], "coefficient_mass_share", problem=problem, method=method, gap_group=g)
                for g in gap_groups
            ]
            ax.bar(x + offsets[mi], vals, width=width, color=COLORS[method], label=method)
        ax.axhline(1.0 / 3.0, color="#333333", lw=0.8, ls="--", alpha=0.55)
        ax.set_title(label)
        ax.set_xticks(x, gap_groups)
        ax.set_xlabel("relative-gap tertile")
    axes[0].set_ylabel("coefficient mass share")
    axes[-1].legend(frameon=False, loc="best", ncol=1)
    fig.suptitle("Where each loss allocates its training signal across pair difficulty", fontweight="bold")
    fig.savefig(out_dir / "05_gap_signal_allocation.png", bbox_inches="tight")
    fig.savefig(out_dir / "05_gap_signal_allocation.pdf", bbox_inches="tight")
    plt.close(fig)

    for metric, fname, ylabel in [
        ("loss_value", "06_loss_shape_value", "loss value"),
        ("gradient_magnitude", "07_loss_shape_gradient", "gradient magnitude"),
    ]:
        fig, axes = plt.subplots(3, len(problems), figsize=(3.1 * len(problems), 6.8), sharex=True, constrained_layout=True)
        if len(problems) == 1:
            axes = axes.reshape(3, 1)
        for col, (problem, label) in enumerate(zip(problems, labels)):
            for row, gap_group in enumerate(["small", "medium", "large"]):
                ax = axes[row, col]
                for method in ["PO/BT", "BOPO-style", "SimPO-style", "Loss-only"]:
                    xs = sorted({float(r["margin"]) for r in results["shape"] if r["problem"] == problem and r["gap_group"] == gap_group and r["method"] == method})
                    ys = [_mean(results["shape"], metric, problem=problem, gap_group=gap_group, method=method, margin=x) for x in xs]
                    ax.plot(xs, ys, color=COLORS[method], lw=1.35, label=method)
                ax.axvline(0.0, color="#333333", lw=0.7, alpha=0.55)
                if row == 0:
                    ax.set_title(label)
                if col == 0:
                    ax.set_ylabel(f"{gap_group}\n{ylabel}")
                if row == 2:
                    ax.set_xlabel("winner-loser margin")
        axes[0, -1].legend(frameon=False, loc="best")
        fig.suptitle("Loss shape visualization", fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    summary_rows: list[dict[str, Any]] = []
    for problem in problems:
        for method in SIGNAL_METHODS:
            summary_rows.append(
                {
                    "Problem": PROBLEM_LABELS[problem].replace("JSSP10x10", "JSSP"),
                    "Method": method,
                    "Active pair ratio": _mean(results["saturation"], "active_pair_ratio", problem=problem, method=method),
                    "Saturated pair ratio": _mean(results["saturation"], "saturated_pair_ratio", problem=problem, method=method),
                    "Top-10% coefficient mass": _mean(results["saturation"], "top10_coefficient_mass", problem=problem, method=method),
                    "Coefficient entropy": _mean(results["saturation"], "coefficient_entropy", problem=problem, method=method),
                    "Transfer consistency": _mean(results["consistency"], "consistency", problem=problem, method=method, scale_name="transfer", gap_bin="all"),
                }
            )
    _write_csv(out_dir / "07_loss_signal_summary_table.csv", summary_rows)
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    ax.axis("off")
    display = [
        [
            row["Problem"],
            row["Method"],
            f"{row['Active pair ratio']:.3f}",
            f"{row['Saturated pair ratio']:.3f}",
            f"{row['Top-10% coefficient mass']:.3f}",
            f"{row['Coefficient entropy']:.3f}",
            f"{row['Transfer consistency']:.3f}",
        ]
        for row in summary_rows
    ]
    columns = ["Problem", "Method", "Active", "Saturated", "Top-10% mass", "Entropy", "Transfer Cons."]
    table = ax.table(cellText=display, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#eef1f5")
            cell.set_text_props(weight="bold")
        elif c == 1 and cell.get_text().get_text() == "Loss-only":
            cell.set_facecolor("#e6f4ee")
    ax.set_title("Loss signal replay diagnosis summary", fontweight="bold", pad=12)
    fig.savefig(out_dir / "08_loss_signal_summary_table.png", bbox_inches="tight")
    fig.savefig(out_dir / "08_loss_signal_summary_table.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Loss signal replay diagnosis\n\n"
            "CPU replay/diagnosis analysis. No training is performed.\n\n"
            "The same preference pairs are used for RL-style, PO/BT, BOPO-style, SimPO-style, and searched loss-only coefficient comparisons.\n"
            "This isolates the loss signal shape from pair selection.\n\n"
            "Main use:\n"
            "- `01_loss_signal_separation.*`: coefficient versus pair quality gap.\n"
            "- `02_advantage_separation_by_rank.*`: net update signal from best to worst solution ranks.\n"
            "- `03_preference_consistency_transfer.*`: one-step replay consistency at the transfer scale.\n"
            "- `04_saturation_analysis.*`: active/saturated/concentrated coefficient diagnostics.\n"
            "- `05_gap_signal_allocation.*`: coefficient mass allocated to small/medium/large-gap pairs.\n"
            "- `06_loss_shape_value.*` and `07_loss_shape_gradient.*`: value and gradient shape as a function of pair margin.\n"
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
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "loss_signal_replay_diagnosis" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect(
        problems=problems,
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        state=str(args.state),
        sharpness=float(args.sharpness),
        max_pairs=max(int(args.max_pairs), 256),
        step_size=float(args.step_size),
    )
    for key, rows in results.items():
        _write_csv(out_dir / f"{key}.csv", rows)
    plot(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
