from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

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

from scripts.final_gradient_behavior_analysis import build_problem_specs, rollout_feature_caches  # noqa: E402
from scripts.plot_loss_signal_replay_diagnosis import (  # noqa: E402
    COLORS,
    PROBLEMS,
    _coefficients_for_method,
    _pair_features,
    _subsample_pref,
    _with_uniform_weight,
)
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


BASELINES = ["RL", "PO/BT", "BOPO-style", "SLL"]
FINE_SIGNAL_METHODS = [*BASELINES, "Loss-only"]
PLOT_COLORS = {**COLORS, "SLL": "#F58518"}
FEATURES = [
    ("margin", "policy margin"),
    ("rank_diff_norm", "rank diff."),
    ("winner_rank_norm", "winner rank"),
    ("loser_rank_norm", "loser rank"),
    ("pool_std", "pool std"),
    ("pool_range", "pool range"),
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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _norm_mean(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / max(float(np.nanmean(arr[np.isfinite(arr)])), 1e-12)


def _quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, qs)
    if np.unique(edges).size < 3:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if abs(hi - lo) <= 1e-12:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_ids(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)


def _gap_matched_baseline(loss_coeff: np.ndarray, base_coeff: np.ndarray, rel_gap: np.ndarray, *, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    loss = _norm_mean(loss_coeff)
    base = _norm_mean(base_coeff)
    edges = _quantile_bins(rel_gap, n_bins)
    bins = _bin_ids(rel_gap, edges)
    matched = np.array(base, copy=True)
    for bi in range(n_bins):
        mask = bins == bi
        if not np.any(mask):
            continue
        scale = float(np.nanmean(loss[mask])) / max(float(np.nanmean(base[mask])), 1e-12)
        matched[mask] = base[mask] * scale
    return matched, bins


def _rmse_curve(loss_coeff: np.ndarray, base_coeff: np.ndarray, rel_gap: np.ndarray, *, n_bins: int = 10) -> float:
    loss = _norm_mean(loss_coeff)
    base = _norm_mean(base_coeff)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    errs: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rel_gap >= lo) & (rel_gap < hi if hi < 1.0 else rel_gap <= hi)
        if not np.any(mask):
            continue
        errs.append((float(np.nanmean(loss[mask])) - float(np.nanmean(base[mask]))) ** 2)
    return float(math.sqrt(float(np.nanmean(errs)))) if errs else float("nan")


def _hardness_group(margin: np.ndarray) -> np.ndarray:
    q1, q2 = np.nanquantile(margin, [0.33, 0.67])
    out = np.full(margin.shape, "medium", dtype=object)
    out[margin <= q1] = "hard"
    out[margin > q2] = "easy"
    return out


def _sll_pair_coefficients(pref: Any, fc: dict[str, torch.Tensor]) -> np.ndarray:
    # SLL is listwise in training; for this pairwise residual diagnostic we use
    # its local winner-loser influence proxy on the same pairs as Loss-only.
    pf = _pair_features(pref, fc)
    margin = _as_np(pf["margin"])
    return 1.0 / (1.0 + np.exp(np.clip(margin, -50.0, 50.0)))


def _coefficients_for_fine_method(
    method: str,
    problem: str,
    pref: Any,
    fc: dict[str, torch.Tensor],
    pairs: dict[str, Any],
    *,
    alpha: float,
) -> np.ndarray:
    if method == "SLL":
        return _sll_pair_coefficients(pref, fc)
    return _coefficients_for_method(method, problem, pref, fc, pairs, alpha=alpha)


def _collect_one_problem(
    *,
    problem: str,
    batches: int,
    seed: int,
    device: str,
    state: str,
    sharpness: float,
    max_pairs: int,
) -> dict[str, Any]:
    specs = build_problem_specs(device, batches)
    source_spec = _target_spec(specs[problem], problem, 1.0)
    alpha = float(source_spec.hf.alpha)
    pairs = {
        "Loss-only": _load_pair(LOSS_ONLY[problem]),
        "Loss+Weighting": _load_pair(WEIGHTING[problem]),
    }
    caches = rollout_feature_caches(source_spec, seed=seed + 117, device=torch.device(device))
    coeffs: dict[str, list[np.ndarray]] = {m: [] for m in FINE_SIGNAL_METHODS}
    scale_sensitivity_rows: list[dict[str, Any]] = []
    feats: dict[str, list[np.ndarray]] = {
        "relative_gap": [],
        "margin": [],
        "rank_diff_norm": [],
        "winner_rank_norm": [],
        "loser_rank_norm": [],
        "pool_std": [],
        "pool_range": [],
        "cache_id": [],
        "instance_id": [],
    }

    for cache_id, raw_fc in enumerate(caches):
        log_prob = _state_log_prob(raw_fc, state, sharpness).detach()
        fc = {**dict(raw_fc), "log_prob": log_prob}
        pref = _with_uniform_weight(pairs["Loss-only"][0](fc), fc)
        pref = _subsample_pref(pref, max_pairs, seed + 1009 * cache_id)
        pf = _pair_features(pref, fc)
        objective = fc["objective"].detach()
        b = pf["b"]
        pool_std = objective.std(dim=1).clamp_min(1e-12)
        pool_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
        feats["relative_gap"].append(_as_np(pf["relative_gap"]))
        feats["margin"].append(_as_np(pf["margin"]))
        feats["rank_diff_norm"].append(_as_np(pf["rank_diff_norm"]))
        feats["winner_rank_norm"].append(_as_np(pf["winner_rank_norm"]))
        feats["loser_rank_norm"].append(_as_np(pf["loser_rank_norm"]))
        feats["pool_std"].append(_as_np(pool_std[b]))
        feats["pool_range"].append(_as_np(pool_range[b]))
        feats["cache_id"].append(np.full((int(b.numel()),), cache_id, dtype=np.int64))
        feats["instance_id"].append(_as_np(b).astype(np.int64))
        for method in FINE_SIGNAL_METHODS:
            coeffs[method].append(_coefficients_for_fine_method(method, problem, pref, fc, pairs, alpha=alpha))
        if cache_id == 0:
            base_objective = raw_fc["objective"].detach()
            centered = base_objective - base_objective.mean(dim=1, keepdim=True)
            center = base_objective.mean(dim=1, keepdim=True)
            for objective_scale in [0.25, 0.50, 1.0, 2.0, 4.0]:
                scaled_objective = center + float(objective_scale) * centered
                scaled_fc = _replace_objective(raw_fc, scaled_objective, log_prob)
                scaled_pref = _with_uniform_weight(pairs["Loss-only"][0](scaled_fc), scaled_fc)
                scaled_pref = _subsample_pref(scaled_pref, max_pairs, seed + 3337)
                for method in FINE_SIGNAL_METHODS:
                    coeff = _coefficients_for_fine_method(method, problem, scaled_pref, scaled_fc, pairs, alpha=alpha)
                    scale_sensitivity_rows.append(
                        {
                            "problem": problem,
                            "objective_scale": float(objective_scale),
                            "method": method,
                            "mean_coefficient": float(np.nanmean(coeff)),
                            "median_coefficient": float(np.nanmedian(coeff)),
                        }
                    )

    return {
        "coeffs": {k: np.concatenate(v) for k, v in coeffs.items()},
        "features": {k: np.concatenate(v) for k, v in feats.items()},
        "scale_sensitivity": scale_sensitivity_rows,
    }


def collect(
    *,
    problems: list[str],
    batches: int,
    seed: int,
    device: str,
    state: str,
    sharpness: float,
    max_pairs: int,
) -> dict[str, list[dict[str, Any]]]:
    if "jssp10x10" in problems:
        _ensure_jssp_data(REPO_ROOT)
    nearest_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []
    hardness_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []

    for problem in problems:
        print(f"[fine] {problem}", flush=True)
        data = _collect_one_problem(
            problem=problem,
            batches=batches,
            seed=seed,
            device=device,
            state=state,
            sharpness=sharpness,
            max_pairs=max_pairs,
        )
        feats = data["features"]
        coeffs = data["coeffs"]
        scale_rows.extend(data["scale_sensitivity"])
        rel_gap = feats["relative_gap"]
        loss_coeff = coeffs["Loss-only"]
        rmses = {m: _rmse_curve(loss_coeff, coeffs[m], rel_gap) for m in BASELINES}
        nearest = min(rmses, key=lambda m: (float("inf") if not np.isfinite(rmses[m]) else rmses[m]))
        for method, rmse in rmses.items():
            nearest_rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "gap_curve_rmse_to_loss_only": rmse,
                    "is_nearest": method == nearest,
                }
            )

        loss_norm = _norm_mean(loss_coeff)
        margin_edges = _quantile_bins(feats["margin"], 8)
        margin_bin = _bin_ids(feats["margin"], margin_edges)
        hardness = _hardness_group(feats["margin"])
        for baseline in BASELINES:
            matched, gap_bin = _gap_matched_baseline(loss_coeff, coeffs[baseline], rel_gap, n_bins=8)
            residual = np.log2((loss_norm + 1e-8) / (matched + 1e-8))
            for gi in range(8):
                for mi in range(8):
                    mask = (gap_bin == gi) & (margin_bin == mi)
                    if not np.any(mask):
                        continue
                    surface_rows.append(
                        {
                            "problem": problem,
                            "baseline": baseline,
                            "nearest_baseline": nearest,
                            "is_nearest": baseline == nearest,
                            "gap_bin": gi,
                            "margin_bin": mi,
                            "mean_log2_loss_over_matched_baseline": float(np.nanmean(residual[mask])),
                            "pair_count": int(np.sum(mask)),
                        }
                    )

            for feature, label in FEATURES:
                corr = _safe_corr(residual, feats[feature])
                dependency_rows.append(
                    {
                        "problem": problem,
                        "baseline": baseline,
                        "nearest_baseline": nearest,
                        "is_nearest": baseline == nearest,
                        "feature": feature,
                        "feature_label": label,
                        "residual_corr": corr,
                        "abs_residual_corr": abs(corr) if np.isfinite(corr) else float("nan"),
                    }
                )

            for group in ["hard", "medium", "easy"]:
                mask = hardness == group
                if not np.any(mask):
                    continue
                hardness_rows.append(
                    {
                        "problem": problem,
                        "baseline": baseline,
                        "nearest_baseline": nearest,
                        "is_nearest": baseline == nearest,
                        "hardness_group": group,
                        "mean_log2_loss_over_matched_baseline": float(np.nanmean(residual[mask])),
                        "loss_coefficient_mass_share": float(np.nansum(loss_norm[mask]) / max(np.nansum(loss_norm), 1e-12)),
                        "matched_baseline_mass_share": float(np.nansum(matched[mask]) / max(np.nansum(matched), 1e-12)),
                        "pair_share": float(np.mean(mask)),
                    }
                )

        keys = feats["cache_id"].astype(np.int64) * 1000000 + feats["instance_id"].astype(np.int64)
        for method in ["Loss-only", *BASELINES]:
            coeff = _norm_mean(coeffs[method])
            inst_coeff: list[float] = []
            inst_std: list[float] = []
            inst_range: list[float] = []
            for key in np.unique(keys):
                mask = keys == key
                inst_coeff.append(float(np.nanmean(coeff[mask])))
                inst_std.append(float(np.nanmean(feats["pool_std"][mask])))
                inst_range.append(float(np.nanmean(feats["pool_range"][mask])))
            instance_rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "nearest_baseline": nearest,
                    "mean_coeff_vs_pool_std_corr": _safe_corr(np.asarray(inst_coeff), np.asarray(inst_std)),
                    "mean_coeff_vs_pool_range_corr": _safe_corr(np.asarray(inst_coeff), np.asarray(inst_range)),
                }
            )

        edges = np.linspace(0.0, 1.0, 11)
        for method in ["Loss-only", nearest]:
            coeff = _norm_mean(coeffs[method])
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (rel_gap >= lo) & (rel_gap < hi if hi < 1.0 else rel_gap <= hi)
                curve_rows.append(
                    {
                        "problem": problem,
                        "method": method,
                        "nearest_baseline": nearest,
                        "bin_mid": float(0.5 * (lo + hi)),
                        "mean_normalized_coefficient": float(np.nanmean(coeff[mask])) if np.any(mask) else float("nan"),
                    }
                )

    return {
        "nearest": nearest_rows,
        "surface": surface_rows,
        "dependency": dependency_rows,
        "hardness": hardness_rows,
        "instance": instance_rows,
        "curves": curve_rows,
        "scale_sensitivity": scale_rows,
    }


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
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    problems = sorted({r["problem"] for r in results["nearest"]}, key=PROBLEMS.index)
    labels = [PROBLEM_LABELS[p].replace("JSSP10x10", "JSSP") for p in problems]

    fig, axes = plt.subplots(1, len(problems), figsize=(3.1 * len(problems), 3.0), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, label in zip(axes, problems, labels):
        nearest = next(r["method"] for r in results["nearest"] if r["problem"] == problem and str(r["is_nearest"]) == "True")
        for method in ["Loss-only", nearest]:
            xs = sorted({float(r["bin_mid"]) for r in results["curves"] if r["problem"] == problem and r["method"] == method})
            ys = [_mean(results["curves"], "mean_normalized_coefficient", problem=problem, method=method, bin_mid=x) for x in xs]
            ax.plot(xs, ys, marker="o", ms=3.0, lw=1.6, color=PLOT_COLORS.get(method, "#555555"), label=method)
        ax.set_title(f"{label}\nnearest: {nearest}")
        ax.set_xlabel("relative objective gap")
    axes[0].set_ylabel("mean-normalized coefficient")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Nearest hand-crafted counterfactual before fine-grained matching", fontweight="bold")
    fig.savefig(out_dir / "01_nearest_counterfactual_gap_curve.png", bbox_inches="tight")
    fig.savefig(out_dir / "01_nearest_counterfactual_gap_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(problems), len(BASELINES), figsize=(2.35 * len(BASELINES), 2.25 * len(problems)), constrained_layout=True)
    if len(problems) == 1:
        axes = axes.reshape(1, len(BASELINES))
    vmax = 2.0
    im = None
    for ri, (problem, label) in enumerate(zip(problems, labels)):
        nearest = next(r["method"] for r in results["nearest"] if r["problem"] == problem and str(r["is_nearest"]) == "True")
        for ci, baseline in enumerate(BASELINES):
            ax = axes[ri, ci]
            mat = np.full((8, 8), np.nan, dtype=np.float64)
            for row in results["surface"]:
                if row["problem"] != problem or row["baseline"] != baseline:
                    continue
                mat[int(row["gap_bin"]), int(row["margin_bin"])] = float(row["mean_log2_loss_over_matched_baseline"])
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            suffix = " *" if baseline == nearest else ""
            if ri == 0:
                ax.set_title(f"{baseline}{suffix}")
            if ci == 0:
                ax.set_ylabel(f"{label}\nobjective-gap quantile\nlow=fine, high=coarse")
            if ri == len(problems) - 1:
                ax.set_xlabel("policy-margin quantile\nlow=hard, high=easy")
            ax.set_xticks([0, 7], ["low\nhard", "high\neasy"])
            ax.set_yticks([0, 7], ["low\nfine", "high\ncoarse"])
    if im is not None:
        fig.colorbar(
            im,
            ax=axes,
            shrink=0.78,
            label="log2 coefficient ratio\nLoss-only / gap-matched baseline\nred=stronger, blue=weaker",
        )
    fig.suptitle("Fine-grained residual after matching each baseline's objective-gap response", fontweight="bold")
    fig.savefig(out_dir / "02_all_methods_gap_matched_residual_surface.png", bbox_inches="tight")
    fig.savefig(out_dir / "02_all_methods_gap_matched_residual_surface.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 3.0), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    x = np.arange(len(BASELINES))
    for ax, problem, label in zip(axes, problems, labels):
        vals = []
        for baseline in BASELINES:
            hard = _mean(results["hardness"], "mean_log2_loss_over_matched_baseline", problem=problem, baseline=baseline, hardness_group="hard")
            easy = _mean(results["hardness"], "mean_log2_loss_over_matched_baseline", problem=problem, baseline=baseline, hardness_group="easy")
            vals.append(hard - easy)
        colors = [PLOT_COLORS.get(b, "#777777") for b in BASELINES]
        ax.bar(x, vals, color=colors, width=0.66)
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(label)
        ax.set_xticks(x, BASELINES, rotation=35, ha="right")
    axes[0].set_ylabel("hard-easy log2 residual")
    fig.suptitle("Policy-hardness selectivity after matching each baseline's gap curve", fontweight="bold")
    fig.savefig(out_dir / "03_all_methods_hardness_residual.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_all_methods_hardness_residual.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 3.1), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, label in zip(axes, problems, labels):
        vals = []
        tick_labels = []
        for baseline in BASELINES:
            dep_rows = [r for r in results["dependency"] if r["problem"] == problem and r["baseline"] == baseline]
            best = max((float(r["abs_residual_corr"]) for r in dep_rows if np.isfinite(float(r["abs_residual_corr"]))), default=float("nan"))
            vals.append(best)
            tick_labels.append(baseline)
        ax.bar(np.arange(len(BASELINES)), vals, color=[PLOT_COLORS.get(b, "#777777") for b in BASELINES], width=0.66)
        ax.set_title(label)
        ax.set_xticks(np.arange(len(BASELINES)), tick_labels, rotation=35, ha="right")
        ax.set_ylim(0.0, max(0.05, np.nanmax(vals) * 1.25))
    axes[0].set_ylabel("max |corr(residual, secondary feature)|")
    fig.suptitle("Residual structure remains after gap matching every baseline", fontweight="bold")
    fig.savefig(out_dir / "04_all_methods_residual_dependency.png", bbox_inches="tight")
    fig.savefig(out_dir / "04_all_methods_residual_dependency.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 3.4), constrained_layout=True)
    x = np.arange(len(problems))
    width = 0.34
    loss_vals = [_mean(results["instance"], "mean_coeff_vs_pool_std_corr", problem=p, method="Loss-only") for p in problems]
    near_vals = []
    near_labels = []
    for p in problems:
        nearest = next(r["method"] for r in results["nearest"] if r["problem"] == p and str(r["is_nearest"]) == "True")
        near_labels.append(nearest)
        near_vals.append(_mean(results["instance"], "mean_coeff_vs_pool_std_corr", problem=p, method=nearest))
    ax.bar(x - width / 2, loss_vals, width=width, color=COLORS["Loss-only"], label="Loss-only")
    ax.bar(x + width / 2, near_vals, width=width, color="#8C8C8C", label="nearest baseline")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, [f"{lab}\n({near})" for lab, near in zip(labels, near_labels)])
    ax.set_ylabel("corr(instance mean coeff., pool std)")
    ax.set_title("Instance-adaptive normalization differs from the nearest loss", fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "05_instance_adaptive_normalization.png", bbox_inches="tight")
    fig.savefig(out_dir / "05_instance_adaptive_normalization.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 3.0), sharey=False, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, label in zip(axes, problems, labels):
        nearest = next(r["method"] for r in results["nearest"] if r["problem"] == problem and str(r["is_nearest"]) == "True")
        for method in ["Loss-only", *BASELINES]:
            xs = sorted({float(r["objective_scale"]) for r in results["scale_sensitivity"] if r["problem"] == problem and r["method"] == method})
            raw = [_mean(results["scale_sensitivity"], "mean_coefficient", problem=problem, method=method, objective_scale=x) for x in xs]
            base = raw[xs.index(1.0)] if 1.0 in xs and np.isfinite(raw[xs.index(1.0)]) else np.nanmean(raw)
            ys = [v / max(base, 1e-12) for v in raw]
            ax.plot(xs, ys, marker="o", ms=3.0, lw=1.45, color=COLORS.get(method, "#555555"), label=method)
        ax.axhline(1.0, color="#333333", lw=0.8, ls="--", alpha=0.55)
        ax.set_xscale("log", base=2)
        ax.set_title(label)
        ax.set_xlabel("counterfactual objective-dispersion scale s\nc' = mean(c) + s(c - mean(c))")
    axes[0].set_ylabel("relative mean coefficient")
    axes[-1].legend(frameon=False, loc="best")
    axes[0].set_ylabel("mean training coefficient\nnormalized by value at s=1")
    fig.suptitle("Counterfactual replay: sensitivity to objective dispersion with policy margins fixed", fontweight="bold")
    fig.savefig(out_dir / "06_counterfactual_objective_scale_sensitivity.png", bbox_inches="tight")
    fig.savefig(out_dir / "06_counterfactual_objective_scale_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for p in problems:
        nearest = next(r["method"] for r in results["nearest"] if r["problem"] == p and str(r["is_nearest"]) == "True")
        for baseline in BASELINES:
            hard = _mean(results["hardness"], "mean_log2_loss_over_matched_baseline", problem=p, baseline=baseline, hardness_group="hard")
            easy = _mean(results["hardness"], "mean_log2_loss_over_matched_baseline", problem=p, baseline=baseline, hardness_group="easy")
            dep = max(
                _mean(results["dependency"], "abs_residual_corr", problem=p, baseline=baseline, feature=f)
                for f, _ in FEATURES
            )
            inst_loss = _mean(results["instance"], "mean_coeff_vs_pool_std_corr", problem=p, method="Loss-only")
            inst_base = _mean(results["instance"], "mean_coeff_vs_pool_std_corr", problem=p, method=baseline)
            lo_scale025 = _mean(results["scale_sensitivity"], "mean_coefficient", problem=p, method="Loss-only", objective_scale=0.25)
            lo_scale4 = _mean(results["scale_sensitivity"], "mean_coefficient", problem=p, method="Loss-only", objective_scale=4.0)
            base_scale025 = _mean(results["scale_sensitivity"], "mean_coefficient", problem=p, method=baseline, objective_scale=0.25)
            base_scale4 = _mean(results["scale_sensitivity"], "mean_coefficient", problem=p, method=baseline, objective_scale=4.0)
            summary_rows.append(
                {
                    "Problem": PROBLEM_LABELS[p].replace("JSSP10x10", "JSSP"),
                    "Baseline": baseline,
                    "Is nearest": baseline == nearest,
                    "Hard residual log2": hard,
                    "Easy residual log2": easy,
                    "Hard-vs-easy residual gap": hard - easy,
                    "Max secondary-feature residual corr": dep,
                    "Loss-only pool-std corr": inst_loss,
                    "Baseline pool-std corr": inst_base,
                    "Loss-only scale4/scale0.25": lo_scale4 / max(lo_scale025, 1e-12),
                    "Baseline scale4/scale0.25": base_scale4 / max(base_scale025, 1e-12),
                }
            )
    _write_csv(out_dir / "06_fine_grained_signature_summary.csv", summary_rows)
    fig, ax = plt.subplots(figsize=(11.0, 2.7))
    ax.axis("off")
    display = [
        [
            r["Problem"],
            r["Baseline"] + ("*" if r["Is nearest"] else ""),
            f"{r['Hard residual log2']:.2f}",
            f"{r['Easy residual log2']:.2f}",
            f"{r['Hard-vs-easy residual gap']:.2f}",
            f"{r['Max secondary-feature residual corr']:.2f}",
            f"{r['Loss-only pool-std corr']:.2f}",
            f"{r['Baseline pool-std corr']:.2f}",
            f"{r['Loss-only scale4/scale0.25']:.2f}",
            f"{r['Baseline scale4/scale0.25']:.2f}",
        ]
        for r in summary_rows
    ]
    cols = ["Problem", "Baseline", "Hard res.", "Easy res.", "Hard-Easy", "Max dep.", "LO std corr", "Base std corr", "LO scale", "Base scale"]
    table = ax.table(cellText=display, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d7d7")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#eef1f5")
            cell.set_text_props(weight="bold")
    ax.set_title("Fine-grained residual signature after nearest-baseline matching", fontweight="bold", pad=10)
    fig.savefig(out_dir / "07_fine_grained_signature_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "07_fine_grained_signature_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Fine-grained loss signature diagnosis\n\n"
            "This analysis first identifies the nearest hand-crafted baseline to searched loss-only on the 1D coefficient-vs-gap curve.\n"
            "Then it gap-matches that nearest baseline and analyzes the residual signature.\n\n"
            "If the residual still depends on margin hardness, rank coordinates, or instance-level pool dispersion, the difference is not explained by the coarse gap curve.\n"
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
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "loss_fine_grained_signature" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect(
        problems=problems,
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        state=str(args.state),
        sharpness=float(args.sharpness),
        max_pairs=max(int(args.max_pairs), 256),
    )
    for key, rows in results.items():
        _write_csv(out_dir / f"{key}.csv", rows)
    plot(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
