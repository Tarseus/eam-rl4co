from __future__ import annotations

import argparse
import csv
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
    _implicit_pref_for_method,
    _load_pair,
    _method_loss,
    _state_log_prob,
    _target_spec,
    _weight_stats_for_pref,
)
from scripts.plot_source_shift_behavior_fingerprint import _rl_loss  # noqa: E402


METHODS = ["RL", "PO", "SLL", "BOPO", "Loss-only", "Loss+Weighting"]
COLORS = {
    "RL": "#666666",
    "PO": "#E69F00",
    "SLL": "#009E73",
    "BOPO": "#CC79A7",
    "Loss-only": "#0072B2",
    "Loss+Weighting": "#D55E00",
}


def _loss_for_method(method: str, problem: str, fc: Mapping[str, torch.Tensor], log_prob: torch.Tensor, pairs: Mapping[str, Any]) -> torch.Tensor:
    if method == "RL":
        return _rl_loss(fc, log_prob)
    return _method_loss(method, problem, fc, log_prob, pairs)


def _solution_influence(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
    state: str,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    objective = fc["objective"].detach()
    lp = _state_log_prob(fc, state, sharpness).detach().clone().requires_grad_(True)
    loss = _loss_for_method(method, problem, fc, lp, pairs)
    loss.backward()
    influence = lp.grad.detach().abs()
    influence = influence / influence.sum(dim=1, keepdim=True).clamp_min(1e-12)
    regret = objective - objective.min(dim=1, keepdim=True).values
    regret = regret / regret.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    return objective, regret, influence


def _pair_influence_from_generated(
    method: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    pref = pairs[method][0](fc)
    if pref.pair_idx is None or pref.num_examples() <= 0:
        return None
    batch = pref.to_pairwise_loss_batch(fc)
    live: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            live[key] = value.detach().clone()
        else:
            live[key] = value
    for key in ("log_prob_w", "log_prob_l"):
        if key not in live or not isinstance(live[key], torch.Tensor):
            return None
        live[key].requires_grad_(True)
    loss = pairs[method][1](live)
    loss.backward()
    gw = live["log_prob_w"].grad
    gl = live["log_prob_l"].grad
    if gw is None or gl is None:
        return None
    b, w, l = pref.pair_idx
    influence = gw.detach().abs() + gl.detach().abs()
    if isinstance(pref.weight, torch.Tensor):
        weight = pref.weight.detach().float().reshape(-1).clamp_min(0.0)
    else:
        weight = torch.ones_like(influence)
    return b.detach(), w.detach(), l.detach(), influence.detach().float() * weight.clamp_min(1e-12)


def _pair_influence_proxy(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
    state: str,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if method == "RL":
        return None
    objective = fc["objective"].detach()
    lp = _state_log_prob(fc, state, sharpness).detach()
    if method in {"Loss-only", "Loss+Weighting"}:
        out = _pair_influence_from_generated(method, {**dict(fc), "log_prob": lp}, pairs)
        if out is not None:
            return out

    pref = _implicit_pref_for_method(method, problem, {**dict(fc), "log_prob": lp}, pairs)
    if pref.pair_idx is None or pref.num_examples() <= 0:
        return None
    b, w, l = pref.pair_idx
    margin = lp[b, w] - lp[b, l]
    if method == "PO":
        alpha = 1.0 if problem in {"ffsp100", "jssp10x10"} else 0.05
        influence = alpha * torch.sigmoid(-(alpha * margin))
    elif method == "BOPO":
        weight = pref.weight.detach().float().reshape(-1) if isinstance(pref.weight, torch.Tensor) else torch.ones_like(margin)
        influence = weight.clamp_min(0.0) * torch.sigmoid(-margin)
    elif method == "SLL":
        influence = torch.sigmoid(-margin)
    else:
        return None
    return b.detach(), w.detach(), l.detach(), influence.detach().float()


def _bin_index(values: torch.Tensor, bins: int) -> torch.Tensor:
    return torch.clamp((values * bins).long(), 0, bins - 1)


def _collect_spectrum_rows(
    *,
    method: str,
    problem: str,
    scale_name: str,
    scale: float,
    target_size: int,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
    state: str,
    sharpness: float,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    objective, regret, sol_influence = _solution_influence(
        method=method,
        problem=problem,
        fc=fc,
        pairs=pairs,
        state=state,
        sharpness=sharpness,
    )
    batch, k = objective.shape
    bin_mass = torch.zeros((bins,), device=objective.device)
    idx = _bin_index(regret.reshape(-1), bins)
    bin_mass.scatter_add_(0, idx, sol_influence.reshape(-1))
    bin_mass = bin_mass / bin_mass.sum().clamp_min(1e-12)

    pair_out = _pair_influence_proxy(
        method=method,
        problem=problem,
        fc=fc,
        pairs=pairs,
        state=state,
        sharpness=sharpness,
    )
    pair_bin_mass = torch.full((bins,), float("nan"), device=objective.device)
    b10 = int(_bin_index(torch.tensor([0.10], device=objective.device), bins)[0].item())
    b30 = int(_bin_index(torch.tensor([0.30], device=objective.device), bins)[0].item())
    low_info_mass = float(bin_mass[b10 : b30 + 1].sum().item())
    useful_pair_mass = float(bin_mass[b30:].sum().item())
    easy_pair_mass = float(bin_mass[: b10 + 1].sum().item())
    flat_sol_mass = sol_influence.reshape(-1)
    pair_ess = float(((flat_sol_mass.sum() ** 2) / (flat_sol_mass.square().sum().clamp_min(1e-12) * max(int(flat_sol_mass.numel()), 1))).item())
    pair_gap_drift_scale = float("nan")
    if pair_out is not None:
        b, w, l, influence = pair_out
        gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
        inst_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
        gap_norm = (gap / inst_range[b]).clamp(0.0, 1.0)
        margin = (_state_log_prob(fc, state, sharpness).detach()[b, w] - _state_log_prob(fc, state, sharpness).detach()[b, l])
        mass = influence.clamp_min(0.0)
        mass = mass / mass.sum().clamp_min(1e-12)
        pair_bin_mass = torch.zeros((bins,), device=objective.device)
        pair_bin_mass.scatter_add_(0, _bin_index(gap_norm, bins), mass)
        low_info_mass = float(mass[gap_norm <= 0.10].sum().item())
        useful_mask = (gap_norm >= 0.25) & (margin <= torch.quantile(margin.detach(), 0.50))
        easy_mask = (gap_norm >= 0.25) & (margin >= torch.quantile(margin.detach(), 0.80))
        useful_pair_mass = float(mass[useful_mask].sum().item())
        easy_pair_mass = float(mass[easy_mask].sum().item())
        n = max(int(mass.numel()), 1)
        pair_ess = float(((mass.sum() ** 2) / (mass.square().sum().clamp_min(1e-12) * n)).item())
        pair_gap_drift_scale = float(gap_norm.detach().std().item())
    else:
        pair_bin_mass = bin_mass
        pair_gap_drift_scale = float(regret.detach().std().item())

    response_rows: list[dict[str, Any]] = []
    response_grid = torch.zeros((6, 7), device=objective.device)
    if pair_out is not None:
        b, w, l, influence = pair_out
        gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
        inst_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
        yval = (gap / inst_range[b]).clamp(0.0, 1.0)
        lp_state = _state_log_prob(fc, state, sharpness).detach()
        margin = lp_state[b, w] - lp_state[b, l]
        xval = torch.clamp(margin / margin.detach().std().clamp_min(1e-6), -2.5, 2.5)
        mass = influence.clamp_min(0.0)
        mass = mass / mass.sum().clamp_min(1e-12)
    else:
        yval = regret.reshape(-1).clamp(0.0, 1.0)
        lp_state = _state_log_prob(fc, state, sharpness).detach()
        margin = (lp_state - lp_state.mean(dim=1, keepdim=True)) / lp_state.std(dim=1, keepdim=True).clamp_min(1e-6)
        xval = torch.clamp(margin.reshape(-1), -2.5, 2.5)
        mass = sol_influence.reshape(-1).clamp_min(0.0)
        mass = mass / mass.sum().clamp_min(1e-12)
    ybin = torch.clamp((yval * 6).long(), 0, 5)
    xbin = torch.clamp((((xval + 2.5) / 5.0) * 7).long(), 0, 6)
    response_grid.index_put_((ybin, xbin), mass, accumulate=True)
    response_grid = response_grid / response_grid.sum().clamp_min(1e-12)
    for gy in range(6):
        for gx in range(7):
            response_rows.append(
                {
                    "problem": problem,
                    "scale_name": scale_name,
                    "scale": float(scale),
                    "target_size": int(target_size),
                    "method": method,
                    "gap_bin": int(gy),
                    "margin_bin": int(gx),
                    "gap_center": (gy + 0.5) / 6,
                    "margin_center": -2.5 + (gx + 0.5) * (5.0 / 7),
                    "response_mass": float(response_grid[gy, gx].detach().cpu().item()),
                }
            )

    if method in {"PO", "BOPO", "SLL", "Loss-only", "Loss+Weighting"}:
        pref = _implicit_pref_for_method(method, problem, {**dict(fc), "log_prob": _state_log_prob(fc, state, sharpness).detach()}, pairs)
        weight_stats = _weight_stats_for_pref(pref, objective, method)
    else:
        weight_stats = {"weight_ess": 1.0, "top10_mass": 0.10, "hi_clip_share": 0.0, "lo_clip_share": 0.0}

    rows: list[dict[str, Any]] = []
    for bin_id in range(bins):
        rows.append(
            {
                "problem": problem,
                "scale_name": scale_name,
                "scale": float(scale),
                "target_size": int(target_size),
                "method": method,
                "bin": int(bin_id),
                "bin_center": (bin_id + 0.5) / bins,
                "solution_influence_mass": float(bin_mass[bin_id].detach().cpu().item()),
                "pair_gap_influence_mass": float(pair_bin_mass[bin_id].detach().cpu().item()) if torch.isfinite(pair_bin_mass[bin_id]) else float("nan"),
            }
        )

    top_mass = (sol_influence * (regret <= 0.10).float()).sum(dim=1).mean().item()
    bad_mass = (sol_influence * (regret >= 0.70).float()).sum(dim=1).mean().item()
    support = float(((sol_influence.reshape(batch, k).sum(dim=1) ** 2) / (sol_influence.reshape(batch, k).square().sum(dim=1).clamp_min(1e-12) * k)).mean().item())
    metrics = {
        "best_solution_mass": float(top_mass),
        "bad_solution_mass": float(bad_mass),
        "solution_support": support,
        "low_info_pair_mass": low_info_mass,
        "useful_hard_pair_mass": useful_pair_mass,
        "easy_pair_mass": easy_pair_mass,
        "pair_ess": pair_ess,
        "pair_gap_std": pair_gap_drift_scale,
        "weight_ess": float(weight_stats["weight_ess"]),
        "top10_weight_mass": float(weight_stats["top10_mass"]),
        "clip_share": float(weight_stats["hi_clip_share"] + weight_stats["lo_clip_share"]),
    }
    return rows, response_rows, metrics


def collect(
    *,
    problems: list[str],
    source_scale: float,
    shifted_scale: float,
    batches: int,
    seed: int,
    device: str,
    state: str,
    sharpness: float,
    bins: int,
) -> dict[str, Any]:
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
    spectrum_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for problem in problems:
        print(f"[problem] {problem}", flush=True)
        for scale_name, scale in [("source", source_scale), ("generalization", shifted_scale)]:
            target_spec = _target_spec(specs[problem], problem, scale)
            target_size = int(target_spec.hf.train_problem_size)
            print(f"[scale] {problem} {scale_name} target_size={target_size}", flush=True)
            caches = rollout_feature_caches(target_spec, seed=seed + int(round(scale * 1000)), device=torch.device(device))
            for cache_id, fc in enumerate(caches):
                for method in METHODS:
                    rows, responses, metrics = _collect_spectrum_rows(
                        method=method,
                        problem=problem,
                        scale_name=scale_name,
                        scale=scale,
                        target_size=target_size,
                        fc=fc,
                        pairs=pair_cache[problem],
                        state=state,
                        sharpness=sharpness,
                        bins=bins,
                    )
                    spectrum_rows.extend(rows)
                    response_rows.extend(responses)
                    metric_rows.append(
                        {
                            "problem": problem,
                            "scale_name": scale_name,
                            "scale": float(scale),
                            "target_size": target_size,
                            "method": method,
                            "cache_id": cache_id,
                            **metrics,
                        }
                    )
    return {"problems": problems, "bins": bins, "spectrum": spectrum_rows, "response": response_rows, "metrics": metric_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], **conds: Any) -> float:
    vals = []
    metric = str(conds.pop("metric"))
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            val = row.get(metric)
            if val is not None and np.isfinite(float(val)):
                vals.append(float(val))
    return float(np.mean(vals)) if vals else float("nan")


def plot(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    problems = results["problems"]
    bins = int(results["bins"])
    spectrum = results["spectrum"]
    response = results["response"]
    metrics = results["metrics"]

    def spectrum_heat(scale_name: str, value: str, fname: str, title: str) -> None:
        fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 3.35), constrained_layout=True)
        if len(problems) == 1:
            axes = [axes]
        mats = []
        for problem in problems:
            mat = np.full((len(METHODS), bins), np.nan)
            for i, method in enumerate(METHODS):
                for b in range(bins):
                    vals = [
                        float(r[value])
                        for r in spectrum
                        if r["problem"] == problem
                        and r["scale_name"] == scale_name
                        and r["method"] == method
                        and int(r["bin"]) == b
                        and np.isfinite(float(r[value]))
                    ]
                    if vals:
                        mat[i, b] = float(np.mean(vals))
            mats.append(mat)
        finite = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
        vmax = max(float(np.nanpercentile(finite, 98)), 1e-6)
        for ax, problem, mat in zip(axes, problems, mats):
            im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(PROBLEM_LABELS[problem])
            ax.set_yticks(np.arange(len(METHODS)), METHODS)
            ax.set_xticks([0, bins // 2, bins - 1], ["small", "medium", "large"])
            ax.set_xlabel("objective gap / regret")
        axes[0].set_ylabel("method")
        fig.colorbar(im, ax=axes, shrink=0.8, label="share of effective influence")
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    spectrum_heat(
        "source",
        "pair_gap_influence_mass",
        "01_source_pair_gap_influence_spectrum",
        "Why loss methods win at search scale: effective pair influence over objective gaps",
    )
    spectrum_heat(
        "generalization",
        "pair_gap_influence_mass",
        "02_generalization_pair_gap_influence_spectrum",
        "Generalization scale: weighting can move influence into a different gap spectrum",
    )

    def response_surfaces(scale_name: str, fname: str, title: str) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(9.7, 5.7), constrained_layout=True)
        axes_flat = axes.reshape(-1)
        mats = []
        for method in METHODS:
            mat = np.zeros((6, 7), dtype=float)
            for gy in range(6):
                for gx in range(7):
                    vals = [
                        float(r["response_mass"])
                        for r in response
                        if r["scale_name"] == scale_name
                        and r["method"] == method
                        and int(r["gap_bin"]) == gy
                        and int(r["margin_bin"]) == gx
                    ]
                    mat[gy, gx] = float(np.mean(vals)) if vals else np.nan
            mats.append(mat)
        finite = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
        vmax = max(float(np.nanpercentile(finite, 98)), 1e-6)
        for ax, method, mat in zip(axes_flat, METHODS, mats):
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set_title(method, color=COLORS[method], fontweight="semibold")
            ax.set_xticks([0, 3, 6], ["wrong/hard", "border", "easy"])
            ax.set_yticks([0, 2, 5], ["tiny", "mid", "large"])
            ax.set_xlabel("model margin")
            ax.set_ylabel("objective gap")
        fig.colorbar(im, ax=axes_flat, shrink=0.82, label="effective response mass")
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    response_surfaces(
        "source",
        "06_source_loss_response_surfaces",
        "Search scale: objective-aware response surfaces distinguish loss methods from baselines",
    )
    response_surfaces(
        "generalization",
        "07_generalization_loss_response_surfaces",
        "Generalization scale: weighting response can drift while loss-only remains smoother",
    )

    summary_metrics = [
        ("useful_hard_pair_mass", "useful hard-pair mass"),
        ("low_info_pair_mass", "low-info pair mass"),
        ("pair_ess", "pair ESS"),
        ("solution_support", "solution support"),
        ("clip_share", "clip share"),
    ]
    for scale_name, fname, title in [
        ("source", "03_source_mechanism_fingerprint", "Search-scale mechanism fingerprint"),
        ("generalization", "04_generalization_mechanism_fingerprint", "Generalization-scale mechanism fingerprint"),
    ]:
        mat = np.zeros((len(METHODS), len(summary_metrics)), dtype=float)
        for i, method in enumerate(METHODS):
            for j, (metric, _) in enumerate(summary_metrics):
                vals = [
                    _mean(metrics, problem=p, scale_name=scale_name, method=method, metric=metric)
                    for p in problems
                ]
                mat[i, j] = np.nanmean(vals)
        norm = (mat - np.nanmean(mat, axis=0, keepdims=True)) / (np.nanstd(mat, axis=0, keepdims=True) + 1e-8)
        fig, ax = plt.subplots(figsize=(8.4, 3.2), constrained_layout=True)
        im = ax.imshow(norm, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        ax.set_yticks(np.arange(len(METHODS)), METHODS)
        ax.set_xticks(np.arange(len(summary_metrics)), [label for _, label in summary_metrics], rotation=25, ha="right")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.4)
        fig.colorbar(im, ax=ax, shrink=0.82, label="column z-score")
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Directly visualize the specific causal claim: weighting's source-scale advantage is tied to
    # concentrated weights; under scale shift the concentration/clipping remains while pair ESS drops.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)
    for ax, scale_name, title in zip(axes, ["source", "generalization"], ["search scale", "generalization scale"]):
        for method in METHODS:
            xvals = [_mean(metrics, problem=p, scale_name=scale_name, method=method, metric="useful_hard_pair_mass") for p in problems]
            yvals = [_mean(metrics, problem=p, scale_name=scale_name, method=method, metric="low_info_pair_mass") for p in problems]
            ess = np.nanmean([_mean(metrics, problem=p, scale_name=scale_name, method=method, metric="pair_ess") for p in problems])
            x = float(np.nanmean(xvals))
            y = float(np.nanmean(yvals))
            size = 80 + 420 * max(min(ess if np.isfinite(ess) else 0.2, 1.0), 0.02)
            ax.scatter(x, y, s=size, color=COLORS[method], alpha=0.86, edgecolor="white", linewidth=0.8)
            ax.text(x, y, method, fontsize=8, ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel("useful hard-pair influence")
        ax.set_ylabel("low-information influence")
        ax.grid(alpha=0.25)
    fig.suptitle("Causal map: good methods put mass on hard informative comparisons, not low-information pressure", fontweight="bold")
    fig.savefig(out_dir / "05_useful_vs_low_information_map.png", bbox_inches="tight")
    fig.savefig(out_dir / "05_useful_vs_low_information_map.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for problem in problems:
        for method in METHODS:
            src_ess = _mean(metrics, problem=problem, scale_name="source", method=method, metric="pair_ess")
            gen_ess = _mean(metrics, problem=problem, scale_name="generalization", method=method, metric="pair_ess")
            src_clip = _mean(metrics, problem=problem, scale_name="source", method=method, metric="clip_share")
            gen_clip = _mean(metrics, problem=problem, scale_name="generalization", method=method, metric="clip_share")
            src_low = _mean(metrics, problem=problem, scale_name="source", method=method, metric="low_info_pair_mass")
            gen_low = _mean(metrics, problem=problem, scale_name="generalization", method=method, metric="low_info_pair_mass")
            rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "pair_ess_drop": src_ess - gen_ess,
                    "clip_increase": gen_clip - src_clip,
                    "low_info_increase": gen_low - src_low,
                }
            )
    _write_csv(out_dir / "generalization_shift_summary.csv", rows)

    drift_rows = []
    for method in METHODS:
        src = np.zeros((6, 7), dtype=float)
        gen = np.zeros((6, 7), dtype=float)
        for gy in range(6):
            for gx in range(7):
                src_vals = [
                    float(r["response_mass"])
                    for r in response
                    if r["scale_name"] == "source"
                    and r["method"] == method
                    and int(r["gap_bin"]) == gy
                    and int(r["margin_bin"]) == gx
                ]
                gen_vals = [
                    float(r["response_mass"])
                    for r in response
                    if r["scale_name"] == "generalization"
                    and r["method"] == method
                    and int(r["gap_bin"]) == gy
                    and int(r["margin_bin"]) == gx
                ]
                src[gy, gx] = float(np.mean(src_vals)) if src_vals else 0.0
                gen[gy, gx] = float(np.mean(gen_vals)) if gen_vals else 0.0
        drift_rows.append({"method": method, "response_l1_drift": float(np.abs(gen - src).sum() / 2.0)})
    _write_csv(out_dir / "response_surface_drift.csv", drift_rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.1), constrained_layout=True)
    xs = np.arange(len(METHODS))
    vals = [r["response_l1_drift"] for r in drift_rows]
    ax.bar(xs, vals, color=[COLORS[m] for m in METHODS], width=0.68)
    ax.set_xticks(xs, METHODS, rotation=20, ha="right")
    ax.set_ylabel("source-to-generalization response drift")
    ax.set_title("Which objective response changes most under scale shift?", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "08_response_surface_drift.png", bbox_inches="tight")
    fig.savefig(out_dir / "08_response_surface_drift.pdf", bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Causal mechanism analysis\n\n"
            "Use these figures as explanatory evidence, not as the main result figure.\n\n"
            "- Figures 01/02 show where each method's effective loss influence lies on the objective-gap spectrum.\n"
            "- Figures 03/04 summarize useful hard-pair mass, low-information mass, effective support, and clipping.\n"
            "- Figure 05 directly contrasts useful hard comparisons against low-information pressure.\n"
            "- Figures 06/07 show the loss response surface over objective gap and current model margin.\n"
            "- Figure 08 quantifies source-to-generalization response-surface drift.\n"
            "- `mechanism_metrics.csv` and `generalization_shift_summary.csv` contain raw values.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--source-scale", type=float, default=1.0)
    parser.add_argument("--shifted-scale", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--state", default="aligned", choices=["sampled", "aligned", "misaligned"])
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "causal_mechanism_loss_weighting" / stamp))
    results = collect(
        problems=problems,
        source_scale=float(args.source_scale),
        shifted_scale=float(args.shifted_scale),
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        state=str(args.state),
        sharpness=float(args.sharpness),
        bins=max(int(args.bins), 6),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "influence_spectrum.csv", results["spectrum"])
    _write_csv(out_dir / "response_surface.csv", results["response"])
    _write_csv(out_dir / "mechanism_metrics.csv", results["metrics"])
    plot(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
