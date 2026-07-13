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

from scripts.final_gradient_behavior_analysis import build_problem_specs, rollout_feature_caches  # noqa: E402
from scripts.plot_scale_generalization_loss_weighting import (  # noqa: E402
    COLORS as BASE_COLORS,
    PROBLEM_LABELS,
    WEIGHTING,
    LOSS_ONLY,
    _ensure_jssp_data,
    _load_pair,
    _method_loss,
    _state_log_prob,
    _target_spec,
    _implicit_pref_for_method,
    _weight_stats_for_pref,
)


METHODS = ["RL", "PO", "BOPO", "SLL", "Loss-only", "Loss+Weighting"]
COLORS = {"RL": "#9D755D", **BASE_COLORS}


def _rl_loss(fc: Mapping[str, torch.Tensor], log_prob: torch.Tensor) -> torch.Tensor:
    reward = -fc["objective"].detach()
    adv = reward - reward.mean(dim=1, keepdim=True)
    adv = adv / adv.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return -(adv.detach() * log_prob).mean()


def _loss_for_method(method: str, problem: str, fc: Mapping[str, torch.Tensor], log_prob: torch.Tensor, pairs: Mapping[str, Any]) -> torch.Tensor:
    if method == "RL":
        return _rl_loss(fc, log_prob)
    return _method_loss(method, problem, fc, log_prob, pairs)


def _update_for_method(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
    state: str,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    objective = fc["objective"].detach()
    lp0 = _state_log_prob(fc, state, sharpness).detach().clone().requires_grad_(True)
    loss = _loss_for_method(method, problem, fc, lp0, pairs)
    loss.backward()
    update = -lp0.grad.detach()
    update = update - update.mean(dim=1, keepdim=True)
    update = update / update.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return objective, update


def _profile_rows(objective: torch.Tensor, update: torch.Tensor, *, bins: int) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    signed: dict[int, list[float]] = {i: [] for i in range(bins)}
    budget: dict[int, list[float]] = {i: [] for i in range(bins)}
    batch, k = objective.shape
    sorted_idx = objective.sort(dim=1, descending=False).indices
    sorted_update = update.gather(1, sorted_idx)
    abs_mass = sorted_update.abs()
    mass_norm = abs_mass / abs_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
    for rank in range(k):
        bin_id = min(int(rank * bins / max(k, 1)), bins - 1)
        signed[bin_id].extend(float(x) for x in sorted_update[:, rank].detach().cpu().numpy().reshape(-1))
        budget[bin_id].extend(float(x) for x in mass_norm[:, rank].detach().cpu().numpy().reshape(-1))
    signed_rows = [
        {"rank_bin": i, "rank_center": (i + 0.5) / bins, "value": float(np.mean(signed[i]))}
        for i in range(bins)
    ]
    budget_rows = [
        {"rank_bin": i, "rank_center": (i + 0.5) / bins, "value": float(np.sum(budget[i]) / max(batch, 1))}
        for i in range(bins)
    ]
    return signed_rows, budget_rows


def _instance_rows(
    objective: torch.Tensor,
    update: torch.Tensor,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    batch, k = objective.shape
    for b in range(batch):
        obj = objective[b]
        upd = update[b]
        order = obj.sort(descending=False).indices
        n = max(int(np.ceil(0.10 * k)), 1)
        separation = (upd[order[:n]].mean() - upd[order[-n:]].mean()).item()
        mass = upd.abs()
        support = (mass.sum() ** 2 / (mass.square().sum().clamp_min(1e-12) * k)).item()
        budget_entropy = (-(mass / mass.sum().clamp_min(1e-8)) * (mass / mass.sum().clamp_min(1e-8)).clamp_min(1e-12).log()).sum()
        budget_entropy = (budget_entropy / np.log(max(k, 2))).item()
        mask = obj[:, None] < obj[None, :]
        consistency = float("nan")
        if mask.any():
            w, l = mask.nonzero(as_tuple=True)
            consistency = (upd[w] > upd[l]).float().mean().item()
        rows.append(
            {
                "separation": float(separation),
                "support": float(support),
                "budget_entropy": float(budget_entropy),
                "consistency": float(consistency),
            }
        )
    if method in {"Loss-only", "Loss+Weighting", "PO", "BOPO", "SLL"}:
        pref = _implicit_pref_for_method(method, problem, fc, pairs)
        weight_stats = _weight_stats_for_pref(pref, objective, method)
    else:
        weight_stats = {"weight_ess": 1.0, "top10_mass": 0.10, "hi_clip_share": 0.0, "lo_clip_share": 0.0}
    for row in rows:
        row["weight_ess"] = float(weight_stats["weight_ess"])
        row["top10_weight_mass"] = float(weight_stats["top10_mass"])
        row["clip_share"] = float(weight_stats["hi_clip_share"] + weight_stats["lo_clip_share"])
    return rows


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
    profile_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    for problem in problems:
        print(f"[problem] {problem}", flush=True)
        for scale_name, scale in [("source", source_scale), ("shifted", shifted_scale)]:
            target_spec = _target_spec(specs[problem], problem, scale)
            target_size = int(target_spec.hf.train_problem_size)
            print(f"[scale] {problem} {scale_name} target_size={target_size}", flush=True)
            caches = rollout_feature_caches(target_spec, seed=seed + int(round(scale * 1000)), device=torch.device(device))
            accum_signed: dict[tuple[str, int], list[float]] = {}
            accum_budget: dict[tuple[str, int], list[float]] = {}
            for cache_id, fc in enumerate(caches):
                for method in METHODS:
                    objective, update = _update_for_method(
                        method=method,
                        problem=problem,
                        fc=fc,
                        pairs=pair_cache[problem],
                        state=state,
                        sharpness=sharpness,
                    )
                    signed, budget = _profile_rows(objective, update, bins=bins)
                    for r in signed:
                        accum_signed.setdefault((method, int(r["rank_bin"])), []).append(float(r["value"]))
                    for r in budget:
                        accum_budget.setdefault((method, int(r["rank_bin"])), []).append(float(r["value"]))
                    for inst_id, row in enumerate(_instance_rows(objective, update, method, problem, fc, pair_cache[problem])):
                        instance_rows.append(
                            {
                                "problem": problem,
                                "scale_name": scale_name,
                                "scale": float(scale),
                                "target_size": target_size,
                                "method": method,
                                "cache_id": cache_id,
                                "instance_id": inst_id,
                                **row,
                            }
                        )
            for method in METHODS:
                for bin_id in range(bins):
                    profile_rows.append(
                        {
                            "problem": problem,
                            "scale_name": scale_name,
                            "scale": float(scale),
                            "target_size": target_size,
                            "method": method,
                            "rank_bin": bin_id,
                            "rank_center": (bin_id + 0.5) / bins,
                            "signed_update": float(np.mean(accum_signed[(method, bin_id)])),
                            "update_budget": float(np.mean(accum_budget[(method, bin_id)])),
                        }
                    )
    return {
        "problems": problems,
        "source_scale": source_scale,
        "shifted_scale": shifted_scale,
        "bins": bins,
        "profiles": profile_rows,
        "instances": instance_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _matrix(rows: list[dict[str, Any]], problem: str, scale_name: str, metric: str, *, bins: int) -> np.ndarray:
    mat = np.full((len(METHODS), bins), np.nan)
    for i, method in enumerate(METHODS):
        for r in rows:
            if r["problem"] == problem and r["scale_name"] == scale_name and r["method"] == method:
                mat[i, int(r["rank_bin"])] = float(r[metric])
    return mat


def _mean(rows: list[dict[str, Any]], problem: str, scale_name: str, method: str, metric: str) -> float:
    vals = [
        float(r[metric])
        for r in rows
        if r["problem"] == problem and r["scale_name"] == scale_name and r["method"] == method
    ]
    return float(np.nanmean(vals)) if vals else float("nan")


def plot(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.7,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    problems = results["problems"]
    bins = int(results["bins"])
    profiles = results["profiles"]
    instances = results["instances"]

    def heatmap_grid(scale_name: str, metric: str, fname: str, title: str, cmap: str, *, center_zero: bool) -> None:
        mats = [_matrix(profiles, p, scale_name, metric, bins=bins) for p in problems]
        vals = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
        if center_zero:
            vmax = max(float(np.nanpercentile(np.abs(vals), 96)), 1e-6)
            vmin = -vmax
        else:
            vmin = 0.0
            vmax = max(float(np.nanpercentile(vals, 98)), 1e-6)
        fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 3.1), constrained_layout=True)
        if len(problems) == 1:
            axes = [axes]
        for ax, problem, mat in zip(axes, problems, mats):
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(PROBLEM_LABELS[problem])
            ax.set_yticks(np.arange(len(METHODS)), METHODS)
            ax.set_xticks([0, bins // 2, bins - 1], ["best", "mid", "worst"])
            ax.set_xlabel("solution rank")
        axes[0].set_ylabel("method")
        fig.colorbar(im, ax=axes, shrink=0.78)
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    heatmap_grid("source", "update_budget", "01_source_update_budget_fingerprint", "Search scale: where does each method spend update budget?", "magma", center_zero=False)
    heatmap_grid("shifted", "update_budget", "02_shifted_update_budget_fingerprint", "Generalization scale: update-budget behavior fingerprint", "magma", center_zero=False)

    drift_mats: list[np.ndarray] = []
    for problem in problems:
        src = _matrix(profiles, problem, "source", "update_budget", bins=bins)
        shifted = _matrix(profiles, problem, "shifted", "update_budget", bins=bins)
        drift_mats.append(shifted - src)
    vals = np.concatenate([m[np.isfinite(m)] for m in drift_mats if np.isfinite(m).any()])
    vmax = max(float(np.nanpercentile(np.abs(vals), 96)), 1e-6)
    fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 3.1), constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem, mat in zip(axes, problems, drift_mats):
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_yticks(np.arange(len(METHODS)), METHODS)
        ax.set_xticks([0, bins // 2, bins - 1], ["best", "mid", "worst"])
        ax.set_xlabel("solution rank")
    axes[0].set_ylabel("method")
    fig.colorbar(im, ax=axes, shrink=0.78, label="shifted budget - source budget")
    fig.suptitle("Behavior drift: how update budget moves under scale shift", fontweight="bold")
    fig.savefig(out_dir / "03_budget_drift_source_to_shift.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_budget_drift_source_to_shift.pdf", bbox_inches="tight")
    plt.close(fig)

    metrics = ["separation", "consistency", "support", "budget_entropy", "weight_ess", "top10_weight_mass", "clip_share"]
    for scale_name, fname, title in [
        ("source", "04_source_metric_fingerprint", "Search scale behavioral metrics"),
        ("shifted", "05_shifted_metric_fingerprint", "Generalization scale behavioral metrics"),
    ]:
        rows = []
        for method in METHODS:
            vals = []
            for metric in metrics:
                vals.append(np.nanmean([_mean(instances, p, scale_name, method, metric) for p in problems]))
            rows.append(vals)
        mat = np.asarray(rows, dtype=np.float64)
        # Column-normalize for visual contrast; raw values are in CSV.
        norm = (mat - np.nanmean(mat, axis=0, keepdims=True)) / (np.nanstd(mat, axis=0, keepdims=True) + 1e-8)
        fig, ax = plt.subplots(figsize=(8.2, 3.0), constrained_layout=True)
        im = ax.imshow(norm, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)
        ax.set_yticks(np.arange(len(METHODS)), METHODS)
        ax.set_xticks(np.arange(len(metrics)), metrics, rotation=25, ha="right")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7.5)
        fig.colorbar(im, ax=ax, shrink=0.82, label="column z-score")
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Source vs Generalization Behavior Fingerprint\n\n"
            "This figure set compares RL, PO, BOPO, SLL, Loss-only, and Loss+Weighting on the searched problem at source scale and shifted scale.\n"
            "The main diagnostic is not final performance; it is where each objective-induced loss spends equal-norm update budget over solutions sorted by quality.\n"
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
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "source_shift_behavior_fingerprint" / stamp))
    results = collect(
        problems=problems,
        source_scale=float(args.source_scale),
        shifted_scale=float(args.shifted_scale),
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        state=str(args.state),
        sharpness=float(args.sharpness),
        bins=max(int(args.bins), 4),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(results, out_dir)
    _write_csv(out_dir / "behavior_profiles.csv", results["profiles"])
    _write_csv(out_dir / "behavior_instance_metrics.csv", results["instances"])
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
