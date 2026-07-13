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
    COLORS,
    METHODS,
    PROBLEM_LABELS,
    PROBLEMS,
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
    loss = _method_loss(method, problem, fc, lp0, pairs)
    loss.backward()
    update = -lp0.grad.detach()
    update = update - update.mean(dim=1, keepdim=True)
    update = update / update.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return objective, update


def _rank_bin_records(objective: torch.Tensor, update: torch.Tensor, *, bins: int) -> list[tuple[int, float]]:
    sorted_idx = objective.sort(dim=1, descending=False).indices
    sorted_update = update.gather(1, sorted_idx)
    k = int(sorted_update.shape[1])
    records: list[tuple[int, float]] = []
    for rank in range(k):
        bin_id = min(int(rank * bins / max(k, 1)), bins - 1)
        vals = sorted_update[:, rank].detach().cpu().numpy().reshape(-1)
        for val in vals:
            records.append((bin_id, float(val)))
    return records


def _instance_metrics(
    objective: torch.Tensor,
    update: torch.Tensor,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    batch, k = objective.shape
    rows: list[dict[str, float]] = []
    for b in range(batch):
        obj = objective[b]
        upd = update[b]
        order = obj.sort(descending=False).indices
        top_n = max(int(np.ceil(0.10 * k)), 1)
        bot_n = max(int(np.ceil(0.10 * k)), 1)
        top_update = upd[order[:top_n]].mean()
        bottom_update = upd[order[-bot_n:]].mean()
        abs_update = upd.abs()
        support = (abs_update.sum() ** 2 / (abs_update.square().sum().clamp_min(1e-12) * k)).item()
        mask = obj[:, None] < obj[None, :]
        if mask.any():
            b_idx, l_idx = mask.nonzero(as_tuple=True)
            consistency = (upd[b_idx] > upd[l_idx]).float().mean().item()
        else:
            consistency = float("nan")
        rows.append(
            {
                "separation": float((top_update - bottom_update).item()),
                "top_update": float(top_update.item()),
                "bottom_update": float(bottom_update.item()),
                "pairwise_consistency": float(consistency),
                "update_support": float(support),
            }
        )
    pref = _implicit_pref_for_method(method, problem, fc, pairs)
    weight_stats = _weight_stats_for_pref(pref, objective, method)
    return rows, weight_stats


def collect(
    *,
    problems: list[str],
    scales: list[float],
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
    rank_values: dict[tuple[str, float, str, int], list[float]] = {}
    instance_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for problem in problems:
        print(f"[problem] {problem}", flush=True)
        for scale in scales:
            target_spec = _target_spec(specs[problem], problem, scale)
            target_size = int(target_spec.hf.train_problem_size)
            print(f"[scale] {problem} target_size={target_size} ratio={scale}", flush=True)
            caches = rollout_feature_caches(target_spec, seed=seed + int(round(scale * 1000)), device=torch.device(device))
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
                    for bin_id, value in _rank_bin_records(objective, update, bins=bins):
                        rank_values.setdefault((problem, float(scale), method, int(bin_id)), []).append(value)
                    per_inst, weight_stats = _instance_metrics(objective, update, method, problem, fc, pair_cache[problem])
                    for local_id, row in enumerate(per_inst):
                        instance_rows.append(
                            {
                                "problem": problem,
                                "method": method,
                                "scale": float(scale),
                                "target_size": target_size,
                                "cache_id": cache_id,
                                "instance_id": local_id,
                                **row,
                                "weight_ess": float(weight_stats["weight_ess"]),
                                "top10_mass": float(weight_stats["top10_mass"]),
                                "clip_share": float(weight_stats["hi_clip_share"] + weight_stats["lo_clip_share"]),
                            }
                        )

    for (problem, scale, method, bin_id), values in rank_values.items():
        summary_rows.append(
            {
                "problem": problem,
                "method": method,
                "scale": scale,
                "rank_bin": bin_id,
                "rank_bin_center": (bin_id + 0.5) / bins,
                "mean_update": float(np.mean(values)),
                "std_update": float(np.std(values)),
                "n": len(values),
            }
        )
    return {
        "problems": problems,
        "scales": scales,
        "bins": bins,
        "state": state,
        "sharpness": sharpness,
        "rank_rows": summary_rows,
        "instance_rows": instance_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_instance(rows: list[dict[str, Any]], problem: str, method: str, scale: float, metric: str) -> float:
    vals = [
        float(r[metric])
        for r in rows
        if r["problem"] == problem and r["method"] == method and abs(float(r["scale"]) - float(scale)) < 1e-12
    ]
    return float(np.nanmean(vals)) if vals else float("nan")


def _rank_matrix(rank_rows: list[dict[str, Any]], problem: str, scale: float, *, bins: int) -> np.ndarray:
    mat = np.full((len(METHODS), bins), np.nan, dtype=np.float64)
    for i, method in enumerate(METHODS):
        for r in rank_rows:
            if r["problem"] == problem and r["method"] == method and abs(float(r["scale"]) - scale) < 1e-12:
                mat[i, int(r["rank_bin"])] = float(r["mean_update"])
    return mat


def plot(results: dict[str, Any], out_dir: Path) -> None:
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
            "legend.fontsize": 7.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    problems = results["problems"]
    scales = [float(s) for s in results["scales"]]
    bins = int(results["bins"])
    rank_rows = results["rank_rows"]
    instance_rows = results["instance_rows"]
    source_scale = 1.0
    shifted_scale = max(scales)

    def heatmap_for_scale(scale: float, fname: str, title: str) -> None:
        mats = [_rank_matrix(rank_rows, p, scale, bins=bins) for p in problems]
        finite = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
        vmax = float(np.nanpercentile(np.abs(finite), 96)) if finite.size else 1.0
        vmax = max(vmax, 1e-6)
        fig, axes = plt.subplots(1, len(problems), figsize=(3.2 * len(problems), 2.75), constrained_layout=True)
        if len(problems) == 1:
            axes = [axes]
        for ax, problem, mat in zip(axes, problems, mats):
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(PROBLEM_LABELS[problem])
            ax.set_yticks(np.arange(len(METHODS)), METHODS)
            ax.set_xticks([0, bins // 2, bins - 1], ["best", "mid", "worst"])
            ax.set_xlabel("solutions sorted by objective")
        axes[0].set_ylabel("method")
        fig.colorbar(im, ax=axes, shrink=0.78, label="equal-norm update to log-prob")
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    heatmap_for_scale(source_scale, "01_ranked_update_heatmap_source", "PO4COPS-style update separation at source scale")
    heatmap_for_scale(shifted_scale, "02_ranked_update_heatmap_shifted", "PO4COPS-style update separation after scale shift")

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 2.85), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    x = np.arange(len(METHODS))
    for ax, problem in zip(axes, problems):
        vals = [_mean_instance(instance_rows, problem, m, shifted_scale, "pairwise_consistency") for m in METHODS]
        src = [_mean_instance(instance_rows, problem, m, source_scale, "pairwise_consistency") for m in METHODS]
        ax.bar(x, vals, color=[COLORS[m] for m in METHODS], width=0.72)
        ax.scatter(x, src, marker="D", s=24, color="#111111", label="source scale")
        ax.axhline(0.5, color="#333333", lw=0.9, ls="--")
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xticks(x, METHODS, rotation=30, ha="right")
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("pairwise update consistency")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Consistency: does the update rank better solutions above worse ones?", fontweight="bold")
    fig.savefig(out_dir / "03_pairwise_consistency_shifted_bar.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_pairwise_consistency_shifted_bar.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 3.0), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        data = [
            [
                float(r["separation"])
                for r in instance_rows
                if r["problem"] == problem and r["method"] == method and abs(float(r["scale"]) - shifted_scale) < 1e-12
            ]
            for method in METHODS
        ]
        parts = ax.violinplot(data, positions=np.arange(len(METHODS)), widths=0.72, showmeans=True, showextrema=False)
        for body, method in zip(parts["bodies"], METHODS):
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("none")
            body.set_alpha(0.72)
        parts["cmeans"].set_color("#111111")
        parts["cmeans"].set_linewidth(1.1)
        ax.axhline(0.0, color="#333333", lw=0.9)
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xticks(np.arange(len(METHODS)), METHODS, rotation=30, ha="right")
    axes[0].set_ylabel("top-10% update minus bottom-10% update")
    fig.suptitle("Advantage-scale distribution: instance-level update separation", fontweight="bold")
    fig.savefig(out_dir / "04_update_separation_distribution.png", bbox_inches="tight")
    fig.savefig(out_dir / "04_update_separation_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    def metric_heatmap(metric: str, fname: str, title: str, cbar_label: str, *, cmap: str = "viridis", reverse: bool = False) -> None:
        mat = np.asarray(
            [[_mean_instance(instance_rows, problem, method, shifted_scale, metric) for problem in problems] for method in METHODS],
            dtype=np.float64,
        )
        if reverse:
            cmap = f"{cmap}_r"
        fig, ax = plt.subplots(figsize=(1.05 * len(problems) + 3.2, 2.7), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(problems)), [PROBLEM_LABELS[p] for p in problems])
        ax.set_yticks(np.arange(len(METHODS)), METHODS)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if mat[i, j] < np.nanmean(mat) else "black")
        fig.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    metric_heatmap(
        "update_support",
        "05_update_support_heatmap",
        "Effective update support at shifted scale",
        "ESS over candidate solutions",
        cmap="magma",
    )
    metric_heatmap(
        "clip_share",
        "06_weight_saturation_heatmap",
        "Weighting-specific fragility: pair weights at clamp boundaries",
        "clipped pair-weight fraction",
        cmap="YlOrRd",
    )

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# PO4COPS-Style Loss/Weighting Mechanism Plots\n\n"
            "Inspired by PO4COPS Figure 3: ranked advantage separation, advantage-scale distributions, consistency, and entropy/support.\n"
            "Here the plotted quantity is the equal-norm loss update on solution log-probability, not final task performance.\n"
            "Source scale is ratio 1.0; shifted scale is the largest requested target/source ratio.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--scales", default="1.0,2.0")
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
    scales = [float(x) for x in str(args.scales).split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "po4cops_style_loss_weighting" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect(
        problems=problems,
        scales=scales,
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        state=str(args.state),
        sharpness=float(args.sharpness),
        bins=max(int(args.bins), 4),
    )
    plot(results, out_dir)
    _write_csv(out_dir / "ranked_update_bins.csv", results["rank_rows"])
    _write_csv(out_dir / "instance_update_metrics.csv", results["instance_rows"])
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
