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
    LOSS_ONLY,
    WEIGHTING,
    PROBLEM_LABELS,
    _ensure_jssp_data,
    _load_pair,
    _state_log_prob,
    _target_spec,
)
from scripts.plot_causal_mechanism_loss_weighting import _pair_influence_proxy  # noqa: E402


METHODS = ["Loss-only", "Loss+Weighting"]
COLORS = {"Loss-only": "#0072B2", "Loss+Weighting": "#D55E00"}


def _bin_mass(values: torch.Tensor, mass: torch.Tensor, bins: int) -> torch.Tensor:
    idx = torch.clamp((values.clamp(0.0, 1.0) * bins).long(), 0, bins - 1)
    out = torch.zeros((bins,), device=values.device, dtype=torch.float32)
    out.scatter_add_(0, idx, mass.float())
    return out / out.sum().clamp_min(1e-12)


def _availability_distribution(objective: torch.Tensor, bins: int) -> torch.Tensor:
    mask = objective[:, :, None] < objective[:, None, :]
    b, w, l = mask.nonzero(as_tuple=True)
    gap = objective[b, l] - objective[b, w]
    obj_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
    gap_norm = (gap / obj_range[b]).clamp(0.0, 1.0)
    mass = torch.ones_like(gap_norm, dtype=torch.float32)
    return _bin_mass(gap_norm, mass, bins)


def _influence_distribution(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, Any],
    state: str,
    sharpness: float,
    bins: int,
) -> torch.Tensor:
    objective = fc["objective"].detach()
    pair_out = _pair_influence_proxy(
        method=method,
        problem=problem,
        fc=fc,
        pairs=pairs,
        state=state,
        sharpness=sharpness,
    )
    if pair_out is None:
        return torch.full((bins,), float("nan"), device=objective.device)
    b, w, l, influence = pair_out
    gap = objective[b, l] - objective[b, w]
    obj_range = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-12)
    gap_norm = (gap / obj_range[b]).clamp(0.0, 1.0)
    return _bin_mass(gap_norm, influence.clamp_min(0.0), bins)


def _metrics_from_distributions(avail: torch.Tensor, infl: torch.Tensor) -> dict[str, float]:
    eps = 1e-12
    tv = 0.5 * (infl - avail).abs().sum()
    cdf_avail = torch.cumsum(avail, dim=0)
    cdf_infl = torch.cumsum(infl, dim=0)
    emd = (cdf_infl - cdf_avail).abs().mean()
    near = max(int(np.ceil(0.15 * int(avail.numel()))), 1)
    small_avail = avail[:near].sum()
    small_infl = infl[:near].sum()
    large_avail = avail[-near:].sum()
    large_infl = infl[-near:].sum()
    entropy = -(infl * infl.clamp_min(eps).log()).sum() / np.log(max(int(infl.numel()), 2))
    return {
        "coverage_tv": float(tv.item()),
        "coverage_emd": float(emd.item()),
        "small_gap_coverage_ratio": float((small_infl / small_avail.clamp_min(eps)).item()),
        "large_gap_coverage_ratio": float((large_infl / large_avail.clamp_min(eps)).item()),
        "influence_entropy": float(entropy.item()),
        "small_gap_available_mass": float(small_avail.item()),
        "small_gap_influence_mass": float(small_infl.item()),
        "large_gap_influence_mass": float(large_infl.item()),
    }


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
    rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for problem in problems:
        print(f"[problem] {problem}", flush=True)
        for scale_name, scale in [("source", source_scale), ("generalization", shifted_scale)]:
            target_spec = _target_spec(specs[problem], problem, scale)
            target_size = int(target_spec.hf.train_problem_size)
            print(f"[scale] {problem} {scale_name} target_size={target_size}", flush=True)
            caches = rollout_feature_caches(target_spec, seed=seed + int(round(scale * 1000)), device=torch.device(device))
            for cache_id, fc in enumerate(caches):
                objective = fc["objective"].detach()
                avail = _availability_distribution(objective, bins)
                for method in METHODS:
                    infl = _influence_distribution(
                        method=method,
                        problem=problem,
                        fc=fc,
                        pairs=pair_cache[problem],
                        state=state,
                        sharpness=sharpness,
                        bins=bins,
                    )
                    metrics = _metrics_from_distributions(avail, infl)
                    rows.append(
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
                    for b in range(bins):
                        spectrum_rows.append(
                            {
                                "problem": problem,
                                "scale_name": scale_name,
                                "scale": float(scale),
                                "target_size": target_size,
                                "method": method,
                                "cache_id": cache_id,
                                "bin": b,
                                "gap_center": (b + 0.5) / bins,
                                "available_mass": float(avail[b].detach().cpu().item()),
                                "influence_mass": float(infl[b].detach().cpu().item()),
                                "coverage_ratio": float((infl[b] / avail[b].clamp_min(1e-12)).detach().cpu().item()),
                            }
                        )
    return {"problems": problems, "bins": bins, "metrics": rows, "spectrum": spectrum_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            "legend.fontsize": 8,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    problems = results["problems"]
    metrics = results["metrics"]
    spectrum = results["spectrum"]

    def mean_metric(problem: str, scale_name: str, method: str, metric: str) -> float:
        vals = [
            float(r[metric])
            for r in metrics
            if r["problem"] == problem and r["scale_name"] == scale_name and r["method"] == method
        ]
        return float(np.mean(vals)) if vals else float("nan")

    drift_rows = []
    for problem in problems:
        cache_ids = sorted({int(r["cache_id"]) for r in spectrum if r["problem"] == problem})
        for method in METHODS:
            for cache_id in cache_ids:
                src = np.asarray(
                    [
                        float(r["influence_mass"])
                        for r in sorted(
                            [
                                r
                                for r in spectrum
                                if r["problem"] == problem
                                and r["scale_name"] == "source"
                                and r["method"] == method
                                and int(r["cache_id"]) == cache_id
                            ],
                            key=lambda x: int(x["bin"]),
                        )
                    ],
                    dtype=float,
                )
                gen = np.asarray(
                    [
                        float(r["influence_mass"])
                        for r in sorted(
                            [
                                r
                                for r in spectrum
                                if r["problem"] == problem
                                and r["scale_name"] == "generalization"
                                and r["method"] == method
                                and int(r["cache_id"]) == cache_id
                            ],
                            key=lambda x: int(x["bin"]),
                        )
                    ],
                    dtype=float,
                )
                if src.size == 0 or gen.size == 0 or src.size != gen.size:
                    continue
                drift_rows.append(
                    {
                        "problem": problem,
                        "method": method,
                        "cache_id": cache_id,
                        "effective_gap_spectrum_drift": float(0.5 * np.abs(gen - src).sum()),
                    }
                )
    _write_csv(out_dir / "effective_gap_spectrum_drift.csv", drift_rows)

    # Primary mechanism figure: source-to-generalization behavior drift.
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.35), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(problems))
    width = 0.34
    for offset, method in [(-width / 2, "Loss-only"), (width / 2, "Loss+Weighting")]:
        vals = [
            float(np.mean([r["effective_gap_spectrum_drift"] for r in drift_rows if r["problem"] == p and r["method"] == method]))
            for p in problems
        ]
        ax.bar(x + offset, vals, width=width, color=COLORS[method], label=method, alpha=0.92)
        for xi, val in zip(x + offset, vals):
            ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(x, [PROBLEM_LABELS[p] for p in problems], rotation=20, ha="right")
    ax.set_ylabel("source-to-generalization drift")
    ax.set_title("Effective gap-spectrum drift", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    diffs = []
    labels = []
    for p in problems:
        lw = np.mean([r["effective_gap_spectrum_drift"] for r in drift_rows if r["problem"] == p and r["method"] == "Loss+Weighting"])
        lo = np.mean([r["effective_gap_spectrum_drift"] for r in drift_rows if r["problem"] == p and r["method"] == "Loss-only"])
        diffs.append(float(lw - lo))
        labels.append(PROBLEM_LABELS[p])
    ax.bar(np.arange(len(problems)), diffs, color=["#B85C00" if d >= 0 else "#0072B2" for d in diffs], width=0.62)
    for xi, val in enumerate(diffs):
        ax.text(xi, val, f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.6)
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.set_xticks(np.arange(len(problems)), labels, rotation=20, ha="right")
    ax.set_ylabel("Weighting drift - Loss-only drift")
    ax.set_title("Extra drift caused by weighting", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Why Loss-only generalizes better: weighting changes the effective loss behavior more under scale shift", fontweight="bold")
    fig.savefig(out_dir / "00_effective_gap_spectrum_drift.png", bbox_inches="tight")
    fig.savefig(out_dir / "00_effective_gap_spectrum_drift.pdf", bbox_inches="tight")
    plt.close(fig)

    # Main figure: coverage mismatch paired across problems.
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), constrained_layout=True)
    main_metrics = [
        ("coverage_tv", "coverage mismatch\nTV distance"),
        ("small_gap_coverage_ratio", "small-gap coverage\ninfluence / availability"),
        ("large_gap_coverage_ratio", "large-gap coverage\ninfluence / availability"),
    ]
    for ax, (metric, ylabel) in zip(axes, main_metrics):
        x = np.arange(len(problems))
        width = 0.34
        for offset, method in [(-width / 2, "Loss-only"), (width / 2, "Loss+Weighting")]:
            vals = [mean_metric(p, "generalization", method, metric) for p in problems]
            ax.bar(x + offset, vals, width=width, color=COLORS[method], label=method, alpha=0.9)
            for xi, val in zip(x + offset, vals):
                ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, [PROBLEM_LABELS[p] for p in problems], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if metric == "small_gap_coverage_ratio":
            ax.axhline(1.0, color="#333333", lw=0.8, ls="--", alpha=0.6)
        if metric == "large_gap_coverage_ratio":
            ax.axhline(1.0, color="#333333", lw=0.8, ls="--", alpha=0.6)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Generalization failure mechanism: weighting distorts pair-gap coverage", fontweight="bold")
    fig.savefig(out_dir / "01_generalization_coverage_mismatch.png", bbox_inches="tight")
    fig.savefig(out_dir / "01_generalization_coverage_mismatch.pdf", bbox_inches="tight")
    plt.close(fig)

    # Gap-spectrum curves at generalization scale.
    fig, axes = plt.subplots(1, len(problems), figsize=(3.05 * len(problems), 2.85), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        rows = [r for r in spectrum if r["problem"] == problem and r["scale_name"] == "generalization"]
        avail = []
        xs = []
        for b in sorted({int(r["bin"]) for r in rows}):
            bin_rows = [r for r in rows if int(r["bin"]) == b]
            xs.append(float(bin_rows[0]["gap_center"]))
            avail.append(float(np.mean([float(r["available_mass"]) for r in bin_rows])))
        ax.plot(xs, avail, color="#333333", lw=2.0, ls="--", label="available pairs")
        for method in METHODS:
            vals = []
            for b in sorted({int(r["bin"]) for r in rows}):
                vals.append(float(np.mean([float(r["influence_mass"]) for r in rows if int(r["bin"]) == b and r["method"] == method])))
            ax.plot(xs, vals, color=COLORS[method], lw=2.1, marker="o", ms=3, label=method)
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xlabel("normalized objective gap")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mass")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("At generalization scale, compare available pair gaps with effective loss influence", fontweight="bold")
    fig.savefig(out_dir / "02_generalization_gap_spectrum.png", bbox_inches="tight")
    fig.savefig(out_dir / "02_generalization_gap_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)

    # Source-to-generalization change in mismatch.
    fig, ax = plt.subplots(figsize=(7.2, 3.25), constrained_layout=True)
    x = np.arange(len(problems))
    width = 0.34
    for offset, method in [(-width / 2, "Loss-only"), (width / 2, "Loss+Weighting")]:
        vals = [
            mean_metric(p, "generalization", method, "coverage_tv") - mean_metric(p, "source", method, "coverage_tv")
            for p in problems
        ]
        ax.bar(x + offset, vals, width=width, color=COLORS[method], label=method, alpha=0.9)
        for xi, val in zip(x + offset, vals):
            ax.text(xi, val, f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5)
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.set_xticks(x, [PROBLEM_LABELS[p] for p in problems], rotation=20, ha="right")
    ax.set_ylabel("gen mismatch - source mismatch")
    ax.set_title("Does the coverage distortion get worse under scale shift?", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "03_scale_shift_mismatch_delta.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_scale_shift_mismatch_delta.pdf", bbox_inches="tight")
    plt.close(fig)

    # Aggregate only for a compact paper inset.
    agg_rows = []
    for scale_name in ["source", "generalization"]:
        for method in METHODS:
            subset = [r for r in metrics if r["scale_name"] == scale_name and r["method"] == method]
            agg_rows.append(
                {
                    "scale_name": scale_name,
                    "method": method,
                    "coverage_tv": float(np.mean([float(r["coverage_tv"]) for r in subset])),
                    "small_gap_coverage_ratio": float(np.mean([float(r["small_gap_coverage_ratio"]) for r in subset])),
                    "large_gap_coverage_ratio": float(np.mean([float(r["large_gap_coverage_ratio"]) for r in subset])),
                    "influence_entropy": float(np.mean([float(r["influence_entropy"]) for r in subset])),
                }
            )
    _write_csv(out_dir / "coverage_mismatch_aggregate.csv", agg_rows)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Loss-only vs Loss+Weighting coverage mismatch\n\n"
            "This analysis only compares Loss-only and Loss+Weighting. It tests whether weighting changes the effective objective-gap influence under scale generalization.\n\n"
            "- Figure 00 is the main figure: source-to-generalization drift of the effective gap spectrum.\n"
            "- Figure 01 is diagnostic only: TV mismatch to the available pair-gap distribution, small-gap coverage, and large-gap coverage.\n"
            "- Figure 02 is diagnostic only: actual gap-spectrum distributions at the generalization scale.\n"
            "- Figure 03 is diagnostic only: whether the availability mismatch increases from source scale to generalization scale.\n"
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
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "loss_only_vs_weighting_coverage" / stamp))
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
    _write_csv(out_dir / "coverage_mismatch_metrics.csv", results["metrics"])
    _write_csv(out_dir / "gap_spectrum.csv", results["spectrum"])
    plot(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
