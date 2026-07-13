from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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

from scripts.plot_loss_fine_grained_signature import (  # noqa: E402
    BASELINES,
    FEATURES,
    PLOT_COLORS,
    _collect_one_problem,
    _gap_matched_baseline,
    _norm_mean,
    _rmse_curve,
    _safe_corr,
)
from scripts.plot_scale_generalization_loss_weighting import (  # noqa: E402
    LOSS_ONLY,
    PROBLEM_LABELS,
    _ensure_jssp_data,
)


CLAIM_FEATURE = {
    "tsp100": ("relative_gap", "relative objective gap"),
    "cvrp100": ("rank_diff_norm", "rank-distance"),
    "ffsp100": ("pool_range", "candidate-pool range"),
    "jssp10x10": ("pool_std", "candidate-pool std"),
}

SHORT_LOSS_LABEL = {
    "tsp100": "cost-gap normalized pair margin",
    "cvrp100": "rank-adaptive biased margin",
    "ffsp100": "exp advantage-scale normalization",
    "jssp10x10": "L1 advantage-scale normalization",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _best_loss_name(problem: str) -> str:
    payload = json.loads((REPO_ROOT / LOSS_ONLY[problem]).read_text(encoding="utf-8"))
    return str(payload.get("f_ir", {}).get("name", ""))


def _quantile_curve(x: np.ndarray, y: np.ndarray, *, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 0:
        return np.asarray([]), np.asarray([])
    x = x[mask]
    y = _norm_mean(y[mask])
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    if np.unique(edges).size < 3:
        edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x) + 1e-9), bins + 1)
    xs: list[float] = []
    ys: list[float] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        pick = (x >= lo) & (x <= hi if i == bins - 1 else x < hi)
        if not np.any(pick):
            continue
        xs.append(float(np.nanmean(x[pick])))
        ys.append(float(np.nanmean(y[pick])))
    return np.asarray(xs), np.asarray(ys)


def _collect_evidence(
    *,
    problems: list[str],
    batches: int,
    seed: int,
    device: str,
    state: str,
    sharpness: float,
    max_pairs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if "jssp10x10" in problems:
        _ensure_jssp_data(REPO_ROOT)
    evidence: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for problem in problems:
        print(f"[evidence] {problem}", flush=True)
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
        loss_coeff = coeffs["Loss-only"]
        rel_gap = feats["relative_gap"]
        rmses = {m: _rmse_curve(loss_coeff, coeffs[m], rel_gap) for m in BASELINES}
        nearest = min(rmses, key=lambda m: (float("inf") if not np.isfinite(rmses[m]) else rmses[m]))

        feature_key, feature_label = CLAIM_FEATURE[problem]
        fx, fy = _quantile_curve(feats[feature_key], loss_coeff, bins=8)
        gx, gy = _quantile_curve(rel_gap, loss_coeff, bins=10)
        bx, by = _quantile_curve(rel_gap, coeffs[nearest], bins=10)

        matched, _ = _gap_matched_baseline(loss_coeff, coeffs[nearest], rel_gap, n_bins=8)
        residual = np.log2((_norm_mean(loss_coeff) + 1e-8) / (matched + 1e-8))
        residual_corrs = {
            feature: _safe_corr(residual, feats[feature])
            for feature, _ in FEATURES
        }
        best_residual_feature = max(
            residual_corrs,
            key=lambda k: -1.0 if not np.isfinite(residual_corrs[k]) else abs(float(residual_corrs[k])),
        )

        scale_by_method: dict[str, list[tuple[float, float]]] = {}
        for row in data["scale_sensitivity"]:
            method = str(row["method"])
            scale_by_method.setdefault(method, []).append((float(row["objective_scale"]), float(row["mean_coefficient"])))
        for vals in scale_by_method.values():
            vals.sort()

        def scale_ratio(method: str) -> float:
            vals = dict(scale_by_method.get(method, []))
            return float(vals.get(4.0, np.nan) / max(vals.get(0.25, np.nan), 1e-12))

        evidence[problem] = {
            "best_loss_name": _best_loss_name(problem),
            "nearest": nearest,
            "rmses": rmses,
            "gap_curve": {"Loss-only": (gx, gy), nearest: (bx, by)},
            "feature_curve": (fx, fy, feature_key, feature_label),
            "scale": {
                "Loss-only": scale_by_method.get("Loss-only", []),
                nearest: scale_by_method.get(nearest, []),
            },
            "residual_corrs": residual_corrs,
            "best_residual_feature": best_residual_feature,
        }
        summary_rows.append(
            {
                "problem": problem,
                "best_loss": evidence[problem]["best_loss_name"],
                "nearest_baseline": nearest,
                "nearest_rmse": rmses[nearest],
                "claim_feature": feature_key,
                "loss_only_scale4_over_scale025": scale_ratio("Loss-only"),
                "nearest_scale4_over_scale025": scale_ratio(nearest),
                "best_residual_feature": best_residual_feature,
                "best_residual_corr": residual_corrs[best_residual_feature],
            }
        )
    return evidence, summary_rows


def _plot(evidence: dict[str, Any], problems: list[str], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.3,
            "legend.fontsize": 7.2,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )

    fig, axes = plt.subplots(4, len(problems), figsize=(3.0 * len(problems), 9.4), constrained_layout=True)
    if len(problems) == 1:
        axes = axes.reshape(4, 1)

    for ci, problem in enumerate(problems):
        item = evidence[problem]
        label = PROBLEM_LABELS[problem].replace("JSSP10x10", "JSSP")
        nearest = item["nearest"]
        title = SHORT_LOSS_LABEL[problem]
        axes[0, ci].set_title(f"{label}\n{title}", fontsize=8.6)

        ax = axes[0, ci]
        for method, (xs, ys) in item["gap_curve"].items():
            ax.plot(xs, ys, marker="o", ms=2.7, lw=1.45, color=PLOT_COLORS.get(method, "#555555"), label=method)
        ax.set_xlabel("relative objective gap")
        if ci == 0:
            ax.set_ylabel("norm. coeff.\nvs gap")
        ax.legend(frameon=False, loc="best")

        ax = axes[1, ci]
        xs, ys, feature_key, feature_label = item["feature_curve"]
        ax.plot(xs, ys, marker="o", ms=2.8, lw=1.55, color=PLOT_COLORS["Loss-only"])
        ax.set_xlabel(feature_label)
        if ci == 0:
            ax.set_ylabel("best-loss coeff.\nvs claim feature")
        ax.text(0.02, 0.95, feature_key, transform=ax.transAxes, va="top", ha="left", fontsize=7.0)

        ax = axes[2, ci]
        for method, vals in item["scale"].items():
            if not vals:
                continue
            xs = np.asarray([v[0] for v in vals], dtype=np.float64)
            ys = np.asarray([v[1] for v in vals], dtype=np.float64)
            base = ys[np.argmin(np.abs(xs - 1.0))]
            ys = ys / max(float(base), 1e-12)
            ax.plot(xs, ys, marker="o", ms=2.8, lw=1.45, color=PLOT_COLORS.get(method, "#555555"), label=method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("objective scale")
        if ci == 0:
            ax.set_ylabel("mean coeff.\nrelative to scale=1")
        ax.legend(frameon=False, loc="best")

        ax = axes[3, ci]
        features = [f for f, _ in FEATURES]
        vals = [item["residual_corrs"][f] for f in features]
        colors = ["#555555" if f != item["best_residual_feature"] else "#C44E52" for f in features]
        ax.bar(np.arange(len(features)), vals, color=colors, width=0.72)
        ax.axhline(0.0, color="#333333", lw=0.75)
        ax.set_xticks(np.arange(len(features)), features, rotation=45, ha="right")
        ax.set_xlabel(f"residual corr after matching {nearest}")
        if ci == 0:
            ax.set_ylabel("corr(residual,\nfeature)")

    fig.suptitle("Evidence for per-problem best-loss mechanisms", fontweight="bold")
    fig.savefig(out_dir / "best_loss_mechanism_evidence.png", bbox_inches="tight")
    fig.savefig(out_dir / "best_loss_mechanism_evidence.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--state", default="aligned", choices=["sampled", "aligned", "misaligned"])
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "figures" / "loss_fine_grained_signature" / "20260501-final-all-methods" / "best_loss_mechanism_evidence"),
    )
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    evidence, rows = _collect_evidence(
        problems=problems,
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        state=str(args.state),
        sharpness=float(args.sharpness),
        max_pairs=max(int(args.max_pairs), 256),
    )
    out_dir = Path(args.out_dir)
    _plot(evidence, problems, out_dir)
    _write_csv(out_dir / "best_loss_mechanism_evidence_summary.csv", rows)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
