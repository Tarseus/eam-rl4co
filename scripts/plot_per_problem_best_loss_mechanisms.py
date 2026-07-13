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


PROBLEM_MECHANISM = {
    "tsp100": "Cost-gap normalized pair margin",
    "cvrp100": "Rank-adaptive biased margin",
    "ffsp100": "Advantage-scale normalization",
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


def _loss_name(problem: str) -> str:
    payload = json.loads((REPO_ROOT / LOSS_ONLY[problem]).read_text(encoding="utf-8"))
    return str(payload.get("f_ir", {}).get("name", ""))


def _qcurve(x: np.ndarray, y: np.ndarray, *, bins: int = 8, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) <= 0:
        return np.asarray([]), np.asarray([])
    x = x[mask]
    y = y[mask]
    if normalize:
        y = _norm_mean(y)
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    if np.unique(edges).size < 3:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if abs(hi - lo) <= 1e-12:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
    xs: list[float] = []
    ys: list[float] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        pick = (x >= lo) & (x <= hi if i == bins - 1 else x < hi)
        if np.any(pick):
            xs.append(float(np.nanmean(x[pick])))
            ys.append(float(np.nanmean(y[pick])))
    return np.asarray(xs), np.asarray(ys)


def _ratio_curve(x: np.ndarray, num: np.ndarray, den: np.ndarray, *, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    n = _norm_mean(num)
    d = _norm_mean(den)
    return _qcurve(x, n / np.maximum(d, 1e-12), bins=bins, normalize=False)


def _scale_series(scale_rows: list[dict[str, Any]], method: str) -> tuple[np.ndarray, np.ndarray]:
    vals = sorted(
        (float(row["objective_scale"]), float(row["mean_coefficient"]))
        for row in scale_rows
        if str(row["method"]) == method
    )
    if not vals:
        return np.asarray([]), np.asarray([])
    xs = np.asarray([v[0] for v in vals], dtype=np.float64)
    ys = np.asarray([v[1] for v in vals], dtype=np.float64)
    base_idx = int(np.argmin(np.abs(xs - 1.0)))
    ys = ys / max(float(ys[base_idx]), 1e-12)
    return xs, ys


def _residual_corrs(loss_coeff: np.ndarray, base_coeff: np.ndarray, rel_gap: np.ndarray, feats: dict[str, np.ndarray]) -> dict[str, float]:
    matched, _ = _gap_matched_baseline(loss_coeff, base_coeff, rel_gap, n_bins=8)
    residual = np.log2((_norm_mean(loss_coeff) + 1e-8) / (matched + 1e-8))
    return {feature: _safe_corr(residual, feats[feature]) for feature, _ in FEATURES}


def _bar_residual_corr(ax: Any, corrs: dict[str, float], highlight: str, *, baseline: str) -> None:
    names = [f for f, _ in FEATURES]
    vals = [corrs[n] for n in names]
    colors = ["#525252" if n != highlight else "#C44E52" for n in names]
    ax.bar(np.arange(len(names)), vals, color=colors, width=0.72)
    ax.axhline(0.0, color="#333333", lw=0.75)
    ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    ax.set_ylabel("corr")
    ax.set_title(f"Residual after matching {baseline}")


def _plot_common_gap(ax: Any, data: dict[str, Any], methods: list[str]) -> None:
    feats = data["features"]
    coeffs = data["coeffs"]
    for method in methods:
        xs, ys = _qcurve(feats["relative_gap"], coeffs[method], bins=10)
        ax.plot(xs, ys, marker="o", ms=3.0, lw=1.55, color=PLOT_COLORS.get(method, "#555555"), label=method)
    ax.set_xlabel("relative objective gap")
    ax.set_ylabel("mean-normalized coefficient")
    ax.legend(frameon=False, loc="best")


def _plot_scale(ax: Any, scale_rows: list[dict[str, Any]], methods: list[str]) -> None:
    for method in methods:
        xs, ys = _scale_series(scale_rows, method)
        ax.plot(xs, ys, marker="o", ms=3.0, lw=1.55, color=PLOT_COLORS.get(method, "#555555"), label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("objective scale")
    ax.set_ylabel("coefficient / coefficient at scale=1")
    ax.legend(frameon=False, loc="best")


def _plot_tsp(problem: str, data: dict[str, Any], out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    feats = data["features"]
    coeffs = data["coeffs"]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.8), constrained_layout=True)
    _plot_common_gap(axes[0, 0], data, ["Loss-only", "SLL"])
    axes[0, 0].set_title("Best-loss concentrates on small cost gaps")

    xs, ys = _ratio_curve(feats["relative_gap"], coeffs["Loss-only"], coeffs["SLL"], bins=10)
    axes[0, 1].plot(xs, ys, marker="o", ms=3.0, lw=1.6, color=PLOT_COLORS["Loss-only"])
    axes[0, 1].axhline(1.0, color="#333333", lw=0.8)
    axes[0, 1].set_xlabel("relative objective gap")
    axes[0, 1].set_ylabel("Loss-only / SLL coefficient")
    axes[0, 1].set_title("Extra pressure is gap-dependent")

    _plot_scale(axes[1, 0], data["scale_sensitivity"], ["Loss-only", "SLL"])
    axes[1, 0].set_title("Scale invariance: objective rescaling barely moves it")

    corrs = _residual_corrs(coeffs["Loss-only"], coeffs["SLL"], feats["relative_gap"], feats)
    _bar_residual_corr(axes[1, 1], corrs, "margin", baseline="SLL")
    fig.suptitle(f"{PROBLEM_LABELS[problem]}: {PROBLEM_MECHANISM[problem]}", fontweight="bold")
    fig.savefig(out_dir / "tsp100_best_loss_mechanism.png", bbox_inches="tight")
    fig.savefig(out_dir / "tsp100_best_loss_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    rows.append({"problem": problem, "best_loss": _loss_name(problem), "mechanism": PROBLEM_MECHANISM[problem], "nearest": "SLL", "scale4_over_scale025": _scale_ratio(data, "Loss-only"), "main_residual_corr": corrs["margin"]})


def _plot_cvrp(problem: str, data: dict[str, Any], out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    feats = data["features"]
    coeffs = data["coeffs"]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.8), constrained_layout=True)
    _plot_common_gap(axes[0, 0], data, ["Loss-only", "SLL"])
    axes[0, 0].set_title("Gap response stays SLL-like")

    xs, ys = _qcurve(feats["rank_diff_norm"], coeffs["Loss-only"], bins=8)
    axes[0, 1].plot(xs, ys, marker="o", ms=3.0, lw=1.6, color=PLOT_COLORS["Loss-only"])
    axes[0, 1].set_xlabel("normalized rank distance")
    axes[0, 1].set_ylabel("mean-normalized coefficient")
    axes[0, 1].set_title("Rank-distance adaptive pressure")

    xs, ys = _ratio_curve(feats["rank_diff_norm"], coeffs["Loss-only"], coeffs["SLL"], bins=8)
    axes[1, 0].plot(xs, ys, marker="o", ms=3.0, lw=1.6, color=PLOT_COLORS["Loss-only"])
    axes[1, 0].axhline(1.0, color="#333333", lw=0.8)
    axes[1, 0].set_xlabel("normalized rank distance")
    axes[1, 0].set_ylabel("Loss-only / SLL coefficient")
    axes[1, 0].set_title("Rank term changes pair emphasis")

    corrs = _residual_corrs(coeffs["Loss-only"], coeffs["SLL"], feats["relative_gap"], feats)
    _bar_residual_corr(axes[1, 1], corrs, "rank_diff_norm", baseline="SLL")
    fig.suptitle(f"{PROBLEM_LABELS[problem]}: {PROBLEM_MECHANISM[problem]}", fontweight="bold")
    fig.savefig(out_dir / "cvrp100_best_loss_mechanism.png", bbox_inches="tight")
    fig.savefig(out_dir / "cvrp100_best_loss_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    rows.append({"problem": problem, "best_loss": _loss_name(problem), "mechanism": PROBLEM_MECHANISM[problem], "nearest": "SLL", "scale4_over_scale025": _scale_ratio(data, "Loss-only"), "main_residual_corr": corrs["rank_diff_norm"]})


def _plot_ffsp(problem: str, data: dict[str, Any], out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    feats = data["features"]
    coeffs = data["coeffs"]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.8), constrained_layout=True)
    _plot_common_gap(axes[0, 0], data, ["Loss-only", "PO/BT"])
    axes[0, 0].set_title("Best-loss avoids raw gap-dominated pressure")

    _plot_scale(axes[0, 1], data["scale_sensitivity"], ["Loss-only", "PO/BT"])
    axes[0, 1].set_title("Objective-scale suppression")

    xs, ys = _qcurve(feats["pool_range"], coeffs["Loss-only"], bins=8)
    axes[1, 0].plot(xs, ys, marker="o", ms=3.0, lw=1.6, color=PLOT_COLORS["Loss-only"])
    axes[1, 0].set_xlabel("candidate-pool objective range")
    axes[1, 0].set_ylabel("mean-normalized coefficient")
    axes[1, 0].set_title("Pool-scale dependence")

    corrs = _residual_corrs(coeffs["Loss-only"], coeffs["PO/BT"], feats["relative_gap"], feats)
    _bar_residual_corr(axes[1, 1], corrs, "pool_range", baseline="PO/BT")
    fig.suptitle(f"{PROBLEM_LABELS[problem]}: {PROBLEM_MECHANISM[problem]}", fontweight="bold")
    fig.savefig(out_dir / "ffsp100_best_loss_mechanism.png", bbox_inches="tight")
    fig.savefig(out_dir / "ffsp100_best_loss_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    rows.append({"problem": problem, "best_loss": _loss_name(problem), "mechanism": PROBLEM_MECHANISM[problem], "nearest": "PO/BT", "scale4_over_scale025": _scale_ratio(data, "Loss-only"), "main_residual_corr": corrs["pool_range"]})


def _plot_jssp(problem: str, data: dict[str, Any], out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    feats = data["features"]
    coeffs = data["coeffs"]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.8), constrained_layout=True)
    _plot_common_gap(axes[0, 0], data, ["Loss-only", "PO/BT"])
    axes[0, 0].set_title("Pair margin is PO-like, but not scale behavior")

    _plot_scale(axes[0, 1], data["scale_sensitivity"], ["Loss-only", "PO/BT"])
    axes[0, 1].set_title("L1 advantage normalization suppresses scale")

    xs, ys = _qcurve(feats["pool_std"], coeffs["Loss-only"], bins=8)
    axes[1, 0].plot(xs, ys, marker="o", ms=3.0, lw=1.6, color=PLOT_COLORS["Loss-only"])
    axes[1, 0].set_xlabel("candidate-pool objective std")
    axes[1, 0].set_ylabel("mean-normalized coefficient")
    axes[1, 0].set_title("Instance-local pool-std adaptation")

    corrs = _residual_corrs(coeffs["Loss-only"], coeffs["PO/BT"], feats["relative_gap"], feats)
    _bar_residual_corr(axes[1, 1], corrs, "pool_std", baseline="PO/BT")
    fig.suptitle(f"{PROBLEM_LABELS[problem].replace('JSSP10x10', 'JSSP')}: {PROBLEM_MECHANISM[problem]}", fontweight="bold")
    fig.savefig(out_dir / "jssp10x10_best_loss_mechanism.png", bbox_inches="tight")
    fig.savefig(out_dir / "jssp10x10_best_loss_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    rows.append({"problem": problem, "best_loss": _loss_name(problem), "mechanism": PROBLEM_MECHANISM[problem], "nearest": "PO/BT", "scale4_over_scale025": _scale_ratio(data, "Loss-only"), "main_residual_corr": corrs["pool_std"]})


def _scale_ratio(data: dict[str, Any], method: str) -> float:
    vals = {
        float(row["objective_scale"]): float(row["mean_coefficient"])
        for row in data["scale_sensitivity"]
        if str(row["method"]) == method
    }
    return float(vals.get(4.0, np.nan) / max(vals.get(0.25, np.nan), 1e-12))


def _plot_problem(problem: str, data: dict[str, Any], out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if problem == "tsp100":
        _plot_tsp(problem, data, out_dir, rows)
    elif problem == "cvrp100":
        _plot_cvrp(problem, data, out_dir, rows)
    elif problem == "ffsp100":
        _plot_ffsp(problem, data, out_dir, rows)
    elif problem == "jssp10x10":
        _plot_jssp(problem, data, out_dir, rows)
    else:
        raise KeyError(problem)


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
        default=str(REPO_ROOT / "figures" / "loss_fine_grained_signature" / "20260501-final-all-methods" / "per_problem_best_loss_mechanisms"),
    )
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    if "jssp10x10" in problems:
        _ensure_jssp_data(REPO_ROOT)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.6,
            "legend.fontsize": 7.4,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for problem in problems:
        print(f"[per-problem] {problem}", flush=True)
        data = _collect_one_problem(
            problem=problem,
            batches=max(int(args.batches), 1),
            seed=int(args.seed),
            device=device,
            state=str(args.state),
            sharpness=float(args.sharpness),
            max_pairs=max(int(args.max_pairs), 256),
        )
        _plot_problem(problem, data, out_dir, rows)
    _write_csv(out_dir / "per_problem_best_loss_mechanisms_summary.csv", rows)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
