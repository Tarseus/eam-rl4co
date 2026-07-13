from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CURVE_DIR = REPO_ROOT / "curves"
OUT_DIR = REPO_ROOT / "figures" / "loss_weighting_relative_scale_curves"

METHODS = ["RL", "PO", "SLL", "BOPO", "Loss-only", "Loss+Weighting"]
COLORS = {
    "RL": "#666666",
    "PO": "#E69F00",
    "SLL": "#009E73",
    "BOPO": "#CC79A7",
    "Loss-only": "#0072B2",
    "Loss+Weighting": "#D55E00",
}
LINESTYLES = {
    "RL": (0, (2, 2)),
    "PO": "-",
    "SLL": "-",
    "BOPO": "-",
    "Loss-only": "-",
    "Loss+Weighting": "-",
}

PANELS = [
    (
        "TSP",
        "source/search size",
        "TSP100",
        "tsp100",
        {
            "RL": "tsp100_valmax_epoch_pomo.csv",
            "PO": "tsp100_po.csv",
            "SLL": "tsp100_sll.csv",
            "BOPO": "tsp100_bopo.csv",
            "Loss-only": "tsp100_loss_only.csv",
            "Loss+Weighting": "tsp100_best_weighting.csv",
        },
    ),
    (
        "TSP",
        "different scale",
        "TSP50",
        "tsp50",
        {
            "RL": "tsp50_valmax_epoch_pomo.csv",
            "PO": "tsp50_po.csv",
            "SLL": "tsp50_sll.csv",
            "BOPO": "tsp50_bopo.csv",
            "Loss-only": "tsp50_loss_only.csv",
            "Loss+Weighting": "tsp50_best_weighting.csv",
        },
    ),
    (
        "CVRP",
        "source/search size",
        "CVRP100",
        "cvrp100",
        {
            "RL": "pomo_cvrp100_epoch.csv",
            "PO": "cvrp100_po.csv",
            "SLL": "cvrp100_sll.csv",
            "BOPO": "cvrp100_bopo.csv",
            "Loss-only": "cvrp100_loss_only.csv",
            "Loss+Weighting": "cvrp100_best_weighting.csv",
        },
    ),
    (
        "CVRP",
        "different scale",
        "CVRP50",
        "cvrp50",
        {
            "RL": "cvrp50_valmax_epoch_pomo.csv",
            "PO": "cvrp50_po.csv",
            "SLL": "cvrp50_sll.csv",
            "BOPO": "cvrp50_bopo.csv",
            "Loss-only": "cvrp50_loss_only.csv",
            "Loss+Weighting": "cvrp50_best_weighting.csv",
        },
    ),
    (
        "FFSP",
        "source/search size",
        "FFSP100",
        "ffsp100",
        {
            "RL": "ffsp100_rl.csv",
            "PO": "ffsp100_base.csv",
            "SLL": "ffsp100_sll.csv",
            "BOPO": "ffsp100_bopo.csv",
            "Loss-only": "ffsp100_loss_only.csv",
            "Loss+Weighting": "ffsp100_weighting.csv",
        },
    ),
    (
        "FFSP",
        "different scale",
        "FFSP50",
        "ffsp50",
        {
            "RL": "ffsp50_rl.csv",
            "PO": "ffsp50_base.csv",
            "SLL": "ffsp50_sll.csv",
            "BOPO": "ffsp50_bopo.csv",
            "Loss-only": "ffsp50_loss_only.csv",
            "Loss+Weighting": "ffsp50_weighting.csv",
        },
    ),
    (
        "JSSP",
        "source/search size",
        "JSSP10x10",
        "jssp10x10",
        {
            "RL": "jssp10x10_rl.csv",
            "PO": "jssp10x10_po.csv",
            "SLL": "jssp10x10_sll.csv",
            "BOPO": "jssp10x10_bopo.csv",
            "Loss-only": "jssp10x10_loss_only.csv",
            "Loss+Weighting": "jssp10x10_best.csv",
        },
    ),
    (
        "JSSP",
        "different scale",
        "JSSP15x15",
        "jssp15x15",
        {
            "RL": "jssp15x15_rl.csv",
            "PO": "jssp15x15_po.csv",
            "SLL": "jssp15x15_sll.csv",
            "BOPO": "jssp15x15_bopo.csv",
            "Loss-only": "jssp15x15_loss_only.csv",
            "Loss+Weighting": "jssp15x15_best.csv",
        },
    ),
]


def _score_column(dataset: str, df: pd.DataFrame) -> str | None:
    if dataset.startswith("jssp"):
        for col in ("val/gap", "test/gap"):
            if col in df.columns and df[col].notna().any():
                return col
        for col in ("val/makespan", "test/makespan"):
            if col in df.columns and df[col].notna().any():
                return col
    for col in ("test/max_aug_reward", "val/max_aug_reward", "test/max_reward", "val/max_reward", "y"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _load_curve(dataset: str, filename: str) -> pd.DataFrame | None:
    path = CURVE_DIR / filename
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    col = _score_column(dataset, df)
    if col is None:
        return None
    if "epoch" in df.columns:
        x = pd.to_numeric(df["epoch"], errors="coerce")
    elif "x" in df.columns:
        x = pd.to_numeric(df["x"], errors="coerce")
    else:
        return None
    y = pd.to_numeric(df[col], errors="coerce")
    curve = pd.DataFrame({"epoch": x, "score": y}).dropna()
    if curve.empty:
        return None
    if dataset.startswith("jssp"):
        curve["score"] = -curve["score"]
    curve = curve.groupby("epoch", as_index=False)["score"].last().sort_values("epoch")
    curve["progress"] = (curve["epoch"] - curve["epoch"].min()) / max(curve["epoch"].max() - curve["epoch"].min(), 1e-9)
    curve["score_smooth"] = curve["score"].rolling(5, min_periods=1, center=True).mean()
    return curve


def _interp(curve: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, curve["progress"].to_numpy(), curve["score_smooth"].to_numpy())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 160)
    panel_data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, object]] = []

    for problem, scale_kind, title, dataset, files in PANELS:
        curves: dict[str, pd.DataFrame] = {}
        for method, filename in files.items():
            curve = _load_curve(dataset, filename)
            if curve is not None:
                curves[method] = curve
        if "Loss-only" not in curves:
            continue
        baseline = _interp(curves["Loss-only"], grid)
        rel: dict[str, np.ndarray] = {}
        for method in METHODS:
            if method not in curves:
                continue
            values = _interp(curves[method], grid) - baseline
            rel[method] = values
            summary_rows.append(
                {
                    "problem": problem,
                    "scale_kind": scale_kind,
                    "dataset": title,
                    "method": method,
                    "final_relative_to_loss_only": float(values[-1]),
                    "auc_relative_to_loss_only": float(np.trapz(values, grid)),
                    "available": True,
                }
            )
        for method in METHODS:
            if method not in rel:
                summary_rows.append(
                    {
                        "problem": problem,
                        "scale_kind": scale_kind,
                        "dataset": title,
                        "method": method,
                        "final_relative_to_loss_only": np.nan,
                        "auc_relative_to_loss_only": np.nan,
                        "available": False,
                    }
                )
        panel_data[(problem, scale_kind)] = rel

    with (OUT_DIR / "relative_curve_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "scale_kind",
                "dataset",
                "method",
                "final_relative_to_loss_only",
                "auc_relative_to_loss_only",
                "available",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(4, 2, figsize=(12.2, 12.0), sharex=True)
    fig.patch.set_facecolor("#fbfaf7")
    for r, problem in enumerate(["TSP", "CVRP", "FFSP", "JSSP"]):
        for c, scale_kind in enumerate(["source/search size", "different scale"]):
            ax = axes[r, c]
            ax.set_facecolor("#fbfaf7")
            rel = panel_data.get((problem, scale_kind), {})
            for method in METHODS:
                if method not in rel:
                    continue
                lw = 2.8 if method in {"Loss-only", "Loss+Weighting"} else 1.55
                alpha = 1.0 if method in {"Loss-only", "Loss+Weighting"} else 0.78
                ax.plot(
                    grid,
                    rel[method],
                    color=COLORS[method],
                    lw=lw,
                    alpha=alpha,
                    linestyle=LINESTYLES[method],
                    label=method,
                )
            ax.axhline(0.0, color="#2d2a24", lw=0.85, alpha=0.65)
            ax.grid(axis="y", color="#ddd6ca", lw=0.7, alpha=0.8)
            ax.set_title(f"{problem}: {scale_kind}", fontsize=11.5, fontweight="semibold")
            if c == 0:
                ax.set_ylabel("score minus Loss-only\nhigher is better")
            if r == 3:
                ax.set_xlabel("training progress")
            missing = [m for m in METHODS if m not in rel]
            if missing:
                ax.text(
                    0.99,
                    0.03,
                    "missing: " + ", ".join(missing),
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=7.8,
                    color="#77716a",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.012))
    fig.suptitle(
        "Relative optimization curves: separating Loss-only from weighting and baselines",
        y=1.035,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "These panels use available training/evaluation logs. The right column is a different-scale probe from existing logs, not a same-checkpoint cross-size evaluation.",
        ha="center",
        fontsize=8.5,
        color="#5b5650",
    )
    fig.tight_layout(rect=(0, 0.022, 1, 0.982))
    out = OUT_DIR / "01_relative_to_loss_only_source_vs_scale_probe.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(summary_rows)
    pivot = df[df["method"].isin(["Loss-only", "Loss+Weighting"])].pivot_table(
        index=["problem", "scale_kind", "dataset"],
        columns="method",
        values="final_relative_to_loss_only",
        aggfunc="first",
    )
    print(pivot.to_string())
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
