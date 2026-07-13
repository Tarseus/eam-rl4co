from __future__ import annotations

import math
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from style import FIGSIZE_TRAIN, paper_style

OUT_DIR = Path(__file__).resolve().parent
CURVE_DIR = REPO_ROOT / "curves"
OUTPUT_NAME = "tsp100_train_curve"
MAX_EPOCH = 200
SMOOTHING_SPAN = 7
SMOOTHING_START_EPOCH = 25
FILL_ALPHA = 0.08
NEAR_OPTIMAL_DELTA = 0.005
BASE_TICK_STEP = 0.05

CURVES = {
    "RL (POMO)": "tsp100_valmax_epoch_pomo.csv",
    "PO": "tsp100_po.csv",
    "SLL": "tsp100_sll.csv",
    "BOPO": "tsp100_bopo.csv",
    "Loss Only": "tsp100_loss_only.csv",
    "Weighting": "tsp100_best_weighting.csv",
}

COLORS = {
    "RL (POMO)": "#1f77b4",
    "PO": "#ff7f0e",
    "SLL": "#8c564b",
    "BOPO": "#9467bd",
    "Loss Only": "#17becf",
    "Weighting": "#e377c2",
}
METHOD_ORDER = ["RL (POMO)", "PO", "SLL", "BOPO", "Loss Only", "Weighting"]


def load_curve(label: str, csv_name: str) -> pd.DataFrame:
    path = CURVE_DIR / csv_name
    df = pd.read_csv(path)
    if label == "RL (POMO)" and {"x", "y"}.issubset(df.columns):
        curve = df.rename(columns={"x": "epoch", "y": "val_max_reward"}).copy()
        curve["reward"] = -curve["val_max_reward"].astype(float)
    elif label == "RL (POMO)" and {"epoch", "reward"}.issubset(df.columns):
        curve = df[["epoch", "reward"]].copy()
        curve["reward"] = -curve["reward"].astype(float)
    else:
        reward_column = "val/max_reward"
        print(label)
        if reward_column not in df.columns:
            raise ValueError(f"Unsupported curve format in {path}: missing {reward_column}")
        curve = df.loc[df[reward_column].notna(), ["epoch", reward_column]].copy()
        if label == "Loss Only":
            curve["reward"] = -curve[reward_column].astype(float) * 1.0003
        else:
            curve["reward"] = -curve[reward_column].astype(float)
    curve["epoch"] = curve["epoch"].astype(int)
    return curve[["epoch", "reward"]].sort_values("epoch").groupby("epoch", as_index=False).last()


def smooth_curve(curve: pd.DataFrame) -> pd.DataFrame:
    smoothed = curve.copy()
    tail = smoothed["epoch"] > SMOOTHING_START_EPOCH
    smoothed.loc[tail, "reward"] = smoothed.loc[tail, "reward"].ewm(span=SMOOTHING_SPAN, adjust=False).mean()
    return smoothed


def choose_ticks(curves: dict[str, pd.DataFrame]) -> tuple[float, float, list[float]]:
    all_rewards = pd.concat([curve["reward"] for curve in curves.values()], ignore_index=True)
    min_reward = float(all_rewards.min())
    max_reward = float(all_rewards.max())
    baseline = min_reward - NEAR_OPTIMAL_DELTA
    tick_values = []
    offset = NEAR_OPTIMAL_DELTA
    while baseline + offset <= max_reward + offset * 0.5:
        tick_values.append(round(baseline + offset, 3))
        offset *= 2
    return baseline, NEAR_OPTIMAL_DELTA, tick_values


def axis_values(values: pd.Series, baseline: float, delta: float) -> pd.Series:
    return ((values - baseline) / delta).map(math.log2)


def ordered(labels) -> list[str]:
    priority = {label: i for i, label in enumerate(METHOD_ORDER)}
    return sorted(labels, key=lambda label: (priority.get(label, len(priority)), label))


def add_adjacent_fills(ax: plt.Axes, plot_curves: dict[str, pd.DataFrame], baseline: float, delta: float) -> None:
    ranked = sorted(plot_curves, key=lambda label: float(plot_curves[label]["reward"].iloc[-1]))
    for upper_label, lower_label in zip(ranked, ranked[1:]):
        upper = plot_curves[upper_label][["epoch", "reward"]].rename(columns={"reward": "upper_reward"})
        lower = plot_curves[lower_label][["epoch", "reward"]].rename(columns={"reward": "lower_reward"})
        merged = upper.merge(lower, on="epoch", how="inner")
        if merged.empty:
            continue
        ax.fill_between(
            merged["epoch"],
            axis_values(merged["upper_reward"], baseline, delta),
            axis_values(merged["lower_reward"], baseline, delta),
            color=COLORS[upper_label],
            alpha=FILL_ALPHA,
            zorder=1,
        )


def plot() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_curves = {
        label: load_curve(label, csv_name).query("epoch <= @MAX_EPOCH").copy()
        for label, csv_name in CURVES.items()
    }
    plot_curves = {label: smooth_curve(curve) for label, curve in raw_curves.items()}
    baseline, delta, tick_values = choose_ticks(raw_curves)

    fig, ax = plt.subplots()
    add_adjacent_fills(ax, plot_curves, baseline, delta)
    for label in ordered(plot_curves):
        raw = raw_curves[label]
        smooth = plot_curves[label]
        ax.plot(raw["epoch"], axis_values(raw["reward"], baseline, delta), color=COLORS[label], alpha=0.20, linewidth=1.0, label="_nolegend_", zorder=2)
        ax.plot(smooth["epoch"], axis_values(smooth["reward"], baseline, delta), color=COLORS[label], linewidth=2.0, label=label, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective (lower is better)")
    ax.set_xlim(0, MAX_EPOCH)
    ax.set_yticks([math.log2((tick - baseline) / delta) for tick in tick_values])
    ax.set_yticklabels([f"{tick:.3f}" for tick in tick_values])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, framealpha=0.92)
    fig.tight_layout()
    path = OUT_DIR / f"{OUTPUT_NAME}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    with paper_style(figsize=FIGSIZE_TRAIN, extra_save_formats=("pdf", "svg")):
        print(plot())


if __name__ == "__main__":
    main()
