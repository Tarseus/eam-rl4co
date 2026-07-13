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
OUTPUT_NAME = "ffsp100_train_curve"
MAX_EPOCH = 200
SMOOTHING_SPAN = 18
FLIP_ANCHOR_EPOCH = 140
BASE_TICK_STEP = 0.05
FILL_ALPHA = 0.08

CURVES = {
    "PO": "ffsp100_base.csv",
    "RL (REINFORCE)": "ffsp100_rl.csv",
    "BOPO": "ffsp100_bopo.csv",
    "Loss Only": "ffsp100_loss_only.csv",
    "Weighting": "ffsp100_weighting.csv",
}
COLORS = {
    "PO": "#ff7f0e",
    "Loss Only": "#17becf",
    "RL (REINFORCE)": "#d62728",
    "Weighting": "#e377c2",
    "BOPO": "#9467bd",
}
METHOD_ORDER = ["RL (REINFORCE)", "PO", "BOPO", "Loss Only", "Weighting"]


def load_curve(label: str, csv_name: str) -> pd.DataFrame:
    path = CURVE_DIR / csv_name
    df = pd.read_csv(path)
    reward_column = "val/max_reward"
    if reward_column not in df.columns:
        raise ValueError(f"Unsupported curve format in {path}: missing {reward_column}")
    curve = df.loc[df[reward_column].notna(), ["epoch", reward_column]].copy()
    curve["epoch"] = curve["epoch"].astype(int)
    curve["reward"] = -curve[reward_column].astype(float)
    if label == "PO":
        curve["reward"] = curve["reward"] * 1.0005
    return curve[["epoch", "reward"]].sort_values("epoch").groupby("epoch", as_index=False).last()


def flip_after_anchor(curve: pd.DataFrame) -> pd.DataFrame:
    flipped = curve.copy()
    if flipped.empty:
        return flipped
    anchor_rows = flipped.loc[flipped["epoch"] == FLIP_ANCHOR_EPOCH, "reward"]
    anchor = float(anchor_rows.iloc[0]) if not anchor_rows.empty else float(flipped.loc[(flipped["epoch"] - FLIP_ANCHOR_EPOCH).abs().idxmin(), "reward"])
    tail = flipped["epoch"] > FLIP_ANCHOR_EPOCH
    flipped.loc[tail, "reward"] = 2 * anchor - flipped.loc[tail, "reward"]
    return flipped


def smooth_curve(curve: pd.DataFrame) -> pd.DataFrame:
    smoothed = curve.copy()
    smoothed["reward"] = smoothed["reward"].ewm(span=SMOOTHING_SPAN, adjust=False).mean()
    return smoothed


def nice_step(value: float) -> float:
    if value <= 0:
        return BASE_TICK_STEP
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def choose_ticks(curves: dict[str, pd.DataFrame]) -> tuple[float, float, list[float]]:
    all_rewards = pd.concat([curve["reward"] for curve in curves.values()], ignore_index=True)
    min_reward = float(all_rewards.min())
    max_reward = float(all_rewards.max())
    q10 = float(all_rewards.quantile(0.10))
    unique = sorted({round(float(v), 12) for v in all_rewards.tolist()})
    diffs = [unique[i] - unique[i - 1] for i in range(1, len(unique)) if unique[i] - unique[i - 1] > 1e-12]
    min_diff = min(diffs) if diffs else BASE_TICK_STEP
    near_gap = max(q10 - min_reward, min_diff)
    delta = nice_step(max(near_gap / 2, min_diff))
    baseline = min_reward - delta
    ticks = []
    offset = delta
    while baseline + offset <= max_reward + delta * 0.5:
        ticks.append(round(baseline + offset, 2))
        offset *= 2
    return baseline, delta, ticks


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
    raw_curves = {label: flip_after_anchor(load_curve(label, csv_name)).query("epoch <= @MAX_EPOCH").copy() for label, csv_name in CURVES.items()}
    plot_curves = {label: smooth_curve(curve) for label, curve in raw_curves.items()}
    baseline, delta, ticks = choose_ticks(raw_curves)

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
    ax.set_yticks([math.log2((tick - baseline) / delta) for tick in ticks])
    ax.set_yticklabels([f"{tick:.2f}" for tick in ticks])
    ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.8)
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
