#!/usr/bin/env python3
"""Plot direct joint-search curves in the canonical paper trajectory style."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_objective_search_figures import (  # noqa: E402
    _add_stage_gradient_background,
    _apply_line_shadow,
    _compress_plateaus,
    _fill_to_bottom_gradient,
    _generation_axis,
    plt,
)


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    for row in rows:
        row["generation"] = int(row["generation"])
        row["best_so_far_fitness"] = float(row["best_so_far_fitness"])
        row["improved_best_so_far"] = str(row["improved_best_so_far"]).lower() == "true"
    return rows


def _plot_one(problem: str, rows: list[dict[str, Any]], output: Path) -> Path:
    generations = [int(row["generation"]) for row in rows]
    fitness = [float(row["best_so_far_fitness"]) for row in rows]
    score_gain = [-value for value in fitness]
    gain_by_generation = dict(zip(generations, score_gain))

    display_generations = _compress_plateaus(generations, score_gain)
    x_positions, tick_positions, tick_labels = _generation_axis(
        display_generations,
        prefix="G",
        start_x=0,
    )
    x_by_generation = dict(zip(display_generations, x_positions))
    display_gains = [gain_by_generation[generation] for generation in display_generations]
    x_left = -1.0
    x_right = float(x_positions[-1]) + 0.8

    fig, ax = plt.subplots(figsize=(11.4, 7.6))
    # The canonical staged plot uses blue for its search region.  Move the
    # blue/orange transition beyond this single-stage axis to retain that look.
    _add_stage_gradient_background(
        ax,
        x_left=x_left,
        phase_boundary_x=x_right + 100.0,
        x_right=x_right,
    )

    line = ax.step(
        x_positions,
        display_gains,
        where="post",
        color="#111111",
        lw=2.6,
        marker="o",
        ms=5.2,
        zorder=4,
    )[0]
    _apply_line_shadow(line, alpha=0.26, offset=(1.5, -1.5))

    improvement_rows = [row for row in rows if bool(row["improved_best_so_far"])]
    improvement_points: list[tuple[float, float, int]] = []
    for row in improvement_rows:
        generation = int(row["generation"])
        if generation not in x_by_generation:
            continue
        improvement_points.append(
            (float(x_by_generation[generation]), float(gain_by_generation[generation]), generation)
        )
    for x, y, _ in improvement_points:
        ax.scatter([x], [y], s=300, color="#4c78a8", alpha=0.10, zorder=5, linewidths=0)
        ax.scatter([x], [y], s=175, color="#4c78a8", alpha=0.18, zorder=6, linewidths=0)
        ax.scatter([x], [y], s=92, color="#4c78a8", edgecolors="#dce7f4", linewidths=1.2, zorder=7)

    final_row = min(rows, key=lambda row: float(row["best_so_far_fitness"]))
    final_generation = int(final_row["generation"])
    final_x = float(x_by_generation[final_generation])
    final_y = float(gain_by_generation[final_generation])
    ax.scatter(
        [final_x],
        [final_y],
        s=245,
        marker="*",
        facecolor="#ffcf4d",
        edgecolor="#6b4b00",
        linewidths=1.05,
        zorder=10,
    )
    ax.annotate(
        f"G{final_generation} best: {final_y:.3f}",
        xy=(final_x, final_y),
        xytext=(0, 54),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="semibold",
        color="#332500",
        bbox=dict(boxstyle="round,pad=0.26", fc="#fff7d6", ec="#b78900", lw=1.0, alpha=0.97),
        arrowprops=dict(arrowstyle="->", color="#9a7400", lw=1.0, shrinkA=5, shrinkB=8),
        zorder=11,
        annotation_clip=False,
    )

    y_min = min(display_gains)
    y_max = max(0.0, max(display_gains))
    y_span = max(y_max - y_min, abs(y_min) * 0.08, 1e-6)
    y_pad = max(y_span * 0.12, 0.01 if problem == "FFSP100" else 0.002)
    y_bottom = y_min - y_pad * 0.35
    y_top = y_max + y_pad
    _fill_to_bottom_gradient(
        ax,
        x_positions,
        display_gains,
        bottom=y_bottom,
        color="#111111",
        alpha=0.040,
        zorder=1.08,
    )

    ax.set_ylabel("score gain", fontsize=14)
    ax.grid(True, alpha=0.45, color="#b0b0b0", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="#888888", lw=1.0, ls=":")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7.4)
    ax.tick_params(axis="x", pad=10)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.text(
        (x_positions[0] + x_positions[-1]) / 2,
        y_top + y_pad * 0.11,
        "Direct Joint Search (Checkpoint-only)",
        ha="center",
        va="bottom",
        fontsize=17,
        color="#222222",
        clip_on=False,
    )
    ax.text(
        x_left + 0.64,
        y_top - y_pad * 0.15,
        r"$\mathrm{score\ gain}=-\Delta_{\mathrm{ckpt}}$" "\n"
        "higher is better\n"
        "early-pruned: checkpoint branch fitness",
        ha="left",
        va="top",
        fontsize=8.0,
        color="#1f1f1f",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#c9c9c9", alpha=0.96),
        zorder=8,
    )
    ax.text(
        x_right - 0.05,
        y_bottom + y_pad * 0.12,
        problem,
        ha="right",
        va="bottom",
        fontsize=11.5,
        fontweight="semibold",
        color="#333333",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsp", type=Path, required=True, help="TSP100 search_trajectory.csv")
    parser.add_argument("--ffsp", type=Path, required=True, help="FFSP100 search_trajectory.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    outputs = [
        _plot_one("TSP100", _load(args.tsp), args.output_dir / "tsp100_joint_search_trajectory.png"),
        _plot_one("FFSP100", _load(args.ffsp), args.output_dir / "ffsp100_joint_search_trajectory.png"),
    ]
    for output in outputs:
        print(output)
        print(output.with_suffix(".pdf"))
        print(output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
