#!/usr/bin/env python3
"""Analyze search-fitness versus full-training final performance."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        raise ValueError("at least three completed candidates are required")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def exact_permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    *,
    use_ranks: bool = False,
) -> float | None:
    if len(x) > 9:
        return None
    total = 0
    extreme = 0
    statistic_x = rankdata(x) if use_ranks else x
    for perm in itertools.permutations(y.tolist()):
        statistic_y = np.asarray(perm, dtype=float)
        if use_ranks:
            statistic_y = rankdata(statistic_y)
        value = float(np.corrcoef(statistic_x, statistic_y)[0, 1])
        if abs(value) + 1e-12 >= abs(observed):
            extreme += 1
        total += 1
    return extreme / total


def load_completed(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = []
    for row in rows:
        try:
            fitness = float(row["search_fitness"])
            final_cost = float(row["final_cost"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(fitness) and math.isfinite(final_cost):
            item = dict(row)
            item["search_fitness"] = fitness
            item["final_cost"] = final_cost
            completed.append(item)
    return completed


def analyze(manifest_csv: Path) -> dict[str, Any]:
    rows = load_completed(manifest_csv)
    search_fitness = np.asarray([row["search_fitness"] for row in rows], dtype=float)
    final_cost = np.asarray([row["final_cost"] for row in rows], dtype=float)
    # Both source metrics are minimization metrics. Negating both expresses the
    # requested higher-is-better utility without changing their correlation.
    fitness_utility = -search_fitness
    final_utility = -final_cost
    pearson, spearman = correlation(fitness_utility, final_utility)
    slope, intercept = np.polyfit(search_fitness, final_cost, 1)
    fitted = intercept + slope * search_fitness
    rmse = float(np.sqrt(np.mean(np.square(fitted - final_cost))))
    return {
        "n": len(rows),
        "orientation": "fitness_utility=-search_fitness; final_utility=-final_cost; higher is better",
        "pearson_r": pearson,
        "pearson_exact_two_sided_permutation_p": exact_permutation_pvalue(
            fitness_utility, final_utility, pearson
        ),
        "spearman_rho": spearman,
        "spearman_exact_two_sided_permutation_p": exact_permutation_pvalue(
            fitness_utility, final_utility, spearman, use_ranks=True
        ),
        "r_squared": pearson**2,
        "linear_fit": {
            "formula": "final_cost = intercept + slope * search_fitness",
            "slope": float(slope),
            "intercept": float(intercept),
            "in_sample_rmse": rmse,
        },
        "positive_pearson": pearson > 0,
        "positive_spearman": spearman > 0,
        "candidate_ids": [row.get("candidate_id", "") for row in rows],
        "search_fitness": search_fitness.tolist(),
        "final_cost": final_cost.tolist(),
        "final_result_sources": [row.get("final_result_source", "") for row in rows],
    }


def short_candidate_id(row: dict[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id", ""))
    label = candidate_id.split("_", 1)[0] if candidate_id else ""
    return "APW" if label == "q00" else label


def plot(rows: list[dict[str, Any]], output: Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    search_fitness = np.asarray([float(row["search_fitness"]) for row in rows])
    final_cost = np.asarray([float(row["final_cost"]) for row in rows])
    fitness_gap = search_fitness - search_fitness.min()
    cost_gap = final_cost - final_cost.min()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.9))

    colors = ["#009E73", "#0072B2", "#0072B2", "#0072B2", "#D55E00"]
    ax.scatter(fitness_gap, cost_gap, s=54, color=colors, edgecolor="white", linewidth=0.6, zorder=3)

    gap_grid = np.concatenate(([0.0], np.geomspace(1e-4, fitness_gap.max(), 240)))
    raw_grid = search_fitness.min() + gap_grid
    fit = result["linear_fit"]
    fitted_gap = fit["intercept"] + fit["slope"] * raw_grid - final_cost.min()
    ax.plot(gap_grid, fitted_gap, color="#6B7280", linewidth=1.5, linestyle="--", zorder=2)

    ax.set_xscale("symlog", linthresh=0.005, linscale=1.0, base=10)
    ax.set_yscale("symlog", linthresh=0.002, linscale=1.0, base=10)
    ax.set_xlim(-0.0012, 0.55)
    ax.set_ylim(-0.0005, 0.14)
    ax.set_xticks([0.0, 0.005, 0.01, 0.03, 0.1, 0.4])
    ax.set_yticks([0.0, 0.002, 0.004, 0.01, 0.03, 0.1])
    formatter = FuncFormatter(lambda value, _: "0" if abs(value) < 1e-15 else f"{value:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    label_offsets = {
        "APW": (6, 7),
        "q01": (-7, 9),
        "q02": (7, -14),
        "q03": (7, -14),
        "q04": (-7, 7),
    }
    for row, xx, yy in zip(rows, fitness_gap, cost_gap):
        label = short_candidate_id(row)
        offset = label_offsets.get(label, (5, 5))
        ax.annotate(
            rf"$\mathrm{{{label}}}$",
            (xx, yy),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha="right" if offset[0] < 0 else "left",
        )
    ax.set_xlabel(r"Search-fitness gap from $\mathrm{APW}$")
    ax.set_ylabel(r"Final-cost gap from $\mathrm{APW}$")
    ax.text(
        0.04,
        0.96,
        rf"Pearson $r = {result['pearson_r']:.4f}$"
        "\n"
        rf"Spearman $\rho = {result['spearman_rho']:.4f}$"
        "\n"
        rf"$n = {result['n']}$",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.grid(alpha=0.16, linewidth=0.7)
    ax.plot([], [], color="#6B7280", linestyle="--", linewidth=1.5, label="Linear fit in raw coordinates")
    ax.legend(loc="lower right")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_candidate_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fieldnames = [
        "candidate",
        "population_rank",
        "fitness_quantile",
        "search_fitness",
        "final_cost",
        "fitness_utility",
        "final_performance_utility",
        "pair_json",
        "final_result_source",
    ]
    records = []
    for row in rows:
        search_fitness = float(row["search_fitness"])
        final_cost = float(row["final_cost"])
        records.append(
            {
                "candidate": short_candidate_id(row),
                "population_rank": row.get("population_rank", ""),
                "fitness_quantile": row.get("fitness_quantile", ""),
                "search_fitness": search_fitness,
                "final_cost": final_cost,
                "fitness_utility": -search_fitness,
                "final_performance_utility": -final_cost,
                "pair_json": row.get("pair_json", ""),
                "final_result_source": row.get("final_result_source", ""),
            }
        )
    with (output_dir / "candidate_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    latex_rows = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Candidate & Population rank & Quantile & Search fitness & Final cost \\",
        r"\midrule",
    ]
    for record in records:
        latex_rows.append(
            f"{record['candidate']} & {record['population_rank']} & "
            f"{float(record['fitness_quantile']):.2f} & "
            f"{record['search_fitness']:.6f} & {record['final_cost']:.6f} \\\\"
        )
    latex_rows.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "candidate_results.tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")


def write_readme(rows: list[dict[str, Any]], result: dict[str, Any], output_dir: Path) -> None:
    first = rows[0]
    last = rows[-1]
    cost_delta = float(last["final_cost"]) - float(first["final_cost"])
    fitness_delta = float(last["search_fitness"]) - float(first["search_fitness"])
    table_rows = [
        f"| {short_candidate_id(row)} | {row.get('population_rank', '')} | "
        f"{float(row['search_fitness']):.6f} | {float(row['final_cost']):.6f} |"
        for row in rows
    ]
    content = f"""# TSP100 search-fitness / final-performance correlation

## Scope

This directory contains the protocol-defined five-candidate analysis. APW (source candidate q00) and q01--q04 are evenly sampled from the 53 completed high-fidelity pairs at population ranks 0, 13, 26, 39, and 52. Both search fitness and final cost are minimization metrics. Correlations are reported after negating both metrics into higher-is-better utilities, which leaves the correlation coefficients unchanged.

## Candidate results

| Candidate | Population rank | Search fitness | Final augmented cost |
|---|---:|---:|---:|
{chr(10).join(table_rows)}

The final cost is the negative of `test/max_aug_reward` on the fixed TSP100 test set with 100 starts and eight augmentations. APW (source candidate q00) reuses the existing canonical fixed-test result; q01--q04 use their completed 200-epoch training runs.

## Correlation statistics

- Pearson correlation: `{result['pearson_r']:.12f}`; exact two-sided permutation p-value: `{result['pearson_exact_two_sided_permutation_p']:.12f}`.
- Spearman correlation: `{result['spearman_rho']:.12f}`; exact two-sided permutation p-value: `{result['spearman_exact_two_sided_permutation_p']:.12f}`.
- Linear-fit R-squared: `{result['r_squared']:.12f}`.
- Linear fit: `final_cost = {result['linear_fit']['intercept']:.12f} + {result['linear_fit']['slope']:.12f} * search_fitness`.
- In-sample RMSE: `{result['linear_fit']['in_sample_rmse']:.12f}`.

The figure places all five candidates in one panel. Both axes show nonnegative degradation gaps from APW on symmetric-log scales, retaining the exact APW zero point while separating the near-optimal q01--q03 candidates from q04. The dashed line is the linear fit computed in the original, untransformed coordinates.
All figure text and mathematical symbols are rendered by the same LaTeX `times` and `amsmath` setup used by the AAAI paper source.

## Phase 1 findings for confirmation

Finding 1: APW and q01--q04 exhibit a strictly monotonic search-fitness/final-cost ordering, with Spearman rho = {result['spearman_rho']:.6f} and exact two-sided p = {result['spearman_exact_two_sided_permutation_p']:.6f} across five candidates.

Finding 2: Search fitness and final cost have Pearson r = {result['pearson_r']:.6f} (R-squared = {result['r_squared']:.6f}; exact two-sided p = {result['pearson_exact_two_sided_permutation_p']:.6f}) across the protocol-defined five points.

Finding 3: The worst-fitness endpoint q04 has a final cost {cost_delta:.6f} higher than APW (7.867153 versus 7.767692), while their search fitness differs by {fitness_delta:.6f}.

Identified 3 findings. Confirm, correct, or add findings before any discussion paragraph is written.

## Provenance

- Candidate definition and source paths: `research/fitness_final_correlation_tsp100/candidates/manifest.csv`
- Sampling protocol: `research/fitness_final_correlation_tsp100/protocol.md`
- Machine-readable statistics: `paper_materials/fitness_final_correlation_tsp100/correlation.json`
- Paper table inputs: `paper_materials/fitness_final_correlation_tsp100/candidate_results.csv`
- LaTeX table: `paper_materials/fitness_final_correlation_tsp100/candidate_results.tex`
- Figure: `paper_materials/fitness_final_correlation_tsp100/fitness_vs_final_performance.png` and `.pdf`
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_completed(args.manifest)
    result = analyze(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "correlation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_candidate_results(rows, args.output_dir)
    write_readme(rows, result, args.output_dir)
    plot(rows, args.output_dir / "fitness_vs_final_performance.png", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
