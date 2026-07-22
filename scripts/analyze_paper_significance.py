from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_materials" / "statistical_analysis" / "results"
OURS = ("USW", "ASW")
HAND_DESIGNED = ("PO4COPs", "SLL", "BOPO")
STANDARD_NEURAL = ("SymNCO",)


def holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    order = np.argsort(np.asarray(pvalues, dtype=np.float64))
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(pvalues[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    # Keep the temporary bootstrap index matrix bounded for the 10,000-instance
    # routing evaluations. Chunking preserves the bootstrap definition and RNG
    # stream while avoiding multi-gigabyte allocations.
    max_indices_per_chunk = 2_000_000
    chunk_size = max(1, min(samples, max_indices_per_chunk // differences.size))
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(
            0,
            differences.size,
            size=(stop - start, differences.size),
        )
        means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def matched_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def compare(
    frame: pd.DataFrame,
    *,
    problem: str,
    left: str,
    right: str,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    subset = frame[frame["problem"] == problem]
    wide = subset.pivot_table(index="instance_id", columns="method", values="cost", aggfunc="first")
    pair = wide[[left, right]].dropna()
    left_values = pair[left].to_numpy(dtype=np.float64)
    right_values = pair[right].to_numpy(dtype=np.float64)
    differences = left_values - right_values
    if differences.size < 2:
        raise ValueError(f"Not enough paired values for {problem}: {left} vs {right}")

    t_result = stats.ttest_rel(left_values, right_values)
    if np.all(differences == 0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(
            stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue
        )
    ci_low, ci_high = bootstrap_mean_ci(
        differences,
        seed=seed,
        samples=bootstrap_samples,
    )
    wins = int((differences < 0).sum())
    ties = int((differences == 0).sum())
    losses = int((differences > 0).sum())
    return {
        "problem": problem,
        "left_method": left,
        "right_method": right,
        "n": int(differences.size),
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        "mean_difference_left_minus_right": float(differences.mean()),
        "median_difference_left_minus_right": float(np.median(differences)),
        "relative_improvement_percent": float(
            (right_values.mean() - left_values.mean()) / right_values.mean() * 100.0
        ),
        "bootstrap_95ci_mean_difference_low": ci_low,
        "bootstrap_95ci_mean_difference_high": ci_high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "probability_of_superiority": float((wins + 0.5 * ties) / differences.size),
        "paired_t_pvalue": float(t_result.pvalue),
        "wilcoxon_pvalue": wilcoxon_p,
        "matched_rank_biserial": matched_rank_biserial(differences),
        "left_is_better": bool(differences.mean() < 0),
    }


def load_jssp_directory(root: Path) -> pd.DataFrame:
    rows = []
    aliases = {
        "po": "PO4COPs",
        "sll": "SLL",
        "bopo": "BOPO",
        "loss_only": "USW",
        "weighting": "ASW",
    }
    for size in ("10x10", "15x15"):
        for legacy, method in aliases.items():
            candidates = [
                root
                / f"jssp{size}_{legacy}_seed12345678_n100_fixedparser"
                / "generated.csv"
            ]
            if size == "15x15" and legacy == "loss_only":
                candidates.insert(
                    0,
                    root
                    / "jssp15x15_loss_only_seed12345678_n100_fixedparser"
                    / "generated.csv",
                )
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                continue
            data = pd.read_csv(path)
            rows.append(
                pd.DataFrame(
                    {
                        "problem": f"jssp{size}",
                        "method": method,
                        "instance_id": data["instance"].astype(str),
                        "cost": data["pred_makespan"].astype(float),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired significance tests for all paper candidate-pool objectives."
    )
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument(
        "--symnco-cvrp50-npy",
        type=Path,
        default=None,
        help=(
            "Optional ordered CVRP50 Sym-NCO cost array. When supplied, add "
            "USW/ASW-vs-SymNCO paired comparisons to the same Holm family."
        ),
    )
    parser.add_argument(
        "--jssp-root",
        type=Path,
        default=REPO_ROOT / "logs" / "eval_jssp_generated",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = []
    for path in args.input:
        data = pd.read_csv(path.resolve())
        required = {"problem", "method", "instance_id", "cost"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(data[list(required)].copy())
    if args.symnco_cvrp50_npy is not None:
        symnco_costs = np.load(args.symnco_cvrp50_npy.resolve())
        if symnco_costs.ndim != 1 or symnco_costs.size != 10_000:
            raise ValueError(
                "The CVRP50 Sym-NCO cost array must contain exactly 10,000 values"
            )
        frames.append(
            pd.DataFrame(
                {
                    "problem": "cvrp50",
                    "method": "SymNCO",
                    "instance_id": [
                        f"cvrp50_{index:05d}" for index in range(symnco_costs.size)
                    ],
                    "cost": symnco_costs.astype(np.float64),
                }
            )
        )
    jssp = load_jssp_directory(args.jssp_root.resolve())
    if not jssp.empty:
        frames.append(jssp)
    if not frames:
        raise ValueError("No per-instance data were supplied")
    frame = pd.concat(frames, ignore_index=True)

    duplicate_counts = frame.groupby(["problem", "method", "instance_id"]).size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if not duplicates.empty:
        raise ValueError(f"Duplicate per-instance rows detected:\n{duplicates.head()}")

    summaries = (
        frame.groupby(["problem", "method"])["cost"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .to_dict(orient="records")
    )
    comparisons: list[dict[str, Any]] = []
    for problem in sorted(frame["problem"].unique()):
        methods = set(frame.loc[frame["problem"] == problem, "method"])
        for ours in OURS:
            if ours not in methods:
                continue
            for baseline in HAND_DESIGNED:
                if baseline in methods:
                    comparisons.append(
                        compare(
                            frame,
                            problem=problem,
                            left=ours,
                            right=baseline,
                            seed=args.seed + len(comparisons),
                            bootstrap_samples=args.bootstrap_samples,
                        )
                    )
            for baseline in STANDARD_NEURAL:
                if baseline in methods:
                    comparisons.append(
                        compare(
                            frame,
                            problem=problem,
                            left=ours,
                            right=baseline,
                            seed=args.seed + len(comparisons),
                            bootstrap_samples=args.bootstrap_samples,
                        )
                    )
        if {"ASW", "USW"} <= methods:
            comparisons.append(
                compare(
                    frame,
                    problem=problem,
                    left="ASW",
                    right="USW",
                    seed=args.seed + len(comparisons),
                    bootstrap_samples=args.bootstrap_samples,
                )
            )

    wilcoxon_adjusted = holm_adjust([row["wilcoxon_pvalue"] for row in comparisons])
    t_adjusted = holm_adjust([row["paired_t_pvalue"] for row in comparisons])
    for row, wilcoxon_p, t_p in zip(comparisons, wilcoxon_adjusted, t_adjusted, strict=True):
        row["wilcoxon_holm_pvalue"] = wilcoxon_p
        row["paired_t_holm_pvalue"] = t_p
        row["significant_better_holm_0p05"] = bool(
            row["left_is_better"] and wilcoxon_p < 0.05
        )
        row["significant_worse_holm_0p05"] = bool(
            not row["left_is_better"] and wilcoxon_p < 0.05
        )
        row["holm_outcome"] = (
            "better"
            if row["significant_better_holm_0p05"]
            else "worse"
            if row["significant_worse_holm_0p05"]
            else "not_significant"
        )

    aggregate: dict[str, dict[str, int]] = {}
    for method in OURS:
        relevant = [
            row
            for row in comparisons
            if row["left_method"] == method and row["right_method"] in HAND_DESIGNED
        ]
        aggregate[method] = {
            "comparisons": len(relevant),
            "significant_wins": sum(row["holm_outcome"] == "better" for row in relevant),
            "significant_losses": sum(row["holm_outcome"] == "worse" for row in relevant),
            "not_significant": sum(
                row["holm_outcome"] == "not_significant" for row in relevant
            ),
        }

    problem_claims = []
    for problem in sorted(frame["problem"].unique()):
        methods = set(frame.loc[frame["problem"] == problem, "method"])
        record: dict[str, Any] = {
            "problem": problem,
            "available_hand_designed": [
                method for method in HAND_DESIGNED if method in methods
            ],
        }
        for ours in OURS:
            rows = [
                row
                for row in comparisons
                if row["problem"] == problem
                and row["left_method"] == ours
                and row["right_method"] in HAND_DESIGNED
            ]
            record[f"{ours.lower()}_beats_every_hand_designed"] = bool(
                rows and all(row["holm_outcome"] == "better" for row in rows)
            )
            record[f"{ours.lower()}_significant_wins"] = sum(
                row["holm_outcome"] == "better" for row in rows
            )
            record[f"{ours.lower()}_significant_losses"] = sum(
                row["holm_outcome"] == "worse" for row in rows
            )
        asw_usw = next(
            (
                row
                for row in comparisons
                if row["problem"] == problem
                and row["left_method"] == "ASW"
                and row["right_method"] == "USW"
            ),
            None,
        )
        record["asw_vs_usw"] = (
            asw_usw["holm_outcome"] if asw_usw is not None else "unavailable"
        )
        problem_claims.append(record)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "method_summaries.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "paired_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(comparisons[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)
    payload = {
        "protocol": {
            "primary_test": "two-sided paired Wilcoxon signed-rank test",
            "familywise_correction": "Holm correction over every reported pairwise comparison",
            "effect_size": "matched-pairs rank-biserial correlation",
            "interval": "95% percentile bootstrap confidence interval for mean paired difference",
            "lower_cost_is_better": True,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "problem_claims": problem_claims,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    (output_dir / "paired_significance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Paired significance analysis",
        "",
        "Lower cost is better. The primary test is a two-sided paired Wilcoxon "
        "signed-rank test on the same test instances, with a single Holm correction "
        "over every reported pairwise comparison. Mean paired differences use "
        "`left - right`; negative values favor the left method.",
        "",
        "## Aggregate candidate-objective conclusion",
        "",
    ]
    for method in OURS:
        counts = aggregate[method]
        report_lines.append(
            f"- {method}: {counts['significant_wins']}/{counts['comparisons']} "
            f"significant wins over hand-designed objectives, "
            f"{counts['significant_losses']} significant losses, and "
            f"{counts['not_significant']} non-significant comparisons."
        )
    report_lines.extend(
        [
            "",
            "## Per-problem claim audit",
            "",
            "| Problem | USW beats every hand-designed objective | "
            "ASW beats every hand-designed objective | ASW vs USW |",
            "|---|---:|---:|---|",
        ]
    )
    for record in problem_claims:
        report_lines.append(
            f"| {record['problem']} | "
            f"{'yes' if record['usw_beats_every_hand_designed'] else 'no'} | "
            f"{'yes' if record['asw_beats_every_hand_designed'] else 'no'} | "
            f"{record['asw_vs_usw']} |"
        )
    report_lines.extend(
        [
            "",
            "## Complete paired comparisons",
            "",
            "| Problem | Left | Right | n | Left mean | Right mean | Mean diff. | "
            "95% bootstrap CI | Holm-adjusted p | Outcome |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in comparisons:
        report_lines.append(
            f"| {row['problem']} | {row['left_method']} | {row['right_method']} | "
            f"{row['n']} | {row['left_mean']:.6f} | {row['right_mean']:.6f} | "
            f"{row['mean_difference_left_minus_right']:.6f} | "
            f"[{row['bootstrap_95ci_mean_difference_low']:.6f}, "
            f"{row['bootstrap_95ci_mean_difference_high']:.6f}] | "
            f"{row['wilcoxon_holm_pvalue']:.3e} | {row['holm_outcome']} |"
        )
    (output_dir / "statistical_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "problems": sorted(frame["problem"].unique().tolist()),
                "rows": len(frame),
                "comparisons": len(comparisons),
                "significant_better": sum(
                    row["significant_better_holm_0p05"] for row in comparisons
                ),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
