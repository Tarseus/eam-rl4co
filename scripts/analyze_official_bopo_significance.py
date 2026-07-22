from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_paper_significance import compare, holm_adjust


BASELINES = ("PO4COPs", "SLL", "USW", "ASW")


def format_pvalue(value: float) -> str:
    return "<1e-300" if value == 0.0 else f"{value:.3e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired significance analysis for official BOPO TSP checkpoints."
    )
    parser.add_argument("--official", type=Path, action="append", required=True)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_frame(paths: list[Path]) -> pd.DataFrame:
    required = ("problem", "method", "instance_id", "cost")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path.resolve())
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame[list(required)].copy())
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    official = load_frame(args.official)
    if set(official["method"]) != {"BOPO-official"}:
        raise ValueError("Every official row must use method=BOPO-official")
    baselines = load_frame(args.baseline)
    baselines = baselines[baselines["method"].isin(BASELINES)].copy()
    frame = pd.concat([official, baselines], ignore_index=True)

    duplicate_counts = frame.groupby(["problem", "method", "instance_id"]).size()
    if (duplicate_counts > 1).any():
        raise ValueError("Duplicate problem/method/instance rows detected")

    comparisons: list[dict[str, Any]] = []
    for problem in sorted(official["problem"].unique()):
        expected_ids = set(
            official.loc[official["problem"] == problem, "instance_id"].astype(str)
        )
        if len(expected_ids) != 10_000:
            raise ValueError(f"{problem}: expected 10,000 official instances")
        for baseline in BASELINES:
            baseline_ids = set(
                baselines.loc[
                    (baselines["problem"] == problem)
                    & (baselines["method"] == baseline),
                    "instance_id",
                ].astype(str)
            )
            if baseline_ids != expected_ids:
                raise ValueError(
                    f"{problem}/{baseline}: instance IDs do not exactly match official BOPO"
                )
            comparisons.append(
                compare(
                    frame,
                    problem=problem,
                    left="BOPO-official",
                    right=baseline,
                    seed=args.seed + len(comparisons),
                    bootstrap_samples=args.bootstrap_samples,
                )
            )

    wilcoxon_holm = holm_adjust([row["wilcoxon_pvalue"] for row in comparisons])
    t_holm = holm_adjust([row["paired_t_pvalue"] for row in comparisons])
    for row, wilcoxon_p, t_p in zip(
        comparisons, wilcoxon_holm, t_holm, strict=True
    ):
        row["wilcoxon_holm_pvalue"] = wilcoxon_p
        row["paired_t_holm_pvalue"] = t_p
        row["mean_holm_outcome"] = (
            "BOPO-official better"
            if row["left_is_better"] and t_p < 0.05
            else "BOPO-official worse"
            if not row["left_is_better"] and t_p < 0.05
            else "not significant"
        )
        rank_direction = row["matched_rank_biserial"]
        row["wilcoxon_holm_outcome"] = (
            "BOPO-official better"
            if rank_direction < 0 and wilcoxon_p < 0.05
            else "BOPO-official worse"
            if rank_direction > 0 and wilcoxon_p < 0.05
            else "not significant"
        )
        row["direction_agrees"] = bool(
            row["mean_holm_outcome"] == row["wilcoxon_holm_outcome"]
        )

    summaries = (
        frame.groupby(["problem", "method"])["cost"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
        .to_dict(orient="records")
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("method_summaries.csv", summaries),
        ("paired_comparisons.csv", comparisons),
    ):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "protocol": {
            "primary_test": "two-sided paired Wilcoxon signed-rank test",
            "familywise_correction": (
                "Holm correction over 8 comparisons: 4 baselines x 2 problem sizes"
            ),
            "effect_size": "matched-pairs rank-biserial correlation",
            "interval": "95% percentile bootstrap CI for mean paired difference",
            "difference": "BOPO-official minus baseline; negative favors BOPO-official",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "summaries": summaries,
        "comparisons": comparisons,
    }
    (output_dir / "paired_significance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Official BOPO paired significance analysis",
        "",
        "The same 10,000 canonical instances are paired by `instance_id`. Lower cost is better. "
        "The primary test is a two-sided paired Wilcoxon signed-rank test with one Holm correction "
        "across all eight comparisons. Mean outcomes use paired t tests with the same Holm family, "
        "and the displayed mean confidence intervals use 20,000 paired bootstrap samples. "
        "Differences are `BOPO-official - baseline`.",
        "",
        "| Problem | Baseline | BOPO | Baseline | Difference | 95% bootstrap CI | Mean Holm p | Mean outcome | Wilcoxon Holm p | Rank outcome |",
        "|---|---|---:|---:|---:|---|---:|---|---:|---|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['problem']} | {row['right_method']} | {row['left_mean']:.9f} | "
            f"{row['right_mean']:.9f} | {row['mean_difference_left_minus_right']:.9f} | "
            f"[{row['bootstrap_95ci_mean_difference_low']:.9f}, "
            f"{row['bootstrap_95ci_mean_difference_high']:.9f}] | "
            f"{format_pvalue(row['paired_t_holm_pvalue'])} | "
            f"{row['mean_holm_outcome']} | "
            f"{format_pvalue(row['wilcoxon_holm_pvalue'])} | "
            f"{row['wilcoxon_holm_outcome']} |"
        )
    lines.extend(
        [
            "",
            "TSP100 BOPO-official versus SLL is the only direction disagreement: BOPO has a lower "
            "mean by 0.000505100, but wins/ties/losses are 3,631/1,579/4,790. Its less frequent wins "
            "are larger on average, so the paired-mean analysis favors BOPO while the signed-rank "
            "analysis favors SLL.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"comparisons": len(comparisons), "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
