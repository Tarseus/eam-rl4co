from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "paper_materials"
    / "statistical_analysis"
    / "final_results"
    / "paired_significance.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "paper_materials"
    / "statistical_analysis"
    / "significance_appendix.tex"
)

GROUPS = (
    ("routing-tsp", "Routing significance results for TSP.", ("tsp100", "tsp50")),
    (
        "routing-cvrp",
        "Routing significance results for CVRP.",
        ("cvrp100", "cvrp50"),
    ),
    (
        "scheduling-ffsp",
        "Scheduling significance results for FFSP.",
        ("ffsp100", "ffsp50"),
    ),
    (
        "scheduling-jssp",
        "Scheduling significance results for JSSP.",
        ("jssp15x15", "jssp10x10"),
    ),
)


def problem_label(problem: str) -> str:
    labels = {
        "tsp50": "TSP50",
        "tsp100": "TSP100",
        "cvrp50": "CVRP50",
        "cvrp100": "CVRP100",
        "ffsp50": "FFSP50",
        "ffsp100": "FFSP100",
        "jssp10x10": r"JSSP10$\times$10",
        "jssp15x15": r"JSSP15$\times$15",
    }
    return labels[problem]


def format_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.2f}"
    if magnitude >= 10:
        return f"{value:.3f}"
    return f"{value:.6f}"


def format_pvalue(value: float) -> str:
    if value == 0:
        return r"$<10^{-300}$"
    if value >= 0.001:
        return f"{value:.3f}"
    exponent = int(f"{value:.1e}".split("e")[1])
    coefficient = value / (10**exponent)
    return rf"${coefficient:.2f}\times10^{{{exponent}}}$"


def outcome_label(outcome: str) -> str:
    return {
        "better": "better",
        "worse": "worse",
        "not_significant": "ns",
    }[outcome]


def table_lines(
    *,
    label_suffix: str,
    caption: str,
    problems: tuple[str, ...],
    comparisons: list[dict[str, Any]],
) -> list[str]:
    selected = [
        row for row in comparisons if str(row["problem"]) in set(problems)
    ]
    order = {problem: index for index, problem in enumerate(problems)}
    selected.sort(key=lambda row: order[str(row["problem"])])
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{caption} Mean difference is left minus right, so negative values and negative rank-biserial correlations favor the left method.}}",
        rf"\label{{tab:paired-significance-{label_suffix}}}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrl}",
        r"\toprule",
        r"Problem & Left & Right & $n$ & Mean diff. & 95\% bootstrap CI & Holm $p$ & Rank-biserial & Outcome \\",
        r"\midrule",
    ]
    previous_problem = None
    for row in selected:
        problem = str(row["problem"])
        if previous_problem is not None and problem != previous_problem:
            lines.append(r"\midrule")
        diff = float(row["mean_difference_left_minus_right"])
        low = float(row["bootstrap_95ci_mean_difference_low"])
        high = float(row["bootstrap_95ci_mean_difference_high"])
        effect = float(row["matched_rank_biserial"])
        lines.append(
            f"{problem_label(problem)} & {row['left_method']} & {row['right_method']} & "
            f"{int(row['n'])} & {format_number(diff)} & "
            f"[{format_number(low)}, {format_number(high)}] & "
            f"{format_pvalue(float(row['wilcoxon_holm_pvalue']))} & "
            f"{effect:.3f} & {outcome_label(str(row['holm_outcome']))} \\\\"
        )
        previous_problem = problem
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate appendix-ready LaTeX tables for paired paper statistics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    comparisons = list(payload["comparisons"])
    comparison_count = len(comparisons)
    lines = [
        r"\section{Paired Statistical Analysis}",
        r"\label{app:paired-significance}",
        "",
        "All candidate-pool objective comparisons use costs paired on identical "
        "test instances. FFSP evaluations additionally reset the augmentation "
        "random stream before each checkpoint. We use a two-sided paired "
        f"Wilcoxon signed-rank test and apply one Holm correction over all {comparison_count} "
        "comparisons in Tables~\\ref{tab:routing-results} "
        "and~\\ref{tab:scheduling-results}. The tables below also report a "
        "20,000-sample percentile-bootstrap 95\\% confidence interval for the "
        "mean paired difference and the matched-pairs rank-biserial effect "
        "size. The recovered CVRP50 SymNCO checkpoint is included inferentially; "
        "other reference and standard-neural-solver rows remain descriptive "
        "because matching per-instance artifacts were not recovered for every "
        "solver.",
        "",
    ]
    for label_suffix, caption, problems in GROUPS:
        lines.extend(
            table_lines(
                label_suffix=label_suffix,
                caption=caption,
                problems=problems,
                comparisons=comparisons,
            )
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "comparisons": len(comparisons),
                "tables": len(GROUPS),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
