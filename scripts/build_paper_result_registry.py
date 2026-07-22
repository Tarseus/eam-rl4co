from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def rounding_tolerance(problem: str) -> float:
    if problem.startswith(("tsp", "cvrp")):
        return 0.00005
    if problem.startswith("ffsp"):
        return 0.0005
    if problem.startswith("jssp"):
        return 0.005
    return 0.00005


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join checkpoint provenance, fresh test means, and significance claims."
    )
    parser.add_argument(
        "--checkpoint-registry",
        type=Path,
        default=REPO_ROOT
        / "paper_materials"
        / "statistical_analysis"
        / "checkpoint_registry.json",
    )
    parser.add_argument("--significance-json", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_materials" / "statistical_analysis",
    )
    args = parser.parse_args()

    checkpoints = json.loads(args.checkpoint_registry.read_text(encoding="utf-8"))
    significance = json.loads(args.significance_json.read_text(encoding="utf-8"))
    means = {
        (str(row["problem"]), str(row["method"])): row
        for row in significance["summaries"]
    }
    comparisons = {
        (
            str(row["problem"]),
            str(row["left_method"]),
            str(row["right_method"]),
        ): row
        for row in significance.get("comparisons", [])
    }
    claims = {
        str(row["problem"]): row for row in significance.get("problem_claims", [])
    }
    hand_designed = ("PO4COPs", "SLL", "BOPO")
    best_hand_by_problem: dict[str, str] = {}
    for problem, _method in means:
        available = [
            (method, means[(problem, method)])
            for method in hand_designed
            if (problem, method) in means
        ]
        if available:
            best_hand_by_problem[problem] = min(
                available,
                key=lambda item: float(item[1]["mean"]),
            )[0]

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        problem = str(checkpoint["problem"])
        method = str(checkpoint["paper_method"])
        summary = means.get((problem, method))
        fresh_mean = float(summary["mean"]) if summary is not None else None
        main_cost = checkpoint.get("main_tex_cost")
        delta = (
            fresh_mean - float(main_cost)
            if fresh_mean is not None and main_cost is not None
            else None
        )
        claim = claims.get(problem, {})
        beats_all = None
        if method in {"USW", "ASW"}:
            beats_all = claim.get(f"{method.lower()}_beats_every_hand_designed")
        best_hand = best_hand_by_problem.get(problem)
        best_hand_comparison = (
            comparisons.get((problem, method, best_hand))
            if best_hand is not None and method in {"USW", "ASW"}
            else None
        )
        asw_usw_comparison = (
            comparisons.get((problem, "ASW", "USW"))
            if method == "ASW"
            else None
        )
        rows.append(
            {
                "problem": problem,
                "paper_method": method,
                "legacy_method": checkpoint["legacy_method"],
                "checkpoint": checkpoint["checkpoint"],
                "checkpoint_epoch": checkpoint.get("checkpoint_epoch"),
                "global_step": checkpoint.get("global_step"),
                "sha256": checkpoint["sha256"],
                "source_checkpoint": checkpoint.get("source_checkpoint"),
                "selection_basis": checkpoint.get("selection_basis"),
                "num_instances": int(summary["count"]) if summary is not None else None,
                "fresh_mean_cost": fresh_mean,
                "main_tex_cost_before_audit": main_cost,
                "fresh_minus_main": delta,
                "main_value_matches_fresh_rounding": (
                    abs(delta) <= rounding_tolerance(problem)
                    if delta is not None
                    else None
                ),
                "beats_every_hand_designed_holm_0p05": beats_all,
                "best_mean_hand_designed": best_hand,
                "vs_best_hand_holm_outcome": (
                    best_hand_comparison["holm_outcome"]
                    if best_hand_comparison is not None
                    else None
                ),
                "vs_best_hand_holm_pvalue": (
                    float(best_hand_comparison["wilcoxon_holm_pvalue"])
                    if best_hand_comparison is not None
                    else None
                ),
                "asw_vs_usw_holm_outcome": (
                    asw_usw_comparison["holm_outcome"]
                    if asw_usw_comparison is not None
                    else None
                ),
                "asw_vs_usw_holm_pvalue": (
                    float(asw_usw_comparison["wilcoxon_holm_pvalue"])
                    if asw_usw_comparison is not None
                    else None
                ),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paper_result_registry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Paper result and checkpoint registry",
        "",
        "Fresh means come from fixed, per-instance evaluations. The old `main.tex` "
        "value is retained as an audit field; it is not treated as ground truth.",
        "",
        "| Problem | Method | Fresh mean | Old main.tex | Match | Epoch | "
        "Checkpoint | vs best-mean hand objective | ASW vs USW |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        fresh = (
            f"{row['fresh_mean_cost']:.6f}"
            if row["fresh_mean_cost"] is not None
            else "--"
        )
        old = (
            f"{float(row['main_tex_cost_before_audit']):.6f}"
            if row["main_tex_cost_before_audit"] is not None
            else "--"
        )
        match = (
            "yes"
            if row["main_value_matches_fresh_rounding"] is True
            else "no"
            if row["main_value_matches_fresh_rounding"] is False
            else "--"
        )
        vs_hand = (
            f"{row['vs_best_hand_holm_outcome']} "
            f"({row['best_mean_hand_designed']}, "
            f"p={row['vs_best_hand_holm_pvalue']:.3e})"
            if row["vs_best_hand_holm_outcome"] is not None
            else "--"
        )
        asw_vs_usw = (
            f"{row['asw_vs_usw_holm_outcome']} "
            f"(p={row['asw_vs_usw_holm_pvalue']:.3e})"
            if row["asw_vs_usw_holm_outcome"] is not None
            else "--"
        )
        lines.append(
            f"| {row['problem']} | {row['paper_method']} | {fresh} | {old} | "
            f"{match} | {row['checkpoint_epoch']} | `{row['checkpoint']}` | "
            f"{vs_hand} | {asw_vs_usw} |"
        )
    (output_dir / "paper_result_registry.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"records": len(rows), "output": str(csv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
