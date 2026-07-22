from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


JSSP_LABELS = ("rl", "po", "sll", "bopo", "h8_usw", "h13_asw")
FFSP_LABELS = ("rl", "po", "sll", "bopo", "usw", "asw")
DISPLAY = {
    "rl": "RL backbone",
    "po": "PO4COPs",
    "sll": "SLL",
    "bopo": "BOPO",
    "usw": "USW",
    "asw": "ASW",
    "h8_usw": "USW (H8)",
    "h13_asw": "H13-ASW",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 10_000):
        stop = min(start + 10_000, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def _require_optimal_reference(root: Path, key_field: str) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("all_optimal") or int(summary.get("optimal_count", -1)) != 100:
        raise ValueError(f"Refusing paper analysis: CP-SAT is not 100/100 optimal at {root}")
    rows = _read_csv(root / "per_instance.csv")
    if len(rows) != 100 or any(row["status"] != "optimal" for row in rows):
        raise ValueError(f"CP-SAT row audit failed at {root}")
    keys = [row[key_field] for row in rows]
    if len(set(keys)) != 100:
        raise ValueError(f"Duplicate CP-SAT instance keys at {root}")
    costs = np.asarray([float(row["objective"]) for row in rows], dtype=np.float64)
    return keys, costs, summary


def _load_neural(
    root: Path,
    labels: tuple[str, ...],
    key_field: str,
    expected_keys: list[str],
    expected_protocol: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    costs: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for label in labels:
        summary = json.loads((root / label / "summary.json").read_text(encoding="utf-8"))
        rows = _read_csv(root / label / "per_instance.csv")
        if summary["protocol"] != expected_protocol or int(summary["instance_count"]) != 100:
            raise ValueError(f"Neural protocol mismatch for {root / label}")
        by_key = {row[key_field]: row for row in rows}
        if len(rows) != 100 or set(by_key) != set(expected_keys):
            raise ValueError(f"Neural instance alignment mismatch for {root / label}")
        costs[label] = np.asarray(
            [float(by_key[key]["cost"]) for key in expected_keys], dtype=np.float64
        )
        summaries[label] = summary
    return costs, summaries


def _problem_summary(
    name: str,
    labels: tuple[str, ...],
    reference: np.ndarray,
    cp_summary: dict[str, Any],
    costs: dict[str, np.ndarray],
    summaries: dict[str, dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    reference_mean = float(reference.mean())
    methods = []
    for label in labels:
        values = costs[label]
        methods.append(
            {
                "label": label,
                "display": DISPLAY[label],
                "mean_cost": float(values.mean()),
                "std_cost": float(values.std(ddof=1)),
                "gap_percent_ratio_of_means": float((values.mean() / reference_mean - 1) * 100),
                "mean_per_instance_gap_percent": float(np.mean((values / reference - 1) * 100)),
                "total_elapsed_sec": float(summaries[label]["total_elapsed_sec"]),
                "mean_elapsed_sec_per_instance": float(
                    summaries[label]["mean_elapsed_sec_per_instance"]
                ),
                "checkpoint_sha256": summaries[label]["checkpoint_sha256"],
                "optimizer_step": int(summaries[label]["optimizer_step"]),
            }
        )
    comparisons = []
    bopo = costs["bopo"]
    preference_labels = labels[labels.index("bopo") + 1 :]
    for offset, label in enumerate(preference_labels):
        delta = costs[label] - bopo
        nonzero = delta[delta != 0]
        comparisons.append(
            {
                "comparison": f"{label}_minus_bopo",
                "mean_delta_cost": float(delta.mean()),
                "bootstrap_95_ci_mean_delta": _bootstrap_ci(
                    delta, 20260720 + offset, bootstrap_samples
                ),
                "wilcoxon_two_sided_p": (
                    float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)
                    if len(nonzero)
                    else 1.0
                ),
                "wins": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
                "losses": int(np.sum(delta > 0)),
            }
        )
    return {
        "problem": name,
        "instance_count": 100,
        "cp_sat": {
            "mean_cost": reference_mean,
            "optimal_count": 100,
            "sum_solver_elapsed_sec": float(cp_summary["sum_solver_elapsed_sec"]),
        },
        "methods": methods,
        "paired_against_bopo": comparisons,
    }


def _latex_cell(value: float, best: bool, digits: int) -> str:
    rendered = f"{value:.{digits}f}"
    return rf"\textbf{{{rendered}}}" if best else rendered


def _write_latex_table(output_dir: Path, payload: dict[str, Any]) -> None:
    ffsp = payload["ffsp1000"]
    jssp = payload["jssp50x20"]
    ffsp_methods = {row["label"]: row for row in ffsp["methods"]}
    jssp_methods = {row["label"]: row for row in jssp["methods"]}
    preference_labels = ("po", "sll", "bopo", "usw", "asw")
    best_ffsp = min(preference_labels, key=lambda label: ffsp_methods[label]["mean_cost"])
    jssp_preference_labels = ("po", "sll", "bopo", "h8_usw", "h13_asw")
    best_jssp = min(
        jssp_preference_labels, key=lambda label: jssp_methods[label]["mean_cost"]
    )

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & \multicolumn{3}{c}{FFSP1000 (H)} & \multicolumn{3}{c}{JSSP50$\times$20 (H)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"& Cost $\downarrow$ & Gap (\%) $\downarrow$ & Time & Cost $\downarrow$ & Gap (\%) $\downarrow$ & Time \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Reference solver}}\\",
        (
            f"CP-SAT & {ffsp['cp_sat']['mean_cost']:.2f} & 0.000 & "
            f"{round(ffsp['cp_sat']['sum_solver_elapsed_sec'])}s & "
            f"{jssp['cp_sat']['mean_cost']:.2f} & 0.000 & "
            f"{round(jssp['cp_sat']['sum_solver_elapsed_sec'])}s \\\\"
        ),
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Standard neural solver}}\\",
    ]

    def method_line(display: str, ffsp_label: str, jssp_label: str) -> str:
        frow = ffsp_methods[ffsp_label]
        jrow = jssp_methods[jssp_label]
        return (
            f"{display} & {_latex_cell(frow['mean_cost'], ffsp_label == best_ffsp, 2)} & "
            f"{_latex_cell(frow['gap_percent_ratio_of_means'], ffsp_label == best_ffsp, 3)} & "
            f"{round(frow['total_elapsed_sec'])}s & "
            f"{_latex_cell(jrow['mean_cost'], jssp_label == best_jssp, 2)} & "
            f"{_latex_cell(jrow['gap_percent_ratio_of_means'], jssp_label == best_jssp, 3)} & "
            f"{round(jrow['total_elapsed_sec'])}s \\\\"
        )

    lines.extend(
        [
            method_line("RL backbone", "rl", "rl"),
            r"\midrule",
            r"\multicolumn{7}{l}{\textit{Hand-designed preference objectives}}\\",
            method_line("PO4COPs", "po", "po"),
            method_line("SLL", "sll", "sll"),
            method_line("BOPO", "bopo", "bopo"),
            r"\midrule",
            r"\multicolumn{7}{l}{\textit{Discovered preference objectives}}\\",
            method_line("USW", "usw", "h8_usw"),
            method_line("ASW", "asw", "h13_asw"),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (output_dir / "large_scale_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit complete FFSP1000/JSSP50x20 paper rows.")
    parser.add_argument("--jssp-neural-root", type=Path, required=True)
    parser.add_argument("--jssp-cp-root", type=Path, required=True)
    parser.add_argument("--ffsp-neural-root", type=Path, required=True)
    parser.add_argument("--ffsp-cp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite complete-row analysis: {output_dir}")
    j_keys, j_ref, j_cp = _require_optimal_reference(args.jssp_cp_root.resolve(), "instance")
    f_keys, f_ref, f_cp = _require_optimal_reference(args.ffsp_cp_root.resolve(), "instance_index")
    j_costs, j_summaries = _load_neural(
        args.jssp_neural_root.resolve(),
        JSSP_LABELS,
        "instance",
        j_keys,
        "jssp50x20_paper_aligned_generated100_b128_g0_v1",
    )
    f_costs, f_summaries = _load_neural(
        args.ffsp_neural_root.resolve(),
        FFSP_LABELS,
        "instance_index",
        f_keys,
        "ffsp1000_fixed_generated100_seed12345678_starts24_aug1_v1",
    )
    payload = {
        "protocol": "complete_large_scale_scheduling_rows_v1",
        "gap_definition": "(mean_method / mean_proven_optimal_cp_sat - 1) * 100%",
        "ffsp1000": _problem_summary(
            "ffsp1000", FFSP_LABELS, f_ref, f_cp, f_costs, f_summaries, args.bootstrap_samples
        ),
        "jssp50x20": _problem_summary(
            "jssp50x20", JSSP_LABELS, j_ref, j_cp, j_costs, j_summaries, args.bootstrap_samples
        ),
        "bootstrap_samples": args.bootstrap_samples,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_latex_table(output_dir, payload)

    for problem, keys, reference, costs, labels in (
        ("ffsp1000", f_keys, f_ref, f_costs, FFSP_LABELS),
        ("jssp50x20", j_keys, j_ref, j_costs, JSSP_LABELS),
    ):
        with (output_dir / f"{problem}_per_instance.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fields = ["instance", "cp_sat", *labels]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, key in enumerate(keys):
                writer.writerow(
                    {
                        "instance": key,
                        "cp_sat": reference[index],
                        **{label: costs[label][index] for label in labels},
                    }
                )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
