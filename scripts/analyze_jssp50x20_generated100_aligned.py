from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


LABELS = ("po", "bopo", "h8_usw", "h13_asw")
SUCCESS_STATUSES = {"optimal", "feasible"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize the locked JSSP50x20 generated-100 evaluation."
    )
    parser.add_argument("--neural-root", type=Path, required=True)
    parser.add_argument("--cp-sat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bootstrap_mean_ci(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    args = _parse_args()
    neural_root = args.neural_root.resolve()
    cp_sat_dir = args.cp_sat_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite aligned analysis: {output_dir}")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    summaries: dict[str, dict[str, object]] = {}
    rows_by_label: dict[str, dict[str, dict[str, str]]] = {}
    expected_instances: list[str] | None = None
    expected_dataset_sha: str | None = None
    expected_hashes: dict[str, str] | None = None

    for label in LABELS:
        label_dir = neural_root / label
        summary = json.loads((label_dir / "summary.json").read_text(encoding="utf-8"))
        rows = _read_csv(label_dir / "per_instance.csv")
        if summary["protocol"] != "jssp50x20_paper_aligned_generated100_b128_g0_v1":
            raise ValueError(f"Protocol mismatch for {label}: {summary['protocol']}")
        if (
            int(summary["instance_count"]),
            int(summary["sampling_seed_base"]),
            int(summary["B"]),
            bool(summary["greedy_injected"]),
            int(summary["augmentation_factor"]),
        ) != (100, 12345678, 128, False, 1):
            raise ValueError(f"Locked neural settings mismatch for {label}")
        if len(rows) != 100:
            raise ValueError(f"Expected 100 neural rows for {label}, found {len(rows)}")

        rows_by_instance = {row["instance"]: row for row in rows}
        if len(rows_by_instance) != 100:
            raise ValueError(f"Duplicate neural instances for {label}")
        instances = [row["instance"] for row in rows]
        hashes = {row["instance"]: row["instance_sha256"] for row in rows}
        seeds = [int(row["sampling_seed"]) for row in rows]
        if seeds != list(range(12345678, 12345678 + 100)):
            raise ValueError(f"Per-instance sampling seeds mismatch for {label}")
        if expected_instances is None:
            expected_instances = instances
            expected_dataset_sha = str(summary["dataset_order_sha256"])
            expected_hashes = hashes
        elif (
            instances != expected_instances
            or str(summary["dataset_order_sha256"]) != expected_dataset_sha
            or hashes != expected_hashes
        ):
            raise ValueError(f"Dataset/order/hash mismatch for {label}")
        summaries[label] = summary
        rows_by_label[label] = rows_by_instance

    assert expected_instances is not None
    assert expected_dataset_sha is not None

    cp_rows = _read_csv(cp_sat_dir / "per_instance.csv")
    if len(cp_rows) != 100:
        raise ValueError(f"Expected 100 CP-SAT rows, found {len(cp_rows)}")
    cp_by_instance = {row["instance_id"]: row for row in cp_rows}
    if len(cp_by_instance) != 100 or set(cp_by_instance) != set(expected_instances):
        raise ValueError("CP-SAT and neural instance sets do not match exactly")
    invalid_cp = [
        row for row in cp_rows
        if row["status"] not in SUCCESS_STATUSES or not row["objective"]
    ]
    if invalid_cp:
        raise ValueError(f"CP-SAT has {len(invalid_cp)} failed or objective-free rows")

    reference = np.array(
        [float(cp_by_instance[name]["objective"]) for name in expected_instances],
        dtype=np.float64,
    )
    if np.max(np.abs(reference - np.rint(reference))) > 0.01:
        raise ValueError("CP-SAT makespans are unexpectedly non-integral")
    reference = np.rint(reference)
    reference_mean = float(reference.mean())
    raw_method_costs = {
        label: np.array(
            [float(rows_by_label[label][name]["cost"]) for name in expected_instances],
            dtype=np.float64,
        )
        for label in LABELS
    }
    if any(np.max(np.abs(values - np.rint(values))) > 0.01 for values in raw_method_costs.values()):
        raise ValueError("Neural makespans are unexpectedly non-integral")
    method_costs = {label: np.rint(values) for label, values in raw_method_costs.items()}

    methods: list[dict[str, object]] = []
    for label in LABELS:
        costs = method_costs[label]
        methods.append(
            {
                "label": label,
                "mean_cost": float(costs.mean()),
                "std_cost": float(costs.std(ddof=1)),
                "median_cost": float(np.median(costs)),
                "gap_percent_ratio_of_means": float((costs.mean() / reference_mean - 1.0) * 100.0),
                "mean_per_instance_gap_percent": float(np.mean((costs / reference - 1.0) * 100.0)),
                "total_elapsed_sec": float(summaries[label]["total_elapsed_sec"]),
                "checkpoint_sha256": summaries[label]["checkpoint_sha256"],
                "optimizer_step": int(summaries[label]["optimizer_step"]),
            }
        )

    comparisons: list[dict[str, object]] = []
    bopo_costs = method_costs["bopo"]
    for offset, label in enumerate(("po", "h8_usw", "h13_asw")):
        delta = method_costs[label] - bopo_costs
        ci_low, ci_high = _bootstrap_mean_ci(
            delta,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + offset,
        )
        nonzero = delta[delta != 0]
        wilcoxon_p = (
            float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)
            if len(nonzero)
            else 1.0
        )
        comparisons.append(
            {
                "comparison": f"{label}_minus_bopo",
                "mean_delta_cost": float(delta.mean()),
                "bootstrap_95_ci_mean_delta": [ci_low, ci_high],
                "wilcoxon_two_sided_p": wilcoxon_p,
                "wins": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
                "losses": int(np.sum(delta > 0)),
            }
        )

    cp_summary_payload = json.loads((cp_sat_dir / "summary.json").read_text(encoding="utf-8"))
    cp_summary_records = cp_summary_payload.get("records", [])
    if len(cp_summary_records) != 1:
        raise ValueError(f"Expected one CP-SAT summary record, found {len(cp_summary_records)}")
    cp_record = cp_summary_records[0]

    payload = {
        "protocol": "jssp50x20_paper_aligned_generated100_b128_g0_v1",
        "instance_count": 100,
        "dataset_order_sha256": expected_dataset_sha,
        "gap_definition": "(mean_method / mean_cp_sat_reference - 1) * 100%",
        "cp_sat": {
            "mean_reference": reference_mean,
            "optimal_count": int(sum(row["status"] == "optimal" for row in cp_rows)),
            "feasible_count": int(sum(row["status"] == "feasible" for row in cp_rows)),
            "failed_count": 0,
            "time_limit_sec_per_instance": 600,
            "outer_workers": int(cp_record["workers"]),
            "search_workers_per_instance": int(cp_record["solver_threads"]),
            "wall_clock_sec": float(cp_record["total_elapsed_s"]),
            "sum_instance_elapsed_sec": float(cp_record["sum_instance_elapsed_s"]),
        },
        "methods": methods,
        "paired_against_bopo": comparisons,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "aligned_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "aligned_per_instance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["instance", "instance_sha256", "cp_sat_status", "cp_sat_cost", *LABELS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in expected_instances:
            writer.writerow(
                {
                    "instance": name,
                    "instance_sha256": rows_by_label["po"][name]["instance_sha256"],
                    "cp_sat_status": cp_by_instance[name]["status"],
                    "cp_sat_cost": cp_by_instance[name]["objective"],
                    **{label: rows_by_label[label][name]["cost"] for label in LABELS},
                }
            )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
