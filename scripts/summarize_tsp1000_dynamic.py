from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_rows(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    grouped: dict[str, dict[int, float]] = {}
    for row in summary["rows"]:
        grouped.setdefault(str(row["method"]), {})[int(row["instance_index"])] = float(
            row["tour_length"]
        )
    result = {}
    for method, indexed in grouped.items():
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError(f"Non-contiguous instance indices for {method}")
        result[method] = np.asarray([indexed[index] for index in expected], dtype=np.float64)
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> list[float]:
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--po-bopo-summary", type=Path, required=True)
    parser.add_argument("--usw-asw-summary", type=Path, required=True)
    parser.add_argument("--lkh-summary", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--zero-shot-po-summary", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    method_rows = {
        **_method_rows(_load_json(args.po_bopo_summary)),
        **_method_rows(_load_json(args.usw_asw_summary)),
    }
    if set(method_rows) != {"po", "bopo", "usw", "asw"}:
        raise ValueError(f"Unexpected methods: {sorted(method_rows)}")
    row_counts = {method: len(values) for method, values in method_rows.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Mismatched row counts: {row_counts}")

    aggregates = {
        method: {
            "num_instances": int(len(values)),
            "mean_tour_length": float(values.mean()),
            "std_tour_length": float(values.std(ddof=1)),
        }
        for method, values in method_rows.items()
    }
    comparisons = {}
    for better, baseline in (
        ("usw", "po"),
        ("asw", "po"),
        ("usw", "bopo"),
        ("asw", "bopo"),
        ("asw", "usw"),
        ("po", "bopo"),
    ):
        improvement = method_rows[baseline] - method_rows[better]
        key = f"{better}_over_{baseline}"
        comparisons[key] = {
            "mean_tour_length_improvement": float(improvement.mean()),
            "relative_improvement_percent": float(
                100.0 * improvement.mean() / method_rows[baseline].mean()
            ),
            "bootstrap_95ci_mean_improvement": _bootstrap_mean_ci(
                improvement,
                seed=args.seed,
                samples=args.bootstrap_samples,
            ),
            "fraction_instances_improved": float((improvement > 0).mean()),
        }

    lkh_payload = _load_json(args.lkh_summary)
    lkh_indexed = {
        int(row["instance_index"]): float(row["tour_length"])
        for row in lkh_payload["rows"]
        if row.get("status") == "ok" and row.get("tour_length") is not None
    }
    lkh_count = len(lkh_indexed)
    if sorted(lkh_indexed) != list(range(lkh_count)):
        raise ValueError("LKH instance indices must be contiguous from zero")
    if lkh_count > len(method_rows["po"]):
        raise ValueError("LKH summary has more instances than the model evaluation")
    lkh = np.asarray(
        [lkh_indexed[index] for index in range(lkh_count)],
        dtype=np.float64,
    )
    lkh_gaps = {}
    for method, values in method_rows.items():
        gaps = 100.0 * (values[:lkh_count] - lkh) / lkh
        lkh_gaps[method] = {
            "num_instances": int(lkh_count),
            "mean_gap_percent": float(gaps.mean()),
            "std_gap_percent": float(gaps.std(ddof=1)),
            "bootstrap_95ci_mean_gap_percent": _bootstrap_mean_ci(
                gaps,
                seed=args.seed,
                samples=args.bootstrap_samples,
            ),
        }

    training_selection = {}
    for method in ("po", "bopo", "usw", "asw"):
        summary = _load_json(args.training_root / method / "summary.json")
        training_selection[method] = {
            "best_mean_tour_length": float(summary["best_mean_tour_length"]),
            "best_step": int(summary["best_step"]),
            "completed_steps": int(summary["completed_steps"]),
        }

    zero_shot_po = None
    if args.zero_shot_po_summary is not None:
        zero_shot_payload = _load_json(args.zero_shot_po_summary)
        zero_shot_values = _method_rows(zero_shot_payload)["po"]
        zero_shot_count = len(zero_shot_values)
        if zero_shot_count > len(method_rows["po"]):
            raise ValueError("Zero-shot summary has more instances than model evaluation")
        zero_shot_mean = float(zero_shot_values.mean())
        zero_shot_po = {
            "num_instances": int(zero_shot_count),
            "mean_tour_length": zero_shot_mean,
            "dynamic_method_improvement": {
                method: {
                    "absolute": float(
                        (zero_shot_values - values[:zero_shot_count]).mean()
                    ),
                    "relative_percent": float(
                        100.0
                        * (zero_shot_values - values[:zero_shot_count]).mean()
                        / zero_shot_mean
                    ),
                    "bootstrap_95ci_mean_improvement": _bootstrap_mean_ci(
                        zero_shot_values - values[:zero_shot_count],
                        seed=args.seed,
                        samples=args.bootstrap_samples,
                    ),
                }
                for method, values in method_rows.items()
            },
        }

    payload = {
        "protocol": {
            "target_size": 1000,
            "num_instances": int(len(method_rows["po"])),
            "num_starts": 1000,
            "num_augment": 8,
            "precision": "32-true",
            "paired_seed": int(args.seed),
            "checkpoint_selection": "best fixed-validation checkpoint",
            "lkh_reference_instances": int(lkh_count),
        },
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "lkh_gaps": lkh_gaps,
        "training_selection": training_selection,
        "common_initialization_zero_shot": zero_shot_po,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
