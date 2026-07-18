from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def _read(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows in {path}, got {len(rows)}")
    keyed = {(row["set"], row["instance"]): row for row in rows}
    if len(keyed) != 20:
        raise ValueError(f"Duplicate instance keys in {path}")
    return keyed


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 2000):
        stop = min(start + 2000, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def _holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=raw.__getitem__)
    adjusted = [math.nan] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bopo", type=Path, required=True)
    parser.add_argument("--h8-usw", type=Path, required=True)
    parser.add_argument("--h13-asw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.bootstrap_samples != 20000 or args.bootstrap_seed != 20260718:
        raise ValueError("Locked analysis requires 20,000 bootstraps and seed 20260718")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite locked analysis: {args.output}")
    rows = {
        "bopo": _read(args.bopo.resolve()),
        "h8_usw": _read(args.h8_usw.resolve()),
        "h13_asw": _read(args.h13_asw.resolve()),
    }
    keys = sorted(rows["bopo"])
    if any(sorted(payload) != keys for payload in rows.values()):
        raise ValueError("Instance keys do not align")
    for key in keys:
        seeds = {payload[key]["sampling_seed"] for payload in rows.values()}
        refs = {payload[key]["reference_makespan"] for payload in rows.values()}
        if len(seeds) != 1 or len(refs) != 1:
            raise ValueError(f"Paired seed/reference mismatch for {key}")

    comparisons = []
    raw_p = []
    for index, label in enumerate(("h8_usw", "h13_asw")):
        candidate = np.asarray(
            [float(rows[label][key]["gap_percent"]) for key in keys], dtype=np.float64
        )
        baseline = np.asarray(
            [float(rows["bopo"][key]["gap_percent"]) for key in keys],
            dtype=np.float64,
        )
        differences = candidate - baseline
        if not np.isfinite(differences).all():
            raise FloatingPointError(f"Non-finite paired differences for {label}")
        test = wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        p_value = float(test.pvalue)
        raw_p.append(p_value)
        comparisons.append(
            {
                "candidate": label,
                "baseline": "bopo",
                "n": 20,
                "candidate_mean_gap_percent": float(candidate.mean()),
                "baseline_mean_gap_percent": float(baseline.mean()),
                "candidate_minus_bopo_mean_gap_pp": float(differences.mean()),
                "bootstrap_95ci_mean_gap_difference_pp": _bootstrap(
                    differences,
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + index,
                ),
                "wins": int((differences < 0).sum()),
                "ties": int((differences == 0).sum()),
                "losses": int((differences > 0).sum()),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_raw_p": p_value,
                "per_instance_gap_difference_pp": differences.tolist(),
            }
        )

    adjusted = _holm(raw_p)
    for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
        comparison["wilcoxon_holm_p"] = adjusted_p
        comparison["passes_locked_significance"] = bool(
            comparison["candidate_minus_bopo_mean_gap_pp"] < 0.0
            and comparison["bootstrap_95ci_mean_gap_difference_pp"][1] < 0.0
            and adjusted_p < 0.05
        )

    descriptive = {}
    for label, payload in rows.items():
        descriptive[label] = {}
        for set_name in ("TA", "DMU"):
            values = [
                float(payload[key]["gap_percent"]) for key in keys if key[0] == set_name
            ]
            descriptive[label][set_name] = {
                "n": len(values),
                "mean_gap_percent": float(np.mean(values)),
            }

    result = {
        "protocol": "jssp50x20_locked_official_ta_dmu_inference_v1",
        "test_only_no_selection": True,
        "statistical_unit": "official JSSP50x20 instance",
        "primary_sets": ["TA", "DMU"],
        "n": 20,
        "primary_test": "two-sided paired Wilcoxon signed-rank",
        "holm_family": ["h8_usw_vs_bopo", "h13_asw_vs_bopo"],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "descriptive_by_set": descriptive,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

