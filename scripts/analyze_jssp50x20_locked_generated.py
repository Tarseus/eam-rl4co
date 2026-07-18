from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def _read(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 256:
        raise ValueError(f"Expected 256 rows in {path}, got {len(rows)}")
    keyed = {int(row["instance_index"]): row for row in rows}
    if sorted(keyed) != list(range(256)):
        raise ValueError(f"Invalid generated indices in {path}")
    return keyed


def _bootstrap(values: np.ndarray, *, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [1.25, 98.75])]


def _holm(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=raw.__getitem__)
    adjusted = [math.nan] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bopo", type=Path, required=True)
    parser.add_argument("--h8-usw", type=Path, required=True)
    parser.add_argument("--h13-asw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    args = parser.parse_args()
    if args.bootstrap_samples != 20000 or args.bootstrap_seed != 20260718:
        raise ValueError("Locked analysis parameter mismatch")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite locked analysis: {args.output}")
    rows = {
        "bopo": _read(args.bopo.resolve()),
        "h8_usw": _read(args.h8_usw.resolve()),
        "h13_asw": _read(args.h13_asw.resolve()),
    }
    for index in range(256):
        seeds = {payload[index]["sampling_seed"] for payload in rows.values()}
        hashes = {payload[index]["instance_sha256"] for payload in rows.values()}
        if len(seeds) != 1 or len(hashes) != 1:
            raise ValueError(f"Pairing mismatch at generated index {index}")

    comparisons = []
    raw_p = []
    baseline = np.asarray(
        [float(rows["bopo"][index]["cost"]) for index in range(256)],
        dtype=np.float64,
    )
    for comparison_index, label in enumerate(("h8_usw", "h13_asw")):
        candidate = np.asarray(
            [float(rows[label][index]["cost"]) for index in range(256)],
            dtype=np.float64,
        )
        differences = candidate - baseline
        if not np.isfinite(differences).all():
            raise FloatingPointError(f"Non-finite differences for {label}")
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
                "n": 256,
                "candidate_mean_cost": float(candidate.mean()),
                "baseline_mean_cost": float(baseline.mean()),
                "candidate_minus_bopo_mean_cost": float(differences.mean()),
                "bootstrap_97p5ci_mean_difference": _bootstrap(
                    differences,
                    samples=20000,
                    seed=20260718 + comparison_index,
                ),
                "wins": int((differences < 0).sum()),
                "ties": int((differences == 0).sum()),
                "losses": int((differences > 0).sum()),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_raw_p": p_value,
            }
        )
    adjusted = _holm(raw_p)
    for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
        comparison["wilcoxon_holm_p"] = adjusted_p
        comparison["passes_locked_replication"] = bool(
            comparison["candidate_minus_bopo_mean_cost"] < 0.0
            and comparison["bootstrap_97p5ci_mean_difference"][1] < 0.0
            and adjusted_p < 0.025
        )
    result = {
        "protocol": "jssp50x20_locked_generated256_inference_v1",
        "test_only_no_selection": True,
        "official_h14_replaced": False,
        "statistical_unit": "independently generated JSSP50x20 instance",
        "n": 256,
        "primary_test": "two-sided paired Wilcoxon signed-rank",
        "holm_family": ["h8_usw_vs_bopo", "h13_asw_vs_bopo"],
        "alpha": 0.025,
        "bootstrap_interval": "97.5% percentile",
        "bootstrap_samples": 20000,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

