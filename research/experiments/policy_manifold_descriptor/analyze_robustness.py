from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import run_experiment as experiment


DESCRIPTORS = (
    "raw_coefficient",
    "logit_euclidean",
    "policy_fisher",
)
N_SUBSETS = 200
SEED = 20260728


def descriptor_blocks(
    candidates: list[experiment.pilot.Candidate],
    probes: list[experiment.pilot.Probe],
    statistics: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    output = {name: [] for name in DESCRIPTORS}
    for candidate in candidates:
        rows = {name: [] for name in DESCRIPTORS}
        for probe, values in zip(probes, statistics, strict=True):
            descriptors, _ = experiment.probe_descriptors(
                candidate, probe, values
            )
            for name in DESCRIPTORS:
                rows[name].append(descriptors[name])
        for name in DESCRIPTORS:
            output[name].append(np.stack(rows[name]))
    return {name: np.stack(rows) for name, rows in output.items()}


def flatten(blocks: np.ndarray, subset: np.ndarray) -> np.ndarray:
    selected = blocks[:, subset, :]
    return selected.reshape(selected.shape[0], -1) / math.sqrt(len(subset))


def primary_target(
    dataset_name: str,
    targets: dict[str, np.ndarray],
) -> np.ndarray:
    if dataset_name == "matched":
        scratch = np.asarray(targets["scratch"])
        warm = np.asarray(targets["warm"])
        return np.column_stack(
            [
                (scratch - scratch.mean()) / scratch.std(),
                (warm - warm.mean()) / warm.std(),
            ]
        )
    return np.asarray(targets["scratch"])[:, None]


def metrics(matrix: np.ndarray, target: np.ndarray) -> dict[str, float]:
    distances = pdist(matrix)
    gaps = pdist(target)
    close = distances <= np.quantile(distances, 0.20)
    return {
        "rho": float(spearmanr(distances, gaps).statistic),
        "false20": float(np.mean(gaps[close] > np.median(gaps))),
    }


def evaluate_subset(
    blocks: dict[str, dict[str, np.ndarray]],
    subset: np.ndarray,
    target: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for descriptor in DESCRIPTORS:
        matrices = {
            bank: flatten(values[descriptor], subset)
            for bank, values in blocks.items()
        }
        for bank, matrix in matrices.items():
            result[f"{descriptor}:{bank}"] = metrics(matrix, target)
        product = np.concatenate(
            [
                matrices["epoch031"] / math.sqrt(2.0),
                matrices["epoch135"] / math.sqrt(2.0),
            ],
            axis=1,
        )
        result[f"{descriptor}:product"] = metrics(product, target)
    return result


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
    }


def main() -> int:
    datasets = experiment.load_datasets()
    banks = {
        name: experiment.load_bank(paths)
        for name, paths in experiment.BANKS.items()
    }
    rng = np.random.default_rng(SEED)
    output: dict[str, Any] = {"datasets": {}}

    for dataset_name, dataset in datasets.items():
        blocks = {
            bank_name: descriptor_blocks(
                dataset["candidates"], probes, statistics
            )
            for bank_name, (probes, statistics) in banks.items()
        }
        target = primary_target(dataset_name, dataset["targets"])
        fixed = {
            "all16": np.arange(16),
            "seed1": np.arange(8),
            "seed2": np.arange(8, 16),
        }
        fixed_results = {
            name: evaluate_subset(blocks, subset, target)
            for name, subset in fixed.items()
        }

        deltas: dict[str, list[float]] = {
            "fisher_minus_logit_rho": [],
            "fisher_minus_logit_false20": [],
            "fisher_minus_raw_rho": [],
            "fisher_minus_raw_false20": [],
        }
        for _ in range(N_SUBSETS):
            subset = np.sort(rng.choice(16, size=8, replace=False))
            values = evaluate_subset(blocks, subset, target)
            fisher = values["policy_fisher:product"]
            logit = values["logit_euclidean:product"]
            raw = values["raw_coefficient:product"]
            deltas["fisher_minus_logit_rho"].append(
                fisher["rho"] - logit["rho"]
            )
            deltas["fisher_minus_logit_false20"].append(
                fisher["false20"] - logit["false20"]
            )
            deltas["fisher_minus_raw_rho"].append(
                fisher["rho"] - raw["rho"]
            )
            deltas["fisher_minus_raw_false20"].append(
                fisher["false20"] - raw["false20"]
            )

        trimmed: dict[str, Any] | None = None
        if dataset_name == "external":
            scores = target[:, 0]
            central_indices = np.argsort(scores)[:-3]
            central_target = target[central_indices]
            central_blocks = {
                bank: {
                    descriptor: values[descriptor][central_indices]
                    for descriptor in DESCRIPTORS
                }
                for bank, values in blocks.items()
            }
            trimmed = {
                "removed_indices": np.argsort(scores)[-3:].tolist(),
                "removed_scores": scores[np.argsort(scores)[-3:]].tolist(),
                "metrics": evaluate_subset(
                    central_blocks, np.arange(16), central_target
                ),
            }

        output["datasets"][dataset_name] = {
            "fixed_subsets": fixed_results,
            "random_r8_deltas": {
                name: summarize(values) for name, values in deltas.items()
            },
            "exploratory_remove_three_largest_scores": trimmed,
        }

    path = experiment.HERE / "results/robustness.json"
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
