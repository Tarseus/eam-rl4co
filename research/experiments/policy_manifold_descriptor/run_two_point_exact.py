from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

import run_two_point as shared


first_order = shared.first_order
LENGTH = shared.LENGTH
DESCRIPTORS = shared.DESCRIPTORS


def probe_descriptors(
    candidate: first_order.pilot.Candidate,
    probe: first_order.pilot.Probe,
    statistics: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    step_log_probs = np.asarray(
        statistics["step_log_probs"], dtype=np.float64
    )
    probability0 = np.exp(step_log_probs)
    trajectory0 = step_log_probs.sum(axis=1)
    coefficient0 = shared.coefficient_at(candidate, probe, trajectory0)
    fisher_terms0 = np.maximum(1.0 / probability0 - 1.0, 0.0)
    direction0 = first_order.unit(
        torch.from_numpy(
            coefficient0[:, None] * np.sqrt(fisher_terms0)
        ).reshape(-1)
    ).numpy()

    probability1, diagnostics = shared.policy_step(
        probability0, coefficient0
    )
    trajectory1 = np.log(probability1).sum(axis=1)
    coefficient1 = shared.coefficient_at(candidate, probe, trajectory1)
    fisher_terms1 = np.maximum(1.0 / probability1 - 1.0, 0.0)
    transported_direction1 = first_order.unit(
        torch.from_numpy(
            coefficient1[:, None] * np.sqrt(fisher_terms1)
        ).reshape(-1)
    ).numpy()
    two_point = np.concatenate(
        [direction0, transported_direction1]
    ) / np.sqrt(2.0)

    diagnostics.update(
        {
            "direction_cosine": float(
                np.dot(direction0, transported_direction1)
            ),
            "trajectory_log_likelihood_change_median": float(
                np.median(np.abs(trajectory1 - trajectory0))
            ),
            "coefficient_relative_change": float(
                np.linalg.norm(coefficient1 - coefficient0)
                / max(np.linalg.norm(coefficient0), 1e-15)
            ),
        }
    )
    return {
        "raw_coefficient": first_order.unit(
            torch.from_numpy(coefficient0)
        ).numpy(),
        "one_point_policy_fisher": direction0,
        "two_point_policy_fisher": two_point,
    }, diagnostics


def bank_grams(
    candidates: list[first_order.pilot.Candidate],
    probes: list[first_order.pilot.Probe],
    statistics: list[dict[str, np.ndarray]],
    *,
    dataset_name: str,
    bank_name: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    count = len(candidates)
    grams = {
        name: np.zeros((count, count), dtype=np.float64)
        for name in DESCRIPTORS
    }
    diagnostic_rows: list[dict[str, float]] = []
    for probe_index, (probe, values) in enumerate(
        zip(probes, statistics, strict=True),
        start=1,
    ):
        vectors = {name: [] for name in DESCRIPTORS}
        for candidate in candidates:
            descriptors, diagnostics = probe_descriptors(
                candidate, probe, values
            )
            diagnostic_rows.append(diagnostics)
            for name, vector in descriptors.items():
                vectors[name].append(vector)
        for name, rows in vectors.items():
            matrix = np.stack(rows)
            grams[name] += matrix @ matrix.T / len(probes)
        print(
            f"[{dataset_name}/{bank_name}] "
            f"probe {probe_index:02d}/{len(probes):02d}",
            flush=True,
        )
    return grams, diagnostic_rows


def gram_distances(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.diag(gram)
    squared = np.maximum(
        diagonal[:, None] + diagonal[None, :] - 2.0 * gram,
        0.0,
    )
    square = np.sqrt(squared)
    upper = np.triu_indices(gram.shape[0], 1)
    return square[upper], square


def metrics(
    gram: np.ndarray,
    primary_gaps: np.ndarray,
    target_gaps: dict[str, np.ndarray],
) -> dict[str, float]:
    distances, square = gram_distances(gram)
    np.fill_diagonal(square, np.inf)
    nearest = np.argmin(square, axis=1)
    gap_square = squareform(primary_gaps)
    close = distances <= np.quantile(distances, 0.20)
    result = {
        f"rho_{name}": float(
            spearmanr(distances, gaps).statistic
        )
        for name, gaps in target_gaps.items()
    }
    result.update(
        {
            "nn_primary_median": float(
                np.median(
                    gap_square[np.arange(gram.shape[0]), nearest]
                )
            ),
            "false20": float(
                np.mean(primary_gaps[close] > np.median(primary_gaps))
            ),
        }
    )
    return result


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        f"{name}_{statistic}": float(function([row[name] for row in rows]))
        for name in rows[0]
        for statistic, function in (
            ("median", np.median),
            ("p05", lambda values: np.quantile(values, 0.05)),
            ("p95", lambda values: np.quantile(values, 0.95)),
        )
    }


def main() -> int:
    torch.set_default_dtype(first_order.pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    datasets = first_order.load_datasets()
    banks = {
        name: first_order.load_bank(paths)
        for name, paths in first_order.BANKS.items()
    }
    output: dict[str, Any] = {"length": LENGTH, "datasets": {}}
    csv_rows: list[dict[str, Any]] = []

    for dataset_name, dataset in datasets.items():
        bank_data: dict[str, dict[str, np.ndarray]] = {}
        diagnostic_data: dict[str, dict[str, float]] = {}
        for bank_name, (probes, statistics) in banks.items():
            bank_data[bank_name], rows = bank_grams(
                dataset["candidates"],
                probes,
                statistics,
                dataset_name=dataset_name,
                bank_name=bank_name,
            )
            diagnostic_data[bank_name] = summarize(rows)

        primary_gaps, target_gaps = first_order.target_geometry(
            dataset_name, dataset["targets"]
        )
        descriptor_metrics: dict[str, dict[str, float]] = {}
        for descriptor in DESCRIPTORS:
            for bank_name in ("epoch031", "epoch135"):
                value = metrics(
                    bank_data[bank_name][descriptor],
                    primary_gaps,
                    target_gaps,
                )
                descriptor_metrics[f"{descriptor}:{bank_name}"] = value
                csv_rows.append(
                    {
                        "dataset": dataset_name,
                        "descriptor": descriptor,
                        "bank": bank_name,
                        **value,
                    }
                )
            product = (
                bank_data["epoch031"][descriptor]
                + bank_data["epoch135"][descriptor]
            ) / 2.0
            value = metrics(product, primary_gaps, target_gaps)
            descriptor_metrics[f"{descriptor}:product"] = value
            csv_rows.append(
                {
                    "dataset": dataset_name,
                    "descriptor": descriptor,
                    "bank": "product",
                    **value,
                }
            )
        output["datasets"][dataset_name] = {
            "n_candidates": len(dataset["candidates"]),
            "metrics": descriptor_metrics,
            "diagnostics": diagnostic_data,
        }

    results_dir = first_order.HERE / "results"
    (results_dir / "two_point.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in csv_rows for key in row})
    with (results_dir / "two_point.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
