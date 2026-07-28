from __future__ import annotations

import csv
import json
import math
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import run_experiment as first_order


LENGTH = 0.03
DESCRIPTORS = (
    "raw_coefficient",
    "one_point_policy_fisher",
    "two_point_policy_fisher",
)


def coefficient_at(
    candidate: first_order.pilot.Candidate,
    probe: first_order.pilot.Probe,
    trajectory_log_likelihood: np.ndarray,
) -> np.ndarray:
    values = torch.as_tensor(
        trajectory_log_likelihood,
        dtype=first_order.pilot.DTYPE,
    ).reshape_as(probe.p0)
    p = values.detach().clone().requires_grad_(True)
    count = p.numel()
    q = torch.full(
        (count,),
        1.0 / count,
        dtype=first_order.pilot.DTYPE,
    )
    batch, model_output = first_order.pilot.pairwise_inputs_importance(
        probe, p, q
    )
    loss = candidate.compiled.loss_fn(
        batch,
        model_output,
        {"alpha": candidate.alpha},
    )
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, dtype=first_order.pilot.DTYPE)
    if loss.numel() != 1:
        loss = loss.mean()
    gradient = torch.autograd.grad(loss, p, allow_unused=True)[0]
    coefficient = torch.zeros_like(p) if gradient is None else -gradient
    if not torch.isfinite(coefficient).all():
        raise RuntimeError(f"non-finite coefficient for {candidate.key}")
    return coefficient.reshape(-1).detach().cpu().numpy()


def policy_step(
    selected_probability: np.ndarray,
    coefficient: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    probability = np.asarray(selected_probability, dtype=np.float64)
    c = np.asarray(coefficient, dtype=np.float64)
    if probability.shape[0] != c.shape[0]:
        raise RuntimeError(
            f"probability/coefficient mismatch: {probability.shape}, {c.shape}"
        )
    fisher_terms = np.maximum(1.0 / probability - 1.0, 0.0)
    speed_squared = np.sum(c[:, None] ** 2 * fisher_terms)
    speed = math.sqrt(max(float(speed_squared), 0.0))
    if speed <= 1e-14:
        return probability.copy(), {
            "step_length": 0.0,
            "speed": speed,
            "max_log_probability_change": 0.0,
        }

    local_speed = np.abs(c[:, None]) * np.sqrt(fisher_terms)
    angle = LENGTH * local_speed / (2.0 * speed)
    square_root = np.sqrt(probability)
    orthogonal_selected = np.sqrt(np.maximum(1.0 - probability, 0.0))
    signed_direction = np.sign(c)[:, None] * orthogonal_selected
    next_square_root = (
        np.cos(angle) * square_root
        + np.sin(angle) * signed_direction
    )
    next_probability = np.square(next_square_root)
    next_probability = np.clip(next_probability, 1e-300, 1.0)
    step_length = math.sqrt(float(np.sum(np.square(2.0 * angle))))
    return next_probability, {
        "step_length": step_length,
        "speed": speed,
        "max_log_probability_change": float(
            np.max(np.abs(np.log(next_probability) - np.log(probability)))
        ),
    }


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
    coefficient0 = coefficient_at(candidate, probe, trajectory0)
    fisher_weight0 = np.maximum(1.0 / probability0 - 1.0, 0.0).sum(axis=1)
    direction0 = first_order.unit(
        torch.from_numpy(coefficient0 * np.sqrt(fisher_weight0))
    ).numpy()

    probability1, diagnostics = policy_step(probability0, coefficient0)
    trajectory1 = np.log(probability1).sum(axis=1)
    coefficient1 = coefficient_at(candidate, probe, trajectory1)
    fisher_weight1 = np.maximum(1.0 / probability1 - 1.0, 0.0).sum(axis=1)
    transported_direction1 = first_order.unit(
        torch.from_numpy(coefficient1 * np.sqrt(fisher_weight1))
    ).numpy()

    two_point = np.concatenate(
        [direction0, transported_direction1]
    ) / math.sqrt(2.0)
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


def bank_matrices(
    candidates: list[first_order.pilot.Candidate],
    probes: list[first_order.pilot.Probe],
    statistics: list[dict[str, np.ndarray]],
    *,
    dataset_name: str,
    bank_name: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    output: dict[str, list[np.ndarray]] = {
        name: [] for name in DESCRIPTORS
    }
    diagnostic_rows: list[dict[str, float]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_vectors = {name: [] for name in DESCRIPTORS}
        for probe, values in zip(probes, statistics, strict=True):
            descriptors, diagnostics = probe_descriptors(
                candidate, probe, values
            )
            diagnostic_rows.append(diagnostics)
            for name, vector in descriptors.items():
                candidate_vectors[name].append(vector)
        for name, vectors in candidate_vectors.items():
            output[name].append(
                np.concatenate(vectors) / math.sqrt(len(vectors))
            )
        print(
            f"[{dataset_name}/{bank_name}] "
            f"{candidate_index:03d}/{len(candidates):03d}",
            flush=True,
        )
    return {
        name: np.stack(vectors) for name, vectors in output.items()
    }, diagnostic_rows


def metrics(
    matrix: np.ndarray,
    primary_gaps: np.ndarray,
    target_gaps: dict[str, np.ndarray],
) -> dict[str, float]:
    distances = pdist(matrix)
    square = squareform(distances)
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
                    gap_square[np.arange(matrix.shape[0]), nearest]
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
            bank_data[bank_name], rows = bank_matrices(
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
            product = np.concatenate(
                [
                    bank_data["epoch031"][descriptor] / math.sqrt(2.0),
                    bank_data["epoch135"][descriptor] / math.sqrt(2.0),
                ],
                axis=1,
            )
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
