from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PILOT_DIR = EXPERIMENTS / "rfps_feature_pilot"
MINIMALITY_DIR = EXPERIMENTS / "rfps_minimality"
INVARIANCE_DIR = EXPERIMENTS / "rfps_invariance_validation"
NONUNIFORM_PATH = EXPERIMENTS / "rfps_nonuniform_q0" / "run_experiment.py"
ROOT = EXPERIMENTS.parents[1]

sys.path[:0] = [
    str(PILOT_DIR),
    str(MINIMALITY_DIR),
    str(INVARIANCE_DIR),
]

import run_pilot as pilot  # noqa: E402
import run_validation as validation  # noqa: E402
from run_two_point_baselines import unit  # noqa: E402


spec = importlib.util.spec_from_file_location("nonuniform_experiment", NONUNIFORM_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {NONUNIFORM_PATH}")
nonuniform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nonuniform)


BANKS = {
    "epoch031": (
        (
            ROOT / "results/rfps_cross_bank_scan/candidate_epoch031_seed1.npz",
            HERE / "data/epoch031_seed1_policy_stats.npz",
        ),
        (
            ROOT / "results/rfps_cross_bank_scan/candidate_epoch031_seed2.npz",
            HERE / "data/epoch031_seed2_policy_stats.npz",
        ),
    ),
    "epoch135": (
        (
            PILOT_DIR / "tsp100_epoch135_rollout_probes.npz",
            HERE / "data/epoch135_seed1_policy_stats.npz",
        ),
        (
            PILOT_DIR / "tsp100_epoch135_rollout_probes_seed2.npz",
            HERE / "data/epoch135_seed2_policy_stats.npz",
        ),
    ),
}

DESCRIPTORS = (
    "raw_coefficient",
    "centered_coefficient",
    "logit_euclidean",
    "policy_fisher",
)


def load_datasets() -> dict[str, dict[str, Any]]:
    matched_candidates, _, matched_targets = nonuniform.load_candidates()
    external_candidates, external_keys = validation.load_candidates(
        validation.DATASETS["external"]["source"]
    )
    external_scores = validation.aligned_scores(
        validation.DATASETS["external"]["targets"]["scratch"],
        external_keys,
    )
    return {
        "matched": {
            "candidates": matched_candidates,
            "targets": matched_targets,
        },
        "external": {
            "candidates": external_candidates,
            "targets": {"scratch": external_scores},
        },
    }


def load_bank(
    paths: tuple[tuple[Path, Path], ...],
) -> tuple[list[pilot.Probe], list[dict[str, np.ndarray]]]:
    probes: list[pilot.Probe] = []
    statistics: list[dict[str, np.ndarray]] = []
    for probe_path, statistic_path in paths:
        seed_probes = pilot.load_probes_npz(probe_path)
        with np.load(statistic_path) as archive:
            step_log_probs = np.asarray(archive["step_log_probs"]).copy()
            step_prob_l2 = np.asarray(archive["step_prob_l2"]).copy()
        if len(seed_probes) != step_log_probs.shape[0]:
            raise RuntimeError(
                f"probe/stat count mismatch for {probe_path}: "
                f"{len(seed_probes)} versus {step_log_probs.shape[0]}"
            )
        for index, probe in enumerate(seed_probes):
            probes.append(probe)
            statistics.append(
                {
                    "step_log_probs": step_log_probs[index],
                    "step_prob_l2": step_prob_l2[index],
                }
            )
    return probes, statistics


def coefficient(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
) -> np.ndarray:
    count = probe.p0.numel()
    q = torch.full((count,), 1.0 / count, dtype=pilot.DTYPE)
    p = probe.p0.detach().clone().requires_grad_(True)
    batch, model_output = pilot.pairwise_inputs_importance(probe, p, q)
    loss = candidate.compiled.loss_fn(
        batch,
        model_output,
        {"alpha": candidate.alpha},
    )
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, dtype=pilot.DTYPE)
    if loss.numel() != 1:
        loss = loss.mean()
    gradient = torch.autograd.grad(loss, p, allow_unused=True)[0]
    value = torch.zeros_like(p) if gradient is None else -gradient
    if not torch.isfinite(value).all():
        raise RuntimeError(f"non-finite coefficient for {candidate.key}")
    return value.reshape(-1).detach().cpu().numpy()


def policy_weights(statistics: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    step_log_probs = np.asarray(statistics["step_log_probs"], dtype=np.float64)
    step_prob_l2 = np.asarray(statistics["step_prob_l2"], dtype=np.float64)
    selected = np.exp(step_log_probs)
    euclidean_terms = 1.0 - 2.0 * selected + step_prob_l2
    fisher_terms = np.exp(-step_log_probs) - 1.0
    if euclidean_terms.min() < -1e-5 or fisher_terms.min() < -1e-5:
        raise RuntimeError("negative policy norm term")
    euclidean = np.maximum(euclidean_terms, 0.0).sum(axis=1)
    fisher = np.maximum(fisher_terms, 0.0).sum(axis=1)
    if not np.isfinite(euclidean).all() or not np.isfinite(fisher).all():
        raise RuntimeError("non-finite policy norm weight")
    return euclidean, fisher


def probe_descriptors(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    statistics: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    c = coefficient(candidate, probe)
    euclidean_weight, fisher_weight = policy_weights(statistics)
    if c.shape != euclidean_weight.shape or c.shape != fisher_weight.shape:
        raise RuntimeError(
            f"coefficient/weight mismatch: {c.shape}, "
            f"{euclidean_weight.shape}, {fisher_weight.shape}"
        )
    centered = c - c.mean()
    descriptors = {
        "raw_coefficient": unit(torch.from_numpy(c)).numpy(),
        "centered_coefficient": unit(torch.from_numpy(centered)).numpy(),
        "logit_euclidean": unit(
            torch.from_numpy(c * np.sqrt(euclidean_weight))
        ).numpy(),
        "policy_fisher": unit(
            torch.from_numpy(c * np.sqrt(fisher_weight))
        ).numpy(),
    }
    denominator = max(float(np.linalg.norm(c)), 1e-15)
    return descriptors, {
        "coefficient_sum_ratio": float(abs(c.sum()) / denominator),
        "euclidean_weight_median": float(np.median(euclidean_weight)),
        "euclidean_weight_p95": float(np.quantile(euclidean_weight, 0.95)),
        "fisher_weight_median": float(np.median(fisher_weight)),
        "fisher_weight_p95": float(np.quantile(fisher_weight, 0.95)),
        "fisher_weight_max": float(fisher_weight.max()),
    }


def bank_matrices(
    candidates: list[pilot.Candidate],
    probes: list[pilot.Probe],
    statistics: list[dict[str, np.ndarray]],
    *,
    dataset_name: str,
    bank_name: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    raw: dict[str, list[np.ndarray]] = {name: [] for name in DESCRIPTORS}
    diagnostics: list[dict[str, float]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_vectors: dict[str, list[np.ndarray]] = {
            name: [] for name in DESCRIPTORS
        }
        for probe, probe_statistics in zip(probes, statistics, strict=True):
            descriptors, values = probe_descriptors(
                candidate, probe, probe_statistics
            )
            diagnostics.append(values)
            for name, vector in descriptors.items():
                candidate_vectors[name].append(vector)
        for name, vectors in candidate_vectors.items():
            raw[name].append(
                np.concatenate(vectors) / math.sqrt(len(vectors))
            )
        print(
            f"[{dataset_name}/{bank_name}] "
            f"{candidate_index:03d}/{len(candidates):03d}",
            flush=True,
        )
    return {name: np.stack(rows) for name, rows in raw.items()}, diagnostics


def target_geometry(
    dataset_name: str,
    targets: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dataset_name == "matched":
        scratch = np.asarray(targets["scratch"])
        warm = np.asarray(targets["warm"])
        joint = np.column_stack(
            [
                (scratch - scratch.mean()) / scratch.std(),
                (warm - warm.mean()) / warm.std(),
            ]
        )
        return pdist(joint), {
            "scratch": pdist(scratch[:, None]),
            "warm": pdist(warm[:, None]),
            "joint": pdist(joint),
        }
    scratch = np.asarray(targets["scratch"])
    return pdist(scratch[:, None]), {"scratch": pdist(scratch[:, None])}


def descriptor_metrics(
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
        f"rho_{name}": float(spearmanr(distances, gaps).statistic)
        for name, gaps in target_gaps.items()
    }
    result.update(
        {
            "nn_primary_median": float(
                np.median(gap_square[np.arange(matrix.shape[0]), nearest])
            ),
            "false20": float(
                np.mean(primary_gaps[close] > np.median(primary_gaps))
            ),
        }
    )
    return result


def neighborhood_agreement(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    first_distances = pdist(first)
    second_distances = pdist(second)
    first_square = squareform(first_distances)
    second_square = squareform(second_distances)
    np.fill_diagonal(first_square, np.inf)
    np.fill_diagonal(second_square, np.inf)
    return {
        "distance_rho": float(
            spearmanr(first_distances, second_distances).statistic
        ),
        "top1_agreement": float(
            np.mean(
                np.argmin(first_square, axis=1)
                == np.argmin(second_square, axis=1)
            )
        ),
    }


def summarize_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
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
    torch.set_default_dtype(pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    datasets = load_datasets()
    banks = {name: load_bank(paths) for name, paths in BANKS.items()}
    results: dict[str, Any] = {"datasets": {}}
    metric_rows: list[dict[str, Any]] = []

    for dataset_name, dataset in datasets.items():
        matrices: dict[str, dict[str, np.ndarray]] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for bank_name, (probes, statistics) in banks.items():
            matrices[bank_name], values = bank_matrices(
                dataset["candidates"],
                probes,
                statistics,
                dataset_name=dataset_name,
                bank_name=bank_name,
            )
            diagnostics[bank_name] = summarize_diagnostics(values)

        primary_gaps, target_gaps = target_geometry(
            dataset_name, dataset["targets"]
        )
        metrics: dict[str, dict[str, float]] = {}
        products: dict[str, np.ndarray] = {}
        for descriptor in DESCRIPTORS:
            for bank_name in ("epoch031", "epoch135"):
                value = descriptor_metrics(
                    matrices[bank_name][descriptor],
                    primary_gaps,
                    target_gaps,
                )
                metrics[f"{descriptor}:{bank_name}"] = value
                metric_rows.append(
                    {
                        "dataset": dataset_name,
                        "descriptor": descriptor,
                        "bank": bank_name,
                        **value,
                    }
                )
            product = np.concatenate(
                [
                    matrices["epoch031"][descriptor] / math.sqrt(2.0),
                    matrices["epoch135"][descriptor] / math.sqrt(2.0),
                ],
                axis=1,
            )
            products[descriptor] = product
            value = descriptor_metrics(
                product, primary_gaps, target_gaps
            )
            metrics[f"{descriptor}:product"] = value
            metric_rows.append(
                {
                    "dataset": dataset_name,
                    "descriptor": descriptor,
                    "bank": "product",
                    **value,
                }
            )

        agreement = {
            bank_name: neighborhood_agreement(
                matrices[bank_name]["logit_euclidean"],
                matrices[bank_name]["policy_fisher"],
            )
            for bank_name in ("epoch031", "epoch135")
        }
        agreement["product"] = neighborhood_agreement(
            products["logit_euclidean"],
            products["policy_fisher"],
        )
        results["datasets"][dataset_name] = {
            "n_candidates": len(dataset["candidates"]),
            "metrics": metrics,
            "agreement": agreement,
            "diagnostics": diagnostics,
        }

    results_path = HERE / "results/results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in metric_rows for key in row})
    with (HERE / "results/metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
