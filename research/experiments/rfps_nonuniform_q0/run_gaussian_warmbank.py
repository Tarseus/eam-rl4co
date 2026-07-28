from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_experiment as experiment


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "rfps_feature_pilot"
OUTPUT = HERE / "results_warmbank_gaussian"
PROBE_BANKS = (
    PILOT / "tsp100_epoch135_rollout_probes.npz",
    PILOT / "tsp100_epoch135_rollout_probes_seed2.npz",
)
CONDITIONS = {
    "uniform": ("uniform", 0.0, 0.0),
    "mid_s010": ("single", 0.50, 0.10),
    "mid_s020": ("single", 0.50, 0.20),
    "mid_s035": ("single", 0.50, 0.35),
    "best_s015": ("single", 0.00, 0.15),
    "best_s025": ("single", 0.00, 0.25),
    "best_s040": ("single", 0.00, 0.40),
    "tails_s010": ("tails", 0.00, 0.10),
    "tails_s020": ("tails", 0.00, 0.20),
    "tails_s035": ("tails", 0.00, 0.35),
}


def as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rank_coordinate(objective: np.ndarray | torch.Tensor) -> np.ndarray:
    values = as_numpy(objective).reshape(-1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(1, values.size - 1)


def gaussian_measure(
    coordinate: np.ndarray,
    specification: tuple[str, float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    shape, center, sigma = specification
    if shape == "uniform":
        raw = np.ones_like(coordinate)
    elif shape == "single":
        raw = np.exp(-0.5 * np.square((coordinate - center) / sigma))
    elif shape == "tails":
        raw = 0.5 * np.exp(-0.5 * np.square(coordinate / sigma))
        raw += 0.5 * np.exp(-0.5 * np.square((coordinate - 1.0) / sigma))
    else:
        raise ValueError(shape)
    raw /= raw.sum()
    q, mixture = experiment.ensure_ess(raw)
    return q, {
        "raw_ess": experiment.normalized_ess(raw),
        "ess": experiment.normalized_ess(q),
        "uniform_mixture": mixture,
        "max_over_uniform": float(q.max() * q.size),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    torch.set_default_dtype(experiment.pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    candidates, keys, targets = experiment.load_candidates()

    probes: list[tuple[int, experiment.pilot.Probe]] = []
    for seed_index, path in enumerate(PROBE_BANKS, start=1):
        probes.extend(
            (seed_index, probe) for probe in experiment.pilot.load_probes_npz(path)
        )

    q_cache: dict[tuple[int, str], torch.Tensor] = {}
    q_rows: list[dict[str, Any]] = []
    for probe_index, (seed_index, probe) in enumerate(probes):
        coordinate = rank_coordinate(probe.objective)
        for condition, specification in CONDITIONS.items():
            q, stats = gaussian_measure(coordinate, specification)
            q_cache[(probe_index, condition)] = torch.as_tensor(
                q, dtype=experiment.pilot.DTYPE
            )
            q_rows.append(
                {
                    "probe": probe_index,
                    "seed_index": seed_index,
                    "condition": condition,
                    **stats,
                }
            )

    raw: dict[tuple[str, str], list[list[np.ndarray]]] = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        candidate_store: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for probe_index, (_, probe) in enumerate(probes):
            for condition in CONDITIONS:
                blocks = experiment.descriptor_blocks(
                    candidate, probe, q_cache[(probe_index, condition)]
                )
                for method, vector in blocks.items():
                    candidate_store[(condition, method)].append(vector)
        for key, vectors in candidate_store.items():
            raw[key].append(vectors)
        print(
            f"[gaussian-q0] {candidate_index + 1:03d}/{len(candidates):03d}",
            flush=True,
        )

    matrices = {key: np.asarray(value) for key, value in raw.items()}
    subsets = {
        "all": np.arange(len(probes)),
        "seed1": np.asarray(
            [index for index, item in enumerate(probes) if item[0] == 1]
        ),
        "seed2": np.asarray(
            [index for index, item in enumerate(probes) if item[0] == 2]
        ),
    }
    rows: list[dict[str, Any]] = []
    for subset_name, subset in subsets.items():
        for condition in CONDITIONS:
            euclidean = experiment.flatten_blocks(
                matrices[(condition, "euclidean_two_point")], subset
            )
            fisher = experiment.flatten_blocks(
                matrices[(condition, "fisher_two_point")], subset
            )
            ef = experiment.agreement(euclidean, fisher)
            for target, scores in targets.items():
                for method in experiment.METHODS:
                    matrix = experiment.flatten_blocks(
                        matrices[(condition, method)], subset
                    )
                    row: dict[str, Any] = {
                        "subset": subset_name,
                        "target": target,
                        "condition": condition,
                        "method": method,
                    }
                    row.update(experiment.descriptor_metrics(matrix, scores))
                    row.update(
                        experiment.order_risk(
                            matrix,
                            scores,
                            seed=experiment.SEED
                            + (0 if subset_name == "all" else int(subset[-1])),
                        )
                    )
                    row.update(ef)
                    rows.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "metrics.csv", rows)
    write_csv(OUTPUT / "q_stats.csv", q_rows)
    payload = {
        "ell": experiment.ELL,
        "n_candidates": len(candidates),
        "n_probes": len(probes),
        "coordinate": "normalized within-probe objective rank; lower is better",
        "conditions": CONDITIONS,
        "metrics": rows,
        "q_stats": q_rows,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary = [
        row
        for row in rows
        if row["subset"] == "all"
        and row["method"] in ("euclidean_two_point", "fisher_two_point")
    ]
    lines = [
        "# Gaussian q0 on the checkpoint-135 probe bank",
        "",
        "The Gaussian coordinate is normalized within-probe tour-cost rank;",
        "rank 0 is the best tour and rank 1 is the worst.",
        "",
        "| Target | Condition | Method | rho | false skip | NN median | E/F top-1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {target} | {condition} | {method} | {rho:.4f} | "
            "{false_skip_mean:.4f} | {nn_median:.5f} | {ef_top1:.3f} |".format(
                **row
            )
        )
    (OUTPUT / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    for target in ("scratch", "warm"):
        target_rows = [
            row
            for row in primary
            if row["target"] == target and row["method"] == "fisher_two_point"
        ]
        target_rows.sort(key=lambda row: row["rho"], reverse=True)
        best = target_rows[0]
        uniform = next(row for row in target_rows if row["condition"] == "uniform")
        print(
            f"{target}: best={best['condition']} rho={best['rho']:.4f} "
            f"false_skip={best['false_skip_mean']:.4f}; "
            f"uniform rho={uniform['rho']:.4f} "
            f"false_skip={uniform['false_skip_mean']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
