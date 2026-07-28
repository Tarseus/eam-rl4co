from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from analyze_existing import (
    INITIAL_ONLY_RUNS,
    RUNS,
    descriptor_metrics,
    flatten_probe_blocks,
    load_run,
)


HERE = Path(__file__).resolve().parent
TWO_POINT = HERE / 'results_two_point'


def initial_blocks(data: dict[str, np.ndarray]) -> np.ndarray:
    n_candidates = len(data['scores'])
    n_probes = data['initial_field'].shape[1] // 102
    block_size = data['initial_field'].shape[1] // n_probes
    return data['initial_field'].reshape(n_candidates, n_probes, block_size)[:, :, :-2]


def normalize_blocks(blocks: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(blocks, axis=2, keepdims=True)
    return blocks / np.maximum(norm, 1e-12)


def gram_descriptor(blocks: np.ndarray) -> np.ndarray:
    normalized = normalize_blocks(blocks)
    gram = np.einsum('cpi,cqi->cpq', normalized, normalized)
    upper = np.triu_indices(blocks.shape[1])
    return gram[:, upper[0], upper[1]]


def probe_objectives(run_path: Path) -> np.ndarray:
    metadata = json.loads((run_path / 'results.json').read_text(encoding='utf-8'))
    probe_path = Path(metadata['probe_source'])
    with np.load(probe_path) as archive:
        return archive['objective'].copy()


def rank_aligned(blocks: np.ndarray, objective: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(blocks)
    for probe_index in range(blocks.shape[1]):
        order = np.argsort(objective[probe_index])
        aligned[:, probe_index, :] = blocks[:, probe_index, order]
    return aligned


def sorted_response(blocks: np.ndarray) -> np.ndarray:
    return np.sort(blocks, axis=2)


def permuted_gram_metrics(
    blocks: np.ndarray,
    scores: np.ndarray,
    *,
    n_trials: int = 100,
) -> dict[str, float]:
    rng = np.random.default_rng(20260728)
    rho = np.empty(n_trials)
    nn = np.empty(n_trials)
    for trial in range(n_trials):
        permuted = np.empty_like(blocks)
        for probe_index in range(blocks.shape[1]):
            order = rng.permutation(blocks.shape[2])
            permuted[:, probe_index, :] = blocks[:, probe_index, order]
        metrics = descriptor_metrics(gram_descriptor(permuted), scores)
        rho[trial] = metrics['rho']
        nn[trial] = metrics['nn_median']
    return {
        'rho_mean': float(np.mean(rho)),
        'rho_q05': float(np.quantile(rho, 0.05)),
        'rho_median': float(np.median(rho)),
        'rho_q95': float(np.quantile(rho, 0.95)),
        'nn_mean': float(np.mean(nn)),
        'nn_q95': float(np.quantile(nn, 0.95)),
    }


def order_resampled_risk(
    matrix: np.ndarray,
    scores: np.ndarray,
    *,
    n_trials: int = 1000,
    skip_quantile: float = 0.20,
    tolerance: float = 0.01,
) -> dict[str, float]:
    distance = squareform(pdist(matrix))
    rng = np.random.default_rng(20260729)
    false_rates = np.empty(n_trials)
    skipped_mae = np.empty(n_trials)
    for trial in range(n_trials):
        order = rng.permutation(len(scores))
        nearest_distance: list[float] = []
        nearest_gap: list[float] = []
        for position in range(1, len(order)):
            candidate = int(order[position])
            previous = order[:position]
            local = int(np.argmin(distance[candidate, previous]))
            neighbor = int(previous[local])
            nearest_distance.append(float(distance[candidate, neighbor]))
            nearest_gap.append(float(abs(scores[candidate] - scores[neighbor])))
        nearest_distance_array = np.asarray(nearest_distance)
        nearest_gap_array = np.asarray(nearest_gap)
        threshold = float(np.quantile(nearest_distance_array, skip_quantile))
        selected = nearest_distance_array <= threshold
        false_rates[trial] = float(np.mean(nearest_gap_array[selected] > tolerance))
        skipped_mae[trial] = float(np.mean(nearest_gap_array[selected]))
    return {
        'false_skip_mean': float(np.mean(false_rates)),
        'false_skip_median': float(np.median(false_rates)),
        'false_skip_q90': float(np.quantile(false_rates, 0.90)),
        'skipped_mae_mean': float(np.mean(skipped_mae)),
        'skipped_mae_q90': float(np.quantile(skipped_mae, 0.90)),
    }


def candidate_bootstrap_delta(
    preferred: np.ndarray,
    reference: np.ndarray,
    scores: np.ndarray,
    *,
    n_trials: int = 2000,
) -> dict[str, float]:
    rng = np.random.default_rng(20260730)
    deltas = np.empty(n_trials)
    n = len(scores)
    for trial in range(n_trials):
        sample = rng.integers(0, n, n)
        preferred_distance = pdist(preferred[sample])
        reference_distance = pdist(reference[sample])
        gap = pdist(scores[sample, None])
        preferred_rho = float(spearmanr(preferred_distance, gap).statistic)
        reference_rho = float(spearmanr(reference_distance, gap).statistic)
        deltas[trial] = preferred_rho - reference_rho
    return {
        'delta_mean': float(np.nanmean(deltas)),
        'delta_q025': float(np.nanquantile(deltas, 0.025)),
        'delta_median': float(np.nanmedian(deltas)),
        'delta_q975': float(np.nanquantile(deltas, 0.975)),
    }


def load_two_point(run_name: str, descriptor_name: str, target_keys: np.ndarray) -> np.ndarray:
    with np.load(TWO_POINT / run_name / 'descriptors.npz') as archive:
        keys = archive['keys']
        matrix = archive[descriptor_name]
    index = {str(key): idx for idx, key in enumerate(keys)}
    return np.stack([matrix[index[str(key)]] for key in target_keys])


def analyze_run(
    run_name: str,
    run_path: Path,
    *,
    include_dynamic: bool,
) -> dict[str, Any]:
    data = load_run(run_path)
    blocks = initial_blocks(data)
    normalized = normalize_blocks(blocks)
    objective = probe_objectives(run_path)
    descriptors = {
        'initial_unit_per_probe': flatten_probe_blocks(normalized),
        'initial_gram': gram_descriptor(blocks),
        'rank_aligned_gram': gram_descriptor(rank_aligned(blocks, objective)),
        'sorted_response_gram': gram_descriptor(sorted_response(blocks)),
    }
    if 'flow_curvature' in data:
        descriptors['reference_kappa_full'] = data['flow_curvature']
    if include_dynamic:
        fisher_turn = load_two_point(
            run_name,
            'fisher_direction_delta_pt',
            data['keys'],
        )
        descriptors['fisher_direction_delta_pt'] = fisher_turn
        second_direction = descriptors['initial_unit_per_probe'] + fisher_turn
        descriptors['two_point_direction_pair'] = np.concatenate(
            [descriptors['initial_unit_per_probe'], second_direction],
            axis=1,
        ) / math.sqrt(2.0)
        descriptors['two_point_direction_jet'] = np.concatenate(
            [descriptors['initial_unit_per_probe'], fisher_turn],
            axis=1,
        ) / math.sqrt(2.0)
        euclidean_turn = load_two_point(
            run_name,
            'euclidean_direction_delta',
            data['keys'],
        )
        descriptors['euclidean_direction_delta'] = euclidean_turn
        euclidean_second_direction = descriptors['initial_unit_per_probe'] + euclidean_turn
        descriptors['euclidean_two_point_direction_pair'] = np.concatenate(
            [descriptors['initial_unit_per_probe'], euclidean_second_direction],
            axis=1,
        ) / math.sqrt(2.0)

    metrics = {
        name: descriptor_metrics(matrix, data['scores'])
        for name, matrix in descriptors.items()
    }
    risk = {
        name: order_resampled_risk(matrix, data['scores'])
        for name, matrix in descriptors.items()
    }
    distance_correlations: dict[str, float] = {}
    unit_distance = pdist(descriptors['initial_unit_per_probe'])
    for name, matrix in descriptors.items():
        distance_correlations[f'unit_vs_{name}'] = float(
            spearmanr(unit_distance, pdist(matrix)).statistic
        )

    bootstrap: dict[str, float] | None = None
    if 'reference_kappa_full' in descriptors:
        bootstrap = candidate_bootstrap_delta(
            descriptors['initial_unit_per_probe'],
            descriptors['reference_kappa_full'],
            data['scores'],
        )
    return {
        'metrics': metrics,
        'order_resampled_risk': risk,
        'distance_correlations': distance_correlations,
        'unit_minus_curvature_bootstrap': bootstrap,
        'permuted_gram': permuted_gram_metrics(blocks, data['scores']),
    }


def main() -> int:
    results: dict[str, Any] = {}
    for run_name, run_path in RUNS.items():
        results[run_name] = analyze_run(
            run_name,
            run_path,
            include_dynamic=True,
        )
        print(f'completed {run_name}', flush=True)
    for run_name, run_path in INITIAL_ONLY_RUNS.items():
        results[run_name] = analyze_run(
            run_name,
            run_path,
            include_dynamic=False,
        )
        print(f'completed {run_name}', flush=True)

    output = HERE / 'results_first_order_controls'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    lines = [
        '# First-order control analysis',
        '',
        '| Run | Unit rho | Curvature rho | Gram rho | Unit NN | Curvature NN |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for run_name in RUNS:
        metrics = results[run_name]['metrics']
        unit_metrics = metrics['initial_unit_per_probe']
        curvature_metrics = metrics['reference_kappa_full']
        gram_metrics = metrics['initial_gram']
        unit_rho = unit_metrics['rho']
        curve_rho = curvature_metrics['rho']
        gram_rho = gram_metrics['rho']
        unit_nn = unit_metrics['nn_median']
        curve_nn = curvature_metrics['nn_median']
        lines.append(
            f'| {run_name} | {unit_rho:.3f} | {curve_rho:.3f} | {gram_rho:.3f} | '
            f'{unit_nn:.5f} | {curve_nn:.5f} |'
        )
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
