from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'rfps_feature_pilot'

RUNS = {
    'matched_scratch_seed1': PILOT / 'results_dual_scratch_seed1_curvature',
    'matched_scratch_seed2': PILOT / 'results_dual_scratch_seed2_curvature',
    'matched_warm_seed1': PILOT / 'results_dual_ckpt135_seed1_curvature',
    'matched_warm_seed2': PILOT / 'results_dual_ckpt135_seed2_curvature',
    'external_scratch_seed1': PILOT / 'results_external65_scratch_seed1_curvature',
    'external_scratch_seed2': PILOT / 'results_external65_scratch_seed2_curvature',
}

INITIAL_ONLY_RUNS = {
    'external_warmprobe_seed1': PILOT / 'results_epoch135_isflow_seed1',
    'external_warmprobe_seed2': PILOT / 'results_epoch135_isflow_seed2',
}


def load_run(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / 'descriptors.npz') as archive:
        return {name: archive[name].copy() for name in archive.files}


def sequential_false_skip(
    distance: np.ndarray,
    scores: np.ndarray,
    *,
    skip_quantile: float = 0.20,
    tolerance: float = 0.01,
) -> dict[str, float]:
    nearest_dist: list[float] = []
    nearest_gap: list[float] = []
    for idx in range(1, len(scores)):
        nearest = int(np.argmin(distance[idx, :idx]))
        nearest_dist.append(float(distance[idx, nearest]))
        nearest_gap.append(float(abs(scores[idx] - scores[nearest])))
    dist = np.asarray(nearest_dist)
    gap = np.asarray(nearest_gap)
    threshold = float(np.quantile(dist, skip_quantile))
    selected = dist <= threshold
    return {
        'threshold': threshold,
        'skip_rate': float(np.mean(selected)),
        'false_skip_rate': float(np.mean(gap[selected] > tolerance)),
        'n_skipped': int(np.count_nonzero(selected)),
    }


def descriptor_metrics(matrix: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    condensed = pdist(matrix, metric='euclidean')
    score_gap = pdist(scores[:, None], metric='euclidean')
    rho = float(spearmanr(condensed, score_gap).statistic)
    square = squareform(condensed)
    masked = square.copy()
    np.fill_diagonal(masked, np.inf)
    nearest = np.argmin(masked, axis=1)
    nn_gap = np.abs(scores - scores[nearest])
    risk = sequential_false_skip(square, scores)
    return {
        'rho': rho,
        'nn_median': float(np.median(nn_gap)),
        'nn_mean': float(np.mean(nn_gap)),
        'false_skip_20pct_tol001': risk['false_skip_rate'],
        'skip_rate': risk['skip_rate'],
        'dimension': int(matrix.shape[1]),
    }


def flatten_probe_blocks(blocks: np.ndarray) -> np.ndarray:
    return blocks.reshape(blocks.shape[0], -1) / math.sqrt(blocks.shape[1])


def derive_initial_descriptors(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_candidates = len(data['scores'])
    n_probes = data['initial_field'].shape[1] // 102
    block_size = data['initial_field'].shape[1] // n_probes
    n_solutions = block_size - 2
    blocks = data['initial_field'].reshape(n_candidates, n_probes, block_size)
    direction = blocks[:, :, :n_solutions]
    norms = np.linalg.norm(direction, axis=2, keepdims=True)
    normalized = direction / np.maximum(norms, 1e-12)
    gram = np.einsum('cpi,cqi->cpq', normalized, normalized)
    upper = np.triu_indices(n_probes)
    return {
        'initial_field': data['initial_field'],
        'initial_unit_per_probe': flatten_probe_blocks(normalized),
        'initial_gram': gram[:, upper[0], upper[1]],
    }


def derive_descriptors(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    scores = data['scores']
    n_candidates = len(scores)
    n_probes = data['initial_field'].shape[1] // 102
    n_solutions = data['flow_curvature'].shape[1] // (2 * n_probes)
    block_size = n_solutions + 2

    positions = data['three_flow_points'].reshape(
        n_candidates,
        n_probes,
        3,
        block_size,
    )
    eta = positions[:, :, :, :n_solutions]
    early = eta[:, :, 1, :] / 2.0 - eta[:, :, 0, :]
    late = eta[:, :, 2, :] / 3.0 - eta[:, :, 1, :] / 2.0
    second_difference = eta[:, :, 2, :] - 2.0 * eta[:, :, 1, :] + eta[:, :, 0, :]

    stored_full_blocks = data['flow_curvature'].reshape(
        n_candidates,
        n_probes,
        2 * n_solutions,
    )
    reconstructed_full = flatten_probe_blocks(
        np.concatenate([early, late], axis=2) / math.sqrt(2.0)
    )

    initial_blocks = data['initial_field'].reshape(n_candidates, n_probes, block_size)
    initial_direction = initial_blocks[:, :, :n_solutions]
    norms = np.linalg.norm(initial_direction, axis=2, keepdims=True)
    normalized = initial_direction / np.maximum(norms, 1e-12)
    gram = np.einsum('cpi,cqi->cpq', normalized, normalized)
    gram_upper = np.triu_indices(n_probes)
    initial_global = data['initial_field']
    global_norm = np.linalg.norm(initial_global, axis=1, keepdims=True)

    full = data['flow_curvature']
    rng = np.random.default_rng(20260728)
    descriptors = {
        'initial_field': data['initial_field'],
        'raw_three_positions': data['three_flow_points'],
        'kappa_full': full,
        'kappa_early': flatten_probe_blocks(early),
        'kappa_late': flatten_probe_blocks(late),
        'position_second_difference': flatten_probe_blocks(second_difference),
        'initial_unit_per_probe': flatten_probe_blocks(normalized),
        'initial_unit_global': initial_global / np.maximum(global_norm, 1e-12),
        'initial_norm_profile': np.log(np.maximum(norms[:, :, 0], 1e-12)),
        'initial_gram': gram[:, gram_upper[0], gram_upper[1]],
    }
    for dimension in (32, 64, 128):
        projection = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(full.shape[1], dimension),
        ) / math.sqrt(dimension)
        descriptors[f'kappa_rp{dimension}'] = full @ projection

    probe_blocks = {
        'kappa_full': stored_full_blocks,
        'kappa_early': early,
        'kappa_late': late,
        'initial_field': initial_blocks,
        'initial_unit_per_probe': normalized,
    }
    distance_ref = pdist(data['flow_curvature'])
    distance_recon = pdist(reconstructed_full)
    reconstruction = {
        'distance_spearman': float(spearmanr(distance_ref, distance_recon).statistic),
        'distance_relative_l2': float(
            np.linalg.norm(distance_ref / np.linalg.norm(distance_ref) - distance_recon / np.linalg.norm(distance_recon))
        ),
    }
    return descriptors, probe_blocks, reconstruction


def probe_count_ablation(
    blocks: np.ndarray,
    scores: np.ndarray,
) -> list[dict[str, float | int]]:
    n_probes = blocks.shape[1]
    rows: list[dict[str, float | int]] = []
    for count in (1, 2, 4, n_probes):
        combinations = list(itertools.combinations(range(n_probes), count))
        for subset_index, subset in enumerate(combinations):
            matrix = flatten_probe_blocks(blocks[:, subset, :])
            metrics = descriptor_metrics(matrix, scores)
            rows.append(
                {
                    'probe_count': count,
                    'subset_index': subset_index,
                    'rho': metrics['rho'],
                    'nn_median': metrics['nn_median'],
                    'false_skip': metrics['false_skip_20pct_tol001'],
                }
            )
    return rows


def summarize_probe_rows(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    counts = sorted({int(row['probe_count']) for row in rows})
    for count in counts:
        selected = [row for row in rows if int(row['probe_count']) == count]
        rho = np.asarray([float(row['rho']) for row in selected])
        nn = np.asarray([float(row['nn_median']) for row in selected])
        false_skip = np.asarray([float(row['false_skip']) for row in selected])
        summary.append(
            {
                'probe_count': count,
                'n_subsets': len(selected),
                'rho_min': float(np.min(rho)),
                'rho_median': float(np.median(rho)),
                'rho_max': float(np.max(rho)),
                'nn_median_of_subsets': float(np.median(nn)),
                'nn_worst': float(np.max(nn)),
                'false_skip_median': float(np.median(false_skip)),
                'false_skip_worst': float(np.max(false_skip)),
            }
        )
    return summary


def align_matrix(
    source: dict[str, np.ndarray],
    target_keys: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    source_index = {str(key): idx for idx, key in enumerate(source['keys'])}
    return np.stack([matrix[source_index[str(key)]] for key in target_keys])


def cross_anchor_analysis(
    loaded: dict[str, dict[str, np.ndarray]],
    derived: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in (1, 2):
        scratch_name = f'matched_scratch_seed{seed}'
        warm_name = f'matched_warm_seed{seed}'
        scratch = loaded[scratch_name]
        warm = loaded[warm_name]
        for descriptor_name in ('kappa_full', 'initial_unit_per_probe', 'initial_gram'):
            for target_name, target in (('scratch', scratch), ('warm', warm)):
                for source_name, source_run, source_data in (
                    ('scratch', scratch_name, scratch),
                    ('warm', warm_name, warm),
                ):
                    matrix = align_matrix(
                        source_data,
                        target['keys'],
                        derived[source_run][descriptor_name],
                    )
                    metrics = descriptor_metrics(matrix, target['scores'])
                    rows.append(
                        {
                            'seed': seed,
                            'descriptor': descriptor_name,
                            'target_label': target_name,
                            'descriptor_anchor': source_name,
                            **metrics,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def h1_decision(all_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    failures: list[str] = []
    scratch_runs = [
        'matched_scratch_seed1',
        'matched_scratch_seed2',
        'external_scratch_seed1',
        'external_scratch_seed2',
    ]
    warm_runs = ['matched_warm_seed1', 'matched_warm_seed2']
    for run_name in scratch_runs:
        full = all_metrics[run_name]['kappa_full']
        early = all_metrics[run_name]['kappa_early']
        if early['rho'] < full['rho'] - 0.05:
            failures.append(f'{run_name}: rho')
        if early['nn_median'] > 1.10 * full['nn_median']:
            failures.append(f'{run_name}: nn')
    for run_name in warm_runs:
        full = all_metrics[run_name]['kappa_full']
        early = all_metrics[run_name]['kappa_early']
        if early['rho'] < full['rho'] - 0.10:
            failures.append(f'{run_name}: warm rho')
    for run_name, metrics in all_metrics.items():
        full = metrics['kappa_full']
        early = metrics['kappa_early']
        if early['false_skip_20pct_tol001'] > full['false_skip_20pct_tol001'] + 0.05:
            failures.append(f'{run_name}: false skip')
    return {
        'passes_preregistered_rule': not failures,
        'failures': failures,
        'component_saved': 'third flow checkpoint and one curvature block',
    }


def main() -> int:
    output = HERE / 'results_existing'
    output.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    derived: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    reconstructions: dict[str, dict[str, float]] = {}
    metric_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    for run_name, run_path in RUNS.items():
        data = load_run(run_path)
        descriptors, blocks, reconstruction = derive_descriptors(data)
        loaded[run_name] = data
        derived[run_name] = descriptors
        reconstructions[run_name] = reconstruction
        metrics[run_name] = {}
        for descriptor_name, matrix in descriptors.items():
            result = descriptor_metrics(matrix, data['scores'])
            metrics[run_name][descriptor_name] = result
            metric_rows.append(
                {
                    'run': run_name,
                    'descriptor': descriptor_name,
                    **result,
                }
            )
        for descriptor_name in ('kappa_full', 'kappa_early', 'initial_unit_per_probe'):
            raw_rows = probe_count_ablation(blocks[descriptor_name], data['scores'])
            for row in summarize_probe_rows(raw_rows):
                probe_rows.append(
                    {
                        'run': run_name,
                        'descriptor': descriptor_name,
                        **row,
                    }
                )

    cross_anchor = cross_anchor_analysis(loaded, derived)
    decision = h1_decision(metrics)
    initial_only_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for run_name, run_path in INITIAL_ONLY_RUNS.items():
        data = load_run(run_path)
        initial_only_metrics[run_name] = {}
        for descriptor_name, matrix in derive_initial_descriptors(data).items():
            result = descriptor_metrics(matrix, data['scores'])
            initial_only_metrics[run_name][descriptor_name] = result
            metric_rows.append(
                {
                    'run': run_name,
                    'descriptor': descriptor_name,
                    **result,
                }
            )
    payload = {
        'metrics': metrics,
        'initial_only_metrics': initial_only_metrics,
        'reconstruction_checks': reconstructions,
        'h1_two_checkpoint_decision': decision,
        'cross_anchor': cross_anchor,
        'probe_count_summary': probe_rows,
    }
    (output / 'results.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    write_csv(output / 'descriptor_metrics.csv', metric_rows)
    write_csv(output / 'probe_count_summary.csv', probe_rows)
    write_csv(output / 'cross_anchor.csv', cross_anchor)

    h1_pass = decision['passes_preregistered_rule']
    lines = [
        '# Existing-descriptor minimality ablations',
        '',
        f'**H1 two-checkpoint decision: {h1_pass}.**',
        '',
        '| Run | Initial rho | Full curvature rho | Early curvature rho | Full NN | Early NN |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for run_name in RUNS:
        initial = metrics[run_name]['initial_field']
        full = metrics[run_name]['kappa_full']
        early = metrics[run_name]['kappa_early']
        initial_rho = initial['rho']
        full_rho = full['rho']
        early_rho = early['rho']
        full_nn = full['nn_median']
        early_nn = early['nn_median']
        lines.append(
            f'| {run_name} | {initial_rho:.3f} | {full_rho:.3f} | '
            f'{early_rho:.3f} | {full_nn:.5f} | {early_nn:.5f} |'
        )
    lines.extend(
        [
            '',
            '## Preregistered H1 failures',
            '',
            *([f'- {item}' for item in decision['failures']] or ['- None.']),
            '',
            '## Interpretation',
            '',
            'This file reports post-protocol computations from already saved descriptors. '
            'Two-point field-change and Euclidean-update baselines require a new descriptor run '
            'and are analyzed separately.',
            '',
        ]
    )
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
