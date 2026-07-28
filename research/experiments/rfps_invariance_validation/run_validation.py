from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'rfps_feature_pilot'
MINIMALITY = HERE.parent / 'rfps_minimality'
sys.path[:0] = [str(PILOT), str(MINIMALITY)]

import run_pilot as pilot  # noqa: E402
from run_step_sweep import coefficient_field  # noqa: E402
from run_two_point_baselines import sphere_parallel_transport, unit  # noqa: E402

ELL = 0.03
BETAS = (0.0, 0.25, 0.5, 1.0)
R_COUNTS = (2, 4, 8, 12, 16)
KAPPAS = (1, 3, 10, 30)
N_SUBSETS = 100
N_ORDERS = 200
SEED = 20260728

PROBE_BANKS = (
    PILOT / 'tsp100_epoch135_rollout_probes.npz',
    PILOT / 'tsp100_epoch135_rollout_probes_seed2.npz',
)

DATASETS = {
    'matched': {
        'source': PILOT / 'results_dual_scratch_seed1_curvature',
        'targets': {
            'scratch': PILOT / 'results_dual_scratch_seed1_curvature',
            'warm': PILOT / 'results_dual_ckpt135_seed1_curvature',
        },
    },
    'external': {
        'source': PILOT / 'results_external65_scratch_seed1_curvature',
        'targets': {'scratch': PILOT / 'results_epoch135_isflow_seed1'},
    },
}


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / 'descriptors.npz') as archive:
        return {name: archive[name].copy() for name in archive.files}


def load_candidates(source: Path) -> tuple[list[pilot.Candidate], np.ndarray]:
    meta = json.loads((source / 'results.json').read_text(encoding='utf-8'))
    candidates, failures = pilot.load_candidates(
        Path(meta['data_source']), meta['score_target']
    )
    if failures:
        raise RuntimeError(f'candidate loading failed: {failures[:3]}')
    keys = read_npz(source)['keys']
    by_key = {candidate.key: candidate for candidate in candidates}
    return [by_key[str(key)] for key in keys], keys


def aligned_scores(path: Path, keys: np.ndarray) -> np.ndarray:
    data = read_npz(path)
    index = {str(key): idx for idx, key in enumerate(data['keys'])}
    return np.asarray([data['scores'][index[str(key)]] for key in keys])


def load_probes() -> list[pilot.Probe]:
    probes: list[pilot.Probe] = []
    for path in PROBE_BANKS:
        probes.extend(pilot.load_probes_npz(path))
    if len(probes) != 16:
        raise RuntimeError(f'expected 16 probes, found {len(probes)}')
    return probes


def helmert_basis(size: int) -> np.ndarray:
    basis = np.zeros((size, size - 1), dtype=np.float64)
    for j in range(1, size):
        scale = math.sqrt(j * (j + 1))
        basis[:j, j - 1] = 1.0 / scale
        basis[j, j - 1] = -j / scale
    return basis


def tilted_distribution(probe: pilot.Probe, beta: float) -> torch.Tensor:
    objective = probe.objective.reshape(-1).detach().cpu().numpy()
    median = float(np.median(objective))
    mad = float(np.median(np.abs(objective - median)))
    scale = max(1.4826 * mad, 1e-12)
    standardized = np.clip((objective - median) / scale, -3.0, 3.0)
    logits = torch.as_tensor(-beta * standardized, dtype=pilot.DTYPE)
    return torch.softmax(logits, dim=0)


def fisher_euclidean_blocks(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    q0: torch.Tensor,
) -> tuple[dict[str, np.ndarray], torch.Tensor, float]:
    _, u0, speed0 = coefficient_field(candidate, probe, q0)
    h0 = torch.sqrt(q0)
    v0 = u0 / (2.0 * h0)
    d0 = unit(v0)

    z1 = torch.log(q0) + ELL * unit(u0)
    q_euclidean = torch.softmax(z1, dim=0)
    _, u_euclidean, _ = coefficient_field(candidate, probe, q_euclidean)
    euclidean = torch.cat([d0, unit(u_euclidean)]) / math.sqrt(2.0)

    if speed0 <= 1e-14:
        q_fisher = q0.clone()
    else:
        q_fisher = pilot.retract_fisher_sphere(q0, u0, ELL / speed0)
    _, u_fisher, _ = coefficient_field(candidate, probe, q_fisher)
    h_fisher = torch.sqrt(q_fisher)
    v_fisher = u_fisher / (2.0 * h_fisher)
    transported = sphere_parallel_transport(h_fisher, h0, v_fisher)
    fisher = torch.cat([d0, unit(transported)]) / math.sqrt(2.0)
    arc = 2.0 * math.acos(float(torch.dot(h0, h_fisher).clamp(-1.0, 1.0)))
    blocks = {
        'one_point': d0.detach().cpu().numpy(),
        'euclidean_two_point': euclidean.detach().cpu().numpy(),
        'fisher_two_point': fisher.detach().cpu().numpy(),
    }
    return blocks, u0, abs(arc - (0.0 if speed0 <= 1e-14 else ELL))


def chart_pair(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    q0: torch.Tensor,
    u0: torch.Tensor,
    basis: torch.Tensor,
    scales: torch.Tensor,
) -> np.ndarray:
    centered = torch.log(q0) - torch.log(q0).mean()
    x0 = basis.T @ centered
    eta0 = scales * x0
    gradient0 = (basis.T @ u0) / scales
    eta1 = eta0 + ELL * unit(gradient0)
    x1 = eta1 / scales
    q1 = torch.softmax(basis @ x1, dim=0)
    _, u1, _ = coefficient_field(candidate, probe, q1)
    gradient1 = (basis.T @ u1) / scales
    pair = torch.cat([unit(gradient0), unit(gradient1)]) / math.sqrt(2.0)
    return pair.detach().cpu().numpy()


def flatten_blocks(blocks: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    selected = blocks if subset is None else blocks[:, subset, :]
    return selected.reshape(selected.shape[0], -1) / math.sqrt(selected.shape[1])


def order_risk(
    matrix: np.ndarray,
    scores: np.ndarray,
    *,
    n_trials: int = N_ORDERS,
    seed: int = SEED + 1,
) -> dict[str, float]:
    distance = squareform(pdist(matrix))
    rng = np.random.default_rng(seed)
    false_rates = np.empty(n_trials)
    skipped_mae = np.empty(n_trials)
    for trial in range(n_trials):
        order = rng.permutation(len(scores))
        nearest_distance = []
        nearest_gap = []
        for position in range(1, len(order)):
            current = int(order[position])
            previous = order[:position]
            neighbor = int(previous[np.argmin(distance[current, previous])])
            nearest_distance.append(distance[current, neighbor])
            nearest_gap.append(abs(scores[current] - scores[neighbor]))
        distances = np.asarray(nearest_distance)
        gaps = np.asarray(nearest_gap)
        selected = distances <= np.quantile(distances, 0.20)
        false_rates[trial] = np.mean(gaps[selected] > 0.01)
        skipped_mae[trial] = np.mean(gaps[selected])
    return {
        'false_skip_mean': float(np.mean(false_rates)),
        'false_skip_q90': float(np.quantile(false_rates, 0.90)),
        'skipped_mae_mean': float(np.mean(skipped_mae)),
    }


def descriptor_metrics(matrix: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    condensed = pdist(matrix)
    gaps = pdist(scores[:, None])
    rho = float(spearmanr(condensed, gaps).statistic)
    distance = squareform(condensed)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    nn_gap = np.abs(scores - scores[nearest])
    return {
        'rho': rho,
        'nn_median': float(np.median(nn_gap)),
        'nn_mean': float(np.mean(nn_gap)),
    }


def agreement(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    ref_distance = pdist(reference)
    cur_distance = pdist(current)
    rank = float(spearmanr(ref_distance, cur_distance).statistic)
    ref_square = squareform(ref_distance)
    cur_square = squareform(cur_distance)
    np.fill_diagonal(ref_square, np.inf)
    np.fill_diagonal(cur_square, np.inf)
    ref_order = np.argsort(ref_square, axis=1)
    cur_order = np.argsort(cur_square, axis=1)
    top1 = float(np.mean(ref_order[:, 0] == cur_order[:, 0]))
    overlaps = [
        len(set(ref_order[i, :5]).intersection(cur_order[i, :5])) / 5.0
        for i in range(len(reference))
    ]
    return {'distance_rank': rank, 'top1': top1, 'top5_overlap': float(np.mean(overlaps))}


def coordinate_specs(size: int) -> list[tuple[int, int, np.ndarray]]:
    specs: list[tuple[int, int, np.ndarray]] = [(1, 0, np.ones(size))]
    rng = np.random.default_rng(SEED + 2)
    for kappa in KAPPAS[1:]:
        base = np.exp(np.linspace(-0.5 * math.log(kappa), 0.5 * math.log(kappa), size))
        for permutation in range(5):
            specs.append((kappa, permutation, base[rng.permutation(size)]))
    return specs


def compute_descriptors(
    probes: list[pilot.Probe],
    *,
    candidate_limit: int | None = None,
    probe_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_probes = probes[:probe_limit] if probe_limit else probes
    basis_np = helmert_basis(selected_probes[0].p0.numel())
    basis = torch.as_tensor(basis_np, dtype=pilot.DTYPE)
    specs = coordinate_specs(basis.shape[1])
    stores: dict[str, Any] = {}
    sanity: dict[str, Any] = {
        'basis_orthonormal_error': float(np.max(np.abs(basis_np.T @ basis_np - np.eye(basis_np.shape[1])))),
        'basis_centering_error': float(np.max(np.abs(basis_np.T @ np.ones(basis_np.shape[0])))),
        'max_fisher_arc_error': 0.0,
    }
    for dataset_name, config in DATASETS.items():
        candidates, keys = load_candidates(config['source'])
        if candidate_limit:
            candidates = candidates[:candidate_limit]
            keys = keys[:candidate_limit]
        raw = {
            beta: {method: [] for method in ('one_point', 'euclidean_two_point', 'fisher_two_point')}
            for beta in BETAS
        }
        chart_raw: dict[tuple[int, int], list[np.ndarray]] = {
            (kappa, permutation): [] for kappa, permutation, _ in specs
        }
        for candidate_index, candidate in enumerate(candidates, start=1):
            per_beta = {
                beta: {method: [] for method in raw[beta]}
                for beta in BETAS
            }
            per_chart: dict[tuple[int, int], list[np.ndarray]] = {
                key: [] for key in chart_raw
            }
            for probe in selected_probes:
                uniform_u0: torch.Tensor | None = None
                for beta in BETAS:
                    q0 = tilted_distribution(probe, beta)
                    blocks, u0, arc_error = fisher_euclidean_blocks(candidate, probe, q0)
                    sanity['max_fisher_arc_error'] = max(
                        sanity['max_fisher_arc_error'], arc_error
                    )
                    for method, vector in blocks.items():
                        per_beta[beta][method].append(vector)
                    if beta == 0.0:
                        uniform_u0 = u0
                if uniform_u0 is None:
                    raise RuntimeError('uniform field was not computed')
                q_uniform = tilted_distribution(probe, 0.0)
                for kappa, permutation, scales_np in specs:
                    scales = torch.as_tensor(scales_np, dtype=pilot.DTYPE)
                    per_chart[(kappa, permutation)].append(
                        chart_pair(candidate, probe, q_uniform, uniform_u0, basis, scales)
                    )
            for beta in BETAS:
                for method in raw[beta]:
                    raw[beta][method].append(np.stack(per_beta[beta][method]))
            for key in chart_raw:
                chart_raw[key].append(np.stack(per_chart[key]))
            print(
                f'[{dataset_name}] {candidate_index:03d}/{len(candidates):03d}',
                flush=True,
            )
        stores[dataset_name] = {
            'keys': keys,
            'blocks': {
                beta: {method: np.stack(values) for method, values in methods.items()}
                for beta, methods in raw.items()
            },
            'chart_blocks': {key: np.stack(values) for key, values in chart_raw.items()},
            'targets': {
                name: aligned_scores(path, keys)
                for name, path in config['targets'].items()
            },
        }
        identity = flatten_blocks(stores[dataset_name]['chart_blocks'][(1, 0)])
        original = flatten_blocks(stores[dataset_name]['blocks'][0.0]['euclidean_two_point'])
        sanity[f'{dataset_name}_identity_distance_error'] = float(
            np.max(np.abs(pdist(identity) - pdist(original)))
        )
    ess = {
        str(beta): [
            float(1.0 / q.square().sum().item() / q.numel())
            for probe in selected_probes
            for q in [tilted_distribution(probe, beta)]
        ]
        for beta in BETAS
    }
    sanity['ess'] = ess
    return stores, sanity


def probe_subsets(count: int, n_probes: int) -> list[np.ndarray]:
    if count == n_probes:
        return [np.arange(n_probes)]
    rng = np.random.default_rng(SEED + count)
    return [np.sort(rng.choice(n_probes, count, replace=False)) for _ in range(N_SUBSETS)]


def experiment_instances(stores: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, store in stores.items():
        for target, scores in store['targets'].items():
            for method, blocks in store['blocks'][0.0].items():
                for count in R_COUNTS:
                    if count > blocks.shape[1]:
                        continue
                    for subset_index, subset in enumerate(probe_subsets(count, blocks.shape[1])):
                        matrix = flatten_blocks(blocks, subset)
                        row = {
                            'dataset': dataset,
                            'target': target,
                            'method': method,
                            'R': count,
                            'subset': subset_index,
                        }
                        row.update(descriptor_metrics(matrix, scores))
                        row.update(order_risk(matrix, scores, seed=SEED + count + subset_index))
                        rows.append(row)
    return rows


def experiment_coordinates(stores: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, store in stores.items():
        invariant = {
            method: flatten_blocks(store['blocks'][0.0][method])
            for method in ('one_point', 'fisher_two_point')
        }
        euclidean_reference = flatten_blocks(store['chart_blocks'][(1, 0)])
        for (kappa, permutation), chart_blocks in store['chart_blocks'].items():
            matrices = {
                'one_point': invariant['one_point'],
                'fisher_two_point': invariant['fisher_two_point'],
                'euclidean_two_point': flatten_blocks(chart_blocks),
            }
            references = {
                'one_point': invariant['one_point'],
                'fisher_two_point': invariant['fisher_two_point'],
                'euclidean_two_point': euclidean_reference,
            }
            for target, scores in store['targets'].items():
                for method, matrix in matrices.items():
                    row = {
                        'dataset': dataset,
                        'target': target,
                        'method': method,
                        'kappa': kappa,
                        'permutation': permutation,
                    }
                    row.update(agreement(references[method], matrix))
                    row.update(descriptor_metrics(matrix, scores))
                    row.update(order_risk(matrix, scores, seed=SEED + 3))
                    rows.append(row)
    return rows


def experiment_nonuniform(
    stores: dict[str, Any], sanity: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, store in stores.items():
        uniform = {
            method: flatten_blocks(blocks)
            for method, blocks in store['blocks'][0.0].items()
        }
        for beta in BETAS:
            matrices = {
                method: flatten_blocks(blocks)
                for method, blocks in store['blocks'][beta].items()
            }
            pair_agreement = agreement(
                matrices['euclidean_two_point'], matrices['fisher_two_point']
            )
            ess = np.asarray(sanity['ess'][str(beta)])
            for target, scores in store['targets'].items():
                for method, matrix in matrices.items():
                    row = {
                        'dataset': dataset,
                        'target': target,
                        'method': method,
                        'beta': beta,
                        'ess_median': float(np.median(ess)),
                        'ess_min': float(np.min(ess)),
                        'ef_distance_rank': pair_agreement['distance_rank'],
                        'ef_top1': pair_agreement['top1'],
                        'ef_top5_overlap': pair_agreement['top5_overlap'],
                    }
                    drift = agreement(uniform[method], matrix)
                    row.update({f'drift_{key}': value for key, value in drift.items()})
                    row.update(descriptor_metrics(matrix, scores))
                    row.update(order_risk(matrix, scores, seed=SEED + 4))
                    rows.append(row)
    return rows


def summarize(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    value_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, members in groups.items():
        result = dict(zip(group_keys, group))
        result['n'] = len(members)
        for key in value_keys:
            values = np.asarray([float(member[key]) for member in members])
            result[f'{key}_median'] = float(np.nanmedian(values))
            result[f'{key}_q025'] = float(np.nanquantile(values, 0.025))
            result[f'{key}_q975'] = float(np.nanquantile(values, 0.975))
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(
    instance_summary: list[dict[str, Any]],
    coordinate_summary: list[dict[str, Any]],
    nonuniform_rows: list[dict[str, Any]],
    sanity: dict[str, Any],
) -> str:
    lines = [
        '# Fisher-invariant validation results',
        '',
        'Protocol: `../protocol.md`. All descriptors use ell = 0.03.',
        '',
        '## Numerical checks',
        '',
        '- Helmert orthonormal error: {:.3e}'.format(sanity['basis_orthonormal_error']),
        '- Helmert centering error: {:.3e}'.format(sanity['basis_centering_error']),
        '- Maximum Fisher arc error: {:.3e}'.format(sanity['max_fisher_arc_error']),
    ]
    for key, value in sanity.items():
        if key.endswith('identity_distance_error'):
            lines.append('- {}: {:.3e}'.format(key, value))
    lines.extend([
        '',
        '## A. Instance-count sweep (median across subsets)',
        '',
        '| Data | Target | Method | R | rho | NN median | False skip |',
        '|---|---|---|---:|---:|---:|---:|',
    ])
    for row in instance_summary:
        lines.append(
            '| {dataset} | {target} | {method} | {R} | {rho_median:.3f} | '
            '{nn_median_median:.5f} | {false_skip_mean_median:.3f} |'.format(**row)
        )
    lines.extend([
        '',
        '## B. Coordinate stress (median across permutations)',
        '',
        '| Data | Target | Method | kappa | rank | top-1 | top-5 | rho |',
        '|---|---|---|---:|---:|---:|---:|---:|',
    ])
    for row in coordinate_summary:
        lines.append(
            '| {dataset} | {target} | {method} | {kappa} | {distance_rank_median:.3f} | '
            '{top1_median:.3f} | {top5_overlap_median:.3f} | {rho_median:.3f} |'.format(**row)
        )
    lines.extend([
        '',
        '## C. Non-uniform starts',
        '',
        '| Data | Target | Method | beta | ESS | E/F rank | E/F top-1 | rho | NN |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in nonuniform_rows:
        lines.append(
            '| {dataset} | {target} | {method} | {beta:.2f} | {ess_median:.3f} | '
            '{ef_distance_rank:.3f} | {ef_top1:.3f} | {rho:.3f} | {nn_median:.5f} |'.format(**row)
        )
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-limit', type=int)
    parser.add_argument('--probe-limit', type=int)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    torch.set_default_dtype(pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    probes = load_probes()
    stores, sanity = compute_descriptors(
        probes,
        candidate_limit=args.candidate_limit,
        probe_limit=args.probe_limit,
    )
    if sanity['basis_orthonormal_error'] > 1e-12:
        raise RuntimeError('Helmert basis is not orthonormal')
    if sanity['basis_centering_error'] > 1e-12:
        raise RuntimeError('Helmert basis does not remove the translation gauge')
    if sanity['max_fisher_arc_error'] > 1e-8:
        raise RuntimeError('Fisher step does not have the requested arc length')
    for key, value in sanity.items():
        if key.endswith('identity_distance_error') and value > 1e-10:
            raise RuntimeError(f'identity chart failed to reproduce baseline: {key}')
    instance_rows = experiment_instances(stores)
    coordinate_rows = experiment_coordinates(stores)
    nonuniform_rows = experiment_nonuniform(stores, sanity)
    instance_summary = summarize(
        instance_rows,
        ('dataset', 'target', 'method', 'R'),
        ('rho', 'nn_median', 'false_skip_mean', 'skipped_mae_mean'),
    )
    coordinate_summary = summarize(
        coordinate_rows,
        ('dataset', 'target', 'method', 'kappa'),
        ('distance_rank', 'top1', 'top5_overlap', 'rho', 'nn_median', 'false_skip_mean'),
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'instance_count_raw.csv', instance_rows)
    write_csv(output / 'instance_count_summary.csv', instance_summary)
    write_csv(output / 'coordinate_stress_raw.csv', coordinate_rows)
    write_csv(output / 'coordinate_stress_summary.csv', coordinate_summary)
    write_csv(output / 'nonuniform.csv', nonuniform_rows)
    payload = {
        'protocol': str(HERE / 'protocol.md'),
        'ell': ELL,
        'sanity': sanity,
        'instance_count_summary': instance_summary,
        'coordinate_stress_summary': coordinate_summary,
        'nonuniform': nonuniform_rows,
    }
    (output / 'results.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    summary = render_summary(instance_summary, coordinate_summary, nonuniform_rows, sanity)
    (output / 'summary.md').write_text(summary, encoding='utf-8')
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
