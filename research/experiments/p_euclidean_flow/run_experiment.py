from __future__ import annotations

import csv
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
CROSS_BANK_DIR = EXPERIMENTS / 'rfps_cross_bank_scan'
sys.path.insert(0, str(CROSS_BANK_DIR))

import run_length_sweep as sweep  # noqa: E402


pilot = sweep.pilot
FLOW_LENGTH = 0.03
STEP_COUNTS = (30, 60)
GEOMETRIES = ('p_euclidean', 'fisher')
RESULTS_DIR = HERE / 'results'


def unit(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= 1e-14:
        return torch.zeros_like(vector)
    return vector / norm


def training_covector(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    probability: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    '''Return c=-dL/dlog(p), with sampling statistics stop-gradiented.'''
    log_density_ratio = (
        torch.log(probability.clamp_min(pilot.T_FLOOR))
        - torch.log(reference.clamp_min(pilot.T_FLOOR))
    )
    trajectory_log_probability = (
        probe.p0 + log_density_ratio.unsqueeze(0)
    ).detach().requires_grad_(True)
    batch, model_output = pilot.pairwise_inputs_importance(
        probe,
        trajectory_log_probability,
        probability,
    )
    loss = candidate.compiled.loss_fn(
        batch,
        model_output,
        {'alpha': candidate.alpha},
    )
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, dtype=pilot.DTYPE)
    if loss.numel() != 1:
        loss = loss.mean()
    if not torch.isfinite(loss):
        raise ValueError('non-finite candidate loss')
    if loss.requires_grad:
        gradient = torch.autograd.grad(
            loss,
            trajectory_log_probability,
            allow_unused=True,
        )[0]
    else:
        gradient = None
    covector = (
        torch.zeros_like(probability)
        if gradient is None
        else -gradient.reshape(-1)
    )
    if not torch.isfinite(covector).all():
        raise ValueError('non-finite training covector')
    return covector.detach()


def metric_fields(
    covector: torch.Tensor,
    probability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    fisher = covector - probability * covector.sum()
    probability_covector = covector / probability.clamp_min(pilot.T_FLOOR)
    p_euclidean = probability_covector - probability_covector.mean()
    return {'p_euclidean': p_euclidean, 'fisher': fisher}


def fisher_speed(
    tangent: torch.Tensor,
    probability: torch.Tensor,
) -> float:
    value = (
        tangent.square() / probability.clamp_min(pilot.T_FLOOR)
    ).sum()
    return float(torch.sqrt(value.clamp_min(0.0)).item())


def flat_direction_at_base(
    tangent: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    return unit(tangent / (2.0 * torch.sqrt(base)))


def fisher_direction(
    tangent: torch.Tensor,
    probability: torch.Tensor,
) -> torch.Tensor:
    return unit(tangent / (2.0 * torch.sqrt(probability)))


def integrate_flow(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    base: torch.Tensor,
    geometry: str,
    steps: int,
    initial_covector: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if geometry not in GEOMETRIES:
        raise ValueError(f'unknown geometry {geometry}')
    probability = base.clone()
    root_path = [torch.sqrt(probability)]
    ds = FLOW_LENGTH / float(steps)
    max_sum_error = 0.0
    max_step_error = 0.0
    minimum_probability = float(probability.min().item())
    stationary_steps = 0

    initial_tangent = metric_fields(initial_covector, probability)[geometry]
    if geometry == 'fisher':
        initial_direction = fisher_direction(initial_tangent, probability)
    else:
        initial_direction = flat_direction_at_base(initial_tangent, base)

    for index in range(steps):
        covector = (
            initial_covector
            if index == 0
            else training_covector(candidate, probe, probability, base)
        )
        tangent = metric_fields(covector, probability)[geometry]
        max_sum_error = max(max_sum_error, abs(float(tangent.sum().item())))
        speed = fisher_speed(tangent, probability)
        if speed <= 1e-14:
            stationary_steps += 1
            root_path.append(torch.sqrt(probability))
            continue
        next_probability = pilot.retract_fisher_sphere(
            probability,
            tangent,
            ds / speed,
        )
        actual_step = sweep.fisher_distance(probability, next_probability)
        max_step_error = max(max_step_error, abs(actual_step - ds))
        if not torch.isfinite(next_probability).all():
            raise ValueError('non-finite flow state')
        if float(next_probability.min().item()) <= 0.0:
            raise ValueError('flow left simplex interior')
        if abs(float(next_probability.sum().item()) - 1.0) > 1e-10:
            raise ValueError('flow state does not sum to one')
        probability = next_probability
        root_path.append(torch.sqrt(probability))
        minimum_probability = min(
            minimum_probability,
            float(probability.min().item()),
        )

    terminal_covector = training_covector(
        candidate,
        probe,
        probability,
        base,
    )
    terminal_tangent = metric_fields(
        terminal_covector,
        probability,
    )[geometry]
    max_sum_error = max(
        max_sum_error,
        abs(float(terminal_tangent.sum().item())),
    )

    if geometry == 'fisher':
        terminal_direction = fisher_direction(
            terminal_tangent,
            probability,
        )
        for source, target in zip(
            reversed(root_path[1:]),
            reversed(root_path[:-1]),
        ):
            terminal_direction = sweep.sphere_parallel_transport(
                source,
                target,
                terminal_direction,
            )
        terminal_direction = unit(terminal_direction)
    else:
        terminal_direction = flat_direction_at_base(terminal_tangent, base)

    descriptor = torch.cat(
        [initial_direction, terminal_direction]
    ) / math.sqrt(2.0)
    endpoint_log = pilot.fisher_log_map(base, probability)
    endpoint_direction = unit(endpoint_log)
    diagnostics = {
        'max_sum_error': max_sum_error,
        'max_step_error': max_step_error,
        'minimum_probability': minimum_probability,
        'stationary_fraction': stationary_steps / float(steps),
        'endpoint_fr_distance': sweep.fisher_distance(base, probability),
    }
    return descriptor.numpy(), endpoint_direction.numpy(), diagnostics


def legacy_controls(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    base: torch.Tensor,
    initial_covector: torch.Tensor,
) -> dict[str, np.ndarray]:
    fisher0 = metric_fields(initial_covector, base)['fisher']
    speed0 = fisher_speed(fisher0, base)
    d0 = fisher_direction(fisher0, base)

    logit_probability = torch.softmax(
        torch.log(base) + FLOW_LENGTH * unit(fisher0),
        dim=0,
    )
    logit_covector = training_covector(
        candidate,
        probe,
        logit_probability,
        base,
    )
    logit_fisher = metric_fields(
        logit_covector,
        logit_probability,
    )['fisher']
    legacy_logit = torch.cat([d0, unit(logit_fisher)]) / math.sqrt(2.0)

    fisher_probability = (
        base.clone()
        if speed0 <= 1e-14
        else pilot.retract_fisher_sphere(
            base,
            fisher0,
            FLOW_LENGTH / speed0,
        )
    )
    fisher_covector = training_covector(
        candidate,
        probe,
        fisher_probability,
        base,
    )
    fisher1 = metric_fields(
        fisher_covector,
        fisher_probability,
    )['fisher']
    transported = sweep.sphere_parallel_transport(
        torch.sqrt(fisher_probability),
        torch.sqrt(base),
        fisher1 / (2.0 * torch.sqrt(fisher_probability)),
    )
    legacy_fisher = torch.cat([d0, unit(transported)]) / math.sqrt(2.0)
    return {
        'one_point': d0.numpy(),
        'legacy_logit_two_point': legacy_logit.numpy(),
        'legacy_fisher_one_step': legacy_fisher.numpy(),
    }


def descriptors_for_probe(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    count = probe.objective.numel()
    base = torch.full((count,), 1.0 / count, dtype=pilot.DTYPE)
    initial_covector = training_covector(candidate, probe, base, base)
    initial_fields = metric_fields(initial_covector, base)
    euclidean_initial = flat_direction_at_base(
        initial_fields['p_euclidean'],
        base,
    )
    fisher_initial = fisher_direction(initial_fields['fisher'], base)
    if (
        torch.linalg.vector_norm(euclidean_initial) > 1e-14
        and torch.linalg.vector_norm(fisher_initial) > 1e-14
    ):
        initial_cosine = float(
            torch.dot(euclidean_initial, fisher_initial).item()
        )
    else:
        initial_cosine = 1.0

    blocks = legacy_controls(
        candidate,
        probe,
        base,
        initial_covector,
    )
    diagnostics: list[dict[str, Any]] = [
        {
            'geometry': 'initial',
            'steps': 0,
            'initial_cosine': initial_cosine,
            'max_sum_error': max(
                abs(float(value.sum().item()))
                for value in initial_fields.values()
            ),
            'max_step_error': 0.0,
            'minimum_probability': float(base.min().item()),
            'stationary_fraction': 0.0,
            'endpoint_fr_distance': 0.0,
        }
    ]
    for steps in STEP_COUNTS:
        for geometry in GEOMETRIES:
            descriptor, endpoint, values = integrate_flow(
                candidate,
                probe,
                base,
                geometry,
                steps,
                initial_covector,
            )
            blocks[f'{geometry}_flow_{steps}'] = descriptor
            blocks[f'{geometry}_endpoint_{steps}'] = endpoint
            diagnostics.append(
                {
                    'geometry': geometry,
                    'steps': steps,
                    'initial_cosine': initial_cosine,
                    **values,
                }
            )
    return blocks, diagnostics


def matrices_for_bank(
    candidates: list[pilot.Candidate],
    probes: list[pilot.Probe],
    *,
    dataset_name: str,
    bank_name: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    raw: dict[str, list[np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_blocks: dict[str, list[np.ndarray]] = {}
        for probe_index, probe in enumerate(probes):
            blocks, rows = descriptors_for_probe(candidate, probe)
            for name, vector in blocks.items():
                candidate_blocks.setdefault(name, []).append(vector)
            for row in rows:
                diagnostics.append(
                    {
                        'dataset': dataset_name,
                        'bank': bank_name,
                        'candidate': index - 1,
                        'probe': probe_index,
                        **row,
                    }
                )
        for name, vectors in candidate_blocks.items():
            raw.setdefault(name, []).append(
                np.concatenate(vectors) / math.sqrt(len(vectors))
            )
        print(
            f'[{dataset_name}/{bank_name}] '
            f'{index:03d}/{len(candidates):03d}',
            flush=True,
        )
    return {name: np.stack(rows) for name, rows in raw.items()}, diagnostics


def distance_comparison(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    left_distance = pdist(left)
    right_distance = pdist(right)
    rho = float(spearmanr(left_distance, right_distance).statistic)
    left_square = squareform(left_distance)
    right_square = squareform(right_distance)
    np.fill_diagonal(left_square, np.inf)
    np.fill_diagonal(right_square, np.inf)
    left_nearest = np.argmin(left_square, axis=1)
    right_nearest = np.argmin(right_square, axis=1)
    return {
        'distance_rho': rho,
        'top1_agreement': float(np.mean(left_nearest == right_nearest)),
    }


def summarize_sanity(rows: list[dict[str, Any]]) -> dict[str, float]:
    flow_rows = [row for row in rows if row['steps'] > 0]
    return {
        'minimum_initial_cosine': float(
            min(row['initial_cosine'] for row in rows)
        ),
        'maximum_tangent_sum_error': float(
            max(row['max_sum_error'] for row in rows)
        ),
        'maximum_substep_length_error': float(
            max(row['max_step_error'] for row in flow_rows)
        ),
        'minimum_probability': float(
            min(row['minimum_probability'] for row in flow_rows)
        ),
        'maximum_stationary_fraction': float(
            max(row['stationary_fraction'] for row in flow_rows)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    torch.set_default_dtype(pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    banks = sweep.load_banks()
    datasets = sweep.load_datasets()
    output: dict[str, Any] = {
        'flow_length': FLOW_LENGTH,
        'step_counts': STEP_COUNTS,
        'datasets': {},
    }
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    archive: dict[str, np.ndarray] = {}

    for dataset_name, dataset in datasets.items():
        bank_matrices: dict[str, dict[str, np.ndarray]] = {}
        for bank_name, probes in banks.items():
            matrices, diagnostics = matrices_for_bank(
                dataset['candidates'],
                probes,
                dataset_name=dataset_name,
                bank_name=bank_name,
            )
            bank_matrices[bank_name] = matrices
            diagnostic_rows.extend(diagnostics)

        primary_gaps, target_gaps = sweep.target_geometry(
            dataset_name,
            dataset['targets'],
        )
        descriptor_names = tuple(next(iter(bank_matrices.values())))
        all_matrices: dict[str, dict[str, np.ndarray]] = {
            name: {} for name in descriptor_names
        }
        for descriptor in descriptor_names:
            for bank_name in ('epoch031', 'epoch135'):
                all_matrices[descriptor][bank_name] = bank_matrices[
                    bank_name
                ][descriptor]
            all_matrices[descriptor]['product'] = np.concatenate(
                [
                    bank_matrices['epoch031'][descriptor] / math.sqrt(2.0),
                    bank_matrices['epoch135'][descriptor] / math.sqrt(2.0),
                ],
                axis=1,
            )

        metrics: dict[str, dict[str, float]] = {}
        for descriptor, matrices in all_matrices.items():
            for bank_name, matrix in matrices.items():
                key = f'{descriptor}:{bank_name}'
                value = sweep.descriptor_metrics(
                    matrix,
                    primary_gaps,
                    target_gaps,
                )
                metrics[key] = value
                metric_rows.append(
                    {
                        'dataset': dataset_name,
                        'descriptor': descriptor,
                        'bank': bank_name,
                        **value,
                    }
                )
                archive[f'{dataset_name}__{bank_name}__{descriptor}'] = matrix

        comparisons: dict[str, dict[str, float]] = {}
        convergence: dict[str, dict[str, float]] = {}
        for bank_name in ('epoch031', 'epoch135', 'product'):
            for steps in STEP_COUNTS:
                comparisons[f'{steps}:{bank_name}'] = distance_comparison(
                    all_matrices[f'p_euclidean_flow_{steps}'][bank_name],
                    all_matrices[f'fisher_flow_{steps}'][bank_name],
                )
            for geometry in GEOMETRIES:
                convergence[f'{geometry}:{bank_name}'] = distance_comparison(
                    all_matrices[f'{geometry}_flow_30'][bank_name],
                    all_matrices[f'{geometry}_flow_60'][bank_name],
                )

        output['datasets'][dataset_name] = {
            'n_candidates': len(dataset['candidates']),
            'metrics': metrics,
            'geometry_comparison': comparisons,
            'convergence': convergence,
        }

    output['sanity'] = summarize_sanity(diagnostic_rows)
    external = output['datasets']['external']
    convergence_ok = all(
        external['convergence'][f'{geometry}:product']['distance_rho']
        >= 0.999
        for geometry in GEOMETRIES
    )
    authoritative_steps = 30 if convergence_ok else 60
    euclidean_metrics = external['metrics'][
        f'p_euclidean_flow_{authoritative_steps}:product'
    ]
    fisher_metrics = external['metrics'][
        f'fisher_flow_{authoritative_steps}:product'
    ]
    distinctness = external['geometry_comparison'][
        f'{authoritative_steps}:product'
    ]
    distinct = (
        distinctness['distance_rho'] < 0.995
        or distinctness['top1_agreement'] < 0.90
    )
    rho_improvement = (
        fisher_metrics['rho_scratch']
        - euclidean_metrics['rho_scratch']
    )
    false_reduction = (
        fisher_metrics['false20']
        <= 0.8 * euclidean_metrics['false20']
        and fisher_metrics['rho_scratch']
        >= euclidean_metrics['rho_scratch'] - 0.005
    )
    supported = distinct and (
        rho_improvement >= 0.01 or false_reduction
    )
    output['decision'] = {
        'authoritative_steps': authoritative_steps,
        'convergence_ok': convergence_ok,
        'distinct': distinct,
        'rho_improvement': rho_improvement,
        'false_reduction_condition': false_reduction,
        'fisher_advantage_supported': supported,
    }

    (RESULTS_DIR / 'results.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    np.savez_compressed(RESULTS_DIR / 'descriptors.npz', **archive)
    write_csv(RESULTS_DIR / 'metrics.csv', metric_rows)
    write_csv(RESULTS_DIR / 'diagnostics.csv', diagnostic_rows)
    print(json.dumps(output['decision'], indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
