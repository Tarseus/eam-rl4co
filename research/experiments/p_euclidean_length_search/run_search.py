from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / 'p_euclidean_flow' / 'run_experiment.py'
RESULTS_DIR = HERE / 'results'
LENGTHS = (0.06, 0.10, 0.20, 0.40)
COARSE_STEPS = 30
AUDIT_STEPS = 60


def load_base():
    spec = importlib.util.spec_from_file_location('p_euclidean_base', BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {BASE_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrices_for_setting(
    candidates,
    probes,
    *,
    bank_name: str,
    length: float,
    steps: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    base.FLOW_LENGTH = length
    raw: dict[str, list[np.ndarray]] = {
        'p_euclidean': [],
        'fisher': [],
    }
    diagnostics: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        blocks = {'p_euclidean': [], 'fisher': []}
        for probe_index, probe in enumerate(probes):
            count = probe.objective.numel()
            p0 = torch.full(
                (count,),
                1.0 / count,
                dtype=base.pilot.DTYPE,
            )
            initial_covector = base.training_covector(
                candidate,
                probe,
                p0,
                p0,
            )
            initial_fields = base.metric_fields(initial_covector, p0)
            d_e = base.flat_direction_at_base(
                initial_fields['p_euclidean'],
                p0,
            )
            d_f = base.fisher_direction(initial_fields['fisher'], p0)
            if (
                torch.linalg.vector_norm(d_e) > 1e-14
                and torch.linalg.vector_norm(d_f) > 1e-14
            ):
                initial_cosine = float(torch.dot(d_e, d_f).item())
            else:
                initial_cosine = 1.0
            for geometry in ('p_euclidean', 'fisher'):
                descriptor, _, values = base.integrate_flow(
                    candidate,
                    probe,
                    p0,
                    geometry,
                    steps,
                    initial_covector,
                )
                blocks[geometry].append(descriptor)
                diagnostics.append(
                    {
                        'length': length,
                        'steps': steps,
                        'bank': bank_name,
                        'candidate': candidate_index,
                        'probe': probe_index,
                        'geometry': geometry,
                        'initial_cosine': initial_cosine,
                        **values,
                    }
                )
        for geometry, vectors in blocks.items():
            raw[geometry].append(
                np.concatenate(vectors) / math.sqrt(len(vectors))
            )
        print(
            f'[{length:.2f}/{steps}/{bank_name}] '
            f'{candidate_index + 1:03d}/{len(candidates):03d}',
            flush=True,
        )
    return {
        geometry: np.stack(rows)
        for geometry, rows in raw.items()
    }, diagnostics


def product_matrices(bank_matrices: dict[str, dict[str, np.ndarray]]):
    return {
        geometry: np.concatenate(
            [
                bank_matrices['scratch'][geometry] / math.sqrt(2.0),
                bank_matrices['epoch135'][geometry] / math.sqrt(2.0),
            ],
            axis=1,
        )
        for geometry in ('p_euclidean', 'fisher')
    }


def evaluate_setting(
    candidates,
    banks,
    primary_gaps,
    target_gaps,
    *,
    length: float,
    steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    per_bank: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    for bank_name in ('scratch', 'epoch135'):
        matrices, rows = matrices_for_setting(
            candidates,
            banks[bank_name],
            bank_name=bank_name,
            length=length,
            steps=steps,
        )
        per_bank[bank_name] = matrices
        diagnostics.extend(rows)
    matrices = product_matrices(per_bank)
    metrics = {
        geometry: base.sweep.descriptor_metrics(
            matrix,
            primary_gaps,
            target_gaps,
        )
        for geometry, matrix in matrices.items()
    }
    comparison = base.distance_comparison(
        matrices['p_euclidean'],
        matrices['fisher'],
    )
    delta_rho = (
        metrics['fisher']['rho_scratch']
        - metrics['p_euclidean']['rho_scratch']
    )
    false_reduction = (
        metrics['p_euclidean']['false20']
        - metrics['fisher']['false20']
    )
    summary = {
        'length': length,
        'steps': steps,
        'metrics': metrics,
        'delta_rho': delta_rho,
        'false20_reduction': false_reduction,
        'geometry_comparison': comparison,
        'sanity': {
            'minimum_initial_cosine': min(
                row['initial_cosine'] for row in diagnostics
            ),
            'maximum_tangent_sum_error': max(
                row['max_sum_error'] for row in diagnostics
            ),
            'maximum_substep_length_error': max(
                row['max_step_error'] for row in diagnostics
            ),
            'minimum_probability': min(
                row['minimum_probability'] for row in diagnostics
            ),
            'maximum_stationary_fraction': max(
                row['stationary_fraction'] for row in diagnostics
            ),
        },
    }
    return summary, diagnostics, matrices


def metric_row(result: dict[str, Any]) -> dict[str, Any]:
    euclidean = result['metrics']['p_euclidean']
    fisher = result['metrics']['fisher']
    comparison = result['geometry_comparison']
    sanity = result['sanity']
    return {
        'length': result['length'],
        'steps': result['steps'],
        'rho_p_euclidean': euclidean['rho_scratch'],
        'rho_fisher': fisher['rho_scratch'],
        'delta_rho': result['delta_rho'],
        'false20_p_euclidean': euclidean['false20'],
        'false20_fisher': fisher['false20'],
        'false20_reduction': result['false20_reduction'],
        'distance_rho': comparison['distance_rho'],
        'top1_agreement': comparison['top1_agreement'],
        'minimum_probability': sanity['minimum_probability'],
        'maximum_tangent_sum_error': sanity['maximum_tangent_sum_error'],
        'maximum_substep_length_error': sanity[
            'maximum_substep_length_error'
        ],
    }


def main() -> int:
    torch.set_default_dtype(base.pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    banks = base.load_banks()
    dataset = base.sweep.load_datasets()['external']
    primary_gaps, target_gaps = base.sweep.target_geometry(
        'external',
        dataset['targets'],
    )

    coarse_results: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    coarse_product: dict[float, dict[str, np.ndarray]] = {}
    output: dict[str, Any] = {
        'status': 'exploratory',
        'selection_metric': 'rho_fisher_minus_rho_p_euclidean',
        'lengths': LENGTHS,
        'coarse_steps': COARSE_STEPS,
        'audit_steps': AUDIT_STEPS,
    }

    for length in LENGTHS:
        result, diagnostics, matrices = evaluate_setting(
            dataset['candidates'],
            banks,
            primary_gaps,
            target_gaps,
            length=length,
            steps=COARSE_STEPS,
        )
        coarse_results.append(result)
        all_diagnostics.extend(diagnostics)
        coarse_product[length] = matrices
        output['coarse_results'] = coarse_results
        (RESULTS_DIR / 'partial_results.json').write_text(
            json.dumps(output, indent=2),
            encoding='utf-8',
        )
        print(json.dumps(metric_row(result), indent=2), flush=True)

    winner = max(
        coarse_results,
        key=lambda result: (
            result['delta_rho'],
            result['false20_reduction'],
            -result['length'],
        ),
    )
    audit, diagnostics, audit_matrices = evaluate_setting(
        dataset['candidates'],
        banks,
        primary_gaps,
        target_gaps,
        length=winner['length'],
        steps=AUDIT_STEPS,
    )
    all_diagnostics.extend(diagnostics)
    convergence = {
        geometry: base.distance_comparison(
            coarse_product[winner['length']][geometry],
            audit_matrices[geometry],
        )
        for geometry in ('p_euclidean', 'fisher')
    }
    output.update(
        {
            'coarse_results': coarse_results,
            'selected_length': winner['length'],
            'selected_coarse_result': winner,
            'selected_audit_result': audit,
            'convergence': convergence,
        }
    )
    (RESULTS_DIR / 'results.json').write_text(
        json.dumps(output, indent=2),
        encoding='utf-8',
    )
    write_csv(
        RESULTS_DIR / 'metrics.csv',
        [metric_row(result) for result in coarse_results] + [metric_row(audit)],
    )
    write_csv(RESULTS_DIR / 'diagnostics.csv', all_diagnostics)
    print(
        json.dumps(
            {
                'selected_length': winner['length'],
                'coarse_delta_rho': winner['delta_rho'],
                'audit_delta_rho': audit['delta_rho'],
                'convergence': convergence,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
