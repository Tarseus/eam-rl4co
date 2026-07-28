from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'rfps_feature_pilot'
sys.path.insert(0, str(PILOT))

import run_pilot as pilot  # noqa: E402
from analyze_existing import descriptor_metrics  # noqa: E402


SOURCE_RUNS = {
    'matched_scratch_seed1': PILOT / 'results_dual_scratch_seed1_curvature',
    'matched_scratch_seed2': PILOT / 'results_dual_scratch_seed2_curvature',
    'matched_warm_seed1': PILOT / 'results_dual_ckpt135_seed1_curvature',
    'matched_warm_seed2': PILOT / 'results_dual_ckpt135_seed2_curvature',
    'external_scratch_seed1': PILOT / 'results_external65_scratch_seed1_curvature',
    'external_scratch_seed2': PILOT / 'results_external65_scratch_seed2_curvature',
}


def unit(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= 1e-14:
        return torch.zeros_like(vector)
    return vector / norm


def sphere_parallel_transport(
    source: torch.Tensor,
    target: torch.Tensor,
    tangent: torch.Tensor,
) -> torch.Tensor:
    denominator = 1.0 + torch.dot(source, target)
    if float(abs(denominator).item()) <= 1e-12:
        raise ValueError('antipodal points do not define stable parallel transport')
    return tangent - torch.dot(tangent, target) / denominator * (source + target)


def fisher_step(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    q: torch.Tensor,
    arc_length: float,
    min_ess_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    u, speed = pilot.importance_field(candidate, probe, q)
    if speed <= 1e-14:
        return q.clone(), u, speed
    next_q, _ = pilot.bounded_importance_step(
        q,
        u,
        arc_length / speed,
        min_ess_ratio,
    )
    return next_q, u, speed


def normalized_position_curvature(states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    directions = [state / float(index + 1) for index, state in enumerate(states)]
    early = directions[1] - directions[0]
    full = torch.cat([early, directions[2] - directions[1]]) / math.sqrt(2.0)
    return early, full


def descriptors_for_probe(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    *,
    ell_max: float,
    min_ess_ratio: float,
) -> dict[str, np.ndarray]:
    sample_count = probe.p0.numel()
    q0 = torch.full(
        (sample_count,),
        1.0 / sample_count,
        dtype=pilot.DTYPE,
    )
    h0 = torch.sqrt(q0)
    u0, speed0 = pilot.importance_field(candidate, probe, q0)
    v0 = u0 / (2.0 * h0)
    first_arc = ell_max / 3.0

    q1, _, _ = fisher_step(
        candidate,
        probe,
        q0,
        first_arc,
        min_ess_ratio,
    )
    u1, _ = pilot.importance_field(candidate, probe, q1)
    h1 = torch.sqrt(q1)
    v1 = u1 / (2.0 * h1)
    transported_v1 = sphere_parallel_transport(h1, h0, v1)

    fisher_states: list[torch.Tensor] = []
    q = q0.clone()
    for _ in range(3):
        q, _, _ = fisher_step(
            candidate,
            probe,
            q,
            first_arc,
            min_ess_ratio,
        )
        fisher_states.append(pilot.fisher_log_map(q0, q))
    fisher_early, fisher_full = normalized_position_curvature(fisher_states)

    z0 = torch.log(q0)
    z = z0.clone()
    euclidean_states: list[torch.Tensor] = []
    euclidean_fields: list[torch.Tensor] = []
    for _ in range(3):
        q_euclidean = torch.softmax(z, dim=0)
        u_euclidean, _ = pilot.importance_field(candidate, probe, q_euclidean)
        euclidean_fields.append(u_euclidean)
        z = z + first_arc * unit(u_euclidean)
        euclidean_states.append(z - z0)
    euclidean_early, euclidean_full = normalized_position_curvature(euclidean_states)
    q_euclidean_1 = torch.softmax(z0 + first_arc * unit(u0), dim=0)
    u_euclidean_1, _ = pilot.importance_field(candidate, probe, q_euclidean_1)

    initial_white = u0 / torch.sqrt(q0)
    fisher_white_1 = u1 / torch.sqrt(q1)
    euclidean_white_1 = u_euclidean_1 / torch.sqrt(q_euclidean_1)
    return {
        'initial_field': initial_white.numpy(),
        'fisher_field_delta_pt': (2.0 * (transported_v1 - v0)).numpy(),
        'fisher_field_delta_ambient': (fisher_white_1 - initial_white).numpy(),
        'fisher_direction_delta': (unit(v1) - unit(v0)).numpy(),
        'fisher_direction_delta_pt': (unit(transported_v1) - unit(v0)).numpy(),
        'fisher_two_fields': (torch.cat([2.0 * v0, 2.0 * transported_v1]) / math.sqrt(2.0)).numpy(),
        'fisher_coarse_kappa_early': fisher_early.numpy(),
        'fisher_coarse_kappa_full': fisher_full.numpy(),
        'euclidean_field_delta': (u_euclidean_1 - u0).numpy(),
        'euclidean_field_delta_white': (euclidean_white_1 - initial_white).numpy(),
        'euclidean_direction_delta': (unit(u_euclidean_1) - unit(u0)).numpy(),
        'euclidean_two_fields': (torch.cat([u0, u_euclidean_1]) / math.sqrt(2.0)).numpy(),
        'euclidean_kappa_early': euclidean_early.numpy(),
        'euclidean_kappa_full': euclidean_full.numpy(),
    }


def load_ordered_inputs(source_path: Path) -> tuple[list[pilot.Candidate], list[pilot.Probe], dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((source_path / 'results.json').read_text(encoding='utf-8'))
    with np.load(source_path / 'descriptors.npz') as archive:
        source = {name: archive[name].copy() for name in archive.files}
    candidates, failures = pilot.load_candidates(
        Path(metadata['data_source']),
        metadata['score_target'],
    )
    if failures:
        raise RuntimeError(f'unexpected candidate loading failures: {failures[:3]}')
    by_key = {candidate.key: candidate for candidate in candidates}
    ordered = [by_key[str(key)] for key in source['keys']]
    probes = pilot.load_probes_npz(Path(metadata['probe_source']))
    return ordered, probes, metadata, source


def run_one(name: str, source_path: Path) -> dict[str, Any]:
    candidates, probes, metadata, source = load_ordered_inputs(source_path)
    raw: dict[str, list[np.ndarray]] = {}
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate_blocks: dict[str, list[np.ndarray]] = {}
        for probe in probes:
            values = descriptors_for_probe(
                candidate,
                probe,
                ell_max=0.10,
                min_ess_ratio=0.4,
            )
            for descriptor_name, vector in values.items():
                candidate_blocks.setdefault(descriptor_name, []).append(vector)
        for descriptor_name, vectors in candidate_blocks.items():
            raw.setdefault(descriptor_name, []).append(
                np.concatenate(vectors) / math.sqrt(len(vectors))
            )
        print(
            f'[{name}] {candidate_index:03d}/{len(candidates):03d}',
            flush=True,
        )

    matrices = {
        descriptor_name: np.stack(vectors)
        for descriptor_name, vectors in raw.items()
    }
    matrices['reference_kappa_full'] = source['flow_curvature']
    metrics = {
        descriptor_name: descriptor_metrics(matrix, source['scores'])
        for descriptor_name, matrix in matrices.items()
    }
    output_dir = HERE / 'results_two_point' / name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / 'descriptors.npz',
        scores=source['scores'],
        keys=source['keys'],
        **matrices,
    )
    payload = {
        'source_results': str(source_path),
        'n_candidates': len(candidates),
        'n_probes': len(probes),
        'ell_max': 0.10,
        'metrics': metrics,
        'cost_model': {
            'field_delta': '2 field evaluations per probe',
            'coarse_curvature': '3 field evaluations per probe',
            'reference_curvature': '30 integration field evaluations plus checkpoint evaluations per probe',
        },
    }
    (output_dir / 'results.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return payload


def main() -> int:
    torch.set_default_dtype(pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    combined: dict[str, Any] = {}
    for name, source_path in SOURCE_RUNS.items():
        combined[name] = run_one(name, source_path)
    output_dir = HERE / 'results_two_point'
    (output_dir / 'results.json').write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    selected = [
        'reference_kappa_full',
        'fisher_field_delta_pt',
        'fisher_coarse_kappa_early',
        'euclidean_field_delta',
        'euclidean_kappa_early',
    ]
    lines = [
        '# Two-point and Euclidean baseline results',
        '',
        '| Run | Descriptor | rho | NN median | False skip |',
        '|---|---|---:|---:|---:|',
    ]
    for run_name, payload in combined.items():
        for descriptor_name in selected:
            item = payload['metrics'][descriptor_name]
            rho = item['rho']
            nn = item['nn_median']
            false_skip = item['false_skip_20pct_tol001']
            lines.append(
                f'| {run_name} | {descriptor_name} | {rho:.3f} | {nn:.5f} | {false_skip:.3f} |'
            )
    (output_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
