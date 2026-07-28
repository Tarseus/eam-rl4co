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
from analyze_first_order_controls import order_resampled_risk  # noqa: E402
from run_two_point_baselines import sphere_parallel_transport, unit  # noqa: E402


STEPS = (0.01, 0.03, 0.10)


CONFIGS = {
    'matched_warmprobe_seed1': {
        'candidate_source': PILOT / 'results_dual_scratch_seed1_curvature',
        'probe_source': PILOT / 'results_dual_ckpt135_seed1_curvature',
        'targets': {
            'scratch': PILOT / 'results_dual_scratch_seed1_curvature',
            'warm': PILOT / 'results_dual_ckpt135_seed1_curvature',
        },
    },
    'matched_warmprobe_seed2': {
        'candidate_source': PILOT / 'results_dual_scratch_seed2_curvature',
        'probe_source': PILOT / 'results_dual_ckpt135_seed2_curvature',
        'targets': {
            'scratch': PILOT / 'results_dual_scratch_seed2_curvature',
            'warm': PILOT / 'results_dual_ckpt135_seed2_curvature',
        },
    },
    'external_warmprobe_seed1': {
        'candidate_source': PILOT / 'results_external65_scratch_seed1_curvature',
        'probe_source': PILOT / 'results_epoch135_isflow_seed1',
        'targets': {
            'scratch': PILOT / 'results_epoch135_isflow_seed1',
        },
    },
    'external_warmprobe_seed2': {
        'candidate_source': PILOT / 'results_external65_scratch_seed2_curvature',
        'probe_source': PILOT / 'results_epoch135_isflow_seed2',
        'targets': {
            'scratch': PILOT / 'results_epoch135_isflow_seed2',
        },
    },
}


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / 'descriptors.npz') as archive:
        return {name: archive[name].copy() for name in archive.files}


def load_config(config: dict[str, Any]) -> tuple[list[pilot.Candidate], list[pilot.Probe], dict[str, dict[str, np.ndarray]]]:
    candidate_metadata = json.loads(
        (config['candidate_source'] / 'results.json').read_text(encoding='utf-8')
    )
    candidates, failures = pilot.load_candidates(
        Path(candidate_metadata['data_source']),
        candidate_metadata['score_target'],
    )
    if failures:
        raise RuntimeError(f'candidate failures: {failures[:3]}')
    candidate_data = read_npz(config['candidate_source'])
    by_key = {candidate.key: candidate for candidate in candidates}
    ordered = [by_key[str(key)] for key in candidate_data['keys']]

    probe_metadata = json.loads(
        (config['probe_source'] / 'results.json').read_text(encoding='utf-8')
    )
    probes = pilot.load_probes_npz(Path(probe_metadata['probe_source']))
    targets = {
        name: read_npz(path)
        for name, path in config['targets'].items()
    }
    return ordered, probes, targets


def coefficient_field(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    sample_count = q.numel()
    log_density_ratio = torch.log(q.clamp_min(pilot.T_FLOOR)) + math.log(sample_count)
    p = (probe.p0 + log_density_ratio.unsqueeze(0)).detach().requires_grad_(True)
    batch, model_output = pilot.pairwise_inputs_importance(probe, p, q)
    loss = candidate.compiled.loss_fn(
        batch,
        model_output,
        {'alpha': candidate.alpha},
    )
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, dtype=pilot.DTYPE)
    if loss.numel() != 1:
        loss = loss.mean()
    gradient = torch.autograd.grad(loss, p, allow_unused=True)[0]
    c = torch.zeros_like(q) if gradient is None else -gradient.reshape(-1)
    u = c - q * c.sum()
    speed = float(torch.sqrt((u.square() / q).sum().clamp_min(0.0)).item())
    return c.detach(), u.detach(), speed


def direction_pairs_for_probe(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
) -> dict[str, np.ndarray]:
    sample_count = probe.p0.numel()
    q0 = torch.full(
        (sample_count,),
        1.0 / sample_count,
        dtype=pilot.DTYPE,
    )
    h0 = torch.sqrt(q0)
    c0, u0, speed0 = coefficient_field(candidate, probe, q0)
    v0 = u0 / (2.0 * h0)
    d0 = unit(v0)
    z0 = torch.log(q0)
    descriptors: dict[str, np.ndarray] = {}

    for step in STEPS:
        z1 = z0 + step * unit(u0)
        q_euclidean = torch.softmax(z1, dim=0)
        c_euclidean, u_euclidean, _ = coefficient_field(candidate, probe, q_euclidean)
        euclidean_pair = torch.cat([d0, unit(u_euclidean)]) / math.sqrt(2.0)
        descriptors[f'euclidean_pair_{step:.2f}'] = euclidean_pair.numpy()
        mean_centered = c_euclidean - c_euclidean.mean()
        mean_pair = torch.cat([d0, unit(mean_centered)]) / math.sqrt(2.0)
        descriptors[f'mean_centered_pair_{step:.2f}'] = mean_pair.numpy()

        raw_z1 = z0 + step * unit(c0)
        q_raw = torch.softmax(raw_z1, dim=0)
        c_raw, _, _ = coefficient_field(candidate, probe, q_raw)
        raw_pair = torch.cat([unit(c0), unit(c_raw)]) / math.sqrt(2.0)
        descriptors[f'raw_coefficient_pair_{step:.2f}'] = raw_pair.numpy()

        if speed0 <= 1e-14:
            q_fisher = q0.clone()
        else:
            q_fisher, _ = pilot.bounded_importance_step(
                q0,
                u0,
                step / speed0,
                0.4,
            )
        _, u_fisher, _ = coefficient_field(candidate, probe, q_fisher)
        h_fisher = torch.sqrt(q_fisher)
        v_fisher = u_fisher / (2.0 * h_fisher)
        transported = sphere_parallel_transport(h_fisher, h0, v_fisher)
        fisher_pair = torch.cat([d0, unit(transported)]) / math.sqrt(2.0)
        descriptors[f'fisher_pair_{step:.2f}'] = fisher_pair.numpy()
    return descriptors


def run_config(
    config_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    candidates, probes, targets = load_config(config)
    raw: dict[str, list[np.ndarray]] = {}
    for candidate_index, candidate in enumerate(candidates, start=1):
        per_candidate: dict[str, list[np.ndarray]] = {}
        for probe in probes:
            for name, vector in direction_pairs_for_probe(candidate, probe).items():
                per_candidate.setdefault(name, []).append(vector)
        for name, vectors in per_candidate.items():
            raw.setdefault(name, []).append(
                np.concatenate(vectors) / math.sqrt(len(vectors))
            )
        print(
            f'[{config_name}] {candidate_index:03d}/{len(candidates):03d}',
            flush=True,
        )
    matrices = {name: np.stack(vectors) for name, vectors in raw.items()}
    target_results: dict[str, Any] = {}
    for target_name, target in targets.items():
        target_index = {str(key): idx for idx, key in enumerate(target['keys'])}
        candidate_keys = read_npz(config['candidate_source'])['keys']
        scores = np.asarray(
            [target['scores'][target_index[str(key)]] for key in candidate_keys]
        )
        target_results[target_name] = {
            name: {
                'metrics': descriptor_metrics(matrix, scores),
                'order_risk': order_resampled_risk(matrix, scores, n_trials=500),
            }
            for name, matrix in matrices.items()
        }
    return {
        'n_candidates': len(candidates),
        'n_probes': len(probes),
        'targets': target_results,
    }


def main() -> int:
    torch.set_default_dtype(pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    results: dict[str, Any] = {}
    for config_name, config in CONFIGS.items():
        results[config_name] = run_config(config_name, config)

    output = HERE / 'results_step_sweep'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    lines = [
        '# Two-point direction-pair step sweep',
        '',
        '| Config | Target | Geometry | Step | rho | NN | False skip mean |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for config_name, config_result in results.items():
        for target_name, descriptors in config_result['targets'].items():
            for descriptor_name, value in descriptors.items():
                geometry, _, step_text = descriptor_name.partition('_pair_')
                metrics = value['metrics']
                risk = value['order_risk']
                rho = metrics['rho']
                nn = metrics['nn_median']
                false_skip = risk['false_skip_mean']
                lines.append(
                    f'| {config_name} | {target_name} | {geometry} | {step_text} | '
                    f'{rho:.3f} | {nn:.5f} | {false_skip:.3f} |'
                )
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
