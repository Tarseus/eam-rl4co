from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
MINIMALITY = HERE.parent / 'rfps_minimality'
PILOT = HERE.parent / 'rfps_feature_pilot'
sys.path.insert(0, str(MINIMALITY))
sys.path.insert(0, str(PILOT))

import run_pilot as pilot  # noqa: E402
import run_step_sweep as sweep  # noqa: E402
from analyze_existing import descriptor_metrics  # noqa: E402
from analyze_first_order_controls import order_resampled_risk  # noqa: E402
from run_two_point_baselines import unit  # noqa: E402


ETA_BASE = 0.03
ETA_PILOT = 0.01
ETA_MIN = 0.005
ETA_MAX = 0.10


@dataclass
class LocalRecord:
    candidate: pilot.Candidate
    probe: pilot.Probe
    q0: torch.Tensor
    z0: torch.Tensor
    d0: torch.Tensor
    pilot_direction: torch.Tensor
    turning_rate: float


def q_along(record: LocalRecord, eta: float) -> torch.Tensor:
    return torch.softmax(record.z0 + eta * record.d0, dim=0)


def forward_kl(record: LocalRecord, eta: float) -> float:
    q = q_along(record, eta)
    return float((record.q0 * (record.z0 - torch.log(q))).sum().item())


def reverse_kl(record: LocalRecord, eta: float) -> float:
    q = q_along(record, eta)
    return float((q * (torch.log(q) - record.z0)).sum().item())


def ess_loss(record: LocalRecord, eta: float) -> float:
    q = q_along(record, eta)
    sample_count = q.numel()
    ess_ratio = 1.0 / float((sample_count * q.square().sum()).item())
    return 1.0 - ess_ratio


def max_log_ratio(record: LocalRecord, eta: float) -> float:
    q = q_along(record, eta)
    return float(torch.max(torch.abs(torch.log(q) - record.z0)).item())


def solve_monotone(
    metric: Callable[[LocalRecord, float], float],
    record: LocalRecord,
    target: float,
    *,
    initial_hi: float = ETA_MAX,
) -> float:
    if target <= 0.0:
        return 0.0
    lo = 0.0
    hi = initial_hi
    while metric(record, hi) < target and hi < 2.0:
        hi *= 2.0
    if metric(record, hi) < target:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if metric(record, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def direction_at(record: LocalRecord, eta: float) -> torch.Tensor:
    q = q_along(record, eta)
    _, u, _ = sweep.coefficient_field(record.candidate, record.probe, q)
    return unit(u)


def make_record(candidate: pilot.Candidate, probe: pilot.Probe) -> LocalRecord:
    sample_count = probe.p0.numel()
    q0 = torch.full(
        (sample_count,),
        1.0 / sample_count,
        dtype=pilot.DTYPE,
    )
    z0 = torch.log(q0)
    _, u0, _ = sweep.coefficient_field(candidate, probe, q0)
    d0 = unit(u0)
    provisional = LocalRecord(
        candidate=candidate,
        probe=probe,
        q0=q0,
        z0=z0,
        d0=d0,
        pilot_direction=d0,
        turning_rate=0.0,
    )
    pilot_direction = direction_at(provisional, ETA_PILOT)
    cosine = float(torch.dot(d0, pilot_direction).clamp(-1.0, 1.0).item())
    angle = math.acos(cosine)
    provisional.pilot_direction = pilot_direction
    provisional.turning_rate = angle / ETA_PILOT
    return provisional


def aggregate_descriptor(
    records: list[LocalRecord],
    etas: list[float],
) -> np.ndarray:
    blocks = []
    for record, eta in zip(records, etas, strict=True):
        d1 = direction_at(record, eta)
        blocks.append(torch.cat([record.d0, d1]) / math.sqrt(2.0))
    return (torch.cat(blocks) / math.sqrt(len(blocks))).numpy()


def step_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    return {
        'mean': mean,
        'median': float(np.median(array)),
        'std': float(array.std()),
        'cv': float(array.std() / mean) if mean > 0 else 0.0,
        'min': float(array.min()),
        'max': float(array.max()),
        'q10': float(np.quantile(array, 0.10)),
        'q90': float(np.quantile(array, 0.90)),
    }


def run_config(config_name: str, config: dict) -> dict:
    candidates, probes, targets = sweep.load_config(config)
    records_by_candidate: list[list[LocalRecord]] = []
    all_records: list[LocalRecord] = []
    for index, candidate in enumerate(candidates, start=1):
        records = [make_record(candidate, probe) for probe in probes]
        records_by_candidate.append(records)
        all_records.extend(records)
        print(f'[{config_name}] prepared {index:03d}/{len(candidates):03d}', flush=True)

    sample_count = all_records[0].q0.numel()
    delta_kl = ETA_BASE**2 / (2.0 * sample_count)
    target_ess_loss = 1.0 - 1.0 / (1.0 + ETA_BASE**2 / sample_count)
    target_max_log = ETA_BASE * math.sqrt(2.0 * math.log(sample_count) / sample_count)
    positive_rates = [record.turning_rate for record in all_records if record.turning_rate > 1e-12]
    median_rate = float(np.median(positive_rates)) if positive_rates else 0.0
    target_angle = ETA_BASE * median_rate

    step_rules: dict[str, Callable[[LocalRecord], float]] = {
        'fixed_003': lambda record: ETA_BASE,
        'fixed_forward_kl': lambda record: solve_monotone(forward_kl, record, delta_kl),
        'fixed_reverse_kl': lambda record: solve_monotone(reverse_kl, record, delta_kl),
        'fixed_ess': lambda record: solve_monotone(ess_loss, record, target_ess_loss),
        'max_log_ratio': lambda record: solve_monotone(max_log_ratio, record, target_max_log),
        'turning_radius': lambda record: float(
            np.clip(
                target_angle / max(record.turning_rate, 1e-12),
                ETA_MIN,
                ETA_MAX,
            )
        ),
    }

    matrices: dict[str, np.ndarray] = {}
    steps: dict[str, list[float]] = {name: [] for name in step_rules}
    for rule_name, rule in step_rules.items():
        candidate_vectors = []
        for index, records in enumerate(records_by_candidate, start=1):
            etas = [rule(record) for record in records]
            steps[rule_name].extend(etas)
            candidate_vectors.append(aggregate_descriptor(records, etas))
            print(
                f'[{config_name}] {rule_name} {index:03d}/{len(candidates):03d}',
                flush=True,
            )
        matrices[rule_name] = np.stack(candidate_vectors)

    target_results = {}
    for target_name, target in targets.items():
        target_index = {str(key): idx for idx, key in enumerate(target['keys'])}
        scores = np.asarray(
            [target['scores'][target_index[candidate.key]] for candidate in candidates]
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
        'step_stats': {name: step_summary(values) for name, values in steps.items()},
        'calibration': {
            'sample_count': sample_count,
            'delta_kl': delta_kl,
            'target_ess_loss': target_ess_loss,
            'target_max_log_ratio': target_max_log,
            'median_turning_rate': median_rate,
            'target_angle': target_angle,
        },
    }


def write_summary(results: dict) -> None:
    lines = [
        '# Adaptive geometric step results',
        '',
        '| Config | Target | Rule | rho | NN | False skip | eta median | eta CV | eta q10--q90 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for config_name, config in results.items():
        for target_name, descriptors in config['targets'].items():
            for rule_name, value in descriptors.items():
                metric = value['metrics']
                risk = value['order_risk']
                stats = config['step_stats'][rule_name]
                rho_value = metric['rho']
                nn_value = metric['nn_median']
                false_skip_value = risk['false_skip_mean']
                median_value = stats['median']
                cv_value = stats['cv']
                q10_value = stats['q10']
                q90_value = stats['q90']
                lines.append(
                    f'| {config_name} | {target_name} | {rule_name} | '
                    f'{rho_value:.3f} | {nn_value:.5f} | '
                    f'{false_skip_value:.3f} | {median_value:.5f} | '
                    f'{cv_value:.3f} | {q10_value:.5f}--{q90_value:.5f} |'
                )
    (HERE / 'results' / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def main() -> int:
    torch.set_default_dtype(pilot.DTYPE)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    results = {
        name: run_config(name, config)
        for name, config in sweep.CONFIGS.items()
    }
    output = HERE / 'results'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    write_summary(results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
