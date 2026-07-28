from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import run_alignment as base


HERE = Path(__file__).resolve().parent
STEPS = 3


def load_primary_descriptors(
    records: list[base.CandidateRecord],
) -> dict[str, np.ndarray]:
    path = HERE / 'results' / 'artifacts.npz'
    if not path.is_file():
        raise FileNotFoundError(
            'The preregistered one-step artifacts are required: ' + str(path)
        )
    with np.load(path) as archive:
        keys = archive['keys'].astype(str)
        sources = archive['sources'].astype(str)
        lookup = {
            (source, key): index
            for index, (source, key) in enumerate(zip(sources, keys, strict=True))
        }
        indices = [lookup[(record.source, record.key)] for record in records]
        return {
            method: np.asarray(archive[method])[indices]
            for method in (
                'one_point',
                'euclidean_two_point',
                'fisher_two_point',
            )
        }


def parameter_displacement(
    initial: dict[str, torch.Tensor],
    policy,
) -> float:
    squared = 0.0
    for name, parameter in policy.named_parameters():
        difference = parameter.detach().double() - initial[name].double()
        squared += float(difference.square().sum().item())
    return math.sqrt(squared)


def gradient_norm(policy) -> float:
    squared = 0.0
    for parameter in policy.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().double().square().sum().item())
    return math.sqrt(squared)


def write_results(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    records: list[base.CandidateRecord],
    descriptors: dict[str, np.ndarray],
    responses: np.ndarray,
    metric_rows: list[dict],
    bootstrap: dict,
    histories: dict[str, list[dict[str, float]]],
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'candidate_records.csv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    with (output_dir / 'metrics.csv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    np.savez_compressed(
        output_dir / 'artifacts.npz',
        responses=responses,
        keys=np.array([record.key for record in records]),
        sources=np.array([record.source for record in records]),
        **descriptors,
    )
    metadata = {
        'smoke': args.smoke,
        'num_steps': args.num_steps,
        'num_train_instances_per_step': args.num_train_instances,
        'num_heldout_instances': args.num_heldout_instances,
        'num_starts': args.num_starts,
        'train_seeds': [
            base.PRIMARY['train_seed'] + step for step in range(args.num_steps)
        ],
        'heldout_seed': base.PRIMARY['heldout_seed'],
        'learning_rate': base.PRIMARY['learning_rate'],
        'weight_decay': base.PRIMARY['weight_decay'],
        'elapsed_seconds': elapsed_seconds,
        'bootstrap': bootstrap,
        'histories': histories,
        'failures': [asdict(record) for record in records if record.error],
    }
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )

    lookup = {(row['group'], row['method']): row for row in metric_rows}
    lines = [
        '# Three-step on-policy actual-update alignment',
        '',
        '- Smoke run: `{}`'.format(args.smoke),
        '- Candidates: {}'.format(len(records)),
        '- On-policy optimizer steps: {}'.format(args.num_steps),
        '- Train / held-out instances: {} / {}'.format(
            args.num_train_instances, args.num_heldout_instances
        ),
        '- POMO starts: {}'.format(args.num_starts),
        '- Elapsed seconds: {:.1f}'.format(elapsed_seconds),
        '',
        '| Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for group in ('matched', 'external', 'pooled'):
        for method in ('one_point', 'euclidean_two_point', 'fisher_two_point'):
            row = lookup[(group, method)]
            lines.append(
                '| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
                    group,
                    method,
                    row['spearman_rho'],
                    row['normalized_nn_error'],
                    row['top1_agreement'],
                    row['top5_overlap'],
                )
            )
    lines.extend([
        '',
        '## Fisher minus Euclidean nearest-neighbor error',
        '',
        '| Group | Mean | 95% low | 95% high |',
        '|---|---:|---:|---:|',
    ])
    for group in ('matched', 'external', 'pooled'):
        item = bootstrap[group]
        lines.append(
            '| {} | {:.4g} | {:.4g} | {:.4g} |'.format(
                group, item['mean'], item['ci95_low'], item['ci95_high']
            )
        )
    fisher_wins = all(
        lookup[(group, 'fisher_two_point')]['spearman_rho']
        > lookup[(group, 'euclidean_two_point')]['spearman_rho']
        and lookup[(group, 'fisher_two_point')]['normalized_nn_error']
        < lookup[(group, 'euclidean_two_point')]['normalized_nn_error']
        for group in ('matched', 'external')
    ) and bootstrap['pooled']['ci95_high'] < 0.0
    lines.extend([
        '',
        '## Preregistered extension decision',
        '',
        '- Fisher promotion rule met: **{}**'.format(fisher_wins),
        '',
    ])
    (output_dir / 'summary.md').write_text(
        '\n'.join(lines), encoding='utf-8'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.candidate_count = 2
        args.num_steps = 2
        args.num_train_instances = 1
        args.num_heldout_instances = 1
        args.num_starts = 8
    else:
        args.candidate_count = base.PRIMARY['candidate_count']
        args.num_steps = STEPS
        args.num_train_instances = base.PRIMARY['num_train_instances']
        args.num_heldout_instances = base.PRIMARY['num_heldout_instances']
        args.num_starts = base.PRIMARY['num_starts']
    if args.output_dir is None:
        args.output_dir = HERE / (
            'smoke_results_3step' if args.smoke else 'results_3step'
        )
    return args


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    env_cls, _ = base.import_policy_runtime()
    template = base.load_policy()
    candidates, records = base.select_candidates(args.candidate_count)
    if args.smoke:
        descriptor_env, descriptor_td = base.make_env_bank(
            env_cls,
            seed=base.PRIMARY['train_seed'],
            num_instances=args.num_train_instances,
        )
        descriptor_probe = base.make_training_graph(
            template,
            descriptor_env,
            descriptor_td,
            seed=base.PRIMARY['train_seed'],
            num_instances=args.num_train_instances,
            num_starts=args.num_starts,
        )
        descriptors = base.compute_descriptors(candidates, descriptor_probe)
    else:
        descriptors = load_primary_descriptors(records)
    print('loaded {} candidates and frozen descriptors'.format(len(candidates)), flush=True)

    heldout_env, heldout_td = base.make_env_bank(
        env_cls,
        seed=base.PRIMARY['heldout_seed'],
        num_instances=args.num_heldout_instances,
    )
    heldout_actions, base_heldout_log_likelihood = base.make_heldout_bank(
        template,
        heldout_env,
        heldout_td,
        seed=base.PRIMARY['heldout_seed'],
        num_instances=args.num_heldout_instances,
        num_starts=args.num_starts,
    )
    train_banks = [
        base.make_env_bank(
            env_cls,
            seed=base.PRIMARY['train_seed'] + step,
            num_instances=args.num_train_instances,
        )
        for step in range(args.num_steps)
    ]

    responses: list[np.ndarray] = []
    valid_indices: list[int] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for index, (candidate, record) in enumerate(zip(candidates, records, strict=True)):
        candidate_start = time.perf_counter()
        history: list[dict[str, float]] = []
        try:
            policy = copy.deepcopy(template)
            initial = {
                name: parameter.detach().clone()
                for name, parameter in policy.named_parameters()
            }
            optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=base.PRIMARY['learning_rate'],
                weight_decay=base.PRIMARY['weight_decay'],
                eps=base.PRIMARY['adam_eps'],
            )
            for step, (train_env, train_td) in enumerate(train_banks):
                probe = base.make_training_graph(
                    policy,
                    train_env,
                    train_td,
                    seed=base.PRIMARY['train_seed'] + step,
                    num_instances=args.num_train_instances,
                    num_starts=args.num_starts,
                )
                loss = base.candidate_loss(candidate, probe)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                current_gradient_norm = gradient_norm(policy)
                optimizer.step()
                history.append({
                    'step': float(step + 1),
                    'loss': float(loss.detach().item()),
                    'gradient_norm': current_gradient_norm,
                })
            current_parameters = {
                name: parameter.detach()
                for name, parameter in policy.named_parameters()
            }
            current_buffers = {
                name: buffer.detach()
                for name, buffer in policy.named_buffers()
            }
            response, response_norm = base.heldout_response(
                policy,
                current_parameters,
                current_buffers,
                heldout_env,
                heldout_td,
                heldout_actions,
                base_heldout_log_likelihood,
                num_instances=args.num_heldout_instances,
                num_starts=args.num_starts,
            )
            record.loss = history[-1]['loss']
            record.gradient_norm = history[-1]['gradient_norm']
            record.parameter_step_norm = parameter_displacement(initial, policy)
            record.response_norm = response_norm
            responses.append(response)
            valid_indices.append(index)
            histories['{}::{}'.format(record.source, record.key)] = history
        except Exception as exc:  # noqa: BLE001
            record.error = '{}: {}'.format(type(exc).__name__, exc)
            print('FAILED {} {}: {}'.format(record.source, record.key, record.error), flush=True)
        record.elapsed_seconds = time.perf_counter() - candidate_start
        print(
            '[{}/{}] {} {} {:.1f}s'.format(
                index + 1,
                len(candidates),
                record.source,
                record.key,
                record.elapsed_seconds,
            ),
            flush=True,
        )

    if len(valid_indices) < 4:
        raise RuntimeError('too few valid three-step responses')
    if len(valid_indices) != len(candidates):
        metric_records = [records[index] for index in valid_indices]
        descriptors = {
            method: matrix[valid_indices]
            for method, matrix in descriptors.items()
        }
    else:
        metric_records = records
    response_matrix = np.stack(responses)
    metric_rows, bootstrap = base.evaluate_metrics(
        descriptors,
        response_matrix,
        metric_records,
    )
    elapsed = time.perf_counter() - start
    write_results(
        args.output_dir,
        args=args,
        records=metric_records,
        descriptors=descriptors,
        responses=response_matrix,
        metric_rows=metric_rows,
        bootstrap=bootstrap,
        histories=histories,
        elapsed_seconds=elapsed,
    )
    print('wrote {}'.format(args.output_dir / 'summary.md'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
