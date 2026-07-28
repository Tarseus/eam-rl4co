from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_validation as dual


METHODS = (
    'node_euclidean',
    'node_fisher_tangent',
    'pair_euclidean',
    'pair_fisher_tangent',
)


def tangent_blocks(
    candidate: dual.actual.pilot.Candidate,
    anchor: dual.Anchor,
) -> dict[str, np.ndarray]:
    margin = anchor.margin.detach().clone().requires_grad_(True)
    batch, model_output = dual.centered_margin_inputs(anchor, margin)
    loss = dual.scalar_loss(candidate, batch, model_output)
    edge_covector = torch.autograd.grad(loss, margin)[0]
    if not torch.isfinite(edge_covector).all():
        raise RuntimeError('non-finite pair-margin covector')
    node_covector = torch.zeros(anchor.probe.p0.numel(), dtype=dual.DTYPE)
    node_covector.index_add_(0, anchor.winner, edge_covector)
    node_covector.index_add_(0, anchor.loser, -edge_covector)
    node_direction = anchor.helmert.T @ node_covector
    fisher_tangent = torch.linalg.solve(
        anchor.fisher_inverse_sqrt, node_direction
    )
    pair_fisher_tangent = torch.sqrt(anchor.pair_fisher_weight) * edge_covector
    blocks = {
        'node_euclidean': dual.unit(node_direction),
        'node_fisher_tangent': dual.unit(fisher_tangent),
        'pair_euclidean': dual.unit(edge_covector),
        'pair_fisher_tangent': dual.unit(pair_fisher_tangent),
    }
    return {
        method: vector.detach().cpu().numpy()
        for method, vector in blocks.items()
    }


def compute_descriptors(
    candidates: list[dual.actual.pilot.Candidate],
    records: list[dual.actual.CandidateRecord],
    anchors: list[dual.Anchor],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    stores = {method: [] for method in METHODS}
    maxima = {
        'common_gradient_ratio': 0.0,
        'relative_shift_loss_change': 0.0,
        'relative_centered_loss_change': 0.0,
        'relative_chain_rule_error': 0.0,
    }
    for index, (candidate, record) in enumerate(zip(candidates, records, strict=True)):
        per_method = {method: [] for method in METHODS}
        for anchor in anchors:
            gate = dual.shift_invariance_gate(candidate, anchor)
            for name, value in zip(maxima, gate, strict=True):
                maxima[name] = max(maxima[name], value)
            if max(gate) > 1e-8:
                raise RuntimeError(
                    '{} failed margin gate: {}'.format(record.key, gate)
                )
            blocks = tangent_blocks(candidate, anchor)
            for method in METHODS:
                per_method[method].append(blocks[method])
        for method in METHODS:
            stores[method].append(
                np.concatenate(per_method[method]) / math.sqrt(len(anchors))
            )
        print(
            '[{}/{}] {} {}'.format(
                index + 1, len(candidates), record.source, record.key
            ),
            flush=True,
        )
    return {method: np.stack(rows) for method, rows in stores.items()}, maxima


def evaluate_target(
    target_name: str,
    responses: np.ndarray,
    descriptors: dict[str, np.ndarray],
    records: list[dual.actual.CandidateRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    groups = {
        'matched': np.array([record.source == 'matched' for record in records]),
        'external': np.array([record.source == 'external' for record in records]),
        'pooled': np.ones(len(records), dtype=bool),
    }
    for group, mask in groups.items():
        target_distance = dual.distance_matrix(responses[mask])
        errors = {}
        for method in METHODS:
            metrics, nn_error = dual.actual.method_metrics(
                dual.distance_matrix(descriptors[method][mask]), target_distance
            )
            rows.append({
                'target': target_name,
                'group': group,
                'method': method,
                **metrics,
            })
            errors[method] = nn_error
        bootstrap[group] = dual.actual.bootstrap_paired_difference(
            errors['node_fisher_tangent'], errors['node_euclidean']
        )
    return rows, bootstrap


def score_metrics(
    descriptors: dict[str, np.ndarray],
    records: list[dual.actual.CandidateRecord],
) -> list[dict[str, Any]]:
    rows = []
    for group in ('matched', 'external'):
        mask = np.array([record.source == group for record in records])
        scores = np.array([record.score for record in records])[mask]
        score_distance = dual.distance_matrix(scores[:, None])
        upper = np.triu_indices(len(scores), k=1)
        for method in METHODS:
            descriptor_distance = dual.distance_matrix(descriptors[method][mask])
            rho = float(spearmanr(
                descriptor_distance[upper], score_distance[upper]
            ).statistic)
            masked = descriptor_distance.copy()
            np.fill_diagonal(masked, np.inf)
            neighbor = np.argmin(masked, axis=1)
            error = np.abs(scores - scores[neighbor])
            random_mean = float(
                (score_distance.sum(axis=1) / (len(scores) - 1)).mean()
            )
            rows.append({
                'group': group,
                'method': method,
                'score_distance_rho': rho,
                'normalized_nn_score_error': float(error.mean() / random_mean),
                'median_nn_score_error': float(np.median(error)),
            })
    return rows


def write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    records: list[dual.actual.CandidateRecord],
    anchors: list[dual.Anchor],
    descriptors: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    bootstraps: dict[str, Any],
    score_rows: list[dict[str, Any]],
    gate: dict[str, float],
    elapsed: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dual.write_csv(output_dir / 'metrics.csv', rows)
    dual.write_csv(output_dir / 'score_metrics.csv', score_rows)
    np.savez_compressed(
        output_dir / 'descriptors.npz',
        keys=np.array([record.key for record in records]),
        sources=np.array([record.source for record in records]),
        **descriptors,
    )
    metadata = {
        'retrospective_exploratory': True,
        'smoke': args.smoke,
        'candidate_count': len(records),
        'anchor_count': len(anchors),
        'gate': gate,
        'bootstraps': bootstraps,
        'elapsed_seconds': elapsed,
    }
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
    lookup = {
        (row['target'], row['group'], row['method']): row for row in rows
    }
    lines = [
        '# Real pair-margin Fisher tangent-metric extension',
        '',
        '- Retrospective/exploratory: `True`',
        '- Smoke: `{}`'.format(args.smoke),
        '- Candidates: {}'.format(len(records)),
        '- Real-margin anchors: {}'.format(len(anchors)),
        '- Elapsed seconds: {:.1f}'.format(elapsed),
        '',
        '| Target | Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for target in ('one_step', 'three_step'):
        for group in ('matched', 'external', 'pooled'):
            for method in METHODS:
                row = lookup[(target, group, method)]
                lines.append(
                    '| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
                        target,
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
        '## Node Fisher tangent minus node Euclidean NN error',
        '',
        '| Target | Group | Mean | 95% low | 95% high |',
        '|---|---|---:|---:|---:|',
    ])
    for target in ('one_step', 'three_step'):
        for group in ('matched', 'external', 'pooled'):
            item = bootstraps[target][group]
            lines.append(
                '| {} | {} | {:.4g} | {:.4g} | {:.4g} |'.format(
                    target,
                    group,
                    item['mean'],
                    item['ci95_low'],
                    item['ci95_high'],
                )
            )
    wins_three = all(
        lookup[('three_step', group, 'node_fisher_tangent')]['spearman_rho']
        > lookup[('three_step', group, 'node_euclidean')]['spearman_rho']
        and lookup[('three_step', group, 'node_fisher_tangent')]['normalized_nn_error']
        < lookup[('three_step', group, 'node_euclidean')]['normalized_nn_error']
        for group in ('matched', 'external')
    )
    pooled_below = bootstraps['three_step']['pooled']['ci95_high'] < 0.0
    one_safe = all(
        lookup[('one_step', group, 'node_fisher_tangent')]['spearman_rho']
        >= lookup[('one_step', group, 'node_euclidean')]['spearman_rho'] - 0.01
        for group in ('matched', 'external')
    )
    promoted = wins_three and pooled_below and one_safe
    lines.extend([
        '',
        '## Preregistered decision',
        '',
        '- Fisher tangent-metric promotion rule met: **{}**'.format(promoted),
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
    args.candidate_count = 2 if args.smoke else 16
    args.anchor_count = 2 if args.smoke else None
    if args.output_dir is None:
        args.output_dir = HERE / (
            'smoke_results_tangent' if args.smoke else 'results_tangent'
        )
    return args


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    candidates, records = dual.actual.select_candidates(args.candidate_count)
    anchors = dual.load_anchors(args.anchor_count)
    descriptors, gate = compute_descriptors(candidates, records, anchors)
    rows = []
    bootstraps = {}
    for target, path in dual.TARGETS.items():
        responses = dual.load_target(path, records)
        target_rows, bootstrap = evaluate_target(
            target, responses, descriptors, records
        )
        rows.extend(target_rows)
        bootstraps[target] = bootstrap
    scores = score_metrics(descriptors, records)
    elapsed = time.perf_counter() - start
    write_outputs(
        args.output_dir,
        args=args,
        records=records,
        anchors=anchors,
        descriptors=descriptors,
        rows=rows,
        bootstraps=bootstraps,
        score_rows=scores,
        gate=gate,
        elapsed=elapsed,
    )
    print('wrote {}'.format(args.output_dir / 'summary.md'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
