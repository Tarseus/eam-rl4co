from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ACTUAL = EXPERIMENTS / 'rfps_actual_update_alignment'
PILOT = EXPERIMENTS / 'rfps_feature_pilot'
sys.path[:0] = [str(ACTUAL), str(PILOT)]

import run_alignment as actual  # noqa: E402


PROBE_PATHS = (
    PILOT / 'tsp100_epoch135_rollout_probes.npz',
    PILOT / 'tsp100_epoch135_rollout_probes_seed2.npz',
)
TARGETS = {
    'one_step': ACTUAL / 'results' / 'artifacts.npz',
    'three_step': ACTUAL / 'results_3step_disjoint' / 'artifacts.npz',
}
METHODS = (
    'node_euclidean',
    'pullback_fisher',
    'pair_euclidean',
    'pair_fisher',
)
DTYPE = torch.float64


@dataclass
class Anchor:
    probe: actual.pilot.Probe
    winner: torch.Tensor
    loser: torch.Tensor
    margin: torch.Tensor
    pair_fisher_weight: torch.Tensor
    helmert: torch.Tensor
    fisher_inverse_sqrt: torch.Tensor
    fisher_eigen_min: float
    fisher_eigen_max: float


def unit(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm.item()) <= 1e-14:
        raise RuntimeError('non-finite or zero descriptor block')
    return vector / norm


def helmert_basis(size: int) -> torch.Tensor:
    basis = torch.zeros((size, size - 1), dtype=DTYPE)
    for column in range(1, size):
        scale = math.sqrt(column * (column + 1))
        basis[:column, column - 1] = 1.0 / scale
        basis[column, column - 1] = -column / scale
    return basis


def objective_pairs(probe: actual.pilot.Probe) -> tuple[torch.Tensor, torch.Tensor]:
    objective = probe.objective
    if objective.shape[0] != 1:
        raise ValueError('each anchor must contain one instance')
    mask = objective[:, :, None] < objective[:, None, :]
    _, winner, loser = mask.nonzero(as_tuple=True)
    return winner, loser


def build_anchor(probe: actual.pilot.Probe) -> Anchor:
    winner, loser = objective_pairs(probe)
    p = probe.p0.reshape(-1).to(dtype=DTYPE)
    margin = p[winner] - p[loser]
    log_weight = F.logsigmoid(margin) + F.logsigmoid(-margin)
    pair_weight = torch.exp(log_weight)
    if not torch.isfinite(pair_weight).all() or bool((pair_weight <= 0).any()):
        raise RuntimeError('invalid exact Bernoulli Fisher weights')

    size = p.numel()
    laplacian = torch.zeros((size, size), dtype=DTYPE)
    laplacian.index_put_((winner, winner), pair_weight, accumulate=True)
    laplacian.index_put_((loser, loser), pair_weight, accumulate=True)
    laplacian.index_put_((winner, loser), -pair_weight, accumulate=True)
    laplacian.index_put_((loser, winner), -pair_weight, accumulate=True)
    helmert = helmert_basis(size)
    reduced = helmert.T @ laplacian @ helmert
    eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
    if not torch.isfinite(eigenvalues).all() or float(eigenvalues.min()) <= 0.0:
        raise RuntimeError('pullback Fisher matrix is not positive definite')
    inverse_sqrt = (
        eigenvectors
        @ torch.diag(eigenvalues.rsqrt())
        @ eigenvectors.T
    )
    return Anchor(
        probe=probe,
        winner=winner,
        loser=loser,
        margin=margin,
        pair_fisher_weight=pair_weight,
        helmert=helmert,
        fisher_inverse_sqrt=inverse_sqrt,
        fisher_eigen_min=float(eigenvalues.min().item()),
        fisher_eigen_max=float(eigenvalues.max().item()),
    )


def scalar_loss(
    candidate: actual.pilot.Candidate,
    batch: dict[str, torch.Tensor],
    model_output: dict[str, torch.Tensor],
) -> torch.Tensor:
    loss = candidate.compiled.loss_fn(
        batch,
        model_output,
        {'alpha': candidate.alpha},
    )
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, dtype=DTYPE)
    if loss.numel() != 1:
        loss = loss.mean()
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite candidate loss')
    return loss


def actual_inputs(
    probe: actual.pilot.Probe,
    p: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return actual.pilot.pairwise_inputs(probe, p)


def centered_margin_inputs(
    anchor: Anchor,
    margin: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    original_batch, model_output = actual_inputs(anchor.probe, anchor.probe.p0)
    batch = dict(original_batch)
    winner_value = 0.5 * margin
    loser_value = -0.5 * margin
    batch['log_prob_w'] = winner_value
    batch['log_prob_l'] = loser_value
    batch['logprob_w'] = winner_value
    batch['logprob_l'] = loser_value
    batch['log_prob_gap'] = loser_value - winner_value
    if 'seq_len_w' in batch and 'seq_len_l' in batch:
        batch['log_prob_w_mean'] = winner_value / batch['seq_len_w']
        batch['log_prob_l_mean'] = loser_value / batch['seq_len_l']
        batch['log_prob_mean_gap'] = (
            batch['log_prob_l_mean'] - batch['log_prob_w_mean']
        )
    return batch, model_output


def shift_invariance_gate(
    candidate: actual.pilot.Candidate,
    anchor: Anchor,
) -> tuple[float, float, float, float]:
    p = anchor.probe.p0.detach().clone().requires_grad_(True)
    batch, model_output = actual_inputs(anchor.probe, p)
    loss = scalar_loss(candidate, batch, model_output)
    gradient = torch.autograd.grad(loss, p)[0]
    denominator = (
        torch.linalg.vector_norm(gradient) * math.sqrt(p.numel())
    ).clamp_min(1e-30)
    common_ratio = float(gradient.sum().abs().item() / denominator.item())
    base_value = float(loss.detach().item())
    relative_shift_change = 0.0
    for shift in (-5.0, -1.0, 1.0, 5.0):
        shifted = anchor.probe.p0 + shift
        shifted_batch, shifted_output = actual_inputs(anchor.probe, shifted)
        shifted_loss = scalar_loss(candidate, shifted_batch, shifted_output)
        change = abs(float(shifted_loss.item()) - base_value) / max(
            abs(base_value), 1e-8
        )
        relative_shift_change = max(relative_shift_change, change)

    margin = anchor.margin.detach().clone().requires_grad_(True)
    centered_batch, centered_output = centered_margin_inputs(anchor, margin)
    centered_loss = scalar_loss(candidate, centered_batch, centered_output)
    edge_gradient = torch.autograd.grad(centered_loss, margin)[0]
    reconstructed = torch.zeros_like(p.reshape(-1))
    reconstructed.index_add_(0, anchor.winner, edge_gradient)
    reconstructed.index_add_(0, anchor.loser, -edge_gradient)
    chain_error = float(
        torch.linalg.vector_norm(reconstructed - gradient.reshape(-1)).item()
        / torch.linalg.vector_norm(gradient.reshape(-1)).clamp_min(1e-30).item()
    )
    centered_change = abs(float(centered_loss.item()) - base_value) / max(
        abs(base_value), 1e-8
    )
    return common_ratio, relative_shift_change, centered_change, chain_error


def descriptor_blocks(
    candidate: actual.pilot.Candidate,
    anchor: Anchor,
) -> dict[str, np.ndarray]:
    margin = anchor.margin.detach().clone().requires_grad_(True)
    batch, model_output = centered_margin_inputs(anchor, margin)
    loss = scalar_loss(candidate, batch, model_output)
    edge_covector = torch.autograd.grad(loss, margin)[0]
    if not torch.isfinite(edge_covector).all():
        raise RuntimeError('non-finite pair-margin covector')

    node_covector = torch.zeros(
        anchor.probe.p0.numel(), dtype=DTYPE
    )
    node_covector.index_add_(0, anchor.winner, edge_covector)
    node_covector.index_add_(0, anchor.loser, -edge_covector)
    centered_node = anchor.helmert.T @ node_covector
    fisher_node = anchor.fisher_inverse_sqrt @ centered_node
    fisher_pair = edge_covector / torch.sqrt(anchor.pair_fisher_weight)
    blocks = {
        'node_euclidean': unit(centered_node),
        'pullback_fisher': unit(fisher_node),
        'pair_euclidean': unit(edge_covector),
        'pair_fisher': unit(fisher_pair),
    }
    return {
        name: vector.detach().cpu().numpy()
        for name, vector in blocks.items()
    }


def load_anchors(limit: int | None = None) -> list[Anchor]:
    probes: list[actual.pilot.Probe] = []
    for path in PROBE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
        probes.extend(actual.pilot.load_probes_npz(path))
    if limit is not None:
        probes = probes[:limit]
    anchors = [build_anchor(probe) for probe in probes]
    return anchors


def compute_descriptors(
    candidates: list[actual.pilot.Candidate],
    records: list[actual.CandidateRecord],
    anchors: list[Anchor],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stores = {method: [] for method in METHODS}
    failures: list[dict[str, str]] = []
    gate_rows: list[dict[str, Any]] = []
    for index, (candidate, record) in enumerate(zip(candidates, records, strict=True)):
        per_method = {method: [] for method in METHODS}
        try:
            for anchor_index, anchor in enumerate(anchors):
                (
                    common_ratio,
                    shift_change,
                    centered_change,
                    chain_error,
                ) = shift_invariance_gate(candidate, anchor)
                gate_rows.append({
                    'source': record.source,
                    'key': record.key,
                    'anchor': anchor_index,
                    'common_gradient_ratio': common_ratio,
                    'relative_shift_loss_change': shift_change,
                    'relative_centered_loss_change': centered_change,
                    'relative_chain_rule_error': chain_error,
                })
                if max(
                    common_ratio, shift_change, centered_change, chain_error
                ) > 1e-8:
                    raise RuntimeError(
                        'margin-only gate failed: common={:.3e} shift={:.3e} '
                        'centered={:.3e} chain={:.3e}'.format(
                            common_ratio, shift_change, centered_change, chain_error
                        )
                    )
                blocks = descriptor_blocks(candidate, anchor)
                for method in METHODS:
                    per_method[method].append(blocks[method])
            for method in METHODS:
                stores[method].append(
                    np.concatenate(per_method[method]) / math.sqrt(len(anchors))
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({
                'source': record.source,
                'key': record.key,
                'error': '{}: {}'.format(type(exc).__name__, exc),
            })
        print(
            '[{}/{}] {} {}'.format(
                index + 1, len(candidates), record.source, record.key
            ),
            flush=True,
        )
    if failures:
        raise RuntimeError('descriptor failures: {}'.format(failures[:3]))
    matrices = {method: np.stack(rows) for method, rows in stores.items()}
    gate_summary = {
        'max_common_gradient_ratio': max(
            row['common_gradient_ratio'] for row in gate_rows
        ),
        'max_relative_shift_loss_change': max(
            row['relative_shift_loss_change'] for row in gate_rows
        ),
        'max_relative_centered_loss_change': max(
            row['relative_centered_loss_change'] for row in gate_rows
        ),
        'max_relative_chain_rule_error': max(
            row['relative_chain_rule_error'] for row in gate_rows
        ),
        'rows': gate_rows,
    }
    return matrices, gate_summary


def load_target(
    path: Path,
    records: list[actual.CandidateRecord],
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        keys = archive['keys'].astype(str)
        sources = archive['sources'].astype(str)
        lookup = {
            (source, key): index
            for index, (source, key) in enumerate(zip(sources, keys, strict=True))
        }
        indices = [lookup[(record.source, record.key)] for record in records]
        return np.asarray(archive['responses'])[indices]


def distance_matrix(matrix: np.ndarray) -> np.ndarray:
    return squareform(pdist(matrix, metric='euclidean'))


def evaluate_target(
    target_name: str,
    responses: np.ndarray,
    descriptors: dict[str, np.ndarray],
    records: list[actual.CandidateRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    groups = {
        'matched': np.array([record.source == 'matched' for record in records]),
        'external': np.array([record.source == 'external' for record in records]),
        'pooled': np.ones(len(records), dtype=bool),
    }
    for group, mask in groups.items():
        actual_distance = distance_matrix(responses[mask])
        errors: dict[str, np.ndarray] = {}
        for method in METHODS:
            metrics, nn_error = actual.method_metrics(
                distance_matrix(descriptors[method][mask]),
                actual_distance,
            )
            rows.append({
                'target': target_name,
                'group': group,
                'method': method,
                **metrics,
            })
            errors[method] = nn_error
        bootstrap[group] = actual.bootstrap_paired_difference(
            errors['pullback_fisher'],
            errors['node_euclidean'],
        )
    return rows, bootstrap


def score_metrics(
    descriptors: dict[str, np.ndarray],
    records: list[actual.CandidateRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ('matched', 'external'):
        mask = np.array([record.source == group for record in records])
        scores = np.array([record.score for record in records])[mask]
        score_distance = distance_matrix(scores[:, None])
        upper = np.triu_indices(len(scores), k=1)
        for method in METHODS:
            descriptor_distance = distance_matrix(descriptors[method][mask])
            rho = float(spearmanr(
                descriptor_distance[upper], score_distance[upper]
            ).statistic)
            masked_distance = descriptor_distance.copy()
            np.fill_diagonal(masked_distance, np.inf)
            neighbor = np.argmin(masked_distance, axis=1)
            nn_error = np.abs(scores - scores[neighbor])
            random_mean = float(
                (score_distance.sum(axis=1) / (len(scores) - 1)).mean()
            )
            rows.append({
                'group': group,
                'method': method,
                'score_distance_rho': rho,
                'normalized_nn_score_error': float(nn_error.mean() / random_mean),
                'median_nn_score_error': float(np.median(nn_error)),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    records: list[actual.CandidateRecord],
    descriptors: dict[str, np.ndarray],
    metric_rows: list[dict[str, Any]],
    bootstraps: dict[str, Any],
    score_rows: list[dict[str, Any]],
    gate_summary: dict[str, Any],
    anchors: list[Anchor],
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'metrics.csv', metric_rows)
    write_csv(output_dir / 'score_metrics.csv', score_rows)
    write_csv(output_dir / 'shift_invariance.csv', gate_summary['rows'])
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
        'gate': {key: value for key, value in gate_summary.items() if key != 'rows'},
        'fisher_spectrum': {
            'min_eigenvalue': min(anchor.fisher_eigen_min for anchor in anchors),
            'max_eigenvalue': max(anchor.fisher_eigen_max for anchor in anchors),
            'max_condition_number': max(
                anchor.fisher_eigen_max / anchor.fisher_eigen_min
                for anchor in anchors
            ),
        },
        'bootstraps': bootstraps,
        'elapsed_seconds': elapsed_seconds,
    }
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )

    lookup = {
        (row['target'], row['group'], row['method']): row
        for row in metric_rows
    }
    lines = [
        '# Real pair-margin Fisher validation',
        '',
        '- Retrospective/exploratory: `True`',
        '- Smoke: `{}`'.format(args.smoke),
        '- Candidates: {}'.format(len(records)),
        '- Real-margin anchors: {}'.format(len(anchors)),
        '- Elapsed seconds: {:.1f}'.format(elapsed_seconds),
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
        '## Pullback Fisher minus node Euclidean NN error',
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

    fisher_wins_three = all(
        lookup[('three_step', group, 'pullback_fisher')]['spearman_rho']
        > lookup[('three_step', group, 'node_euclidean')]['spearman_rho']
        and lookup[('three_step', group, 'pullback_fisher')]['normalized_nn_error']
        < lookup[('three_step', group, 'node_euclidean')]['normalized_nn_error']
        for group in ('matched', 'external')
    )
    pooled_interval = bootstraps['three_step']['pooled']['ci95_high'] < 0.0
    one_step_safe = all(
        lookup[('one_step', group, 'pullback_fisher')]['spearman_rho']
        >= lookup[('one_step', group, 'node_euclidean')]['spearman_rho'] - 0.01
        for group in ('matched', 'external')
    )
    promoted = fisher_wins_three and pooled_interval and one_step_safe
    lines.extend([
        '',
        '## Preregistered decision',
        '',
        '- Fisher promotion rule met: **{}**'.format(promoted),
        '- Maximum common-gradient ratio: `{:.3e}`'.format(
            gate_summary['max_common_gradient_ratio']
        ),
        '- Maximum common-shift loss change: `{:.3e}`'.format(
            gate_summary['max_relative_shift_loss_change']
        ),
        '- Maximum centered-margin loss change: `{:.3e}`'.format(
            gate_summary['max_relative_centered_loss_change']
        ),
        '- Maximum margin chain-rule error: `{:.3e}`'.format(
            gate_summary['max_relative_chain_rule_error']
        ),
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
        args.output_dir = HERE / ('smoke_results' if args.smoke else 'results')
    return args


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    candidates, records = actual.select_candidates(args.candidate_count)
    anchors = load_anchors(args.anchor_count)
    print(
        'loaded {} candidates and {} anchors'.format(len(candidates), len(anchors)),
        flush=True,
    )
    descriptors, gate_summary = compute_descriptors(candidates, records, anchors)
    metric_rows: list[dict[str, Any]] = []
    bootstraps: dict[str, Any] = {}
    for target_name, target_path in TARGETS.items():
        responses = load_target(target_path, records)
        rows, bootstrap = evaluate_target(
            target_name, responses, descriptors, records
        )
        metric_rows.extend(rows)
        bootstraps[target_name] = bootstrap
    score_rows = score_metrics(descriptors, records)
    elapsed = time.perf_counter() - start
    write_outputs(
        args.output_dir,
        args=args,
        records=records,
        descriptors=descriptors,
        metric_rows=metric_rows,
        bootstraps=bootstraps,
        score_rows=score_rows,
        gate_summary=gate_summary,
        anchors=anchors,
        elapsed_seconds=elapsed,
    )
    print('wrote {}'.format(args.output_dir / 'summary.md'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
