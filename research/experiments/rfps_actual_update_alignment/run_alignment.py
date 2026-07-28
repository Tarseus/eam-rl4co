from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from torch.func import functional_call


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PILOT_DIR = EXPERIMENTS / 'rfps_feature_pilot'
MINIMALITY_DIR = EXPERIMENTS / 'rfps_minimality'
INVARIANCE_DIR = EXPERIMENTS / 'rfps_invariance_validation'
REPO_ROOT = HERE.parents[2]
RUNTIME_POMO = (
    PILOT_DIR / 'find_pref_runtime' / 'rl4co' / 'models' / 'zoo' / 'pomo'
)
sys.path[:0] = [str(PILOT_DIR), str(MINIMALITY_DIR), str(INVARIANCE_DIR)]

import run_pilot as pilot  # noqa: E402
from run_validation import fisher_euclidean_blocks  # noqa: E402


CHECKPOINT = REPO_ROOT / 'tsp100_epoch_135.ckpt'
RUNS = {
    'matched': (
        PILOT_DIR
        / 'warmstart_snapshot'
        / 'runs'
        / 'pref_loss_tsp100_discovery'
        / '20260317-131507'
    ),
    'external': (
        PILOT_DIR
        / 'snapshot'
        / 'runs'
        / 'pref_loss_tsp100_discovery_scratch_only'
        / '20260408-092546'
    ),
}
PRIMARY = {
    'candidate_count': 16,
    'num_train_instances': 2,
    'num_heldout_instances': 2,
    'num_starts': 100,
    'train_seed': 2026072801,
    'heldout_seed': 2026072802,
    'policy_seed': 1234,
    'learning_rate': 3e-4,
    'weight_decay': 1e-6,
    'adam_eps': 1e-8,
    'ell': 0.03,
}


@dataclass
class CandidateRecord:
    source: str
    key: str
    score: float
    rank_index: int
    loss: float | None = None
    gradient_norm: float | None = None
    parameter_step_norm: float | None = None
    response_norm: float | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def import_policy_runtime():
    import rl4co.models.zoo.pomo as pomo_package

    runtime_path = str(RUNTIME_POMO.resolve())
    if runtime_path not in pomo_package.__path__:
        pomo_package.__path__.append(runtime_path)
    from rl4co.envs import TSPEnv
    from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy

    return TSPEnv, PO4COPsTSPPolicy


def load_policy():
    _, policy_cls = import_policy_runtime()
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state_dict = {
        key.removeprefix('policy.'): value
        for key, value in payload['state_dict'].items()
        if key.startswith('policy.')
    }
    torch.manual_seed(PRIMARY['policy_seed'])
    policy = policy_cls(
        env_name='tsp',
        start_node='pomo',
        eval_type='argmax',
        train_decode_type='sampling',
        val_decode_type='greedy',
        test_decode_type='greedy',
    )
    incompatible = policy.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    return policy


def select_candidates(count: int) -> tuple[list[pilot.Candidate], list[CandidateRecord]]:
    selected: list[pilot.Candidate] = []
    records: list[CandidateRecord] = []
    for source, run_dir in RUNS.items():
        candidates, failures = pilot.load_candidates(run_dir, 'final')
        if failures:
            raise RuntimeError(f'{source} candidate compile failures: {failures[:3]}')
        ordered = sorted(candidates, key=lambda item: (item.score, item.key))
        indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
        if len(set(indices.tolist())) != count:
            raise RuntimeError(f'{source}: quantile selection produced duplicate indices')
        for index in indices:
            candidate = ordered[int(index)]
            if not candidate.key.startswith('g_ref::'):
                raise RuntimeError(f'non-reference builder in primary set: {candidate.key}')
            selected.append(candidate)
            records.append(
                CandidateRecord(
                    source=source,
                    key=candidate.key,
                    score=float(candidate.score),
                    rank_index=int(index),
                )
            )
    return selected, records


def unflatten_starts(
    tensor: torch.Tensor,
    batch_size: int,
    num_starts: int,
) -> torch.Tensor:
    tail = tensor.shape[1:]
    return tensor.reshape(num_starts, batch_size, *tail).transpose(0, 1)


def make_env_bank(
    env_cls,
    *,
    seed: int,
    num_instances: int,
):
    env = env_cls(
        generator_params={'num_loc': 100},
        device='cpu',
        seed=seed,
    )
    batch = env.generator(num_instances)
    return env, env.reset(batch)


def make_training_graph(
    policy,
    env,
    reset_td,
    *,
    seed: int,
    num_instances: int,
    num_starts: int,
) -> pilot.Probe:
    torch.manual_seed(seed)
    policy.train()
    out = policy(
        reset_td,
        env,
        phase='train',
        num_starts=num_starts,
        return_actions=False,
        return_entropy=False,
        return_sum_log_likelihood=True,
    )
    reward = unflatten_starts(out['reward'], num_instances, num_starts).detach()
    log_likelihood = unflatten_starts(
        out['log_likelihood'], num_instances, num_starts
    )
    objective = -reward
    advantage = reward - reward.mean(dim=1, keepdim=True)
    seq_len = torch.full_like(log_likelihood, 100.0)
    entropy = torch.zeros_like(log_likelihood)
    return pilot.Probe(
        objective=objective,
        p0=log_likelihood,
        seq_len=seq_len,
        advantage=advantage,
        entropy=entropy,
    )


def make_heldout_bank(
    policy,
    env,
    reset_td,
    *,
    seed: int,
    num_instances: int,
    num_starts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    policy.eval()
    with torch.inference_mode():
        out = policy(
            reset_td,
            env,
            phase='train',
            num_starts=num_starts,
            return_actions=True,
            return_entropy=False,
            return_sum_log_likelihood=True,
        )
    actions = out['actions'].detach()
    log_likelihood = unflatten_starts(
        out['log_likelihood'], num_instances, num_starts
    ).detach()
    return actions, log_likelihood


def detached_probe_blocks(probe: pilot.Probe) -> list[pilot.Probe]:
    blocks: list[pilot.Probe] = []
    for index in range(probe.p0.shape[0]):
        values = {}
        for name in ('objective', 'p0', 'seq_len', 'advantage', 'entropy'):
            tensor = getattr(probe, name)[index : index + 1]
            values[name] = tensor.detach().to(dtype=pilot.DTYPE)
        blocks.append(pilot.Probe(**values))
    return blocks


def compute_descriptors(
    candidates: list[pilot.Candidate],
    probe: pilot.Probe,
) -> dict[str, np.ndarray]:
    blocks = detached_probe_blocks(probe)
    stores: dict[str, list[np.ndarray]] = {
        'one_point': [],
        'euclidean_two_point': [],
        'fisher_two_point': [],
    }
    q0 = torch.full(
        (probe.p0.shape[1],),
        1.0 / probe.p0.shape[1],
        dtype=pilot.DTYPE,
    )
    for candidate in candidates:
        per_method = {name: [] for name in stores}
        for block in blocks:
            descriptor, _, arc_error = fisher_euclidean_blocks(candidate, block, q0)
            if arc_error > 1e-8:
                raise RuntimeError(f'Fisher arc error {arc_error:.3e}')
            for name in stores:
                per_method[name].append(descriptor[name])
        for name in stores:
            vector = np.concatenate(per_method[name]) / math.sqrt(len(blocks))
            if not np.isfinite(vector).all():
                raise RuntimeError(f'non-finite {name} for {candidate.key}')
            stores[name].append(vector)
    return {name: np.stack(rows) for name, rows in stores.items()}


def candidate_loss(
    candidate: pilot.Candidate,
    probe: pilot.Probe,
) -> torch.Tensor:
    batch, model_output = pilot.pairwise_inputs(probe, probe.p0)
    loss = candidate.compiled.loss_fn(
        batch,
        model_output,
        {'alpha': candidate.alpha},
    )
    if not isinstance(loss, torch.Tensor):
        loss = torch.as_tensor(loss, dtype=probe.p0.dtype)
    if loss.numel() != 1:
        loss = loss.mean()
    if not torch.isfinite(loss):
        raise RuntimeError('non-finite loss')
    return loss


def first_adam_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients: tuple[torch.Tensor | None, ...],
    *,
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> tuple[dict[str, torch.Tensor], float, float]:
    updated: dict[str, torch.Tensor] = {}
    grad_sq = 0.0
    step_sq = 0.0
    for (name, parameter), gradient in zip(named_parameters, gradients, strict=True):
        if gradient is None:
            updated[name] = parameter.detach()
            continue
        effective_gradient = gradient.detach() + weight_decay * parameter.detach()
        step = -learning_rate * effective_gradient / (
            effective_gradient.abs() + eps
        )
        updated[name] = parameter.detach() + step
        grad_sq += float(gradient.detach().double().square().sum().item())
        step_sq += float(step.double().square().sum().item())
    return updated, math.sqrt(grad_sq), math.sqrt(step_sq)


def heldout_response(
    policy,
    parameters: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    env,
    reset_td,
    actions: torch.Tensor,
    base_log_likelihood: torch.Tensor,
    *,
    num_instances: int,
    num_starts: int,
) -> tuple[np.ndarray, float]:
    policy.eval()
    with torch.inference_mode():
        out = functional_call(
            policy,
            (parameters, buffers),
            (reset_td, env),
            {
                'phase': 'train',
                'num_starts': num_starts,
                'return_actions': False,
                'return_entropy': False,
                'return_sum_log_likelihood': True,
                'forced_actions': actions,
            },
            strict=False,
        )
    updated_log_likelihood = unflatten_starts(
        out['log_likelihood'], num_instances, num_starts
    )
    delta = updated_log_likelihood - base_log_likelihood
    delta = delta - delta.mean(dim=1, keepdim=True)
    raw_norm = float(torch.linalg.vector_norm(delta.double()).item())
    if not math.isfinite(raw_norm) or raw_norm <= 1e-12:
        raise RuntimeError(f'invalid held-out response norm: {raw_norm}')
    return (delta / raw_norm).reshape(-1).double().numpy(), raw_norm


def distance_matrix(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        raise ValueError('at least two rows are required')
    return squareform(pdist(matrix, metric='euclidean'))


def nearest_indices(distance: np.ndarray, k: int = 1) -> np.ndarray:
    masked = distance.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argsort(masked, axis=1)[:, :k]


def method_metrics(
    descriptor_distance: np.ndarray,
    actual_distance: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    n = len(actual_distance)
    upper = np.triu_indices(n, k=1)
    rho = float(spearmanr(
        descriptor_distance[upper],
        actual_distance[upper],
    ).statistic)
    descriptor_nn = nearest_indices(descriptor_distance, 1)[:, 0]
    actual_nn = nearest_indices(actual_distance, 1)[:, 0]
    query = np.arange(n)
    nn_error = actual_distance[query, descriptor_nn]
    random_mean = float(
        (actual_distance.sum(axis=1) / max(1, n - 1)).mean()
    )
    top5_size = min(5, n - 1)
    descriptor_top5 = nearest_indices(descriptor_distance, top5_size)
    actual_top5 = nearest_indices(actual_distance, top5_size)
    top5_overlap = np.mean([
        len(set(descriptor_top5[i]).intersection(actual_top5[i])) / top5_size
        for i in range(n)
    ])
    metrics = {
        'spearman_rho': rho,
        'normalized_nn_error': float(nn_error.mean() / random_mean),
        'top1_agreement': float(np.mean(descriptor_nn == actual_nn)),
        'top5_overlap': float(top5_overlap),
        'mean_nn_actual_distance': float(nn_error.mean()),
        'mean_random_actual_distance': random_mean,
    }
    return metrics, nn_error


def bootstrap_paired_difference(
    fisher_error: np.ndarray,
    euclidean_error: np.ndarray,
    *,
    seed: int = 20260728,
    draws: int = 10000,
) -> dict[str, float]:
    difference = fisher_error - euclidean_error
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        'mean': float(difference.mean()),
        'ci95_low': float(np.quantile(means, 0.025)),
        'ci95_high': float(np.quantile(means, 0.975)),
    }


def evaluate_metrics(
    descriptors: dict[str, np.ndarray],
    responses: np.ndarray,
    records: list[CandidateRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    groups = {
        'matched': np.array([r.source == 'matched' for r in records]),
        'external': np.array([r.source == 'external' for r in records]),
        'pooled': np.ones(len(records), dtype=bool),
    }
    for group, mask in groups.items():
        actual_distance = distance_matrix(responses[mask])
        errors: dict[str, np.ndarray] = {}
        for method, matrix in descriptors.items():
            metrics, nn_error = method_metrics(
                distance_matrix(matrix[mask]),
                actual_distance,
            )
            rows.append({'group': group, 'method': method, **metrics})
            errors[method] = nn_error
        bootstrap[group] = bootstrap_paired_difference(
            errors['fisher_two_point'],
            errors['euclidean_two_point'],
        )
    return rows, bootstrap


def write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    records: list[CandidateRecord],
    descriptors: dict[str, np.ndarray],
    responses: np.ndarray,
    metric_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
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
        keys=np.array([r.key for r in records]),
        sources=np.array([r.source for r in records]),
        **descriptors,
    )
    metadata = {
        'smoke': bool(args.smoke),
        'configuration': {
            'candidate_count_per_source': args.candidate_count,
            'num_train_instances': args.num_train_instances,
            'num_heldout_instances': args.num_heldout_instances,
            'num_starts': args.num_starts,
            **{
                key: value
                for key, value in PRIMARY.items()
                if key not in {
                    'candidate_count',
                    'num_train_instances',
                    'num_heldout_instances',
                    'num_starts',
                }
            },
        },
        'checkpoint': str(CHECKPOINT.resolve()),
        'checkpoint_sha256': sha256(CHECKPOINT),
        'torch_version': torch.__version__,
        'elapsed_seconds': elapsed_seconds,
        'bootstrap': bootstrap,
        'failures': [asdict(r) for r in records if r.error],
    }
    (output_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )

    lookup = {(row['group'], row['method']): row for row in metric_rows}
    lines = [
        '# Fixed-chart actual-update alignment',
        '',
        f'- Smoke run: `{bool(args.smoke)}`',
        f'- Candidates: {len(records)}',
        f'- Train / held-out instances: {args.num_train_instances} / '
        f'{args.num_heldout_instances}',
        f'- POMO starts: {args.num_starts}',
        f'- Elapsed seconds: {elapsed_seconds:.1f}',
        '',
        '| Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for group in ('matched', 'external', 'pooled'):
        for method in ('one_point', 'euclidean_two_point', 'fisher_two_point'):
            row = lookup[(group, method)]
            lines.append(
                f"| {group} | {method} | {row['spearman_rho']:.3f} | "
                f"{row['normalized_nn_error']:.3f} | "
                f"{row['top1_agreement']:.3f} | {row['top5_overlap']:.3f} |"
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
            f"| {group} | {item['mean']:.4g} | "
            f"{item['ci95_low']:.4g} | {item['ci95_high']:.4g} |"
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
        '## Preregistered decision',
        '',
        f'- Fisher promotion rule met: **{fisher_wins}**',
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
        args.num_train_instances = 1
        args.num_heldout_instances = 1
        args.num_starts = 8
    else:
        args.candidate_count = PRIMARY['candidate_count']
        args.num_train_instances = PRIMARY['num_train_instances']
        args.num_heldout_instances = PRIMARY['num_heldout_instances']
        args.num_starts = PRIMARY['num_starts']
    if args.output_dir is None:
        args.output_dir = HERE / ('smoke_results' if args.smoke else 'results')
    return args


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    np.random.seed(PRIMARY['train_seed'] % (2**32))

    env_cls, _ = import_policy_runtime()
    policy = load_policy()
    candidates, records = select_candidates(args.candidate_count)
    print(f'loaded {len(candidates)} candidates', flush=True)

    heldout_env, heldout_td = make_env_bank(
        env_cls,
        seed=PRIMARY['heldout_seed'],
        num_instances=args.num_heldout_instances,
    )
    heldout_actions, base_heldout_log_likelihood = make_heldout_bank(
        policy,
        heldout_env,
        heldout_td,
        seed=PRIMARY['heldout_seed'],
        num_instances=args.num_heldout_instances,
        num_starts=args.num_starts,
    )
    train_env, train_td = make_env_bank(
        env_cls,
        seed=PRIMARY['train_seed'],
        num_instances=args.num_train_instances,
    )
    train_probe = make_training_graph(
        policy,
        train_env,
        train_td,
        seed=PRIMARY['train_seed'],
        num_instances=args.num_train_instances,
        num_starts=args.num_starts,
    )
    print('built common on-policy training graph and held-out bank', flush=True)

    descriptors = compute_descriptors(candidates, train_probe)
    print('computed fixed-chart descriptors', flush=True)

    named_parameters = list(policy.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    buffers = {
        name: buffer.detach()
        for name, buffer in policy.named_buffers()
    }
    responses: list[np.ndarray] = []
    valid_indices: list[int] = []
    for index, (candidate, record) in enumerate(zip(candidates, records, strict=True)):
        candidate_start = time.perf_counter()
        try:
            loss = candidate_loss(candidate, train_probe)
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=index < len(candidates) - 1,
                allow_unused=True,
            )
            updated, gradient_norm, step_norm = first_adam_parameters(
                named_parameters,
                gradients,
                learning_rate=PRIMARY['learning_rate'],
                weight_decay=PRIMARY['weight_decay'],
                eps=PRIMARY['adam_eps'],
            )
            response, response_norm = heldout_response(
                policy,
                updated,
                buffers,
                heldout_env,
                heldout_td,
                heldout_actions,
                base_heldout_log_likelihood,
                num_instances=args.num_heldout_instances,
                num_starts=args.num_starts,
            )
            record.loss = float(loss.detach().item())
            record.gradient_norm = gradient_norm
            record.parameter_step_norm = step_norm
            record.response_norm = response_norm
            responses.append(response)
            valid_indices.append(index)
        except Exception as exc:  # noqa: BLE001
            record.error = f'{type(exc).__name__}: {exc}'
            print(f'FAILED {record.source} {record.key}: {record.error}', flush=True)
        record.elapsed_seconds = time.perf_counter() - candidate_start
        print(
            f'[{index + 1}/{len(candidates)}] {record.source} {record.key} '
            f'{record.elapsed_seconds:.1f}s',
            flush=True,
        )

    if len(valid_indices) < 4:
        raise RuntimeError('too few valid actual-update responses')
    if len(valid_indices) != len(candidates):
        candidates = [candidates[i] for i in valid_indices]
        metric_records = [records[i] for i in valid_indices]
        descriptors = {
            name: matrix[valid_indices]
            for name, matrix in descriptors.items()
        }
    else:
        metric_records = records
    response_matrix = np.stack(responses)
    metric_rows, bootstrap = evaluate_metrics(
        descriptors,
        response_matrix,
        metric_records,
    )
    elapsed_seconds = time.perf_counter() - start_time
    write_outputs(
        args.output_dir,
        args=args,
        records=metric_records,
        descriptors=descriptors,
        responses=response_matrix,
        metric_rows=metric_rows,
        bootstrap=bootstrap,
        elapsed_seconds=elapsed_seconds,
    )
    print(f'wrote {args.output_dir / "summary.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
