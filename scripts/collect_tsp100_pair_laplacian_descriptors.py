#!/usr/bin/env python3
"""Collect checkpoint-conditioned TSP100 loss-behaviour descriptors.

The experiment evaluates archived generated losses on real policy rollouts from
one fixed checkpoint.  It compares three objects:

1. the canonical Bradley--Terry Fisher Laplacian, whose conductance is
   p(1-p) and is independent of the candidate loss;
2. a loss-specific empirical-Fisher Laplacian, whose conductance is the
   squared derivative with respect to each pair margin;
3. the signed first-order node drift induced by the loss.

Nodes are reordered by tour-cost rank before descriptors are averaged, making
the matrices comparable across independently generated TSP100 instances.
Historical fitness is carried only as provenance and is never used to build or
evaluate the descriptor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PTP_ROOT = REPO_ROOT / "PTP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PTP_ROOT) not in sys.path:
    sys.path.insert(0, str(PTP_ROOT))

from fitness.free_loss_fidelity import (  # noqa: E402
    _load_policy_weights_from_checkpoint,
    _rl4co_build_env,
    _rl4co_build_policy,
    _rl4co_objective_from_reward,
    _rl4co_rollout,
    extract_feature_cache,
    prepare_pairwise_loss_batch,
)
from fitness.ptp_high_fidelity import _set_seed  # noqa: E402
from scripts.final_gradient_behavior_analysis import build_problem_specs  # noqa: E402


EPS = 1e-12


class TorchOps:
    """Operator surface used by the archived generated-loss programs."""

    abs = staticmethod(torch.abs)
    add = staticmethod(torch.add)
    clamp = staticmethod(torch.clamp)
    div = staticmethod(torch.div)
    exp = staticmethod(torch.exp)
    log = staticmethod(torch.log)
    logsigmoid = staticmethod(torch.nn.functional.logsigmoid)
    mean = staticmethod(torch.mean)
    median = staticmethod(torch.median)
    mul = staticmethod(torch.mul)
    neg = staticmethod(torch.neg)
    norm = staticmethod(torch.linalg.vector_norm)
    pow = staticmethod(torch.pow)
    relu = staticmethod(torch.relu)
    sigmoid = staticmethod(torch.sigmoid)
    softplus = staticmethod(torch.nn.functional.softplus)
    sign = staticmethod(torch.sign)
    sqrt = staticmethod(torch.sqrt)
    sub = staticmethod(torch.sub)
    sum = staticmethod(torch.sum)
    tanh = staticmethod(torch.tanh)
    ones_like = staticmethod(torch.ones_like)
    zeros_like = staticmethod(torch.zeros_like)

    @staticmethod
    def max(input: torch.Tensor, other: torch.Tensor | float | None = None) -> torch.Tensor:
        if other is None:
            return torch.max(input)
        other_t = torch.as_tensor(other, dtype=input.dtype, device=input.device)
        return torch.maximum(input, other_t)

    maximum = max

    @staticmethod
    def scalar_tensor(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64)


@dataclass(frozen=True)
class Probe:
    seed: int
    batch_id: int
    feature_cache: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PairContext:
    full_batch: dict[str, torch.Tensor]
    instance: torch.Tensor
    winner: torch.Tensor
    loser: torch.Tensor
    winner_rank: torch.Tensor
    loser_rank: torch.Tensor
    batch_size: int
    num_nodes: int


@dataclass
class SeedDescriptor:
    candidate_id: str
    seed: int
    matrix: np.ndarray
    drift: np.ndarray
    spectrum: np.ndarray
    log_margin_mass: float
    shift_leakage: float
    active_fraction: float
    loss_value: float
    status: str = "ok"
    error: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None and limit > 0:
        records = records[: int(limit)]
    if not records:
        raise ValueError(f"No loss records found in {path}")
    ids = [_candidate_id(record) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Loss record file contains duplicate candidate IDs")
    return records


def _candidate_id(record: Mapping[str, Any]) -> str:
    return str(
        record.get("candidate_id")
        or record.get("f_id")
        or (record.get("loss") or {}).get("id")
        or "unknown"
    )


def _loss_ir(record: Mapping[str, Any]) -> Mapping[str, Any]:
    loss = record.get("loss")
    if isinstance(loss, Mapping) and isinstance(loss.get("ir"), Mapping):
        return loss["ir"]
    if isinstance(record.get("f_ir"), Mapping):
        return record["f_ir"]
    raise KeyError(f"Record {_candidate_id(record)} has no loss IR")


def _compile_loss(record: Mapping[str, Any]):
    ir = _loss_ir(record)
    source = str(ir["code"])
    namespace: dict[str, Any] = {"ops": TorchOps, "torch": torch}
    exec(compile(source, f"<archived-loss:{_candidate_id(record)}>", "exec"), namespace)
    generated = namespace.get("generated_loss")
    if not callable(generated):
        raise TypeError("Loss source did not define generated_loss")
    return generated


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > EPS)


def _all_pairs_context(feature_cache: Mapping[str, torch.Tensor]) -> PairContext:
    objective = feature_cache["objective"].detach()
    log_prob = feature_cache["log_prob"].detach()
    batch_size, num_nodes = objective.shape
    instance, winner, loser = (
        objective[:, :, None] < objective[:, None, :]
    ).nonzero(as_tuple=True)

    order = objective.argsort(dim=1, descending=False)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(num_nodes, device=objective.device)[None, :].expand(
        batch_size, num_nodes
    )
    ranks.scatter_(1, order, rank_values)

    cost_w = objective[instance, winner]
    cost_l = objective[instance, loser]
    gap = cost_l - cost_w
    cost_std = objective.std(dim=1, unbiased=False).clamp_min(EPS)
    advantage = feature_cache.get("advantage")
    if not isinstance(advantage, torch.Tensor):
        reward_like = -objective
        advantage = reward_like - reward_like.mean(dim=1, keepdim=True)
    adv_w = advantage[instance, winner]
    adv_l = advantage[instance, loser]

    full_batch = {
        "log_prob_w": log_prob[instance, winner],
        "log_prob_l": log_prob[instance, loser],
        "cost_a": cost_w,
        "cost_b": cost_l,
        "cost_gap": gap,
        "advantage_w": adv_w,
        "advantage_l": adv_l,
        "advantage_gap": adv_w - adv_l,
        "delta_rank": (
            ranks[instance, loser] - ranks[instance, winner]
        ).to(dtype=objective.dtype)
        / float(max(num_nodes - 1, 1)),
        "delta_regret": gap,
        "delta_z": gap / cost_std[instance],
        "weight": torch.ones_like(gap),
    }
    return PairContext(
        full_batch=full_batch,
        instance=instance,
        winner=winner,
        loser=loser,
        winner_rank=ranks[instance, winner],
        loser_rank=ranks[instance, loser],
        batch_size=int(batch_size),
        num_nodes=int(num_nodes),
    )


def _matrix_and_drift(
    context: PairContext,
    conductance: torch.Tensor,
    grad_w: torch.Tensor,
    grad_l: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a trace-normalised rank-space Laplacian and signed node drift."""

    matrices: list[torch.Tensor] = []
    drifts: list[torch.Tensor] = []
    for instance_id in range(context.batch_size):
        pick = context.instance == instance_id
        c = conductance[pick].clamp_min(0)
        rw = context.winner_rank[pick]
        rl = context.loser_rank[pick]
        c_sum = c.sum()
        if not torch.isfinite(c_sum) or float(c_sum.item()) <= EPS:
            matrix = torch.zeros(
                (context.num_nodes, context.num_nodes),
                dtype=c.dtype,
                device=c.device,
            )
        else:
            c = c / c_sum
            matrix = torch.zeros(
                (context.num_nodes, context.num_nodes),
                dtype=c.dtype,
                device=c.device,
            )
            degree = torch.zeros(context.num_nodes, dtype=c.dtype, device=c.device)
            degree.scatter_add_(0, rw, c)
            degree.scatter_add_(0, rl, c)
            diag = torch.arange(context.num_nodes, device=c.device)
            matrix[diag, diag] = degree
            matrix.index_put_((rw, rl), -c, accumulate=True)
            matrix.index_put_((rl, rw), -c, accumulate=True)
        matrices.append(matrix)

        drift_original = torch.zeros(
            context.num_nodes, dtype=grad_w.dtype, device=grad_w.device
        )
        drift_original.scatter_add_(0, context.winner[pick], grad_w[pick])
        drift_original.scatter_add_(0, context.loser[pick], grad_l[pick])
        drift_ranked = torch.zeros_like(drift_original)
        drift_ranked.scatter_(0, context.winner_rank.new_tensor(
            torch.arange(context.num_nodes, device=grad_w.device)
        ), drift_original)
        # The previous scatter is identity; reorder explicitly using node ranks.
        drift_ranked.zero_()
        node_ranks = torch.empty(
            context.num_nodes, dtype=torch.long, device=grad_w.device
        )
        node_ranks[context.winner[pick]] = context.winner_rank[pick]
        node_ranks[context.loser[pick]] = context.loser_rank[pick]
        drift_ranked.scatter_(0, node_ranks, drift_original)
        drift_norm = torch.linalg.vector_norm(drift_ranked)
        if float(drift_norm.item()) > EPS:
            drift_ranked = drift_ranked / drift_norm
        drifts.append(drift_ranked)

    matrix_mean = torch.stack(matrices).mean(dim=0)
    matrix_norm = torch.linalg.matrix_norm(matrix_mean)
    if float(matrix_norm.item()) > EPS:
        matrix_mean = matrix_mean / matrix_norm
    drift_mean = torch.stack(drifts).mean(dim=0)
    drift_norm = torch.linalg.vector_norm(drift_mean)
    if float(drift_norm.item()) > EPS:
        drift_mean = drift_mean / drift_norm
    return (
        matrix_mean.detach().double().cpu().numpy(),
        drift_mean.detach().double().cpu().numpy(),
    )


def _evaluate_on_context(
    record: Mapping[str, Any],
    context: PairContext,
    *,
    alpha: float,
) -> dict[str, Any]:
    ir = _loss_ir(record)
    expects = list((ir.get("implementation_hint") or {}).get("expects") or [])
    batch = prepare_pairwise_loss_batch(context.full_batch, expects)
    if "log_prob_w" not in batch:
        batch["log_prob_w"] = context.full_batch["log_prob_w"]
    if "log_prob_l" not in batch:
        batch["log_prob_l"] = context.full_batch["log_prob_l"]

    log_prob_w = batch["log_prob_w"].detach().clone().requires_grad_(True)
    log_prob_l = batch["log_prob_l"].detach().clone().requires_grad_(True)
    batch["log_prob_w"] = log_prob_w
    batch["log_prob_l"] = log_prob_l
    if isinstance(batch.get("weight"), torch.Tensor):
        batch["weight"] = batch["weight"].detach()

    generated_loss = _compile_loss(record)
    value = generated_loss(batch, context.full_batch, {"alpha": float(alpha)})
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=log_prob_w.device, dtype=log_prob_w.dtype)
    if value.numel() != 1:
        value = value.mean()
    if not torch.isfinite(value):
        raise ValueError("non-finite loss")

    if not value.requires_grad:
        grad_w = torch.zeros_like(log_prob_w)
        grad_l = torch.zeros_like(log_prob_l)
    else:
        grads = torch.autograd.grad(
            value,
            (log_prob_w, log_prob_l),
            allow_unused=True,
            retain_graph=False,
        )
        grad_w = grads[0] if grads[0] is not None else torch.zeros_like(log_prob_w)
        grad_l = grads[1] if grads[1] is not None else torch.zeros_like(log_prob_l)
    if not torch.isfinite(grad_w).all() or not torch.isfinite(grad_l).all():
        raise ValueError("non-finite pair gradient")

    margin_grad = 0.5 * (grad_w - grad_l)
    shift_grad = 0.5 * (grad_w + grad_l)
    conductance = margin_grad.square()
    margin_mass = conductance.sum()
    shift_mass = shift_grad.square().sum()
    total_pair_energy = margin_mass + shift_mass
    active = (grad_w.abs() + grad_l.abs()) > 1e-12
    matrix, drift = _matrix_and_drift(
        context, conductance, grad_w, grad_l
    )
    return {
        "matrix": matrix,
        "drift": drift,
        "loss_value": float(value.detach().double().cpu().item()),
        "margin_mass": float(margin_mass.detach().double().cpu().item()),
        "shift_leakage": float(
            (shift_mass / total_pair_energy.clamp_min(EPS)).detach().double().cpu().item()
        ),
        "active_fraction": float(active.double().mean().detach().cpu().item()),
    }


def _canonical_bt_matrix(
    context: PairContext,
    *,
    alpha: float,
) -> np.ndarray:
    margin = (
        context.full_batch["log_prob_w"] - context.full_batch["log_prob_l"]
    )
    probability = torch.sigmoid(float(alpha) * margin)
    conductance = probability * (1.0 - probability)
    # Dummy gradients are used only because the shared constructor also emits
    # drift. The canonical matrix itself is independent of a candidate loss.
    zeros = torch.zeros_like(conductance)
    matrix, _ = _matrix_and_drift(context, conductance, zeros, zeros)
    return matrix


def _collect_probes(
    checkpoint: Path,
    *,
    seeds: Sequence[int],
    batches: int,
    batch_size: int,
    num_rollouts: int,
    device: torch.device,
) -> list[Probe]:
    base_spec = build_problem_specs(str(device), batches)["tsp100"]
    hf = replace(base_spec.hf, pomo_size=int(num_rollouts), device=str(device))
    spec = replace(
        base_spec,
        hf=hf,
        batch_size=int(batch_size),
        batches=int(batches),
    )
    env = _rl4co_build_env(spec.hf, 100).to(device)
    policy, rollout_strategy = _rl4co_build_policy(spec.hf, env)
    _load_policy_weights_from_checkpoint(policy, str(checkpoint))
    policy = policy.to(device)
    policy.eval()

    probes: list[Probe] = []
    for seed in seeds:
        for batch_id in range(int(batches)):
            rollout_seed = int(seed) + 1009 * batch_id
            _set_seed(rollout_seed)
            with torch.no_grad():
                reward, log_prob = _rl4co_rollout(
                    env,
                    policy,
                    int(batch_size),
                    int(num_rollouts),
                    phase="train",
                    rollout_strategy=rollout_strategy,
                    device=device,
                    precision=spec.hf.precision,
                    cfg_like=spec.hf,
                )
            objective = _rl4co_objective_from_reward(reward.float(), spec.hf)
            log_prob = log_prob.float()
            reward = reward.float()
            seq_len = torch.full_like(log_prob, 100.0)
            cache = extract_feature_cache(
                objective,
                log_prob,
                extra={
                    "advantage": reward - reward.mean(dim=1, keepdim=True),
                    "seq_len": seq_len,
                    "log_prob_mean": log_prob / seq_len,
                },
            )
            probes.append(Probe(seed=int(seed), batch_id=batch_id, feature_cache=cache))
    return probes


def _aggregate_seed_descriptor(
    candidate_id: str,
    seed: int,
    pieces: Sequence[dict[str, Any]],
) -> SeedDescriptor:
    matrix = np.mean(np.stack([piece["matrix"] for piece in pieces]), axis=0)
    matrix_norm = float(np.linalg.norm(matrix))
    if matrix_norm > EPS:
        matrix = matrix / matrix_norm
    drift = np.mean(np.stack([piece["drift"] for piece in pieces]), axis=0)
    drift = _normalise_rows(drift[None, :])[0]
    spectrum = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    spectrum[np.abs(spectrum) < 1e-12] = 0.0
    return SeedDescriptor(
        candidate_id=candidate_id,
        seed=int(seed),
        matrix=matrix,
        drift=drift,
        spectrum=spectrum,
        log_margin_mass=float(
            np.mean([math.log10(piece["margin_mass"] + EPS) for piece in pieces])
        ),
        shift_leakage=float(np.mean([piece["shift_leakage"] for piece in pieces])),
        active_fraction=float(np.mean([piece["active_fraction"] for piece in pieces])),
        loss_value=float(np.mean([piece["loss_value"] for piece in pieces])),
    )


def _feature_blocks(
    descriptors: Sequence[SeedDescriptor],
    canonical_by_seed: Mapping[int, np.ndarray],
) -> dict[str, np.ndarray]:
    matrices = np.stack([row.matrix.reshape(-1) for row in descriptors])
    matrices = _normalise_rows(matrices)
    drifts = _normalise_rows(np.stack([row.drift for row in descriptors]))
    spectra = _normalise_rows(np.stack([row.spectrum[1:] for row in descriptors]))
    scalar = np.asarray(
        [
            [row.log_margin_mass, row.shift_leakage, row.active_fraction]
            for row in descriptors
        ],
        dtype=np.float64,
    )
    scalar_std = scalar.std(axis=0, keepdims=True)
    scalar = (scalar - scalar.mean(axis=0, keepdims=True)) / np.maximum(
        scalar_std, 1e-8
    )
    canonical = np.stack(
        [canonical_by_seed[row.seed].reshape(-1) for row in descriptors]
    )
    canonical = _normalise_rows(canonical)
    return {
        "canonical_bt_matrix": canonical,
        "loss_laplacian_matrix": matrices,
        "loss_laplacian_spectrum": spectra,
        "signed_node_drift": drifts,
        "matrix_plus_drift": np.concatenate(
            [matrices / math.sqrt(2.0), drifts / math.sqrt(2.0)], axis=1
        ),
        "matrix_drift_and_scale": np.concatenate(
            [
                matrices / math.sqrt(3.0),
                drifts / math.sqrt(3.0),
                scalar / math.sqrt(3.0 * max(scalar.shape[1], 1)),
            ],
            axis=1,
        ),
    }


def _retrieval_metrics(
    features: np.ndarray,
    candidate_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, float]:
    features = np.asarray(features, dtype=np.float64)
    candidate_ids = np.asarray(candidate_ids)
    seeds = np.asarray(seeds)
    ranks: list[float] = []
    top1: list[float] = []
    top5: list[float] = []
    for index in range(len(features)):
        allowed = seeds != seeds[index]
        allowed_indices = np.flatnonzero(allowed)
        distances = np.linalg.norm(features[allowed] - features[index], axis=1)
        order = np.argsort(distances, kind="mergesort")
        ordered_ids = candidate_ids[allowed_indices[order]]
        hits = np.flatnonzero(ordered_ids == candidate_ids[index])
        if hits.size == 0:
            continue
        rank = float(hits[0] + 1)
        ranks.append(rank)
        top1.append(float(rank <= 1))
        top5.append(float(rank <= 5))

    unique_ids = np.unique(candidate_ids)
    means = np.stack([features[candidate_ids == item].mean(axis=0) for item in unique_ids])
    grand = features.mean(axis=0)
    between = float(np.mean(np.sum((means - grand) ** 2, axis=1)))
    within_values = []
    for item, mean in zip(unique_ids, means):
        local = features[candidate_ids == item]
        within_values.extend(np.sum((local - mean) ** 2, axis=1).tolist())
    within = float(np.mean(within_values)) if within_values else float("nan")
    return {
        "queries": float(len(ranks)),
        "top1": float(np.mean(top1)) if top1 else float("nan"),
        "top5": float(np.mean(top5)) if top5 else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "between_variance": between,
        "within_variance": within,
        "between_within_ratio": between / max(within, EPS),
        "chance_top1": 1.0 / max(len(unique_ids), 1),
    }


def _neighbour_overlap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    k: int,
) -> float:
    first = np.asarray(first)
    second = np.asarray(second)
    n = len(first)
    if n <= 1:
        return float("nan")
    k = min(max(int(k), 1), n - 1)
    overlaps = []
    for index in range(n):
        dist_a = np.linalg.norm(first - first[index], axis=1)
        dist_b = np.linalg.norm(second - second[index], axis=1)
        dist_a[index] = np.inf
        dist_b[index] = np.inf
        near_a = set(np.argsort(dist_a)[:k].tolist())
        near_b = set(np.argsort(dist_b)[:k].tolist())
        overlaps.append(len(near_a & near_b) / float(k))
    return float(np.mean(overlaps))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_probe_bank(path: Path, probes: Sequence[Probe]) -> None:
    np.savez_compressed(
        path,
        seed=np.asarray([probe.seed for probe in probes], dtype=np.int64),
        batch_id=np.asarray([probe.batch_id for probe in probes], dtype=np.int64),
        objective=np.stack(
            [probe.feature_cache["objective"].detach().cpu().numpy() for probe in probes]
        ),
        log_prob=np.stack(
            [probe.feature_cache["log_prob"].detach().cpu().numpy() for probe in probes]
        ),
        advantage=np.stack(
            [probe.feature_cache["advantage"].detach().cpu().numpy() for probe in probes]
        ),
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    records_path = args.records.resolve()
    records = _load_records(records_path, args.limit)
    seeds = [int(value) for value in args.seeds]
    if len(seeds) < 2:
        raise ValueError("At least two sampling seeds are required")

    device = torch.device(args.device)
    started = time.time()
    print(
        f"[collect] checkpoint={checkpoint} records={len(records)} seeds={seeds} "
        f"batches={args.batches} batch_size={args.batch_size} K={args.num_rollouts}",
        flush=True,
    )
    probes = _collect_probes(
        checkpoint,
        seeds=seeds,
        batches=args.batches,
        batch_size=args.batch_size,
        num_rollouts=args.num_rollouts,
        device=device,
    )
    _save_probe_bank(output_dir / "probe_bank.npz", probes)
    contexts = {
        (probe.seed, probe.batch_id): _all_pairs_context(probe.feature_cache)
        for probe in probes
    }

    canonical_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        matrices = [
            _canonical_bt_matrix(contexts[(seed, batch_id)], alpha=args.alpha)
            for batch_id in range(args.batches)
        ]
        matrix = np.mean(np.stack(matrices), axis=0)
        norm = np.linalg.norm(matrix)
        canonical_by_seed[seed] = matrix / norm if norm > EPS else matrix

    descriptors: list[SeedDescriptor] = []
    failures: list[dict[str, str]] = []
    for record_id, record in enumerate(records):
        candidate_id = _candidate_id(record)
        print(
            f"[loss {record_id + 1:03d}/{len(records):03d}] {candidate_id}",
            flush=True,
        )
        try:
            for seed in seeds:
                pieces = [
                    _evaluate_on_context(
                        record,
                        contexts[(seed, batch_id)],
                        alpha=args.alpha,
                    )
                    for batch_id in range(args.batches)
                ]
                descriptors.append(
                    _aggregate_seed_descriptor(candidate_id, seed, pieces)
                )
        except Exception as exc:  # keep failures visible
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if failures and not args.allow_failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failures)} loss programs failed; see failures.json")

    candidate_ids = [row.candidate_id for row in descriptors]
    descriptor_seeds = [row.seed for row in descriptors]
    blocks = _feature_blocks(descriptors, canonical_by_seed)
    retrieval = {
        name: _retrieval_metrics(block, candidate_ids, descriptor_seeds)
        for name, block in blocks.items()
    }

    unique_ids = sorted(set(candidate_ids))
    candidate_matrix = np.stack(
        [
            np.mean(
                np.stack(
                    [row.matrix for row in descriptors if row.candidate_id == item]
                ),
                axis=0,
            ).reshape(-1)
            for item in unique_ids
        ]
    )
    candidate_drift = np.stack(
        [
            np.mean(
                np.stack(
                    [row.drift for row in descriptors if row.candidate_id == item]
                ),
                axis=0,
            )
            for item in unique_ids
        ]
    )
    candidate_matrix = _normalise_rows(candidate_matrix)
    candidate_drift = _normalise_rows(candidate_drift)
    neighbour_overlap = _neighbour_overlap(
        candidate_matrix, candidate_drift, k=args.neighbour_k
    )
    random_overlap = min(args.neighbour_k, max(len(unique_ids) - 1, 1)) / max(
        len(unique_ids) - 1, 1
    )

    matrix_metric = retrieval["loss_laplacian_matrix"]
    canonical_metric = retrieval["canonical_bt_matrix"]
    combined_metric = retrieval["matrix_plus_drift"]
    useful = bool(
        matrix_metric["top5"] >= 0.5
        and matrix_metric["between_within_ratio"] > 1.0
        and matrix_metric["top1"] > 3.0 * matrix_metric["chance_top1"]
    )
    standalone_sufficient = bool(
        matrix_metric["top1"] >= combined_metric["top1"] - 1e-12
        and neighbour_overlap >= max(0.5, 2.0 * random_overlap)
    )
    verdict = {
        "matrix_useful_as_sensitivity_descriptor": useful,
        "matrix_sufficient_as_complete_loss_descriptor": standalone_sufficient,
        "recommended_descriptor": (
            "loss_laplacian_matrix"
            if standalone_sufficient
            else "(signed_node_drift, loss_laplacian_matrix)"
        ),
        "reason": (
            "The loss-specific Laplacian is reproducible across sampled probes."
            if useful
            else "The loss-specific Laplacian did not pass the preregistered reproducibility thresholds."
        ),
        "canonical_bt_note": (
            "At a fixed checkpoint and sampler, p(1-p) is shared by all losses; "
            "it cannot distinguish candidate losses without a loss-specific link or precision."
        ),
    }

    descriptor_rows = [
        {
            "candidate_id": row.candidate_id,
            "seed": row.seed,
            "loss_value": row.loss_value,
            "log_margin_mass": row.log_margin_mass,
            "shift_leakage": row.shift_leakage,
            "active_fraction": row.active_fraction,
            "lambda2": float(row.spectrum[1]) if len(row.spectrum) > 1 else 0.0,
            "lambda_max": float(row.spectrum[-1]),
        }
        for row in descriptors
    ]
    _write_csv(output_dir / "descriptor_rows.csv", descriptor_rows)
    np.savez_compressed(
        output_dir / "descriptors.npz",
        candidate_ids=np.asarray(candidate_ids),
        seeds=np.asarray(descriptor_seeds, dtype=np.int64),
        matrices=np.stack([row.matrix for row in descriptors]),
        drifts=np.stack([row.drift for row in descriptors]),
        spectra=np.stack([row.spectrum for row in descriptors]),
        canonical_seeds=np.asarray(seeds, dtype=np.int64),
        canonical_matrices=np.stack([canonical_by_seed[seed] for seed in seeds]),
    )

    result = {
        "protocol": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "records": str(records_path),
            "records_sha256": _sha256(records_path),
            "loss_count_requested": len(records),
            "loss_count_completed": len(unique_ids),
            "seeds": seeds,
            "batches_per_seed": int(args.batches),
            "instances_per_batch": int(args.batch_size),
            "rollouts_per_instance": int(args.num_rollouts),
            "alpha": float(args.alpha),
            "node_alignment": "ascending tour-cost rank",
            "fitness_used": False,
        },
        "retrieval": retrieval,
        "matrix_drift_neighbour_overlap_at_k": {
            "k": int(args.neighbour_k),
            "observed": neighbour_overlap,
            "random_reference": random_overlap,
        },
        "verdict": verdict,
        "failures": failures,
        "wall_time_seconds": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# TSP100 checkpoint-conditioned pair-Laplacian audit",
        "",
        f"- Losses: {len(unique_ids)}/{len(records)}",
        f"- Seeds: {seeds}; batches/seed: {args.batches}; instances/batch: {args.batch_size}",
        f"- Canonical BT matrix top-1 retrieval: {canonical_metric['top1']:.3f}",
        f"- Loss-Laplacian top-1/top-5: {matrix_metric['top1']:.3f}/{matrix_metric['top5']:.3f}",
        f"- Loss-Laplacian between/within ratio: {matrix_metric['between_within_ratio']:.3f}",
        f"- Signed-drift top-1: {retrieval['signed_node_drift']['top1']:.3f}",
        f"- Matrix+drift top-1: {combined_metric['top1']:.3f}",
        f"- Matrix/drift neighbourhood overlap@{args.neighbour_k}: {neighbour_overlap:.3f} "
        f"(random reference {random_overlap:.3f})",
        "",
        f"**Matrix useful:** {useful}",
        f"**Matrix sufficient alone:** {standalone_sufficient}",
        f"**Recommended descriptor:** `{verdict['recommended_descriptor']}`",
        "",
        "The canonical p(1-p) matrix is identical across candidate losses at a fixed "
        "checkpoint and sampler. Candidate discrimination therefore comes from the "
        "loss-specific margin influence, not from p alone.",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2234, 3234])
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--neighbour-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-failures", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
