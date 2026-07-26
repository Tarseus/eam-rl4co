"""Budget-preserving matched-branch layouts and certified two-program loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MatchedBranchLayout:
    """Forced actions for exactly ``rollouts`` complete branches per instance."""

    forced_prefix_actions: torch.Tensor
    group_ids: torch.Tensor
    depth: int
    group_sizes: tuple[int, ...]

    def validate(self, *, problem_size: int) -> None:
        actions = self.forced_prefix_actions
        groups = self.group_ids
        if actions.ndim != 3:
            raise ValueError("forced_prefix_actions must have shape [B, R, depth+1]")
        if groups.shape != actions.shape[:2]:
            raise ValueError("group_ids must have shape [B, R]")
        if actions.shape[-1] != self.depth + 1:
            raise ValueError("forced action length must equal depth+1")
        if actions.dtype != torch.long:
            raise TypeError("forced_prefix_actions must use torch.long")
        if int(actions.min()) < 0 or int(actions.max()) >= problem_size:
            raise ValueError("forced actions must be valid node indices")
        if sum(self.group_sizes) != actions.shape[1]:
            raise ValueError("group sizes must partition the rollout axis")
        sorted_rows = actions.sort(dim=-1).values
        if torch.any(sorted_rows[..., 1:] == sorted_rows[..., :-1]):
            raise ValueError("a TSP forced prefix cannot revisit a node")
        for batch_index in range(actions.shape[0]):
            for group_id, expected_size in enumerate(self.group_sizes):
                mask = groups[batch_index] == group_id
                if int(mask.sum()) != expected_size:
                    raise ValueError("group_ids do not match group_sizes")
                branch = actions[batch_index, mask]
                if self.depth > 0 and not torch.equal(
                    branch[:, : self.depth],
                    branch[:1, : self.depth].expand_as(branch[:, : self.depth]),
                ):
                    raise ValueError("branches in a group must share their prefix")
                if torch.unique(branch[:, self.depth]).numel() != expected_size:
                    raise ValueError("next actions must be distinct within a group")


def _balanced_group_sizes(rollouts: int, capacity: int) -> tuple[int, ...]:
    if rollouts < 2:
        raise ValueError("at least two rollouts are required")
    if capacity < 2:
        raise ValueError("the anchor state must have at least two feasible actions")
    group_count = (rollouts + capacity - 1) // capacity
    if group_count > rollouts // 2:
        raise ValueError("cannot partition rollouts into preference groups of size >=2")
    quotient, remainder = divmod(rollouts, group_count)
    sizes = tuple(
        quotient + int(group_index < remainder)
        for group_index in range(group_count)
    )
    if min(sizes) < 2 or max(sizes) > capacity:
        raise AssertionError("invalid balanced preference-group partition")
    return sizes


def build_tsp_matched_branch_layout(
    *,
    batch_size: int,
    problem_size: int,
    rollouts: int,
    depth: int,
    seed: int,
    device: torch.device | str,
) -> MatchedBranchLayout:
    """Reallocate a standard TSP rollout axis without adding trajectories.

    Prefixes are sampled with a detached CPU generator and therefore require no
    additional policy forward pass. Every branch remains a full TSP trajectory;
    only the first ``depth+1`` actions are forced.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if problem_size < 3:
        raise ValueError("problem_size must be at least three")
    if depth < 0 or depth > problem_size - 2:
        raise ValueError("depth must leave at least two feasible next actions")
    capacity = problem_size - depth
    group_sizes = _balanced_group_sizes(int(rollouts), capacity)
    actions = torch.empty(
        (batch_size, rollouts, depth + 1), dtype=torch.long
    )
    group_ids = torch.empty((batch_size, rollouts), dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for batch_index in range(batch_size):
        offset = 0
        for group_id, group_size in enumerate(group_sizes):
            permutation = torch.randperm(problem_size, generator=generator)
            prefix = permutation[:depth]
            candidates = permutation[depth : depth + group_size]
            branch = torch.cat(
                (
                    prefix[None, :].expand(group_size, -1),
                    candidates[:, None],
                ),
                dim=-1,
            )
            actions[batch_index, offset : offset + group_size] = branch
            group_ids[batch_index, offset : offset + group_size] = group_id
            offset += group_size
    layout = MatchedBranchLayout(
        forced_prefix_actions=actions.to(device),
        group_ids=group_ids.to(device),
        depth=int(depth),
        group_sizes=group_sizes,
    )
    layout.validate(problem_size=problem_size)
    return layout


@dataclass(frozen=True)
class MatchedBranchEdgeContext:
    """Detached inputs visible to the two searched source programs."""

    objective_gap: torch.Tensor
    winner_rank: torch.Tensor
    loser_rank: torch.Tensor
    rank_span: torch.Tensor
    depth_fraction: torch.Tensor
    feasible_fraction: torch.Tensor
    training_progress: torch.Tensor


SelectorProgram = Callable[[MatchedBranchEdgeContext], torch.Tensor]
MarginProgram = Callable[[MatchedBranchEdgeContext], torch.Tensor]


@dataclass(frozen=True)
class MatchedBranchGroupTrace:
    batch_index: int
    group_id: int
    sorted_branch_indices: torch.Tensor
    edge_indices: torch.Tensor
    selected: torch.Tensor
    pressure: torch.Tensor
    boundary: torch.Tensor


@dataclass(frozen=True)
class MatchedBranchLossResult:
    loss: torch.Tensor
    groups: tuple[MatchedBranchGroupTrace, ...]
    skipped_groups: int


def _edge_context(
    normalized_objective: torch.Tensor,
    edge_indices: torch.Tensor,
    *,
    depth: int,
    problem_size: int,
    training_progress: float,
) -> MatchedBranchEdgeContext:
    winner, loser = edge_indices
    denominator = max(len(normalized_objective) - 1, 1)
    edge_count = edge_indices.shape[1]
    dtype = normalized_objective.dtype
    device = normalized_objective.device
    return MatchedBranchEdgeContext(
        objective_gap=(
            normalized_objective[loser] - normalized_objective[winner]
        ).detach(),
        winner_rank=(winner.to(dtype) / denominator).detach(),
        loser_rank=(loser.to(dtype) / denominator).detach(),
        rank_span=((loser - winner).to(dtype) / denominator).detach(),
        depth_fraction=torch.full(
            (edge_count,), depth / problem_size, dtype=dtype, device=device
        ),
        feasible_fraction=torch.full(
            (edge_count,),
            len(normalized_objective) / problem_size,
            dtype=dtype,
            device=device,
        ),
        training_progress=torch.full(
            (edge_count,),
            float(training_progress),
            dtype=dtype,
            device=device,
        ),
    )


def certified_matched_branch_loss(
    local_logp: torch.Tensor,
    objective: torch.Tensor,
    layout: MatchedBranchLayout,
    *,
    selector_program: SelectorProgram,
    margin_program: MarginProgram,
    problem_size: int,
    training_progress: float,
    residual_radius: float = 2.0,
    strict_gap_tolerance: float = 1e-12,
) -> MatchedBranchLossResult:
    """Execute two arbitrary detached programs inside a convex loss wrapper."""

    layout.validate(problem_size=problem_size)
    if local_logp.shape != layout.group_ids.shape:
        raise ValueError("local_logp must have shape [B, R]")
    if objective.shape != local_logp.shape:
        raise ValueError("objective must have shape [B, R]")
    if not torch.isfinite(local_logp).all() or not torch.isfinite(objective).all():
        raise ValueError("local_logp and objective must be finite")
    if residual_radius < 0:
        raise ValueError("residual_radius must be nonnegative")

    group_losses: list[torch.Tensor] = []
    traces: list[MatchedBranchGroupTrace] = []
    skipped = 0
    for batch_index in range(local_logp.shape[0]):
        for group_id in range(len(layout.group_sizes)):
            original_indices = torch.nonzero(
                layout.group_ids[batch_index] == group_id,
                as_tuple=False,
            ).squeeze(-1)
            group_objective = objective[batch_index, original_indices].detach()
            order = torch.argsort(group_objective, stable=True)
            sorted_indices = original_indices[order]
            sorted_objective = group_objective[order]
            objective_range = sorted_objective[-1] - sorted_objective[0]
            if float(objective_range) <= strict_gap_tolerance:
                skipped += 1
                continue
            normalized = (
                sorted_objective - sorted_objective[0]
            ) / objective_range
            sorted_logp = local_logp[batch_index, sorted_indices]
            edge_indices = torch.triu_indices(
                len(sorted_indices),
                len(sorted_indices),
                offset=1,
                device=local_logp.device,
            )
            context = _edge_context(
                normalized,
                edge_indices,
                depth=layout.depth,
                problem_size=problem_size,
                training_progress=training_progress,
            )
            selected_raw = selector_program(context)
            proposed_margin = margin_program(context)
            expected_shape = context.objective_gap.shape
            if selected_raw.shape != expected_shape:
                raise ValueError("selector program returned an invalid shape")
            if proposed_margin.shape != expected_shape:
                raise ValueError("margin program returned an invalid shape")
            if not torch.isfinite(proposed_margin).all():
                raise ValueError("margin program must return finite values")
            selected = selected_raw.detach().to(torch.bool)
            selected = selected & (
                context.objective_gap > strict_gap_tolerance
            )
            if not selected.any():
                skipped += 1
                continue
            detached_margin = proposed_margin.detach().to(sorted_logp.dtype)
            centered = detached_margin - detached_margin[selected].mean()
            target = 2.0 * context.objective_gap.to(sorted_logp.dtype)
            target = target + centered.clamp(
                min=-float(residual_radius), max=float(residual_radius)
            )
            winner, loser = edge_indices
            policy_margin = sorted_logp[winner] - sorted_logp[loser]
            normalizer = int(selected.sum())
            selected_loss = F.softplus(target[selected] - policy_margin[selected])
            group_losses.append(selected_loss.sum() / (4.0 * normalizer))

            pressure = torch.zeros_like(policy_margin)
            pressure[selected] = (
                torch.sigmoid(target[selected] - policy_margin[selected])
                / (4.0 * normalizer)
            )
            boundary = torch.zeros(
                len(sorted_indices) - 1,
                dtype=sorted_logp.dtype,
                device=sorted_logp.device,
            )
            pressure_difference = torch.zeros(
                len(sorted_indices),
                dtype=sorted_logp.dtype,
                device=sorted_logp.device,
            )
            pressure_difference.scatter_add_(0, winner, pressure)
            pressure_difference.scatter_add_(0, loser, -pressure)
            boundary = pressure_difference.cumsum(dim=0)[:-1]
            traces.append(
                MatchedBranchGroupTrace(
                    batch_index=batch_index,
                    group_id=group_id,
                    sorted_branch_indices=sorted_indices.detach(),
                    edge_indices=edge_indices.detach(),
                    selected=selected.detach(),
                    pressure=pressure.detach(),
                    boundary=boundary.detach(),
                )
            )
    if not group_losses:
        raise ValueError("no strict preference edge was selected in the batch")
    return MatchedBranchLossResult(
        loss=torch.stack(group_losses).mean(),
        groups=tuple(traces),
        skipped_groups=skipped,
    )
