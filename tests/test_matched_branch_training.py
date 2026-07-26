from __future__ import annotations

import torch

from rl4co.envs import TSPEnv
from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy
from rl4co.utils.ops import unbatchify
from ptp_discovery.matched_branch_training import (
    MatchedBranchEdgeContext,
    build_tsp_matched_branch_layout,
    certified_matched_branch_loss,
)


def _all_edges(context: MatchedBranchEdgeContext) -> torch.Tensor:
    return torch.ones_like(context.objective_gap, dtype=torch.bool)


def _rank_margin(context: MatchedBranchEdgeContext) -> torch.Tensor:
    return 1.5 * context.rank_span - 0.25


def test_layout_preserves_rollout_budget_and_group_invariants():
    layout = build_tsp_matched_branch_layout(
        batch_size=3,
        problem_size=100,
        rollouts=100,
        depth=75,
        seed=1234,
        device="cpu",
    )
    assert layout.forced_prefix_actions.shape == (3, 100, 76)
    assert layout.group_ids.shape == (3, 100)
    assert layout.group_sizes == (25, 25, 25, 25)
    layout.validate(problem_size=100)


def test_certified_loss_has_raw_b_gradient_identity_bound_and_convex_hessian():
    layout = build_tsp_matched_branch_layout(
        batch_size=1,
        problem_size=8,
        rollouts=4,
        depth=4,
        seed=5,
        device="cpu",
    )
    logp = torch.tensor(
        [[-1.4, -0.7, -2.0, -1.1]], dtype=torch.double, requires_grad=True
    )
    objective = torch.tensor([[3.0, 1.0, 4.0, 2.0]], dtype=torch.double)

    def scalar_loss(values: torch.Tensor) -> torch.Tensor:
        return certified_matched_branch_loss(
            values,
            objective,
            layout,
            selector_program=_all_edges,
            margin_program=_rank_margin,
            problem_size=8,
            training_progress=0.3,
        ).loss

    result = certified_matched_branch_loss(
        logp,
        objective,
        layout,
        selector_program=_all_edges,
        margin_program=_rank_margin,
        problem_size=8,
        training_progress=0.3,
    )
    gradient = torch.autograd.grad(result.loss, logp, create_graph=True)[0]
    trace = result.groups[0]
    count = len(trace.sorted_branch_indices)
    gap = torch.zeros((count - 1, count), dtype=torch.double)
    gap[torch.arange(count - 1), torch.arange(count - 1)] = 1.0
    gap[torch.arange(count - 1), torch.arange(1, count)] = -1.0
    expected_sorted_descent = gap.T @ trace.boundary
    expected_descent = torch.zeros(count, dtype=torch.double)
    expected_descent[trace.sorted_branch_indices] = expected_sorted_descent
    torch.testing.assert_close(-gradient.squeeze(0), expected_descent)
    assert float(trace.boundary.min()) >= 0.0
    assert float(trace.boundary.max()) <= 0.25

    hessian = torch.autograd.functional.hessian(scalar_loss, logp)
    matrix = hessian.reshape(logp.numel(), logp.numel())
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert float(eigenvalues.min()) >= -1e-10


def test_constant_margin_shift_cancels_exactly():
    layout = build_tsp_matched_branch_layout(
        batch_size=1,
        problem_size=8,
        rollouts=4,
        depth=4,
        seed=7,
        device="cpu",
    )
    logp = torch.tensor([[-0.2, -0.5, -1.0, -1.8]], dtype=torch.double)
    objective = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.double)

    def shifted(context: MatchedBranchEdgeContext) -> torch.Tensor:
        return _rank_margin(context) + 19.0

    base = certified_matched_branch_loss(
        logp,
        objective,
        layout,
        selector_program=_all_edges,
        margin_program=_rank_margin,
        problem_size=8,
        training_progress=0.0,
    ).loss
    translated = certified_matched_branch_loss(
        logp,
        objective,
        layout,
        selector_program=_all_edges,
        margin_program=shifted,
        problem_size=8,
        training_progress=0.0,
    ).loss
    torch.testing.assert_close(base, translated)


def test_layout_and_certified_loss_backpropagate_through_one_policy_call():
    env = TSPEnv(generator_params={"num_loc": 8})
    td = env.reset(env.generator(2))
    policy = PO4COPsTSPPolicy(
        env_name=env.name,
        embed_dim=32,
        num_encoder_layers=2,
        decoder_layer_num=1,
        qkv_dim=8,
        num_heads=4,
        feedforward_hidden=64,
    )
    layout = build_tsp_matched_branch_layout(
        batch_size=2,
        problem_size=8,
        rollouts=4,
        depth=4,
        seed=1234,
        device="cpu",
    )
    output = policy(
        td,
        env,
        phase="train",
        num_starts=4,
        return_actions=True,
        return_sum_log_likelihood=False,
        forced_prefix_actions=layout.forced_prefix_actions,
    )
    step_logp = unbatchify(output["log_likelihood"], (0, 4))
    reward = unbatchify(output["reward"], (0, 4))
    actions = unbatchify(output["actions"], (0, 4))
    result = certified_matched_branch_loss(
        step_logp[:, :, layout.depth],
        -reward,
        layout,
        selector_program=_all_edges,
        margin_program=_rank_margin,
        problem_size=8,
        training_progress=0.2,
    )
    result.loss.backward()

    torch.testing.assert_close(
        actions[:, :, : layout.depth + 1],
        layout.forced_prefix_actions,
    )
    expected = torch.arange(8).expand_as(actions)
    torch.testing.assert_close(actions.sort(dim=-1).values, expected)
    gradient_l1 = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in policy.parameters()
        if parameter.grad is not None
    )
    assert gradient_l1 > 0.0
