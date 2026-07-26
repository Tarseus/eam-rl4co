from __future__ import annotations

import torch

from fitness.ptp_high_fidelity import HighFidelityConfig
from ptp_discovery.tangent_branch_probe import (
    _collect_tangent_policy_state,
    parameter_tangent_influence,
)


def test_parameter_tangent_influence_matches_direct_gradient_inner_product():
    weight = torch.tensor(
        [[0.3, -0.7], [1.2, 0.5]], dtype=torch.double, requires_grad=True
    )
    left = torch.tensor([0.4, -1.1], dtype=torch.double)
    hidden = weight @ left
    local = torch.stack((hidden[0] + hidden[1], hidden[0] - hidden[1]))
    terminal = torch.stack((2.0 * hidden[0], -0.5 * hidden[1]))
    target = torch.softmax(local, dim=0) @ torch.tensor(
        [0.0, 1.0], dtype=torch.double
    )
    local_influence, terminal_influence = parameter_tangent_influence(
        (local, terminal), target, (weight,)
    )

    local_coefficient = torch.tensor([0.2, -0.8], dtype=torch.double)
    terminal_coefficient = torch.tensor([1.1, 0.4], dtype=torch.double)
    candidate = (
        local_coefficient @ local + terminal_coefficient @ terminal
    )
    target_gradient = torch.autograd.grad(target, weight, retain_graph=True)[0]
    candidate_gradient = torch.autograd.grad(candidate, weight)[0]
    direct = (target_gradient * candidate_gradient).sum()
    tangent = (
        local_influence @ local_coefficient
        + terminal_influence @ terminal_coefficient
    )
    torch.testing.assert_close(tangent, direct)


def test_small_tsp_tangent_collector_returns_finite_nonzero_influence():
    config = HighFidelityConfig(
        problem="tsp",
        backend="rl4co",
        env_name="tsp",
        generator_params={"num_loc": 8},
        policy_name="pomo",
        policy_kwargs={
            "po4cops_compat": True,
            "embed_dim": 32,
            "num_encoder_layers": 2,
            "decoder_layer_num": 1,
            "qkv_dim": 8,
            "num_heads": 4,
            "feedforward_hidden": 64,
            "tanh_clipping": 10,
            "eval_type": "argmax",
        },
        rollout_strategy="auto",
        objective_sign="neg_reward",
        train_problem_size=8,
        valid_problem_sizes=(8,),
        train_batch_size=2,
        pomo_size=8,
        device="cpu",
        seed=1234,
    )
    probes = _collect_tangent_policy_state(
        config,
        policy_state="early",
        checkpoint_path=None,
        instances=2,
        branch_count=3,
        depths=(2,),
        device="cpu",
    )
    assert len(probes) == 1
    probe = probes[0]
    probe.validate()
    assert probe.local_target_influence.shape == (2, 3)
    assert probe.terminal_target_influence.shape == (2, 3)
    assert float(probe.local_target_influence.abs().sum()) > 0.0
    assert float(probe.terminal_target_influence.abs().sum()) > 0.0
