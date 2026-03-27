import torch

from rl4co.models.zoo.mgl_jssp.sampling import (
    Solutions,
    po_loss,
    rl_loss,
    trajectory_log_probs,
)


def test_trajectory_log_probs_sums_selected_action_log_probs() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            [
                [1.0, 1.0],
                [3.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    trajs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    actual = trajectory_log_probs(logits, trajs)
    expected = torch.log_softmax(logits[0, 0], dim=-1)[0]
    expected += torch.log_softmax(logits[0, 1], dim=-1)[1]
    expected_1 = torch.log_softmax(logits[1, 0], dim=-1)[1]
    expected_1 += torch.log_softmax(logits[1, 1], dim=-1)[0]

    assert torch.allclose(actual, torch.stack([expected, expected_1]))


def test_rl_and_po_losses_are_finite_and_return_quality_ratio() -> None:
    samples = Solutions(
        mss=torch.tensor([10.0, 12.0, 15.0], dtype=torch.float32),
        logits=torch.tensor(
            [
                [[3.0, 0.0], [2.0, 0.0]],
                [[1.5, 0.5], [1.0, 0.0]],
                [[0.5, 1.5], [0.0, 2.0]],
            ],
            dtype=torch.float32,
        ),
        trajs=torch.tensor([[0, 0], [0, 0], [1, 1]], dtype=torch.long),
    )

    rl_loss_value, rl_ratio = rl_loss(samples)
    po_loss_value, po_ratio = po_loss(samples)

    assert torch.isfinite(rl_loss_value)
    assert torch.isfinite(po_loss_value)
    assert rl_ratio == 1.5
    assert 0.0 <= po_ratio <= 1.0
