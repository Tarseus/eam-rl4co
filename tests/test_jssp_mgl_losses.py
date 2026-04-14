import torch
import torch.nn.functional as F

from rl4co.models.zoo.mgl_jssp.sampling import (
    Solutions,
    po_loss,
    rl_loss,
    trajectory_log_probs,
)


def _reference_rl_loss(samples: Solutions) -> tuple[torch.Tensor, float]:
    log_probs = trajectory_log_probs(samples.logits, samples.trajs)
    rewards = -samples.mss
    advantage = rewards - rewards.mean()
    advantage = advantage / (advantage.std(unbiased=False) + 1e-8)
    loss = -(advantage.detach() * log_probs).mean()
    best = samples.mss.min().clamp_min(1e-8)
    worst = samples.mss.max().clamp_min(1e-8)
    return loss, float((worst / best).item())


def _reference_original_jssp_po_loss(
    samples: Solutions, alpha: float = 1.0
) -> tuple[torch.Tensor, float]:
    log_probs = trajectory_log_probs(samples.logits, samples.trajs)
    makespans = samples.mss
    pair_mask = torch.triu(
        torch.ones((makespans.shape[0], makespans.shape[0]), dtype=torch.bool), diagonal=1
    )
    if not pair_mask.any():
        best = makespans.min().clamp_min(1e-8)
        worst = makespans.max().clamp_min(1e-8)
        return log_probs.sum() * 0.0, float((worst / best).item())

    left_idx, right_idx = pair_mask.nonzero(as_tuple=True)
    left_ms = makespans[left_idx]
    right_ms = makespans[right_idx]
    unequal = left_ms != right_ms
    if not unequal.any():
        best = makespans.min().clamp_min(1e-8)
        worst = makespans.max().clamp_min(1e-8)
        return log_probs.sum() * 0.0, float((worst / best).item())

    left_idx = left_idx[unequal]
    right_idx = right_idx[unequal]
    left_ms = left_ms[unequal]
    right_ms = right_ms[unequal]

    better_is_left = left_ms < right_ms
    better_idx = torch.where(better_is_left, left_idx, right_idx)
    worse_idx = torch.where(better_is_left, right_idx, left_idx)
    score_diff = alpha * (log_probs[better_idx] - log_probs[worse_idx])
    loss = -F.logsigmoid(score_diff).mean()
    best = makespans.min().clamp_min(1e-8)
    worst = makespans.max().clamp_min(1e-8)
    return loss, float((worst / best).item())


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
    ref_rl_loss_value, ref_rl_ratio = _reference_rl_loss(samples)
    ref_po_loss_value, ref_po_ratio = _reference_original_jssp_po_loss(samples)

    assert torch.isfinite(rl_loss_value)
    assert torch.isfinite(po_loss_value)
    torch.testing.assert_close(rl_loss_value, ref_rl_loss_value)
    torch.testing.assert_close(po_loss_value, ref_po_loss_value)
    assert rl_ratio == ref_rl_ratio == 1.5
    assert po_ratio == ref_po_ratio == 1.5


def test_po_loss_alpha_scales_pairwise_margin() -> None:
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

    po_loss_value, po_ratio = po_loss(samples, alpha=0.25)
    ref_po_loss_value, ref_po_ratio = _reference_original_jssp_po_loss(samples, alpha=0.25)

    assert torch.isfinite(po_loss_value)
    torch.testing.assert_close(po_loss_value, ref_po_loss_value)
    assert po_ratio == ref_po_ratio == 1.5
