import pytest
import torch

from rl4co.models.rl.reinforce.preference_losses import slim_loss


def test_slim_selects_best_rollout_strictly_per_instance_and_normalizes_length():
    reward = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])
    log_likelihood = torch.tensor(
        [[-4.0, -2.0, -3.0], [-1.0, -2.0, -5.0]], requires_grad=True
    )
    lengths = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])

    loss = slim_loss(reward, log_likelihood, sequence_length=lengths)

    assert loss.item() == pytest.approx((1.0 + 1.25) / 2.0)
    loss.backward()
    expected_grad = torch.tensor([[0.0, -0.25, 0.0], [0.0, 0.0, -0.125]])
    assert torch.equal(log_likelihood.grad, expected_grad)


def test_slim_uses_first_sample_for_tied_best_reward():
    reward = torch.tensor([[2.0, 2.0, 1.0]])
    log_likelihood = torch.tensor([[-3.0, -1.0, -4.0]], requires_grad=True)

    loss = slim_loss(reward, log_likelihood, sequence_length=2)
    loss.backward()

    assert loss.item() == pytest.approx(1.5)
    assert torch.equal(log_likelihood.grad, torch.tensor([[-0.5, 0.0, 0.0]]))


@pytest.mark.parametrize(
    ("reward", "log_likelihood"),
    [
        (torch.ones(3), torch.ones(3)),
        (torch.ones(2, 3), torch.ones(2, 4)),
        (torch.empty(0, 3), torch.empty(0, 3)),
    ],
)
def test_slim_rejects_invalid_pool_shapes(reward, log_likelihood):
    with pytest.raises(ValueError):
        slim_loss(reward, log_likelihood)


def test_slim_rejects_nonpositive_or_unbroadcastable_lengths():
    reward = torch.ones(2, 3)
    log_likelihood = torch.ones(2, 3)
    with pytest.raises(ValueError, match="strictly positive"):
        slim_loss(reward, log_likelihood, sequence_length=0)
    with pytest.raises(ValueError, match="broadcastable"):
        slim_loss(reward, log_likelihood, sequence_length=torch.ones(4))
