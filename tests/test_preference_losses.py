import torch
import torch.nn.functional as F

from rl4co.models.rl.reinforce.preference_losses import pl_loss, po_loss


def _ptp_po_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, alpha: float):
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp_pair = alpha * (log_likelihood[:, :, None] - log_likelihood[:, None, :])
    pf_log = torch.log(torch.sigmoid(logp_pair))
    return -(pf_log * preference).mean()


def _ptp_pl_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, alpha: float):
    sorted_idx = reward.sort(dim=1, descending=True).indices
    logp = alpha * log_likelihood
    max_logp = logp.max(1, keepdim=True).values
    logp = logp - max_logp
    exp_logp = torch.exp(logp)
    one_hot = F.one_hot(sorted_idx, num_classes=reward.size(1)).float()
    till_mat = torch.tril(torch.ones_like(one_hot))
    sum_exp = (till_mat @ one_hot @ exp_logp.unsqueeze(-1)).squeeze(-1)
    return torch.mean(torch.log(exp_logp) - torch.log(sum_exp))


def test_po_loss_matches_ptp_formula():
    torch.manual_seed(0)
    reward = torch.randn(4, 6)
    log_likelihood = torch.randn(4, 6)
    alpha = 1.2
    ref = _ptp_po_loss(reward, log_likelihood, alpha)
    loss, pref_rate = po_loss(reward, log_likelihood, alpha=alpha)
    assert torch.isfinite(loss)
    assert torch.isfinite(pref_rate)
    assert torch.allclose(loss, ref, atol=1e-6)


def test_po_loss_sensitive_to_reward_order():
    reward = torch.tensor([[1.0, 0.0, -1.0]])
    log_likelihood = torch.tensor([[0.1, -0.2, 0.3]])
    loss_1, _ = po_loss(reward, log_likelihood, alpha=1.0)
    loss_2, _ = po_loss(reward[:, [1, 0, 2]], log_likelihood, alpha=1.0)
    assert torch.abs(loss_1 - loss_2) > 1e-6


def test_pl_loss_matches_ptp_and_stable_impl():
    torch.manual_seed(42)
    reward = torch.randn(3, 5)
    log_likelihood = torch.randn(3, 5)
    alpha = 0.7
    ref = _ptp_pl_loss(reward, log_likelihood, alpha)
    loss_ptp = pl_loss(reward, log_likelihood, alpha=alpha, impl="ptp")
    loss_stable = pl_loss(reward, log_likelihood, alpha=alpha, impl="stable")
    assert torch.isfinite(loss_ptp)
    assert torch.isfinite(loss_stable)
    assert torch.allclose(loss_ptp, ref, atol=1e-6)
    assert torch.allclose(loss_stable, ref, atol=1e-6)
