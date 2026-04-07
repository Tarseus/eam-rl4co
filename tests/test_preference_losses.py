import pytest
import torch
import torch.nn.functional as F

from rl4co.models.rl.reinforce.preference_losses import bopo_loss, pl_loss, po_loss, sll_loss


def _ptp_po_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, alpha: float):
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp_pair = alpha * (log_likelihood[:, :, None] - log_likelihood[:, None, :])
    pf_log = torch.log(torch.sigmoid(logp_pair))
    return -(pf_log * preference).mean()


def _exp_po_loss(reward: torch.Tensor, log_likelihood: torch.Tensor, alpha: float):
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp_pair = alpha * (log_likelihood[:, :, None] - log_likelihood[:, None, :])
    return -(logp_pair * preference).mean()


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


def _paper_bopo_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float,
    k: int,
    sequence_length: float,
):
    sorted_idx = torch.argsort(reward, descending=True, dim=-1)
    batch_size, pool_size = reward.shape
    idx = sorted_idx[:, :: pool_size // k]
    bs_idx = torch.arange(batch_size).view(-1, 1).expand(-1, k)
    mean_log_prob = log_likelihood / sequence_length
    objective = -reward
    better_obj = objective[bs_idx[:, [0]], idx[:, [0]]]
    worse_obj = objective[bs_idx[:, 1:], idx[:, 1:]]
    guidance = worse_obj / better_obj
    better_score = alpha * mean_log_prob[torch.arange(batch_size), idx[:, 0]]
    worse_score = alpha * mean_log_prob.gather(1, idx[:, 1:])
    pair_score = guidance * (better_score.unsqueeze(1) - worse_score)
    return -F.logsigmoid(pair_score).mean()


def _bopo_all_pairs_reference(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float,
    selected_idx: torch.Tensor,
):
    objective = -reward
    losses = []
    for b in range(reward.size(0)):
        selected = selected_idx[b]
        selected_obj = objective[b, selected]
        selected_logp = alpha * log_likelihood[b, selected]
        for i in range(selected.numel()):
            for j in range(i + 1, selected.numel()):
                factor = selected_obj[j] / selected_obj[i]
                losses.append(-F.logsigmoid(factor * (selected_logp[i] - selected_logp[j])))
    return torch.stack(losses).mean()


def _listmle_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float,
    temperature: float,
):
    sorted_idx = reward.sort(dim=1, descending=True).indices
    scores = (alpha * log_likelihood / temperature).gather(1, sorted_idx)
    suffix_logsumexp = torch.logcumsumexp(scores.flip(dims=[1]), dim=1).flip(dims=[1])
    return -(scores - suffix_logsumexp).mean()


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


def test_po_loss_matches_exponential_formula():
    torch.manual_seed(0)
    reward = torch.randn(4, 6)
    log_likelihood = torch.randn(4, 6)
    alpha = 1.2
    ref = _exp_po_loss(reward, log_likelihood, alpha)
    loss, pref_rate = po_loss(
        reward, log_likelihood, alpha=alpha, impl="exponential"
    )
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


def test_bopo_loss_matches_paper_anchor_best_reference():
    reward = torch.tensor(
        [
            [-10.0, -20.0, -12.0, -40.0, -15.0, -25.0, -17.0, -30.0],
            [-8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0],
        ]
    )
    log_likelihood = torch.tensor(
        [
            [-1.0, -2.2, -1.1, -3.0, -1.3, -2.4, -1.5, -2.8],
            [-0.9, -1.0, -1.2, -1.3, -1.5, -1.8, -2.0, -2.2],
        ]
    )
    alpha = 1.0
    k = 4
    seq_len = 5.0

    ref = _paper_bopo_loss(reward, log_likelihood, alpha, k, seq_len)
    loss, pair_count = bopo_loss(
        reward,
        log_likelihood,
        alpha=alpha,
        pair_mode="anchor_best",
        select_strategy="paper",
        select_k=k,
        sequence_length=seq_len,
    )

    assert torch.isfinite(loss)
    assert torch.allclose(loss, ref, atol=1e-6)
    assert pair_count.item() == reward.size(0) * (k - 1)


def test_bopo_loss_paper_strategy_requires_even_stride():
    reward = -torch.arange(10, dtype=torch.float32).view(1, -1)
    log_likelihood = torch.randn_like(reward)

    with pytest.raises(ValueError, match="num_pomo % select_k == 0"):
        bopo_loss(
            reward,
            log_likelihood,
            select_strategy="paper",
            select_k=6,
        )


def test_bopo_loss_all_pairs_matches_reference():
    reward = torch.tensor([[-10.0, -12.0, -20.0, -40.0]])
    log_likelihood = torch.tensor([[-1.0, -1.3, -2.1, -3.4]])
    selected_idx = torch.tensor([[0, 1, 2, 3]])

    ref = _bopo_all_pairs_reference(reward, log_likelihood, alpha=1.0, selected_idx=selected_idx)
    loss, pair_count = bopo_loss(
        reward,
        log_likelihood,
        alpha=1.0,
        pair_mode="all_pairs",
        select_strategy="top_k",
        select_k=4,
    )

    assert torch.allclose(loss, ref, atol=1e-6)
    assert pair_count.item() == 6.0


def test_bopo_loss_quantile_selection_is_real_quantile():
    reward = torch.tensor([[-1.0, -2.0, -100.0, -101.0]])
    log_likelihood = torch.tensor([[-0.1, -0.2, -5.0, -5.2]])

    loss_q, count_q = bopo_loss(
        reward,
        log_likelihood,
        select_strategy="quantile",
        select_quantile=0.75,
        pair_mode="anchor_best",
    )
    loss_topk, count_topk = bopo_loss(
        reward,
        log_likelihood,
        select_strategy="top_k",
        select_k=2,
        pair_mode="anchor_best",
    )

    assert torch.isfinite(loss_q)
    assert count_q.item() >= 1.0
    assert not torch.allclose(loss_q, loss_topk) or count_q.item() != count_topk.item()


def test_bopo_loss_rejects_unknown_pair_mode():
    reward = torch.tensor([[-1.0, -2.0]])
    log_likelihood = torch.tensor([[-0.1, -0.2]])

    with pytest.raises(ValueError, match="pair_mode"):
        bopo_loss(reward, log_likelihood, pair_mode="bad_mode")


def test_sll_loss_matches_listmle_reference():
    reward = torch.tensor(
        [
            [3.0, 1.0, 2.0, 0.0],
            [0.5, 0.2, 0.1, -0.3],
        ]
    )
    log_likelihood = torch.tensor(
        [
            [0.2, -0.6, 0.1, -1.0],
            [1.5, 0.3, -0.2, -0.7],
        ]
    )
    alpha = 1.3
    temperature = 0.7

    ref = _listmle_loss(reward, log_likelihood, alpha, temperature)
    actual = sll_loss(
        reward,
        log_likelihood,
        alpha=alpha,
        impl="sll",
        temperature=temperature,
    )

    assert torch.isfinite(actual)
    assert torch.allclose(actual, ref, atol=1e-6)


def test_sll_loss_prefers_scores_aligned_with_reward_order():
    reward = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    aligned_scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    misaligned_scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    aligned_loss = sll_loss(reward, aligned_scores, impl="sll")
    misaligned_loss = sll_loss(reward, misaligned_scores, impl="sll")

    assert aligned_loss < misaligned_loss


def test_sll_loss_rejects_bad_impl_and_temperature():
    reward = torch.tensor([[1.0, 0.0]])
    log_likelihood = torch.tensor([[0.1, -0.2]])

    with pytest.raises(ValueError, match="Unknown sll_loss impl"):
        sll_loss(reward, log_likelihood, impl="bad")
    with pytest.raises(ValueError, match="temperature"):
        sll_loss(reward, log_likelihood, impl="sll", temperature=0.0)


def test_sll_aux_variants_prefer_better_alignment():
    reward = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    aligned_scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    misaligned_scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    for impl in ("slim", "listnet"):
        aligned_loss = sll_loss(reward, aligned_scores, impl=impl)
        misaligned_loss = sll_loss(reward, misaligned_scores, impl=impl)
        assert aligned_loss < misaligned_loss
