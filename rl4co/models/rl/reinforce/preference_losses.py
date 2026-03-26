from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def po_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float = 1.0,
    impl: Literal["bt", "exponential"] = "bt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise preference loss used in PO-style POMO training.

    Args:
        reward: Tensor of shape [batch, pomo], higher is better.
        log_likelihood: Tensor of shape [batch, pomo], sum log-prob per trajectory.
        alpha: Scale applied to log-likelihood.
        impl: Pairwise preference function.
            - "bt": Bradley-Terry / logistic preference, using ``logsigmoid``.
            - "exponential": Exponential preference from the PO4COPs paper,
              using the raw scaled log-prob gap.

    Returns:
        loss: Scalar tensor.
        pref_rate: Mean of preference matrix, useful for diagnostics.
    """
    if impl not in {"bt", "exponential"}:
        raise ValueError(f"Unknown po_loss impl: {impl}")

    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp = alpha * log_likelihood
    logp_pair = logp[:, :, None] - logp[:, None, :]
    if impl == "bt":
        pf_log = F.logsigmoid(logp_pair)
    else:
        pf_log = logp_pair
    loss = -(pf_log * preference).mean()
    pref_rate = preference.mean()
    return loss, pref_rate


def pl_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float = 1.0,
    impl: Literal["ptp", "stable"] = "stable",
) -> torch.Tensor:
    """Listwise (Plackett-Luce-style) ranking loss aligned with PTP.

    Args:
        reward: Tensor of shape [batch, pomo], higher is better.
        log_likelihood: Tensor of shape [batch, pomo], sum log-prob per trajectory.
        alpha: Scale applied to log-likelihood.
        impl: "ptp" reproduces the original PTP one-hot + tril formulation.
              "stable" uses an equivalent cumulative sum without [B, P, P] tensors.

    Returns:
        loss: Scalar tensor.
    """
    if impl not in {"ptp", "stable"}:
        raise ValueError(f"Unknown pl_loss impl: {impl}")

    sorted_idx = reward.sort(dim=1, descending=True).indices
    logp = alpha * log_likelihood
    # Match the historical PTP formulation exactly (see `tests/test_preference_losses.py::_ptp_pl_loss`):
    # - Normalize logp by a per-row max in the *original* order
    # - Build denominators as prefix sums over the reward-ranked permutation
    # - Use the original-order numerator `log(exp_logp)` (even though denominators are rank-ordered)
    max_logp = logp.max(dim=1, keepdim=True).values
    logp = logp - max_logp
    exp_logp = torch.exp(logp)
    exp_logp_rank = exp_logp.gather(1, sorted_idx)

    if impl == "ptp":
        one_hot = F.one_hot(sorted_idx, num_classes=reward.size(1)).to(exp_logp.dtype)
        till_mat = torch.tril(torch.ones_like(one_hot))
        sum_exp = (till_mat @ one_hot @ exp_logp.unsqueeze(-1)).squeeze(-1)
    else:
        sum_exp = exp_logp_rank.cumsum(dim=1)

    loss = torch.mean(torch.log(exp_logp) - torch.log(sum_exp))
    return loss
