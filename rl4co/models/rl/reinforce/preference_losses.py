from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def po_loss(
    reward: torch.Tensor, log_likelihood: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise BT-style preference loss used in PTP POMO.

    Args:
        reward: Tensor of shape [batch, pomo], higher is better.
        log_likelihood: Tensor of shape [batch, pomo], sum log-prob per trajectory.
        alpha: Scale applied to log-likelihood.

    Returns:
        loss: Scalar tensor.
        pref_rate: Mean of preference matrix, useful for diagnostics.
    """
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp = alpha * log_likelihood
    logp_pair = logp[:, :, None] - logp[:, None, :]
    pf_log = F.logsigmoid(logp_pair)
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
    logp_sorted = logp.gather(1, sorted_idx)
    max_logp = logp_sorted.max(dim=1, keepdim=True).values
    logp_sorted = logp_sorted - max_logp
    exp_logp = torch.exp(logp_sorted)

    if impl == "ptp":
        one_hot = F.one_hot(sorted_idx, num_classes=reward.size(1)).to(exp_logp.dtype)
        till_mat = torch.tril(torch.ones_like(one_hot))
        sum_exp = (till_mat @ one_hot @ exp_logp.unsqueeze(-1)).squeeze(-1)
    else:
        sum_exp = exp_logp.cumsum(dim=1)

    loss = torch.mean(torch.log(exp_logp) - torch.log(sum_exp))
    return loss
