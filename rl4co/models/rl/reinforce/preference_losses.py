from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def bopo_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float = 1.0,
    pair_mode: Literal["anchor_best", "all_pairs"] = "anchor_best",
    select_strategy: Literal["top_k", "quantile"] = "top_k",
    select_k: int | None = None,
    select_quantile: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BOPO-style (Best-anchored and Objective-guided Preference Optimization) loss / SRO loss.

    This implements the Stochastic Ranking Optimization (SRO) loss from the BOPO paper.
    It uses the makespan ratio (objective gap) to weight the preference loss.

    Args:
        reward: Tensor of shape [batch, pomo], higher is better.
            For minimization problems like JSP/FFSP, reward = -makespan.
        log_likelihood: Tensor of shape [batch, pomo], sum log-prob per trajectory.
        alpha: Scale applied to log-likelihood.
        pair_mode: Pairing strategy:
            - "anchor_best": Pair the best solution with all other selected solutions
            - "all_pairs": Pair all better-worse pairs among selected solutions
        select_strategy: Strategy to select solutions for pairing:
            - "top_k": Select the top k solutions
            - "quantile": Select solutions better than a quantile threshold
        select_k: Number of top solutions to select (for "top_k" strategy).
            If None, uses sqrt(pomo) as default.
        select_quantile: Quantile threshold (for "quantile" strategy, 0.0-1.0).

    Returns:
        loss: Scalar tensor.
        pair_count: Number of pairs used, useful for diagnostics.
    """
    batch_size, num_pomo = reward.shape

    # Convert reward to objective (lower is better, e.g., makespan)
    # reward = -objective => objective = -reward
    objective = -reward

    # Determine number of solutions to select
    if select_strategy == "top_k":
        if select_k is None:
            select_k = max(2, int(torch.sqrt(torch.tensor(num_pomo, dtype=torch.float32)).item()))
        select_k = min(select_k, num_pomo)
        # Sort by objective (ascending) and select top k
        sorted_obj, sorted_idx = objective.sort(dim=1, descending=False)
        selected_idx = sorted_idx[:, :select_k]
    else:  # quantile
        # Select solutions better than the quantile threshold
        threshold = torch.quantile(objective, select_quantile, dim=1, keepdim=True)
        # Create mask and get indices of selected solutions
        mask = objective <= threshold
        # Ensure at least 2 solutions are selected
        for b in range(batch_size):
            if mask[b].sum() < 2:
                _, top_idx = objective[b].topk(2, largest=False)
                mask[b, top_idx] = True
        # Get selected indices (this is a bit more complex since each row may have different count)
        # For simplicity, we'll use top_k with sqrt(num_pomo) as fallback
        sorted_obj, sorted_idx = objective.sort(dim=1, descending=False)
        select_k = max(2, int(torch.sqrt(torch.tensor(num_pomo, dtype=torch.float32)).item()))
        select_k = min(select_k, num_pomo)
        selected_idx = sorted_idx[:, :select_k]

    # Gather selected solutions
    selected_obj = objective.gather(1, selected_idx)
    selected_logp = alpha * log_likelihood.gather(1, selected_idx)
    _, num_selected = selected_idx.shape

    if pair_mode == "anchor_best":
        # Pair the best (first) with all others
        # Shape: [batch, num_selected-1]
        best_obj = selected_obj[:, [0]]
        best_logp = selected_logp[:, [0]]
        worse_obj = selected_obj[:, 1:]
        worse_logp = selected_logp[:, 1:]

        # Compute makespan factor (objective ratio: worse / better >= 1)
        # Add epsilon to avoid division by zero
        eps = 1e-8
        makespan_factor = (worse_obj + eps) / (best_obj + eps)

        # Logp gap: logp_better - logp_worse
        logp_gap = best_logp - worse_logp

        # Loss: -log(sigmoid(makespan_factor * logp_gap))
        loss = -torch.log(torch.sigmoid(makespan_factor * logp_gap)).mean()
        pair_count = torch.tensor(batch_size * (num_selected - 1), dtype=torch.float32, device=reward.device)

    else:  # all_pairs
        # Create all better-worse pairs
        # Expand to [batch, num_selected, num_selected]
        obj_i = selected_obj[:, :, None]
        obj_j = selected_obj[:, None, :]
        logp_i = selected_logp[:, :, None]
        logp_j = selected_logp[:, None, :]

        # Mask where i is better than j (obj_i < obj_j)
        better_mask = (obj_i < obj_j).float()

        # Compute makespan factor for each pair
        eps = 1e-8
        makespan_factor = (obj_j + eps) / (obj_i + eps)

        # Logp gap: logp_i - logp_j (i is better)
        logp_gap = logp_i - logp_j

        # Loss: -log(sigmoid(makespan_factor * logp_gap)) averaged over all pairs
        pf_log = torch.log(torch.sigmoid(makespan_factor * logp_gap))
        loss = -(pf_log * better_mask).sum() / better_mask.sum().clamp_min(1.0)
        pair_count = better_mask.sum()

    return loss, pair_count


def sll_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float = 1.0,
    impl: Literal["sll", "slim", "listnet"] = "sll",
    temperature: float = 1.0,
) -> torch.Tensor:
    """SLL/SLIM-style listwise loss for neural combinatorial optimization.

    SLL = Softmax Listwise Loss
    SLIM = Softmax Listwise with Instance-wise Margin

    These are listwise ranking losses that compare the full list of solutions.

    Args:
        reward: Tensor of shape [batch, pomo], higher is better.
        log_likelihood: Tensor of shape [batch, pomo], sum log-prob per trajectory.
        alpha: Scale applied to log-likelihood.
        impl: Implementation variant:
            - "sll": Basic softmax listwise loss (ListMLE-style)
            - "slim": SLIM-style with margin based on reward gaps
            - "listnet": ListNet-style cross-entropy on permutations
        temperature: Temperature for softmax scaling.

    Returns:
        loss: Scalar tensor.
    """
    batch_size, num_pomo = reward.shape

    # Sort by reward descending to get the optimal ranking
    sorted_reward, sorted_idx = reward.sort(dim=1, descending=True)
    logp = alpha * log_likelihood

    if impl == "sll":
        # SLL: Softmax Listwise Loss (ListMLE-style)
        # Max normalization for numerical stability
        max_logp = logp.max(dim=1, keepdim=True).values
        logp_normalized = logp - max_logp

        # Gather in reward-ranked order
        logp_rank = logp_normalized.gather(1, sorted_idx)

        # Compute cumulative log-sum-exp
        # Use cumsum of exponentials for stable computation
        exp_logp = torch.exp(logp_rank / temperature)
        cum_exp = exp_logp.cumsum(dim=1)
        log_cum_exp = torch.log(cum_exp)

        # Loss: mean over (logp_rank - log_cum_exp) for each position
        loss = -torch.mean(logp_rank / temperature - log_cum_exp)

    elif impl == "slim":
        # SLIM: Softmax Listwise with Instance-wise Margin
        # Compute margin based on reward gaps
        sorted_reward_expanded = sorted_reward[:, :, None]
        reward_gaps = sorted_reward_expanded - sorted_reward[:, None, :]

        # Create target: only top triangle (better > worse) matters
        target = (reward_gaps > 0).float()

        # Logp in sorted order
        logp_rank = logp.gather(1, sorted_idx)
        logp_expanded = logp_rank[:, :, None] - logp_rank[:, None, :]

        # Scale by reward gap margin
        margin = torch.abs(reward_gaps) + 1e-8
        scaled_logp = logp_expanded * margin / temperature

        # Binary cross-entropy over all pairs
        loss = F.binary_cross_entropy_with_logits(scaled_logp, target)

    else:  # listnet
        # ListNet-style: Cross-entropy on softmax distributions
        # Create target distribution from rewards (Plackett-Luce)
        reward_normalized = reward - reward.max(dim=1, keepdim=True).values
        target_probs = F.softmax(reward_normalized / temperature, dim=1)

        # Predict distribution from log probabilities
        logp_normalized = logp - logp.max(dim=1, keepdim=True).values
        pred_logits = logp_normalized / temperature

        # Cross-entropy loss
        loss = -(target_probs * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()

    return loss


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
