from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def bopo_loss(
    reward: torch.Tensor,
    log_likelihood: torch.Tensor,
    alpha: float = 1.0,
    pair_mode: Literal["anchor_best", "all_pairs"] = "anchor_best",
    select_strategy: Literal["paper", "top_k", "quantile"] = "paper",
    select_k: int | None = None,
    select_quantile: float = 0.5,
    sequence_length: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BOPO-style (Best-anchored and Objective-guided Preference Optimization) loss / SRO loss.

    This implements the Stochastic Ranking Optimization (SRO) loss used by BOPO.
    The paper-faithful path is ``select_strategy="paper"``, which:
    1. sorts each candidate pool by reward/objective,
    2. picks evenly spaced candidates via stride ``B // K``,
    3. forms best-anchored pairs,
    4. weights each pair by the objective ratio.

    Args:
        reward: Tensor of shape [batch, pomo], higher is better.
            For minimization problems like JSP/FFSP, reward = -makespan.
        log_likelihood: Tensor of shape [batch, pomo], trajectory log-prob score.
            When ``sequence_length`` is provided, this is normalized to the
            paper's mean-per-step log-prob before pairing.
        alpha: Scale applied to log-likelihood.
        pair_mode: Pairing strategy:
            - "anchor_best": Pair the best solution with all other selected solutions
            - "all_pairs": Pair all better-worse pairs among selected solutions
        select_strategy: Strategy to select solutions for pairing:
            - "paper": BOPO paper-faithful evenly spaced rank selection
            - "top_k": Select the top k solutions
            - "quantile": Select solutions better than a quantile threshold
        select_k: Number of top solutions to select (for "top_k" strategy).
            If None, uses sqrt(pomo) as default.
        select_quantile: Quantile threshold (for "quantile" strategy, 0.0-1.0).
        sequence_length: Optional scalar or tensor broadcastable to ``reward``.
            When set, ``log_likelihood`` is divided by this value so the BOPO
            score uses mean log-prob per decoding step like the official code.

    Returns:
        loss: Scalar tensor.
        pair_count: Number of pairs used, useful for diagnostics.
    """
    if pair_mode not in {"anchor_best", "all_pairs"}:
        raise ValueError(f"Unknown bopo pair_mode: {pair_mode}")
    if select_strategy not in {"paper", "top_k", "quantile"}:
        raise ValueError(f"Unknown bopo select_strategy: {select_strategy}")
    if not 0.0 <= float(select_quantile) <= 1.0:
        raise ValueError(f"select_quantile must be in [0, 1], got {select_quantile}.")

    batch_size, num_pomo = reward.shape

    # Convert reward to objective (lower is better, e.g., makespan)
    # reward = -objective => objective = -reward
    objective = -reward
    log_score = log_likelihood
    if sequence_length is not None:
        if not torch.is_tensor(sequence_length):
            sequence_length = torch.tensor(
                float(sequence_length), device=log_likelihood.device, dtype=log_likelihood.dtype
            )
        log_score = log_score / sequence_length.clamp_min(1.0)

    if select_k is None:
        default_k = max(2, int(torch.sqrt(torch.tensor(num_pomo, dtype=torch.float32)).item()))
    else:
        default_k = int(select_k)
    if default_k < 2:
        raise ValueError(f"BOPO selection requires at least 2 candidates, got {default_k}.")

    per_pair_losses: list[torch.Tensor] = []
    pair_count_value = 0
    eps = 1e-8

    for batch_idx in range(batch_size):
        objective_row = objective[batch_idx]
        log_score_row = alpha * log_score[batch_idx]

        if select_strategy == "paper":
            if num_pomo % default_k != 0:
                raise ValueError(
                    f"BOPO paper strategy requires num_pomo % select_k == 0, got "
                    f"num_pomo={num_pomo}, select_k={default_k}."
                )
            stride = num_pomo // default_k
            selected = reward[batch_idx].sort(descending=True).indices[::stride]
            if selected.numel() != default_k:
                raise RuntimeError(
                    f"BOPO paper selection expected exactly {default_k} items, got {selected.numel()}."
                )
        elif select_strategy == "top_k":
            k_eff = min(default_k, num_pomo)
            selected = objective_row.sort(descending=False).indices[:k_eff]
        else:
            threshold = torch.quantile(objective_row, float(select_quantile))
            selected = torch.nonzero(objective_row <= threshold, as_tuple=False).flatten()
            if selected.numel() < 2:
                selected = objective_row.topk(2, largest=False).indices
            selected = selected[torch.argsort(objective_row[selected], descending=False)]

        selected_obj = objective_row[selected]
        selected_logp = log_score_row[selected]

        if pair_mode == "anchor_best":
            better_obj = selected_obj[0].expand(selected_obj.numel() - 1)
            worse_obj = selected_obj[1:]
            better_logp = selected_logp[0].expand(selected_logp.numel() - 1)
            worse_logp = selected_logp[1:]
            makespan_factor = (worse_obj + eps) / (better_obj + eps)
            logp_gap = better_logp - worse_logp
            per_pair_losses.append(-F.logsigmoid(makespan_factor * logp_gap))
            pair_count_value += int(selected_obj.numel() - 1)
        else:
            for better_pos in range(selected_obj.numel()):
                for worse_pos in range(better_pos + 1, selected_obj.numel()):
                    makespan_factor = (selected_obj[worse_pos] + eps) / (selected_obj[better_pos] + eps)
                    logp_gap = selected_logp[better_pos] - selected_logp[worse_pos]
                    per_pair_losses.append(-F.logsigmoid(makespan_factor * logp_gap).unsqueeze(0))
                    pair_count_value += 1

    if pair_count_value == 0:
        return log_likelihood.sum() * 0.0, torch.tensor(0.0, device=reward.device)

    loss = torch.cat([x.reshape(-1) for x in per_pair_losses]).mean()
    pair_count = torch.tensor(float(pair_count_value), dtype=torch.float32, device=reward.device)
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
    if impl not in {"sll", "slim", "listnet"}:
        raise ValueError(f"Unknown sll_loss impl: {impl}")
    if float(temperature) <= 0.0:
        raise ValueError(f"sll temperature must be > 0, got {temperature}.")

    batch_size, num_pomo = reward.shape

    # Sort by reward descending to get the optimal ranking
    sorted_reward, sorted_idx = reward.sort(dim=1, descending=True)
    logp = alpha * log_likelihood

    if impl == "sll":
        # SLL: Softmax Listwise Loss (ListMLE-style)
        # Standard ListMLE uses the suffix partition:
        #   sum_{j=i..P} exp(score_rank[j])
        # rather than a prefix sum. The previous implementation used a forward
        # cumsum, which optimizes the wrong ordering objective.
        scaled_logp = logp / temperature
        logp_rank = scaled_logp.gather(1, sorted_idx)
        suffix_logsumexp = torch.logcumsumexp(logp_rank.flip(dims=[1]), dim=1).flip(dims=[1])
        loss = -(logp_rank - suffix_logsumexp).mean()

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
