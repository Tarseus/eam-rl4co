def generated_loss(batch, model_output, extra):
    log_prob_w = batch['log_prob_w']
    log_prob_l = batch['log_prob_l']
    cost_gap = batch.get('cost_gap', ops.zeros_like(log_prob_w))
    # Compute pairwise difference weighted by cost gap
    diff = ops.sub(log_prob_l, log_prob_w)
    weighted_diff = ops.mul(diff, cost_gap)
    # Apply hinge: only penalize if weighted diff > 0 (i.e., incorrect ordering)
    hinge_loss = ops.relu(weighted_diff)
    # Average over pairs to get scalar loss
    loss = ops.mean(hinge_loss)
    return loss
