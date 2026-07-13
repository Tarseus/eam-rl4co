def generated_loss(batch, model_output, extra):
    log_prob_w = batch['log_prob_w']
    log_prob_l = batch['log_prob_l']
    weight = batch.get('weight', None)
    cost_a = batch.get('cost_a', None)
    cost_b = batch.get('cost_b', None)
    # Compute the difference in log probabilities
    diff_log_prob = log_prob_w - log_prob_l
    # Preference signal: logsigmoid of the difference
    pref_signal = ops.logsigmoid(diff_log_prob)
    # Optionally multiply by the cost difference to emphasize preferred options
    cost_diff = (cost_b - cost_a) if (cost_a is not None and cost_b is not None) else 1.0
    weighted_signal = pref_signal * cost_diff
    # Aggregate over batch to produce scalar loss
    loss = -ops.mean(weighted_signal)
    return loss
