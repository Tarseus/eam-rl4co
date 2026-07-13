def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    alpha = float(extra.get('alpha', 1.0))
    lambda_ = float(extra.get('lambda', 0.5))
    diff_log_prob = lpw - lpl
    diff_cost = cost_b - cost_a
    x = alpha * diff_log_prob - lambda_ * diff_cost
    x_clamped = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x_clamped)
    weighted_losses = losses * weight
    loss = ops.mean(weighted_losses)
    return loss
