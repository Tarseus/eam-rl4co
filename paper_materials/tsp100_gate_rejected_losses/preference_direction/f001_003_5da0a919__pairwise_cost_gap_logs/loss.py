def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    alpha = float(extra.get('alpha', 1.0))
    beta = float(extra.get('beta', 1.0))
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    diff_probs = lpw - lpl
    gap = cost_b - cost_a
    x = alpha * diff_probs - beta * gap
    x = ops.clamp(x, -20.0, 20.0)
    loss = -ops.logsigmoid(x) * weight
    return ops.mean(loss)
