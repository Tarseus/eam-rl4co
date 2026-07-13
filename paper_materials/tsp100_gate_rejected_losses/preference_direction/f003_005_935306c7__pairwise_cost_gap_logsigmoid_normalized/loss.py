def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    alpha = float(extra.get('alpha', 1.0))
    scale = float(extra.get('hyperparams', {}).get('scale', 1.0))
    lambda_ = float(extra.get('hyperparams', {}).get('lambda', 1.0))
    eps = 1e-6
    gap = cost_b - cost_a
    norm = ops.mean(ops.abs(gap)) + eps
    scaled_gap = gap / norm
    x = alpha * scale * (lpw - lpl) - lambda_ * scaled_gap
    x = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x)
    weighted_losses = losses * weight
    loss = ops.mean(weighted_losses)
    return loss
