def generated_loss(batch, model_output, extra):
    eps = 1e-8
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    scale = float(extra.get('hyperparams', {}).get('scale', 1.9496803035382082))
    gap = cost_b - cost_a
    x = alpha * scale * (lpw - lpl) - 0.1 * gap
    x = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x)
    weighted_losses = losses * weight
    norm_factor = ops.sum(weight) + eps
    median_loss = ops.median(weighted_losses)
    loss = median_loss / norm_factor
    return loss
