def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', None)
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    scale = float(extra.get('hyperparams', {}).get('scale', 1.9496803035382082))
    gap = (cost_b - cost_a).detach()
    x = alpha * scale * (lpw - lpl) - 0.1 * gap
    x = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x)
    if weight is not None:
        weighted_losses = losses * weight
        loss = ops.sum(weighted_losses) / ops.sum(weight)
    else:
        loss = ops.sum(losses)
    return loss
