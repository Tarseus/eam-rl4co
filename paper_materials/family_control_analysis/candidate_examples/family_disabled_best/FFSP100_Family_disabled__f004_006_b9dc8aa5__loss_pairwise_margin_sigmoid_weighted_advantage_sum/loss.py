def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', None)
    advantage_gap = batch['advantage_gap']
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    margin = ops.sub(lpw, lpl)
    scaled_margin = ops.mul(alpha, margin)
    smooth_margin = ops.sigmoid(scaled_margin)
    weight_factor = ops.abs(advantage_gap)
    loss = ops.mul(ops.sub(ops.ones_like(smooth_margin), smooth_margin), weight_factor)
    if weight is not None:
        loss = ops.mul(loss, weight)
    return ops.sum(loss)
