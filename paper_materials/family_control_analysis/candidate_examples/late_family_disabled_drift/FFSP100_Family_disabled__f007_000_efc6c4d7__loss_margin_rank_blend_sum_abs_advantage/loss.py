def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    adv_w = batch['advantage_w']
    adv_l = batch['advantage_l']
    weight = batch.get('weight', None)
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    margin = ops.sub(lpw, lpl)
    scaled_margin = ops.mul(alpha, margin)
    clamped_margin = ops.clamp(scaled_margin, -10.0, 10.0)
    smooth_margin = ops.tanh(clamped_margin)
    weight_factor = ops.add(ops.abs(adv_w), ops.abs(adv_l))
    hinge = ops.relu(ops.sub(ops.ones_like(smooth_margin), smooth_margin))
    loss = ops.mul(hinge, weight_factor)
    if weight is not None:
        loss = ops.mul(loss, weight)
    return ops.sum(loss)
