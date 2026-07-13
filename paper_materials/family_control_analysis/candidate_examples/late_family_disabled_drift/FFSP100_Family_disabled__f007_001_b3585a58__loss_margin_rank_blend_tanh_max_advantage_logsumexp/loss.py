def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    adv_w = batch['advantage_w']
    adv_l = batch['advantage_l']
    weight = batch.get('weight', None)
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    adv_max = ops.maximum(adv_w, adv_l)
    margin = ops.sub(lpw, lpl)
    scaled_margin = ops.mul(alpha, margin)
    clamped_margin = ops.clamp(scaled_margin, -10.0, 10.0)
    smooth_margin = ops.tanh(clamped_margin)
    hinge = ops.relu(ops.sub(ops.ones_like(smooth_margin), smooth_margin))
    weight_factor = ops.abs(adv_max)
    weighted_loss = ops.mul(hinge, weight_factor)
    if weight is not None:
        weighted_loss = ops.mul(weighted_loss, weight)
        normalizer = ops.sum(weight)
    else:
        normalizer = ops.sum(ops.ones_like(weighted_loss))
    max_loss = ops.max(weighted_loss)
    shifted = ops.sub(weighted_loss, max_loss)
    sum_exp = ops.sum(ops.exp(shifted))
    lse = ops.add(max_loss, ops.log(sum_exp))
    return ops.div(lse, normalizer)
