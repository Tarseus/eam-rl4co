def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    adv_gap = batch['advantage_gap']
    weight = batch.get('weight', None)
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    margin = ops.mul(alpha, ops.sub(lpw, lpl))
    clamped_margin = ops.clamp(margin, -10.0, 10.0)
    smooth_margin = ops.tanh(clamped_margin)
    weight_factor = ops.abs(adv_gap)
    mean_abs_wf = ops.mean(weight_factor)
    eps = 1e-8
    norm_weight = ops.div(weight_factor, ops.add(mean_abs_wf, eps))
    hinge = ops.relu(ops.sub(ops.ones_like(smooth_margin), smooth_margin))
    loss = ops.mul(hinge, norm_weight)
    if weight is not None:
        loss = ops.mul(loss, weight)
    return ops.mean(loss)
