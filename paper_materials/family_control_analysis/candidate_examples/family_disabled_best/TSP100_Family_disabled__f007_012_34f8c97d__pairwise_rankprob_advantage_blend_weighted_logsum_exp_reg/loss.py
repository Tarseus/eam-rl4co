def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch['weight']
    advantage_w = batch['advantage_w']

    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    beta = float(extra.get('beta', extra.get('hyperparams', {}).get('beta', 1.0)))
    gamma = float(extra.get('gamma', extra.get('hyperparams', {}).get('gamma', 0.1)))
    scale = float(extra.get('hyperparams', {}).get('scale', 2.0))
    eps = 1e-12

    logits = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_w))
    logits = ops.clamp(logits, -20.0, 20.0)

    zeros = ops.zeros_like(logits)
    stacked_logits = ops.stack([zeros, logits], dim=1)
    logsumexp_vals = ops.logsumexp(stacked_logits, dim=1)

    loss_terms = ops.add(ops.neg(logits), logsumexp_vals)

    weighted_loss = ops.mul(loss_terms, weight)
    agg_loss = ops.sum(weighted_loss)

    weighted_adv = ops.mul(weight, advantage_w)
    adv_sum = ops.sum(weighted_adv) + eps
    adv_prob = ops.div(weighted_adv, adv_sum)

    adv_entropy = ops.neg(ops.sum(ops.mul(adv_prob, ops.log(ops.add(adv_prob, eps)))))

    reg_loss = ops.mul(gamma, adv_entropy)

    final_loss = ops.add(agg_loss, reg_loss)
    return final_loss
