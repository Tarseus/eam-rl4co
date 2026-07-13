def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch['weight']
    advantage_gap = batch['advantage_gap']
    delta_rank = batch['delta_rank']

    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    beta = float(extra.get('beta', extra.get('hyperparams', {}).get('beta', 1.0)))
    gamma = float(extra.get('gamma', extra.get('hyperparams', {}).get('gamma', 0.1)))
    scale = float(extra.get('hyperparams', {}).get('scale', 2.0))
    eps = 1e-12

    total_weight_raw = ops.sum(weight)
    norm_weight = ops.div(weight, ops.add(total_weight_raw, eps))

    x = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_gap))
    loss = ops.softplus(x)

    weighted_loss = ops.mul(loss, norm_weight)
    sum_weighted_loss = ops.sum(weighted_loss)

    poly_norm = ops.sqrt(ops.add(total_weight_raw, eps))
    norm_loss = ops.div(sum_weighted_loss, poly_norm)

    delta_rank_norm = ops.normalize(delta_rank)
    entropy_reg = ops.mul(gamma, ops.mean(ops.mul(delta_rank_norm, delta_rank_norm)))

    final_loss = ops.add(norm_loss, entropy_reg)
    return final_loss
