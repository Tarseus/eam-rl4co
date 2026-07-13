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

    x = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_gap))
    x = ops.clamp(x, -20.0, 20.0)
    loss = ops.neg(ops.logsigmoid(x))

    weighted_loss = ops.mul(loss, weight)
    sum_weighted_loss = ops.sum(weighted_loss)
    total_weight = ops.sum(weight)

    weighted_avg_loss = ops.div(sum_weighted_loss, total_weight)
    weight_norm = ops.add(ops.sqrt(total_weight), eps)
    norm_loss = ops.div(weighted_avg_loss, weight_norm)

    delta_rank_norm = ops.normalize(delta_rank)
    delta_rank_squared = ops.mul(delta_rank_norm, delta_rank_norm)
    poly_reg = ops.mean(ops.softplus(delta_rank_squared))

    final_loss = ops.add(norm_loss, ops.mul(gamma, poly_reg))
    return final_loss
