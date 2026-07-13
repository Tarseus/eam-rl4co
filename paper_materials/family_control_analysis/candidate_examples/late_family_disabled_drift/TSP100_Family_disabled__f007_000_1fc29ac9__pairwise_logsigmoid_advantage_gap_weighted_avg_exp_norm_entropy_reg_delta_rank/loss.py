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

    exp_weights = ops.exp(weight)
    total_exp_weight = ops.add(ops.sum(exp_weights), eps)
    norm_weights = ops.div(exp_weights, total_exp_weight)

    x = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_gap))
    x = ops.clamp(x, -20.0, 20.0)
    loss = ops.neg(ops.logsigmoid(x))

    weighted_loss = ops.mul(loss, norm_weights)
    avg_loss = ops.sum(weighted_loss)

    delta_rank_norm = ops.normalize(delta_rank)
    entropy_reg = ops.mul(gamma, ops.mean(ops.softplus(delta_rank_norm)))

    final_loss = ops.add(avg_loss, entropy_reg)
    return final_loss
