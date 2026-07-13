def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch['weight']
    advantage_gap = batch['advantage_gap']
    epsilon = 1e-6
    mean_adv = ops.mean(advantage_gap)
    margin = ops.sub(ops.exp(ops.clamp(ops.abs(mean_adv), 0.0, 1.0)), 1.0)
    norm_per_sample = ops.norm(advantage_gap)
    norm = ops.maximum(ops.mean(norm_per_sample), epsilon)
    x_raw = ops.div(ops.sub(lpw, lpl), norm)
    x = ops.mul(margin, x_raw)
    x = ops.clamp(x, -20.0, 20.0)
    loss = ops.neg(ops.logsigmoid(x))
    loss_weighted = ops.mul(loss, weight)
    loss_mean = ops.div(ops.sum(loss_weighted), ops.maximum(ops.sum(weight), epsilon))
    return loss_mean
