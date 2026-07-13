def generated_loss(batch, model_output, extra):
    log_prob_w = batch['log_prob_w']
    log_prob_l = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(log_prob_w))
    cost_gap = batch.get('cost_gap', ops.zeros_like(log_prob_w))
    diff = ops.sub(log_prob_l, log_prob_w)
    weighted_diff = ops.mul(ops.mul(diff, cost_gap), weight)
    hinge_loss = ops.relu(weighted_diff)
    loss = ops.div(ops.sum(hinge_loss), ops.sum(weight))
    return loss
