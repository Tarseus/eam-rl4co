def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    adv_w = batch['advantage_w']
    adv_l = batch['advantage_l']
    cost_gap = batch.get('cost_gap', ops.zeros_like(lpw))
    margin = float(extra.get('margin', 0.1))

    adv_gap = ops.tanh(ops.sub(adv_w, adv_l))
    cost_penalty = ops.relu(cost_gap)
    rank_diff = ops.sub(lpw, lpl)
    margin_diff = ops.relu(ops.sub(margin, ops.mul(adv_gap, rank_diff)))

    loss_per_pair = ops.mul(ops.mul(margin_diff, cost_penalty), weight)
    total_weight = ops.sum(weight) + 1e-8
    loss = ops.div(ops.sum(loss_per_pair), total_weight)
    return loss
