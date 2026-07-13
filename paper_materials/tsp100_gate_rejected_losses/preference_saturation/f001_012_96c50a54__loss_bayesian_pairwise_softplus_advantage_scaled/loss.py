def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    adv_w = batch['advantage_w']
    adv_l = batch['advantage_l']
    beta = float(extra.get('beta', extra.get('hyperparams', {}).get('beta', 1.0)))
    adv_gap = ops.sub(adv_w, adv_l)
    scaled_adv = ops.mul(beta, adv_gap)
    margin = ops.sub(lpw, lpl)
    # Negative of product of scaled advantage gap and margin
    neg_product = ops.neg(ops.mul(scaled_adv, margin))
    losses = ops.mul(weight, ops.softplus(neg_product))
    return ops.mean(losses)
