# f001_012_96c50a54: loss_bayesian_pairwise_softplus_advantage_scaled

Category: **Preference-saturation failures**

Why this is useful: These formulas collapse to zero or near-zero effective gradients on the gate batch. They may encode a reasonable hinge, softplus, or advantage idea, but their link/thresholding makes the preference signal inactive or flat.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `1`, pair `6`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['log_prob_w_direction', 'log_prob_l_direction', 'saturation', 'swap']`
- observed: `{'grad_w_pass_rate': 0.0, 'grad_l_pass_rate': 0.0, 'effective_grad_ratio': 0.0, 'loss': 0.6931473016738892, 'loss_swap': 0.6931473016738892, 'swap_ok': False}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

Shift from margin and rank-based losses to a probabilistic Bayesian-inspired paradigm using softplus on scaled advantage differences to robustly capture preference with smooth gradients, combining advantage signal directly and weighting by pair importance. This avoids margin thresholds and clamps by using softplus as a probabilistic likelihood surrogate.

## Pseudocode

```text
adv_gap = advantage_w - advantage_l
scaled_adv = beta * adv_gap
loss = mean(weight * softplus(-scaled_adv * (log_prob_w - log_prob_l)))
```

## Code

```python
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
```
