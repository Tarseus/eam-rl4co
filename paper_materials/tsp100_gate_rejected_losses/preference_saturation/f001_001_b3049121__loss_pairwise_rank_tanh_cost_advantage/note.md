# f001_001_b3049121: loss_pairwise_rank_tanh_cost_advantage

Category: **Preference-saturation failures**

Why this is useful: These formulas collapse to zero or near-zero effective gradients on the gate batch. They may encode a reasonable hinge, softplus, or advantage idea, but their link/thresholding makes the preference signal inactive or flat.

Source: `runs\pref_loss_tsp100_discovery\20260317-102442`, generation `1`, pair `10`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['log_prob_w_direction', 'log_prob_l_direction', 'saturation', 'swap']`
- observed: `{'grad_w_pass_rate': 0.0, 'grad_l_pass_rate': 0.0, 'effective_grad_ratio': 0.0, 'loss': 0.5666666626930237, 'loss_swap': 0.5666666626930237, 'swap_ok': False}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

A pairwise ranking loss that combines a tanh-transformed advantage gap with a cost gap penalty, using a hinge-like margin on the rank difference. This paradigm shift replaces the logistic link with a margin ranking approach linked via a tanh non-linearity, aggregating losses by weighted sum to emphasize significant pairwise disagreements prioritized by cost and advantage signals.

## Pseudocode

```text
adv_gap = tanh(advantage_w - advantage_l)
cost_penalty = relu(cost_gap)
rank_diff = log_prob_w - log_prob_l
margin = 0.1
margin_diff = relu(margin - adv_gap * rank_diff)
loss_per_pair = margin_diff * cost_penalty * weight
loss = sum(loss_per_pair) / sum(weight)
```

## Code

```python
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
```
