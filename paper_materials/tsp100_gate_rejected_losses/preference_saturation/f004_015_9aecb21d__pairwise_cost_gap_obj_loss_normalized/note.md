# f004_015_9aecb21d: pairwise_cost_gap_obj_loss_normalized

Category: **Preference-saturation failures**

Why this is useful: These formulas collapse to zero or near-zero effective gradients on the gate batch. They may encode a reasonable hinge, softplus, or advantage idea, but their link/thresholding makes the preference signal inactive or flat.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `4`, pair `0`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['log_prob_w_direction', 'log_prob_l_direction', 'saturation']`
- observed: `{'grad_w_pass_rate': 0.0, 'grad_l_pass_rate': 0.0, 'effective_grad_ratio': 0.0, 'loss': 0.0, 'loss_swap': 4.5333333015441895, 'swap_ok': True}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

This loss normalizes the pairwise hinge loss by the total sum of weights to ensure scale-invariance and prevent large batch size effects, promoting stable training regardless of batch composition.

## Pseudocode

```text
diff = log_prob_l - log_prob_w
weighted_diff = diff * cost_gap * weight
hinge_loss = max(0, weighted_diff)
loss = sum(hinge_loss) / sum(weight)
```

## Code

```python
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
```
