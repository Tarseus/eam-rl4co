# f000_012_50dd0498: pairwise_cost_gap_obj_loss

Category: **Preference-saturation failures**

Why this is useful: These formulas collapse to zero or near-zero effective gradients on the gate batch. They may encode a reasonable hinge, softplus, or advantage idea, but their link/thresholding makes the preference signal inactive or flat.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `0`, pair `1`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['log_prob_w_direction', 'log_prob_l_direction', 'saturation']`
- observed: `{'grad_w_pass_rate': 0.0, 'grad_l_pass_rate': 0.0, 'effective_grad_ratio': 0.0, 'loss': 0.0, 'loss_swap': 4.5333333015441895, 'swap_ok': True}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

This loss encourages ordering of pairs based on the cost gap signal, which measures the difference in actual costs between chosen and alternative options, promoting preference for lower-cost choices. Incorporating the log probabilities ensures it is a likelihood-based comparison, suitable for RL settings where relative rankings are crucial.

## Pseudocode

```text
loss = mean_over_pairs( max(0, (log_prob_l - log_prob_w) * cost_gap) )
```

## Code

```python
def generated_loss(batch, model_output, extra):
    log_prob_w = batch['log_prob_w']
    log_prob_l = batch['log_prob_l']
    cost_gap = batch.get('cost_gap', ops.zeros_like(log_prob_w))
    # Compute pairwise difference weighted by cost gap
    diff = ops.sub(log_prob_l, log_prob_w)
    weighted_diff = ops.mul(diff, cost_gap)
    # Apply hinge: only penalize if weighted diff > 0 (i.e., incorrect ordering)
    hinge_loss = ops.relu(weighted_diff)
    # Average over pairs to get scalar loss
    loss = ops.mean(hinge_loss)
    return loss
```
