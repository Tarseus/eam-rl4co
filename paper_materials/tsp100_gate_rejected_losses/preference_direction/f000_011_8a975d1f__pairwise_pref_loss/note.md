# f000_011_8a975d1f: pairwise_pref_loss

Category: **Preference-direction failures**

Why this is useful: These formulas are executable and have an interpretable pairwise structure, but the joint preference gate detects wrong swap behavior or wrong log-probability gradient direction. They show that plausible ingredients do not guarantee correct preference semantics.

Source: `runs\pref_loss_tsp100_discovery\20260317-102442`, generation `0`, pair `7`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['swap']`
- observed: `{'grad_w_pass_rate': 1.0, 'grad_l_pass_rate': 1.0, 'effective_grad_ratio': 1.0, 'loss': 2.1787893772125244, 'loss_swap': -6.712122440338135, 'swap_ok': False}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

This loss compares pairwise log probabilities to encourage the model to assign higher probabilities to preferred options. By using logsigmoid on the difference, it emphasizes relative ranking, making the model learn to prefer better options consistently. This approach directly encodes preference signals and leverages pairwise comparisons, leading to robust preference modeling.

## Pseudocode

```text
loss = -mean(logsigmoid(log_prob_w - log_prob_l) * (cost_b - cost_a))
```

## Code

```python
def generated_loss(batch, model_output, extra):
    log_prob_w = batch['log_prob_w']
    log_prob_l = batch['log_prob_l']
    weight = batch.get('weight', None)
    cost_a = batch.get('cost_a', None)
    cost_b = batch.get('cost_b', None)
    # Compute the difference in log probabilities
    diff_log_prob = log_prob_w - log_prob_l
    # Preference signal: logsigmoid of the difference
    pref_signal = ops.logsigmoid(diff_log_prob)
    # Optionally multiply by the cost difference to emphasize preferred options
    cost_diff = (cost_b - cost_a) if (cost_a is not None and cost_b is not None) else 1.0
    weighted_signal = pref_signal * cost_diff
    # Aggregate over batch to produce scalar loss
    loss = -ops.mean(weighted_signal)
    return loss
```
