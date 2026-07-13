# f003_005_935306c7: pairwise_cost_gap_logsigmoid_normalized

Category: **Preference-direction failures**

Why this is useful: These formulas are executable and have an interpretable pairwise structure, but the joint preference gate detects wrong swap behavior or wrong log-probability gradient direction. They show that plausible ingredients do not guarantee correct preference semantics.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `3`, pair `25`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['swap']`
- observed: `{'grad_w_pass_rate': 1.0, 'grad_l_pass_rate': 1.0, 'effective_grad_ratio': 1.0, 'loss': 0.9421698451042175, 'loss_swap': 0.5088366270065308, 'swap_ok': False}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

Fixed novelty issue by introducing normalization of the cost gap using its mean absolute value to reduce redundancy with similar losses. This avoids direct subtraction of cost_b and cost_a raw values and helps stabilize gradients. Also clarified parameter naming for stability and kept the pairwise signals. This preserves the original goal of encouraging higher log probabilities for lower cost options while being sufficiently distinct.

## Pseudocode

```text
gap = cost_b - cost_a
norm = mean(abs(gap)) + 1e-6
scaled_gap = gap / norm
x = alpha * scale * (log_prob_w - log_prob_l) - lambda * scaled_gap
x = clamp(x, -20, 20)
losses = -logsigmoid(x)
weighted_losses = losses * weight
loss = mean(weighted_losses)
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    alpha = float(extra.get('alpha', 1.0))
    scale = float(extra.get('hyperparams', {}).get('scale', 1.0))
    lambda_ = float(extra.get('hyperparams', {}).get('lambda', 1.0))
    eps = 1e-6
    gap = cost_b - cost_a
    norm = ops.mean(ops.abs(gap)) + eps
    scaled_gap = gap / norm
    x = alpha * scale * (lpw - lpl) - lambda_ * scaled_gap
    x = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x)
    weighted_losses = losses * weight
    loss = ops.mean(weighted_losses)
    return loss
```
