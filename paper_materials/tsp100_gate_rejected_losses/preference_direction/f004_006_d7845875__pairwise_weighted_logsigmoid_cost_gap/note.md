# f004_006_d7845875: pairwise_weighted_logsigmoid_cost_gap

Category: **Preference-direction failures**

Why this is useful: These formulas are executable and have an interpretable pairwise structure, but the joint preference gate detects wrong swap behavior or wrong log-probability gradient direction. They show that plausible ingredients do not guarantee correct preference semantics.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `4`, pair `16`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['swap']`
- observed: `{'grad_w_pass_rate': 1.0, 'grad_l_pass_rate': 1.0, 'effective_grad_ratio': 1.0, 'loss': 2.4504446983337402, 'loss_swap': 0.1837780624628067, 'swap_ok': False}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

This loss uses pairwise log-sigmoid of the difference in log probabilities, weighted by the cost gap signal to emphasize pairs with larger cost differences. It aims to push higher probabilities towards lower cost outcomes while robustly aggregating through weighted pairwise comparisons, which should improve preference learning especially when cost differences are meaningful.

## Pseudocode

```text
losses = -logsigmoid(alpha * (log_prob_w - log_prob_l) - lambda * (cost_b - cost_a))
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
    lambda_ = float(extra.get('lambda', 0.5))
    diff_log_prob = lpw - lpl
    diff_cost = cost_b - cost_a
    x = alpha * diff_log_prob - lambda_ * diff_cost
    x_clamped = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x_clamped)
    weighted_losses = losses * weight
    loss = ops.mean(weighted_losses)
    return loss
```
