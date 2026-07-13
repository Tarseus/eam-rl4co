# f001_003_5da0a919: pairwise_cost_gap_logs

Category: **Preference-direction failures**

Why this is useful: These formulas are executable and have an interpretable pairwise structure, but the joint preference gate detects wrong swap behavior or wrong log-probability gradient direction. They show that plausible ingredients do not guarantee correct preference semantics.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `1`, pair `11`.

## Gate Diagnosis

- pair_reason: `cheap_gate_failed`
- joint_gate_reason: `joint_preference_violation`
- joint_where_failed: `['swap']`
- observed: `{'grad_w_pass_rate': 1.0, 'grad_l_pass_rate': 1.0, 'effective_grad_ratio': 1.0, 'loss': 5.1718363761901855, 'loss_swap': 0.07183623313903809, 'swap_ok': False}`
- co_reason: `None`
- co_failed_gate: `None`, kind `None`

## Intuition

This loss penalizes pairs with large cost differences (cost_b - cost_a) while encouraging higher log probabilities for preferred options. Using a sigmoid-based link function on the difference in log probs and cost gap creates a smooth, differentiable surface that emphasizes preference consistency, aiding stable RL training in combinatorial optimization scenarios.

## Pseudocode

```text
loss = sum(-logsigmoid(alpha * (log_prob_w - log_prob_l) - beta * (cost_b - cost_a)))
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    alpha = float(extra.get('alpha', 1.0))
    beta = float(extra.get('beta', 1.0))
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    diff_probs = lpw - lpl
    gap = cost_b - cost_a
    x = alpha * diff_probs - beta * gap
    x = ops.clamp(x, -20.0, 20.0)
    loss = -ops.logsigmoid(x) * weight
    return ops.mean(loss)
```
