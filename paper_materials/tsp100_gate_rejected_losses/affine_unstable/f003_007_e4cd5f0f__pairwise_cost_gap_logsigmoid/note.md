# f003_007_e4cd5f0f: pairwise_cost_gap_logsigmoid

Category: **Affine-unstable candidates**

Why this is useful: These formulas pass the basic winner/loser preference gate, but CO-alignment rejects them because the loss changes under irrelevant objective affine transformations. They are good examples of plausible cost-gap objectives that use the objective scale too literally.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `3`, pair `21`.

## Gate Diagnosis

- pair_reason: `co_gate_failed`
- joint_gate_reason: `ok`
- joint_where_failed: `[]`
- observed: `{'grad_w_pass_rate': 1.0, 'grad_l_pass_rate': 1.0, 'effective_grad_ratio': 1.0, 'loss': 0.5655002593994141, 'loss_swap': 0.8488335013389587, 'swap_ok': True}`
- co_reason: `affine_invariance_violation`
- co_failed_gate: `AffineInvariance`, kind `affine_invariance_violation`

## Intuition

This loss leverages the cost gap signal (cost_b - cost_a) to encourage the model to prefer lower-cost options. Using logsigmoid on the scaled difference between log probabilities ensures smooth and bounded gradients that push the model toward better ordering. Pairwise operations and direct cost gap signal focus the training on cost-based preferences, improving the policy's ability to distinguish better options robustly.

## Pseudocode

```text
losses = -logsigmoid(alpha * scale * (log_prob_w - log_prob_l) - lambda * (cost_b - cost_a))
loss = mean(losses)
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
    scale = float(extra.get('scale', 1.5))
    lambda_ = float(extra.get('lambda', 0.1))
    diff_logprob = alpha * scale * (lpw - lpl)
    gap = cost_b - cost_a
    x = diff_logprob - lambda_ * gap
    x_clamped = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x_clamped)
    weighted_losses = losses * weight
    loss = ops.mean(weighted_losses)
    return loss
```
