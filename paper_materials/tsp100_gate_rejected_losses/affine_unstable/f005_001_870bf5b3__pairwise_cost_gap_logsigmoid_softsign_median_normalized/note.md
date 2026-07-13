# f005_001_870bf5b3: pairwise_cost_gap_logsigmoid_softsign_median_normalized

Category: **Affine-unstable candidates**

Why this is useful: These formulas pass the basic winner/loser preference gate, but CO-alignment rejects them because the loss changes under irrelevant objective affine transformations. They are good examples of plausible cost-gap objectives that use the objective scale too literally.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `5`, pair `7`.

## Gate Diagnosis

- pair_reason: `co_gate_failed`
- joint_gate_reason: `ok`
- joint_where_failed: `[]`
- observed: `{'grad_w_pass_rate': 1.0, 'grad_l_pass_rate': 1.0, 'effective_grad_ratio': 1.0, 'loss': -0.3180064857006073, 'loss_swap': 0.3180064857006073, 'swap_ok': True}`
- co_reason: `affine_invariance_violation`
- co_failed_gate: `AffineInvariance`, kind `affine_invariance_violation`

## Intuition

Repaired to reduce similarity with prior loss by replacing logsigmoid link with a softsign-based smoothed hinge, still emphasizing the cost_gap as quality signal. This eliminates exact logsigmoid usage and median normalization is preserved for scale invariance. The softsign transformation softens gradient behavior and better differentiates from prior implementations.

## Pseudocode

```text
diff = alpha * scale * (log_prob_w - log_prob_l) - 0.1 * (cost_b - cost_a)
diff_clamped = clamp(diff, -20, 20)
losses = -diff_clamped / (1 + abs(diff_clamped))  # softsign-based loss
weighted_losses = losses * weight
normalized_loss = median(weighted_losses) / (sum(weight) + epsilon)
```

## Code

```python
def generated_loss(batch, model_output, extra):
    eps = 1e-8
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', ops.ones_like(lpw))
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    scale = float(extra.get('hyperparams', {}).get('scale', 1.9496803035382082))
    diff = alpha * scale * (lpw - lpl) - 0.1 * (cost_b - cost_a)
    diff_clamped = ops.clamp(diff, -20.0, 20.0)
    losses = - diff_clamped / (1.0 + ops.abs(diff_clamped))
    weighted_losses = losses * weight
    norm_factor = ops.sum(weight) + eps
    loss = ops.median(weighted_losses) / norm_factor
    return loss
```
