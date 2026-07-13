# f000_001_4b362843: loss_logsigmoid_000_cost_structure_shifted

Category: **Affine-unstable candidates**

Why this is useful: These formulas pass the basic winner/loser preference gate, but CO-alignment rejects them because the loss changes under irrelevant objective affine transformations. They are good examples of plausible cost-gap objectives that use the objective scale too literally.

Source: `runs\pref_loss_tsp100_discovery\20260317-131507`, generation `0`, pair `4`.

## Gate Diagnosis

- pair_reason: `co_gate_failed`
- joint_gate_reason: `ok`
- joint_where_failed: `[]`
- observed: `{'grad_w_pass_rate': 1.0, 'grad_l_pass_rate': 1.0, 'effective_grad_ratio': 1.0, 'loss': 0.47336772084236145, 'loss_swap': 1.0115199089050293, 'swap_ok': True}`
- co_reason: `affine_invariance_violation`
- co_failed_gate: `AffineInvariance`, kind `affine_invariance_violation`

## Intuition

rule_based: pairwise logsigmoid loss with shifted aggregation structure

## Pseudocode

```text
losses = -logsigmoid(clamp(alpha*scale*(lpw - lpl) - 0.1*gap, -20, 20))
weighted_losses = losses * weight if weight is not None
loss = sum(weighted_losses) / sum(weight) if weight is not None else sum(losses)
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', None)
    cost_a = batch['cost_a']
    cost_b = batch['cost_b']
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    scale = float(extra.get('hyperparams', {}).get('scale', 1.9496803035382082))
    gap = (cost_b - cost_a).detach()
    x = alpha * scale * (lpw - lpl) - 0.1 * gap
    x = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x)
    if weight is not None:
        weighted_losses = losses * weight
        loss = ops.sum(weighted_losses) / ops.sum(weight)
    else:
        loss = ops.sum(losses)
    return loss
```
