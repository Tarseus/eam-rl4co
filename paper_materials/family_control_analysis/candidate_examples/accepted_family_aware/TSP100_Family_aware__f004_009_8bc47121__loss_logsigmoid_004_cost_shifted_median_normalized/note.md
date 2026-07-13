# TSP100 Family-aware: f004_009_8bc47121 ? loss_logsigmoid_004_cost_shifted_median_normalized

Why collected: Family-aware TSP best: bounded normalized cost-gap modulation.

Generation: `4`  
Family label: `Cost / Pairwise margin`  
Family signature: `pairwise_margin|cost_gap|logsigmoid|median|scale_invariance`  
Motifs: `extra_regularizer; raw_cost_additive; bounded_or_normalized; logsigmoid_backbone; plain_pairwise_margin`

## Intuition

Pairwise logsigmoid loss with cost gap, normalized by sum of weights to enforce scale-invariance and maintain stable aggregation via median.

## Pseudocode

```text
losses = -logsigmoid(clamp(alpha*scale*(lpw - lpl) - 0.1*(cost_b - cost_a), -20, 20))
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
    gap = cost_b - cost_a
    x = alpha * scale * (lpw - lpl) - 0.1 * gap
    x = ops.clamp(x, -20.0, 20.0)
    losses = -ops.logsigmoid(x)
    weighted_losses = losses * weight
    norm_factor = ops.sum(weight) + eps
    median_loss = ops.median(weighted_losses)
    loss = median_loss / norm_factor
    return loss
```
