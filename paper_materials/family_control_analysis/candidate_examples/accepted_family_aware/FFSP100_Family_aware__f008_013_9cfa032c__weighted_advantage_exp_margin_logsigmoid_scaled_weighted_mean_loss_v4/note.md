# FFSP100 Family-aware: f008_013_9cfa032c ? weighted_advantage_exp_margin_logsigmoid_scaled_weighted_mean_loss_v4

Why collected: Family-aware FFSP best: simple advantage-scale calibrated logsigmoid.

Generation: `8`  
Family label: `Advantage / Pairwise margin`  
Family signature: `pairwise_margin|advantage_gap|logsigmoid|weighted_mean|normalization_exponential_margin`  
Motifs: `advantage_heavy; bounded_or_normalized; logsigmoid_backbone; plain_pairwise_margin`

## Intuition

Replaced the linear scaling of (log_prob_w - log_prob_l) by normalizing advantage_gap with an exponential margin term to reduce similarity to prior losses. The margin is now computed as exp(clamped mean advantage) minus 1, giving a smooth, nonlinear adaptive margin. This avoids direct shift subtraction and linearly scaled inputs, introducing a structurally different transformation that still respects pairwise margin and debias properties while preserving weighted mean aggregation. This structural shift improves novelty and numeric stability.

## Pseudocode

```text
epsilon=1e-6
mean_adv = mean(advantage_gap)
margin = exp(clamp(abs(mean_adv), 0, 1)) - 1
norm = max(mean(norm(advantage_gap)), epsilon)
x_raw = (log_prob_w - log_prob_l) / norm
x = margin * x_raw
x = clamp(x, -20, 20)
loss = -logsigmoid(x)
loss_weighted = loss * weight
return sum(loss_weighted) / max(sum(weight), epsilon)
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch['weight']
    advantage_gap = batch['advantage_gap']
    epsilon = 1e-6
    mean_adv = ops.mean(advantage_gap)
    margin = ops.sub(ops.exp(ops.clamp(ops.abs(mean_adv), 0.0, 1.0)), 1.0)
    norm_per_sample = ops.norm(advantage_gap)
    norm = ops.maximum(ops.mean(norm_per_sample), epsilon)
    x_raw = ops.div(ops.sub(lpw, lpl), norm)
    x = ops.mul(margin, x_raw)
    x = ops.clamp(x, -20.0, 20.0)
    loss = ops.neg(ops.logsigmoid(x))
    loss_weighted = ops.mul(loss, weight)
    loss_mean = ops.div(ops.sum(loss_weighted), ops.maximum(ops.sum(weight), epsilon))
    return loss_mean
```
