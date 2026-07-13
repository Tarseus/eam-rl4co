# TSP100 Family-disabled: f007_001_7b75011d ? pairwise_logsigmoid_advantage_gap_weighted_avg_sqrt_norm_entropy_reg_delta_rank_poly

Why collected: Late no-family drift example selected for motif `extra_regularizer`.

Generation: `7`  
Family label: `Rank / Pairwise margin`  
Family signature: `pairwise_margin|advantage_gap_delta_rank_entropy_reg_poly|logsigmoid|weighted_avg_sqrtnorm|none`  
Motifs: `extra_regularizer; advantage_heavy; bounded_or_normalized; logsigmoid_backbone`

## Intuition

Replaced exponential normalization with square root normalization of total weight to reduce similarity with previously seen exponential and logarithmic normalizations. Additionally, replaced the linear regularization on normalized delta_rank with a polynomial (quadratic) penalty using the square of the normalized delta_rank passed through softplus to encourage smoother gradients and stronger penalization of large deviations. This introduces a different aggregation and penalty scheme while preserving the core signals and pairwise margin loss structure, ensuring novelty and stability.

## Pseudocode

```text
total_weight = sum(weight)
weight_norm = sqrt(total_weight) + eps
x = clamp(alpha * scale * (log_prob_w - log_prob_l) + beta * advantage_gap, -20, 20)
loss = -logsigmoid(x)
weighted_loss = loss * weight
sum_weighted_loss = sum(weighted_loss)
weighted_avg_loss = sum_weighted_loss / total_weight
norm_loss = weighted_avg_loss / weight_norm
delta_rank_norm = normalize(delta_rank)
poly_reg = mean(softplus(delta_rank_norm ** 2))
final_loss = norm_loss + gamma * poly_reg
return final_loss
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch['weight']
    advantage_gap = batch['advantage_gap']
    delta_rank = batch['delta_rank']

    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    beta = float(extra.get('beta', extra.get('hyperparams', {}).get('beta', 1.0)))
    gamma = float(extra.get('gamma', extra.get('hyperparams', {}).get('gamma', 0.1)))
    scale = float(extra.get('hyperparams', {}).get('scale', 2.0))
    eps = 1e-12

    x = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_gap))
    x = ops.clamp(x, -20.0, 20.0)
    loss = ops.neg(ops.logsigmoid(x))

    weighted_loss = ops.mul(loss, weight)
    sum_weighted_loss = ops.sum(weighted_loss)
    total_weight = ops.sum(weight)

    weighted_avg_loss = ops.div(sum_weighted_loss, total_weight)
    weight_norm = ops.add(ops.sqrt(total_weight), eps)
    norm_loss = ops.div(weighted_avg_loss, weight_norm)

    delta_rank_norm = ops.normalize(delta_rank)
    delta_rank_squared = ops.mul(delta_rank_norm, delta_rank_norm)
    poly_reg = ops.mean(ops.softplus(delta_rank_squared))

    final_loss = ops.add(norm_loss, ops.mul(gamma, poly_reg))
    return final_loss
```
