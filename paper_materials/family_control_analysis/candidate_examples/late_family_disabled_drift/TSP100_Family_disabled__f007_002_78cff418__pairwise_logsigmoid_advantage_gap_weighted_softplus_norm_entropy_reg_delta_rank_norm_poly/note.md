# TSP100 Family-disabled: f007_002_78cff418 ? pairwise_logsigmoid_advantage_gap_weighted_softplus_norm_entropy_reg_delta_rank_norm_poly

Why collected: Late no-family drift example selected for motif `extra_regularizer`.

Generation: `7`  
Family label: `Rank / Pairwise margin`  
Family signature: `pairwise_margin|advantage_gap_delta_rank_entropy_reg|softplus|weighted_softplus_poly_norm|weight_normalization_scale_invariant`  
Motifs: `extra_regularizer; advantage_heavy; bounded_or_normalized; logsigmoid_backbone`

## Intuition

To reduce similarity with prior losses, replaced the weighted sum with a weighted softplus aggregation to smooth loss contributions and changed log normalization to polynomial (sqrt) normalization of total weight for scale invariance. This changes behavior theoretically and algorithmically while maintaining the benefit of weighting and normalization. Also replaced entropy regularization on normalized delta_rank with a smooth squared term (square via multiply) to give a different regularizer form, enhancing novelty and potentially improving gradient properties.

## Pseudocode

```text
total_weight_raw = sum(weight)
norm_weight = weight / (total_weight_raw + eps)
weighted_loss = softplus(alpha * scale * (log_prob_w - log_prob_l) + beta * advantage_gap) * norm_weight
sum_weighted_loss = sum(weighted_loss)
poly_norm = sqrt(total_weight_raw + eps)
norm_loss = sum_weighted_loss / poly_norm
delta_rank_norm = normalize(delta_rank)
entropy_reg = gamma * mean(delta_rank_norm * delta_rank_norm)
final_loss = norm_loss + entropy_reg
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

    total_weight_raw = ops.sum(weight)
    norm_weight = ops.div(weight, ops.add(total_weight_raw, eps))

    x = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_gap))
    loss = ops.softplus(x)

    weighted_loss = ops.mul(loss, norm_weight)
    sum_weighted_loss = ops.sum(weighted_loss)

    poly_norm = ops.sqrt(ops.add(total_weight_raw, eps))
    norm_loss = ops.div(sum_weighted_loss, poly_norm)

    delta_rank_norm = ops.normalize(delta_rank)
    entropy_reg = ops.mul(gamma, ops.mean(ops.mul(delta_rank_norm, delta_rank_norm)))

    final_loss = ops.add(norm_loss, entropy_reg)
    return final_loss
```
