# TSP100 Family-disabled: f007_000_1fc29ac9 ? pairwise_logsigmoid_advantage_gap_weighted_avg_exp_norm_entropy_reg_delta_rank

Why collected: Late no-family drift example selected for motif `extra_regularizer`.

Generation: `7`  
Family label: `Rank / Pairwise margin`  
Family signature: `pairwise_margin|advantage_gap_delta_rank_entropy_reg|logsigmoid|weighted_avg_exp_norm|none`  
Motifs: `extra_regularizer; advantage_heavy; bounded_or_normalized; logsigmoid_backbone`

## Intuition

Shift aggregation from weighted sum with logarithmic normalization to a weighted average with exponential normalization of weights, maintaining the pairwise margin and advantage_gap signal. This structure normalizes the pair weights exponentially to emphasize medium-sized weights more evenly, combined with a softplus regularization on normalized delta_rank for stability and better generalization.

## Pseudocode

```text
exp_weights = exp(weight)
total_exp_weight = sum(exp_weights) + eps
norm_weights = exp_weights / total_exp_weight
x = clamp(alpha * scale * (log_prob_w - log_prob_l) + beta * advantage_gap, -20, 20)
loss = -logsigmoid(x)
weighted_loss = loss * norm_weights
avg_loss = sum(weighted_loss)
delta_rank_norm = normalize(delta_rank)
entropy_reg = gamma * mean(softplus(delta_rank_norm))
final_loss = avg_loss + entropy_reg
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

    exp_weights = ops.exp(weight)
    total_exp_weight = ops.add(ops.sum(exp_weights), eps)
    norm_weights = ops.div(exp_weights, total_exp_weight)

    x = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_gap))
    x = ops.clamp(x, -20.0, 20.0)
    loss = ops.neg(ops.logsigmoid(x))

    weighted_loss = ops.mul(loss, norm_weights)
    avg_loss = ops.sum(weighted_loss)

    delta_rank_norm = ops.normalize(delta_rank)
    entropy_reg = ops.mul(gamma, ops.mean(ops.softplus(delta_rank_norm)))

    final_loss = ops.add(avg_loss, entropy_reg)
    return final_loss
```
