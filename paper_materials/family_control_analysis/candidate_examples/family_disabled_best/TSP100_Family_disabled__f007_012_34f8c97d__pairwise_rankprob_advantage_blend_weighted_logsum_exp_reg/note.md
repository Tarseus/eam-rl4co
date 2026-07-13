# TSP100 Family-disabled: f007_012_34f8c97d ? pairwise_rankprob_advantage_blend_weighted_logsum_exp_reg

Why collected: Family-disabled TSP best: rank-prob/advantage entropy regularized formula, weaker than family-aware best.

Generation: `7`  
Family label: `Prob / Rank-prob`  
Family signature: `pairwise_rankprob|advantage_w_entropy_reg|logsumexp|weighted_sum|entropy_regularization`  
Motifs: `extra_regularizer; advantage_heavy; rankprob_or_probabilistic; unstructured_weighted_sum; bounded_or_normalized`

## Intuition

Shift from margin-based pairwise loss to a probabilistic rank-based paradigm using log-sum-exp to model relative likelihoods. Instead of a margin, it computes relative probabilities of winning weighted by advantage signals and aggregates via weighted log-sum-exp to emphasize harder pairs. Regularization is applied on the weighted advantage distribution entropy encouraging weight diversity and stability.

## Pseudocode

```text
weighted_adv = weight * advantage_w
logits = alpha * scale * (log_prob_w - log_prob_l) + beta * advantage_w
logits_clamped = clamp(logits, -20, 20)
loss_terms = -logits_clamped + logsumexp(ops.stack([zeros_like(logits), logits_clamped], dim=1), dim=1)
weighted_loss = loss_terms * weight
agg_loss = sum(weighted_loss)
adv_sum = sum(weighted_adv) + eps
adv_prob = weighted_adv / adv_sum
adv_entropy = -sum(adv_prob * log(adv_prob + eps))
reg_loss = gamma * adv_entropy
final_loss = agg_loss + reg_loss
return final_loss
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch['weight']
    advantage_w = batch['advantage_w']

    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    beta = float(extra.get('beta', extra.get('hyperparams', {}).get('beta', 1.0)))
    gamma = float(extra.get('gamma', extra.get('hyperparams', {}).get('gamma', 0.1)))
    scale = float(extra.get('hyperparams', {}).get('scale', 2.0))
    eps = 1e-12

    logits = ops.add(ops.mul(alpha * scale, ops.sub(lpw, lpl)), ops.mul(beta, advantage_w))
    logits = ops.clamp(logits, -20.0, 20.0)

    zeros = ops.zeros_like(logits)
    stacked_logits = ops.stack([zeros, logits], dim=1)
    logsumexp_vals = ops.logsumexp(stacked_logits, dim=1)

    loss_terms = ops.add(ops.neg(logits), logsumexp_vals)

    weighted_loss = ops.mul(loss_terms, weight)
    agg_loss = ops.sum(weighted_loss)

    weighted_adv = ops.mul(weight, advantage_w)
    adv_sum = ops.sum(weighted_adv) + eps
    adv_prob = ops.div(weighted_adv, adv_sum)

    adv_entropy = ops.neg(ops.sum(ops.mul(adv_prob, ops.log(ops.add(adv_prob, eps)))))

    reg_loss = ops.mul(gamma, adv_entropy)

    final_loss = ops.add(agg_loss, reg_loss)
    return final_loss
```
