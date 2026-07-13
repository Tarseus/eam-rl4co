# FFSP100 Family-disabled: f004_006_b9dc8aa5 ? loss_pairwise_margin_sigmoid_weighted_advantage_sum

Why collected: Family-disabled FFSP best: advantage-weighted sigmoid/sum formula that never beats the baseline.

Generation: `4`  
Family label: `Rank / Pairwise margin`  
Family signature: `pairwise_margin_rank_blend|advantage_gap|sigmoid_linear|sum|weighted`  
Motifs: `advantage_heavy; unstructured_weighted_sum; plain_pairwise_margin`

## Intuition

Maintains the pairwise margin approach with sigmoid smoothing of the scaled margin but replaces mean aggregation with sum aggregation to emphasize total pairwise loss contribution. This change in aggregation structure shifts focus from average per pair to total accumulated loss, suitable for scenarios valuing aggregate penalty.

## Pseudocode

```text
margin = log_prob_w - log_prob_l
scaled_margin = alpha * margin
smooth_margin = sigmoid(scaled_margin)
weight_factor = abs(advantage_gap)
loss_per_pair = (1 - smooth_margin) * weight_factor * weight
final_loss = sum(loss_per_pair)
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    weight = batch.get('weight', None)
    advantage_gap = batch['advantage_gap']
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    margin = ops.sub(lpw, lpl)
    scaled_margin = ops.mul(alpha, margin)
    smooth_margin = ops.sigmoid(scaled_margin)
    weight_factor = ops.abs(advantage_gap)
    loss = ops.mul(ops.sub(ops.ones_like(smooth_margin), smooth_margin), weight_factor)
    if weight is not None:
        loss = ops.mul(loss, weight)
    return ops.sum(loss)
```
