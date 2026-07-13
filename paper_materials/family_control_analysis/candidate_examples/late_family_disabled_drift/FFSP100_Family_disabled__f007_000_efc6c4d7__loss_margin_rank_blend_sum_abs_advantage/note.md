# FFSP100 Family-disabled: f007_000_efc6c4d7 ? loss_margin_rank_blend_sum_abs_advantage

Why collected: Late no-family drift example selected for motif `advantage_blend_tanh_relu`.

Generation: `7`  
Family label: `Rank / Pairwise margin`  
Family signature: `pairwise_margin_advantage_blend|advantage_combined_sum_abs|tanh_relu|sum|weighted`  
Motifs: `advantage_heavy; advantage_blend_tanh_relu; unstructured_weighted_sum; bounded_or_normalized; plain_pairwise_margin`

## Intuition

Change aggregation from mean to sum and use the sum of absolute advantages from winner and loser as weighting. Maintains tanh smoothing and hinge loss with clamping for stability, but aggregates total pair loss by summation to emphasize total batch loss magnitude.

## Pseudocode

```text
lpw = batch['log_prob_w']
lpl = batch['log_prob_l']
adv_w = batch['advantage_w']
adv_l = batch['advantage_l']
margin = alpha * (lpw - lpl)
smooth_margin = tanh(clamp(margin, -10, 10))
weight_factor = abs(adv_w) + abs(adv_l)
hinge = relu(1 - smooth_margin)
loss = hinge * weight_factor
if weight exists:
  loss *= weight
final_loss = sum(loss)
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    adv_w = batch['advantage_w']
    adv_l = batch['advantage_l']
    weight = batch.get('weight', None)
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    margin = ops.sub(lpw, lpl)
    scaled_margin = ops.mul(alpha, margin)
    clamped_margin = ops.clamp(scaled_margin, -10.0, 10.0)
    smooth_margin = ops.tanh(clamped_margin)
    weight_factor = ops.add(ops.abs(adv_w), ops.abs(adv_l))
    hinge = ops.relu(ops.sub(ops.ones_like(smooth_margin), smooth_margin))
    loss = ops.mul(hinge, weight_factor)
    if weight is not None:
        loss = ops.mul(loss, weight)
    return ops.sum(loss)
```
