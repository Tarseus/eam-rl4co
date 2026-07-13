# FFSP100 Family-disabled: f007_002_03e05fbc ? loss_margin_rank_blend_tanh_weighted_advantage_gap_norm

Why collected: Late no-family drift example selected for motif `advantage_blend_tanh_relu`.

Generation: `7`  
Family label: `Rank / Pairwise margin`  
Family signature: `pairwise_margin_rank_blend|advantage_gap|tanh_relu|mean|weight_normalization`  
Motifs: `extra_regularizer; advantage_heavy; advantage_blend_tanh_relu; bounded_or_normalized; plain_pairwise_margin`

## Intuition

Normalize the weighting by advantage_gap to ensure scale-invariance and prevent instability caused by large advantage magnitude variations. This normalization stabilizes the pairwise loss by scaling weights to have unit mean absolute value, preserving relative weighting but ensuring consistent magnitude across batches.

## Pseudocode

```text
lpw = batch['log_prob_w']
lpl = batch['log_prob_l']
adv_gap = batch['advantage_gap']
weight = batch['weight'] if present else 1
margin = alpha * (lpw - lpl)
clamped_margin = clamp(margin, -10, 10)
smooth_margin = tanh(clamped_margin)
weight_factor = abs(adv_gap)
normalized_weight = weight_factor / (mean(weight_factor) + eps)
hinge = relu(1 - smooth_margin)
loss = hinge * normalized_weight
if weight exists:
  loss *= weight
final_loss = mean(loss)
```

## Code

```python
def generated_loss(batch, model_output, extra):
    lpw = batch['log_prob_w']
    lpl = batch['log_prob_l']
    adv_gap = batch['advantage_gap']
    weight = batch.get('weight', None)
    alpha = float(extra.get('alpha', extra.get('hyperparams', {}).get('alpha', 1.0)))
    margin = ops.mul(alpha, ops.sub(lpw, lpl))
    clamped_margin = ops.clamp(margin, -10.0, 10.0)
    smooth_margin = ops.tanh(clamped_margin)
    weight_factor = ops.abs(adv_gap)
    mean_abs_wf = ops.mean(weight_factor)
    eps = 1e-8
    norm_weight = ops.div(weight_factor, ops.add(mean_abs_wf, eps))
    hinge = ops.relu(ops.sub(ops.ones_like(smooth_margin), smooth_margin))
    loss = ops.mul(hinge, norm_weight)
    if weight is not None:
        loss = ops.mul(loss, weight)
    return ops.mean(loss)
```
