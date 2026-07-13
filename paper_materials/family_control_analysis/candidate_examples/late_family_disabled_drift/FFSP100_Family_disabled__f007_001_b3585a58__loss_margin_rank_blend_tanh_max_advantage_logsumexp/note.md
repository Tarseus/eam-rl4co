# FFSP100 Family-disabled: f007_001_b3585a58 ? loss_margin_rank_blend_tanh_max_advantage_logsumexp

Why collected: Late no-family drift example selected for motif `advantage_blend_tanh_relu`.

Generation: `7`  
Family label: `Rank / Pairwise margin`  
Family signature: `pairwise_margin_advantage_blend|advantage_combined_max|tanh_relu|logsumexp_normalized|weighted`  
Motifs: `advantage_heavy; rankprob_or_probabilistic; advantage_blend_tanh_relu; bounded_or_normalized; plain_pairwise_margin`

## Intuition

Instead of averaging loss over pairs, use log-sum-exp aggregation for a smooth maximum effect that prioritizes hardest pairs. Retains pairwise margin blending with advantage max weighting and tanh smoothing for numerical stability. This structure shift improves robustness by focusing more on significant loss pairs while keeping weighting and clamping intact.

## Pseudocode

```text
lpw = batch['log_prob_w']
lpl = batch['log_prob_l']
adv_w = batch['advantage_w']
adv_l = batch['advantage_l']
adv_max = maximum(adv_w, adv_l)
margin = alpha * (lpw - lpl)
clamped_margin = clamp(margin, -10, 10)
smooth_margin = tanh(clamped_margin)
hinge = relu(1 - smooth_margin)
weighted_loss = hinge * abs(adv_max)
if weight exists:
  weighted_loss *= weight
final_loss = logsumexp(weighted_loss) - log(sum(weight_or_ones))
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
    adv_max = ops.maximum(adv_w, adv_l)
    margin = ops.sub(lpw, lpl)
    scaled_margin = ops.mul(alpha, margin)
    clamped_margin = ops.clamp(scaled_margin, -10.0, 10.0)
    smooth_margin = ops.tanh(clamped_margin)
    hinge = ops.relu(ops.sub(ops.ones_like(smooth_margin), smooth_margin))
    weight_factor = ops.abs(adv_max)
    weighted_loss = ops.mul(hinge, weight_factor)
    if weight is not None:
        weighted_loss = ops.mul(weighted_loss, weight)
        normalizer = ops.sum(weight)
    else:
        normalizer = ops.sum(ops.ones_like(weighted_loss))
    max_loss = ops.max(weighted_loss)
    shifted = ops.sub(weighted_loss, max_loss)
    sum_exp = ops.sum(ops.exp(shifted))
    lse = ops.add(max_loss, ops.log(sum_exp))
    return ops.div(lse, normalizer)
```
