# H9: Source-Normalized ASW Weighting Interaction for JSSP50x20

## Status

Conditionally preregistered while the sole H8 leader continuation is running.
Launch only if absolute steps450 and500 do not improve the locked H8 step250
validation32 mean `3083.937515`. If H8 improves that lock, run its prescribed
paired inference instead and do not launch H9.

## Observed Motivation

H7 source-normalized sparse USW matched BOPO's gradient scale but missed its
mean by `3.062500`. H8 then added mean-one relative-gap temperature and reached
`3083.937515`, beating matched BOPO `3085.312515`, but its paired validation
effect was uncertain: 17 wins, one tie, 14 losses, bootstrap 95% CI
`[-9.749956, 7.218765]`, Holm-adjusted p `0.966267`. Its fixed continuation
through absolute step400 has not improved the step250 lock.

The remaining preregistered failure class is whether ASW's bounded
gap-regret weights can make the source-normalized sparse objective more
consistent across validation instances without changing link curvature,
trajectory scale, pair geometry, rollout semantics, or optimizer scale.

## Hypothesis

Use H7's source-length-calibrated sigmoid loss unchanged, but replace uniform
weights with H5 ASW's gap-regret-blend weights on the same 15 rank-stratified
best-anchor pairs. This isolates the ASW weighting interaction at matched
source scale. It is a separately labelled source-normalized ASW variant, not
the original dense ASW claim and not a retry of H4-H8.

## Locked Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Seed and ordered dynamic training stream: `12345678`, start index 0.
- Validation stream: identical locked H3-H8 validation32 stream.
- JSSP50x20, FP32, physical instance batch one.
- B=128 rollouts and K=16 selected candidates per instance.
- Exactly 15 rank-stratified best-anchor pairs, strictly instance-local.
- H5 ASW builder is byte-for-byte unchanged, including its bounded
  gap-regret-blend weight formula.
- H7 source-normalized loss is byte-for-byte unchanged: source factor 100,
  global mean absolute advantage normalization, sigmoid link, and normalized
  weighted aggregation.
- Alpha is fixed to `0`, the validation-leading dense ASW calibration and the
  H5 sparse-ASW setting; no alpha sweep is allowed.
- Fresh Adam, LR `1e-5`, weight decay `1e-6`. This matches H7/H8's
  source-normalized first-order scale and is not a new LR search.
- Fixed matched BOPO threshold: `3085.312515`; lower is better.
- No paired inference, TA, DMU, or final-test data may be read for selection.

## Stages and Gates

1. Run exactly one fresh-root update. Require step0 and step1 validation,
   finite loss, finite strictly positive gradient, B128/K16, physical batch
   one, exactly 15 instance-local pairs, and a clean exact-log error scan.
2. Only after smoke passes, run one fresh-root 300-update screen with
   validation at steps0/50/100/150/200/250/300.
3. Continue only a validation checkpoint strictly below `3085.312515`.
   Otherwise H9 is negative and must not be retried or extended.
4. Paired validation inference is allowed only after a mean-gate win and uses
   the already locked bootstrap/Wilcoxon/Holm family.

## Failure Interpretation

If H9 misses BOPO, bounded ASW gap-regret weighting does not rescue the
source-normalized sparse objective. The next bounded family must change link
curvature; it must not repeat H4-H9 LR, weight decay, alpha, geometry,
trajectory normalization, or relative-gap-temperature settings.
