# H8: Mean-Preserving Relative-Gap Temperature for JSSP50x20

## Status

Preregistered only after H7 completed below the fixed matched-BOPO gate. Launch
exactly one smoke, then one screen only if the smoke passes. Never reuse an
existing output or log target.

## Observed Motivation

H7 corrected the 10x source-to-target trajectory-length mismatch: its 300-step
mean gradient was `0.517274`, close to matched BOPO's `0.569430`. Its best
validation32 mean nevertheless remained `3088.375015`, which is `3.062500`
above BOPO `3085.312515`.

On the deterministic first training instance, H7's 15 local pairs had mean
absolute advantage gap `111.399`, relative objective gaps from `0.0139` to
`0.0637`, H7 logit mean `-0.025924`, and BOPO-form logit mean `-0.029905`.
Thus global link scale is already matched closely, but H7 applies one global
gap divisor to every pair despite materially different objective separations.

## Hypothesis

Preserve H7's source-length-normalized base logit and its global mean scale,
but multiply the base logit by each pair's absolute advantage gap divided by
the within-instance mean absolute gap:

`base = 100 * (log_prob_w_mean - log_prob_l_mean) / mean_abs_gap`

`temperature = abs(advantage_gap) / mean_abs_gap`

`logit = base * temperature`

The temperature has mean one by construction. It should keep first-order scale
near H7/BOPO while placing more preference pressure on the widest-quality
comparisons among the same 15 local anchors. This is a relative-gap-temperature
USW variant, not the original USW claim and not a BOPO replacement.

## Locked Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Seed and ordered dynamic training stream: `12345678`, start index 0.
- Validation stream: identical locked H3-H7 validation32 stream.
- JSSP50x20, FP32, physical instance batch one.
- B=128 rollouts and K=16 selected candidates per instance.
- Exactly 15 rank-stratified best-anchor pairs, strictly instance-local.
- H5/H7 builder remains byte-for-byte unchanged.
- The only change from H7 is the mean-one pair-relative advantage temperature
  inside the sigmoid link. Source-length factor 100, uniform weights, normalized
  weighted aggregation, optimizer, and all data streams remain fixed.
- Fresh Adam, LR `1e-5`, weight decay `1e-6`.
- Fixed matched BOPO threshold: `3085.312515`; lower is better.
- No paired inference, TA, DMU, or final-test read is allowed for selection.

## Stages and Gates

1. Run exactly one fresh-root update. Require one step-0 and one step-1
   validation, finite loss, finite strictly positive gradient, B128/K16,
   physical batch one, exactly 15 instance-local pairs, and a clean exact-log
   error scan.
2. Only after smoke passes, run one fresh-root 300-update screen with
   validation at steps 0/50/100/150/200/250/300.
3. Continue only a validation checkpoint strictly below `3085.312515`.
   Otherwise H8 is negative and must not be retried or continued.

## Failure Interpretation

If H8 misses BOPO, stronger pair-relative objective-gap emphasis at matched
mean scale is not sufficient. The next bounded family must change link
curvature or return to a separately labelled ASW source-normalized weighting
interaction; it must not repeat H4-H8 LR, weight decay, geometry, trajectory
scale, or relative-gap-temperature settings.
