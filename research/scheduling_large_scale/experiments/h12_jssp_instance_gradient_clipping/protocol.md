# H12: Instance-Level Gradient Clipping for JSSP50x20

## Status

Preregistered only after H11 completed cleanly without beating the fixed matched
BOPO threshold. Launch exactly one smoke, then one screen only if the smoke
passes. Never reuse an existing output or log target.

## Observed Motivation

H8's linear relative-gap-temperature USW is the family mean leader at step500
(`3080.624992` versus matched BOPO `3085.312515`) but fails paired inference:
only 15/32 instances improve and the bootstrap interval crosses zero. H11's
square-root temperature reduces gap emphasis and reaches a boundary-close
step250 mean of `3085.437469`, but misses BOPO by `0.124954` and regresses at
step300. H11 has mean gradient `0.520051`, yet individual instance updates can
spike far above that scale (maximum observed pre-clip norm exceeds `6`).

Because physical batch is one, every optimizer update is driven by exactly one
JSSP instance. A fixed gradient-norm cap is therefore an explicit
instance-level robustness intervention: it limits how strongly an atypical
instance can move the shared policy without altering candidate selection,
pair construction, or the per-instance objective.

## Hypothesis

Apply `clip_grad_norm_(parameters, 1.0)` after the finite/nonzero pre-clip
gradient audit and before each Adam step while restoring H8's byte-identical
linear relative-gap-temperature loss and 15-edge builder. This should preserve
H8's mean optimization signal while reducing update outliers that contribute
to cross-instance response heterogeneity.

This is a separately labelled robust-training USW variant, not the original
USW claim and not an H8 or H11 retry.

## Locked Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Seed and ordered dynamic training stream: `12345678`, start index 0.
- Validation stream: identical locked H3-H11 validation32 stream.
- JSSP50x20, FP32, physical instance batch one.
- B=128 rollouts and K=16 selected candidates per instance.
- H8 loss and H5 builder artifacts are reused byte-for-byte.
- Exactly 15 rank-stratified best-anchor pairs, strictly instance-local.
- The only optimization change is max gradient norm `1.0` per physical
  instance update. Pre-clip and post-clip norms must both be logged.
- Fresh Adam, LR `1e-5`, weight decay `1e-6`.
- Fixed matched BOPO threshold: `3085.312515`; lower is better.
- H8 step500 `3080.624992` remains the family mean leader.
- No paired inference, TA, DMU, or final-test read is allowed for selection.

## Stages and Gates

1. Run exactly one fresh-root update. Require one step-0 and one step-1
   validation, finite loss, finite strictly positive pre/post-clip gradients,
   `post_clip <= 1.000001`, B128/K16, physical batch one, exactly 15 local
   pairs, and a clean exact-log error scan.
2. Only after smoke passes, run one fresh-root 300-update screen with
   validation at steps0/50/100/150/200/250/300.
3. Continue to the matched step500 budget only if a scheduled screen checkpoint
   is strictly below `3085.312515`. Otherwise H12 is negative and must not be
   retried or continued.
4. Run paired validation inference only for a validation-selected checkpoint
   below BOPO. A paper-facing win still requires a bootstrap CI wholly below
   zero and Holm-adjusted Wilcoxon `p<0.05`.

## Failure Interpretation

If H12 misses BOPO or remains paired-inconclusive, clipping whole-instance
update magnitude is insufficient. The next bounded robustness hypothesis must
target within-instance pair influence (for example, a preregistered robust
pair-loss aggregation) while preserving H8's loss signal, local builder, and
matched protocol. Do not repeat H4-H12 LR, weight decay, alpha, pair geometry,
source normalization, ASW weighting, temperature-curvature, or gradient-cap
settings.
