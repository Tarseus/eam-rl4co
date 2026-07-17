# H10: Source-Normalized ASW Weighting After H8 Inference Failure

## Status

Preregistered after H8 completed its matched step500 budget and failed the
locked paired validation inference gate. H9 remains unexecuted because its
precondition (no H8 improvement after step250) was false. H10 carries the
already prepared byte-identical source-normalized ASW mechanics into a fresh
protocol whose observed basis is H8's step500 cross-instance inconsistency.

## Observed Motivation

H8 step500 reached validation32 `3080.624992`, `4.687523` below matched BOPO
`3085.312515`. The paired effect nevertheless failed: only 15 of 32 instances
favored H8, bootstrap 95% CI `[-13.374925, 3.187508]`, raw Wilcoxon
`p=0.568454`, and Holm-adjusted `p=0.966267`. The average gain is therefore
carried by relatively large improvements on a minority of instances.

H8 already rules in source-length normalization and pair-relative gap
emphasis for mean optimization. The next bounded variable is whether H5 ASW's
bounded gap-regret weights on the same local anchors distribute preference
pressure more consistently than uniform weights.

## Hypothesis

Combine H7's byte-identical source-length-calibrated sigmoid loss with H5's
byte-identical ASW gap-regret-blend builder and alpha `0`. Relative-gap
temperature is removed, so the single change from H7 is uniform-to-ASW pair
weighting. This should preserve BOPO-scale gradients while emphasizing
reliable gap/regret evidence within each instance, improving the validation
win count and paired uncertainty relative to H8.

This is a separately labelled source-normalized ASW variant. It is not the
original dense ASW claim, not an H8 retry, and not an H9 execution.

## Locked Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Fresh Adam; LR `1e-5`, weight decay `1e-6`.
- Seed `12345678`, ordered dynamic stream from index0, and the identical
  locked validation32 stream.
- JSSP50x20, FP32, physical instance batch one.
- B=128 rollouts and K=16 selected candidates per instance.
- Exactly 15 rank-stratified best-anchor pairs, strictly instance-local.
- H7 source factor100, global mean absolute advantage normalization, sigmoid
  link, and normalized weighted aggregation are byte-for-byte unchanged.
- H5 ASW builder and bounded gap-regret weights are byte-for-byte unchanged.
- Alpha is fixed at `0`; no alpha, LR, or weight-decay sweep is allowed.
- The committed artifact pair is
  `research/scheduling_large_scale/experiments/h9_jssp_source_normalized_asw/artifacts/asw_source_normalized/best_pair.json`;
  it was prepared before H8 step450 and never executed under H9.
- Mean screening threshold remains matched BOPO `3085.312515`; lower is
  better. H8's `3080.624992` remains the current USW family leader.
- No TA, DMU, or final-test data may be read for selection.

## Stages and Gates

1. Run exactly one fresh-root update. Require step0 and step1 validation,
   finite loss, finite strictly positive gradient, B128/K16, physical batch
   one, exactly 15 instance-local pairs, and a clean exact-log scan.
2. Only after the smoke passes, run exactly one fresh-root 300-update screen
   with validation at steps0/50/100/150/200/250/300.
3. Continue or run paired inference only for a checkpoint strictly below
   matched BOPO `3085.312515`. Otherwise H10 is negative and is not retried.
4. A paper-facing claim still requires the locked bootstrap CI wholly below
   zero and Holm-adjusted Wilcoxon `p<0.05`; mean selection alone is not a win.

## Failure Interpretation

If H10 misses BOPO or remains paired-inconclusive, bounded ASW weighting does
not solve the cross-instance heterogeneity. The next bounded family must
change link curvature or explicitly optimize instance-level robustness; it
must not repeat H4-H10 LR, WD, alpha, geometry, source normalization, or
relative-gap-temperature settings.
