# H13: H8-to-ASW Micro-Adaptation for JSSP50x20

## Status

Preregistered manually after the recurring automation was paused and the user
requested a small ASW continuation from the H8 leader. This is a sequential
hybrid experiment, not evidence that original ASW independently beats BOPO.

## Observed Motivation

H8 USW is the matched-mean family leader at absolute step500: validation32
`3080.624992`, `4.687523` below matched BOPO `3085.312515`. H8 nevertheless
failed paired inference because its advantage is heterogeneous across
instances. H10's ASW-weighted source-normalized objective did not win when
trained from the original common checkpoint, but its bounded instance-local
weights may act as a gentle adapter when initialized from H8's strong policy.

## Hypothesis

Initialize from H8 step500 and run only 50 fresh-Adam updates using H10's
byte-identical source-normalized ASW pair artifact. A tenfold lower LR than H8
(`1e-6`) should preserve the H8 policy while allowing ASW weights to make a
small corrective move.

Two conclusions are separated:

- Retention: a positive-update checkpoint below BOPO `3085.312515` shows that
  the H8-to-ASW hybrid retains a matched-mean advantage.
- Contribution: a checkpoint below the H8 initialization `3080.624992` is
  required to claim that ASW micro-adaptation improved the H8 policy.

## Locked Protocol

- Initial checkpoint:
  `logs/scheduling_large_scale/h8_relative_gap_temperature/jssp50x20/continue500_from_step250_20260718_0655/usw_relative_gap_temperature/best.ckpt`.
- The checkpoint must reproduce validation32 `3080.624992` at micro-step0.
- Fresh Adam, LR `1e-6`, weight decay `1e-6`.
- Method `asw`, alpha `0`.
- ASW artifact:
  `research/scheduling_large_scale/experiments/h9_jssp_source_normalized_asw/artifacts/asw_source_normalized/best_pair.json`.
- Seed `12345678`; dynamic training stream starts at index500 so H8's first
  500 instances are not replayed.
- JSSP50x20 FP32, B=128, K=16, physical instance batch one.
- Exactly 15 rank-stratified pairs, strictly instance-local.
- Validation uses the identical locked validation32 stream at micro-steps
  0/10/20/30/40/50.
- No LR, alpha, weight-decay, checkpoint, or budget sweep is allowed.
- No TA, DMU, or final-test data may be read for selection.

## Stages and Gates

1. Run exactly one fresh-root update on an otherwise idle GPU. Require exact
   step0 reproduction, finite loss, finite strictly positive gradient,
   B128/K16, physical batch one, exactly 15 local pairs, and a clean log.
2. Only if smoke passes, run one fresh-root 50-update micro-screen.
3. Select only positive-update scheduled validation gates. Report retention
   against BOPO and contribution against H8 separately.
4. Paired validation inference is allowed only if a positive-update checkpoint
   is below BOPO. A paper-facing claim still requires a bootstrap CI wholly
   below zero and Holm-adjusted Wilcoxon `p<0.05`.

## Failure Interpretation

If all positive-update gates exceed BOPO, ASW micro-adaptation destroys H8's
mean advantage. If they remain below BOPO but above H8, ASW preserves rather
than improves the inherited USW policy. Only a gate below H8 supports a direct
ASW contribution claim.
