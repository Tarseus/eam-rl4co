# H5: Rank-Stratified Anchor Geometry for JSSP50x20

## Status

Preregistered after H4 Phase B exhausted all six dense USW/ASW calibration
settings without beating the matched BOPO validation32 minimum.

## Hypothesis

At JSSP50x20, the discovered USW/ASW formulas are not primarily limited by
learning rate, weight decay, or the tested ASW regret exponents. Their dense
approximately 8,100-pair update geometry dilutes the rank-stratified anchor
signal that makes matched BOPO stable. Holding the discovered loss and ASW
weight formula fixed while replacing only pair construction with the same
instance-local rank-stratified 15-edge anchor star will improve validation32.

This is explicitly an **anchor-geometry USW/ASW variant**, not the original
dense objective.

## Locked Common Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Seed and ordered dynamic training stream: `12345678` and data start index 0.
- Validation stream: the identical locked 32 instances used by H3/H4.
- JSSP shape: 50x20; FP32; physical instance batch 1.
- Rollouts remain B=128 per instance and K=16.
- Fresh Adam; no optimizer state is transferred.
- Every pair is constructed strictly within its source instance.
- No TA, DMU, or final-test data may be read for selection.
- Fixed matched BOPO threshold: validation32 mean `3085.312515`; lower is
  better.

## Single Mechanistic Change

Sort the 128 candidates within each instance. Select ranks
`0,8,16,...,120`, matching the native BOPO `B/K` stride. Construct the 15
pairs `(rank 0, rank r)` for `r=8,...,120`.

- USW variant: retain the original discovered USW loss; use uniform weights.
- ASW variant: retain the original discovered loss and gap-regret-blend weight
  formula; compute those weights only on the 15 locked anchor pairs.

No candidate crosses instance boundaries, and the rollout pool is not reduced.

## Preregistered Variants and Gates

Family-leading dense calibrations are transferred without a new sweep:

1. USW anchor geometry: LR `5e-7`, WD `1e-6`.
2. ASW anchor geometry: LR `1e-6`, WD `1e-6`, alpha `0`.

Stage A is a sequential one-update smoke for both variants. Each must have one
step-0 validation, one finite loss, a finite strictly positive gradient, exactly
15 instance-local pairs, B128/K16, physical batch one, and a clean exact-log
error scan.

If both smokes pass, Stage B runs each variant in a fresh root for 300 updates,
with validation at steps 0/50/100/150/200/250/300. A variant is a leader only
if its mean is strictly below `3085.312515`. Only a leader may continue from
its validation-selected checkpoint to the matched 500-update gate and paired
validation inference. A non-leader is stopped and never retried.

## Failure Interpretation

If both anchor variants miss BOPO, sparse geometry alone is ruled out under the
family-leading calibrations. The next hypothesis must be separately
preregistered and bounded; H4 dense settings and H5 roots must not be reused.
