# H6: Sparse-USW Update-Scale Calibration for JSSP50x20

## Status

Conditionally preregistered while the final H5 ASW screen is still running.
Launch only after the authoritative H5 queue exits naturally. Do not launch if
another H6 root or trainer already exists.

## Observed Motivation

H5 USW with the BOPO-matched rank-stratified 15-edge anchor geometry completed
300 clean updates at LR `5e-7`. Its validation32 curve improved at every gate
from step 150 through step 300:
`3091.281303 -> 3090.812531 -> 3088.875000 -> 3088.406235`. The terminal
checkpoint is still `3.093720` above matched BOPO `3085.312515`, so it is not a
leader and cannot be continued.

The 300-step H5 USW mean gradient norm was `5.285728`, versus `0.569430` over
the matched BOPO run. The mean LR-times-gradient scale is therefore about
`2.64e-6` for H5 USW and `5.69e-6` for BOPO. This suggests that transferring
the dense-family LR into the sparse geometry under-scaled the optimizer update.

## Hypothesis

Holding the discovered USW loss and H5 rank-stratified 15-edge instance-local
pair geometry fixed, increasing LR from `5e-7` to `1e-6` will approximately
match BOPO's observed first-order update scale and reach a validation32 mean
strictly below `3085.312515` within 300 updates.

This is a single, bounded sparse-USW update-scale calibration. It is not a
continuation of the negative H5 checkpoint and it is not a dense-pair retry.

## Locked Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Seed and ordered dynamic training stream: `12345678`, start index 0.
- Validation stream: the identical locked H3/H4/H5 validation32 stream.
- JSSP50x20, FP32, physical instance batch one.
- Per-instance rollout semantics remain B=128 and K=16.
- Exactly 15 rank-stratified best-anchor pairs, all strictly instance-local.
- H5 USW loss and H5 USW builder artifacts are unchanged.
- Fresh Adam, LR `1e-6`, weight decay `1e-6`.
- Fixed matched BOPO threshold: `3085.312515`; lower is better.
- No TA, DMU, paired inference, or final-test read is allowed for selection.

## Stages and Gates

1. Run one fresh-root update. Require one step-0 and one step-1 validation,
   finite loss, finite strictly positive gradient, B128/K16, physical batch
   one, exactly 15 instance-local pairs, and a clean exact-log error scan.
2. Only after the smoke passes, run one fresh-root 300-update screen with
   validation at steps 0/50/100/150/200/250/300.
3. Continue from a validation-selected H6 checkpoint only if its mean is
   strictly below `3085.312515`. Otherwise H6 is negative and must not be
   retried or continued.

## Failure Interpretation

If H6 misses BOPO, sparse geometry plus matched first-order update scale is
insufficient. The next bounded hypothesis must change the discovered loss's
trajectory-scale normalization rather than repeat LR, weight-decay, alpha, or
pair-geometry settings already tested in H4-H6.
