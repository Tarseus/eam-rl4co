# H7: Source-Length-Calibrated USW for JSSP50x20

## Status

Conditionally preregistered while the sole authoritative H6 screen is still
running. Launch only if H6 completes without a validation32 mean strictly
below matched BOPO `3085.312515`, after confirming no active H7 trainer and
fresh output/log targets.

## Observed Motivation

H5 sparse USW preserved the discovered loss and BOPO-matched 15-edge geometry,
but its mean gradient norm was `5.285728` versus matched BOPO `0.569430`, a
ratio of about `9.28`. JSSP50x20 trajectories contain 1000 decisions versus
100 decisions in the JSSP10x10 source regime where the USW loss was discovered.
The loss uses total-trajectory log-probability differences, so its gradient
scale is expected to grow with trajectory length. H6 matched first-order update
scale by doubling LR without correcting this loss curvature; through step 200
its best scheduled validation32 mean was `3089.781181`, still `4.468666` above
BOPO.

## Hypothesis

Replace the USW total-log-probability difference with a source-length-calibrated
mean-log-probability difference:

`100 * (log_prob_w_mean - log_prob_l_mean)`

This is equivalent to dividing the original total-log-probability difference
by `seq_len / 100`, which is exactly 10 for JSSP50x20. Holding the original
advantage-gap normalization and H5 pair geometry fixed should reduce the H5
gradient scale to approximately BOPO's. LR `1e-5` is then mechanically chosen
to keep the expected LR-times-gradient scale near `5e-6`, rather than forming
a new LR sweep.

This is explicitly a trajectory-normalized USW variant, not the original USW
claim and not a retry of H4-H6.

## Locked Protocol

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`.
- Seed and ordered dynamic training stream: `12345678`, start index 0.
- Validation stream: the identical locked H3-H6 validation32 stream.
- JSSP50x20, FP32, physical instance batch one.
- Per-instance rollout semantics remain B=128 and K=16.
- Exactly 15 rank-stratified best-anchor pairs, strictly instance-local.
- H5 builder is byte-for-byte unchanged.
- The only loss change is total log-probability difference to
  `100 * mean-log-probability difference`; advantage-gap normalization,
  sigmoid link, uniform weights, and normalized weighted aggregation remain.
- Fresh Adam, LR `1e-5`, weight decay `1e-6`.
- With physical batch one, the scalar loss is exactly the required per-instance
  loss before averaging; no cross-instance pair or reduction is possible.
- Fixed matched BOPO threshold: `3085.312515`; lower is better.
- No paired inference, TA, DMU, or final-test read is allowed for selection.

## Stages and Gates

1. Run exactly one fresh-root update. Require one step-0 and one step-1
   validation, finite loss, finite strictly positive gradient, B128/K16,
   physical batch one, exactly 15 instance-local pairs, and a clean exact-log
   error scan.
2. Only after the smoke passes, run one fresh-root 300-update screen with
   validation at steps 0/50/100/150/200/250/300.
3. Continue from a validation-selected H7 checkpoint only if its mean is
   strictly below `3085.312515`. Otherwise H7 is negative and must not be
   retried or continued.

## Failure Interpretation

If H7 misses BOPO, the source-to-target trajectory-length mismatch is not the
remaining cause under sparse local anchors and BOPO-matched first-order update
scale. The next bounded hypothesis must change the advantage-gap normalization
or link shape, not repeat H4-H7 LR, weight decay, alpha, geometry, or trajectory
normalization settings.
