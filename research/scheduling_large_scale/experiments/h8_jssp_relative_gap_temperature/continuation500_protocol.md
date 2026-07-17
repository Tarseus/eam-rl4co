# H8 Leader Continuation to Absolute Step 500

## Status

Preregistered after H8 step250 beat the matched BOPO validation32 mean but
failed the locked paired-inference gate. This continuation changes no method,
loss, builder, optimizer, data, rollout, or validation parameter.

## Observed Basis

- Matched BOPO best: `3085.312515` at step500.
- H8 screen best: `3083.937515` at step250, a mean improvement of `1.375000`.
- H8 paired result: 17 wins, one tie, 14 losses; bootstrap 95% CI
  `[-9.749956, 7.218765]`; Holm-adjusted p `0.966267`.
- H8 step300 regressed to `3092.625008`, so the validation-selected step250
  checkpoint remains the only eligible resume source.

## Hypothesis

H8's matched-scale relative-gap objective produced a late improvement at
step250, as did earlier sparse USW variants. Preserving its exact optimizer
state and extending only this mean-gate leader to BOPO's absolute 500-update
budget may produce a stronger scheduled validation checkpoint whose paired
effect is distinguishable from zero.

## Locked Continuation

- Resume exactly
  `logs/scheduling_large_scale/h8_relative_gap_temperature/jssp50x20/screen300_20260718_0620/usw_relative_gap_temperature/best.ckpt`,
  whose payload records optimizer step250 and best cost `3083.937515`.
- Add exactly 250 updates, ending at absolute optimizer step500.
- Preserve optimizer state; do not create fresh Adam.
- Preserve LR `1e-5`, weight decay `1e-6`, seed `12345678`, data start index
  `0`, common checkpoint, FP32, physical batch one, B=128, and K=16.
- Preserve H8's source-length factor 100, mean-one pair-relative gap
  temperature, and byte-identical H5 rank-stratified 15-edge builder.
- All pairs remain strictly instance-local.
- Validate on the identical locked validation32 stream every 50 steps through
  absolute steps300/350/400/450/500.
- Use one fresh, collision-free output root. Never append to the screen root.

## Gates and Decision Rule

Require 250 unique contiguous updates at absolute steps251-500, finite loss,
finite strictly positive gradients, B128/K16, physical batch one, exactly 15
instance-local pairs, and a clean exact-log error scan.

The existing step250 lock remains selected unless a scheduled continuation
mean is strictly below `3083.937515`. Repeat paired validation inference only
for a newly improved lock. If no later gate improves the lock, H8 remains a
mean-only, statistically inconclusive result and is not extended again; the
next bounded hypothesis is the separately labelled source-normalized ASW
weighting interaction described by H8's failure plan.

Do not read TA, DMU, or final-test data during this continuation or selection.
