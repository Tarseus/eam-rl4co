# H1 Protocol: 100-Update Common-Checkpoint Screen

Status: confirmatory, locked before execution.

## Question

From one target-selected common checkpoint and fresh Adam, do original USW and
ASW show a better 100-update validation trajectory than matched PO and BOPO at
FFSP1000 and JSSP50x20?

## Common Conditions

- Methods: PO, BOPO, original USW, original ASW.
- Seed: 12345678; data start index: 0.
- Optimizer: fresh Adam, learning rate 1e-5, weight decay 1e-6.
- Updates: 100; gradient accumulation: 1 physical instance per update.
- Validate at step 0 and step 100 on the same eight dynamically generated
  validation instances. These are not final test instances.
- No method may resume optimizer state or read final fixed paired test data.

## FFSP1000

- Common checkpoint: `downloads/ffsp100/loss_only.ckpt`, policy SHA
  `d92e8319b91406c0a80d1844b6fc42c419b8bd71d66a35e0f56d9d1a6d9dca83`.
- Three stages, four machines per stage, exactly 24 machine-order starts.
- BF16 autocast with cuDNN SDPA disabled before model construction.
- Validation uses 24 starts, augmentation 1, count 8, batch size 8.

## JSSP50x20

- Common checkpoint: `downloads/jssp15x15/weighting/checkpoint.ckpt`, policy SHA
  `5a90dc4027c08ef7bb2192a84a752a638bd4c59c46a24bf2a57c3721baaf8584`.
- FP32, B=128, K=16, one physical instance per update.
- Validation uses 128 rollouts per instance, count 8, batch size 8.
- Every BOPO anchored pair and every USW/ASW pair stays within the current
  physical instance.

## Measurements

- Step-0 and step-100 validation mean cost.
- Per-update loss, gradient norm, elapsed time, peak allocated/reserved memory,
  candidate count, and pair count where applicable.
- Initial policy SHA and `fresh_optimizer=true` must match within each problem.
- Scan logs for OOM, NaN, Inf, traceback, segmentation fault, and killed jobs.

## Gate

Do not inspect final tests. After all four methods finish for a problem, compare
their step-100 fixed-validation means and change from their common step-0 mean.
Methods with finite stable trajectories remain eligible for the locked
500-update gate; any pruning or LR adjustment must be justified from validation
only and recorded before execution.
