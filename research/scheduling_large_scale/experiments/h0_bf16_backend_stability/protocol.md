# H0 Addendum: FFSP1000 BF16 Backend Stability

Status: confirmatory, locked before execution.

## Observation

FFSP1000 BOPO and USW completed a finite BF16 update and the post-update
validation, then exited with signal 11. Both runs emitted the same cuDNN SDPA
backward stride-materialization warning. Their step-1 checkpoints were already
saved, so numerical feasibility and process teardown are separate questions.

## Hypothesis

The exit-time segmentation fault is caused by the cuDNN SDPA backend rather
than the FFSP objective or checkpoint. Disabling only cuDNN SDPA should retain
the BF16 memory advantage and produce a clean process exit.

## Fixed Test

- Rerun FFSP1000 BOPO from the locked common `loss_only.ckpt` checkpoint.
- Keep fresh Adam, seed 12345678, 24 starts, one physical instance, one update,
  BF16 autocast, LR, weight decay, and one-instance pre/post validation fixed.
- Before importing the training entry point, call
  `torch.backends.cuda.enable_cudnn_sdp(False)`; do not change objective logic.
- Do not read final test data.

## Pass Criteria

- Finite loss and nonzero finite gradient norm.
- Exactly 24 candidates for the physical instance.
- Peak memory below the 24 GiB device limit.
- Step-1 checkpoint and post-update validation are written.
- Process exits with code 0 and no OOM, NaN, Inf, traceback, or segmentation
  fault.

## Decision

If the fixed test passes, use cuDNN-SDPA-disabled BF16 for all FFSP1000
training and validation jobs. If it OOMs or still segfaults, return to the outer
loop for a different memory-safe backend or rematerialization strategy before
any 100-update screen.
