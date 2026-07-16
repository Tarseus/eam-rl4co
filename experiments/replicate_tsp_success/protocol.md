# Replicate the successful TSP1000 route on CVRP1000

Status: confirmatory protocol locked before remote preflight and launch.

## Hypothesis

The TSP1000 result transferred poorly when CVRP1000 USW/ASW restored objective-specific CVRP100 Adam states at LR `3e-4`. Starting all objectives from the same CVRP100 PO weights with fresh Adam, LR `1e-5`, large effective instance batches and a strong PO anchor should preserve the stable PO direction while allowing USW and ASW to improve per-instance route selection.

## Common initialization and optimizer

- Checkpoint: `downloads/cvrp100/po/checkpoint.ckpt` for PO, USW and ASW.
- Fresh Adam for every branch; do not restore checkpoint optimizer state when changing objective.
- LR `1e-5`, weight decay `1e-6`, BF16 mixed training.
- Capacity50 dynamic CVRP1000, seed1234.
- Effective instance batch128. Probe physical batch4 first, then choose the largest common safe physical batch in `{4,2,1}` and set accumulation to `128 / physical_batch`.
- Candidate pools, pairs, weights, anchors and losses are computed per instance before averaging over the physical instance batch.

## Branch definitions

- PO: exponential PO, K64, PO alpha `0.05`.
- USW-PO-anchor: uniform dense all-pairs builder, K64, preference alpha `0.01`, PO anchor weight `0.90`, PO anchor alpha `0.05`.
- ASW-PO-anchor: adaptive gap-square weighting, K40, preference alpha `0.01`, detached preference weights, PO anchor weight `0.90`, PO anchor alpha `0.05`.

## Short screen and step0 gate

- Run 100 optimizer updates for all three branches.
- Use the same matched dynamic instance indices `300000000..300012799` in every branch.
- Validation is the locked AGFN capacity50 validation32 with 100 starts and augmentation1 at continuation step0 and step100.
- The three step0 per-instance validation vectors must be exactly equal. A mean-only match is insufficient.
- Require finite losses and gradients, forced replay error at most `1e-5`, no OOM/Traceback/nonfinite values, and no output reuse or optimizer-step reset.
- Short-screen output directories are immutable and unique per branch and launch attempt.

## Promotion and continuation

- Lock candidates only after the short screen passes all integrity gates.
- Resume each selected branch from its own verified step100 `last.ckpt`, including its fresh-Adam state.
- Continue for 900 additional optimizer updates to absolute continuation step1000.
- Start promoted-run data at index `300012800`; no short-screen instance may repeat.
- Keep branch definitions, LR, batch semantics, replay checks and validation32 protocol unchanged.

## Final evaluation and success

- Only after candidate lock, evaluate the exact official AGFN capacity50 fixed128 dataset with ordered indices0..127, 100 starts, 8 augmentations and FP32.
- Verify checkpoint SHA, dataset SHA, configuration, exact row coverage and route feasibility.
- Compare USW and ASW separately with the same-start, same-continuation-budget PO branch using aligned per-instance costs and a paired bootstrap.
- Success requires both candidate-minus-PO 95% confidence intervals to lie wholly below zero. Otherwise continue evidence-driven matched experiments or conclude negatively after sufficiently powered failures.
