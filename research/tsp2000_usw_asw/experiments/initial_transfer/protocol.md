# Initial TSP2000 transfer protocol

Status: confirmatory protocol locked before execution.

## Hypothesis

The TSP1000 USW K64/alpha=0.01 advantage transfers to TSP2000 because the larger within-instance candidate pool continues to improve preference estimation at the larger graph size.

## Training

- Resume the verified TSP1000 USW K64/alpha=0.01 optimizer-step1000 `last.ckpt`.
- Train on dynamic TSP2000 instances for 1000 additional optimizer updates.
- Use K64, alpha 0.01, batch32, BF16, seed1234, fresh `data_start_index=40000000`.
- Run a matched PO K20 control from its equal-budget TSP1000 endpoint with the same batch, seed, data start, and 1000-update budget.
- First run a bounded memory smoke probe; a smaller common batch is allowed only if batch32 does not fit.

## Evaluation

- Proxy: fixed validation16, 1000 starts, augmentation1 every 100 steps.
- Final: aligned fixed100, 1000 starts, 8 augment, FP32.
- USW succeeds only if mean is below PO and paired bootstrap candidate-minus-PO 95% CI is wholly below zero.
- Dataset hash and indices 0..99 must be recorded before interpreting results.

