# H0 Protocol: Target-Scale Feasibility

Status: confirmatory, locked before remote execution.

## Question

Can the existing FFSP and JSSP networks execute one target-scale training update on a 24 GiB GPU while preserving the intended candidate-pool semantics?

## Fixed Conditions

- One physical instance per microbatch.
- Fresh Adam; no optimizer state restored.
- FFSP1000: 3 stages, 4 machines/stage, 24 starts, FP32 first.
- JSSP50x20: B=128, K=16, FP32 first.
- PO, BOPO, USW and ASW use the same exact common checkpoint bytes within each problem.
- USW and ASW use the frozen source objective artifacts without semantic modification.
- No final test data are read.

## Probe Order

1. Evaluate source PO/BOPO/USW/ASW checkpoints on a fresh small target validation subset to select one common checkpoint per problem.
2. Run one forward/backward update for PO, BOPO, USW and ASW from that common checkpoint.
3. Record loss, gradient norm, elapsed time, peak allocated/reserved memory, candidate/pair count, and finite-value checks.
4. Verify cloned models are tensor-for-tensor identical before their objective-specific update.

## Pass Criteria

- No OOM, NaN, Inf, or traceback.
- Finite nonzero gradient norm.
- All methods start from identical network tensors.
- FFSP pools contain exactly 24 candidates for one instance.
- JSSP pools contain exactly B=128 candidates for one instance; K=16 filtering and all pairs remain instance-local.
- If FP32 fails only for memory, BF16 may be tested as an explicitly recorded engineering adaptation.

## Promotion

Only passing methods proceed to the locked 100-update validation screen. Physical batch remains one unless a separate capacity probe proves a larger batch safe.
