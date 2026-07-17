# Large-Scale FFSP/JSSP Findings

## Research Question

Can original USW and ASW, from an identical strong checkpoint and fresh optimizer, outperform matched PO and BOPO at FFSP1000 and JSSP50x20 without materially degrading absolute solution quality?

## Current Understanding

Architectural scale transfer is not the main risk. MatNet parameters depend on the fixed three-stage/four-machine structure rather than the number of jobs, and MGL parameters depend on feature dimensions rather than the number of jobs or machines. The main risks are target-scale rollout/backward memory, throughput, objective saturation, and preserving a fair common initialization.

The clean comparison is checkpoint-backed continuation: select one strong common checkpoint per problem using only a fresh target validation set, clone it exactly, reset Adam, and expose every method to the same ordered target instances. This follows the successful routing pattern while removing initialization and optimizer-state confounds.

## Key Results

- FFSP100 USW loaded into an FFSP1000 environment and encoded a `(1,1000,12)` instance without parameter mismatch.
- JSSP15x15 USW completed full greedy construction at 50x20 and 100x20 with B=1.
- RL/PO/BOPO/USW/ASW checkpoint core parameter names and tensor shapes are identical within FFSP and within JSSP.

## Patterns and Insights

- Large scheduling scale should be measured in jobs/operations, not copied mechanically from routing node counts.
- Zero-shot compatibility does not establish trainability at B=128 or 24 starts; backward probes are mandatory.
- A fixed source-scale epoch count is misleading because target rollouts have very different costs. Optimizer-update gates plus validation promotion provide a more defensible continuation budget.

## Lessons and Constraints

- FFSP's current paper evaluation loader restores `num_job=100`; it must rebuild the environment for FFSP1000.
- JSSP checkpoint `allowed_shapes=[[15,15]]` does not block direct benchmark inference but must be overridden for target-scale dataloaders/continuation.
- Final tests cannot be used for checkpoint or hyperparameter selection.
- Any PO-anchored or otherwise mixed objective must be labeled a variant and cannot substitute for the original USW/ASW claim.

## Open Questions

- Which source checkpoint gives the best locked zero-shot target validation mean and should become the common initialization?
- What physical/effective batch fits FFSP1000 with 24 starts and JSSP50x20 with B=128?
- Do original USW and ASW retain useful gradient signal at the target scales, or do they require only alpha/LR calibration?

## Optimization Trajectory

No target-scale optimizer update has been run yet. H0 feasibility probes are next.
