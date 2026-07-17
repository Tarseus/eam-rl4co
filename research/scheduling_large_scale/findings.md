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
- The direct full-graph FFSP path completed one PO, BOPO, USW, and ASW update at a reduced 10-job smoke scale.
- The direct MGL path completed one PO, BOPO, USW, and ASW update at 5x5 with B=8/K=2 and physical batch one.
- Fresh target-validation checkpoint selection on eight generated instances locked FFSP1000 to `downloads/ffsp100/loss_only.ckpt` (mean 804.625; state SHA `d92e8319...d9dca83`) and JSSP50x20 to `downloads/jssp15x15/weighting/checkpoint.ckpt` (mean 3103.875061; state SHA `5a90dc40...af8584`). The final fixed test sets were not read.
- JSSP50x20 B=128/K=16 passed all four FP32 one-update probes from the same locked ASW checkpoint and fresh Adam. Update times were 4.40--4.47 seconds and peak allocated memory was 11.56--11.58 GiB.
- JSSP50x20 pair diagnostics stayed instance-local: BOPO formed 15 anchored pairs; USW and ASW each formed 8117 pairs from one 128-candidate physical instance. All four initial policy SHAs were identical.
- FFSP1000 PO, BOPO, USW, and ASW with 24 starts each exhausted a 24 GiB RTX 3090 during the FP32 backward pass (about 23.68 GiB in use). The pre-registered BF16 engineering adaptation completed successfully; no FFSP method is promoted beyond H0 yet.
- FFSP1000 BF16 produced finite nonzero-gradient updates for all four objectives with exactly 24 candidates in 47--49 seconds and about 18.04 GiB peak allocated memory. PO, BOPO, USW, and ASW gradient norms were 189.1890, 0.3448, 6.5276, and 5.4116 respectively.
- cuDNN SDPA caused a repeatable exit-time segmentation fault after otherwise complete BOPO/USW/ASW updates on GPU4. The locked BOPO rerun with only cuDNN SDPA disabled matched the original loss/gradient (0.689702/0.344631), used 17.94 GiB, wrote post-update validation/checkpoints, and exited cleanly. This backend is now fixed for FFSP1000.
- An exact FP32 fallback now samples without a graph and checkpoint-replays the same actions plus the same RandomOneHot encoder RNG state. At FFSP10 it matched the full-graph PO loss and gradient norm exactly; all four methods had zero replay error and instance-local pairing. At FFSP1000, PO replay retained all 24 starts, had zero replay error and a finite gradient, used only 0.215 GiB peak allocation, and took 89.4 seconds per update.

## Patterns and Insights

- Large scheduling scale should be measured in jobs/operations, not copied mechanically from routing node counts.
- Zero-shot compatibility does not establish trainability at B=128 or 24 starts; backward probes are mandatory.
- A fixed source-scale epoch count is misleading because target rollouts have very different costs. Optimizer-update gates plus validation promotion provide a more defensible continuation budget.
- JSSP50x20 is comfortably trainable in FP32 at the locked B=128 semantics, while FFSP1000's full 24-start graph sits beyond the 24 GiB FP32 boundary even at physical batch one. Precision, not candidate-count reduction, is the first allowed FFSP adaptation because the 24 machine-order starts are part of the protocol.
- BF16 full graph with cuDNN SDPA disabled is about 1.9x faster than exact FP32 replay at FFSP1000, while replay is the low-memory numerical-audit path. Both preserve the complete 24-candidate pool.

## Lessons and Constraints

- FFSP's current paper evaluation loader restores `num_job=100`; it must rebuild the environment for FFSP1000.
- JSSP checkpoint `allowed_shapes=[[15,15]]` does not block direct benchmark inference but must be overridden for target-scale dataloaders/continuation.
- Final tests cannot be used for checkpoint or hyperparameter selection.
- Any PO-anchored or otherwise mixed objective must be labeled a variant and cannot substitute for the original USW/ASW claim.

## Open Questions

- Do the four objectives separate by the 100- and 500-update validation gates from their identical common checkpoints?
- Is effective batch one sufficient for stable 100-update FFSP/JSSP screening, or will the 500-update gate require gradient accumulation?
- Do original USW and ASW retain useful gradient signal at the target scales, or do they require only alpha/LR calibration?

## Optimization Trajectory

Target-scale optimization is still gated on H0, but four JSSP50x20 diagnostic updates are complete. Reduced-scale smoke gradient norms were:

- FFSP: PO 0.5782, BOPO 0.06649, USW 0.3853, ASW 0.3221.
- JSSP: PO 3.4307, BOPO 1.7471, USW 0.1706, ASW 0.1986.

The four FFSP variants shared state SHA `997238c32efaf8c5336d2beda94da6f13d4616f21982d9326e5fb6c9f2791e96`.
The four JSSP variants shared state SHA `da29f48d0e904ece1fd2dc66986bd599888a64622f2a906324673c52f81d6c24`.

JSSP50x20 target-scale H0 gradient norms were PO 27.1649, BOPO 0.4844, USW 1.2189, and ASW 1.4536. FFSP1000 FP32 is ruled out on 24 GiB for all four objectives; BF16 with cuDNN SDPA disabled is the locked FFSP engineering configuration. H1's 100-update validation screen is next.

H1 screening is active on g51. The clean FFSP1000 BF16 screen uses a new output root: PO then ASW on GPU0/PID 3918803, BOPO on GPU1/PID 3917615, and USW on GPU6/PID 3917593. The JSSP50x20 PO->BOPO->USW->ASW chain remains on GPU4/PID 3903972. JSSP PO finished 100 updates with validation 3132.375 versus its same-run step-0 value 3129.374939, so its step-0 checkpoint remains selected while BOPO is running.
