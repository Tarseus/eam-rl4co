# Confirmatory protocol: promote screened USW/ASW candidates to 1000 steps

This protocol is locked before any of the six screening runs reaches continuation step 300.

## Objective

Select one USW configuration and one ASW configuration from the existing 300-step exploratory screen, then continue each configuration to exactly 1000 optimizer/continuation steps from its method-specific pilot initialization without changing the USW/ASW method definition.

## Screening selection rule

- A run is eligible only if it reaches continuation step 300 with `summary.json`, has no OOM/NaN/Traceback, and has maximum absolute forced-replay error equal to zero.
- Select the USW run with the lowest fixed validation32 mean at continuation step 300 among `usw_k40_a005`, `usw_k40_a001`, and `usw_k64_a001`.
- Select the ASW run with the lowest fixed validation32 mean at continuation step 300 among `asw_k40_a005`, `asw_k40_a001`, and `asw_k40_a001_detach`.
- If step-300 means are exactly tied, use the lower step-200 mean, then the lower step-100 mean.
- The matched PO-300 control is used to interpret screening progress, not to change the within-method selection rule.

## Continuation protocol

- Resume from the selected run's `last.ckpt` at optimizer step 300, not from an earlier validation-best checkpoint.
- Restore both model and Adam optimizer state using `--resume`.
- Use `--additional-steps 700`, producing absolute target optimizer step 1000.
- Preserve the selected run's method, K/num-starts, alpha, detach flag, learning rate, weight decay, batch size, precision, seed, validation protocol, and forced-replay validation.
- Advance `data_start_index` from 20,000,000 to 20,038,400 (`20,000,000 + 300 * 128`) so steps 301-1000 consume the same non-overlapping dynamic-instance stream that a single 1000-step run would consume.
- Keep base train batch at 128 or below. Do not disable forced replay.
- Write continuation output to a new directory whose config records `resume_optimizer_step=300`, `additional_steps=700`, and absolute `steps=1000`.

## Final checkpoint and evaluation

- The candidate eligible for the final claim is `last.ckpt` at absolute optimizer step 1000. A checkpoint from an earlier step cannot satisfy the equal-budget criterion.
- Evaluate on `data/tsp/tsp1000_test_seed1234.npz` only after verifying SHA256 `fc544f59443a9352f4f3317603d39fae8123cba8502864cae8355e595555f865` and exact instance-index alignment with the PO result.
- Use fixed100, 1000 starts, 8 augmentations, and FP32.
- Success requires mean tour length below PO 27.13804609298706 and a paired-bootstrap 95% confidence interval for candidate-minus-PO with both bounds below zero.

## Method-identity boundary

This promotion changes training configuration only. It does not change the USW/ASW weighting formula, pair target, instance-local pair construction, model architecture, canonical data, or evaluation protocol.

## Predictions

- USW K64/alpha=0.01 currently has the strongest step-100 proxy, consistent with improved per-instance preference estimation from larger K.
- ASW alpha=0.01 currently improves over alpha=0.05, while detach and non-detach are nearly tied at step 100; later validation points decide the promotion.
