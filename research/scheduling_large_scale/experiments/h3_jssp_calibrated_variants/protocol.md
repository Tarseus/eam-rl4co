# H3 Protocol: JSSP50x20 calibrated USW/ASW variants

Status: locked before any H3 target-scale training launch.

## Question

Can a separately labelled, low-learning-rate calibration recover the original
USW/ASW objectives at JSSP50x20, following the checkpoint-backed continuation
strategy that succeeded at TSP1000/CVRP1000?

The original H1 result remains negative and is never overwritten. Every run in
this protocol is a calibrated variant.

## Fixed controls

- Common policy initialization:
  `downloads/jssp15x15/weighting/checkpoint.ckpt`, policy SHA256
  `5a90dc4027c08ef7bb2192a84a752a638bd4c59c46a24bf2a57c3721baaf8584`.
- Fresh Adam, FP32, weight decay `1e-6`, physical instance batch `1`.
- JSSP shape `50x20`, `B=128`, `K=16`, greedy injection disabled in training.
- Seed `12345678`, training data start index `0`.
- Candidate pools and all pairs remain strictly within each instance.
- Validation uses the same deterministic validation stream as H1 expanded from
  8 to 32 instances; it is validation-only and not an official/final test set.
- Validate at optimizer steps `0`, `50`, and `100`; select by the best 32-instance
  validation mean, never by terminal step alone.
- Frozen original controls are H1 PO step0 and selected BOPO. No PO continuation
  is reused because its step500 artifact has invalid duplicate-run provenance.

## Phase A variants

| Variant | Objective | Learning rate | ASW alpha | Purpose |
|---|---|---:|---:|---|
| `usw_lr5e6` | original USW kernel/builder | `5e-6` | n/a | moderate LR reduction |
| `usw_lr2e6` | original USW kernel/builder | `2e-6` | n/a | strong LR reduction |
| `asw_lr5e6_a1` | original ASW | `5e-6` | `1.0` | LR-only control |
| `asw_lr5e6_a0` | original ASW | `5e-6` | `0.0` | remove raw regret-scale exponent |
| `asw_lr5e6_a025` | original ASW | `5e-6` | `0.25` | weak regret-scale dependence |
| `asw_lr5e6_a05` | original ASW | `5e-6` | `0.5` | intermediate regret-scale dependence |

`alpha` is not presented as an original-objective result: changing it creates an
ASW calibration variant. USW's reference all-pairs builder does not consume
`alpha`, so only its optimizer LR is varied in Phase A.

## Execution gates

1. Run a one-update smoke for all six variants in collision-free output
   directories. Require finite loss, finite nonzero gradient, exact
   `candidate_count_per_instance=128`, physical batch `1`, and instance-local
   pair diagnostics.
2. Only smoke-passing variants run the 100-update screen. Never reuse an output
   directory and never append a second run to an existing history.
3. A family promotes at most one variant: its best step-50/100 validation mean
   must improve on its own identical step-0 mean, show no error/nonfinite marker,
   and be the family minimum on the locked 32-instance validation set.
4. If no variant in a family improves step0, that family stops. If a variant
   promotes, continue it in a new directory/checkpoint chain to an absolute
   500-update gate with validation every 100 steps.
5. Official TA50x20 evaluation is run only after a family candidate is locked.
   Hyperparameters are never selected on TA/DMU/final benchmark results.

## Interpretation

- A low-LR USW recovery supports optimizer-scale mismatch as the main H1 failure.
- ASW alpha improvement over the `alpha=1` LR control supports source-scale
  pair-weight saturation as a transfer failure.
- If the LR-only ASW control wins, the evidence favors optimizer scale rather
  than weighting calibration.
- These are exploratory calibrated variants and cannot replace the original H1
  claim without being reported under their new labels.
