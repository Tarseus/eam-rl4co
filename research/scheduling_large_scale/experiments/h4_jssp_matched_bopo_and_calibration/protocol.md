# H4 Protocol: matched BOPO control and stability-calibrated JSSP50x20 search

Status: locked before any H4 training launch.

## Question

Can validation-selected USW or ASW calibration beat BOPO when every method is
measured on the identical H3 validation32 stream, common checkpoint, rollout
semantics, training instances, and optimizer-update gates?

H1 and H3 results remain unchanged. H4 is a separately labelled calibration
and matched-control cycle.

## Fair-comparison controls

- Common initialization: `downloads/jssp15x15/weighting/checkpoint.ckpt`, policy
  SHA256 `5a90dc4027c08ef7bb2192a84a752a638bd4c59c46a24bf2a57c3721baaf8584`.
- Fresh Adam, FP32, physical instance batch `1`, JSSP `50x20`, `B=128`, `K=16`,
  seed `12345678`, and training data start index `0`.
- All candidate pools and PO/BOPO/USW/ASW pairs remain strictly within one
  physical instance.
- Validation is the existing deterministic H3 validation32 stream: count `32`,
  batch size `8`, rollout count `128`, seed offset `80000003`. It is not final
  test data.
- Select checkpoints only by validation32 mean. Never select parameters on
  TA/DMU or any final benchmark.
- Every output root is collision-free and single-execution. A submission timeout
  requires a process/root audit; it never authorizes an automatic retry.

## Stage 0: matched BOPO reference

Run original BOPO from the common checkpoint with LR `1e-5`, weight decay
`1e-6`, and 500 optimizer updates. Validate at steps `0`, `50`, `100`, then
every 50 steps through `500`. The minimum validation32 mean is the H4 BOPO
reference. Require 500 finite losses, finite nonzero gradients, B128/K16,
physical batch one, 15 best-anchored instance-local pairs, and a clean error
scan.

The already completed H3 locks (USW LR `2e-6` step300 and ASW LR `5e-6`,
alpha `0`, step200) may be compared to this reference because their fixed
initialization, data order, and validation32 stream are identical. They are not
retrained or reselected after seeing BOPO.

## Success and continuation gates

1. A candidate wins the screening gate only if its locked validation32 mean is
   strictly lower than the minimum matched-BOPO mean.
2. If either existing H3 lock wins, freeze that parameterization and checkpoint;
   next run paired validation-only inference before any TA/DMU evaluation.
3. If neither wins, run Phase B below. Do not extend a regressed terminal
   checkpoint merely to spend more updates.
4. A Phase-B family promotes at most one configuration by its best scheduled
   validation mean. Continue only a configuration that beats matched BOPO.
5. Final claims require candidate-minus-BOPO paired bootstrap 95% CI below zero
   and the locked paired Wilcoxon/Holm procedure. Mean-only victory is a search
   gate, not the final statistical claim.

## Conditional Phase B search

The H3 trajectories improve at intermediate steps and regress later, indicating
optimizer-scale instability rather than insufficient update count. If needed,
screen the following bounded configurations from the common checkpoint for 300
updates with validation every 50 steps:

| Family | Label | LR | Weight decay | Alpha |
|---|---|---:|---:|---:|
| USW | `usw_lr1e6_wd1e6` | `1e-6` | `1e-6` | n/a |
| USW | `usw_lr5e7_wd1e6` | `5e-7` | `1e-6` | n/a |
| USW | `usw_lr1e6_wd0` | `1e-6` | `0` | n/a |
| ASW | `asw_lr2e6_a0` | `2e-6` | `1e-6` | `0` |
| ASW | `asw_lr1e6_a0` | `1e-6` | `1e-6` | `0` |
| ASW | `asw_lr1e6_a025` | `1e-6` | `1e-6` | `0.25` |

All variants retain the original objective artifacts. Alpha changes remain
explicitly labelled ASW calibration variants.

## Stop condition

H4 stops only after at least one locked USW/ASW candidate beats the best matched
BOPO checkpoint on the fixed validation32 stream and passes paired validation
inference, or after the bounded Phase-B space is exhausted and a new mechanistic
hypothesis is preregistered. It does not stop merely because a job completed.
