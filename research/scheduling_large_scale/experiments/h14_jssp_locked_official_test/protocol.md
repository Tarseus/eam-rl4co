# H14: Locked JSSP50x20 Official-Test Inference

## Status and purpose

Preregistered after validation-only checkpoint selection and before evaluating
H8 or H13 on official TA/DMU data. The user explicitly authorized final-test
evaluation. Test results may be reported but must never select or modify a
checkpoint, training budget, loss, pair builder, rollout count, or seed.

FFSP100 is not rerun: the fixed 1,000-instance test at dataset SHA256
`673cf591412242a3e6b197ca8c2c1bc59d5e1b1c50ede55aaef9c6c4a40b6446`
already evaluated the four locked paper checkpoints with 24 starts and 128
augmentations. Its existing global 52-comparison Holm family is the final
FFSP100 evidence and will be audited rather than consumed again.

## Locked JSSP checkpoints

- BOPO step500:
  `logs/scheduling_large_scale/h4_matched_bopo/jssp50x20/bopo_lr1e5_screen500_20260718_0110/best.ckpt`,
  SHA256 `ca0a0e4d0b90a74c91f09daf087d230e14ac81274cfc9ceb6dc9e6244c8fb44d`.
- H8 USW absolute step500:
  `logs/scheduling_large_scale/h8_relative_gap_temperature/jssp50x20/continue500_from_step250_20260718_0655/usw_relative_gap_temperature/best.ckpt`,
  SHA256 `9b5054b5a9b6038840c389a9802dc8d2e15a059c11a417a443ec930058b58ddc`.
- H13 H8-to-ASW micro-step30:
  `logs/scheduling_large_scale/h13_h8_to_asw_micro/jssp50x20/screen50_20260718_1254/asw_micro_lr1e6/best.ckpt`,
  SHA256 `0b25c8b50fda168b01bf6c0820e910c0cc72bf1066d5cb0a1eb70ada1ee4011c`.

H13 remains a sequential 500-USW-plus-30-ASW hybrid, not original ASW and not
a matched-500-update objective comparison.

## Locked data and inference protocol

- Official BOPO repository commit:
  `2739fbfe39a478755173c892b80c4b06f9f59b05`.
- TA50x20: `ta61.jsp` through `ta70.jsp`.
- DMU50x20: `dmu36.jsp` through `dmu40.jsp` and `dmu76.jsp` through
  `dmu80.jsp`.
- Primary statistical unit: one official instance; total `n=20`.
- Identical per-instance sampling seeds, base `12345678`.
- `B'=128`, one greedy rollout injected, K is irrelevant during inference.
- Primary effect: candidate-minus-BOPO paired gap-percent difference; lower is
  better.
- Two-sided paired Wilcoxon signed-rank tests for exactly two comparisons:
  H8 versus BOPO and H13 versus BOPO.
- One Holm family over those two p-values at familywise alpha `0.05`.
- 20,000-sample percentile bootstrap 95% CI for each mean paired gap
  difference, seed `20260718` plus comparison index.
- A candidate passes only when its mean paired difference is negative, the
  bootstrap upper bound is below zero, and its Holm-adjusted Wilcoxon p-value
  is below `0.05`.
- TA and DMU subset means/win counts are descriptive only. They do not create
  extra tests or selection opportunities.

## Execution and failure discipline

Each checkpoint is evaluated once in a collision-free output directory. An
SSH timeout is reconciled against processes, output roots, and logs before any
retry. Checkpoint SHA, saved method/step, dataset commit, exact instance names,
file SHA256, finite costs, and aligned instance keys must all pass. After the
three outputs finish, run the locked analysis once. Whatever the result, do
not retrain, change B/seeds, add instances, or rerun a favorable subset.

