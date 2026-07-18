# H15: Independent Higher-Power JSSP50x20 Generated Test

## Motivation and scope

H14 was directionally positive on the 20 official TA+DMU instances but failed
both locked significance gates. H14 is frozen and remains the official-test
conclusion. H15 is a separately labelled independent-distribution replication
that tests the same already-frozen checkpoints with higher power. It cannot
replace, pool with, or reinterpret H14, and it cannot trigger model changes.

## Locked population and execution

- Same BOPO/H8/H13 checkpoints and SHA256 values as H14.
- Generate exactly 256 JSSP50x20 instances with the committed
  `_dynamic_instance` generator, generator seed `314159265`, indices 0--255.
- Each method sees byte-identical costs/machines for each index; record a
  content SHA256 per instance.
- Sampling seed base `271828182`, so instance i uses `271828182+i` for every
  method.
- B'=128, one greedy rollout injected, physical instance batch one.
- One execution per checkpoint in collision-free roots. No retry into an
  existing root and no selection from partial results.

## Locked inference

- Primary statistical unit: generated instance, `n=256`.
- Candidate-minus-BOPO raw makespan difference; lower is better.
- Exactly two two-sided paired Wilcoxon tests: H8-vs-BOPO and H13-vs-BOPO.
- One Holm family over the two tests.
- Conservative replication threshold: Holm-adjusted p `<0.025`.
- 20,000-sample percentile bootstrap 97.5% CI for the mean paired difference,
  using percentiles 1.25 and 98.75, seed `20260718` plus comparison index.
- A candidate passes only if mean difference is negative, the 97.5% CI upper
  bound is below zero, and Holm p is below 0.025.

H15 reports generated-distribution significance only. H14 official-test
non-significance and the fact that H13 is a sequential hybrid remain explicit.

