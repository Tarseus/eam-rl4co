# Exploratory protocol: low-LR effect amplification after validation-best failures

The locked FFSP50 epoch117 and JSSP15x15 epoch13 candidates both improved mean direction but failed the pre-specified two-sided Wilcoxon plus global Holm threshold. The paper test instances are now considered spent for any later checkpoint selection. This experiment changes only the optimization schedule and uses fresh, pre-generated confirmation data.

## FFSP50 USW restart

- Source: validation-best epoch117, SHA256 `0b52e9de24c31cc373290c3397081c085914e682981b406bde645a6b068fab1d`.
- Rationale: the original MultiStepLR had already reduced LR from `1e-4` to about `1e-6` after epoch110, and the remaining epochs drifted without creating a large signed-rank effect.
- Intervention: restore Adam state, force LR `1e-5`, train 10 additional epochs, and select exactly one checkpoint by `val/reward` (maximize).
- USW objective, pair JSON, 24 starts, per-instance candidate pools, model, and training data semantics remain unchanged.
- Fresh confirmation set: 2000 FFSP50 instances generated once with seed `20260717`.
- Fresh NPZ SHA256: `6aa464b4b8c95f192d208a880592616250784aa82bd7b3666045da30ccb178aa`.
- Confirmation requires reevaluating PO4COPs, BOPO, ASW, and the locked USW candidate with identical seed reset, 24 starts, and 128 augmentations. No intermediate checkpoint is evaluated.

## JSSP15x15 USW restart

- Source: validation-best epoch13, SHA256 `93b79af77a42156753882f5d37c4be1b1e9b814d9890852b666de278be1ac3ff`.
- Rationale: LR `2e-4` reached the best validation gap at epoch13 and then regressed. A 4x downshift may preserve direction while strengthening the effect.
- Intervention: restore Adam state, force LR `5e-5`, train 5 additional epochs, and select exactly one checkpoint by `val/gap` (minimize).
- USW objective, K=16, B=128, anchored instance-local pairs, 15x15-only buckets, and model remain unchanged.
- Fresh confirmation set: 1000 JSSP15x15 instances generated once with seed `20260717`; a dummy reference value of 1 is appended because inference comparison uses only `pred_makespan`.
- Ordered dataset digest: `507d448904649b3e9b372a561c1d19da060806b42857b0df930dd3dcfa71542e`, defined as SHA256 over newline-joined lowercase content-SHA256/filename records in filename order.
- Confirmation requires reevaluating PO4COPs, SLL, BOPO, ASW, and the locked USW candidate with B128, greedy0, no augmentation, and common per-instance sampling seeds. No intermediate checkpoint is evaluated.

## Statistical decision

The fresh confirmation family retains the two-sided paired Wilcoxon test and a Holm correction over the complete candidate-objective comparison family available on the fresh sets. A candidate must have a lower mean and adjusted p < 0.05 against every hand-designed objective. Results are exploratory until the fresh confirmation is complete.
