# Exploratory protocol: CVRP50 ASW-after-USW short continuation

## Hypothesis

The CVRP50 USW checkpoint has a stronger policy representation than the current ASW checkpoint. A short, low-learning-rate continuation using the unchanged ASW objective may preserve enough of that representation while learning adaptive weights to produce a second proposed-method variant that outperforms the strongest recorded external learned baseline. The variant does not need to outperform USW.

## Configuration

- Initialization: USW epoch775, SHA256 `1a4270b5976586b20441360c11e5bba1361cae4368df2bd5a817fc9247a41e32`.
- Objective: existing CVRP50 ASW `best_pair.json`; no loss or weighting formula edits.
- Optimizer: restored Adam, forced LR `1e-5`.
- Screen: 10 additional epochs, fixed validation set, monitor `val/max_aug_reward` in max mode.
- Selection: after the full screen completes, lock exactly one checkpoint by maximum `val/max_aug_reward`.
- Promotion gate: the locked checkpoint may receive the single paper-test evaluation only if its `val/max_aug_reward` is strictly better than the source USW checkpoint on the same validation monitor. The verified source epoch775 checkpoint records `-10.429018020629883` in max mode. If the gate fails, close H3 without reading or running the paper test.
- Reporting label: `ASW-after-USW variant`.

## External-baseline decision rule

- Target baseline: recorded CVRP50 `SymNCO, aug_max` mean `10.43764` in `all_problem_results_no_eam_max_aug_comparable_time.csv`.
- Recovered baseline artifact: `downloads/standard_baselines/symnco/cvrp50/symnco_cvrp50_epoch=1000.ckpt`, SHA256 `0f13962d9f1770d7fdad36e9d04cc8bc3d888ef5c72e47136fdb4fe890dc6cef`.
- The rounded table mean is not used for paired inference. Only after locking the variant and passing the source-validation promotion gate, re-evaluate SymNCO, locked USW, and the variant on the exact same ordered paper instances with verified seeds, augmentation, and method-appropriate rollout settings.
- Success requires both USW and the `ASW-after-USW variant` to have lower means than SymNCO and Holm-adjusted two-sided paired Wilcoxon p-values below `0.05` for both `USW vs SymNCO` and `ASW-after-USW vs SymNCO`.
- These two comparisons form a separate fixed Holm family preregistered before reading the variant's paper-test result. The original 52-comparison paper family remains unchanged.

## User-authorized gate override (2026-07-17 14:32 CST)

- The user explicitly authorized the already validation-locked epoch778 candidate to receive its one paper-test evaluation despite failing the source-USW validation promotion gate.
- This override changes only whether the fixed candidate may be evaluated. Epoch778 remains the unique checkpoint selected by the completed validation screen; no paper-test result may be used to select another epoch.
- Re-evaluate the fixed USW epoch775, fixed `ASW-after-USW variant` epoch778, and SymNCO checkpoint once on the exact ordered CVRP50 paper instances.
- Keep checkpoint/data SHA checks and seed1234. Reproduce the comparable-time `aug_max` table setting for every method: 50 starts with 8 augmentations. Keep the two-sided paired Wilcoxon tests and preregistered two-comparison Holm family unchanged.
