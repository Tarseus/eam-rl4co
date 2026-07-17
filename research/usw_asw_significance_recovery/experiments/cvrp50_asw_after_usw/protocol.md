# Exploratory protocol: CVRP50 ASW-after-USW short continuation

## Hypothesis

The CVRP50 USW checkpoint has a stronger policy representation than the current ASW checkpoint. A short, low-learning-rate continuation using the unchanged ASW objective can preserve that representation while learning the adaptive weights, producing a candidate that improves validation without changing pair construction.

## Configuration

- Initialization: USW epoch775, SHA256 `1a4270b5976586b20441360c11e5bba1361cae4368df2bd5a817fc9247a41e32`.
- Objective: existing CVRP50 ASW `best_pair.json`; no loss or weighting formula edits.
- Optimizer: restored Adam, forced LR `1e-5`.
- Screen: 10 additional epochs, fixed validation set, monitor `val/max_aug_reward` in max mode.
- Selection: at most one validation-best checkpoint; do not run the paper test until the screen is complete and the candidate is locked.
- Reporting label: `ASW-after-USW variant`.

## Promotion rule

Promote only if validation beats the source USW validation value under the same inference configuration. A promoted checkpoint must then beat the locked USW paper mean `10.437088506984711` and achieve replacement Holm-adjusted p < 0.05 on ASW-vs-USW.
