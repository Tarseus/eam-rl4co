# Weighting Formula Scale Shift

CPU replay analysis for the discovered best weighting builders. The script applies the same weighting formula at the search scale and a transfer scale, then records the pre-clamp formula score, final pair weight, clamp shares, and operand-level distribution shifts.

This is not a new training run. It diagnoses how second-stage weighting changes the pair measure induced over the candidate pool when the problem scale changes.

Note: this folder is a formula-pressure diagnostic. For the main paper explanation, prefer `../weighting_delta_cost_ckpt/`, whose x-axis is the actual pair-level Delta cost from checkpoint-generated trajectories.

Related robustness replay:
- `../weighting_generalization_sampled/`: same analysis with raw sampled rollout log-probabilities instead of the aligned diagnostic state.

Writing note:
- `interpretation.md` contains the recommended paper argument and formula-level observations.
