# Weighting Formula Scale Shift

CPU replay analysis for the discovered best weighting builders. The script applies the same weighting formula at the search scale and a transfer scale, then records the pre-clamp formula score, final pair weight, clamp shares, and operand-level distribution shifts.

State: `sampled`; sharpness: `1.0`.

This is not a new training run. It diagnoses how second-stage weighting changes the pair measure induced over the candidate pool when the problem scale changes.
