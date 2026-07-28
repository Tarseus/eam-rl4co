# RFPS adaptive-step findings

## Current understanding

Pending preregistered experiments. Analytically, a fixed Fisher/KL radius at a uniform
empirical distribution is expected to be almost equivalent to a fixed log-weight step
after per-probe direction normalization.

## Open questions

- Does exact KL root solving create any meaningful candidate-specific step variation?
- Does a local turning radius preserve or erase the useful second-point signal?
- Should the final radius be calibrated from a real optimizer update rather than from
  descriptor labels?
