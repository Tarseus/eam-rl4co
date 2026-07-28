# Research log

## 2026-07-28: protocol lock

Locked the candidate set, scratch/warm targets, shared-bank construction,
reference measures, step length, and decision metrics before cross-policy
scoring.

## 2026-07-28: shared-bank construction

Teacher-forced both policies on both trajectory groups. A first attempt to use
one 200-start POMO call failed the own-likelihood sanity check; splitting the
two 100-start groups reproduced stored likelihoods exactly.

## 2026-07-28: numerical audit

The anchor likelihood ratios underflowed to zero on the opposite policy's
trajectories. Added the mathematically required simplex-interior floor of
`1e-12`; no ESS tuning was needed.

## 2026-07-28: confirmatory result

Anchor-IS beat uniform weighting on the mixed bank for both targets and both
probe seeds. Pair exposure alone failed the locked tail-risk decision.

## 2026-07-28: outer-loop comparison

Compared the mixed-bank results against the already validated checkpoint-135
uniform bank, then ran an exploratory pair-exposure follow-up on that bank.
The existing uniform bank remained clearly better. Final direction: retain the
uniform checkpoint probe and do not add non-uniform initialization.

Recomputing the preregistered all-pair endpoint revealed that tied objectives
make it slightly non-uniform. This materially changed one mixed-bank global
correlation but did not change tail risk. On the deployed warm bank it changed
rho by less than 0.001, confirming that it adds no useful component.

## 2026-07-28: Gaussian shape stress test

Tested best-centered, middle-centered, and best/worst two-tail Gaussian
reference measures over normalized tour-cost rank, with three widths per
family. Uniform remained best for scratch and warm targets on both probe
seeds. The closest Gaussian had normalized ESS 0.985 and converged toward,
rather than improved upon, the uniform result.
