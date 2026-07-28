# H2 findings and outer-loop decision

The real policy manifold solved the mathematical degeneracy of the uniform
rollout-weight simplex: its one-point Fisher descriptor materially changes
neighborhoods and improves external global fitness-gap correlation. The
follow-up experiment shows that this does not imply a useful short policy
flow. A second point at Fisher--Rao length 0.03 reduces external correlation
and more than doubles local false-skip risk.

Current interpretation:

1. Policy Fisher detects sensitivity through low-probability actions that
   raw trajectory coefficients miss.
2. That sensitivity is valuable for global separation, especially for the
   epoch-31 bank.
3. The same rare-action weighting makes local neighborhoods fragile.
4. Re-evaluating the candidate after an intrinsic policy step does not repair
   that fragility.

Therefore this branch should not replace the current RFPS screening
descriptor. It may motivate a future, separately evaluated risk or global
diversity feature, but that would be a new hypothesis and not a rescue of H2.
