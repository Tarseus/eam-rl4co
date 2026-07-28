# RFPS adaptive-step findings

## Current understanding

The projected RFPS vector is already the Fisher natural gradient on the empirical
simplex. The invariant descriptor should normalize it with the Fisher norm, take one
closed-form Fisher exponential step, and parallel-transport the second unit direction
back to the first tangent space. This two-derivative Fisher pair slightly but
consistently improves rho over the Euclidean pair without restoring curvature.

## Open questions

- How large is the optimizer-induced Fisher arc at checkpoint-135 under one stateless
  Adam update?
- Does a model-pullback tangent justify its network backward cost over the cheap
  empirical-simplex tangent?

## Final decision

- Exact KL, reverse-KL and ESS radii collapse to eta 0.03 with nearly zero
  candidate variation at the uniform normalized start.
- Max-coordinate and turning-radius adaptation add variation but no reliable metric
  gain; turning radius also costs a third derivative.
- Keep one global Fisher arc length, but calibrate it from the observed trajectory
  log-probability displacement of a real stateless optimizer update.
