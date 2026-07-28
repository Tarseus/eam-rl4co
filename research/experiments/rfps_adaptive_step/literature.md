# Narrow literature note: geometric update radii

- Kakade, *A Natural Policy Gradient* (NeurIPS 2001): natural gradient is the steepest
  policy direction under the distribution-space metric rather than raw parameter
  Euclidean distance.
- Schulman et al., *Trust Region Policy Optimization* (2015): replaces a fixed penalty
  scale by a KL-bounded policy update and uses line search to satisfy the trust region.
- Pajarinen et al., *Compatible Natural Gradient Policy Search* (2019): connects
  natural-gradient updates and KL trust regions for exponential-family policies and
  highlights entropy loss as a distinct quantity.
- McMahan, *A Survey of Algorithms and Analysis for Adaptive Online Learning* (JMLR
  2017): adaptive mirror descent changes the regularizer or learning-rate schedule from
  observed data; it motivates separating an intrinsic divergence from its adaptive
  radius.

For RFPS, these works motivate KL/Fisher length as the first candidate. They do not
imply that a candidate-specific radius is useful after the RFPS direction has already
been normalized.
