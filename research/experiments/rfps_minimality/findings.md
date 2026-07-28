# RFPS minimality findings

## Current understanding

The useful object is not a three-point Fisher--Rao curvature. A two-point directional
response is sufficient: normalize the complete program-pair gradient direction at a
fixed checkpoint probe, take one small empirical-logit step, recompute and normalize
the direction, and concatenate the two directions. This uses two derivatives and one
probe bank while retaining high global correlation and the curvature descriptor tail
safety.

## Patterns and insights

- Raw gradient magnitude is mostly nuisance; per-probe direction normalization is
  responsible for the large global-correlation gain.
- Initial direction alone has dangerous near-neighbor collisions. The second point
  resolves those collisions, which is the real incremental value of local response.
- Three flow points, fine integration, log maps, curvature residuals, and Fisher
  transport do not add measurable value on the available data.
- Euclidean and Fisher one-step direction pairs are nearly indistinguishable.
- One fixed checkpoint probe bank predicts both scratch and warm fitness differences.
- Gram features mostly re-encode the same initial directions and inherit their tail
  collision problem.

## Lessons and constraints

- Prefer a simpler descriptor whenever it meets the locked accuracy and risk margins.
- Do not interpret large curvature as candidate quality; the descriptor is only a
  behavior-neighborhood marker.
- Keep real on-policy training as the source of fitness.
- Compare normalized baselines before attributing gains to higher-order structure.
- Optimize screening descriptors for tail collision risk, not only pairwise rho.

## Open questions

- Does the two-point direction pair retain the same behavior when g varies?
- Does it improve best-so-far under a fixed real-training budget in a live search?
- What is the smallest probe-bank size that preserves tail-risk control?

## Final method decision

The deployable descriptor is
`Psi(C) = concat_r[d0^(r), d1^(r)] / sqrt(2R)`, with `R=8` and a single
checkpoint rollout bank. `d0` is the centered unit negative log-probability
gradient of the full `(f,g)` loss. `d1` is recomputed after
`q1 = softmax(log(q0) + 0.03*d0)`. Euclidean nearest-neighbor distance drives only
soft training-budget allocation with a nonzero audit floor. Real on-policy training
remains the sole source of fitness.
