# Fisher-invariant descriptor validation protocol

**Locked before new results are computed: 2026-07-28.**

## Question

The current one-point direction, Euclidean two-point response, and Fisher
two-point response are close when evaluated from a uniform empirical
distribution with a short step. This study tests whether that similarity is
caused by insufficient task-instance coverage, by the chosen log-weight
coordinates, or by the locally uniform starting distribution.

## Fixed data and labels

- Candidate sets: the existing 40 matched candidates and 65 external
  candidates used in Table 3.
- Fitness labels: matched scratch, matched warm-start, and external scratch;
  no descriptor-derived or imputed fitness is allowed.
- Rollout probes: concatenate the two independently sampled checkpoint-135
  banks, each containing 8 TSP100 instances and 100 POMO trajectories per
  instance. The maximum bank therefore has 16 real instances.
- Candidate programs, rollout tensors, fitness labels, step length
  `ell = 0.03`, and the false-skip definition remain fixed.

## Descriptors

For every candidate and probe, compute:

1. `one_point`: the Fisher-unit initial direction `d0`;
2. `euclidean_two_point`: `d0` concatenated with the unit response after one
   centered-logit Euclidean step of coordinate length `0.03`;
3. `fisher_two_point`: `d0` concatenated with the second Fisher-unit direction
   after one Fisher--Rao geodesic step of arc length `0.03` and parallel
   transport to the initial tangent space.

All methods use the same candidate loss evaluations. No network parameter is
updated and no new rollout is generated.

## Experiment A: task-instance count

- Pool the 16 independent probes.
- Evaluate `R in {2, 4, 8, 12, 16}`.
- For each `R < 16`, draw 100 deterministic random subsets without
  replacement; for `R = 16`, use the full pool.
- Report the median and 95% empirical interval across subsets for pairwise
  Spearman rho and nearest-neighbor fitness error. Report false-skip risk with
  200 random history orders per subset.

**H-A.** Increasing `R` primarily narrows subset variability. It is not
expected to systematically enlarge the Fisher--Euclidean effect because both
methods start at a uniform distribution and share the dominant first
direction.

## Experiment B: coordinate reparameterization stress test

- Represent centered logits in a deterministic orthonormal Helmert basis
  `z = B x`, which removes the softmax translation gauge.
- Reparameterize the same local chart by `eta = A x`, where diagonal `A` has
  condition number `kappa in {1, 3, 10, 30}`.
- For each `kappa > 1`, use five fixed random permutations of log-spaced
  diagonal scales. The distribution, candidate, and loss are unchanged.
- Recompute the Euclidean unit step and direction pair in each `eta` chart.
- Report pairwise-distance rank agreement, top-1 nearest-neighbor agreement,
  top-5 neighbor overlap, rho, NN fitness error, and false-skip risk relative
  to the identity chart.
- The Fisher descriptor is computed from the probability distribution and its
  Fisher metric, so its corresponding chart-drift quantities must equal their
  invariant limits up to numerical tolerance.

**H-B.** Euclidean neighbor relations degrade as `kappa` increases, whereas
the Fisher descriptor is invariant. This is the direct test of the geometry,
not a test that Fisher must have a much larger fitness correlation in one
preferred chart.

## Experiment C: non-uniform starting distributions

- For each probe, robustly standardize its minimization objective with median
  and MAD, clip to `[-3, 3]`, and define
  `q0(beta) proportional to exp(-beta * standardized_objective)`.
- Evaluate `beta in {0, 0.25, 0.5, 1.0}` using all 16 probes.
- Record the effective-sample-size ratio of every starting distribution.
- Recompute all three descriptors from each common `q0(beta)` and report rho,
  nearest-neighbor error, false-skip risk, pairwise-distance agreement between
  Euclidean and Fisher two-point descriptors, and nearest-neighbor agreement.

**H-C.** Moving away from the uniform start removes the constant-factor
equivalence between Fisher and Euclidean norms. Euclidean--Fisher descriptor
and neighbor agreement should therefore decrease with lower ESS. A larger
fitness-metric gap is plausible but is not required for confirmation.

## Sanity checks

1. At `beta = 0` and `kappa = 1`, reproduce the existing uniform-start
   Euclidean and Fisher metrics within numerical tolerance.
2. Every descriptor and metric must be finite.
3. The Helmert basis must satisfy `B^T B = I` and `B^T 1 = 0`.
4. Fisher geodesic steps must have arc-length error below `1e-8` unless the
   zero-gradient guard is active.
5. Results discovered after this lock but not specified above are labeled
   exploratory.

## Decision rule

- Keep the current lightweight Fisher construction if H-B is confirmed and
  it does not materially worsen false-skip risk relative to Euclidean two
  points under H-C.
- Do not increase the deployment probe count merely to manufacture a larger
  method gap; increase it only if Experiment A shows a material reduction in
  uncertainty or tail risk.
- Do not tune `beta`, `kappa`, or the coordinate chart to maximize the reported
  Fisher advantage.
