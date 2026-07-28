# Findings: fixed-chart descriptors versus realized policy updates

## Bottom line

With one fixed representation of the empirical weights, Fisher transport does
not predict realized neural-policy updates better than an equally expensive
Euclidean two-point construction. The preregistered Fisher promotion rule fails
for both one-step and three-step targets.

This removes the practical justification for making Fisher--Rao invariance the
core of the method. Coordinate invariance remains mathematically true, but it
does not buy measurable update alignment in the fixed implementation tested
here.

## What was tested

- The trusted `tsp100_epoch_135.ckpt` PO4COPs-compatible TSP100 POMO policy.
- Sixteen score-quantile candidates from each of two independently collected
  high-fidelity sets (32 candidates total).
- Both sets fix `g = g_ref`, the dense all-pairs builder. This experiment varies
  `f`; it is not evidence about varying preference builders.
- A fixed log-weight chart and `ell = 0.03`; no coordinate rescaling stress.
- Real Adam parameter updates followed by held-out teacher-forced trajectory
  log-likelihood changes.
- A one-step common on-policy microstep and a separate three-step experiment in
  which every candidate resamples from its own updated policy at each step.

The valid three-step run uses training seeds `2026072811`, `2026072812`, and
`2026072813`, disjoint from held-out seed `2026072802`.

## Primary one-step result

| Set | Descriptor | Distance rho | Normalized NN error | Top-1 |
|---|---|---:|---:|---:|
| matched | one point | 0.975 | 0.267 | 0.938 |
| matched | Euclidean two point | 0.975 | 0.267 | 0.938 |
| matched | Fisher two point | 0.975 | 0.267 | 0.938 |
| external | one point | 0.804 | 0.397 | 0.625 |
| external | Euclidean two point | 0.802 | 0.397 | 0.625 |
| external | Fisher two point | 0.800 | 0.397 | 0.688 |

The pooled Fisher-minus-Euclidean nearest-neighbor error is approximately
`-1.1e-7`, with a bootstrap 95% interval ending at zero. It is numerically tiny
and fails the strict improvement rule.

## Valid three-step on-policy result

| Set | Descriptor | Distance rho | Normalized NN error | Top-1 |
|---|---|---:|---:|---:|
| matched | one point | 0.971 | 0.529 | 0.562 |
| matched | Euclidean two point | 0.971 | 0.529 | 0.562 |
| matched | Fisher two point | 0.971 | 0.529 | 0.562 |
| external | one point | 0.676 | 0.519 | 0.500 |
| external | Euclidean two point | 0.675 | 0.519 | 0.500 |
| external | Fisher two point | 0.673 | 0.519 | 0.562 |

The pooled Fisher-minus-Euclidean nearest-neighbor error is about `-5.25e-6`,
again with a bootstrap interval ending at zero. Fisher does not satisfy the
registered rule on either source or pooled.

## Why the three descriptors look the same

At `ell = 0.03`, the Euclidean and Fisher two-point distance matrices have
Pearson correlation `0.999986`. Their mean absolute pairwise-distance
difference is `0.00114`, and they disagree on the nearest neighbor for only one
of 32 candidates. The one-point and Euclidean two-point matrices correlate at
`0.9999998`.

Thus, the absence of a Fisher gain is not a rounding artifact. In this regime,
the second probe barely changes the candidate topology, and the choice of
transport changes it even less.

## Protocol erratum

The first three-step diagnostic accidentally reused held-out seed `2026072802`
as its second training seed. It is stored under `results_3step` for audit but is
invalid as evidence. The error was documented in the protocol and committed
before rerunning. Only `results_3step_disjoint` is used above.

## Method consequence

The clean interpretation is:

1. Do not argue that Fisher geometry is necessary merely because the empirical
   distribution admits reparameterizations. A fixed chart removes that
   practical problem.
2. Do not call the core descriptor Fisher-invariant on the basis of current
   evidence. The realized-update tests do not support that emphasis.
3. For a minimal screening method, the initial joint loss-induced direction is
   currently sufficient. If a second point is retained to target later
   nonlinear behavior, use the fixed-chart Euclidean version unless a direct
   budget-to-best-program ablation demonstrates incremental value.
4. Keep Fisher normalization or transport only as an optional implementation
   choice, not as the claimed source of performance or novelty.

This result does not erase earlier correlations with high-fidelity fitness:
fitness prediction and realized local-update alignment are different targets.
It does mean that any paper claim that the Fisher construction is closer to
training must be removed or supported by a new fixed-chart experiment where it
actually beats the simpler descriptor.

## Scope limits

- Warm-start checkpoint only; from-scratch parameter updates were not run.
- `g` is fixed, so the result cannot establish topology for joint `f/g`
  mutations.
- Two training and two held-out instances per response, with 100 POMO starts.
- The conclusion is specific to the registered `ell = 0.03` rather than a
  post-hoc step-length sweep.
