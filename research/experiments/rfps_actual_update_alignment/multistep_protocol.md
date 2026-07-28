# Three-step on-policy extension protocol

## Motivation

The preregistered one-step experiment is the primary fixed-chart test, but one
optimizer step is expected to be dominated by the initial parameter gradient.
A two-point descriptor is intended to represent the beginning of nonlinear
training behavior. This extension therefore tests the descriptors against a
short realized training path rather than changing the coordinate chart or the
virtual step length.

This protocol is committed after the one-step result and before any three-step
alignment result is computed. The one-step result remains primary and is not
replaced.

## Frozen design

- Reuse the exact 16 `matched` and 16 `external` candidates selected by the
  primary protocol, the same `tsp100_epoch_135.ckpt`, and fixed `g_ref`.
- Reuse the initial one-point, Euclidean two-point, and Fisher two-point
  descriptors from the primary run (`ell = 0.03`, fixed log-weight chart).
- Start every candidate from the same checkpoint.
- Run exactly three Adam steps with learning rate `3e-4`, weight decay `1e-6`,
  and one optimizer state maintained across the three steps.
- At every step, use two fixed TSP100 instances and 100 POMO starts. Instance
  banks are generated once with seeds `2026072811`, `2026072812`, and
  `2026072813`, and shared across candidates.
- Before each rollout, reset the sampling RNG to the same step-specific seed
  for all candidates. Because the current policies differ after step one, the
  resulting trajectories may differ; each rollout is nevertheless sampled
  from the policy being updated and is therefore on-policy.
- Compile and evaluate the actual candidate loss with the fixed dense
  all-pairs builder at every step. No importance sampling is used.
- Evaluate the final policy on the same disjoint teacher-forced held-out bank
  as the primary experiment. Center and unit-normalize its log-likelihood
  change exactly as in the primary protocol.

## Metrics and decision

Report the same distance Spearman correlation, normalized actual nearest-
neighbor error, top-1 agreement, top-5 overlap, and paired bootstrap interval
as the primary experiment, separately for each source and pooled.

Fisher is considered practically supported by the short realized flow only if
it beats Euclidean in both distance correlation and normalized nearest-neighbor
error on both sources and the pooled bootstrap 95% interval for
`NN_error(Fisher) - NN_error(Euclidean)` is strictly below zero.

If Fisher again fails this rule, coordinate invariance must not be used as the
core justification under a fixed implementation. The method should be reduced
to the cheapest descriptor that retains the observed actual-update alignment.

## Integrity rules

- Commit this extension protocol and its implementation before execution.
- Three steps, seeds, candidate membership, optimizer, and descriptor settings
  may not be tuned after viewing results.
- Record every candidate failure. Do not replace failed responses with scores
  or descriptor neighbors.

## Protocol erratum recorded before the valid rerun

The first implementation used the originally written formula
`2026072801 + step`. Its second training seed was therefore `2026072802`,
which equals the frozen held-out seed. That run violated the disjoint-bank rule
and is retained only as an invalid diagnostic under `results_3step`; it must not
be cited as evidence. The corrected seeds above are the sole change for the
valid rerun, whose outputs go to `results_3step_disjoint`. Candidate selection,
optimizer, number of steps, descriptors, held-out bank, and metrics remain
unchanged.
