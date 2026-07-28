# Real pair-margin Fisher validation protocol

## Research question

Does the Fisher geometry induced directly by real pairwise policy margins
produce a candidate descriptor that is more faithful to realized policy
updates than the same margin gradient treated in a Euclidean space?

This experiment removes the virtual empirical-distribution flow. It never
uses an absolute log-probability as a candidate coordinate.

## Fixed candidates and anchors

- Reuse the deterministic 16-candidate score-quantile subset from each of the
  `matched` and `external` high-fidelity sources (32 candidates total).
- Both sources use the same dense all-pairs builder `g_ref`; this experiment
  therefore tests variation in `f` with `g` fixed.
- Use the two checkpoint rollout banks
  `tsp100_epoch135_rollout_probes.npz` and
  `tsp100_epoch135_rollout_probes_seed2.npz`, giving 16 independent TSP100
  anchor instances with 100 POMO trajectories each.
- For every objective-ordered pair `e=(w,l)`, use only the real margin

  `s_e = log pi(tau_w) - log pi(tau_l)`.

- No temperature, margin clipping, probability floor, or learned calibration
  is permitted. Fisher factors are evaluated in float64 using stable log-space
  formulas.

## Shift-invariance gate

The existing programs expose `log_prob_w` and `log_prob_l` separately, while
the proposed search space will expose only their difference. Before descriptor
evaluation, every selected candidate must pass both checks on the anchors:

1. common shifts in `{-5,-1,1,5}` change the loss by at most `1e-8` relative;
2. the common-mode gradient ratio is at most `1e-8`.

A failure is reported and excluded rather than projected or silently repaired.
The gate itself was audited before this protocol: all 32 selected candidates
pass to numerical precision. Actual alignment metrics have not yet been
computed for the new descriptors.

## Shared pair statistical manifold

Let `p` be the centered vector of trajectory log-probabilities, `B` the
oriented pair--trajectory incidence matrix, and `s = Bp`. Each pair defines the
conditional preference probability

`mu_e = sigmoid(s_e)`.

The shared convex potential and its Hessian are

`Phi(p) = sum_e log(1 + exp((Bp)_e))`,

`F_p = B^T diag(mu_e (1-mu_e)) B`.

Because adding a constant to every component of `p` changes no margin, all
node-space quantities are represented in a fixed Helmert basis `H` for the
centered `(S-1)`-dimensional quotient. The realizable Fisher matrix is

`F = H^T F_p H`.

`F` is shared by all candidates at the same anchor. The builder/loss programs
affect the loss covector, not the metric.

## Compared descriptors

For a candidate `C`, differentiate its scalar loss with respect to the
independent pair margins at the real anchor:

`a_C = d L_C / d s`.

The realizable node covector is `c_C = B^T a_C`. Compare:

1. `node_euclidean`: `H^T c_C`.
2. `pullback_fisher`: `F^{-1/2} H^T c_C`, so ordinary dot products of stored
   vectors equal the dual Fisher inner product.
3. `pair_euclidean` (diagnostic): the ambient edge covector `a_C`.
4. `pair_fisher` (diagnostic):
   `diag(mu(1-mu))^{-1/2} a_C`.

Each descriptor block is unit normalized per anchor, then the 16 blocks are
concatenated and divided by `sqrt(16)`. Unit normalization removes candidate
loss scale but does not remove the non-scalar Fisher anisotropy.

The two node-space methods are the primary comparison. The pair-space methods
diagnose whether enforcing Bradley--Terry consistency matters.

## Frozen targets and metrics

Reuse the previously generated, candidate-keyed held-out response vectors:

- one real Adam step from `results/artifacts.npz`;
- the valid three-step on-policy path with disjoint train/held-out banks from
  `results_3step_disjoint/artifacts.npz`.

For each target, report by source and pooled:

1. Spearman correlation between descriptor and actual-response distances;
2. normalized actual nearest-neighbor error;
3. top-1 agreement and top-5 overlap with actual-response neighbors;
4. a paired candidate-bootstrap 95% interval for
   `NN_error(pullback_fisher) - NN_error(node_euclidean)`.

High-fidelity score-difference correlation and nearest-neighbor score error are
secondary metrics. No target score is used to tune the descriptor.

## Decision rule

The real-margin Fisher geometry is promoted as the core descriptor only if:

- on the valid three-step target, `pullback_fisher` has higher distance
  correlation and lower normalized nearest-neighbor error than
  `node_euclidean` on both candidate sources; and
- the pooled three-step bootstrap interval lies strictly below zero; and
- on the one-step target, Fisher is not worse by more than `0.01` in distance
  correlation on either source.

Otherwise retain the margin-only search-space restriction but use the simpler
Euclidean realizable gradient descriptor. Pair-space diagnostics cannot satisfy
the promotion rule by themselves.

## Integrity

- Commit this protocol and the implementation before computing descriptors.
- Label the study retrospective/exploratory because its realized-update targets
  were generated in earlier experiments, although the new descriptor outcomes
  are unknown at protocol time.
- Record all non-finite losses, gradients, Fisher spectra, and zero descriptor
  blocks. Do not impute failures.
