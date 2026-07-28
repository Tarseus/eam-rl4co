# Protocol: two-point flow on the real policy manifold

Date locked: 2026-07-28

## Hypothesis

The first-order categorical-Fisher descriptor adds a real global behavior
signal but is less safe than the raw coefficient for local deduplication.
Re-evaluating the candidate coefficient after one short intrinsic policy
update will recover local information without losing the first-order
Fisher signal.

## Locked construction

Use the same candidates, banks, probes, and categorical product manifold as
in `protocol.md`.  For candidate \(C\), compute

\[
c_i^0=-\frac{\partial L_C}{\partial p_i}
\]

and its policy-Fisher tangent.  Move the product policy by the closed-form
categorical Fisher exponential map for a total Fisher--Rao length
\(\ell=0.03\).  Only the probabilities of the already selected actions are
needed to obtain the perturbed trajectory log-likelihoods \(p_i^1\).
Recompute

\[
c_i^1=-\frac{\partial L_C}{\partial p_i}\bigg|_{p=p^1}
\]

and construct the second policy-Fisher tangent.  Parallel transport along
the same categorical geodesics returns the second tangent to the original
policy.  Concatenate the two unit directions with \(1/\sqrt2\) scaling.

No weight clipping, channel concatenation, learned bank weighting, adaptive
length, or additional screening rule is allowed.

## Controls

1. raw trajectory coefficient;
2. one-point policy Fisher from H1;
3. two-point policy Fisher.

The primary question is whether the second point improves over one-point
Fisher and closes its local-safety gap to the raw coefficient.

## Success criterion

On the external product bank, two-point Fisher must:

1. retain at least 0.7495 scratch rho (no more than 0.005 below the H1
   value 0.7545); and
2. reduce false20 by at least 20% relative to one-point Fisher (below
   3.03%).

Matched joint rho and false20 are reported as a transfer check.  Failure of
either external condition stops the two-point branch.

## Sanity checks

- The global Fisher step length must equal 0.03 within \(10^{-6}\).
- Perturbed selected probabilities must be finite and in \((0,1]\).
- Zero coefficient produces no policy change.
- The second-point construction uses exactly one additional candidate-loss
  derivative and no network update.
