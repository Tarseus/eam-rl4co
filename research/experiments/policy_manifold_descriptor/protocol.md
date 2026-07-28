# Protocol: preference-program gradients on the real policy manifold

Date locked: 2026-07-28

## Question

Does representing a preference program's loss gradient on the product of the
model's actual categorical action simplices provide a useful first-order
descriptor that the uniform rollout-weight simplex cannot provide?

## Hypothesis

The current rollout-weight construction starts from \(q_i=1/M\), where the
Fisher metric is a constant multiple of the Euclidean metric.  By contrast,
the checkpoint policy distributions \(\pi_\theta(a\mid s)\) are nonuniform.
Therefore categorical Fisher weighting should change candidate neighborhoods
at first order and should improve either fitness-gap prediction or local
false-skip safety relative to an equally normalized logit-Euclidean gradient.

## Locked construction

For trajectory \(i\), let

\[
c_i=-\frac{\partial L_C}{\partial p_i},
\qquad
p_i=\sum_t \log \pi(a_{it}\mid s_{it}).
\]

At a visited decision state, the loss gradient with respect to policy logits
is proportional to

\[
c_i(e_{a_{it}}-\pi_{it}).
\]

The exact squared norms needed for the product-manifold descriptors are

\[
W_i^{\mathrm{E}}
=\sum_t\left(1-2\pi_{it}(a_{it})
             +\sum_a\pi_{it}(a)^2\right),
\]

\[
W_i^{\mathrm{F}}
=\sum_t\left(\frac{1}{\pi_{it}(a_{it})}-1\right).
\]

For each probe, compare four unit-normalized vectors:

1. raw trajectory coefficient \(c\);
2. centered trajectory coefficient \(c-\bar c\);
3. logit-Euclidean policy gradient
   \(c_i\sqrt{W_i^{\mathrm E}}\);
4. categorical-Fisher policy gradient
   \(c_i\sqrt{W_i^{\mathrm F}}\).

Probe vectors are concatenated with \(1/\sqrt R\) scaling.  Epoch-31,
epoch-135, and their equal-weight product are evaluated.  No second point,
curvature, adaptive length, learned weighting, or additional search module is
allowed in this experiment.

## Data

- Same 16 TSP100 instances already used by the RFPS studies: seeds 20260728
  and 20260729, eight instances per seed.
- 100 POMO rollouts per instance.
- Checkpoints: epoch 31 and epoch 135.
- Candidate sets:
  - 40 matched candidates with scratch and warm-start outcomes;
  - 65 external candidates with scratch outcomes.

The 40-candidate set is exploratory.  The 65-candidate set is validation for
this locked descriptor definition, but is not claimed to be globally unseen
because it has been inspected in earlier RFPS experiments.

## Metrics

- Spearman correlation between descriptor distance and fitness gap.
- Median nearest-neighbor fitness gap.
- False-skip fraction among the closest 20% of candidate pairs.
- Fisher versus logit-Euclidean pairwise-distance rank correlation and
  top-1 nearest-neighbor agreement.

For the matched set, the primary correlation is the distance to the
two-dimensional standardized scratch/warm target.  For the external set, it
is scratch fitness gap.

## Success criteria

The hypothesis is supported only if the Fisher descriptor satisfies both:

1. it changes neighborhoods materially on the external set
   (distance rank correlation below 0.995 or top-1 agreement below 0.90
   relative to logit-Euclidean); and
2. it provides useful validation behavior: either primary correlation
   improves by at least 0.01, or false-skip decreases by at least 20%
   relative without reducing primary correlation by more than 0.005.

Anything weaker is recorded as no practical advantage.  A positive result
would justify a separately preregistered two-point policy-flow experiment;
a negative result stops this branch.

## Sanity checks

- Per-step selected log probabilities must sum to the stored trajectory
  log-likelihood within \(10^{-5}\).
- Rerun objectives and actions must reproduce the existing bank for the same
  seed.
- All norm weights must be finite and nonnegative.
- The forced POMO first action contributes zero to both norms.
