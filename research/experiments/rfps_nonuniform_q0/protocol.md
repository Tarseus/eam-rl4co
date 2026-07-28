# Non-uniform reference measure for RFPS

## Question

Does a statistically grounded non-uniform reference distribution improve the
ability of normalized one- and two-point descriptors to predict real scratch
and warm-start training differences?

The experiment separates two sources of non-uniformity:

1. **Anchor correction.** Scratch and checkpoint-135 policies share one mixed
   rollout bank. Self-normalized multiple-importance weights represent either
   policy on that support.
2. **Pair exposure.** A candidate-independent preference-pair proposal induces
   an endpoint marginal over trajectories.

## Locked data

- Candidate set: the 40 candidates shared by existing scratch and warm
  high-fidelity evaluations.
- Real targets: existing scratch and checkpoint-135 fitness deltas.
- Instances: the two existing seeds, eight TSP100 instances each.
- Shared support: concatenate 100 scratch and 100 checkpoint-135 trajectories
  generated for the same instance.
- Anchors: scratch policy seed 1234 and `tsp100_epoch_135.ckpt`.
- Step length: Fisher--Rao arc length `ell = 0.03`.

No network training is run. The policies only teacher-force fixed trajectories
to obtain cross-policy log likelihoods.

## Reference measures

Let the shared-bank proposal be
`mu = 0.5 * pi_scratch + 0.5 * pi_warm`. For anchor `a`, define
`b_a(m) proportional to pi_a(tau_m) / mu(tau_m)`.

The locked conditions are:

1. `uniform`: equal mass over all 200 trajectories.
2. `anchor_is`: `b_a`.
3. `all_pair_endpoint`: endpoint marginal of all objective-ordered pairs under
   a uniform trajectory base. It must equal `uniform` up to rounding.
4. `gap_endpoint`: endpoint marginal with candidate-independent exposure
   `rho_ij proportional to rank_gap(i,j)`.
5. `anchor_gap_endpoint`: endpoint marginal of
   `b_a(i) * b_a(j) * rank_gap(i,j)`.
6. `anchor_hard_endpoint`: endpoint marginal of
   `b_a(i) * b_a(j) * sigmoid(-0.05 * (p_a(i)-p_a(j)))`.

The searched loss is never used to construct `q0`. Every non-uniform measure is
mixed with the smallest uniform component needed to keep normalized ESS at
least 0.40.

## Descriptors

For every condition and anchor:

- `one_point`: per-probe Fisher-unit initial direction.
- `fisher_two_point`: direction after one candidate-induced Fisher step.
- `euclidean_two_point`: matched two-derivative logit-Euclidean baseline.

Effective log probabilities use `p_a(q) = p_a + log(q / q0_a)`, so the
reference state always evaluates the actual anchor logits.

## Metrics and decision rule

Primary method: `fisher_two_point`.

Primary metrics:

- Spearman correlation between descriptor distance and absolute real-fitness
  difference.
- Mean false-skip rate over 200 random history orders at a 20% skip rate and
  fitness tolerance 0.01.

Secondary metrics are nearest-neighbor fitness gaps, Euclidean/Fisher
neighborhood agreement, normalized ESS, maximum importance ratio, and
descriptor drift from the uniform shared-bank baseline.

An initialization is supported only if it improves primary correlation without
worsening false-skip rate by more than 0.02, or reduces false-skip rate without
lowering correlation by more than 0.02. Effects must have the same sign on both
probe seeds when evaluated separately.

## Hypotheses

- **H1:** Anchor importance correction improves its corresponding real target.
- **H2:** Dense all-pairs exposure is uninformative because every trajectory
  has equal graph degree.
- **H3:** Candidate-independent pair exposure improves tail screening.
- **H4:** Combining anchor correction and pair exposure is most useful.
- **H5:** Non-uniform `q0` separates Fisher and Euclidean neighborhoods, but
  separation alone is not considered useful.

