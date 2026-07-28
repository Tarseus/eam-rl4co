# Non-uniform RFPS reference measure: findings

## Decision

Keep the current candidate-independent **uniform reference measure on the
checkpoint-135 rollout bank**. Neither anchor importance correction nor fixed
pair-exposure weighting improves the deployed descriptor over that baseline.

## What the shared-bank experiment showed

The mixed bank contains 100 scratch and 100 checkpoint-135 trajectories per
instance. Teacher-forced likelihoods reproduce the generating-policy values
exactly. Scratch and warm policies have effectively disjoint support:

- normalized anchor-IS ESS is 0.50;
- scratch assigns only about `1e-10` mass to the warm half after the required
  interior floor, and warm behaves symmetrically;
- cross-anchor log importance weights are roughly `-1775` to `-2384`.

Relative to an inappropriate uniform measure on this mixed support, anchor-IS
is clearly better and the sign is consistent on both probe seeds:

| Anchor/target | Uniform rho | Anchor-IS rho | Uniform false skip | Anchor-IS false skip |
|---|---:|---:|---:|---:|
| scratch | 0.336 | 0.586 | 0.111 | 0.036 |
| warm | 0.390 | 0.652 | 0.000 | 0.000 |

This supports importance correction *if a mixed bank must be used*. It does
not show that the mixed bank is a better probe. Since the two policies do not
overlap, the correction merely selects the corresponding half of the stored
bank; it does not transfer useful information across anchors.

## Pair exposure

The actual builder enumerates all objective-ordered pairs. Its endpoint
marginal is not exactly uniform because the strict preference mask omits edges
between tied-objective tours. Every mixed-bank instance contains tied groups
(46--55 groups per seed, maximum multiplicity 8), producing a maximum absolute
coordinate deviation of `1.48e-4`. Although small in `q`, this changes mixed-
bank scratch rho from 0.336 to 0.390 without improving false-skip risk. The
effect depends on how duplicate/tied support points are represented and is not
treated as a robust gain.

On the mixed bank, rank-gap exposure raises scratch rho from 0.336 to 0.415 but
worsens false-skip risk from 0.111 to 0.143 and does not improve warm rho. When
combined with anchor-IS, rank-gap exposure adds only 0.004--0.005 rho. A paired
candidate bootstrap places zero inside the 95% interval for this incremental
gain (`[-0.0016, 0.0094]` scratch and `[-0.0019, 0.0176]` warm).

The decisive follow-up applies pair exposure to the current checkpoint-135
bank. Its literal all-pair endpoint changes scratch/warm rho only from
0.8113/0.8606 to 0.8118/0.8609. More purposeful rank-gap and hard-pair
exposures are worse:

| Target | Uniform rho | All-pair rho | Rank-gap rho | Hard-pair rho | Uniform false skip | Rank-gap false skip |
|---|---:|---:|---:|---:|---:|---:|
| scratch | 0.811 | 0.812 | 0.804 | 0.805 | 0.031 | 0.112 |
| warm | 0.861 | 0.861 | 0.846 | 0.857 | 0.000 | 0.000 |

Both probe seeds have the same direction of change. Candidate-independent
pair exposure is therefore not retained.

## Fisher versus Euclidean

Non-uniform anchors do make the two neighborhoods less identical: top-1
agreement falls from about 0.975 to 0.875--0.900. This does not create a Fisher
accuracy advantage. Under anchor-IS, scratch rho is 0.590 for the Euclidean
pair and 0.586 for the Fisher pair; warm rho is 0.651 and 0.652. The reason to
retain Fisher geometry remains coordinate invariance, not an engineered
non-uniform start.

## Interpretation

The reference bank is better viewed as a **behavioral probe** than as an
importance-sampling estimator of the eventual training distribution. A
checkpoint-135 bank supplies informative margins and solution diversity and
predicts both scratch and warm fitness differences better than anchor-matched
mixed-bank measures. Uniform weighting gives every observed probe trajectory
equal voice; the candidate-induced `q1` then records how the searched loss
wants to redistribute that common probe.

## Technical lessons

- Fisher--Rao calculations require an interior `q0`; likelihood-ratio
  underflow must be floored before square-root coordinates are formed.
- POMO's multistart dimension cannot concatenate two duplicate 100-start sets
  into one 200-start decoder call. The groups must be teacher-forced
  separately and concatenated afterward.
- Importance weighting is not enough if zero-mass trajectories still enter
  unweighted rank/min statistics. This is another reason not to use the
  nearly disjoint mixed support as the deployed probe.

## Artifacts

- Locked protocol: `protocol.md`
- Shared-bank builder: `build_shared_bank.py`
- Main experiment: `run_experiment.py`
- Confirmatory tables: `results_floor/metrics.csv`, `results_floor/q_stats.csv`
- Warm-bank pair-exposure follow-up:
  `results_warmbank_pair_exposure/metrics.csv`
- Literal all-pair warm-bank follow-up: `results_warmbank_allpair/metrics.csv`
