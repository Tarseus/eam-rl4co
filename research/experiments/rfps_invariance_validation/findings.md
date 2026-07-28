# Fisher-invariant validation: findings

The three experiments in `protocol.md` were run on the locked 40-candidate
matched set and 65-candidate external set, using 16 checkpoint-135 TSP100
rollout probes and the fixed step length `ell = 0.03`.

## Decision

Collecting more task instances does **not** create a larger Fisher--Euclidean
performance gap. It mainly reduces the variance caused by which instances are
used. Non-uniform starting distributions also do not separate the methods
reliably. The clean empirical reason to retain Fisher geometry is instead that
it preserves the candidate-neighbor relation under coordinate
reparameterization, while an equally expensive Euclidean two-point descriptor
does not.

## A. Number of task instances

- From `R=2` to `R=12`, the 95% subset interval width of Fisher two-point rho
  contracts from `0.042--0.114` to `0.012--0.014` across the three labels.
- At `R=16`, Fisher minus Euclidean rho is `0.0043` on matched scratch,
  `0.0051` on matched warm, and `0.0006` on external scratch.
- The gap does not grow monotonically with `R`; using more than eight probes is
  therefore a stability/compute choice rather than a new source of
  discrimination.

## B. Coordinate reparameterization

- Fisher pairwise distances, top-1 neighbors, and top-5 neighborhoods have
  agreement `1.000` for every tested condition number and permutation.
- At coordinate condition number 30, Euclidean top-1 agreement falls to
  `0.900` on matched-40 and `0.831` on external-65; top-5 overlap falls to
  `0.930` and `0.852`.
- On matched scratch, three of five condition-30 coordinate permutations raise
  Euclidean false-skip risk from `0.0219` in the identity chart to `0.1469`
  under the same 200 history orders. Fisher false-skip risk stays exactly
  `0.0219`.

The last result matters more than the high global distance-rank agreement:
screening decisions depend on the nearest-neighbor tail, and that tail changes
under an arbitrary Euclidean coordinate scale.

## C. Non-uniform starts

- At `beta=1`, median ESS ratio falls to `0.521`, so the test reaches a
  materially non-uniform empirical distribution.
- Euclidean/Fisher distance-rank agreement remains `0.997--0.998`.
- Top-1 agreement and fitness correlation change non-monotonically, and the
  method with the larger rho changes across labels and beta values.

This experiment does not support adding a non-uniform anchor. The final method
keeps a single uniform start and treats the coordinate stress test, rather than
an engineered start distribution, as the direct validation of Fisher
invariance.

## Artifacts

- Raw and summarized tables: `results/*.csv`
- Machine-readable summary: `results/results.json`
- Publication figure: `results/fig_validation.pdf`
- Reproduction: `python run_validation.py` followed by
  `python plot_results.py`
