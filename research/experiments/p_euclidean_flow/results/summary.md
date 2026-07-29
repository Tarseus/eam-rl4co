# Result: true probability-Euclidean versus Fisher response flow

Date completed: 2026-07-29

## Outcome

The locked scratch + epoch-135 experiment does **not** support a practical
predictive advantage for the Fisher--Rao flow over a true Euclidean flow in
fixed probability coordinates at arc length `0.03`.

This is a stronger control than the earlier `log p` / logit-Euclidean
retraction.  Both methods receive the same probability-coordinate training
covector.  They differ only in the metric used to raise that covector:

\[
v_{\mathrm E}=P(c\oslash p),
\qquad
v_{\mathrm{FR}}=c-p\mathbf 1^\top c.
\]

The result is therefore not caused by accidentally feeding an
already-Fisher-raised vector to the Euclidean baseline.

## Primary external result (65 candidates, 30 substeps)

The preregistered primary bank is the equal-weight product of scratch and
epoch 135.

| Descriptor | Fitness-gap rho | false20 |
|---|---:|---:|
| one-point | 0.708857 | 0.067308 |
| legacy logit two-point | **0.709913** | 0.067308 |
| legacy Fisher one-step | 0.707533 | 0.067308 |
| true p-Euclidean flow | 0.707822 | 0.067308 |
| Fisher--Rao flow | 0.708097 | 0.067308 |

Fisher improves rho over true p-Euclidean by only `0.000276`; their false20
values are identical.  Their pairwise-distance rankings have Spearman rho
`0.999533`, with top-1 nearest-neighbor agreement `0.938462`.  Thus they fail
both parts of the locked success criterion: the descriptors are not
materially distinct on the primary bank, and Fisher has no meaningful
predictive gain.

## Per-bank external result

| Bank | Geometry | Fitness-gap rho | false20 | E/FR distance rho | E/FR top-1 agreement |
|---|---|---:|---:|---:|---:|
| scratch | p-Euclidean | 0.639613 | 0.062350 | 0.999815 | 0.892308 |
| scratch | Fisher--Rao | 0.639412 | 0.066826 | 0.999815 | 0.892308 |
| epoch 135 | p-Euclidean | 0.718586 | 0.018735 | 0.999748 | 0.969231 |
| epoch 135 | Fisher--Rao | 0.718748 | 0.016279 | 0.999748 | 0.969231 |

The scratch bank alone changes a few nearest-neighbor identities, but Fisher
is slightly worse there.  At epoch 135 Fisher has a `0.000162` rho gain and a
`13.1%` false20 reduction, both below the preregistered practical threshold.

## Matched scratch/warm transfer check (40 candidates)

On the product bank, the joint-fitness-distance rho is `0.786318` for true
p-Euclidean and `0.786935` for Fisher.  Both have false20 `0.019231`.  Their
distance rankings have rho `0.999939`, with top-1 agreement `0.975`.

## Numerical audit

- Minimum initial E/FR direction cosine: `0.9999999999999997`.
- Maximum tangent-sum error in the authoritative 30-step runs: `8.73e-11`.
  One 60-step convergence-audit trajectory reached `1.46e-10`, narrowly above
  the locked absolute `1e-10` tolerance; this is disclosed as a tolerance
  miss rather than silently counted as a pass.
- Maximum substep-length error: `3.34e-12`.
- Minimum probability reached: `0.0072366` from the uniform start `0.01`.
- Stationary fraction: `0`.
- All 30-versus-60-step distance correlations exceed `0.99985`.

The two fields therefore were genuinely integrated.  The isolated 60-step
absolute cancellation residual does not affect the authoritative 30-step
descriptors, and the negative result is not explained by a hidden equality
or a nonconverged solver.

## Interpretation

At the uniform start, the normalized fields are exactly equal.  Over a short
Fisher arc of `0.03`, probabilities remain close enough to uniform that the
metric-dependent separation is only a higher-order perturbation.  It changes
some individual nearest neighbors but barely changes the global distance
ordering or fitness prediction.

The experiment supports a limited conclusion: Fisher--Rao remains the
intrinsic and positivity-compatible geometry for probability distributions,
but this particular short local descriptor does not obtain an empirical
screening advantage merely from using the Fisher metric.  The paper should
not claim that fixed-coordinate Euclidean geometry is experimentally
inferior on this audit.

Raw aggregate evidence is in `results.json` and `metrics.csv`; per-flow
numerical diagnostics are in `diagnostics.csv`.  `descriptors.npz` is retained
locally as a regenerable 84 MB artifact and is not intended for source control.
