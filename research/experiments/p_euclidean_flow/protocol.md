# Protocol: true probability-Euclidean versus Fisher response flow

Date locked: 2026-07-29

## Question

When rollout probabilities have fixed semantic coordinates, does raising the
same preference-training covector with the Fisher--Rao metric produce a more
useful short response-flow descriptor than raising it with the standard
Euclidean metric on the embedded probability simplex?

This is not the earlier logit-Euclidean control.  That control moved in
`z = log p` using an already Fisher-raised tangent.  The present experiment
changes the metric used to turn the common covector into a tangent vector.

## Locked training covector

For a candidate program pair `C=(f,g)` and an interior rollout distribution
`p`, recompute the same fixed-bank preference inputs used by RFPS.  Let

\[
c_i(p)=-\frac{\partial L_C}{\partial\log p_i}.
\]

The sampling weights and the statistics constructed from them are evaluated
at the current `p` but remain detached during this derivative.  This is the
same fixed-batch training semi-gradient used by the existing RFPS code.  It
defines a covector field even if it is not the total derivative of one global
scalar `L_C(p)`.  Both geometries receive exactly the same `c(p)`.

The corresponding probability-coordinate covector is

\[
a(p)=\frac{\partial L_C}{\partial p}=-c(p)\oslash p.
\]

## Two metric-raised fields

Let

\[
P=I-\frac1N\mathbf1\mathbf1^\top.
\]

The negative projected Euclidean field in the fixed probability coordinates
is

\[
v_{\mathrm E}(p)=-P a(p)=P\bigl(c(p)\oslash p\bigr).
\]

The negative Fisher--Rao field is

\[
v_{\mathrm{FR}}(p)
=-\left(\operatorname{diag}(p)-pp^\top\right)a(p)
=c(p)-p\,\mathbf1^\top c(p).
\]

Both tangents must sum to zero.  At the uniform starting distribution
`p0=(1/N,...,1/N)`, they are exactly proportional:

\[
v_{\mathrm E}(p_0)=N v_{\mathrm{FR}}(p_0).
\]

Thus the normalized first direction is shared; any later separation must be
caused by the metric after the probability distribution becomes nonuniform.

## Locked flow and numerical integration

For `g in {E, FR}`, follow the integral curve of its own field while using
Fisher--Rao arc length only as a common statistical clock:

\[
\frac{dp_g}{ds}
=\frac{v_g(p_g)}{\|v_g(p_g)\|_{\mathrm{FR}}},
\qquad
p_g(0)=p_0,
\qquad
s\in[0,0.03].
\]

Multiplying a vector field by a positive scalar changes only its time
parameterization, not its integral curves.  The shared clock therefore does
not turn the Euclidean field into a Fisher field; it ensures equal
distributional displacement budgets.

Use 30 exponential-Euler substeps on the square-root sphere for both fields.
The exponential map is a positivity-preserving numerical retraction shared by
the two ODEs.  Repeat with 60 substeps only as a convergence audit.  No length
tuning, clipping, projection rescue, learned weighting, or extra channel is
allowed.

## Locked descriptors

For the Euclidean flow, normalize directions with the Euclidean norm and use
the flat connection:

\[
d_{\mathrm E}(p)=v_{\mathrm E}(p)/\|v_{\mathrm E}(p)\|_2.
\]

For the Fisher flow, map the tangent to square-root-sphere coordinates,
normalize there, and parallel transport the terminal direction backward
along the discretized flow path to `p0`:

\[
d_{\mathrm{FR}}(p)
=\operatorname{unit}\!\left(v_{\mathrm{FR}}(p)/(2\sqrt p)\right).
\]

The primary descriptor concatenates the initial and terminal directions:

\[
\Phi_g(C)=\frac1{\sqrt2}
\left[d_g(p_0)\,\|\,\operatorname{PT}_{p_g(0)\leftarrow p_g(0.03)}
d_g(p_g(0.03))\right].
\]

For the Euclidean flat connection, parallel transport is the identity.  At
the uniform base, its tangent coordinates differ from square-root coordinates
only by a shared scalar, so the initial normalized blocks are directly
comparable.

Secondary diagnostics report endpoint distributions and endpoint log-map
directions, but they are not used to choose the primary result.

## Data and controls

- The same 40 matched candidates with scratch and warm-start outcomes.
- The same 65 external candidates with scratch outcomes.
- Sixteen TSP100 probes per policy bank: two fixed seeds with eight instances
  each.
- Report epoch-31, epoch-135, and their equal-weight product.  The product is
  the preregistered primary audit for continuity with the equal-arc control.
- Controls: one-point direction, the existing logit-Euclidean two-point
  descriptor, and the existing one-step Fisher descriptor.

## Metrics

- Spearman correlation between descriptor distance and the true fitness gap.
- Median nearest-neighbor fitness gap.
- False-skip fraction among the closest 20% of candidate pairs.
- Pairwise-distance rank correlation and top-1 agreement between true
  p-Euclidean and Fisher descriptors.
- Thirty-versus-sixty-step distance correlation and nearest-neighbor
  agreement.

For the matched set, the primary target is the two-dimensional standardized
scratch/warm fitness distance.  For the external set, it is scratch fitness
gap.

## Success criterion

The claim that Fisher provides a practically useful metric advantage is
supported only if, on the external product bank:

1. the two descriptors are materially distinct (distance rank correlation
   below 0.995 or top-1 agreement below 0.90); and
2. Fisher either improves rho by at least 0.01, or reduces false20 by at least
   20% while losing no more than 0.005 rho, relative to true p-Euclidean.

Matched joint rho and false20 are a transfer check.  Failure of the locked
criterion is recorded without changing length, normalization, or channels.

## Sanity checks

- `sum(v_E)` and `sum(v_FR)` are below `1e-10` in absolute value.
- At uniform `p0`, the two normalized initial directions have cosine at least
  `1-1e-10` for every nonstationary candidate/probe.
- Every probability remains finite, strictly positive, and sums to one.
- Every nonstationary substep has Fisher--Rao length `0.03 / steps` within
  `1e-8` numerical tolerance.
- Thirty- and sixty-step results must have pairwise-distance rho at least
  0.999 for each primary descriptor; otherwise the 30-step result is not
  trusted.
- Stationary or non-finite candidates are audited rather than interpreted as
  close neighbors.

## User-directed amendment before metric inspection

Amended on 2026-07-29 after the first run had computed descriptors for part of
the epoch-31 bank but before it had produced any aggregate metric or result
file.  The user requested that checkpoint selection not be part of this test.

- Replace epoch 31 with the scratch policy bank.
- Evaluate scratch, epoch 135, and their equal-weight product.
- The scratch + epoch-135 product becomes the preregistered primary audit.
- All fields, flow length, step counts, descriptors, metrics, and success
  thresholds above remain unchanged.

The interrupted epoch-31 descriptors were never aggregated or inspected and
are not retained as results.
