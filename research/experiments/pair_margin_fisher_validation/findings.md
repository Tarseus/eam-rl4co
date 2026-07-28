# Findings: real pair-margin geometry

## Established facts

- All 32 selected programs are exactly margin-only on the 16 checkpoint
  anchors: common shifts change no loss, and the maximum relative chain-rule
  reconstruction error is `3.40e-16`.
- Real margins make the geometries genuinely different. The pullback Fisher
  condition number reaches `8.35e6`, and its descriptor-distance matrix has
  Pearson correlation only `0.784` with the realizable Euclidean matrix.
- The direct dual/natural Fisher descriptor does not align better with the
  realized Adam update.

## Dual Fisher result

For the one-step target, the realizable Euclidean versus pullback-Fisher
distance correlations are:

- matched: `0.982` versus `0.917`;
- external: `0.698` versus `0.614`.

For the valid three-step on-policy target:

- matched: `0.955` versus `0.862`;
- external: `0.622` versus `0.461`.

The Euclidean descriptor also wins the secondary high-fidelity score metrics.
The preregistered Fisher promotion rule fails.

## Interpretation

`F^{-1/2} dL` represents the loss covector under the dual Fisher metric, or
equivalently the Fisher-natural update. It strongly amplifies directions with
small `mu(1-mu)`, which are saturated, low-information pairs. The actual search
evaluation uses Adam rather than a pair-manifold natural-gradient optimizer,
so this change of update direction is not faithful to the target training
dynamics.

This negative result does not yet answer whether Fisher is useful as the metric
for an update direction supplied by the actual optimizer. That distinct
question uses a primal tangent representation `F^{1/2} v` and requires a
separate locked extension.

## Fisher tangent-metric extension

The locked extension kept the realizable Euclidean direction and measured it
with the primal Fisher metric. This also failed its promotion rule.

For one-step realized updates, Euclidean versus Fisher-tangent distance
correlations were:

- matched: `0.982` versus `0.973`;
- external: `0.698` versus `0.665`.

For valid three-step on-policy updates:

- matched: `0.955` versus `0.940`;
- external: `0.622` versus `0.590`.

The methods were not equivalent: their descriptor-distance matrices had
Pearson correlation `0.971` and mean absolute distance difference `0.092`.
Fisher therefore made a real change, but that change selected a less faithful
training topology.

## Current conclusion

The margin-only representation is strongly supported as a clean restriction:

`a_C = dL_C/ds`, followed by `c_C = B^T a_C`, exactly reconstructs the loss
gradient with respect to trajectory log-probabilities while discarding all
absolute-log-probability dependence and unrealizable pair-cycle components.

Neither inverse-Fisher naturalization nor primal-Fisher tangent measurement
improves alignment with Adam training. The standalone Bradley--Terry pair
Fisher omits the model Jacobian and Adam preconditioner that determine the
realized policy update. Under the current optimizer, the minimal supported
descriptor is therefore the realizable Euclidean margin gradient `B^T dL/ds`.

Fisher would become mechanically aligned only after either incorporating the
optimizer/model pullback or changing training itself to a natural-gradient
method. Both are materially different methods and are not justified by the
present screening evidence.
