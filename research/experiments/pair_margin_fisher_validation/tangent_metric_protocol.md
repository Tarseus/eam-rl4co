# Fisher tangent-metric extension protocol

## Why this is a separate test

The completed real-margin experiment used `F^{-1/2} dL`. That is the dual
Fisher representation of a loss covector and changes the implied update into a
natural-gradient direction. It performed worse against training carried out by
Adam.

This extension keeps the existing realizable Euclidean direction fixed and
uses Fisher only to measure that tangent. It tests a different hypothesis and
is committed before the new descriptors are computed.

## Frozen objects

Reuse without change:

- the same 32 candidates and their margin-only gate;
- the same 16 real checkpoint margin anchors;
- the same incidence matrices, Bernoulli factors, and Helmert quotient;
- the same one-step and valid disjoint three-step realized-update targets;
- the same metrics, grouping, bootstrap procedure, and secondary score
  analysis.

No temperature, clipping, damping, or probability floor is introduced.

## Added descriptors

Let `z_C = H^T B^T dL_C/ds` be the realizable Euclidean direction and
`F = H^T B^T diag(mu(1-mu)) B H` the shared pullback Fisher metric.

Add:

1. `node_fisher_tangent = F^{1/2} z_C`;
2. `pair_fisher_tangent = diag(mu(1-mu))^{1/2} dL_C/ds`.

Thus ordinary dot products of the stored node feature equal `z_C^T F z_D`.
Unlike the earlier `F^{-1/2}` feature, this does not claim that training follows
a natural gradient. It asks whether Fisher local distinguishability is the
right metric for the direction already supplied by the current optimization
proxy.

Every block is unit normalized exactly as before. The primary comparison is
`node_fisher_tangent` versus `node_euclidean`; the ambient pair feature is a
diagnostic.

## Decision rule

Promote the tangent Fisher metric only if:

- on the valid three-step target it has higher distance correlation and lower
  normalized nearest-neighbor error than `node_euclidean` on both candidate
  sources;
- the pooled three-step bootstrap interval for
  `NN_error(node_fisher_tangent) - NN_error(node_euclidean)` is strictly below
  zero; and
- it loses no more than `0.01` distance correlation on either one-step source.

Otherwise the real-margin restriction remains useful, but Fisher geometry is
not retained in the screening descriptor.

## Status

This is a retrospective exploratory extension. Earlier targets and the failure
of the dual-Fisher descriptor are known; tangent-metric outcomes are not known
at commit time.
