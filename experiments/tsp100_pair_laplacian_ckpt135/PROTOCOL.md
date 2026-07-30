# TSP100 checkpoint-conditioned pair-Laplacian protocol

This audit uses one fixed TSP100 epoch-135 checkpoint and multiple independent
rollout seeds. Every archived loss is evaluated on the same sampled all-pairs
comparison graph within a seed.

The audit deliberately separates:

- canonical Bradley--Terry information, `p(1-p)`, which is shared by all losses
  at a fixed checkpoint and sampler;
- loss-specific empirical-Fisher conductance,
  `(dL / d margin_e)^2`;
- signed first-order node drift.

Nodes are aligned by ascending tour-cost rank. Each per-instance Laplacian is
normalised before averaging, while raw margin-gradient scale and common-shift
leakage are retained as separate scalars.

Utility is evaluated without fitness:

1. cross-seed retrieval of the same loss;
2. between-loss versus within-loss variance;
3. neighbourhood overlap with the actual signed one-step node drift.

The matrix is considered a useful sensitivity descriptor when top-5 retrieval
is at least 0.5, between/within variance exceeds 1, and top-1 retrieval exceeds
three times chance. It is considered sufficient alone only if it matches the
matrix-plus-drift retrieval and has substantial drift-neighbourhood overlap.
