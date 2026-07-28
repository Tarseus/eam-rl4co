# Real pair-margin Fisher tangent-metric extension

- Retrospective/exploratory: `True`
- Smoke: `False`
- Candidates: 32
- Real-margin anchors: 16
- Elapsed seconds: 10.2

| Target | Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |
|---|---|---|---:|---:|---:|---:|
| one_step | matched | node_euclidean | 0.982 | 0.267 | 0.938 | 0.838 |
| one_step | matched | node_fisher_tangent | 0.973 | 0.267 | 0.938 | 0.788 |
| one_step | matched | pair_euclidean | 0.879 | 0.307 | 0.875 | 0.663 |
| one_step | matched | pair_fisher_tangent | 0.845 | 0.366 | 0.562 | 0.650 |
| one_step | external | node_euclidean | 0.698 | 0.411 | 0.750 | 0.688 |
| one_step | external | node_fisher_tangent | 0.665 | 0.430 | 0.688 | 0.675 |
| one_step | external | pair_euclidean | 0.777 | 0.422 | 0.625 | 0.725 |
| one_step | external | pair_fisher_tangent | 0.694 | 0.450 | 0.562 | 0.700 |
| one_step | pooled | node_euclidean | 0.893 | 0.308 | 0.688 | 0.706 |
| one_step | pooled | node_fisher_tangent | 0.866 | 0.319 | 0.719 | 0.644 |
| one_step | pooled | pair_euclidean | 0.861 | 0.344 | 0.594 | 0.619 |
| one_step | pooled | pair_fisher_tangent | 0.744 | 0.432 | 0.406 | 0.525 |
| three_step | matched | node_euclidean | 0.955 | 0.529 | 0.562 | 0.725 |
| three_step | matched | node_fisher_tangent | 0.940 | 0.529 | 0.562 | 0.675 |
| three_step | matched | pair_euclidean | 0.827 | 0.548 | 0.500 | 0.562 |
| three_step | matched | pair_fisher_tangent | 0.801 | 0.569 | 0.438 | 0.562 |
| three_step | external | node_euclidean | 0.622 | 0.553 | 0.688 | 0.700 |
| three_step | external | node_fisher_tangent | 0.590 | 0.569 | 0.625 | 0.675 |
| three_step | external | pair_euclidean | 0.646 | 0.520 | 0.562 | 0.713 |
| three_step | external | pair_fisher_tangent | 0.624 | 0.551 | 0.562 | 0.713 |
| three_step | pooled | node_euclidean | 0.835 | 0.478 | 0.594 | 0.556 |
| three_step | pooled | node_fisher_tangent | 0.811 | 0.495 | 0.625 | 0.550 |
| three_step | pooled | pair_euclidean | 0.779 | 0.477 | 0.406 | 0.550 |
| three_step | pooled | pair_fisher_tangent | 0.703 | 0.549 | 0.344 | 0.506 |

## Node Fisher tangent minus node Euclidean NN error

| Target | Group | Mean | 95% low | 95% high |
|---|---|---:|---:|---:|
| one_step | matched | 0 | 0 | 0 |
| one_step | external | 0.004959 | 0 | 0.01325 |
| one_step | pooled | 0.00304 | 0 | 0.007288 |
| three_step | matched | 0 | 0 | 0 |
| three_step | external | 0.007726 | 0 | 0.02062 |
| three_step | pooled | 0.009826 | -0.0005939 | 0.02817 |

## Preregistered decision

- Fisher tangent-metric promotion rule met: **False**
