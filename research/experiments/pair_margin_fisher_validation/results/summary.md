# Real pair-margin Fisher validation

- Retrospective/exploratory: `True`
- Smoke: `False`
- Candidates: 32
- Real-margin anchors: 16
- Elapsed seconds: 10.4

| Target | Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |
|---|---|---|---:|---:|---:|---:|
| one_step | matched | node_euclidean | 0.982 | 0.267 | 0.938 | 0.838 |
| one_step | matched | pullback_fisher | 0.917 | 0.308 | 0.875 | 0.750 |
| one_step | matched | pair_euclidean | 0.879 | 0.307 | 0.875 | 0.663 |
| one_step | matched | pair_fisher | 0.926 | 0.309 | 0.875 | 0.725 |
| one_step | external | node_euclidean | 0.698 | 0.411 | 0.750 | 0.688 |
| one_step | external | pullback_fisher | 0.614 | 0.436 | 0.625 | 0.763 |
| one_step | external | pair_euclidean | 0.777 | 0.422 | 0.625 | 0.725 |
| one_step | external | pair_fisher | 0.688 | 0.442 | 0.562 | 0.738 |
| one_step | pooled | node_euclidean | 0.893 | 0.308 | 0.688 | 0.706 |
| one_step | pooled | pullback_fisher | 0.801 | 0.319 | 0.625 | 0.663 |
| one_step | pooled | pair_euclidean | 0.861 | 0.344 | 0.594 | 0.619 |
| one_step | pooled | pair_fisher | 0.840 | 0.436 | 0.500 | 0.619 |
| three_step | matched | node_euclidean | 0.955 | 0.529 | 0.562 | 0.725 |
| three_step | matched | pullback_fisher | 0.862 | 0.547 | 0.562 | 0.650 |
| three_step | matched | pair_euclidean | 0.827 | 0.548 | 0.500 | 0.562 |
| three_step | matched | pair_fisher | 0.853 | 0.554 | 0.500 | 0.537 |
| three_step | external | node_euclidean | 0.622 | 0.553 | 0.688 | 0.700 |
| three_step | external | pullback_fisher | 0.461 | 0.554 | 0.625 | 0.725 |
| three_step | external | pair_euclidean | 0.646 | 0.520 | 0.562 | 0.713 |
| three_step | external | pair_fisher | 0.505 | 0.584 | 0.562 | 0.712 |
| three_step | pooled | node_euclidean | 0.835 | 0.478 | 0.594 | 0.556 |
| three_step | pooled | pullback_fisher | 0.728 | 0.499 | 0.469 | 0.506 |
| three_step | pooled | pair_euclidean | 0.779 | 0.477 | 0.406 | 0.550 |
| three_step | pooled | pair_fisher | 0.748 | 0.486 | 0.438 | 0.506 |

## Pullback Fisher minus node Euclidean NN error

| Target | Group | Mean | 95% low | 95% high |
|---|---|---:|---:|---:|
| one_step | matched | 0.01231 | 0 | 0.03294 |
| one_step | external | 0.006449 | 0 | 0.01773 |
| one_step | pooled | 0.003125 | 0.0003517 | 0.006625 |
| three_step | matched | 0.01201 | 0 | 0.03265 |
| three_step | external | 0.0003991 | -0.01709 | 0.01828 |
| three_step | pooled | 0.01183 | -0.001256 | 0.03171 |

## Preregistered decision

- Fisher promotion rule met: **False**
- Maximum common-gradient ratio: `1.459e-16`
- Maximum common-shift loss change: `0.000e+00`
- Maximum centered-margin loss change: `0.000e+00`
- Maximum margin chain-rule error: `3.397e-16`
