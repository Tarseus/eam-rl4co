# Fixed-chart actual-update alignment

- Smoke run: `False`
- Candidates: 32
- Train / held-out instances: 2 / 2
- POMO starts: 100
- Elapsed seconds: 44.7

| Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|
| matched | one_point | 0.975 | 0.267 | 0.938 | 0.800 |
| matched | euclidean_two_point | 0.975 | 0.267 | 0.938 | 0.800 |
| matched | fisher_two_point | 0.975 | 0.267 | 0.938 | 0.800 |
| external | one_point | 0.804 | 0.397 | 0.625 | 0.775 |
| external | euclidean_two_point | 0.802 | 0.397 | 0.625 | 0.750 |
| external | fisher_two_point | 0.800 | 0.397 | 0.688 | 0.738 |
| pooled | one_point | 0.926 | 0.308 | 0.688 | 0.694 |
| pooled | euclidean_two_point | 0.926 | 0.308 | 0.688 | 0.706 |
| pooled | fisher_two_point | 0.926 | 0.308 | 0.688 | 0.713 |

## Fisher minus Euclidean nearest-neighbor error

| Group | Mean | 95% low | 95% high |
|---|---:|---:|---:|
| matched | 0 | 0 | 0 |
| external | -2.199e-07 | -6.598e-07 | 0 |
| pooled | -1.1e-07 | -3.299e-07 | 0 |

## Preregistered decision

- Fisher promotion rule met: **False**
