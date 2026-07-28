# Three-step on-policy actual-update alignment

- Smoke run: `False`
- Candidates: 32
- On-policy optimizer steps: 3
- Train / held-out instances: 2 / 2
- POMO starts: 100
- Elapsed seconds: 99.0

| Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|
| matched | one_point | 0.971 | 0.529 | 0.562 | 0.812 |
| matched | euclidean_two_point | 0.971 | 0.529 | 0.562 | 0.812 |
| matched | fisher_two_point | 0.971 | 0.529 | 0.562 | 0.812 |
| external | one_point | 0.676 | 0.519 | 0.500 | 0.688 |
| external | euclidean_two_point | 0.675 | 0.519 | 0.500 | 0.688 |
| external | fisher_two_point | 0.673 | 0.519 | 0.562 | 0.675 |
| pooled | one_point | 0.870 | 0.458 | 0.531 | 0.556 |
| pooled | euclidean_two_point | 0.870 | 0.458 | 0.531 | 0.569 |
| pooled | fisher_two_point | 0.870 | 0.458 | 0.531 | 0.569 |

## Fisher minus Euclidean nearest-neighbor error

| Group | Mean | 95% low | 95% high |
|---|---:|---:|---:|
| matched | 0 | 0 | 0 |
| external | -1.049e-05 | -3.148e-05 | 0 |
| pooled | -5.247e-06 | -1.574e-05 | 0 |

## Preregistered extension decision

- Fisher promotion rule met: **False**
