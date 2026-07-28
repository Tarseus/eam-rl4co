# Three-step on-policy actual-update alignment

- Smoke run: `False`
- Candidates: 32
- On-policy optimizer steps: 3
- Train / held-out instances: 2 / 2
- POMO starts: 100
- Elapsed seconds: 161.9

| Group | Method | Distance rho | Normalized NN error | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|
| matched | one_point | 0.665 | 0.698 | 0.438 | 0.538 |
| matched | euclidean_two_point | 0.665 | 0.698 | 0.438 | 0.538 |
| matched | fisher_two_point | 0.667 | 0.698 | 0.438 | 0.538 |
| external | one_point | 0.708 | 0.510 | 0.562 | 0.637 |
| external | euclidean_two_point | 0.704 | 0.510 | 0.625 | 0.625 |
| external | fisher_two_point | 0.698 | 0.510 | 0.562 | 0.613 |
| pooled | one_point | 0.695 | 0.573 | 0.375 | 0.456 |
| pooled | euclidean_two_point | 0.695 | 0.573 | 0.406 | 0.463 |
| pooled | fisher_two_point | 0.695 | 0.573 | 0.375 | 0.450 |

## Fisher minus Euclidean nearest-neighbor error

| Group | Mean | 95% low | 95% high |
|---|---:|---:|---:|
| matched | 0 | 0 | 0 |
| external | 3.234e-07 | 0 | 9.703e-07 |
| pooled | 1.617e-07 | 0 | 4.851e-07 |

## Preregistered extension decision

- Fisher promotion rule met: **False**
