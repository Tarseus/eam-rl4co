# H1 result summary

The locked first-order policy-manifold hypothesis is supported relative to
the logit-Euclidean control, but not yet sufficient for local screening.

| dataset | descriptor | primary rho | false20 |
|---|---|---:|---:|
| matched | raw coefficient | 0.8582 | 1.27% |
| matched | logit-Euclidean | **0.8581** | **1.28% |
| matched | policy Fisher | 0.8505 | 2.56% |
| external | raw coefficient | 0.7317 | **1.67%** |
| external | logit-Euclidean | 0.7220 | 11.97% |
| external | policy Fisher | **0.7545** | 3.78% |

External policy-Fisher versus logit-Euclidean distance rho is 0.9614 and
top-1 agreement is 83.1%. See `results.json` and `robustness.json` for
bank-specific, seed-split, random-subset, and trimmed diagnostics.

The external improvement is present in both independent eight-instance
seeds and remains after removing the three largest scratch scores. The raw
coefficient nevertheless retains better local false-skip behavior.
