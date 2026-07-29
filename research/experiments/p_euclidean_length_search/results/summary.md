# Exploratory result: response length for a Fisher advantage

Date completed: 2026-07-29

## Direct answer

Use Fisher arc length **`0.10`**.  On the locked scratch + epoch-135 product
bank, Fisher scores above true probability-Euclidean under the selection
metric.

At the operational 30-substep resolution:

| Geometry | Fitness-gap rho | false20 |
|---|---:|---:|
| true p-Euclidean | 0.706713 | 0.066667 |
| Fisher--Rao | **0.707213** | **0.064748** |

Thus Fisher gains `+0.000499` rho and also has the lower false20 value.

At 60 substeps, the rho advantage remains and becomes `+0.001123`:

| Geometry | Fitness-gap rho | false20 |
|---|---:|---:|
| true p-Euclidean | 0.706410 | **0.066194** |
| Fisher--Rao | **0.707533** | 0.068396 |

The rho ordering is numerically converged, but the false20 ordering is not.
Accordingly, the defensible tuned claim is only that Fisher has higher rho at
length `0.10`, not that it dominates every screening metric.

## Length search

| Length | p-E rho | Fisher rho | Fisher - p-E | Minimum probability |
|---:|---:|---:|---:|---:|
| 0.06 | 0.707503 | **0.707651** | +0.000148 | 4.92e-3 |
| **0.10** | 0.706713 | **0.707213** | **+0.000499** | 2.52e-3 |
| 0.20 | **0.697663** | 0.692119 | -0.005544 | 3.12e-8 |
| 0.40 | **0.640502** | 0.622807 | -0.017695 | 5.97e-43 |

Lengths `0.20` and `0.40` push some trajectories close to the simplex
boundary and make Fisher worse.  The `0.40` integration also has a substep
length error of `0.0127`, so it is not a trustworthy operating point.

## Numerical audit at the selected length

- 30-versus-60 distance rho: `0.999886` for p-Euclidean and `0.999845` for
  Fisher.
- Minimum probability: `0.002515`.
- Maximum substep-length error: `1.00e-12`.
- Initial direction cosine: `0.9999999999999997`.
- Fisher-versus-p-Euclidean distance rho at 60 steps: `0.999532`.

The positive rho difference at `0.10` is therefore not caused by a failed
integrator.  However, the two descriptors remain extremely similar.

## Evidence boundary

This length was selected on the same external targets used to measure the
gap.  It is an exploratory, tuned comparison.  It establishes the requested
existence result---there is a stable tested length at which Fisher rho is
higher than true p-Euclidean rho---but it is not an unbiased estimate of a
general Fisher advantage.

Aggregate evidence is in `results.json` and `metrics.csv`; all per-trajectory
diagnostics are in `diagnostics.csv`.
