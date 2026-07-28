# H2 two-point policy-flow result

The locked two-point policy-Fisher hypothesis failed both external-set
conditions.

| descriptor | external scratch rho | external false20 | matched joint rho | matched false20 |
|---|---:|---:|---:|---:|
| raw coefficient | 0.7325 | **2.01%** | **0.8585** | **1.28%** |
| one-point policy Fisher | **0.7546** | 3.80% | 0.8504 | 2.55% |
| two-point policy Fisher | 0.7326 | 7.09% | 0.8565 | 5.73% |

The preregistered external requirements were rho at least 0.7495 and
false20 below 3.03%. Two-point Fisher satisfies neither.

Every nonstationary step has Fisher--Rao length exactly 0.03. The median
cosine between the first and transported second directions is approximately
0.995, while the median coefficient change is below 0.5%. Despite this small
global change, local neighbor order is sensitive enough that false skips
increase. The epoch-135 component accounts for most of the external
correlation loss.

Decision: stop the two-point policy-flow branch. Do not tune length, clip
Fisher weights, concatenate channels, or learn bank weights on these results.
Retain H1 only as evidence that the actual policy manifold exposes a distinct
first-order global behavior signal.
