# RFPS minimality research log

## 2026-07-28 — Protocol lock

The study is reframed from adding geometric machinery to identifying the minimal
sufficient behavior descriptor. Metrics, baselines, datasets, and deletion rules
were fixed before running the new ablations. Existing curvature results are treated
as prior evidence; newly introduced baselines remain unobserved at protocol lock.

## 2026-07-28 — Inner-loop ablations

- The first two flow checkpoints reproduced full three-point curvature.
- Three coarse field evaluations reproduced the 30-step flow.
- Raw two-point field differences failed; unit direction changes reproduced curvature.
- Per-probe normalized initial directions beat curvature on global correlation but
  produced dangerous tail collisions under sequential screening.
- Concatenating the initial and perturbed unit directions retained the first-order
  correlation gain while matching curvature tail risk.
- A one-step empirical-logit Euclidean perturbation matched the Fisher step across
  two probe seeds, scratch and warm labels, and the external 65-candidate set.
- A single checkpoint probe bank predicted both scratch and warm labels better than
  maintaining separate descriptor anchors.

## 2026-07-28 — Outer-loop decision

Replace the three-point Fisher--Rao curvature system with one two-point directional
response vector. Keep only one fixed checkpoint rollout bank, two loss derivatives,
one empirical-logit step of size 0.03, soft screening, and a nonzero audit rate.
Remove the geometric integration stack and correct the previous overbroad claim that
second-order curvature generally outperforms first-order descriptors.

## 2026-07-28 — Paper method rewrite

Replaced RFPS v3 with Response-Field Program-pair Search v4. The paper now defines
one checkpoint bank, a projected unit direction, one self-induced Euclidean
log-weight perturbation of size 0.03, and the concatenated two-point direction
descriptor. It reports all minimality controls, narrows the related-work boundary to
gradient-equivalence screening, archives the v3 source, and compiles to an 11-page
PDF without unresolved references or layout warnings.
