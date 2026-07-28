# RFPS adaptive-step research log

## 2026-07-28 — Protocol lock

The experiment separates intrinsic trust-radius calibration from candidate-specific
adaptation. It preregisters KL, reverse-KL/entropy, ESS, max-log-ratio, and local
turning-radius rules before running them on the frozen RFPS candidate sets.

## 2026-07-28 — Adaptive-radius results

Exact forward KL, reverse KL, and ESS constraints all solved back to eta 0.03000 with
negligible candidate variation and identical screening metrics. Max-log-ratio and
turning-radius rules created substantial step variation but no stable rho, NN, or
false-skip gain. Candidate-specific adaptation on the free empirical simplex is
therefore rejected.

## 2026-07-28 — Geometric reformulation

Čencov uniqueness changes the role of geometry: Fisher should define the unit tangent,
exponential step, and comparison of d0/d1, while the scalar arc length must be supplied
by training because the invariant metric is only unique up to a constant. Existing
Fisher-pair ablations show a small consistent rho gain over the Euclidean pair at no
additional derivative cost. The selected next method uses this lightweight invariant
pair and calibrates one global arc length from a stateless real optimizer update.
