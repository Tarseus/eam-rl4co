# Exploratory protocol: choose a length where Fisher beats true p-Euclidean

Date locked: 2026-07-29

## Status and objective

This is an **exploratory hyperparameter search**, not a confirmatory test.
The user explicitly removes the previous fixed length `0.03` and asks only
whether some longer response horizon makes the Fisher descriptor score above
the true fixed-coordinate probability-Euclidean descriptor.

The sole selection objective is

\[
\Delta_\rho(\ell)
=\rho_{\mathrm{FR}}(\ell)-\rho_{p\text{-E}}(\ell).
\]

Absolute rho is recorded but is not part of length selection.  A length is a
positive result whenever `Delta_rho > 0`; the selected length maximizes it.

## Locked quantities

- Candidate set: the 65-candidate external set with scratch fitness targets.
- Policy banks: scratch and epoch 135, with their equal-weight descriptor
  product as the selection bank.
- Probes: all 16 fixed TSP100 probes per bank.
- Metric raising, common training covector, common Fisher arc-length clock,
  retraction, parallel transport, and two-direction descriptor are exactly
  those in `../p_euclidean_flow/protocol.md`.
- Candidate lengths: `0.06`, `0.10`, `0.20`, and `0.40`.
- Coarse integration: 30 equal-arc substeps for both geometries at every
  length.
- Tie-breaks: larger Fisher false20 reduction, then shorter length.
- After selecting a length, repeat only that length with 60 substeps as a
  numerical-convergence audit.  The 60-step result cannot change which
  length was selected.

Nothing else may change across lengths.  In particular, there is no switch
between flow and endpoint descriptors, no bank reweighting, and no metric
selection after seeing results.

## Reporting

For every length report:

- Fisher and p-Euclidean fitness-gap rho;
- `Delta_rho`;
- false20 for both geometries;
- pairwise-distance rho and top-1 nearest-neighbor agreement between the two
  geometries;
- minimum probability and numerical errors.

Because the target data are used to select the length, the winning gap is
descriptive evidence only.  It must be labeled tuned/exploratory in any paper
and cannot be presented as an unbiased generalization estimate.
