# Fixed-chart actual-update alignment protocol

## Question

When the empirical distribution is represented in one fixed coordinate system,
does the Fisher two-point descriptor predict the effect of a real neural-policy
update better than an equally expensive Euclidean two-point descriptor?

This experiment does **not** use coordinate transformations. It therefore does
not treat reparameterization invariance as evidence by itself. The only
practical question is alignment with an actual parameter update.

## Frozen experimental objects

- Policy: `tsp100_epoch_135.ckpt`, loaded into the isolated PO4COPs-compatible
  TSP100 POMO policy used by `rfps_feature_pilot`.
- Candidate sets:
  - `matched`: the 40 high-fidelity pairs in the 2026-03-17 run;
  - `external`: the 65 high-fidelity pairs in the 2026-04-08 run.
- Both sources contain only `g_ref`, the same dense all-pairs preference
  builder. The experiment therefore tests variation in `f` with `g` fixed and
  must not be presented as evidence about varying builders.
- To keep real updates tractable on CPU, select 16 candidates from each source
  by sorting on `(score, key)` and taking the deterministic indices nearest to
  16 equally spaced rank quantiles. Selection is fixed before any alignment
  result is computed.
- Optimizer: a fresh Adam state for every candidate, matching warm-start
  evaluation, with learning rate `3e-4`, weight decay `1e-6`, and PyTorch
  defaults for the remaining hyperparameters.
- One optimizer step is the primary target. It is an on-policy microstep from
  the shared checkpoint, not a claim about an entire training run.

## Common random numbers and held-out response

For each candidate, the policy starts from identical checkpoint parameters.

1. Generate one fixed on-policy training bank from the checkpoint with a common
   random seed. The sampled actions, objectives, and computational graph of
   their log-likelihoods are shared by every candidate.
2. Construct the candidate loss using the actual `g_ref` all-pairs builder and
   the compiled candidate `f`.
3. Take one real Adam parameter step.
4. On a disjoint fixed instance bank, teacher-force actions sampled once from
   the unmodified checkpoint. Record

   `Delta_C = log pi_{theta_C'}(tau) - log pi_theta(tau)`.

The held-out actions are never used to form the training loss. Teacher forcing
removes action-resampling noise while still evaluating the updated network.
Within each held-out instance, center `Delta_C` over starts. Concatenate the
centered blocks and normalize the resulting vector to unit Euclidean norm. This
fixed-chart vector is the primary actual-update response. Its pairwise
Euclidean distance matrix is `D_actual`.

The primary run uses two TSP100 training instances, two disjoint held-out
instances, and 100 POMO starts per instance. A smaller configuration may be
used only for implementation smoke tests and may not enter the reported
conclusion.

## Compared descriptors

All methods use the same training bank, uniform `q0`, fixed log-weight
coordinates, four-digit arc/step length `ell = 0.03`, and the same probe
aggregation. Each per-probe direction is unit normalized before concatenation.

1. `one_point`: initial Fisher-unit loss-induced direction `d0`.
2. `euclidean_two_point`: concatenate `d0` with the unit loss-induced
   direction at one fixed-coordinate Euclidean/logit step.
3. `fisher_two_point`: concatenate `d0` with the second Fisher-unit direction
   after a Fisher--Rao geodesic step, parallel transported to the initial
   tangent space.

The Euclidean and Fisher variants each evaluate the candidate field twice.
No chart rescaling or alternative encoding is used in the primary comparison.

## Metrics

For each source and for the pooled candidates, report:

1. Spearman correlation between the upper triangles of the descriptor distance
   matrix and `D_actual` (primary global metric).
2. Mean actual-response distance to each descriptor's nearest non-self
   neighbor, divided by the mean distance to a uniformly random non-self
   neighbor (primary local metric; lower is better).
3. Top-1 and top-5 neighbor overlap with neighbors under `D_actual`.
4. The paired per-query difference in actual nearest-neighbor error between
   Fisher and Euclidean, with a candidate bootstrap 95% interval.

Also report update loss, gradient norm, parameter-step norm, and actual-response
norm as sanity checks. Non-finite or zero-response candidates are failures and
must be listed, not silently imputed.

## Decision rule

The Fisher geometry is promoted as a practically necessary part of the method
only if:

- it has higher distance Spearman correlation and lower normalized
  nearest-neighbor error than Euclidean on both candidate sources; and
- the pooled bootstrap 95% interval for
  `NN_error(Fisher) - NN_error(Euclidean)` lies below zero.

If this rule is not met, the result supports the two-point joint response but
not a claim that Fisher geometry is closer to the realized training update.
The paper must then use the simpler fixed-coordinate two-point descriptor as
the core method, with Fisher invariance described only as an optional
coordinate-robust implementation property.

## Integrity rules

- Commit this protocol before computing any actual-update alignment result.
- Keep candidate selection, seeds, optimizer, `ell`, and primary metrics fixed.
- Smoke tests may change only runtime parameters and are excluded from results.
- Record exceptions and exact software/configuration metadata.
- Do not use high-fidelity scores to tune the descriptors or the conclusion.
