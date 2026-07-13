# Checkpoint-Rollout Delta-Cost View of Weighting Transfer

This is the figure to use for explaining why weighting is harder to generalize.

Main figure: `delta_cost_rawmargin_vs_weight_4x2_ckpt.png`.

CVRP/FFSP cost-free contrast: `rank_span_vs_weight_cvrp_ffsp_ckpt.png`.

The x-axis is no longer an abstract formula pressure. It is the actual pair-level cost gap produced by neural-network rollouts:

```text
Delta cost = cost(loser trajectory) - cost(winner trajectory)
```

The y-axis is the weight assigned by the discovered weighting builder to that same pair. Each curve is sorted by Delta cost and binned by Delta-cost quantiles; the faint points are sampled rollout pairs.

For separating CVRP and FFSP, use normalized rank span instead of cost:

```text
rank span = (rank(loser) - rank(winner)) / (K - 1)
```

This axis is computed identically for both problems after sorting each checkpoint-generated candidate pool by objective value. It removes the raw cost-unit issue and asks where the weighting rule places mass along the candidate-pool order.

Normalized policy margin is useful as a backup diagnostic:

```text
margin_norm = |logp_w - logp_l| / std_instance(logp)
```

This axis is not shown in the main figure because the two retained columns are enough. It is still exported in `margin_norm_weight_binned.csv`. CVRP is almost flat on this axis because its builder is a cost-gap rule. FFSP has a strong negative slope because the final builder explicitly contains `margin_norm^-0.6`.

The right column of the 4x2 figure uses raw policy margin:

```text
raw_margin = |logp_w - logp_l|
```

This is the axis that makes FFSP100 and FFSP50 visibly different. Normalized margin hides the scale change by construction; raw margin reveals that the checkpoint-generated policy-margin distribution contracts strongly at FFSP50. In the current replay, FFSP raw-margin median changes from about 14.59 at FFSP100 to 7.63 at FFSP50, and the 90th percentile changes from about 36.46 to 18.68. The weighting curve remains a decreasing function of raw margin, so the rule is acting on a compressed policy-margin range at the smaller scale.

## What This View Shows

Loss search fixes the semantic kernel: winner should be preferred over loser. Weighting search changes the empirical measure over the checkpoint-generated pair table. The Delta-cost plot shows this directly: for the same kind of pairwise semantic relation, the weighting formula induces different curves over the actual trajectory gaps produced at different scales.

## Problem-Specific Readout

TSP is mostly saturated. Under the final TSP weighting formula, the median weight is already at the upper clamp on both TSP100 and TSP50. The Delta-cost curve climbs quickly and then becomes flat. This means TSP weighting is not learning a smoothly transferable importance rule over all gaps; much of the rule behaves like a thresholded/saturated selection of sufficiently separated pairs. The correlation between Delta cost and weight drops from about 0.57 at TSP100 to 0.29 at TSP50 because many medium and large gaps are already indistinguishable after clamping.

CVRP is the cleanest scale-shift example. The final CVRP weighting formula is essentially an absolute-gap rule with a lower clamp. On checkpoint rollouts, the Delta-cost distribution shifts downward from CVRP100 to CVRP50: median Delta cost changes from about 0.28 to 0.20, and the 90th percentile from about 0.70 to 0.56. Since the formula uses raw Delta cost, this directly changes the induced weight curve. Many more pairs remain at the lower clamp on the smaller scale. This is the easiest case to describe as failure of scale calibration.

FFSP is qualitatively different and very useful for the paper. Larger Delta cost does not monotonically mean larger weight. The discovered FFSP rule combines a tie-zone, normalized margin, and an inverse rank-span factor. In the ckpt rollout plot, the weight is high for small-to-moderate Delta-cost pairs and decreases for larger Delta-cost pairs. This is exactly the point: weighting is not just "bigger objective gap means stronger supervision." It is a learned sampling measure that prefers a particular region of the trajectory-pair manifold. That region can move when the policy or problem size changes.

The clearest same-axis contrast between CVRP and FFSP is normalized policy margin. On checkpoint rollouts, CVRP has almost no margin/weight correlation: about 0.05 at CVRP100 and 0.03 in the CVRP50 replay. FFSP has a strong negative correlation: about -0.63 at both FFSP100 and FFSP50. Thus CVRP and FFSP differ not only in cost scale, but in what kind of pair the weighting rule selects. CVRP remains a gap-driven measure; FFSP is a low-margin-pair measure.

For FFSP scale transfer specifically, use raw policy margin. FFSP50 is not separated by the final weight histogram, and it is not separated by normalized margin; it is separated by the raw policy-margin scale. This is consistent with the performance table: FFSP50 is worse than FFSP100, but the final weighting curve can look deceptively similar because the formula normalizes margin and then clips many low-margin pairs. The raw-margin panel shows that the underlying checkpoint signal has changed before normalization and clipping.

JSSP shows raw-scale expansion. From 10x10 to 15x15, the trajectory Delta-cost distribution shifts strongly upward: median Delta cost changes from about 17 to about 40, and the 90th percentile from about 49 to about 103. The weighting curve therefore reaches the upper clamp over a wider portion of the pair table. This supports the argument that raw or partially normalized gap-based weighting is tied to the scale of generated schedules.

## Paper-Ready Claim

The key sentence:

> Weighting is harder to transfer because it is a learned measure over the checkpoint-generated trajectory pairs. Its input is not only the preference label, but the empirical geometry of the rollout table: which Delta-cost values appear, how they correlate with rank span and policy margin, and where the formula's clamps become active.

Recommended paragraph:

> To make the transfer failure concrete, we plot each discovered weighting rule against the actual Delta cost of pairs generated by trained checkpoints. This view exposes weighting as a scale-calibrated pair measure. In CVRP, moving from CVRP100 to CVRP50 shifts the Delta-cost distribution downward, so the same raw-gap weighting rule assigns more pairs to the lower-clamp region. In TSP, the rule rapidly saturates at the upper clamp, making medium and large gaps indistinguishable. FFSP reveals a different failure mode: because the rule includes rank-span and margin terms, larger objective gaps can receive lower weight, so the rule selects a specific band of rollout pairs rather than a scale-invariant preference strength. JSSP expands the raw Delta-cost range from 10x10 to 15x15 and pushes more pairs toward the upper clamp. These patterns explain why weighting can improve a source scale while generalizing worse than the loss kernel: it learns where to allocate mass in the source checkpoint's pair distribution, not a universally stable semantic comparison.

If the reviewer asks why CVRP and FFSP are fundamentally different, use the raw-margin sentence:

> The distinction becomes clearest when the weight curve is plotted against two actual checkpoint-generated pair coordinates. Along Delta cost, CVRP behaves like a gap-weighting rule. Along raw policy margin, FFSP reveals the scale effect: FFSP50 has a compressed margin range, so the same low-margin selector operates on a different signal scale before normalization and clipping.
