# Why Weighting Is Harder to Generalize

核心说法不要写成“loss 做不到 weighting”。更稳的写法是：

> Loss search learns the semantic kernel of an already materialized pairwise comparison, whereas weighting search learns a pool-conditioned measure over which comparisons receive training mass. This measure is calibrated on the empirical geometry of the source-scale candidate pool, so it is easier to overfit to scale-specific distributions of gaps, margins, ranks, and clamp boundaries.

## What the Figure Shows

Figure: `weighting_formula_scale_shift.png`.

The left column plots `log10(pre-clamp score / upper clamp)`. The vertical line at 0 marks the upper clamp boundary. Values to the right would be clipped to the maximum weight. The middle column plots the final weight after all clamps, normalized to each formula's clamp interval. The right column shows how much pair mass lies at the lower clamp, inside the active interval, or at the upper clamp.

This directly visualizes the induced pair measure. Even when the preference semantics are unchanged, the formula may assign a different distribution of mass to the pair table after the problem scale changes.

## Formula-Level Observations

TSP weighting uses

```text
weight = clamp((gap / instance_obj_mad) * (|logp_w-logp_l| / instance_log_prob_std) * instance_regret_mean, 0.2, 2.5)
```

In this replay, TSP100 to TSP50 is relatively stable: the high-clamp share changes from about 58.3% to 57.2%. This is useful as a control. The argument should not claim that every weighting formula catastrophically shifts under every scale transfer. Instead, TSP shows the other side of the problem: the rule works partly because a source-scale clamp calibration turns a large part of the formula into a saturated measure.

CVRP weighting uses raw absolute objective gap:

```text
weight = clamp(cost_l - cost_w, 0.3, 5.0)
```

This is the clearest example of poor transfer calibration. From CVRP100 to CVRP50, the pre-clamp median shifts from -0.49 to -0.74 on the log10 scale relative to the upper clamp; the high-clamp share collapses from 9.4% to 0.4%; the low-clamp share increases from 10.4% to 18.2%; and the final-weight KS distance is 0.24. The semantic pair relation is the same, but the measure over pairs is not: the same formula allocates much less mass to large-gap comparisons at the smaller scale.

FFSP weighting uses

```text
weight = clamp(clamp(sigmoid(5 * rank_span) / margin_norm^0.6 * 1[gap/MAD > 0.12], 0.15, 2.5) / rank_span, 0.15, 2.5)
```

This replay does not show a large raw distribution shift; instead it shows saturation. About half of the pairs are at the upper clamp, and the rest sit inside the active interval. That means the rule is dominated by a source-calibrated threshold/clamp window. Small changes in which pairs leave the upper clamp can change the effective signal, even if aggregate weight histograms look similar. For FFSP-specific discussion, also use `figures/scale_transfer_replay_diagnosis/ffsp_failure_probe`, which probes how clamp release affects rank-signal separation.

JSSP weighting uses

```text
weight = clamp((gap/MAD) * regret_mean^0.9 * sigmoid(3.5 * rank_diff) + 0.05 * regret_mean^0.75, 0.1, 3.0)
```

The JSSP probe shows a mild shift toward larger weights from 10x10 to 15x15, with the high-clamp share increasing from 38.4% to 43.0%. Because the local TA source data were unavailable, the script used the repository's random shape-probe fallback; treat this as qualitative formula stress evidence, not as a formal benchmark result.

## Paper-Ready Wording

The transfer weakness of weighting is therefore not a failure of pairwise preference semantics. It is a failure mode of the learned sampling measure. The best weighting formulas do not merely multiply the loss by a harmless scalar; they combine empirical pool statistics such as objective dispersion, log-probability dispersion, rank span, regret scale, and thresholded gap regions, then pass the result through hard clamps. These operations are calibrated at the source problem size. When the candidate-pool geometry changes, the same formula can move different pairs into the lower clamp, active interval, or upper clamp, thereby changing which comparisons dominate training. Loss-only objectives keep the comparison kernel more invariant; weighting adds an extra distribution-sensitive layer that decides which parts of the pair manifold count.

Recommended phrasing:

> Weighting search is a search over pair measures, not merely over scalar multipliers. Its gains are source-distribution gains: it can identify which pair regions are useful under the search-scale candidate-pool geometry. But because that geometry changes with problem size, the learned measure is less portable than the loss kernel itself.
