# FFSP Clip Signature

Main figure: `ffsp_clip_signature_ckpt.png`.

This figure follows the user's clue that FFSP transfer degradation may be related to clipping. The evidence is more subtle than a large source-vs-transfer change in the marginal weight distribution.

## What Is Actually Happening

For the discovered FFSP builder, the effective pre-final score is

```text
pre_final =
  clamp(sigmoid(5 * rank_span) / margin_norm^0.6 * 1[gap/MAD > 0.12], 0.15, 2.5)
  / rank_span

weight = clamp(pre_final, 0.15, 2.5)
```

where

```text
margin_norm = |logp_w - logp_l| / std_instance(logp)
```

The final clamp hides a large part of the rule's dynamic range. On checkpoint rollouts, about half of all pairs are upper-clipped:

- FFSP100 ckpt @ FFSP100: 54.6% upper-clipped.
- FFSP100 ckpt @ FFSP50: 54.5% upper-clipped.
- FFSP50 ckpt @ FFSP50: 53.4% upper-clipped.

So the important statement should not be "FFSP clip share changes dramatically when scale changes." It does not, at least on these final checkpoint rollouts. The defensible statement is:

> FFSP's weighting rule is strongly saturated. Transfer risk comes from relying on a narrow low-margin/local-rank region whose pre-clamp variation is mostly hidden by the upper clamp, rather than from a large visible shift in the final weight histogram.

## Why This Makes FFSP Different

The lower-left panel shows that clipping is concentrated on low-margin pairs. In the lowest margin bin, essentially all pairs are upper-clipped. At high margin, only about 11-12% of pairs are upper-clipped.

This is the FFSP-specific behavior missing from the raw Delta-cost plot. FFSP is not a gap-strength weighting rule. It is closer to:

> select ambiguous low-margin pairs, especially local rank-neighbor comparisons, then clip many of them to the same maximum weight.

This explains why FFSP can look visually similar across FFSP100 and FFSP50 in final-weight plots: final weights are already saturated. The mechanism is still different from CVRP, but the scale-transfer difference is hidden behind the clamp.

## Paper Wording

Use a careful version:

> FFSP does not exhibit a large source-to-target shift in the marginal final-weight curve. Instead, the discovered rule is already highly saturated at the source scale: roughly half of the pairs exceed the upper clamp, and clipping is concentrated on low-margin pairs. Thus the final weight distribution can look stable even though the rule depends on a narrow region of the checkpoint-generated pair manifold. This is a different transfer risk from CVRP: CVRP shows raw-gap scale drift, whereas FFSP shows saturation of a low-margin pair selector.
