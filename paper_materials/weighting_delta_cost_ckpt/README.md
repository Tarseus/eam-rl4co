# Delta-Cost vs Weight on Checkpoint Rollouts

This analysis loads downloaded weighting checkpoints, rolls out candidate trajectories, applies the discovered weighting builder, and plots pair weight against the actual pair-level cost gap produced by the checkpoint policy.

Main figure: `delta_cost_vs_weight_ckpt.png`.

Combined 4x2 figure: `delta_cost_rawmargin_vs_weight_4x2_ckpt.png`.

CVRP-vs-FFSP cost-free comparison: `rank_span_vs_weight_cvrp_ffsp_ckpt.png`.

The columns are raw pair-level `delta_cost = cost_loser - cost_winner` and raw policy margin `abs(logp_w - logp_l)`. The y-axis is the discovered builder's pair weight. Lines are quantile-binned means; shaded regions are interquartile bands; faint points are sampled pairs. Normalized-margin data are still exported as `margin_norm_weight_binned.csv`, but are not used in the main figure.
