# Official SLIM Baseline Recovery Log

| Time | Event | Evidence / decision |
|---|---|---|
| 2026-07-18 16:14 CST | Protocol lock | Historical local SLL is excluded. True SLIM is fixed to per-instance best sampled pseudo-label cross-entropy. Primary eight settings precede the four large-scale extensions. Only idle g51 GPUs 5 and 7 may be used. |
| 2026-07-18 16:28 CST | Smoke launch audit | The first two queues failed before model construction because Hydra requires `+trainer.limit_*` for new structured keys. A scheduling retry then exposed an incompatible disabled-progress-bar override before training. Both failures are preserved and excluded. |
| 2026-07-18 16:32 CST | Backbone smokes | Fresh-root TSP50, CVRP50, and FFSP50 one-batch smokes completed with finite SLIM losses 0.866, 0.532, and 1.355. JSSP10x10 completed its train/validation computation with finite loss 2.151, then failed only because the generated validation split has no reference gap for the `val/gap` checkpoint monitor; the monitor is changed to mean `val/reward`. The routing/FFSP smoke configs inherited `test=true` and performed an unintended post-smoke test look. Those outputs are ignored, cannot select any setting, and full configurations are now locked to `test=false`. |
