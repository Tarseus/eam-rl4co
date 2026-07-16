# CVRP1000 TSP-Route Research Log

| Date | Stage | Observation / decision |
|---|---|---|
| 2026-07-16 | bootstrap | Initialized a dedicated project rather than treating the incompatible legacy `cvrp1000_po_bopo` experiment as the target replication. |
| 2026-07-16 | merged remote snapshot | On g53, legacy PO and BOPO completed at continuation step1000 with validation32 means `128.064276` and `128.407585`; legacy USW/ASW completed at step100 with `161.208176` and `163.000050`. GPUs3-5 were idle. |
| 2026-07-16 | diagnosis | Earlier CVRP1000 USW/ASW short runs were not matched comparisons: they loaded independent method-specific CVRP100 checkpoints and restored their old optimizer state at LR3e-4 with batch1. This does not reproduce the successful TSP1000 route. |
| 2026-07-16 | implementation smoke | Added CVRP forced replay with activation checkpointing and instance-batched per-instance loss aggregation. Small-CVRP tests passed for exact reward/log-likelihood replay, finite backward, and replay error0. USW and ASW loaded from the same PO checkpoint were tensor-for-tensor identical before the objective-specific update. |
| 2026-07-16 | pre-execution protocol revision | The earlier preflight draft used the CVRP100 PO checkpoint and a 100-step screen. Before any target run launched, revised this to the already-trained strong CVRP1000 PO step1000 source and the TSP route's exact 200-step screen. This makes the source analogous to TSP PO-5075 and the final comparison more stringent. |
| 2026-07-16 | protocol lock | Locked the final TSP route on CVRP1000: strong CVRP1000 PO step1000 source, fresh Adam, LR1e-5, effective batch128, anchor0.90, USW K64/non-detach, ASW K40/detach, screen200 then resume to total1000, official fixed128 paired bootstrap. |
