# CVRP1000 TSP-route replication research log

| Date | Type | Summary |
|---|---|---|
| 2026-07-16 | bootstrap | The four automation-requested research files were absent. Initialized a dedicated project rather than silently treating the incompatible legacy `cvrp1000_po_bopo` experiment as the target replication. |
| 2026-07-16 | merged remote snapshot | On locked host g53, legacy PO and BOPO were complete at continuation step1000 with validation32 means `128.064276` and `128.407585`. Legacy USW/ASW were complete at step100 with means `161.208176` and `163.000050`. No matching process remained and no OOM/NaN/nonfinite/Traceback/replay mismatch was found. GPUs3-5 were idle. |
| 2026-07-16 | code verification | Targeted local tests for CVRP forced replay, batched memory-efficient preference training and PO-anchor composition all passed (3 selected tests). |
| 2026-07-16 | protocol | Locked a 100-step same-checkpoint/fresh-Adam screen followed by optimizer-resumed continuation to total step1000, with effective batch128, USW K64/alpha0.01, ASW K40/alpha0.01/detach, and PO anchor0.90/alpha0.05. |

