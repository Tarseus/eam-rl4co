# CVRP1000 PO/BOPO Research Log

| # | Date | Type | Summary |
|---:|---|---|---|
| 1 | 2026-07-15 | bootstrap | User requested CVRP1000 PO/BOPO training, with PO required to match or slightly beat the architecture's attainable cost, followed by larger FFSP and JSSP experiments. |
| 2 | 2026-07-15 | literature | Public CVRP1000 POMO costs are not directly portable because reported capacities vary substantially. Locked the repository-native capacity150/demand1..9 distribution and a fixed100 self-calibration protocol. |
| 3 | 2026-07-15 | protocol | PO and BOPO will share the CVRP100 PO initialization, deterministic data stream, 20 rollouts, batch1, alpha0.05, LR1e-5, and an initial 1000-step budget. Final comparison uses fixed100/100 starts/8 augment/FP32. |
| 4 | 2026-07-15 | data | Generated fixed CVRP1000 test100 seed1234 (SHA `57342809...e9ccc`) and validation32 seed4321 (SHA `5df80493...bdf8f`) with capacity150 and integer demands1..9. |
