# CVRP1000 PO/BOPO Research Log

| # | Date | Type | Summary |
|---:|---|---|---|
| 1 | 2026-07-15 | bootstrap | User requested CVRP1000 PO/BOPO training, with PO required to match or slightly beat the architecture's attainable cost, followed by larger FFSP and JSSP experiments. |
| 2 | 2026-07-15 | literature | Public CVRP1000 POMO costs are not directly portable because reported capacities vary substantially. Locked the repository-native capacity150/demand1..9 distribution and a fixed100 self-calibration protocol. |
| 3 | 2026-07-15 | protocol | PO and BOPO will share the CVRP100 PO initialization, deterministic data stream, 20 rollouts, batch1, alpha0.05, LR1e-5, and an initial 1000-step budget. Final comparison uses fixed100/100 starts/8 augment/FP32. |
| 4 | 2026-07-15 | data | Generated fixed CVRP1000 test100 seed1234 (SHA `57342809...e9ccc`) and validation32 seed4321 (SHA `5df80493...bdf8f`) with capacity150 and integer demands1..9. |
| 5 | 2026-07-15 | calibration | Evaluated the common CVRP100 PO checkpoint on canonical CVRP1000 fixed100/100-start/8-augment/FP32: mean cost `59.875156478881834`; locked the 0.5% stretch target at `59.57578069648743`. |
| 6 | 2026-07-15 | probe | One-step PO and BOPO probes both completed with finite loss/gradients and under 2 GiB reserved GPU memory. A remote-version-only constructor argument was removed before launch. |
| 7 | 2026-07-15 | launch | Started PO PID `1686437` on g53 GPU0 and BOPO PID `1685545` on g53 GPU1, each for 1000 continuation steps from the same checkpoint and deterministic stream. At the launch check PO was step 64 and BOPO step 87; no OOM, NaN, or process failure. |
