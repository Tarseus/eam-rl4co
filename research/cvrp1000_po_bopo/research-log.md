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
| 8 | 2026-07-15 | monitor | BOPO reached step100 and improved validation32 from `60.896614` to `60.565683`; PO was healthy at step96 with finite loss/gradients. The final bounded SSH capture for PO step100 hit a banner timeout, so no repeat polling was performed. |
| 9 | 2026-07-15 | correction | User rejected capacity150 self-calibration and required a paper-anchored capacity50 experiment. Locked AGFN ICLR 2025 Table 1 as the reference: official fixed128 dataset, POMO no-aug `233.093524`, POMO x8 `192.78563`. |
| 10 | 2026-07-15 | stop | Stopped and preserved obsolete capacity150 PO PID `1686437` at step186 and BOPO PID `1685545` at step189; GPUs0/1 were released. |
| 11 | 2026-07-15 | data | Downloaded the exact AGFN official CVRP1000 tensor (SHA `b8dd0c...d8ec`) and converted all 128 ordered rows to NPZ (SHA `58982b...f295d`). Demand recovery was exact and full distance verification had max error `2.22e-16`. |
