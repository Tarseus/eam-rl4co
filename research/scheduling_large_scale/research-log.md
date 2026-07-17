# Scheduling Large-Scale Research Log

Append-only decision and experiment timeline.

| # | Date | Type | Summary |
|---|---|---|---|
| 1 | 2026-07-17 | bootstrap | Locked FFSP1000 as the FFSP main scale and JSSP50x20 as the train/main scale with JSSP100x20 as stress evaluation. Existing FFSP100 and JSSP15x15 checkpoint parameterizations are size-independent. |
| 2 | 2026-07-17 | sanity | FFSP100 USW weights loaded into an FFSP1000 environment and completed encoding for input shape (1,1000,12). JSSP15x15 USW completed full B=1 greedy rollouts on random 50x20 and 100x20 instances. All five method checkpoints have identical core parameter keys/shapes within each problem. |
| 3 | 2026-07-17 | protocol | Locked identical common policy weights, fresh Adam, matched data/order/budget, instance-local pools and pairs, staged update gates, final paired inference, and 1% FFSP / 2% JSSP non-inferiority bounds. Original USW/ASW are tested before any explicitly labeled variant. |
| 4 | 2026-07-17 | continuity | Created the 20-minute `autoresearch-ffsp1000-and-jssp50x20` heartbeat. Initial g51 check found GPU1 idle; other GPUs were occupied. No remote job was launched yet. |
