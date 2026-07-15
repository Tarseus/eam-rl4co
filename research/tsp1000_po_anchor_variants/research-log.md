# TSP1000 PO-Anchored USW/ASW Research Log

| Date | Type | Summary |
|---|---|---|
| 2026-07-15 | scope reset | User stopped the prior scheduled add2000 work and authorized unrestricted USW/ASW variants initialized from the verified PO optimizer-step5075 checkpoint. Old g48/g52 training jobs were terminated cleanly and their outputs preserved. |
| 2026-07-15 | implementation | Added a per-instance convex PO anchor for free-loss methods: `(1-w)*preference + w*PO_exponential`; forced replay remains enabled. Targeted tests passed. |
| 2026-07-15 | smoke | Remote one-step smoke tests on g53 passed for both variants. USW: replay_error 0, loss -0.294850, grad_norm 8.9456. ASW: replay_error 0, loss -0.101911, grad_norm 11.1655. No OOM, NaN, or traceback. |
| 2026-07-15 | launch | Started the initial 200-step screen from the same PO-5075 weights with a fresh optimizer: USW-PO-anchor on g53 GPU4 PID1713672 and ASW-PO-anchor-detach on GPU5 PID1713669. Both use anchor weight0.90, LR1e-5, batch128, and validation32 every100 steps. Initial validation32 mean is25.838266 for both. |
| 2026-07-15 | live check | Both screens are healthy. USW reached step6 at21.27s/step (6.02 instances/s,385.1 trajectories/s), loss-0.17925, grad_norm0.7822, replay_error0, peak allocated/reserved18.81/19.68GiB, ETA69min. ASW reached step8 at16.25s/step (7.88 instances/s,315.1 trajectories/s), loss-0.20734, grad_norm0.9382, replay_error0, peak allocated/reserved15.79/23.21GiB, ETA52min. No OOM, NaN, or traceback; wait for next tick. |
