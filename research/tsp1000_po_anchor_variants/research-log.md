# TSP1000 PO-Anchored USW/ASW Research Log

| Date | Type | Summary |
|---|---|---|
| 2026-07-15 | scope reset | User stopped the prior scheduled add2000 work and authorized unrestricted USW/ASW variants initialized from the verified PO optimizer-step5075 checkpoint. Old g48/g52 training jobs were terminated cleanly and their outputs preserved. |
| 2026-07-15 | implementation | Added a per-instance convex PO anchor for free-loss methods: `(1-w)*preference + w*PO_exponential`; forced replay remains enabled. Targeted tests passed. |

