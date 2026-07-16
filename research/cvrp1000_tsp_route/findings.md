# CVRP1000 TSP-Route Findings

## Current conclusion

The previous CVRP1000 USW/ASW numbers cannot test the TSP1000 success hypothesis because initialization, optimizer state, learning rate, rollout count, and effective batch were all confounded. The new experiment removes those confounders.

The completed legacy held-out means were PO `128.064276`, BOPO `128.407585`, USW `161.208176`, and ASW `163.000050`. Only the PO result is reused: its locked step-1000 best checkpoint becomes the common model-weight source. The legacy preference runs remain negative context only.

The implementation integrity gate currently passes locally: forced replay is exact, the loss is aggregated per instance, candidate pairs remain instance-local, and USW/ASW begin from identical strong-PO weights. No CVRP1000 performance conclusion is made until the locked remote screen produces held-out results.

The first remote launch attempt used the superseded CVRP100-source draft. Once concurrent commit `f6bff1bd0` exposed the final strong-PO protocol, those three mismatched probe PIDs were contained and their output directories retained as excluded evidence. The correct physical-batch128 probes now run only USW and ASW from the locked strong PO-step1000 source on g53 GPUs4/5. At the sole launch verification each had exactly one Python process and only its config file, so no memory or integrity result is claimed this tick.

## Locked constraints

- Instance batch and per-instance rollout K are separate axes.
- Preference candidate pools, filtering, and pairing never cross instances.
- Step-0 per-instance validation costs must be exactly identical across variants.
- Screen indices begin at `300000000`; promoted continuation begins at `300025600`.
- Every output directory is single-use; any optimizer-step reset or collision invalidates it.
- Official fixed128 evaluation is deferred until a candidate is locked.
