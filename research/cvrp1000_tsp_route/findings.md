# CVRP1000 TSP-route replication findings

## Current understanding

The successful TSP1000 recipe is now the confirmatory route for CVRP1000: start every branch from the same CVRP100 PO checkpoint, create a fresh Adam optimizer when the objective changes, use LR `1e-5`, and stabilize USW/ASW with a `0.90` exponential-PO anchor (`alpha=0.05`). USW uses K64 and preference alpha `0.01`; ASW uses K40, preference alpha `0.01`, and detached preference weights.

The earlier CVRP1000 runs are not a valid test of this hypothesis. The completed legacy PO and BOPO validation32 means were `128.064276` and `128.407585` at continuation step1000. Legacy USW and ASW degraded to `161.208176` and `163.000050` after 100 updates, but they started from separate CVRP100 preference checkpoints and restored their old Adam states at LR `3e-4` with K50/alpha `0.03`. Those results motivate the PO-anchor/fresh-Adam correction but cannot refute it.

Local targeted tests pass for CVRP forced replay, same-shape instance batching, and PO-anchor loss composition. The system Python emitted a known TorchRL binary warning and a post-exit Windows diagnostic, but pytest returned success with all three selected tests passing.

## Locked constraints

- Dataloader instance batch and per-instance rollout K remain separate.
- All preference pairs and weights stay within each instance.
- Step0 validation outputs must be exactly identical across PO, USW and ASW.
- The short screen consumes indices `300000000..300012799` for each matched branch; the promoted continuation starts at `300012800`, so stages never overlap.
- Every output directory is single-use. Any step reset, collision, replay mismatch, OOM, NaN, nonfinite value or traceback invalidates the affected artifact until contained.
- Canonical official128 evaluation is deferred until candidates are locked after training.

## Open question

The largest safe physical instance batch for CVRP1000 with K64/K40 on a 24 GiB GPU is not yet verified. The locked effective batch is128; physical batch will be selected by one-update probes and the remainder supplied through gradient accumulation.

