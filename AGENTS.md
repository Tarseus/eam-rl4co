# JSSP batching project rules

Goal:
Upgrade the current single-instance JSSP/MGL training pipeline into a bucketed batching pipeline.

Final target:
- Support batching for same-shape JSSP instances
- Support shape buckets at least for 10x10, 15x15, 20x20
- Keep per-instance rollout count B semantics unchanged
- RL, PO, and BOPO must all aggregate losses per instance first, then average across instances
- Pair construction for PO and BOPO must stay strictly within each instance

Non-goals:
- Do not implement mixed-shape padding in the first implementation
- Do not silently mix different shapes in one batch
- Do not change unrelated model logic
- Do not rename major config structures unless necessary

Implementation preference:
- First make 10x10 same-shape batching work
- Then generalize to bucket-by-shape
- Use minimal invasive changes
- Add explicit runtime checks for unsupported shapes or mixed-shape batches
- Add smoke tests for RL, PO, and BOPO

Important invariants:
- dataloader batch dimension and rollout dimension must remain semantically separate
- candidate pools must remain instance-local
- BOPO filtering and anchored pairing must remain instance-local
- final loss = mean over instances of per-instance loss