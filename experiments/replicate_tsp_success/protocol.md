# Replicate the successful TSP1000 route on CVRP1000

Status: confirmatory protocol synchronized to commit `f6bff1bd0` before target execution.

## Question and common source

Can the final successful TSP1000 PO-anchor route make both a USW variant and an ASW variant significantly outperform the same strong PO source on CVRP1000?

- Capacity50 CVRP1000 with uniform coordinates and integer demands1..9.
- Common source: g53 `logs/cvrp1000_agfn_capacity50/po_1000steps/best.ckpt`, SHA256 `94cbb5e1c7b6391f31ae113c1beae81491cdfffad17199d1db0ac51205a9fd26`.
- The source fixed32/100-start/augment1 mean is `128.06427645683289`.
- USW and ASW both load only these model weights and create fresh Adam optimizers because the objective changes. They must have exactly identical step0 per-instance validation costs.

## Locked variants

- USW-PO-anchor: dense uniform all-pairs, K64, preference alpha `0.01`, weights not detached, PO anchor weight `0.90`, exponential-PO alpha `0.05`.
- ASW-PO-anchor: adaptive gap-square weighting, K40, preference alpha `0.01`, detached preference weights, PO anchor weight `0.90`, exponential-PO alpha `0.05`.
- Pair artifacts are respectively locked at SHA256 `022bf1a3e81b7a01c346153cf1e338e93a2a9c50e5f52b288c62f4b8cf084304` and `9652dddda2eeca7879514c57126a9601bbee12974ba5905ee5ce1d8ee83f6471`.

## Shared controls and hardware fallback

- Seed1234, dynamic data start index `300000000`, BF16, LR `1e-5`, weight decay `1e-6`.
- Effective instance batch128. Probe physical batch128 first; if it cannot fit a24 GiB GPU, use the largest common safe physical batch and gradient accumulation to preserve effective batch128.
- Rollout K remains per-instance and separate from the dataloader batch dimension.
- Candidate pools, weights and pair construction remain strictly within each instance; per-instance losses are averaged across instances.
- Validation uses `data/vrp/agfn_vrp1000_capacity50_val_seed4321.npz` (SHA `8de3059f...d35e94`), fixed32,100 starts, augmentation1, every100 optimizer steps.
- Probe instances use a separate index range and never overlap the screen or promoted continuation.
- Every launch gets a unique output directory. OOM, NaN/nonfinite, traceback, replay mismatch above `1e-5`, optimizer-step reset or output collision is immediately preserved and contained.

## Screen and promotion

- First run a one-update integrity/VRAM probe for each variant.
- Then run both variants for200 updates on matched indices `300000000..300025599`.
- Promote only when step200 validation improves over the shared step0 validation.
- Restore the exact step200 `last.ckpt`, including its fresh-Adam state, for800 more updates to absolute continuation step1000.
- Promoted data starts at `300025600`, so no screen instance is replayed.
- Candidate locking uses held-out validation only.

## Final evaluation and success

- Evaluate the source PO and locked step1000 variants on official AGFN capacity50 fixed128 (SHA `58982b91...a4f295d`), exact indices0..127,100 starts,8 augmentations, FP32.
- Report every aligned per-instance cost and mean.
- Use a100,000-sample paired bootstrap over candidate-minus-PO costs with seed1234.
- Each variant must have a lower mean than PO and a wholly negative paired-bootstrap95% confidence interval; both must pass for overall success.

