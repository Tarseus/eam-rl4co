# CVRP1000 TSP-Success-Route Replication Protocol

Status: CONFIRMATORY SCREENING protocol locked before remote execution.

## Question

Can the final successful TSP1000 training route make both a USW variant and an ASW variant significantly outperform the same strong PO source checkpoint on CVRP1000?

## Common source

- Problem: uniform CVRP1000 with integer demand 1..9 and vehicle capacity 50.
- Source checkpoint: `logs/cvrp1000_agfn_capacity50/po_1000steps/best.ckpt` on g53.
- Source checkpoint SHA256: `94cbb5e1c7b6391f31ae113c1beae81491cdfffad17199d1db0ac51205a9fd26`.
- Source held-out fixed32 mean: `128.06427645683289` with 100 starts, augmentation 1, BF16.
- Both variants load exactly these model weights. Because the objective changes, both start a fresh Adam optimizer; the PO checkpoint optimizer is not restored.
- Step-0 validation must be identical across variants. Any mismatch invalidates the screen.

## Locked variants

1. `usw_poanchor_w090_k64_a001_screen200`
   - CVRP100 USW dense uniform all-pairs artifact SHA256 `022bf1a3e81b7a01c346153cf1e338e93a2a9c50e5f52b288c62f4b8cf084304`.
   - 64 rollouts per instance, preference alpha 0.01, weights not detached.
   - Convex PO anchor weight 0.90, exponential PO alpha 0.05.
2. `asw_poanchor_w090_k40_a001_detach_screen200`
   - CVRP100 ASW artifact SHA256 `9652dddda2eeca7879514c57126a9601bbee12974ba5905ee5ce1d8ee83f6471`.
   - 40 rollouts per instance, preference alpha 0.01, preference weights detached.
   - Convex PO anchor weight 0.90, exponential PO alpha 0.05.

## Shared training controls

- Seed 1234; deterministic dynamic instance indices starting at 300,000,000.
- Same instance indices for both variants.
- Effective instance batch 128, BF16, Adam LR `1e-5`, weight decay `1e-6`.
- Preferred execution is physical batch 128 with memory-efficient forced replay and exact replay verification. A feasibility probe runs first. If a 24 GiB GPU cannot hold it, use the largest common physical batch that fits and gradient accumulation to keep effective batch 128; record this as a protocol-preserving hardware fallback.
- Rollout count remains per-instance: USW 64 and ASW 40. Dataloader batch and rollout axes remain separate.
- Candidate pools and USW/ASW pair construction remain strictly instance-local. Loss is computed per instance and then averaged across instances.
- Validation: `data/vrp/agfn_vrp1000_capacity50_val_seed4321.npz`, SHA256 `8de3059ffffd6594e10b3e91e9dfeeae271342c240518656aacf8bb260d35e94`, fixed32, 100 starts, augmentation 1, every 100 optimizer steps.
- Every fresh or resumed run uses a new output directory. A history step reset marks the directory corrupt.

## Screen and promotion

- Run a one-update integrity/VRAM probe for each variant before the 200-step screen. Require finite loss and gradients, replay error at most `1e-5`, and the expected batch/rollout semantics.
- Run both variants for 200 optimizer updates unless OOM, NaN/nonfinite metrics, replay mismatch, or clear held-out collapse occurs.
- Promote a variant only if its step-200 validation mean is below its shared step-0 mean.
- Promotion restores the exact step-200 `last.ckpt`, including Adam state, for 800 additional updates to absolute step 1000.
- The promoted data stream starts at `300025600 = 300000000 + 200 * 128`, so no screen instances are replayed.
- Candidate selection uses held-out validation only; the official test set is not used for tuning.

## Final evaluation and success criterion

- Official dataset: `data/vrp/agfn_vrp1000_capacity50_test128.npz`, SHA256 `58982b9168fb21c5defe36d6e1d366f973830d4c7d3c06533d6a4ae52a4f295d`.
- Evaluate the PO source and each locked step-1000 variant on exactly aligned indices 0..127 with 100 starts, 8 augmentations, FP32, and full forward evaluation.
- For each method, report all 128 per-instance costs and the mean.
- Statistical test: 100,000-sample paired bootstrap over candidate-minus-PO costs, seed 1234.
- Success requires each variant independently to have lower mean cost than the PO source and a wholly negative paired-bootstrap 95% confidence interval.
- Mixed objectives are reported explicitly as `USW-PO-anchor` and `ASW-PO-anchor`, not as pure USW/ASW.
