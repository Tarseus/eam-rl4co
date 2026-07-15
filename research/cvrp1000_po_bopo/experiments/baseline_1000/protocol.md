# CVRP1000 PO/BOPO baseline protocol

Status: preregistration draft; commit before remote launch.

## Data

- CVRP1000 with coordinates uniform on the unit square.
- Customer demands are integers 1 through 9.
- Vehicle capacity is 150, matching `CVRPGenerator.CAPACITIES[1000]`.
- Validation: 32 fixed instances, seed 4321, SHA256 `5df804938d3fcb8e5c7d0a9d39ca07c4a7ed4aec85cf3f195e46e83c603bdf8f`.
- Final test: 100 fixed instances, seed 1234, SHA256 `57342809aac9c3cc29b5898d04e52fbf5d20087a0da263e8b6bbb9505fbe9ccc`; preserve exact row order 0..99.

## Training

- Common initialization: `downloads/cvrp100/po/checkpoint.ckpt`.
- Methods: PO (`po_loss`, Bradley-Terry implementation) and BOPO (`anchor_best`, paper selection, K=10).
- Initial budget: 1000 continuation optimizer steps for each method.
- Per-instance rollouts: 20 for both methods.
- Train batch size: 1; no cross-instance pair construction is possible.
- Adam, LR 1e-5, weight decay 1e-6, BF16 mixed precision.
- Identical deterministic instance indices and action seeds for both methods.
- Proxy validation every 100 steps: fixed32, 100 starts, augmentation 1.

## Evaluation and decision

- Architecture calibration: evaluate the unmodified CVRP100 PO checkpoint on fixed100 with 100 starts, 8 augment, FP32.
- PO must complete at least 1000 continuation steps and achieve mean cost no worse than the architecture calibration; aim for at least 0.5% lower cost.
- BOPO is evaluated at the same training budget and on exactly aligned fixed100 rows.
- If PO misses, resume both methods in matched chunks from verified checkpoints; do not declare completion from validation32.
