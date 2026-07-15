# CVRP1000 PO/BOPO protocol aligned to AGFN (ICLR 2025)

Status: preregistration; must be committed before evaluation or training.

## Paper anchor

- Paper: *Adversarial Generative Flow Network for Solving Vehicle Routing Problems*, ICLR 2025.
- CVRP1000 distribution: depot/customer coordinates uniform on `[0,1]^2`, integer demands uniform in `1..9`, vehicle capacity `50`.
- Official test set: 128 instances from `ZHANG-NI/AGFN`, artifact `data/cvrp/testDataset-1000.pt`, Git-LFS SHA256 `b8dd0cee789fbda29faf0c68a6de7e351782396a13e2b235c00ca71d6bd9d8ec`.
- Paper Table 1 POMO target: cost `233.093524` without augmentation and `192.78563` with eightfold augmentation.

## Data conversion and verification

- Download the exact official tensor; do not substitute a newly sampled test set.
- Preserve all 128 source rows in order.
- Recover integer demand as `round(normalized_demand * 50)` and require every value to be in `1..9` with zero residual.
- Extract depot/customer coordinates and recompute the complete Euclidean distance matrix for every source row; require max absolute disagreement with the official stored matrix at most `1e-10`.
- Record the converted NPZ SHA256 before evaluation.

## Training

- Common initialization: `downloads/cvrp100/po/checkpoint.ckpt` for both PO and BOPO.
- Capacity `50` in dynamic training and all validation/evaluation data.
- Initial matched budget: 1000 continuation optimizer steps each.
- Per-instance rollouts: 20; train batch size: 1.
- PO: Bradley-Terry preference objective. BOPO: `anchor_best`, paper selection, K=10.
- Adam, LR `1e-5`, weight decay `1e-6`, BF16 mixed precision.
- Identical deterministic instance indices and action seeds for PO and BOPO.

## Evaluation and decision

- Exact official 128 rows, 100 starts, 8 augmentations, FP32.
- Primary paper reproduction target: PO mean cost `<= 192.78563` after at least 1000 continuation steps.
- Stretch target: PO mean cost `<= 191.82170185` (0.5% below the paper POMO result).
- Always report zero-shot, PO, and BOPO per-instance results on identical source indices.
- If PO misses after 1000 steps, continue PO and BOPO in matched chunks from verified checkpoints; do not weaken the paper target.
- The earlier capacity150 experiment is obsolete exploratory work and cannot satisfy this protocol.
