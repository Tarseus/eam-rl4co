# TSP1000 additional-2000 protocol for USW and ASW

Status: confirmatory USW protocol and exploratory ASW-stability protocol locked before execution.

## Corrected scope

The user requires 2000 additional training steps on TSP1000. PO has already completed the matched `from1000_add2000` run, whose locked canonical fixed100 mean is 25.953238353729247.

## USW

- Resume the verified TSP1000 USW K64/alpha=0.01 optimizer-step1000 endpoint.
- Add exactly 2000 optimizer updates on fresh dynamic TSP1000 instances.
- Keep K64, alpha0.01, batch128, LR1e-5, BF16, seed1234, and forced replay checks.
- Use `data_start_index=40000000`, validation32/1000-start/augmentation1 every 100 updates.

## ASW

- Resume the TSP1000 ASW K40/detached optimizer-step1000 endpoint.
- Add exactly 2000 optimizer updates on the same fresh-data index schedule.
- Reduce alpha from 0.01 to 0.001 to address the observed late saturation and drift; keep K40, detached weights, batch128, LR1e-5, BF16, and seed1234.
- Use validation32/1000-start/augmentation1 every 100 updates and preserve prior checkpoints.

## Final decision

- Evaluate final/best candidates on the aligned canonical fixed100 dataset with 1000 starts, 8 augment, and FP32.
- Record dataset SHA and exact indices 0..99.
- Each method is compared to the matched PO `from1000_add2000` per-instance results.
- ASW is complete only if its mean is below PO and the paired candidate-minus-PO bootstrap 95% CI lies wholly below zero.

