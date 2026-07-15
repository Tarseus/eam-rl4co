# ASW restart from preserved step400 best

Status: user-directed protocol amendment locked before restart.

## Reason

The first ASW additional-2000 launch resumed the degraded optimizer-step1000 endpoint for budget alignment. The user requires continuation from the best previously observed checkpoint instead. That run is stopped and retained only as an aborted diagnostic.

## Source checkpoint

- Path: `logs/tsp1000_dynamic/promote1000_20260714/asw_k40_a001_detach_from300/best.ckpt`
- SHA256: `b9917a1d3e39bf549e395ba132ae2fe5efa01f3ca32a4b13a97c1966de743e13`
- Method: ASW
- Optimizer/best step: 400
- Preserved validation32 best mean: 26.856803596019745

## Restart configuration

- Add 2000 optimizer updates, ending at absolute optimizer step2400.
- Keep K40, detached ASW weights, batch128, LR1e-5, BF16, seed1234.
- Use alpha0.001 as the anti-drift intervention.
- Use fresh `data_start_index=50000000`; do not reuse the aborted endpoint run's ds40m slice.
- Validate on fixed32 with 1000 starts and augmentation1 every 100 continuation steps.
- Preserve the aborted output and write to a new directory.

## Interpretation and final decision

- This is a best-checkpoint continuation rather than an equal-endpoint-budget comparison; report both the 2000 added updates and absolute step2400 transparently.
- Run canonical fixed100/1000-start/8-augment/FP32 evaluation on the selected checkpoint.
- ASW succeeds only if its mean is below matched PO 25.953238353729247 and the paired candidate-minus-PO bootstrap 95% CI lies wholly below zero.

