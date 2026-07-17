# Locked protocol: FFSP50 and JSSP15x15 USW validation-best recovery

This is a confirmatory evaluation of two checkpoints selected only by their original validation monitors. No paper-test result was used to choose either checkpoint.

## FFSP50 USW

- Checkpoint: `g52:logs/train/runs/ffsp50_loss_only_recover_valbest_20260716-2210/checkpoints/epoch_117.ckpt`
- SHA256: `0b52e9de24c31cc373290c3397081c085914e682981b406bde645a6b068fab1d`
- Data SHA256: `6b1a46fb75fbdb1e23433c0a11ae9e54adb18a3306df7b9e46ccc357f1d4f372`
- Evaluation: 1000 fixed instances, 24 starts, 128 augmentations, seed 1234.
- Success requires a lower mean than both PO4COPs and BOPO and Holm-adjusted p < 0.05 for both comparisons after replacement in the unchanged 52-comparison family.

## JSSP15x15 USW

- Checkpoint: `g52:logs/train/runs/mgl-jssp-bopo-lossonly_15x15_recover_valbest_20260716-2210/checkpoints/epoch_013.ckpt`
- SHA256: `93b79af77a42156753882f5d37c4be1b1e9b814d9890852b666de278be1ac3ff`
- Ordered 100-file dataset SHA256: `cf30cda4099e756540c5fe15bd6a47849a6b74a3325bedb0bb7066db40c9294c`
- Evaluation: B=128, greedy=0, no augmentation, per-instance sampling seed `12345678 + i`.
- Success requires a lower mean than BOPO and Holm-adjusted p < 0.05 after replacement in the unchanged family.

All pair construction, candidate pools, rollout semantics, and model logic remain unchanged.
