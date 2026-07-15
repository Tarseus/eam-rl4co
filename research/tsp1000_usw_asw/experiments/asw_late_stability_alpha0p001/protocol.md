# ASW late-stability recovery protocol

Status: exploratory protocol locked before execution.

## Motivation

ASW K40/alpha=0.01/detached reached its best validation32 mean 26.856804 at absolute optimizer step400, then deteriorated through step1000. This indicates late preference saturation or drift rather than failure to learn.

## Intervention

- Resume the preserved ASW optimizer-step400 best checkpoint.
- Keep K40 and detached ASW weights unchanged.
- Reduce alpha from 0.01 to 0.001.
- Continue exactly 600 fresh-data updates to absolute optimizer step1000 using `data_start_index=30000000`.
- Keep batch128, LR1e-5, BF16, validation32/1000-start/augmentation1 every 100 steps.

## Decision rule

- Reject on NaN, replay error, or renewed validation collapse.
- At step1000, run canonical fixed100/1000-start/8-augment/FP32 evaluation.
- ASW succeeds only if mean is below PO 27.13804609298706 and paired candidate-minus-PO bootstrap 95% CI is wholly below zero.

