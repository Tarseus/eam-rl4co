# Initial PO-Anchor Screen Protocol

Status: CONFIRMATORY SCREENING protocol locked before launch.

## Source

- PO checkpoint: optimizer/global/epoch step5075.
- SHA256: `99e2d20ebb1c4379d0a0d1a67fbbdc2b5c08c65725350f46b0c280507f20a9eb`.
- Source canonical fixed100 mean: `25.57453468322754`.
- Initialize model weights from this checkpoint and start a fresh Adam optimizer because the objective changes.

## Variants

Both runs use the same TSP1000 dynamic instance indices, seed1234, batch128, BF16 training, validation32/1000-start/augment1 every100 steps, forced replay verification, LR1e-5, and weight decay1e-6.

1. `usw_poanchor_w090_k64_a001_screen200`
   - USW dense all-pairs discovered loss.
   - K64, alpha0.01.
   - Convex PO-anchor weight0.90, exponential PO alpha0.05.
2. `asw_poanchor_w090_k40_a001_detach_screen200`
   - ASW discovered weighted builder/loss with detached preference weights.
   - K40, alpha0.01.
   - Convex PO-anchor weight0.90, exponential PO alpha0.05.

## Decisions

- Step0 validation must reproduce identically across variants because weights share the same source checkpoint.
- Continue both through200 steps unless OOM, NaN, nonfinite metrics, nonzero replay error, or clear validation collapse occurs.
- Promote a method checkpoint only if its validation32 beats its own step0 source value.
- If either method fails, next inner-loop branches adjust anchor weight/LR and may introduce explicit hard-pair mining; all remain variants.
- Promoted variants continue to a locked total continuation budget before canonical fixed100 evaluation.
- Final success requires each method independently to beat source PO mean25.57453468322754 with a wholly negative paired bootstrap95% CI on exactly aligned100 rows.

