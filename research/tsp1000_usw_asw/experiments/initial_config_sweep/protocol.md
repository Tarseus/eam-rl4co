# Exploratory protocol: K, alpha, and ASW weight gradient

This sweep was launched before the autoresearch workspace was initialized, so it is exploratory rather than preregistered confirmatory evidence.

## Interventions

- USW: K40/alpha0.05, K40/alpha0.01, K64/alpha0.01.
- ASW: K40/alpha0.05, K40/alpha0.01, K40/alpha0.01 with detached pair weights.
- All runs use batch 128, seed 1234, learning rate 1e-5, 300 continuation steps, fixed validation32, and encoder-only fused SDPA.

## Predictions

- Lower alpha should slow late logistic saturation.
- Larger K should improve preference estimates and provide more hard pairs.
- Detaching ASW weights should help if the policy is gaming the weighting rule; it may hurt if full-gradient reweighting is essential.

## Decision rule

Rank by validation32 only for screening. Compare the 300-continuation-step branches against a PO control trained for the same 300 continuation steps; do not use PO fixed100=27.138046 as the short-sweep baseline. Evaluate promising checkpoints on canonical fixed100 only to decide which configurations advance. Final success requires continuing a candidate for 1000 continuation steps, then beating the PO-1000 fixed100 mean 27.138046 with a paired bootstrap 95% confidence interval below zero.

The method-specific pilot best checkpoints occur at different optimizer steps (PO 75, USW 75, ASW 100). Therefore reports must distinguish continuation step from absolute optimizer step; the controlled comparison uses continuation-step budget.
