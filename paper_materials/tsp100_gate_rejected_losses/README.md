# TSP100 Gate-Rejected Loss Candidates: Three Failure Modes

This folder is the clean paper-material set for TSP100 loss-search gate rejections. It intentionally contains only three conceptual failure modes: `affine_unstable`, `preference_direction`, and `preference_saturation`. Interface/observable mismatches and runtime/autograd errors are excluded.

Each candidate has `loss.py`, `metadata.json`, and `note.md`.

## How to Use in the Paper

- **Affine-unstable:** locally prefers the winner, but treats raw objective scale/shift as semantically meaningful.

- **Preference-direction:** has pairwise ingredients, but the gradient or swap semantics are wrong.

- **Preference-saturation:** uses a hinge/link/threshold that collapses useful gradient signal.

## Affine-unstable candidates

These formulas pass the basic winner/loser preference gate, but CO-alignment rejects them because the loss changes under irrelevant objective affine transformations. They are good examples of plausible cost-gap objectives that use the objective scale too literally.

- `f000_001_4b362843` `loss_logsigmoid_000_cost_structure_shifted` -> `affine_unstable/f000_001_4b362843__loss_logsigmoid_000_cost_structure_shifted/note.md`; where_failed=`[]`, co=`affine_invariance_violation`

- `f003_007_e4cd5f0f` `pairwise_cost_gap_logsigmoid` -> `affine_unstable/f003_007_e4cd5f0f__pairwise_cost_gap_logsigmoid/note.md`; where_failed=`[]`, co=`affine_invariance_violation`

- `f002_001_b6517213` `loss_logsigmoid_003_cost_shifted_median` -> `affine_unstable/f002_001_b6517213__loss_logsigmoid_003_cost_shifted_median/note.md`; where_failed=`[]`, co=`affine_invariance_violation`

- `f005_001_870bf5b3` `pairwise_cost_gap_logsigmoid_softsign_median_normalized` -> `affine_unstable/f005_001_870bf5b3__pairwise_cost_gap_logsigmoid_softsign_median_normalized/note.md`; where_failed=`[]`, co=`affine_invariance_violation`

## Preference-direction failures

These formulas are executable and have an interpretable pairwise structure, but the joint preference gate detects wrong swap behavior or wrong log-probability gradient direction. They show that plausible ingredients do not guarantee correct preference semantics.

- `f000_011_8a975d1f` `pairwise_pref_loss` -> `preference_direction/f000_011_8a975d1f__pairwise_pref_loss/note.md`; where_failed=`['swap']`, co=`None`

- `f001_003_5da0a919` `pairwise_cost_gap_logs` -> `preference_direction/f001_003_5da0a919__pairwise_cost_gap_logs/note.md`; where_failed=`['swap']`, co=`None`

- `f003_005_935306c7` `pairwise_cost_gap_logsigmoid_normalized` -> `preference_direction/f003_005_935306c7__pairwise_cost_gap_logsigmoid_normalized/note.md`; where_failed=`['swap']`, co=`None`

- `f004_006_d7845875` `pairwise_weighted_logsigmoid_cost_gap` -> `preference_direction/f004_006_d7845875__pairwise_weighted_logsigmoid_cost_gap/note.md`; where_failed=`['swap']`, co=`None`

## Preference-saturation failures

These formulas collapse to zero or near-zero effective gradients on the gate batch. They may encode a reasonable hinge, softplus, or advantage idea, but their link/thresholding makes the preference signal inactive or flat.

- `f000_012_50dd0498` `pairwise_cost_gap_obj_loss` -> `preference_saturation/f000_012_50dd0498__pairwise_cost_gap_obj_loss/note.md`; where_failed=`['log_prob_w_direction', 'log_prob_l_direction', 'saturation']`, co=`None`

- `f004_015_9aecb21d` `pairwise_cost_gap_obj_loss_normalized` -> `preference_saturation/f004_015_9aecb21d__pairwise_cost_gap_obj_loss_normalized/note.md`; where_failed=`['log_prob_w_direction', 'log_prob_l_direction', 'saturation']`, co=`None`

- `f001_012_96c50a54` `loss_bayesian_pairwise_softplus_advantage_scaled` -> `preference_saturation/f001_012_96c50a54__loss_bayesian_pairwise_softplus_advantage_scaled/note.md`; where_failed=`['log_prob_w_direction', 'log_prob_l_direction', 'saturation', 'swap']`, co=`None`

- `f001_001_b3049121` `loss_pairwise_rank_tanh_cost_advantage` -> `preference_saturation/f001_001_b3049121__loss_pairwise_rank_tanh_cost_advantage/note.md`; where_failed=`['log_prob_w_direction', 'log_prob_l_direction', 'saturation', 'swap']`, co=`None`
