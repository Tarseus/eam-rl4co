# Official SLIM Baseline Recovery Findings

## Current understanding

- Historical `sll_loss` results measure a custom listwise objective, not published SLIM.
- The main BOPO/PO comparisons remain valid, but every SLL/SLIM-labelled row must be replaced or relabelled.
- Unit tests pass locally and on g51 (19/19). One-batch backbone smokes produced finite losses for TSP50 (0.866), CVRP50 (0.532), FFSP50 (1.355), and JSSP10x10 (2.151). The JSSP computation itself passed but its first checkpoint callback required a validation-monitor correction because the generated split lacks reference gaps.
- An unintended inherited `test=true` caused the routing/FFSP smoke only to touch fixed test data. Those values are discarded and no choice was made from them; every full configuration is now `test=false` until a validation-selected checkpoint is locked.
