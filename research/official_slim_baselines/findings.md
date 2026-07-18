# Official SLIM Baseline Recovery Findings

## Current understanding

- Historical `sll_loss` results measure a custom listwise objective, not published SLIM.
- The main BOPO/PO comparisons remain valid, but every SLL/SLIM-labelled row must be replaced or relabelled.
- Unit tests pass locally and on g51 (19/19). All eight primary one-batch smokes now pass with finite losses: TSP50 0.866, TSP100 1.287, CVRP50 0.532, CVRP100 0.864, FFSP50 1.355, FFSP100 1.805, JSSP10x10 2.151, and JSSP15x15 2.501.
- g51's current shared `jssp_bopo/train` contains only 10x10 instances. JSSP15 therefore uses the already-existing shape-isolated 96-instance `jssp_scale_probe/15x15/train` split; its validation stream is the 100 filtered 15x15 instances from the common validation directory. This is explicit in the config and prevents silent mixed-shape loading.
- An unintended inherited `test=true` caused the routing/FFSP smoke only to touch fixed test data. Those values are discarded and no choice was made from them; every full configuration is now `test=false` until a validation-selected checkpoint is locked.
- Full validation-only training for all eight primary paper problems is queued on the only idle g51 GPUs (5 and 7) under collision-free roots. No full-run test data have been read.
- The first full gates are healthy: JSSP10 reached epoch1 validation makespan 887.274 with train loss 1.361629, while JSSP15 reached epoch0 validation makespan 1434.110 with train loss 2.367374. Both continued beyond those gates and both wrapper logs have zero error-pattern matches.
