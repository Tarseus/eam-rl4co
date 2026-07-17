# USW/ASW significance recovery findings

## Current understanding

- FFSP50 is dominated by exact makespan ties. More test instances alone cannot repair the current USW mean direction; a better validation-selected checkpoint is required.
- JSSP15x15 USW is not merely underpowered: the current checkpoint is worse than BOPO in mean. The existing epoch13 validation-best checkpoint is the least invasive recovery candidate.
- CVRP50 ASW already significantly beats PO4COPs, SLL, and BOPO. Its unresolved target is ASW versus USW. Two lower-LR continuations worsened validation, so evaluating them would be test-set fishing.

## Lessons and constraints

- Keep the original two-sided Wilcoxon plus global Holm family; do not switch to a more favorable test post hoc.
- Select checkpoints only by validation data, then evaluate once on the locked paper instances.
- An ASW continuation initialized from USW must be reported as `ASW-after-USW`, not silently substituted for the original single-stage ASW method.

## Open questions

- Does FFSP50 epoch117 reduce ties in the favorable direction enough to survive global Holm correction?
- Does JSSP15x15 epoch13 reverse the mean and signed-rank effects versus BOPO?
- Can a short ASW phase preserve or improve the strong CVRP50 USW policy?
