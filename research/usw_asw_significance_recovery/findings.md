# USW/ASW significance recovery findings

## Current understanding

- FFSP50 epoch117 improved the locked mean to 49.635, below PO4COPs 49.642 and BOPO 49.655, but the effects remain dominated by ties. The two-sided Wilcoxon raw p-values are 0.7268 and 0.2838 and both 52-family Holm p-values are 1.0, so this candidate does not recover either win.
- JSSP15x15 epoch13 reversed the mean direction against BOPO (1301.490 versus 1303.180), but the 51/2/47 signed outcomes give raw p=0.3369 and 52-family Holm p=1.0. It therefore does not recover the locked win.
- CVRP50 ASW already significantly beats PO4COPs, SLL, and BOPO. Its unresolved target is ASW versus USW. Two lower-LR continuations worsened validation, so evaluating them would be test-set fishing.
- The CVRP50 `ASW-after-USW variant` validation-only screen is running on g52. It is not the original ASW method and no paper-test result has been read for selection.
- The two failed USW candidates corrected their mean directions but not their effect sizes. Follow-up low-LR screens are therefore running from the validation-best checkpoints: FFSP50 reheats from its post-milestone LR near 1e-6 to 1e-5, while JSSP15x15 drops from 2e-4 to 5e-5. Fresh independent confirmation sets were generated and hashed before either screen started.

## Lessons and constraints

- Keep the original two-sided Wilcoxon plus global Holm family; do not switch to a more favorable test post hoc.
- Select checkpoints only by validation data, then evaluate once on the locked paper instances.
- An ASW continuation initialized from USW must be reported as `ASW-after-USW`, not silently substituted for the original single-stage ASW method.
- Improving the mean direction was insufficient for both locked USW candidates; neither paper checkpoint registry entry should be replaced.

## Open questions

- Can the short ASW phase improve the fixed CVRP50 validation score over the source USW checkpoint strongly enough to justify locking one `ASW-after-USW variant` candidate for a single paper-test evaluation?
- Can either low-LR USW screen improve validation enough to justify one evaluation of all methods on its untouched fresh confirmation set?
