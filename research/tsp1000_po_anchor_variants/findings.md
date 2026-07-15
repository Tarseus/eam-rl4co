# TSP1000 PO-Anchored USW/ASW Findings

## Objective

Starting from the verified PO optimizer-step5075 checkpoint, find an explicitly labeled USW variant and ASW variant that each significantly beat the source PO checkpoint on canonical fixed100.

## Current Understanding

Pure USW/ASW initially learned quickly but lost their advantage during long continuation. The deployed USW uses dense uniform all-pairs construction; hard-pair emphasis is implicit in its logistic derivative rather than explicit mining. The new variants retain their respective preference signals while anchoring optimization with the stable exponential PO objective.

The one-step remote smoke tests passed for both variants with exact forced replay (`replay_error=0`). USW had loss -0.294850 and grad norm 8.9456; ASW had loss -0.101911 and grad norm 11.1655. This rules out a dead preference gradient at initialization and confirms that the hybrid objective is active.

## Constraints

- Preserve forced replay equality checks.
- Keep base instance batch at or below128.
- Never optimize directly on canonical fixed100; use held-out validation for selection and canonical fixed100 only at locked promotion boundaries.
- Label all mixed objectives as variants, not original USW/ASW.
- Do not conclude until both variants independently clear the paired canonical criterion.
