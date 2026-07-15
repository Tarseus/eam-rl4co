# CVRP1000 PO/BOPO Findings

## Current understanding

**Protocol correction:** the user requires direct comparison to a published result. The capacity150 self-calibration below is now obsolete exploratory work. The confirmatory protocol follows AGFN (ICLR 2025): capacity50, demand1..9, exact official 128-instance CVRP1000 test set, with POMO targets 233.093524 without augmentation and **192.78563 with eightfold augmentation**. PO must reproduce or beat 192.78563 after at least 1000 continuation steps; the 0.5% stretch target is 191.82170185.

The repository uses the POMO-style heavy encoder (six attention layers) and light one-layer decoder through `PO4COPsCVRPPolicy`. Public large-scale CVRP results show that this architecture's cross-size generalization is highly sensitive to the vehicle capacity and decode budget, so an external absolute cost cannot be copied into this project without matching the data distribution.

The repository's native CVRP1000 generator uses capacity 150 and integer demands 1 through 9. The primary target will therefore be established by evaluating the existing CVRP100 PO checkpoint on a locked local CVRP1000 fixed100 set with 100 starts, eight geometric augmentations, and FP32. PO must finish at least 1000 continuation steps and match or improve that cost; the optimization target is a 0.5% improvement. BOPO receives the same initialization, data indices, rollout count, and continuation-step budget.

The calibration completed on g53 at a mean cost of **59.875156478881834** over the 100 fixed instances. This is the hard PO threshold. The preregistered 0.5% improvement target is **59.57578069648743**.

## Literature calibration

- POMO introduced the six-layer encoder, one-layer decoder, multistart training, and eightfold augmentation used by this codebase: https://arxiv.org/abs/2010.16011
- AGFN reports CVRP1000 POMO cost 233.093524 without augmentation and 192.78563 with eightfold augmentation, but its CVRP1000 capacity is 50, so these costs are not directly comparable to this repository's capacity-150 setting: https://proceedings.iclr.cc/paper_files/paper/2025/file/b210c387381713a14a4f5a607aff3520-Paper-Conference.pdf
- Recent large-scale studies use other capacity schedules such as 250 for CVRP1000 and document severe degradation of the original POMO light decoder under cross-size transfer. This supports measuring the actual checkpoint rather than importing a headline number.

## Constraints

- PO and BOPO start from the same CVRP100 PO checkpoint.
- Candidate pools and BOPO anchor pairs remain instance-local.
- The initial implementation uses train batch size one; no mixed-instance aggregation ambiguity is permitted.
- Proxy validation does not establish success. Only the locked fixed100 evaluation counts.
- If PO misses the cost target after the first 1000-step chunk, training continues in matched chunks rather than weakening the target.

## Launch health

One-step CVRP1000 probes completed for both objectives. PO used about 1.91 GiB reserved GPU memory and produced finite loss and gradients. BOPO produced loss 0.693075, grad norm 0.1730, and about 1.90 GiB reserved memory. The full 1000-step PO and BOPO jobs were then launched on g53 GPUs 0 and 1 respectively.

The first formal boundary reached was BOPO continuation step 100: validation32 improved from 60.896614 at step 0 to 60.565683. PO was healthy at continuation step 96 in the same snapshot. A final bounded SSH attempt to capture PO's imminent step-100 validation hit a banner timeout, so the run stops polling and preserves the last verified state for the next tick.

After the user's correction, both capacity150 jobs were stopped and preserved at PO step186 and BOPO step189. They cannot be resumed or cited as satisfying the paper-aligned experiment.

The exact 981 MiB AGFN official tensor was downloaded on g53 and matched its Git-LFS SHA256. Conversion preserved source indices 0..127, recovered only integer demands 1..9 at capacity50, and reproduced every stored pairwise distance with maximum absolute error `2.22e-16`. The compact evaluation NPZ SHA256 is `58982b91...a4f295d`.
