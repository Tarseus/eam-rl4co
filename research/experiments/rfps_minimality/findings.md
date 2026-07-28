# RFPS minimality findings

## Current understanding

Existing evidence shows that the three-point Fisher--Rao curvature residual predicts
pairwise real-fitness differences better than the initial full field. This does not
yet establish that three checkpoints, Fisher geometry, multiple probes, and dual
anchors are all necessary.

## Patterns and insights

- Pending preregistered ablations.

## Lessons and constraints

- Prefer a simpler descriptor whenever it meets the locked accuracy and risk margins.
- Do not interpret large curvature as candidate quality; the descriptor is only a
  behavior-neighborhood marker.
- Keep real on-policy training as the source of fitness.

## Open questions

- Can the first two flow points replace all three?
- Can a two-point gradient change replace log maps and curvature construction?
- Does Euclidean virtual updating match Fisher--Rao updating?
- How many probes and anchors are actually necessary?
