# TSP2000 USW/ASW Findings

## Current understanding

- TSP1000 established that USW K64 with alpha 0.01 can significantly beat equal-budget PO.
- TSP2000 increases memory pressure, so a matched batch-size probe is required before the locked 1000-step transfer begins.
- PO, USW, and ASW must be compared on aligned fixed100 instances under the same final inference protocol.

## Open questions

- Whether the TSP1000 USW advantage transfers to TSP2000.
- Which ASW stability configuration can independently beat PO at TSP2000.

## Scope correction

The user clarified that “2000” meant 2000 training steps on TSP1000, not TSP2000. The two-step memory probe is retained only as an engineering note; the TSP2000 transfer study was stopped and must not be interpreted as an active research objective.

