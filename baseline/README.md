# Global baselines (high-fidelity metrics)

This directory is a cross-run cache for the high-fidelity (HF) `po_loss` baseline.

During `PTP/ptp_discovery/free_loss_eoh_loop.py`, we compute a stable baseline key
from the HF config (env/policy/size/epochs/etc.). If `baseline/<key>/baseline.json`
exists, the run reuses it instead of re-training the baseline.

Generated files (ignored by git):

- `baseline/<key>/baseline.json`: the baseline evaluation output (JSON)
- `baseline/<key>/epoch_objectives.json`: `epoch_eval.objectives` as a plain list

To register a previous run’s baseline into this cache, use:

- `python scripts/register_baseline.py --run-dir <path-to-run-dir>`

