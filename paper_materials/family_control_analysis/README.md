# Family-Control Analysis Materials

This folder contains paper-writing material for explaining family control / no-family ablation, analogous to the no-gate rejected-loss material.

## Files

- `interpretation.md`: ready-to-use explanation and suggested paper wording.
- `summary.csv`: run-level comparison for TSP100 and FFSP100.
- `generation_metrics.csv`: per-generation family and motif statistics.
- `motif_metrics.csv`: all-vs-late motif shares.
- `high_fidelity_candidates.csv`: high-fidelity candidates with family labels and motif tags.
- `candidate_examples/`: representative accepted and no-family drift formulas, each with `loss.py`, `metadata.json`, and `note.md`.
- `figures/`: copied no-family ablation figures from `paper_main_figures_ready`.

## Main Takeaway

Family control should be described as population-level hypothesis control, not raw diversity maximization. Without it, the search can produce more distinct-looking signatures while still drifting toward surface-novel but correlated descendants, such as entropy-regularized rank-prob losses on TSP100 or advantage-blend/tanh-ReLU variants on FFSP100.
