from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_objective_search_figures as objective_search


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    objective_search.ACTIVE_TASK = "TSP100"
    path = objective_search._plot_search_trajectory(
        OUT_DIR,
        output_name="tsp100_no_gate_ablation.png",
        overlay_no_gate=True,
        custom_nogate_style=True,
        show_rejected_count_bars=True,
        show_no_gate_replay=False,
    )
    print(path)


if __name__ == "__main__":
    main()
