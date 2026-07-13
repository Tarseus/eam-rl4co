from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
plot_source = REPO_ROOT / "scripts" / "plot_objective_search_figures.py"
_spec = importlib.util.spec_from_file_location("_plot_objective_search_figures", plot_source)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load module from {plot_source}")
plot_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_module)

_ensure_dir = plot_module._ensure_dir
_plot_search_trajectory_nogate_custom = plot_module._plot_search_trajectory_nogate_custom


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot only the no-gate search trajectory in its custom style.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "figures/objective_search_nogate"),
        type=str,
    )
    parser.add_argument(
        "--task",
        default=plot_module.DEFAULT_TASK,
        choices=sorted(plot_module.LOSS_RUNS),
        help="Problem/run bundle to plot.",
    )
    args = parser.parse_args()

    plot_module.ACTIVE_TASK = args.task
    outdir = Path(args.output_dir).resolve()
    _ensure_dir(outdir)
    path = _plot_search_trajectory_nogate_custom(outdir)
    print(json.dumps({"figure": str(path), "figure_svg": str(path.with_suffix('.svg'))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
