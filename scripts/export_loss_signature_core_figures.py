from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

CORE_FILES = [
    "02_all_methods_gap_matched_residual_surface.png",
    "02_all_methods_gap_matched_residual_surface.pdf",
    "06_counterfactual_objective_scale_sensitivity.png",
    "06_counterfactual_objective_scale_sensitivity.pdf",
    "surface.csv",
    "scale_sensitivity.csv",
    "nearest.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "figures" / "loss_fine_grained_signature" / "20260501-final-all-methods"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "figures" / "loss_fine_grained_signature" / "core_02_06"),
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in CORE_FILES:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    (out_dir / "README.md").write_text(
        "# Core loss-signature figures\n\n"
        "This lean export keeps only the two mechanism figures used in the main text.\n\n"
        "## 02_all_methods_gap_matched_residual_surface\n\n"
        "Rows are problems and columns are existing preference methods. Each heatmap cell is\n"
        "`log2(loss-only coefficient / gap-matched baseline coefficient)` after matching each\n"
        "baseline's marginal objective-gap response to loss-only. The x-axis is policy-margin\n"
        "quantile (`low=hard`, `high=easy`) and the y-axis is objective-gap quantile\n"
        "(`low=fine`, `high=coarse`). Red means loss-only gives a stronger coefficient;\n"
        "blue means it gives a weaker coefficient.\n\n"
        "## 06_counterfactual_objective_scale_sensitivity\n\n"
        "The x-axis scales objective dispersion within the same replay pool while keeping policy\n"
        "margins fixed: `c' = mean(c) + s(c - mean(c))`. The y-axis is the mean coefficient\n"
        "normalized by the method's own value at `s=1`. This reveals whether a loss has\n"
        "objective-dispersion-adaptive temperature.\n\n"
        "Copied files:\n"
        + "\n".join(f"- `{name}`" for name in copied)
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] copied {len(copied)} files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
