from __future__ import annotations

import sys
import shutil
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_ffsp_control_ablation_figures as ffsp_ablation


def _rename(source: Path, target_name: str) -> Path:
    target = OUT_DIR / target_name
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
        try:
            source.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ffsp_ablation.BUILDER_OUT = OUT_DIR
    png, pdf, summary = ffsp_ablation.plot_ffsp_builder_geometry_weight_ablation()
    print(_rename(png, "ffsp100_weighting_ablation.png"))
    print(_rename(pdf, "ffsp100_weighting_ablation.pdf"))
    print(_rename(summary, "ffsp100_weighting_ablation_summary.csv"))


if __name__ == "__main__":
    main()
