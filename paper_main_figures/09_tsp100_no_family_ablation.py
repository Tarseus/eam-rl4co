from __future__ import annotations

import sys
import shutil
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_tsp_family_ablation as tsp_family


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tsp_family.OUT_DIR = OUT_DIR
    tsp_family.main()
    source = OUT_DIR / "04_tsp_family_ablation.png"
    target = OUT_DIR / "tsp100_no_family_ablation.png"
    if source.exists():
        shutil.copy2(source, target)
        try:
            source.unlink(missing_ok=True)
        except OSError:
            pass
    print(target)


if __name__ == "__main__":
    main()
