from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from curves import plot_ffsp


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_ffsp.OUTPUT_DIR = OUT_DIR
    plot_ffsp.plot_ffsp_dataset("ffsp100", plot_ffsp.FFSP_DATASETS["ffsp100"])
    print(OUT_DIR / "ffsp100_train_curve.png")


if __name__ == "__main__":
    main()
