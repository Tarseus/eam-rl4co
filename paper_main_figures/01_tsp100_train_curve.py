from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from curves import plot_train_curves as train_curves


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_curves.OUTPUT_DIR = OUT_DIR
    train_curves.plot_dataset("tsp100", train_curves.DATASETS["tsp100"])
    print(OUT_DIR / "tsp100_train_curve.png")


if __name__ == "__main__":
    main()
