from __future__ import annotations

import sys
import subprocess
import shutil
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_weighting_only_branch_ablation as weighting_ablation


def _rename(source: Path, target_name: str) -> Path:
    target = OUT_DIR / target_name
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
        try:
            source.unlink(missing_ok=True)
        except OSError:
            pass
    return target


def _safe_git_text(commit: str, rel_path: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT.as_posix()}",
            "show",
            f"{commit}:{rel_path}",
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weighting_ablation.OUT_DIR = OUT_DIR
    weighting_ablation._git_text = _safe_git_text
    png, pdf, summary = weighting_ablation._plot()
    print(_rename(png, "tsp100_weighting_ablation.png"))
    print(_rename(pdf, "tsp100_weighting_ablation.pdf"))
    print(_rename(summary, "tsp100_weighting_ablation_summary.csv"))


if __name__ == "__main__":
    main()
