from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
ENTRYPOINT = REPO_ROOT / "scripts" / "train_ffsp1000_objectives.py"

torch.backends.cuda.enable_cudnn_sdp(False)
print(
    f"cudnn_sdp_enabled={torch.backends.cuda.cudnn_sdp_enabled()}",
    flush=True,
)
sys.argv[0] = str(ENTRYPOINT)
runpy.run_path(str(ENTRYPOINT), run_name="__main__")
