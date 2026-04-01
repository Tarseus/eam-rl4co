from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
ptp_root = repo_root / "PTP"
for path in (repo_root, ptp_root):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from fitness.free_loss_fidelity import _should_aggressive_cuda_cleanup


def test_aggressive_cuda_cleanup_enabled_for_ffsp100() -> None:
    cfg = {
        "env_name": "ffsp",
        "generator_params": {"num_job": 100},
        "train_problem_size": 100,
    }
    assert _should_aggressive_cuda_cleanup(cfg) is True


def test_aggressive_cuda_cleanup_disabled_for_smaller_ffsp() -> None:
    cfg = {
        "env_name": "ffsp",
        "generator_params": {"num_job": 50},
        "train_problem_size": 50,
    }
    assert _should_aggressive_cuda_cleanup(cfg) is False


def test_aggressive_cuda_cleanup_disabled_for_other_envs() -> None:
    cfg = {
        "env_name": "tsp",
        "generator_params": {"num_loc": 100},
        "train_problem_size": 100,
    }
    assert _should_aggressive_cuda_cleanup(cfg) is False
