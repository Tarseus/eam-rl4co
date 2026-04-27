from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
ptp_root = repo_root / "PTP"
for path in (repo_root, ptp_root):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from fitness.free_loss_fidelity import (
    _aggressive_cuda_cleanup_mode,
    _effective_precision_mode,
    _should_aggressive_cuda_cleanup,
    _should_run_aggressive_cleanup,
)


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


def test_ffsp_matnet_mixed_precision_forces_fp32() -> None:
    cfg = {
        "env_name": "ffsp",
        "policy_name": "matnet",
        "precision": "16-mixed",
    }
    assert _effective_precision_mode(cfg) == "32-true"


def test_other_policies_keep_requested_precision() -> None:
    cfg = {
        "env_name": "tsp",
        "policy_name": "am",
        "precision": "16-mixed",
    }
    assert _effective_precision_mode(cfg) == "16-mixed"


def test_aggressive_cuda_cleanup_mode_defaults_to_step_for_ffsp100() -> None:
    cfg = {
        "env_name": "ffsp",
        "generator_params": {"num_job": 100},
        "train_problem_size": 100,
    }
    assert _aggressive_cuda_cleanup_mode(cfg) == "step"


def test_aggressive_cuda_cleanup_epoch_mode_skips_step_cleanup() -> None:
    cfg = {
        "env_name": "ffsp",
        "generator_params": {"num_job": 100},
        "train_problem_size": 100,
        "aggressive_cuda_cleanup_mode": "epoch",
    }
    assert _aggressive_cuda_cleanup_mode(cfg) == "epoch"
    assert _should_run_aggressive_cleanup(cfg, when="step") is False
    assert _should_run_aggressive_cleanup(cfg, when="epoch") is True
    assert _should_run_aggressive_cleanup(cfg, when="phase") is True
