from __future__ import annotations

import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
ptp_root = repo_root / "PTP"
for path in (repo_root, ptp_root):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

from fitness.free_loss_fidelity import (
    get_rollout_debug_events,
    reset_rollout_debug_events,
    _log_rollout_input_debug,
)


class _DummyEnv:
    name = "ffsp"
    num_stage = 3
    num_machine = 4
    flatten_stages = False


class _DummyPolicy:
    train_decode_type = "sampling"
    decoders = []


def test_rollout_debug_events_capture_summary() -> None:
    reset_rollout_debug_events()
    _log_rollout_input_debug(
        cfg_like={"cuda_diagnostics_enabled": True},
        env=_DummyEnv(),
        policy=_DummyPolicy(),
        td={"run_time": torch.ones(2, 3)},
        phase="train",
        rollout_strategy="policy_multistart",
        device=torch.device("cpu"),
        precision="32-true",
        batch_size=2,
        num_rollouts=1,
        include_decoder_cache=False,
        message="test rollout debug",
    )
    events = get_rollout_debug_events()
    assert len(events) == 1
    assert events[0]["message"] == "test rollout debug"
    assert events[0]["summary"]["phase"] == "train"
    assert events[0]["summary"]["td"]["items"]["run_time"]["shape"] == [2, 3]
    reset_rollout_debug_events()
