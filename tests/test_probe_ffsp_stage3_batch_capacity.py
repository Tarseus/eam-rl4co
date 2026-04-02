from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.probe_ffsp_stage3_batch_capacity import _effective_fixed_pomo, _parse_batch_sizes


def test_parse_batch_sizes_explicit_values() -> None:
    assert _parse_batch_sizes("8,4,8,16", min_batch_size=1, max_batch_size=32) == [4, 8, 16]


def test_parse_batch_sizes_range_generation() -> None:
    assert _parse_batch_sizes(None, min_batch_size=2, max_batch_size=5) == [2, 3, 4, 5]


def test_effective_fixed_pomo_prefers_explicit_value() -> None:
    cfg = {"pomo_size": 4, "train_problem_size": 100}
    assert _effective_fixed_pomo(cfg, 24) == 24


def test_effective_fixed_pomo_falls_back_to_problem_size() -> None:
    cfg = {"pomo_size": None, "train_problem_size": 100}
    assert _effective_fixed_pomo(cfg, None) == 100
