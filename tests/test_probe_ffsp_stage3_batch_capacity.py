from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import tempfile

from scripts.probe_ffsp_stage3_batch_capacity import (
    _effective_fixed_pomo,
    _parse_batch_sizes,
    _parse_explicit_fids,
    _pick_target_fids,
)


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


def test_parse_explicit_fids() -> None:
    assert _parse_explicit_fids("f1, f2,,f3") == ["f1", "f2", "f3"]


def test_pick_target_fids_prefers_failed_stage3_pairs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        pairs_path = run_dir / "pairs.jsonl"
        rows = [
            {"generation": 0, "pair_index": 1, "f_id": "f_ok", "pair_reason": "ok_stage3_offline_minitrain"},
            {"generation": 1, "pair_index": 2, "f_id": "f_bad_a", "pair_reason": "stage3_early_pruned"},
            {"generation": 1, "pair_index": 3, "f_id": "f_bad_b", "pair_reason": "stage3_runtime_error"},
        ]
        pairs_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        out = _pick_target_fids(run_dir, explicit_fids_raw=None, failed_only=True, max_targets=None)
        assert out == ["f_bad_b", "f_bad_a"]
