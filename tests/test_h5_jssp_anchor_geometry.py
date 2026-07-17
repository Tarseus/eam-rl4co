from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PTP_ROOT = REPO_ROOT / "PTP"
if str(PTP_ROOT) not in sys.path:
    sys.path.insert(0, str(PTP_ROOT))

from ptp_discovery.pref_builder_compiler import compile_preference_builder
from ptp_discovery.pref_builder_ir import ir_from_json


ARTIFACT_ROOT = (
    REPO_ROOT
    / "research"
    / "scheduling_large_scale"
    / "experiments"
    / "h5_jssp_anchor_geometry"
    / "artifacts"
)


@pytest.mark.parametrize("variant", ["usw_stratified_anchor", "asw_stratified_anchor"])
def test_h5_builder_keeps_exactly_15_instance_local_pairs_with_ties(variant: str) -> None:
    payload = json.loads((ARTIFACT_ROOT / variant / "best_builder.json").read_text())
    builder = compile_preference_builder(ir_from_json(payload["ir"]))

    objective = torch.arange(128, dtype=torch.float32).repeat(2, 1)
    objective[:, :16] = 0.0
    feature_cache = {
        "objective": objective,
        "log_prob": torch.zeros_like(objective),
        "rank": torch.arange(128, dtype=torch.float32).repeat(2, 1),
        "instance_obj_mad": torch.ones(2),
        "instance_regret_mean": torch.ones(2),
    }
    pref = builder.build_fn(feature_cache, {"alpha": 0.0})
    b_idx, winner_idx, loser_idx = pref.pair_idx

    assert pref.num_examples() == 30
    assert torch.bincount(b_idx, minlength=2).tolist() == [15, 15]
    assert torch.equal(winner_idx.view(2, 15), torch.zeros((2, 15), dtype=torch.long))
    assert torch.equal(
        loser_idx.view(2, 15),
        torch.arange(8, 128, 8, dtype=torch.long).repeat(2, 1),
    )
    if pref.weight is not None:
        assert pref.weight.shape == (30,)
        assert torch.isfinite(pref.weight).all()
