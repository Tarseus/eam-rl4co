from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from PTP.ptp_discovery import pref_loss_coevo_loop as loop


SOURCE_LOSSES = Path("PTP/seeds/ffsp100_main_loss_init_gen0.jsonl")
GATE_REPLAY = Path("PTP/seeds/ffsp100_main_loss_gen0_gate_replay.json")
CONFIG = Path(
    "PTP/configs/experiment/pref_loss_coevo/"
    "no_two_stage_ffsp100_coevo_matched_init_hf1.yaml"
)


def _source_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in SOURCE_LOSSES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _proposals() -> list[dict]:
    return loop._propose_joint_pairs_for_generation(
        generation=0,
        pop_pairs=16,
        rng=random.Random(1234),
        llm_cfg={
            "joint_pair": {
                "enabled": True,
                "init_seed": {
                    "enabled": True,
                    "origin_label": "MATCHED_FFSP_INIT",
                    "source_losses_path": str(SOURCE_LOSSES),
                    "source_generation": 0,
                    "strict_count": True,
                    "gate_replay_path": str(GATE_REPLAY),
                },
            }
        },
    )


def test_ffsp100_seed_comes_from_authoritative_source_artifacts() -> None:
    replay = json.loads(GATE_REPLAY.read_text(encoding="utf-8"))
    assert replay["source_artifacts"]["losses_sha256"] == (
        "cf07bda06a5b812748ab738099ee90659c145fb765e2c63b0fb10d5ce741af60"
    )
    assert replay["source_artifacts"]["pairs_sha256"] == (
        "8574ad21bba8708a4a765a3c21c5ac25e3e8d92ed24822e54b16cd35c8cc44db"
    )
    assert [row["id"] for row in _source_rows()] == [
        "f000_000_943aeba8",
        "f000_001_2a9aa97a",
        "f000_002_492f2c81",
        "f000_003_9e96a1c3",
        "f000_004_a9e3fbe7",
        "f000_005_c2f26c44",
        "f000_006_8e095760",
        "f000_007_be370055",
        "f000_008_ca885fc4",
        "f000_009_0001ab92",
        "f000_010_8c1f3fbe",
        "f000_011_8502969a",
        "f000_012_372ae734",
        "f000_013_87b4c8d2",
        "f000_014_26963073",
        "f000_015_ffab0ed9",
    ]


def test_ffsp100_gate_replay_pins_pair_order_and_five_hf_candidates() -> None:
    replay = loop._load_matched_joint_gate_replay(
        replay_path=str(GATE_REPLAY),
        seeded_rows=_source_rows(),
        expected_count=16,
    )
    assert [entry["source_loss_id"] for entry in replay["entries"]] == [
        "f000_015_ffab0ed9",
        "f000_012_372ae734",
        "f000_014_26963073",
        "f000_013_87b4c8d2",
        "f000_001_2a9aa97a",
        "f000_009_0001ab92",
        "f000_005_c2f26c44",
        "f000_011_8502969a",
        "f000_006_8e095760",
        "f000_000_943aeba8",
        "f000_007_be370055",
        "f000_004_a9e3fbe7",
        "f000_010_8c1f3fbe",
        "f000_002_492f2c81",
        "f000_008_ca885fc4",
        "f000_003_9e96a1c3",
    ]
    eligible = [entry for entry in replay["entries"] if entry["hf_eligible"]]
    assert [
        (entry["pair_index"], entry["source_loss_id"], entry["runtime_loss_signature"])
        for entry in eligible
    ] == [
        (1, "f000_012_372ae734", "372ae734dbeab4702e548ec8d5d00c424152572a"),
        (2, "f000_014_26963073", "26963073976d00c833bd1ab0290551e462d2fd9b"),
        (8, "f000_006_8e095760", "8e0957608d28a33ec331a4e7a2446350d7f617c2"),
        (11, "f000_004_a9e3fbe7", "a9e3fbe715bcdfee5f8c5a51c1774e2f1f14ebca"),
        (12, "f000_010_8c1f3fbe", "8c1f3fbeabab4c7e6c6e480a66f76d6502914b27"),
    ]


def test_ffsp100_proposals_use_reference_builder_and_problem_origin() -> None:
    proposals = _proposals()
    assert all(proposal["builder_ir"].name == "ref_all_pairs_builder" for proposal in proposals)
    assert all(proposal["origin"] == "MATCHED_FFSP_INIT" for proposal in proposals)
    assert all(proposal["op_type"] == "JOINT_PAIR_MATCHED_INIT" for proposal in proposals)
    replay_order = sorted(
        proposals,
        key=lambda proposal: proposal["matched_gate_replay"]["pair_index"],
    )
    assert replay_order[1]["source_loss_id"] == "f000_012_372ae734"
    assert replay_order[12]["source_loss_id"] == "f000_010_8c1f3fbe"


def test_ffsp100_hf1_config_is_checkpoint_only_and_strict_matched_init() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["search_mode"] == "joint_pair"
    assert cfg["budgets"]["generations"] == 20
    assert cfg["hf_epochs"] == 1
    assert cfg["hf_instances_per_epoch"] == 1000
    assert cfg["scratch_hf_epochs"] == 0
    assert cfg["warmstart_hf_epochs"] == 1
    assert cfg["baseline"]["include_scratch"] is False
    assert cfg["baseline"]["checkpoint_epoch"] == 100
    assert cfg["joint_pair_llm"]["init_seed"] == {
        "enabled": True,
        "origin_label": "MATCHED_FFSP_INIT",
        "source_losses_path": str(SOURCE_LOSSES).replace("\\", "/"),
        "source_generation": 0,
        "strict_count": True,
        "gate_replay_path": str(GATE_REPLAY).replace("\\", "/"),
    }