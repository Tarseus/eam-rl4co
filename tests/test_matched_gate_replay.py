from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest

from PTP.ptp_discovery import pref_loss_coevo_loop as loop


SOURCE_LOSSES = Path("PTP/seeds/tsp100_main_loss_init_gen0.jsonl")
GATE_REPLAY = Path("PTP/seeds/tsp100_main_loss_gen0_gate_replay.json")


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
                    "source_losses_path": str(SOURCE_LOSSES),
                    "source_generation": 0,
                    "strict_count": True,
                    "gate_replay_path": str(GATE_REPLAY),
                },
            }
        },
    )


def _entries(proposal: dict) -> tuple[dict, dict]:
    common = {
        "op_type": "JOINT_PAIR_MATCHED_INIT",
        "origin": "MATCHED_TSP_INIT",
        "joint_pair_id": proposal["joint_pair_id"],
        "source_loss_id": proposal["source_loss_id"],
        "source_loss_signature": proposal["source_loss_signature"],
        "source_loss_canonical_signature": proposal[
            "source_loss_canonical_signature"
        ],
        "matched_gate_replay": proposal["matched_gate_replay"],
    }
    g_entry = {
        **common,
        "id": f"g_{proposal['source_loss_id']}",
        "signature": loop._sig_pref_builder(loop._ref_builder_ir()),
        "ir": asdict(loop._ref_builder_ir()),
    }
    f_entry = {
        **common,
        "id": proposal["source_loss_id"],
        "signature": loop._sig_free_loss(proposal["loss_ir"]),
        "ir": asdict(proposal["loss_ir"]),
    }
    return g_entry, f_entry


def test_gate_replay_manifest_pins_original_order_and_runtime_losses() -> None:
    replay = loop._load_matched_joint_gate_replay(
        replay_path=str(GATE_REPLAY),
        seeded_rows=_source_rows(),
        expected_count=16,
    )

    assert [entry["source_loss_id"] for entry in replay["entries"]] == [
        "f000_015_8d035555",
        "f000_012_50dd0498",
        "f000_014_355129a9",
        "f000_013_c593f56c",
        "f000_001_4b362843",
        "f000_009_236322cf",
        "f000_005_dfa4a10a",
        "f000_011_cf248071",
        "f000_006_3c0966c5",
        "f000_000_e8e19b52",
        "f000_007_c17cc502",
        "f000_004_46117a24",
        "f000_010_98ad5566",
        "f000_002_d78865a2",
        "f000_008_a10ec70e",
        "f000_003_9fe5f4eb",
    ]
    eligible = [entry for entry in replay["entries"] if entry["hf_eligible"]]
    assert [
        (entry["pair_index"], entry["source_loss_id"])
        for entry in eligible
    ] == [
        (7, "f000_011_cf248071"),
        (8, "f000_006_3c0966c5"),
    ]
    assert [entry["runtime_loss_signature"] for entry in eligible] == [
        "005b37a74e716cb8c976c550f4bf3869bf3f6346",
        "e552a82f46765d463e769a0b0dfdc65de168e469",
    ]


def test_matched_proposals_keep_raw_population_but_attach_original_pair_order() -> None:
    proposals = _proposals()

    assert [proposal["source_loss_id"] for proposal in proposals] == [
        row["id"] for row in _source_rows()
    ]
    replay_order = sorted(
        proposals,
        key=lambda proposal: proposal["matched_gate_replay"]["pair_index"],
    )
    assert [proposal["source_loss_id"] for proposal in replay_order[:9]] == [
        "f000_015_8d035555",
        "f000_012_50dd0498",
        "f000_014_355129a9",
        "f000_013_c593f56c",
        "f000_001_4b362843",
        "f000_009_236322cf",
        "f000_005_dfa4a10a",
        "f000_011_cf248071",
        "f000_006_3c0966c5",
    ]


def test_gate_replay_schedules_only_original_hf_candidates_with_frozen_ir() -> None:
    proposals = {
        proposal["source_loss_id"]: proposal for proposal in _proposals()
    }

    g_entry, f_entry = _entries(proposals["f000_006_3c0966c5"])
    record = loop._build_matched_joint_gate_replay_record(
        generation=0,
        pair_index=8,
        g_entry=g_entry,
        f_entry=f_entry,
        eval_signature="hf1-test",
        joint_pair_id=g_entry["joint_pair_id"],
    )

    assert record["pair_ok"] is True
    assert record["matched_gate_replay"] is True
    assert record["joint_gate_repaired"] is False
    assert record["builder_gate_repaired"] is False
    assert loop._sig_free_loss(loop.free_loss_ir_from_json(record["f_ir"])) == (
        "e552a82f46765d463e769a0b0dfdc65de168e469"
    )

    g_rejected, f_rejected = _entries(proposals["f000_015_8d035555"])
    rejected = loop._build_matched_joint_gate_replay_record(
        generation=0,
        pair_index=0,
        g_entry=g_rejected,
        f_entry=f_rejected,
        eval_signature="hf1-test",
        joint_pair_id=g_rejected["joint_pair_id"],
    )
    assert rejected["pair_ok"] is False
    assert rejected["pair_reason"] == (
        "matched_replay_original_cheap_gate_failed"
    )


def test_gate_replay_worker_skips_live_sandbox_and_loss_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = next(
        proposal
        for proposal in _proposals()
        if proposal["source_loss_id"] == "f000_006_3c0966c5"
    )
    g_entry, f_entry = _entries(proposal)
    replay = proposal["matched_gate_replay"]
    f_entry["ir"] = replay["runtime_loss_ir"]

    def _unexpected_sandbox(*_args, **_kwargs):
        raise AssertionError("frozen generation-zero loss must not rerun sandbox")

    monkeypatch.setattr(loop, "_run_stage0_sandbox_gate", _unexpected_sandbox)
    record = loop._evaluate_pair_worker(
        {
            "generation": 0,
            "pair_index": 8,
            "g_entry": g_entry,
            "f_entry": f_entry,
            "cfg_yaml": {
                "cheap_gate_batch_size": 4,
                "cheap_gate_k": 8,
                "builder_max_pairs_per_instance": 4096,
                "stage0_sandbox_gate_enabled": True,
                "stage0_sandbox_gate_only_when_hf": False,
                "joint_gate_repair_enabled": True,
            },
            "device_str": "cpu",
            "operator_whitelist": [],
            "run_dir": None,
            "cheap_gate_on": False,
            "high_fidelity_on": False,
            "matched_joint_init": True,
            "matched_gate_replay": True,
            "matched_runtime_loss_signature": replay[
                "runtime_loss_signature"
            ],
            "eval_budget_signature": "matched-gate-replay-worker-test",
        }
    )

    assert record["pair_ok"] is True
    assert record["matched_gate_replay"] is True
    assert record["joint_gate_repaired"] is False
    assert record["f_id_before_repair"] is None
    assert record["f_id_after_repair"] is None


def test_gate_replay_manifest_rejects_runtime_ir_drift(tmp_path: Path) -> None:
    replay = json.loads(GATE_REPLAY.read_text(encoding="utf-8"))
    eligible = next(entry for entry in replay["entries"] if entry["hf_eligible"])
    eligible["runtime_loss_ir"]["name"] = "drifted_loss"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(replay), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime loss signature mismatch"):
        loop._load_matched_joint_gate_replay(
            replay_path=str(changed),
            seeded_rows=_source_rows(),
            expected_count=16,
        )
