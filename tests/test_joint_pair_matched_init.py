from __future__ import annotations

import json
import random
from dataclasses import asdict

import pytest

from PTP.ptp_discovery import pref_loss_coevo_loop as loop


def _loss_row(index: int, *, generation: int = 0) -> dict:
    ir = loop._ref_loss_ir()
    ir.name = f"matched_loss_{index}"
    signature = loop._sig_free_loss(ir)
    return {
        "generation": generation,
        "index": index,
        "id": f"f{generation:03d}_{index:03d}",
        "signature": signature,
        "ir": asdict(ir),
        "prompt_sha1": f"prompt-{index}",
        "prompt_path": "source-prompt.txt",
        "llm_seed": 100 + index,
    }


def test_joint_pair_generation_zero_reuses_matched_tsp_losses(tmp_path, monkeypatch) -> None:
    source = tmp_path / "losses.jsonl"
    rows = [_loss_row(1), _loss_row(0), _loss_row(0, generation=1)]
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    def _unexpected_llm_call(**_kwargs):
        raise AssertionError("generation zero must not call the joint LLM")

    monkeypatch.setattr(
        loop.joint_pair_llm_ops,
        "generate_joint_pair_candidate_with_meta",
        _unexpected_llm_call,
    )
    proposals = loop._propose_joint_pairs_for_generation(
        generation=0,
        pop_pairs=2,
        rng=random.Random(1234),
        llm_cfg={
            "joint_pair": {
                "enabled": True,
                "init_seed": {
                    "enabled": True,
                    "source_losses_path": str(source),
                    "source_generation": 0,
                    "strict_count": True,
                },
            }
        },
    )

    assert [proposal["loss_ir"].name for proposal in proposals] == [
        "matched_loss_0",
        "matched_loss_1",
    ]
    assert all(proposal["builder_ir"].name == "ref_all_pairs_builder" for proposal in proposals)
    assert all(proposal["origin"] == "MATCHED_TSP_INIT" for proposal in proposals)
    assert [proposal["source_loss_id"] for proposal in proposals] == [
        "f000_000",
        "f000_001",
    ]


def test_joint_pair_matched_init_requires_exact_population_by_default(tmp_path) -> None:
    source = tmp_path / "losses.jsonl"
    source.write_text(json.dumps(_loss_row(0)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seed count differs"):
        loop._propose_joint_pairs_for_generation(
            generation=0,
            pop_pairs=2,
            rng=random.Random(1234),
            llm_cfg={
                "joint_pair": {
                    "enabled": True,
                    "init_seed": {
                        "enabled": True,
                        "source_losses_path": str(source),
                    },
                }
            },
        )


def test_hf_runtime_entry_preserves_matched_metadata_through_json() -> None:
    original = {
        "id": "g000_000_deadbeef",
        "op_type": "JOINT_PAIR_MATCHED_INIT",
        "origin": "MATCHED_TSP_INIT",
        "joint_pair_id": "jp000_000_matched_deadbeef",
        "source_loss_id": "f000_000_deadbeef",
        "source_loss_signature": "deadbeef",
        "ir": asdict(loop._ref_builder_ir()),
    }
    merged = loop._hf_entry_with_runtime_ir(
        original,
        entry_id=original["id"],
        runtime_ir=asdict(loop._ref_builder_ir()),
    )
    round_tripped = json.loads(json.dumps(merged))

    assert round_tripped["op_type"] == "JOINT_PAIR_MATCHED_INIT"
    assert round_tripped["origin"] == "MATCHED_TSP_INIT"
    assert round_tripped["joint_pair_id"] == original["joint_pair_id"]
    assert round_tripped["source_loss_id"] == original["source_loss_id"]
    loop._assert_matched_joint_init_builder(
        round_tripped,
        context="serialized HF task test",
    )


def test_matched_joint_population_requires_exact_loss_and_builder_signatures() -> None:
    g_entries = []
    f_entries = []
    for index in range(2):
        loss_ir = loop._ref_loss_ir()
        loss_ir.name = f"matched_loss_{index}"
        loss_signature = loop._sig_free_loss(loss_ir)
        joint_pair_id = f"jp000_{index:03d}_matched_{loss_signature[:8]}"
        common = {
            "op_type": "JOINT_PAIR_MATCHED_INIT",
            "origin": "MATCHED_TSP_INIT",
            "joint_pair_id": joint_pair_id,
            "source_loss_id": f"f000_{index:03d}_{loss_signature[:8]}",
            "source_loss_signature": loss_signature,
            "source_loss_canonical_signature": loss_signature,
        }
        g_entries.append(
            {
                **common,
                "id": f"g000_{index:03d}",
                "signature": loop._sig_pref_builder(loop._ref_builder_ir()),
                "ir": asdict(loop._ref_builder_ir()),
            }
        )
        f_entries.append(
            {
                **common,
                "id": f"f000_{index:03d}_{loss_signature[:8]}",
                "signature": loss_signature,
                "ir": asdict(loss_ir),
            }
        )

    manifest = loop._validate_matched_joint_population(
        g_entries=g_entries,
        f_entries=f_entries,
        expected_count=2,
    )

    assert manifest["status"] == "strict_match"
    assert manifest["expected_count"] == 2
    assert [row["source_loss_id"] for row in manifest["entries"]] == [
        f_entries[0]["source_loss_id"],
        f_entries[1]["source_loss_id"],
    ]

    changed = json.loads(json.dumps(g_entries))
    changed[1]["ir"] = asdict(loop._ref_builder_ir())
    changed[1]["ir"]["name"] = "not_the_reference_builder"
    with pytest.raises(RuntimeError, match="builder changed"):
        loop._validate_matched_joint_population(
            g_entries=changed,
            f_entries=f_entries,
            expected_count=2,
        )


def test_hf_worker_explicit_matched_flag_bypasses_builder_repair(monkeypatch) -> None:
    def _unexpected_builder_repair(*_args, **_kwargs):
        raise AssertionError("matched HF worker must never repair the reference builder")

    monkeypatch.setattr(loop, "_repair_builder_candidate_loop", _unexpected_builder_repair)
    g_entry = {
        "id": "g000_000_311f50c2",
        "op_type": "JOINT_PAIR_MATCHED_INIT",
        "origin": "MATCHED_TSP_INIT",
        "joint_pair_id": "jp000_000_matched_test",
        "ir": asdict(loop._ref_builder_ir()),
    }
    rec = loop._evaluate_pair_worker(
        {
            "generation": 0,
            "pair_index": 0,
            "g_entry": json.loads(json.dumps(g_entry)),
            "f_entry": {
                "id": "f000_000_test",
                "op_type": "JOINT_PAIR_MATCHED_INIT",
                "joint_pair_id": "jp000_000_matched_test",
                "ir": asdict(loop._ref_loss_ir()),
            },
            "matched_joint_init": True,
            "cfg_yaml": {
                "cheap_gate_batch_size": 4,
                "cheap_gate_k": 8,
                "builder_max_pairs_per_instance": 4096,
                "builder_min_instance_weight_cv": 0.10,
                "builder_min_instance_weight_cv_pass_rate": 0.75,
                "builder_gate_repair_enabled": True,
                "builder_gate_repair_max_attempts": 8,
            },
            "device_str": "cpu",
            "operator_whitelist": [],
            "run_dir": None,
            "cheap_gate_on": True,
            "high_fidelity_on": False,
            "eval_budget_signature": "matched-hf-task-test",
        }
    )

    assert rec["matched_joint_init"] is True
    assert rec["builder_gate_repaired"] is False
    assert rec["g_id_before_repair"] is None
    assert rec["g_id_after_repair"] is None


def test_joint_pair_runtime_preserves_matched_init_and_skips_llm(
    tmp_path,
    monkeypatch,
) -> None:
    import yaml

    source = tmp_path / "losses.jsonl"
    source.write_text(
        "\n".join(json.dumps(_loss_row(index)) for index in range(2)) + "\n",
        encoding="utf-8",
    )

    def _unexpected_llm_call(**_kwargs):
        raise AssertionError("runtime generation zero must use the matched init seed")

    monkeypatch.setattr(
        loop.joint_pair_llm_ops,
        "generate_joint_pair_candidate_with_meta",
        _unexpected_llm_call,
    )

    def _unexpected_builder_repair(*_args, **_kwargs):
        raise AssertionError("matched reference builder must bypass searched-builder repair")

    monkeypatch.setattr(loop, "_repair_builder_candidate_loop", _unexpected_builder_repair)

    cfg = {
        "seed": 1234,
        "output_root": str(tmp_path / "runs"),
        "search_mode": "joint_pair",
        "generations": 1,
        "pop_g": 2,
        "pop_f": 2,
        "elite_g": 1,
        "elite_f": 1,
        "pairing_budget_per_gen": 2,
        "cheap_gate_on": True,
        "high_fidelity_on": False,
        "eval_stages": {
            "stage0_gate": False,
            "stage1_proxy": False,
            "stage2_micro_unroll": False,
            "stage3_high_fidelity": False,
        },
        "backend": "rl4co",
        "env_name": "tsp",
        "policy_name": "pomo",
        "generator_params": {"num_loc": 20},
        "hf_epochs": 0,
        "hf_instances_per_epoch": 0,
        "train_problem_size": 20,
        "valid_problem_sizes": [20],
        "train_batch_size": 8,
        "validation_batch_size": 8,
        "num_validation_episodes": 8,
        "pomo_size": 8,
        "device": "cpu",
        "devices": ["cpu"],
        "mp": {"enabled": False},
        "llm_init_only": True,
        "builder_min_instance_weight_cv": 0.1,
        "builder_min_instance_weight_cv_pass_rate": 0.75,
        "builder_gate_repair_enabled": True,
        "population": {
            "n_candidates_pair": 2,
            "n_candidates_loss": 2,
            "n_candidates_builder": 2,
            "keep_top_k": 1,
        },
        "joint_pair_llm": {
            "enabled": True,
            "offline_mode": False,
            "n_candidates": 2,
            "init_seed": {
                "enabled": True,
                "source_losses_path": str(source),
                "source_generation": 0,
                "strict_count": True,
            },
        },
        "builder_llm": {"enabled": False},
        "loss_llm": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loop.run_pref_loss_coevo(str(cfg_path))

    run_dir = sorted((tmp_path / "runs").iterdir())[-1]
    loss_rows = [
        json.loads(line)
        for line in (run_dir / "losses.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(loss_rows) == 2
