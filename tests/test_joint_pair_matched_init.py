from __future__ import annotations

import json
import random
from dataclasses import asdict

import pytest

from PTP.ptp_discovery import pref_loss_coevo_loop as loop


def _loss_row(index: int, *, generation: int = 0) -> dict:
    ir = loop._ref_loss_ir()
    ir.name = f"matched_loss_{index}"
    return {
        "generation": generation,
        "index": index,
        "id": f"f{generation:03d}_{index:03d}",
        "signature": f"{index + 1:040x}",
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
