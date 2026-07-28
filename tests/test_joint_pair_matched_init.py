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
