from __future__ import annotations

from pathlib import Path


def test_best_pair_artifact_entry_prefers_pair_ir_over_elite(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    best_pair = {
        "g_id": "g_best",
        "f_id": "f_best",
        "g_ir": {"code": "def generated_builder(x, extra):\n    return x\n"},
        "f_ir": {"code": "def generated_loss(batch, model_output, extra):\n    return 0.0\n"},
    }
    # Candidate maps contain mismatching elites; helper should prefer best_pair embedded IR.
    g_map = {"g_elite": {"id": "g_elite", "ir": {"code": "elite"}}}
    f_map = {"f_elite": {"id": "f_elite", "ir": {"code": "elite"}}}

    g_entry = loop._best_pair_artifact_entry(
        cid="g_best",
        best_pair=best_pair,
        cid_key="g_id",
        ir_key="g_ir",
        candidate_map=g_map,
        compiled_map={},
        ref_ir_fn=None,
    )
    f_entry = loop._best_pair_artifact_entry(
        cid="f_best",
        best_pair=best_pair,
        cid_key="f_id",
        ir_key="f_ir",
        candidate_map=f_map,
        compiled_map={},
        ref_ir_fn=None,
    )

    assert g_entry is not None and g_entry["id"] == "g_best"
    assert f_entry is not None and f_entry["id"] == "f_best"


def test_best_pair_eval_metadata_exposes_stage3_result(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    best_pair = {
        "generation": 9,
        "phase": "loss",
        "stage": "high_fidelity",
        "stage_final": "high_fidelity",
        "score": -0.03,
        "final_score": -0.03,
        "stages_enabled": {
            "stage0_gate": True,
            "stage1_proxy": False,
            "stage2_micro_unroll": False,
            "stage3_high_fidelity": True,
        },
        "stages_ran": ["stage0_gate", "stage3_high_fidelity"],
        "stages_skipped": {
            "stage1_proxy": "disabled",
            "stage2_micro_unroll": "disabled",
        },
        "pair_ok": True,
        "pair_reason": "ok_stage3_offline_minitrain",
        "compare_target": "incumbent",
        "metric_mode": "minimize",
        "improve_eps": 0.0,
        "reference_score": -0.01,
        "better_than_incumbent": True,
    }

    meta = loop._best_pair_eval_metadata(best_pair)

    assert meta["stage_final"] == "high_fidelity"
    assert meta["final_score"] == -0.03
    assert meta["stages_ran"] == ["stage0_gate", "stage3_high_fidelity"]
    assert meta["stages_skipped"]["stage1_proxy"] == "disabled"
    assert meta["best_pair_generation"] == 9
    assert meta["best_pair_phase"] == "loss"


def test_pair_score_history_accumulates_repeated_evals(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    history_map = {}
    rec1 = {
        "g_id": "g_best",
        "f_id": "f_best",
        "generation": 1,
        "pair_index": 2,
        "phase": "loss",
        "stage": "high_fidelity",
        "stage_final": "high_fidelity",
        "final_score": -0.01,
        "eval_budget_signature": "sigA",
    }
    rec2 = {
        "g_id": "g_best",
        "f_id": "f_best",
        "generation": 3,
        "pair_index": 4,
        "phase": "builder",
        "stage": "high_fidelity",
        "stage_final": "high_fidelity",
        "final_score": -0.03,
        "eval_budget_signature": "sigA",
    }

    loop._append_pair_score_history(history_map, rec1)
    loop._append_pair_score_history(history_map, rec2)

    history = history_map[loop._pair_history_key("g_best", "f_best")]
    assert [item["score"] for item in history] == [-0.01, -0.03]
    assert history[0]["generation"] == 1
    assert history[1]["generation"] == 3


def test_score_history_summary_reports_latest_and_mean(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    summary = loop._score_history_summary(
        [
            {"score": -0.01},
            {"score": -0.03},
            {"score": -0.02},
        ]
    )

    assert summary["count"] == 3
    assert summary["best"] == -0.03
    assert summary["worst"] == -0.01
    assert summary["latest"] == -0.02
    assert summary["mean"] == (-0.01 - 0.03 - 0.02) / 3


def test_resolve_best_pair_record_prefers_hf_record_over_later_gate_refresh(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    best_so_far = {
        "score": -0.031,
        "builder_id": "g_best",
        "loss_id": "f_best",
        "stage_final": "high_fidelity",
        "generation": 4,
        "phase": "loss",
    }
    pair_records = [
        {
            "generation": 6,
            "pair_index": 4,
            "g_id": "g_best",
            "f_id": "f_best",
            "stage": "gate",
            "stage_final": "none",
            "score": 0.0,
            "final_score": None,
            "phase": "loss",
        }
    ]
    pair_cache_records = [
        {
            "generation": 4,
            "pair_index": 4,
            "g_id": "g_best",
            "f_id": "f_best",
            "stage": "high_fidelity",
            "stage_final": "high_fidelity",
            "score": -0.031,
            "final_score": -0.031,
            "phase": "loss",
            "fitness": {"delta_mean": -0.031},
            "g_ir": {"code": "def generated_builder(x, extra):\n    return x\n"},
            "f_ir": {"code": "def generated_loss(batch, model_output, extra):\n    return 0.0\n"},
        }
    ]

    resolved = loop._resolve_best_pair_record(
        best_so_far=best_so_far,
        pair_records=pair_records,
        pair_cache_records=pair_cache_records,
        metric_mode="minimize",
    )

    assert resolved is not None
    assert resolved["generation"] == 4
    assert resolved["pair_index"] == 4
    assert resolved["stage_final"] == "high_fidelity"
    assert resolved["final_score"] == -0.031
