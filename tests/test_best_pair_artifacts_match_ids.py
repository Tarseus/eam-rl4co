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

