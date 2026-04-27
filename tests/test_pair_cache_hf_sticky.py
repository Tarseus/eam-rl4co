from __future__ import annotations

from pathlib import Path


def test_merge_pair_records_keeps_high_fidelity_terminal(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    from fitness.pref_loss_fidelity import _merge_pair_records

    existing = {
        "stage": "high_fidelity",
        "pair_ok": True,
        "pair_reason": "ok_stage3_offline_minitrain",
        "fitness": {"hf_like_score": -0.03},
        "score": -0.03,
        "generation": 9,
        "pair_index": 18,
        "proxy_metrics": {"cheap_effective_grad_ratio": 1.0},
    }
    incoming = {
        "stage": "gate",
        "pair_ok": True,
        "pair_reason": "ok_gate_only",
        "score": 0.5,
        "generation": 10,
        "pair_index": 3,
        "seed_signature": "seed-a",
        "proxy_metrics": {"cheap_effective_grad_ratio": 0.9},
    }

    merged = _merge_pair_records(existing, incoming)

    assert merged["stage"] == "high_fidelity"
    assert merged["score"] == -0.03
    assert merged["generation"] == 9
    assert merged["pair_index"] == 18
    assert merged["pair_reason"] == "ok_stage3_offline_minitrain"
    assert merged["fitness"] == {"hf_like_score": -0.03}
