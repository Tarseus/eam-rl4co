from __future__ import annotations

from pathlib import Path


def test_stage3_fidelity_key_step_and_epoch(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    assert loop._stage3_fidelity_key({"f1_steps": 200, "hf_epochs": 0, "hf_instances_per_epoch": 0}) == "K200"
    assert (
        loop._stage3_fidelity_key({"f1_steps": 200, "hf_epochs": 1, "hf_instances_per_epoch": 500})
        == "epoch1_inst500"
    )


def test_resolve_stage3_baseline_mini_eval_path(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {"f1_steps": 200, "hf_epochs": 0, "hf_instances_per_epoch": 0}
    baseline_cfg = {"mini_eval_paths": {200: "k200.json", 1000: "k1000.json", "default": "fallback.json"}}
    assert loop._resolve_stage3_baseline_mini_eval_path(cfg, baseline_cfg) == "k200.json"

    cfg2 = {"f1_steps": 5000, "hf_epochs": 0, "hf_instances_per_epoch": 0}
    assert loop._resolve_stage3_baseline_mini_eval_path(cfg2, baseline_cfg) == "fallback.json"


def test_normalize_stage3_multifidelity_cfg_defaults(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = loop._normalize_stage3_multifidelity_cfg({"enabled": True})
    assert cfg["enabled"] is True
    assert isinstance(cfg["rounds"], list)
    assert len(cfg["rounds"]) >= 2
    assert cfg["rounds"][0].get("f1_steps") == 200
    assert cfg["rounds"][1].get("f1_steps") == 1000


def test_select_stage3_promotions_respects_improve_eps(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    records = [
        {"pair_ok": True, "g_id": "g1", "f_id": "f1", "score": -0.012},
        {"pair_ok": True, "g_id": "g2", "f_id": "f2", "score": -0.020},
        {"pair_ok": True, "g_id": "g3", "f_id": "f3", "score": 0.100},
    ]
    promoted = loop._select_stage3_promotions(
        records,
        promote_top_m=0,
        promote_if_better_than_incumbent=True,
        incumbent_ref_score=-0.015,
        metric_mode="minimize",
        improve_eps=0.004,
        always_include_pair=None,
    )
    # -0.020 beats -0.015 by >0.004; -0.012 does not.
    assert ("g2", "f2") in promoted
    assert ("g1", "f1") not in promoted

    promoted2 = loop._select_stage3_promotions(
        records,
        promote_top_m=1,
        promote_if_better_than_incumbent=False,
        incumbent_ref_score=-0.015,
        metric_mode="minimize",
        improve_eps=0.004,
        always_include_pair=("g3", "f3"),
    )
    # Always include + top-1 by score (minimize => most negative).
    assert promoted2[0] == ("g3", "f3")
    assert ("g2", "f2") in promoted2

