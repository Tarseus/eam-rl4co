from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_builder_intuition_nonempty_gate(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    ir = PreferenceBuilderIR(
        name="no_intuition_builder",
        intuition="",
        implementation_hint=PreferenceBuilderImplementationHint(
            expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise"
        ),
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, w_idx, l_idx = mask.nonzero(as_tuple=True)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, w_idx, l_idx), weight=None, meta={})\n"
        ),
    )

    ok, failure = loop.validate_builder_candidate(
        ir,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
        },
    )
    assert ok is False
    assert isinstance(failure, dict)
    assert failure.get("stage") == "interpretability"


def test_free_loss_intuition_nonempty_static_gate(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    from ptp_discovery.free_loss_gates import run_static_gates
    from ptp_discovery.free_loss_ir import FreeLossIR, FreeLossImplementationHint

    ir = FreeLossIR(
        name="no_intuition_loss",
        intuition="",
        pseudocode="loss = mean(log_prob_w - log_prob_l)",
        hyperparams={},
        operators_used=["logsigmoid"],
        implementation_hint=FreeLossImplementationHint(
            expects=["log_prob_w", "log_prob_l", "delta_z", "weight"],
            returns="scalar",
            mode="pairwise",
        ),
        code=(
            "def generated_loss(batch, model_output, extra):\n"
            "    x = batch['log_prob_w'] - batch['log_prob_l']\n"
            "    return x.mean()\n"
        ),
        theoretical_basis="",
    )

    res = run_static_gates(ir, operator_whitelist=[])
    assert res.ok is False
    assert "Missing intuition" in str(res.reason)

