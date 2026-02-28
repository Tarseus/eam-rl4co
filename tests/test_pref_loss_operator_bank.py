from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_pref_builder_ir_parses_optional_fields(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    from ptp_discovery.pref_builder_ir import ir_from_json

    ir = ir_from_json(
        {
            "name": "b1",
            "intuition": "demo",
            "hyperparams": {"geometry_family": "dense_all_pairs"},
            "operators_used": ["sampled_pairs", "clamp"],
            "implementation_hint": {"expects": ["objective", "log_prob"], "returns": "PrefBatch", "mode": "pairwise"},
            "code": "def generated_builder(feature_cache, extra):\n    return PrefBatch(mode='pairwise', pair_idx=(None,None,None), weight=None, meta={})\n",
        }
    )

    assert ir.hyperparams["geometry_family"] == "dense_all_pairs"
    assert ir.operators_used == ["sampled_pairs", "clamp"]

    ir2 = ir_from_json(
        {
            "name": "b2",
            "intuition": "demo",
            "hyperparams": "bad",
            "operators_used": {"foo": 1},
            "implementation_hint": {"expects": ["objective"], "returns": "PrefBatch", "mode": "pairwise"},
            "code": "def generated_builder(feature_cache, extra):\n    return PrefBatch(mode='pairwise', pair_idx=(None,None,None), weight=None, meta={})\n",
        }
    )
    assert ir2.hyperparams == {}
    assert ir2.operators_used == ["foo"]


def test_builder_new_prompt_builders_embed_parent_payload(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint
    import ptp_discovery.pref_builder_llm_ops as ops

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("BASE PROMPT", encoding="utf-8")
    parent = PreferenceBuilderIR(
        name="parent",
        intuition="demo",
        implementation_hint=PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise"),
        hyperparams={"geometry_family": "dense_all_pairs", "cap_family": "uncapped_full", "weight_family": "uniform_none", "constraint_family": "none"},
        operators_used=["all_pairs"],
        code="def generated_builder(feature_cache, extra):\n    return PrefBatch(mode='pairwise', pair_idx=(None,None,None), weight=None, meta={})\n",
    )

    paradigm_prompt, _ = ops.build_paradigm_shift_prompt(
        str(prompt_path),
        parents=[parent, parent],
        parents_fitness=[{"fitness": 1.0}, {"fitness": 2.0}],
        global_feedback={"unit": True},
    )
    assert "PARENTS_JSON" in paradigm_prompt
    assert "geometry_family" in paradigm_prompt
    assert "GLOBAL_FEEDBACK_JSON" in paradigm_prompt

    structure_prompt, _ = ops.build_structure_shift_prompt(
        str(prompt_path),
        parent=parent,
        parent_fitness={"fitness": 1.0},
        global_feedback={"unit": True},
    )
    assert "PARENT_JSON" in structure_prompt
    assert "operators_used" in structure_prompt

    constraint_prompt, _ = ops.build_constraint_inject_prompt(
        str(prompt_path),
        parent=parent,
        parent_fitness={"fitness": 1.0},
        global_feedback={"unit": True},
    )
    assert "constraint_family" in constraint_prompt


def test_operator_bank_expand_and_family_quota(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import random
    import ptp_discovery.pref_loss_coevo_loop as loop

    rng = random.Random(0)
    plan = loop._expand_operator_bank(
        {
            "operator_bank": {
                "init": [
                    {"name": "gen", "count": 2},
                    {"name": "PARADIGM_SHIFT", "count": 1},
                    {"name": "repair", "count": 9},
                ]
            }
        },
        0,
        rng,
        side="builder",
    )
    assert sorted(plan) == ["GEN", "GEN", "PARADIGM_SHIFT"]

    ranked = [
        {"id": "a1", "fitness": 0.1, "family_signature": "fam_a"},
        {"id": "a2", "fitness": 0.2, "family_signature": "fam_a"},
        {"id": "b1", "fitness": 0.3, "family_signature": "fam_b"},
        {"id": "u1", "fitness": 0.4, "family_signature": "unknown"},
    ]
    elites = loop._select_elites_with_family_quota(
        ranked,
        3,
        metric_mode="minimize",
        min_per_family=1,
        max_per_family=2,
        include_unknown=False,
    )
    assert [e["id"] for e in elites][:2] == ["a1", "b1"]
    assert "u1" not in [e["id"] for e in elites[:2]]


def test_operator_contract_gates(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint
    from ptp_discovery.free_loss_ir import FreeLossIR, FreeLossImplementationHint
    import ptp_discovery.pref_loss_coevo_loop as loop

    builder_parent = PreferenceBuilderIR(
        name="bp",
        intuition="parent",
        implementation_hint=PreferenceBuilderImplementationHint(expects=["objective"], returns="PrefBatch", mode="pairwise"),
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "uniform_none",
            "constraint_family": "none",
        },
        operators_used=["all_pairs"],
        code="def generated_builder(feature_cache, extra):\n    return PrefBatch(mode='pairwise', pair_idx=(None,None,None), weight=None, meta={})\n",
    )
    builder_child = PreferenceBuilderIR(
        name="bc",
        intuition="child",
        implementation_hint=builder_parent.implementation_hint,
        hyperparams={
            "geometry_family": "anchor_star",
            "cap_family": "anchor_single",
            "weight_family": "uniform_none",
            "constraint_family": "none",
        },
        operators_used=["anchor_best"],
        code=builder_parent.code,
    )
    ok, fail = loop._validate_builder_operator_contract(
        builder_child,
        "BUILDER_PARADIGM_SHIFT",
        [builder_parent, builder_parent],
    )
    assert ok is True
    bad_ok, bad_fail = loop._validate_builder_operator_contract(
        builder_parent,
        "BUILDER_STRUCTURE_SHIFT",
        [builder_parent],
    )
    assert bad_ok is False
    assert bad_fail["stage"] == "operator_contract"

    loss_parent = FreeLossIR(
        name="lp",
        intuition="parent",
        pseudocode="x",
        hyperparams={
            "paradigm_family": "pairwise_margin",
            "signal_family": "logprob_gap",
            "link_family": "logsigmoid",
            "agg_family": "mean",
            "constraint_family": "weight_optional",
        },
        operators_used=["logsigmoid"],
        implementation_hint=FreeLossImplementationHint(expects=["log_prob_w"], returns="scalar", mode="pairwise"),
        code="def generated_loss(batch, model_output, extra):\n    return batch['log_prob_w'].mean()\n",
        theoretical_basis="",
    )
    loss_child = FreeLossIR(
        name="lc",
        intuition="child",
        pseudocode="x",
        hyperparams={
            "paradigm_family": "pairwise_rank",
            "signal_family": "delta_rank",
            "link_family": "sigmoid",
            "agg_family": "mean",
            "constraint_family": "weight_optional",
        },
        operators_used=["sigmoid"],
        implementation_hint=loss_parent.implementation_hint,
        code=loss_parent.code,
        theoretical_basis="",
    )
    ok2, _ = loop._validate_loss_operator_contract(loss_child, "LOSS_PARADIGM_SHIFT", [loss_parent, loss_parent])
    assert ok2 is True
    bad_ok2, bad_fail2 = loop._validate_loss_operator_contract(loss_parent, "LOSS_CONSTRAINT_INJECT", [loss_parent])
    assert bad_ok2 is False
    assert bad_fail2["stage"] == "operator_contract"
