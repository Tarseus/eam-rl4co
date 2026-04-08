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

    reweight_prompt, _ = ops.build_structure_shift_prompt(
        str(prompt_path),
        parent=parent,
        parent_fitness={"fitness": 1.0},
        global_feedback={
            "builder_search_space": {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
                "seed_weight_families": ["gap_linear", "gap_sigmoid"],
                "allow_freeform_weight_family": True,
            },
            "llm_call": {"search_operator": "STRUCTURE_SHIFT"},
        },
    )
    assert "REWEIGHT_ONLY_OPERATOR_SEMANTICS" in reweight_prompt
    assert "Preserve `weight_family` and `constraint_family`" in reweight_prompt


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


def test_select_resident_population_respects_fixed_size_and_family_diversity(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    ranked = [
        {"id": "a1", "fitness": 0.1, "family_signature": "fam_a"},
        {"id": "a2", "fitness": 0.2, "family_signature": "fam_a"},
        {"id": "b1", "fitness": 0.3, "family_signature": "fam_b"},
    ]

    resident_plain = loop._select_resident_population(
        ranked,
        2,
        metric_mode="minimize",
        family_diversity_cfg={"enabled": False},
    )
    assert [e["id"] for e in resident_plain] == ["a1", "a2"]

    resident_diverse = loop._select_resident_population(
        ranked,
        2,
        metric_mode="minimize",
        family_diversity_cfg={
            "enabled": True,
            "min_per_family": 1,
            "elite_max_per_family": 1,
            "include_unknown": False,
        },
    )
    assert [e["id"] for e in resident_diverse] == ["a1", "b1"]


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


def test_reweight_only_builder_operator_contracts(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint
    import ptp_discovery.pref_loss_coevo_loop as loop

    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    parent = PreferenceBuilderIR(
        name="parent_gap_linear",
        intuition="parent",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "gap_linear"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = gap\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs', 'weight_family': 'gap_linear'})\n"
        ),
    )
    paradigm_child = PreferenceBuilderIR(
        name="child_gap_sigmoid",
        intuition="child",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_sigmoid",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "sigmoid"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = ops.sigmoid(gap)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs', 'weight_family': 'gap_sigmoid'})\n"
        ),
    )
    ok, fail = loop._validate_builder_operator_contract(
        paradigm_child,
        "BUILDER_PARADIGM_SHIFT",
        [parent, parent],
        search_space_cfg={"enabled": True, "mode": "reweight_only", "fixed_pair_builder": "all_pairs"},
    )
    assert ok is True
    assert fail == {}

    bad_structure = PreferenceBuilderIR(
        name="bad_structure",
        intuition="bad",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_sigmoid",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "sigmoid"],
        code=paradigm_child.code,
    )
    ok2, fail2 = loop._validate_builder_operator_contract(
        bad_structure,
        "BUILDER_STRUCTURE_SHIFT",
        [parent],
        search_space_cfg={"enabled": True, "mode": "reweight_only", "fixed_pair_builder": "all_pairs"},
    )
    assert ok2 is False
    assert fail2["reason"] == "weight_family_not_preserved"

    constraint_child = PreferenceBuilderIR(
        name="constraint_child",
        intuition="constraint child",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "gap_linear_clamped_safe",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = ops.clamp(gap, 0.0, 10.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs', 'weight_family': 'gap_linear'})\n"
        ),
    )
    ok3, fail3 = loop._validate_builder_operator_contract(
        constraint_child,
        "BUILDER_CONSTRAINT_INJECT",
        [parent],
        search_space_cfg={"enabled": True, "mode": "reweight_only", "fixed_pair_builder": "all_pairs"},
    )
    assert ok3 is True
    assert fail3 == {}
