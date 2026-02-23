from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _good_builder_json(*, name: str = "good_builder") -> str:
    # Must match PreferenceBuilderIR JSON contract.
    code = (
        "def generated_builder(feature_cache, extra):\n"
        "    objective = feature_cache['objective']\n"
        "    mask = objective[:, :, None] < objective[:, None, :]\n"
        "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
        "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'all_pairs'})\n"
    )
    payload = {
        "name": name,
        "intuition": "test_builder",
        "implementation_hint": {"expects": ["objective", "log_prob"], "returns": "PrefBatch", "mode": "pairwise"},
        "code": code,
    }
    import json

    return json.dumps(payload)


def test_pref_builder_compile_and_run(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import torch
    from fitness.free_loss_fidelity import PrefBatch, extract_feature_cache

    from ptp_discovery.pref_builder_compiler import compile_preference_builder
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    ir = PreferenceBuilderIR(
        name="unit_builder",
        intuition="unit test",
        implementation_hint=PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise"),
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={})\n"
        ),
    )

    compiled = compile_preference_builder(ir, operator_whitelist=[])
    fc = extract_feature_cache(
        objective=torch.rand(4, 8),
        log_prob=torch.randn(4, 8),
    )
    out = compiled.build_fn(fc, {"seed": 0})

    assert isinstance(out, PrefBatch)
    assert out.mode == "pairwise"
    assert out.pair_idx is not None
    b, w, l = out.pair_idx
    assert b.ndim == w.ndim == l.ndim == 1
    assert b.numel() == w.numel() == l.numel()


def test_builder_repair_path(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    bad = PreferenceBuilderIR(
        name="bad_builder",
        intuition="bad",
        implementation_hint=PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise"),
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    # Invalid: pair_idx missing => runtime validation failure\n"
            "    return PrefBatch(mode='pairwise', pair_idx=None, weight=None, meta={'bad': True})\n"
        ),
    )

    ok, fail = loop.validate_builder_candidate(
        bad,
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
    assert isinstance(fail, dict)

    good = PreferenceBuilderIR(
        name="repaired_builder",
        intuition="good",
        implementation_hint=bad.implementation_hint,
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'fixed': True})\n"
        ),
    )

    def _fake_repair_with_meta(*args, **kwargs):
        return good, {"llm_op": "REPAIR", "prompt_path": "dummy", "prompt_sha1": "deadbeef"}

    monkeypatch.setattr(loop.builder_llm_ops, "repair_pref_builder_with_meta", _fake_repair_with_meta)
    monkeypatch.setattr(
        loop.builder_llm_ops,
        "m3_simplify_builder_with_meta",
        lambda *args, **kwargs: (bad, {"llm_op": "M3", "prompt_path": "dummy", "prompt_sha1": "deadbeef"}),
    )

    repaired, meta = loop._repair_builder_candidate_loop(
        bad,
        failure_report=fail,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
        },
        llm_prompts={"builder_m3": "dummy_m3", "builder_repair": "dummy_repair"},
        global_feedback={"unit": True},
        max_attempts=1,
        simplify_first=True,
    )

    assert repaired is not None
    ok2, _ = loop.validate_builder_candidate(
        repaired,
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
    assert ok2 is True
    assert isinstance(meta, dict)
