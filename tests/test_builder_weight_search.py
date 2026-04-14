from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _patch_fast_proxy(monkeypatch, loop_module):
    def _fast_rollout_feature_cache(*args, **kwargs):  # noqa: ANN001
        batch_size = int(kwargs.get("batch_size", 8) or 8)
        cfg = kwargs.get("cfg")
        k = 16
        try:
            pomo_size = getattr(cfg, "pomo_size", None)
            if pomo_size is not None:
                k = int(pomo_size)
        except Exception:  # noqa: BLE001
            k = 16
        return loop_module._dummy_feature_cache(batch_size=batch_size, k=k, variant="visible")

    def _fast_proxy_metrics_for_pair_on_batch(*args, **kwargs):  # noqa: ANN001
        pref_batch = kwargs.get("pref_batch")
        pair_count = int(getattr(pref_batch, "num_examples", lambda: 0)())
        return {
            "loss": 1.0,
            "loss_swap": None,
            "effective_grad_ratio": 0.5,
            "grad_w_pass_rate": 1.0,
            "grad_l_pass_rate": 1.0,
            "swap_ok": True,
            "ess_ratio": 0.5,
            "pair_count": pair_count,
            "joint_ok": True,
            "joint_reason": "ok",
            "joint_trace": {"failed_gate": None, "observed": {"loss": 1.0, "effective_grad_ratio": 0.5}},
        }

    monkeypatch.setattr(loop_module, "build_or_get_rollout_feature_cache", _fast_rollout_feature_cache)
    monkeypatch.setattr(loop_module, "proxy_metrics_for_pair_on_batch", _fast_proxy_metrics_for_pair_on_batch)


def test_validate_builder_reweight_only_rejects_pair_structure_change(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    weighted_all_pairs = PreferenceBuilderIR(
        name="weighted_all_pairs",
        intuition="keep all pairs and only reweight by objective gap",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "fixed_pair_reweight_only",
        },
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    denom = torch.zeros(int(objective.shape[0]), dtype=objective.dtype, device=objective.device)\n"
            "    denom.index_add_(0, b_idx, gap)\n"
            "    counts = torch.bincount(b_idx.to(dtype=torch.int64), minlength=int(objective.shape[0])).to(dtype=objective.dtype)\n"
            "    weight = gap / denom[b_idx].clamp_min(1e-6)\n"
            "    weight = weight * counts[b_idx].clamp_min(1.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )
    ok, fail = loop.validate_builder_candidate(
        weighted_all_pairs,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
            "search_space": {"enabled": True, "mode": "reweight_only", "fixed_pair_builder": "all_pairs"},
        },
    )
    assert ok is True
    assert fail == {}

    changed_pairs = PreferenceBuilderIR(
        name="anchor_best_weighted",
        intuition="changes pair topology so it should be rejected",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "anchor_star",
            "cap_family": "anchor_single",
            "weight_family": "gap_linear",
            "constraint_family": "fixed_pair_reweight_only",
        },
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    best = objective.argmin(dim=1, keepdim=True)\n"
            "    all_idx = torch.arange(objective.shape[1], device=objective.device)[None, :].expand_as(objective)\n"
            "    b_idx = torch.arange(objective.shape[0], device=objective.device)[:, None].expand_as(all_idx)\n"
            "    winner_idx = best.expand_as(all_idx)\n"
            "    loser_idx = all_idx\n"
            "    mask = winner_idx != loser_idx\n"
            "    b = b_idx[mask]\n"
            "    w = winner_idx[mask]\n"
            "    l = loser_idx[mask]\n"
            "    gap = objective[b, l] - objective[b, w]\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b, w, l), weight=gap, meta={'builder': 'anchor_best'})\n"
        ),
    )
    ok2, fail2 = loop.validate_builder_candidate(
        changed_pairs,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
            "search_space": {"enabled": True, "mode": "reweight_only", "fixed_pair_builder": "all_pairs"},
        },
    )
    assert ok2 is False
    assert fail2["reason"] == "pair_structure_changed_under_reweight_only"


def test_builder_only_prefers_imported_loss(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import yaml
    import ptp_discovery.pref_loss_coevo_loop as loop

    _patch_fast_proxy(monkeypatch, loop)

    loss_path = tmp_path / "best_loss.json"
    loss_path.write_text(
        json.dumps(
            {
                "id": "f_best",
                "score": -0.123,
                "ir": {
                    "name": "seed_loss",
                    "intuition": "seed",
                    "pseudocode": "loss = -logsigmoid(lpw-lpl)",
                    "hyperparams": {"scale": 1.0},
                    "operators_used": ["logsigmoid"],
                    "implementation_hint": {
                        "expects": ["log_prob_w", "log_prob_l", "weight"],
                        "returns": "scalar",
                        "mode": "pairwise",
                    },
                    "code": (
                        "def generated_loss(batch, model_output, extra):\n"
                        "    x = batch['log_prob_w'] - batch['log_prob_l']\n"
                        "    return (-ops.logsigmoid(x)).mean()\n"
                    ),
                    "theoretical_basis": "",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = {
        "seed": 0,
        "output_root": str(tmp_path / "runs"),
        "search_mode": "builder_only",
        "generations": 1,
        "pop_g": 4,
        "pop_f": 4,
        "elite_g": 2,
        "elite_f": 2,
        "pairing_budget_per_gen": 4,
        "cheap_gate_on": True,
        "high_fidelity_on": False,
        "backend": "rl4co",
        "env_name": "tsp",
        "policy_name": "pomo",
        "generator_params": {"num_loc": 20},
        "hf_epochs": 0,
        "hf_instances_per_epoch": 0,
        "train_problem_size": 20,
        "valid_problem_sizes": [20],
        "train_batch_size": 8,
        "pomo_size": 16,
        "device": "cpu",
        "proxy_problem_size": 20,
        "proxy_batch_size": 8,
        "proxy_batches": 1,
        "seed_with_po4cops_default": False,
        "builder_min_coverage": 0.0,
        "loss_transfer_seed": {
            "enabled": True,
            "source_loss_path": str(loss_path),
        },
        "builder_search_space": {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
        },
        "builder_llm": {"enabled": False},
        "loss_llm": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loop.run_pref_loss_coevo(str(cfg_path))

    run_dir = sorted((tmp_path / "runs").iterdir())[-1]
    pair_lines = [json.loads(line) for line in (run_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert pair_lines, "expected at least one evaluated pair"
    assert all(str(rec.get("f_id", "")).startswith("fseed_000_") for rec in pair_lines)


def test_reweight_only_freeform_prompt_and_seed_pool(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_builder_llm_ops as builder_ops
    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "seed_weight_families": ["gap_linear", "gap_sigmoid"],
            "allow_uniform_none": False,
            "allow_freeform_weight_family": True,
        }
    )

    assert cfg["allow_freeform_weight_family"] is True
    assert cfg["allowed_weight_families"] == []
    assert cfg["seed_weight_families"] == ["gap_linear", "gap_sigmoid"]

    prompt_path = _repo_root() / "PTP" / "prompts" / "pref_builder_generation.txt"
    prompt, _ = builder_ops.build_generation_prompt(
        str(prompt_path),
        global_feedback={"builder_search_space": cfg},
    )
    assert "Seed weight_family values for the handcrafted initial pool" in prompt
    assert "Allowed weight_family values" not in prompt

    rng = random.Random(0)
    rng._pref_builder_search_space_cfg = cfg  # type: ignore[attr-defined]
    pool = loop._make_builtin_builder_irs(rng, 8)
    observed = {str(ir.hyperparams.get("weight_family")) for ir in pool}
    assert observed <= {"gap_linear", "gap_sigmoid"}


def test_reweight_only_none_rejects_instance_weight_normalization(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    normalized_builder = PreferenceBuilderIR(
        name="all_pairs_gap_linear_normed",
        intuition="keeps all pairs but performs instance-level weight normalization",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "clamped_instance_norm",
        },
        operators_used=["all_pairs", "gap_linear", "clamp", "normalize"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    batch_size = int(objective.shape[0])\n"
            "    counts = torch.bincount(b_idx.to(torch.int64), minlength=batch_size).to(dtype=objective.dtype)\n"
            "    mean_raw = torch.zeros(batch_size, dtype=objective.dtype, device=objective.device)\n"
            "    mean_raw.index_add_(0, b_idx, gap)\n"
            "    mean_raw = mean_raw / counts.clamp_min(1.0)\n"
            "    weight = gap / mean_raw[b_idx].clamp_min(1e-6)\n"
            "    weight = weight.clamp(0.25, 4.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )

    ok, fail = loop.validate_builder_candidate(
        normalized_builder,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
            "search_space": {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
                "pair_weight_normalization": "none",
            },
        },
    )
    assert ok is False
    assert fail["reason"] == "instance_weight_normalization_forbidden"


def test_feature_cache_exposes_instance_stats(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    feature_cache = loop._dummy_feature_cache(batch_size=3, k=7, variant="visible")
    for key in (
        "instance_num_rollouts",
        "instance_obj_mean",
        "instance_obj_std",
        "instance_obj_mad",
        "instance_obj_range",
        "instance_log_prob_mean",
        "instance_log_prob_std",
        "instance_regret_mean",
        "instance_regret_std",
    ):
        assert key in feature_cache
        assert tuple(feature_cache[key].shape) == (3,)
        assert torch.isfinite(feature_cache[key]).all().item()


def test_reweight_only_require_instance_stats_rejects_pair_only_builder(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    pair_only_builder = PreferenceBuilderIR(
        name="all_pairs_gap_only",
        intuition="pair-level gap weighting only",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "clamp_only",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = gap.clamp(0.25, 4.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )

    ok, fail = loop.validate_builder_candidate(
        pair_only_builder,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
            "search_space": {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
                "pair_weight_normalization": "none",
                "require_instance_stats": True,
                "instance_stat_keys": ["instance_obj_std", "instance_log_prob_std"],
            },
        },
    )
    assert ok is False
    assert fail["reason"] == "instance_stats_required"


def test_reweight_only_require_instance_conditioning_rejects_unused_instance_stat(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    impl = PreferenceBuilderImplementationHint(
        expects=["objective", "log_prob", "instance_obj_std"],
        returns="PrefBatch",
        mode="pairwise",
    )
    stat_but_not_conditioned = PreferenceBuilderIR(
        name="all_pairs_gap_with_unused_instance_stat",
        intuition="reads an instance stat but never uses it in the weighting path",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "clamp_only",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    instance_obj_std = feature_cache['instance_obj_std']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    _unused = instance_obj_std[b_idx]\n"
            "    weight = gap.clamp(0.25, 4.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )

    ok, fail = loop.validate_builder_candidate(
        stat_but_not_conditioned,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
            "search_space": {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
                "pair_weight_normalization": "none",
                "require_instance_stats": True,
                "require_instance_conditioning": True,
                "instance_stat_keys": ["instance_obj_std"],
            },
        },
    )
    assert ok is False
    assert fail["reason"] == "instance_conditioning_required"


def test_reweight_only_min_instance_weight_cv_rejects_flat_within_instance_weighting(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    impl = PreferenceBuilderImplementationHint(
        expects=["objective", "log_prob", "instance_obj_std"],
        returns="PrefBatch",
        mode="pairwise",
    )
    flat_within_instance_builder = PreferenceBuilderIR(
        name="all_pairs_instance_scalar_only",
        intuition="mentions gap and instance stats but leaves weights flat within each instance",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "clamp_only",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    instance_obj_std = feature_cache['instance_obj_std']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = gap * 0.0 + instance_obj_std[b_idx].clamp(0.25, 4.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )

    ok, fail = loop.validate_builder_candidate(
        flat_within_instance_builder,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
            "min_instance_weight_cv": 0.05,
            "min_instance_weight_cv_pass_rate": 1.0,
            "search_space": {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
                "pair_weight_normalization": "none",
                "require_instance_stats": True,
                "require_instance_conditioning": True,
                "instance_stat_keys": ["instance_obj_std"],
            },
        },
    )
    assert ok is False
    assert fail["reason"] == "instance_weight_cv_too_low"


def test_reweight_only_prompt_covers_m3_and_repair_none_mode(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_builder_llm_ops as builder_ops
    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "pair_weight_normalization": "none",
            "seed_weight_families": ["gap_linear"],
            "allow_uniform_none": False,
        }
    )
    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    parent = PreferenceBuilderIR(
        name="parent",
        intuition="seed",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = gap.clamp(0.25, 4.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )

    repair_prompt, _ = builder_ops.build_repair_prompt(
        str(_repo_root() / "PTP" / "prompts" / "pref_builder_repair.txt"),
        failed_ir=parent,
        failure_reason={"reason": "sandbox_gate_failed"},
        global_feedback={
            "builder_search_space": cfg,
            "llm_call": {"op_type": "REPAIR", "search_operator": "REPAIR"},
        },
    )
    m3_prompt, _ = builder_ops.build_m3_prompt(
        str(_repo_root() / "PTP" / "prompts" / "pref_builder_m3.txt"),
        candidate=parent,
        failure_reason={"reason": "sandbox_gate_failed"},
        global_feedback={
            "builder_search_space": cfg,
            "llm_call": {"op_type": "M3", "search_operator": "M3"},
        },
    )

    assert "do not repair by adding per-instance weight mean/sum normalization" in repair_prompt.lower()
    assert "do not reintroduce per-instance weight mean/sum normalization" in m3_prompt.lower()


def test_reweight_only_m2_prompt_preserves_weighting_families(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_builder_llm_ops as builder_ops
    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "pair_weight_normalization": "none",
            "seed_weight_families": ["gap_linear"],
            "allow_uniform_none": False,
        }
    )
    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    parent = PreferenceBuilderIR(
        name="parent",
        intuition="seed",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = gap.clamp(0.25, 4.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )
    prompt, _ = builder_ops.build_m2_prompt(
        str(_repo_root() / "PTP" / "prompts" / "pref_builder_m2.txt"),
        parent=parent,
        parent_fitness={"fitness": -0.1},
        global_feedback={
            "builder_search_space": cfg,
            "llm_call": {"op_type": "M2", "search_operator": "TUNE"},
        },
    )
    prompt_lower = prompt.lower()
    assert "preserve geometry_family, cap_family, weight_family, and constraint_family" in prompt_lower
    assert "good tuning targets are beta, gamma, tau, band edges, clamp bounds, and denominator eps" in prompt_lower


def test_reweight_only_m2_contract_rejects_family_change(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    parent = PreferenceBuilderIR(
        name="parent",
        intuition="seed",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code="def generated_builder(feature_cache, extra):\n    pass\n",
    )
    child = PreferenceBuilderIR(
        name="child",
        intuition="bad tune changed family",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_bandpass",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "gap_bandpass", "clamp"],
        code="def generated_builder(feature_cache, extra):\n    pass\n",
    )

    ok, fail = loop._validate_builder_operator_contract(  # noqa: SLF001
        child,
        "M2",
        [parent],
        search_space_cfg={
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "pair_weight_normalization": "none",
        },
    )
    assert ok is False
    assert fail["reason"] == "weight_family_not_preserved"


def test_builder_novelty_rejects_duplicate_structure(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint

    impl = PreferenceBuilderImplementationHint(expects=["objective", "log_prob"], returns="PrefBatch", mode="pairwise")
    ir = PreferenceBuilderIR(
        name="builder",
        intuition="seed",
        implementation_hint=impl,
        hyperparams={
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": "gap_linear",
            "constraint_family": "fixed_pair_reweight_only",
        },
        operators_used=["all_pairs", "gap_linear", "clamp"],
        code=(
            "def generated_builder(feature_cache, extra):\n"
            "    objective = feature_cache['objective']\n"
            "    mask = objective[:, :, None] < objective[:, None, :]\n"
            "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
            "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]\n"
            "    weight = gap.clamp(0.25, 4.0)\n"
            "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={'builder': 'all_pairs'})\n"
        ),
    )
    fp = loop._builder_fingerprint(ir)  # noqa: SLF001
    ok, payload = loop._check_builder_novelty(  # noqa: SLF001
        candidate=ir,
        bank=[{"sig": fp["sig"], "name": ir.name, **fp}],
        max_similarity=0.95,
        neighbors=1,
    )
    assert ok is False
    assert payload["stage"] == "novelty"


def test_builder_prompt_includes_exploration_guidance(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_builder_llm_ops as builder_ops
    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "pair_weight_normalization": "none",
            "seed_weight_families": ["gap_linear"],
            "allow_uniform_none": False,
        }
    )
    prompt, _ = builder_ops.build_generation_prompt(
        str(_repo_root() / "PTP" / "prompts" / "pref_builder_generation.txt"),
        global_feedback={
            "builder_search_space": cfg,
            "builder_search": {
                "explore_mode": True,
                "stagnation_generations": 3,
                "avoid_families": ["gap_linear", "gap_rank_blend"],
            },
        },
    )
    prompt_lower = prompt.lower()
    assert "builder_exploration_guidance" in prompt_lower
    assert "avoid dominant weighting-family patterns" in prompt_lower


def test_builder_prompt_includes_family_quota_guidance(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_builder_llm_ops as builder_ops
    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "pair_weight_normalization": "none",
            "seed_weight_families": ["gap_linear", "gap_bandpass"],
            "allow_uniform_none": False,
        }
    )
    prompt, _ = builder_ops.build_generation_prompt(
        str(_repo_root() / "PTP" / "prompts" / "pref_builder_generation.txt"),
        global_feedback={
            "builder_search_space": cfg,
            "builder_search": {
                "missing_weight_families": ["gap_bandpass"],
                "target_weight_family": "gap_bandpass",
            },
        },
    )
    prompt_lower = prompt.lower()
    assert "builder_family_quota_guidance" in prompt_lower
    assert "prefer producing a valid candidate with `weight_family = gap_bandpass`" in prompt_lower


def test_reweight_only_default_excludes_uniform_none(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import random
    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
        }
    )

    assert cfg["allow_uniform_none"] is False
    assert "uniform_none" not in cfg["seed_weight_families"]

    rng = random.Random(0)
    rng._pref_builder_search_space_cfg = cfg  # type: ignore[attr-defined]
    pool = loop._make_builtin_builder_irs(rng, 8)
    observed = {str(ir.hyperparams.get("weight_family")) for ir in pool}
    assert "uniform_none" not in observed


def test_reweight_only_builtin_pool_supports_clamp_only_weights(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import random
    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "pair_weight_normalization": "none",
            "seed_weight_families": ["gap_linear"],
            "allow_uniform_none": False,
        }
    )

    assert cfg["pair_weight_normalization"] == "none"

    rng = random.Random(0)
    rng._pref_builder_search_space_cfg = cfg  # type: ignore[attr-defined]
    ir = loop._make_builtin_builder_irs(rng, 1)[0]

    assert ir.hyperparams["constraint_family"] == "clamp_only"
    assert "mean_raw" not in ir.code
    assert "mean_weight" not in ir.code
    assert "weight = raw.clamp(clamp_lo, clamp_hi)" in ir.code


def test_builder_proposal_family_quota_backfills_missing_families(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import random
    import ptp_discovery.pref_loss_coevo_loop as loop

    out = loop._propose_builders_for_generation(
        generation=0,
        pop_g=2,
        elites_g=[],
        diverse_elites_g=[],
        rng=random.Random(0),
        llm_cfg={
            "builder": {
                "enabled": False,
                "seed_reserve": 0,
                "search_space": {
                    "enabled": True,
                    "mode": "reweight_only",
                    "fixed_pair_builder": "all_pairs",
                    "pair_weight_normalization": "none",
                    "seed_weight_families": ["gap_linear", "gap_bandpass"],
                    "allow_uniform_none": False,
                },
                "proposal_family_quota": {
                    "enabled": True,
                    "min_per_weight_family": 1,
                    "target_weight_families": ["gap_linear", "gap_bandpass"],
                    "seed_missing_families": True,
                },
            }
        },
        operator_whitelist=[],
        global_feedback={},
        llm_init_only=False,
        carry_elites=False,
    )

    observed = [loop._builder_weight_family_label(item["ir"]) for item in out]  # noqa: SLF001
    assert observed[:2] == ["gap_linear", "gap_bandpass"]


def test_builder_runtime_prompt_context_and_reweight_necessity(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_builder_llm_ops as builder_ops

    ctx = builder_ops.build_runtime_prompt_context(
        loss_observables=("seq_len",),
        mode="pairwise",
    )

    preferred = set(ctx["preferred_cheap_keys"])
    assert {"log_prob", "obj_z", "objective", "rank", "regret"} <= preferred
    assert {"instance_obj_std", "instance_obj_mad", "instance_log_prob_std", "instance_regret_mean"} <= preferred
    assert "seq_len" in set(ctx["available_keys"])
    assert "instance_obj_mean" in set(ctx["available_keys"])
    assert "instance_feature_keys" in ctx
    assert "seq_len" not in set(ctx["preferred_cheap_keys"])
    assert "entropy" in set(ctx["blocked_optional_keys"])
    assert "log_prob_step" in set(ctx["blocked_optional_keys"])

    prompt_path = _repo_root() / "PTP" / "prompts" / "pref_builder_generation.txt"
    prompt, _ = builder_ops.build_generation_prompt(
        str(prompt_path),
        global_feedback={
            "builder_search_space": {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
                "pair_weight_normalization": "none",
                "allow_freeform_weight_family": True,
                "seed_weight_families": ["gap_rank_blend"],
                "require_instance_stats": True,
                "require_instance_conditioning": True,
                "instance_stat_keys": ["instance_obj_std", "instance_log_prob_std", "instance_regret_mean"],
            }
        },
        prompt_context=ctx,
    )

    assert "RUNTIME_CONTEXT_JSON" in prompt
    assert "objective" in prompt
    assert "obj_z" in prompt
    assert "rank" in prompt
    assert "regret" in prompt
    assert "loss batch is flattened and does not carry `b_idx`" in prompt
    assert "Do not divide weights by per-instance sums or means." in prompt
    assert "Candidate weighting must explicitly use one or more instance-level statistics" in prompt
    assert "pair_signal / instance_stat" in prompt or "pair signal must be explicitly rescaled or modulated by an instance statistic" in prompt


def test_builtin_reweight_builders_use_instance_stats_when_required(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import random
    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_compiler import compile_preference_builder

    cfg = loop._normalize_builder_search_space_cfg(
        {
            "enabled": True,
            "mode": "reweight_only",
            "fixed_pair_builder": "all_pairs",
            "pair_weight_normalization": "none",
            "seed_weight_families": ["gap_linear"],
            "allow_uniform_none": False,
            "require_instance_stats": True,
            "require_instance_conditioning": True,
            "instance_stat_keys": ["instance_obj_mad", "instance_log_prob_std", "instance_regret_mean"],
        }
    )
    rng = random.Random(0)
    rng._pref_builder_search_space_cfg = cfg  # type: ignore[attr-defined]
    ir = loop._make_builtin_builder_irs(rng, 1)[0]
    assert "instance_obj_mad" in ir.code

    feature_cache = loop._dummy_feature_cache(batch_size=2, k=6, variant="visible")
    compiled = compile_preference_builder(ir)
    pref_batch = compiled.build_fn(feature_cache, {})
    assert pref_batch.weight is not None
    assert torch.isfinite(pref_batch.weight).all().item()

    ok, fail = loop.validate_builder_candidate(
        ir,
        operator_whitelist=[],
        gate_cfg={
            "min_pairs": 1,
            "min_coverage": 1.0,
            "max_pairs_per_instance": 4096,
            "weight_nonneg": True,
            "semantic_tolerance": 0.0,
            "semantic_min_pass_rate": 1.0,
            "min_instance_weight_cv": 0.10,
            "min_instance_weight_cv_pass_rate": 1.0,
            "search_space": cfg,
        },
    )
    assert ok is True
    assert fail == {}


def test_builtin_multi_signal_reweight_builders_preserve_all_pairs(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.pref_builder_compiler import compile_preference_builder

    feature_cache = loop._dummy_feature_cache(batch_size=4, k=8, variant="visible")
    objective = feature_cache["objective"]
    template_mask = objective[:, :, None] < objective[:, None, :]
    template_pair_idx = template_mask.nonzero(as_tuple=True)

    families = {
        "gap_rank_blend": "gap_rank",
        "gap_regret_blend": "gap_regret",
        "margin_rank_blend": "margin_rank",
        "margin_regret_blend": "margin_regret",
        "gap_bandpass": "gap_bandpass",
    }

    for family, signal_family in families.items():
        cfg = loop._normalize_builder_search_space_cfg(
            {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
                "seed_weight_families": [family],
                "allow_uniform_none": False,
            }
        )
        rng = random.Random(0)
        rng._pref_builder_search_space_cfg = cfg  # type: ignore[attr-defined]
        ir = loop._make_builtin_builder_irs(rng, 1)[0]
        assert ir.hyperparams["weight_family"] == family
        assert ir.hyperparams["signal_family"] == signal_family

        compiled = compile_preference_builder(ir)
        pref_batch = compiled.build_fn(feature_cache, {})
        assert pref_batch.weight is not None
        assert torch.isfinite(pref_batch.weight).all().item()
        assert bool((pref_batch.weight >= 0).all().item())

        built_pair_idx = pref_batch.pair_idx
        assert built_pair_idx is not None
        for built, expected in zip(built_pair_idx, template_pair_idx):
            assert torch.equal(built, expected)

        b_idx = built_pair_idx[0]
        for batch_id in range(int(objective.shape[0])):
            mask = b_idx == batch_id
            assert bool(mask.any().item())
            mean_weight = pref_batch.weight[mask].mean()
            assert torch.isclose(mean_weight, torch.tensor(1.0, dtype=mean_weight.dtype), atol=1e-4, rtol=1e-4).item()


def test_reweight_only_builder_operator_bank_uses_five_search_classes(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import random
    import ptp_discovery.pref_loss_coevo_loop as loop

    plan = loop._expand_operator_bank(
        {
            "search_space": {
                "enabled": True,
                "mode": "reweight_only",
                "fixed_pair_builder": "all_pairs",
            },
            "operator_bank": {
                "init": [
                    {"name": "GEN", "count": 1},
                    {"name": "PARADIGM_SHIFT", "count": 2},
                    {"name": "STRUCTURE_SHIFT", "count": 3},
                    {"name": "CONSTRAINT_INJECT", "count": 4},
                    {"name": "XOVER", "count": 1},
                ]
            },
        },
        generation=0,
        rng=random.Random(0),
        side="builder",
    )

    assert "GEN" not in plan
    assert "MUTATE" not in plan
    assert plan.count("PARADIGM_SHIFT") == 3
    assert plan.count("STRUCTURE_SHIFT") == 3
    assert plan.count("CONSTRAINT_INJECT") == 4
    assert plan.count("XOVER") == 1


def test_aggregate_proxy_metrics_excludes_gate_signal_from_proxy_score(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    from fitness.pref_loss_fidelity import aggregate_proxy_metrics

    score_a, agg_a = aggregate_proxy_metrics(
        [
            {
                "loss": 1.0,
                "effective_grad_ratio": 0.1,
                "ess_ratio": 0.5,
            }
        ],
        proxy_weights={"effective_grad_ratio": 99.0, "ess_ratio": 0.1},
    )
    score_b, agg_b = aggregate_proxy_metrics(
        [
            {
                "loss": 1.0,
                "effective_grad_ratio": 0.9,
                "ess_ratio": 0.5,
            }
        ],
        proxy_weights={"effective_grad_ratio": 99.0, "ess_ratio": 0.1},
    )

    assert score_a == score_b
    assert score_a == 1.0 + 0.1 * (1.0 - 0.5)
    assert agg_a["proxy_effective_grad_ratio_mean"] == 0.1
    assert agg_b["proxy_effective_grad_ratio_mean"] == 0.9


def test_gate_only_pass_uses_neutral_score(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop
    from ptp_discovery.free_loss_gates import JointPreferenceGateResult, PreferenceBuilderGateResult

    monkeypatch.setattr(
        loop,
        "run_preference_builder_gates",
        lambda *args, **kwargs: PreferenceBuilderGateResult(
            ok=True,
            reason="ok",
            pair_count=12,
            coverage=1.0,
            semantic_pass_rate=0.3,
            trace={"failed_gate": None},
        ),
    )
    monkeypatch.setattr(
        loop,
        "run_joint_preference_gates",
        lambda *args, **kwargs: JointPreferenceGateResult(
            ok=True,
            reason="ok",
            effective_grad_ratio=0.8,
            trace={"failed_gate": None, "observed": {"effective_grad_ratio": 0.8}},
        ),
    )
    monkeypatch.setattr(loop, "_run_co_alignment_gates_for_loss", lambda *args, **kwargs: {"co_ok": True, "co_reason": "ok"})

    rec = loop._evaluate_pair_worker(
        {
            "generation": 0,
            "pair_index": 0,
            "g_entry": {"id": "g_ref", "ir": loop.asdict(loop._ref_builder_ir())},
            "f_entry": {"id": "f_ref", "ir": loop.asdict(loop._ref_loss_ir())},
            "cfg_yaml": {
                "cheap_gate_batch_size": 4,
                "cheap_gate_k": 8,
                "builder_max_pairs_per_instance": 4096,
            },
            "device_str": "cpu",
            "operator_whitelist": [],
            "run_dir": None,
            "cheap_gate_on": True,
            "high_fidelity_on": False,
            "eval_budget_signature": "test",
        }
    )

    assert rec["pair_ok"] is True
    assert rec["pair_reason"] == "ok_gate_only"
    assert rec["score"] == 0.0
    assert rec["proxy_metrics"]["cheap_effective_grad_ratio"] == 0.8
    assert rec["proxy_metrics"]["cheap_semantic_pass_rate"] == 0.3


def test_builder_llm_exception_is_logged(monkeypatch, caplog):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import random
    import ptp_discovery.pref_loss_coevo_loop as loop
    import ptp_discovery.pref_builder_llm_ops as builder_ops

    def _boom(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(builder_ops, "generate_pref_builder_candidate_with_meta", _boom)

    with caplog.at_level(logging.WARNING):
        out = loop._propose_builders_for_generation(
            generation=0,
            pop_g=1,
            elites_g=[],
            diverse_elites_g=[],
            rng=random.Random(0),
            llm_cfg={
                "builder": {
                    "enabled": True,
                    "parent_p": 2,
                    "seed_reserve": 0,
                    "search_space": {
                        "enabled": True,
                        "mode": "reweight_only",
                        "fixed_pair_builder": "all_pairs",
                    },
                    "operator_bank": {
                        "init": [
                            {"name": "GEN", "count": 1},
                        ]
                    },
                    "repair": {"enabled": False},
                },
                "prompts": {
                    "builder_generation": "PTP/prompts/pref_builder_generation.txt",
                },
            },
            operator_whitelist=[],
            global_feedback={},
            llm_init_only=True,
            carry_elites=False,
        )

    assert out == []
    assert "Builder LLM proposal failed at gen=0" in caplog.text
    assert ("OPENAI_API_KEY is not set" in caplog.text) or ("openai package is not installed" in caplog.text)
    assert "Builder generation 0 produced zero valid proposals while llm_init_only=true" in caplog.text
