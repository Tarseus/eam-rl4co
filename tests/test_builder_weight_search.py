from __future__ import annotations

import json
import random
from pathlib import Path


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
