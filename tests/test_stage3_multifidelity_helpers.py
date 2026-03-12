from __future__ import annotations

from pathlib import Path
import pytest


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


def test_compute_builder_constraint_state_prefers_lowest_cost_within_slack(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    records = [
        {"pair_ok": True, "g_id": "g1", "f_id": "f_fixed", "final_score": 1.00, "descriptor": {"g": {"pair_count": 120}}},
        {"pair_ok": True, "g_id": "g2", "f_id": "f_fixed", "final_score": 1.01, "descriptor": {"g": {"pair_count": 40}}},
        {"pair_ok": True, "g_id": "g3", "f_id": "f_fixed", "final_score": 1.20, "descriptor": {"g": {"pair_count": 10}}},
    ]
    state = loop._compute_builder_constraint_state(
        records=records,
        perf_by_builder={"g1": 1.00, "g2": 1.01, "g3": 1.20},
        metric_mode="minimize",
        slack=0.02,
    )

    assert state["best_perf"] == 1.0
    assert state["selected"]["builder_id"] == "g2"
    assert [x["builder_id"] for x in state["feasible"]] == ["g2", "g1"]
    assert [x["builder_id"] for x in state["infeasible"]] == ["g3"]


def test_select_stage3_builder_promotions_uses_feasible_then_cost(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    records = [
        {"pair_ok": True, "g_id": "g1", "f_id": "f_fixed", "score": 1.00, "descriptor": {"g": {"pair_count": 120}}},
        {"pair_ok": True, "g_id": "g2", "f_id": "f_fixed", "score": 1.01, "descriptor": {"g": {"pair_count": 40}}},
        {"pair_ok": True, "g_id": "g3", "f_id": "f_fixed", "score": 1.20, "descriptor": {"g": {"pair_count": 10}}},
    ]
    promoted = loop._select_stage3_builder_promotions(
        records,
        promote_top_m=2,
        metric_mode="minimize",
        slack=0.02,
        fixed_loss_id="f_fixed",
        always_include_pair=None,
    )

    assert promoted == [("g2", "f_fixed"), ("g1", "f_fixed")]


def test_select_best_builder_cost_from_archive_blocks_performance_drift(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    archive = {
        "g1": {"builder_id": "g1", "perf": 1.00, "cost": 120.0},
        "g2": {"builder_id": "g2", "perf": 1.01, "cost": 40.0},
        "g3": {"builder_id": "g3", "perf": 1.03, "cost": 10.0},
    }

    selected = loop._select_best_builder_cost_from_archive(
        builder_archive=archive,
        metric_mode="minimize",
        slack=0.02,
    )

    assert selected["best_perf"] == 1.0
    assert selected["selected"]["builder_id"] == "g2"
    assert [x["builder_id"] for x in selected["feasible"]] == ["g2", "g1"]
    assert [x["builder_id"] for x in selected["infeasible"]] == ["g3"]


def test_select_best_builder_cost_from_archive_reanchors_when_better_perf_arrives(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    archive = {
        "g1": {"builder_id": "g1", "perf": 1.00, "cost": 120.0},
        "g2": {"builder_id": "g2", "perf": 1.01, "cost": 40.0},
        "g4": {"builder_id": "g4", "perf": 0.98, "cost": 200.0},
    }

    selected = loop._select_best_builder_cost_from_archive(
        builder_archive=archive,
        metric_mode="minimize",
        slack=0.02,
    )

    assert selected["best_perf"] == 0.98
    assert selected["threshold"] == 1.0
    assert selected["selected"]["builder_id"] == "g1"
    assert [x["builder_id"] for x in selected["feasible"]] == ["g1", "g4"]
    assert [x["builder_id"] for x in selected["infeasible"]] == ["g2"]


def test_build_pair_descriptor_keeps_builder_memory_metrics(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    desc = loop._build_pair_descriptor(
        builder_gate_trace={
            "coverage": 0.5,
            "pair_count": 64,
            "semantic_pass_rate": 1.0,
            "memory_peak_allocated_delta_mb": 12.5,
            "memory_peak_reserved_delta_mb": 16.0,
        },
        proxy_agg={"effective_grad_ratio": 0.2, "ess_ratio": 0.1, "loss": 0.3},
        pair_count_cap=128,
        loss_scale=1.0,
        bins=8,
    )

    assert desc["g"]["pair_count"] == 64
    assert desc["g"]["memory_peak_allocated_mb"] == 12.5
    assert desc["g"]["memory_peak_reserved_mb"] == 16.0


def test_stage3_multiseed_compare_cfg_uses_calibration_defaults(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {
        "scratch_init_seed": 12345,
        "improve_eps_calibration": {
            "enabled": True,
            "N": 8,
            "seed_stride": 111,
        },
        "baseline": {},
    }

    plan = loop._stage3_multiseed_compare_cfg(cfg)
    assert plan["enabled"] is True
    assert plan["n_seeds"] == 8
    assert plan["seed0"] == 12345 + 999
    assert plan["seed_stride"] == 111


def test_stage3_baseline_multiseed_cache_key_differs_by_fidelity(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    monkeypatch.setattr(
        loop,
        "_build_stage3_eval_signature",
        lambda cfg: {"fidelity": loop._stage3_fidelity_key(cfg), "env": "tsp"},
    )

    cfg_200 = {"f1_steps": 200, "hf_epochs": 0, "hf_instances_per_epoch": 0}
    cfg_1000 = {"f1_steps": 1000, "hf_epochs": 0, "hf_instances_per_epoch": 0}

    key_200 = loop._stage3_baseline_multiseed_cache_key(
        cfg_200,
        include_scratch=True,
        seed0=1,
        seed_stride=2,
    )
    key_1000 = loop._stage3_baseline_multiseed_cache_key(
        cfg_1000,
        include_scratch=True,
        seed0=1,
        seed_stride=2,
    )

    assert key_200 != key_1000
    assert key_200.startswith("K200__")
    assert key_1000.startswith("K1000__")


def test_aggregate_stage3_baseline_multiseed_records_prefers_best_seed(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    per_init_base = {
        "scratch": {"aggregated_objective": 10.0},
        "ckpt_135": {"aggregated_objective": 7.0},
    }
    per_seed = {
        "100": {
            "seed": 100,
            "per_init": {
                "scratch": {
                    "aggregated_objective": 9.5,
                    "val_objective_by_size": {"100": 9.5},
                },
                "ckpt_135": {
                    "aggregated_objective": 6.8,
                    "val_objective_by_size": {"100": 6.8},
                },
            },
        },
        "200": {
            "seed": 200,
            "per_init": {
                "scratch": {
                    "aggregated_objective": 9.0,
                    "val_objective_by_size": {"100": 9.0},
                },
                "ckpt_135": {
                    "aggregated_objective": 6.9,
                    "val_objective_by_size": {"100": 6.9},
                },
            },
        },
    }

    agg = loop._aggregate_stage3_baseline_multiseed_records(
        per_init_base=per_init_base,
        per_seed=per_seed,
    )

    assert agg["best_per_init"]["scratch"]["seed"] == 200
    assert agg["best_per_init"]["scratch"]["aggregated_objective"] == 9.0
    assert agg["best_per_init"]["ckpt_135"]["seed"] == 100
    assert agg["best_per_init"]["ckpt_135"]["aggregated_objective"] == 6.8
    assert agg["samples"] == pytest.approx([-0.35, -0.55])


def test_annotate_stage_fields_does_not_mark_disabled_proxy_as_ran(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    ran, skipped = loop._annotate_stage_fields(
        {
            "stage": "high_fidelity",
            "fitness": {"delta_mean": -0.03},
            "proxy_score": 0.0,
            "proxy_metrics": {"cheap_effective_grad_ratio": 1.0},
            "builder_gate_ok": True,
            "joint_gate_ok": True,
        },
        eval_stages={
            "stage0_gate": True,
            "stage1_proxy": False,
            "stage2_micro_unroll": False,
            "stage3_high_fidelity": True,
        },
    )

    assert ran == ["stage0_gate", "stage3_high_fidelity"]
    assert skipped["stage1_proxy"] == "disabled"


def test_resolve_stage3_baseline_reference_entry_prefers_multiseed_best(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    entry, source = loop._resolve_stage3_baseline_reference_entry(
        "scratch",
        per_init_base={"scratch": {"aggregated_objective": 10.0}},
        multiseed_cache={
            "best_per_init": {
                "scratch": {
                    "seed": 321,
                    "aggregated_objective": 8.5,
                    "val_objective_by_size": {"100": 8.5},
                }
            }
        },
    )

    assert source == "multiseed_best"
    assert entry["seed"] == 321
    assert entry["aggregated_objective"] == 8.5


def test_stage3_init_specs_accept_arbitrary_checkpoint_list(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {
        "baseline": {
            "include_scratch": True,
            "checkpoints": [
                "baseline/tsp50_epoch_50.ckpt",
                "baseline/tsp50_epoch_100.ckpt",
                "baseline/custom_final.ckpt",
            ],
        }
    }

    init_specs = loop._stage3_init_specs_from_baseline_cfg(cfg)

    assert init_specs == [
        ("scratch", None),
        ("ckpt_50", "baseline/tsp50_epoch_50.ckpt"),
        ("ckpt_100", "baseline/tsp50_epoch_100.ckpt"),
        ("ckpt_custom_final", "baseline/custom_final.ckpt"),
    ]


def test_iter_stage3_scenario_cfgs_merges_scenario_specific_overrides(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {
        "problem": "tsp",
        "env_name": "tsp",
        "train_problem_size": 100,
        "valid_problem_sizes": [100],
        "generator_params": {
            "offline_train_path": "offline_data/tsp100_train.pt",
            "offline_val_paths": {"100": "offline_data/tsp100_val.pt"},
        },
        "baseline": {
            "include_scratch": True,
            "checkpoints": ["baseline/epoch_135.ckpt", "baseline/epoch_409.ckpt"],
            "mini_eval_paths": {200: "baseline/mini_eval/baseline_minitrain_tsp100_K200.json"},
            "scenarios": [
                {
                    "name": "tsp50",
                    "train_problem_size": 50,
                    "valid_problem_sizes": [50],
                    "generator_params": {
                        "offline_train_path": "offline_data/tsp50_train.pt",
                        "offline_val_paths": {"50": "offline_data/tsp50_val.pt"},
                    },
                    "baseline": {
                        "checkpoints": [
                            "baseline/tsp50_epoch_50.ckpt",
                            "baseline/tsp50_epoch_100.ckpt",
                        ],
                        "mini_eval_paths": {200: "baseline/mini_eval/baseline_minitrain_tsp50_K200.json"},
                    },
                },
                {
                    "name": "cvrp100",
                    "problem": "cvrp",
                    "env_name": "cvrp",
                    "train_problem_size": 100,
                    "valid_problem_sizes": [100],
                    "generator_params": {
                        "offline_train_path": "offline_data/cvrp100_train.pt",
                        "offline_val_paths": {"100": "offline_data/cvrp100_val.pt"},
                    },
                    "baseline": {
                        "checkpoints": [
                            "baseline/cvrp100_epoch_100.ckpt",
                            "baseline/cvrp100_epoch_200.ckpt",
                        ],
                        "mini_eval_paths": {200: "baseline/mini_eval/baseline_minitrain_cvrp100_K200.json"},
                    },
                },
            ],
        },
    }

    scenarios = loop._iter_stage3_scenario_cfgs(cfg)

    assert [item["name"] for item in scenarios] == ["tsp50", "cvrp100"]
    assert scenarios[0]["cfg"]["train_problem_size"] == 50
    assert scenarios[0]["cfg"]["generator_params"]["offline_train_path"] == "offline_data/tsp50_train.pt"
    assert scenarios[0]["cfg"]["baseline"]["checkpoints"] == [
        "baseline/tsp50_epoch_50.ckpt",
        "baseline/tsp50_epoch_100.ckpt",
    ]
    assert "scenarios" not in scenarios[0]["cfg"]["baseline"]
    assert scenarios[1]["cfg"]["problem"] == "cvrp"
    assert scenarios[1]["cfg"]["env_name"] == "cvrp"
    assert scenarios[1]["cfg"]["baseline"]["mini_eval_paths"][200] == (
        "baseline/mini_eval/baseline_minitrain_cvrp100_K200.json"
    )


def test_build_stage3_eval_signature_tracks_scenario_name_and_all_checkpoints(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    monkeypatch.setattr(loop, "_file_sha1_cached", lambda path: f"sha1:{path}")

    cfg = {
        "problem": "cvrp",
        "env_name": "cvrp",
        "train_problem_size": 50,
        "valid_problem_sizes": [50],
        "f1_steps": 200,
        "generator_params": {
            "offline_train_path": "offline_data/cvrp50_train.pt",
            "offline_val_paths": {"50": "offline_data/cvrp50_val.pt"},
        },
        "baseline": {
            "include_scratch": True,
            "checkpoints": [
                "baseline/cvrp50_epoch_50.ckpt",
                "baseline/cvrp50_epoch_100.ckpt",
            ],
        },
        "stage3_scenario_name": "cvrp50_mid_late",
    }

    sig = loop._build_stage3_eval_signature(cfg)

    assert sig["scenario_name"] == "cvrp50_mid_late"
    assert sig["include_scratch"] is True
    assert sig["checkpoints"] == [
        {
            "name": "ckpt_50",
            "path": "baseline/cvrp50_epoch_50.ckpt",
            "sha1": "sha1:baseline/cvrp50_epoch_50.ckpt",
        },
        {
            "name": "ckpt_100",
            "path": "baseline/cvrp50_epoch_100.ckpt",
            "sha1": "sha1:baseline/cvrp50_epoch_100.ckpt",
        },
    ]


def test_build_stage3_eval_signature_supports_online_epoch_mode(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {
        "problem": "tsp",
        "env_name": "tsp",
        "train_problem_size": 100,
        "valid_problem_sizes": [100],
        "hf_epochs": 1,
        "hf_instances_per_epoch": 100000,
        "generator_params": {"num_loc": 100},
        "baseline": {
            "include_scratch": True,
            "checkpoints": ["baseline/tsp100_epoch_135.ckpt"],
        },
        "stage3_scenario_name": "tsp100_online_epoch1",
    }

    monkeypatch.setattr(loop, "_file_sha1_cached", lambda path: f"sha1:{path}")

    sig = loop._build_stage3_eval_signature(cfg)

    assert sig["protocol"] == "stage3_online_minitrain_v1"
    assert sig["data"] == {"mode": "online", "generator_params": {"num_loc": 100}}
    assert sig["hf_epochs"] == 1
    assert sig["hf_instances_per_epoch"] == 100000
    assert sig["checkpoints"] == [
        {
            "name": "ckpt_135",
            "path": "baseline/tsp100_epoch_135.ckpt",
            "sha1": "sha1:baseline/tsp100_epoch_135.ckpt",
        }
    ]
