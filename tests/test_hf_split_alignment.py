from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_free_cfg_propagates_split_fields(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {
        "backend": "rl4co",
        "env_name": "tsp",
        "generator_params": {"num_loc": 20},
        "policy_name": "pomo",
        "policy_kwargs": {},
        "rollout_strategy": "auto",
        "objective_sign": "neg_reward",
        "hf_epochs": 20,
        "hf_instances_per_epoch": 100000,
        "train_problem_size": 20,
        "valid_problem_sizes": [20],
        "train_batch_size": 64,
        "pomo_size": 16,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "alpha": 0.05,
        "device": "cpu",
        "num_validation_episodes": 10000,
        "validation_batch_size": 64,
        "scratch_hf_epochs": 10,
        "warmstart_hf_epochs": 10,
        "baseline_epoch_compare_offset": 0,
        "baseline": {"checkpoint": "baseline/epoch_409.ckpt", "checkpoint_epoch": 409},
    }

    hf_cfg = loop._build_hf_cfg(cfg, seed=0, device_str="cpu")
    free_cfg = loop._build_free_cfg(cfg, hf_cfg=hf_cfg)

    assert int(getattr(free_cfg, "scratch_hf_epochs", 0) or 0) == 10
    assert int(getattr(free_cfg, "warmstart_hf_epochs", 0) or 0) == 10
    assert int(getattr(free_cfg, "baseline_epoch_compare_offset", 0)) == 0


def test_simple_runtime_config_lifts_split_hf_epochs_from_budgets(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {
        "preset": "simple",
        "search_mode": "builder_only",
        "budgets": {
            "hf_epochs": 10,
            "scratch_hf_epochs": 5,
            "warmstart_hf_epochs": 5,
        },
    }

    runtime_cfg, _ = loop._resolve_runtime_config(cfg)

    assert int(runtime_cfg.get("hf_epochs", 0) or 0) == 10
    assert int(runtime_cfg.get("scratch_hf_epochs", 0) or 0) == 5
    assert int(runtime_cfg.get("warmstart_hf_epochs", 0) or 0) == 5


def test_ffsp100_filters_seq_len_and_log_prob_mean(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = {
        "backend": "rl4co",
        "env_name": "ffsp",
        "generator_params": {"num_stage": 3, "num_machine": 4, "num_job": 100},
        "policy_name": "matnet",
        "policy_kwargs": {},
        "rollout_strategy": "auto",
        "objective_sign": "neg_reward",
        "train_problem_size": 100,
        "valid_problem_sizes": [100],
        "train_batch_size": 50,
        "pomo_size": 24,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "alpha": 1.0,
        "device": "cpu",
        "num_validation_episodes": 1000,
        "validation_batch_size": 50,
        "loss_observables": ["seq_len", "log_prob_mean", "advantage"],
    }

    resolved = loop._resolve_loss_observables(cfg)
    assert resolved == ("advantage",)

    hf_cfg = loop._build_hf_cfg(cfg, seed=0, device_str="cpu")
    assert tuple(hf_cfg.loss_observables) == ("advantage",)


def test_external_baseline_slice_split_calls(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    calls: list[dict] = []

    def _fake_baseline(metrics_csv_path, *, value_col, start_epoch, num_epochs, objective_sign):  # noqa: ANN001
        calls.append(
            {
                "metrics_csv_path": str(metrics_csv_path),
                "value_col": str(value_col),
                "start_epoch": int(start_epoch),
                "num_epochs": int(num_epochs),
                "objective_sign": str(objective_sign),
            }
        )
        return [float(i) for i in range(int(num_epochs))]

    monkeypatch.setattr(loop, "baseline_epoch_objectives_from_metrics_csv", _fake_baseline)

    cfg = {
        "seed": 0,
        "output_root": str(tmp_path / "runs"),
        "generations": 0,
        "pop_g": 2,
        "pop_f": 2,
        "elite_g": 1,
        "elite_f": 1,
        "pairing_budget_per_gen": 1,
        "cheap_gate_on": False,
        "high_fidelity_on": False,
        # HF config (signature only; no HF evaluation is run in this test).
        "backend": "rl4co",
        "env_name": "tsp",
        "generator_params": {"num_loc": 20},
        "policy_name": "pomo",
        "policy_kwargs": {},
        "rollout_strategy": "auto",
        "objective_sign": "neg_reward",
        "hf_epochs": 20,
        "hf_instances_per_epoch": 100000,
        "train_problem_size": 20,
        "valid_problem_sizes": [20],
        "train_batch_size": 64,
        "pomo_size": 16,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "alpha": 0.05,
        "device": "cpu",
        "num_validation_episodes": 10000,
        "validation_batch_size": 64,
        "size_aggregation": "cvar",
        "size_cvar_alpha": 0.2,
        # Split HF eval alignment.
        "scratch_hf_epochs": 10,
        "warmstart_hf_epochs": 10,
        "baseline_epoch_compare_offset": 0,
        "early_eval_epochs": 2,
        "early_eval_instances_per_epoch": 10000,
        # External baseline.
        "baseline": {
            "metrics_csv": "baseline/metrics.csv",
            "checkpoint": "baseline/epoch_409.ckpt",
            "checkpoint_epoch": 409,
            "scratch_start_epoch": 0,
            "val_column": "val/reward",
        },
        # Proxy signature.
        "proxy_problem_size": 20,
        "proxy_batch_size": 8,
        "proxy_batches": 1,
    }

    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loop.run_pref_loss_coevo(str(cfg_path))

    assert len(calls) == 2

    scratch = calls[0]
    warm = calls[1]
    assert scratch["start_epoch"] == 0
    assert scratch["num_epochs"] == 10
    assert warm["start_epoch"] == 410
    assert warm["num_epochs"] == 10
