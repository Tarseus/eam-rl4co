from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_stage3_baseline_auto_cache_cleans_between_inits(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cleanup_calls: list[str] = []
    writes: list[tuple[str, dict]] = []

    monkeypatch.setattr(loop, "_stage3_cleanup_eval_device", lambda device_str: cleanup_calls.append(str(device_str)))
    monkeypatch.setattr(loop, "_build_stage3_eval_signature", lambda cfg: {"sig": "ffsp-baseline-test"})
    monkeypatch.setattr(loop, "_build_hf_cfg", lambda cfg, seed, device_str: object())
    monkeypatch.setattr(
        loop,
        "_stage3_pre_minitrain_eval",
        lambda **kwargs: ({100: 10.0}, 10.0),
    )
    monkeypatch.setattr(
        loop,
        "_evaluate_stage3_reference_baseline",
        lambda **kwargs: {"size_objectives": {100: 9.5}},
    )
    monkeypatch.setattr(
        loop,
        "_extract_stage3_size_objectives",
        lambda fitness, valid_sizes: ({100: 9.5}, 9.5),
    )
    monkeypatch.setattr(loop, "_atomic_write_json", lambda path, payload: writes.append((str(path), dict(payload))))

    mini_eval_path = tmp_path / "baseline_ffsp100_test.json"
    cfg = {
        "env_name": "ffsp",
        "problem": "ffsp",
        "objective_sign": "neg_reward",
        "train_problem_size": 100,
        "valid_problem_sizes": [100],
        "num_validation_episodes": 4,
        "train_batch_size": 32,
        "validation_batch_size": 50,
        "f1_steps": 200,
        "seed": 1234,
        "generator_params": {
            "num_stage": 3,
            "num_machine": 4,
            "num_job": 100,
            "flatten_stages": False,
        },
        "baseline": {
            "include_scratch": True,
            "checkpoints": ["baseline/ffsp_matnet_po_paper100_epoch_100.ckpt"],
            "mini_eval_path": str(mini_eval_path),
        },
    }

    prepared = loop._ensure_stage3_baseline_mini_eval(
        cfg_yaml=cfg,
        operator_whitelist=[],
        device_str="cuda:2",
    )

    assert prepared["regenerated"] is True
    assert len(writes) == 2
    assert writes[0][1]["complete"] is False
    assert list(writes[0][1]["per_init"].keys()) == ["scratch"]
    assert writes[1][1]["complete"] is True
    assert set(writes[1][1]["per_init"].keys()) == {"scratch", "ckpt_100"}
    assert cleanup_calls == ["cuda:2", "cuda:2"]


def test_stage3_baseline_auto_cache_resumes_from_partial_json(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cleanup_calls: list[str] = []
    pre_calls: list[str | None] = []
    eval_calls: list[str | None] = []

    mini_eval_path = tmp_path / "baseline_ffsp100_resume.json"
    partial_payload = {
        "schema_version": 1,
        "created_at": "2026-04-15 18:00:00",
        "config_path": None,
        "eval_signature": {"sig": "ffsp-baseline-test"},
        "per_init": {
            "scratch": {
                "aggregated_objective": 9.5,
                "init_checkpoint": None,
            }
        },
        "reference": {
            "builder_ir": {},
            "loss_ir": {},
        },
        "complete": False,
    }
    mini_eval_path.write_text(json.dumps(partial_payload), encoding="utf-8")

    monkeypatch.setattr(loop, "_BASELINE_MINI_EVAL_CACHE", {})
    monkeypatch.setattr(loop, "_stage3_cleanup_eval_device", lambda device_str: cleanup_calls.append(str(device_str)))
    monkeypatch.setattr(loop, "_build_stage3_eval_signature", lambda cfg: {"sig": "ffsp-baseline-test"})
    monkeypatch.setattr(loop, "_build_hf_cfg", lambda cfg, seed, device_str: object())
    monkeypatch.setattr(
        loop,
        "_stage3_pre_minitrain_eval",
        lambda **kwargs: pre_calls.append(kwargs.get("init_checkpoint")) or ({100: 10.0}, 10.0),
    )
    monkeypatch.setattr(
        loop,
        "_evaluate_stage3_reference_baseline",
        lambda **kwargs: eval_calls.append(kwargs.get("init_ckpt")) or {"size_objectives": {100: 9.5}},
    )
    monkeypatch.setattr(
        loop,
        "_extract_stage3_size_objectives",
        lambda fitness, valid_sizes: ({100: 9.5}, 9.5),
    )

    cfg = {
        "env_name": "ffsp",
        "problem": "ffsp",
        "objective_sign": "neg_reward",
        "train_problem_size": 100,
        "valid_problem_sizes": [100],
        "num_validation_episodes": 4,
        "train_batch_size": 32,
        "validation_batch_size": 50,
        "f1_steps": 200,
        "seed": 1234,
        "generator_params": {
            "num_stage": 3,
            "num_machine": 4,
            "num_job": 100,
            "flatten_stages": False,
        },
        "baseline": {
            "include_scratch": True,
            "checkpoints": ["baseline/ffsp_matnet_po_paper100_epoch_100.ckpt"],
            "mini_eval_path": str(mini_eval_path),
        },
    }

    prepared = loop._ensure_stage3_baseline_mini_eval(
        cfg_yaml=cfg,
        operator_whitelist=[],
        device_str="cuda:3",
    )

    payload = json.loads(mini_eval_path.read_text(encoding="utf-8"))
    assert prepared["regenerated"] is True
    assert pre_calls == ["baseline/ffsp_matnet_po_paper100_epoch_100.ckpt"]
    assert eval_calls == ["baseline/ffsp_matnet_po_paper100_epoch_100.ckpt"]
    assert cleanup_calls == ["cuda:3"]
    assert payload["complete"] is True
    assert set(payload["per_init"].keys()) == {"scratch", "ckpt_100"}
