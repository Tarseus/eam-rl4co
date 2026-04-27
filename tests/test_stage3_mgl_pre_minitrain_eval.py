from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_stage3_pre_minitrain_eval_uses_native_mgl_model(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import fitness.free_loss_fidelity as fidelity
    import ptp_discovery.pref_loss_coevo_loop as loop

    cleanup_calls: list[str] = []
    weight_loads: list[str] = []
    eval_calls: list[dict[str, object]] = []

    class _FakeModel:
        def __init__(self) -> None:
            self.eval_called = False
            self.device = None

        def to(self, device):  # noqa: ANN001
            self.device = str(device)
            return self

        def eval(self):
            self.eval_called = True
            return self

    model = _FakeModel()

    monkeypatch.setattr(loop, "_stage3_cleanup_eval_device", lambda device_str: cleanup_calls.append(str(device_str)))
    monkeypatch.setattr(loop, "_abs_from_repo_root", lambda path: f"ABS::{path}")
    monkeypatch.setattr(fidelity, "_rl4co_build_env", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic env builder should not run for mgl_jssp")))
    monkeypatch.setattr(fidelity, "_rl4co_build_policy", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generic policy builder should not run for mgl_jssp")))
    monkeypatch.setattr(fidelity, "_build_mgl_jssp_model", lambda cfg, problem_size: model)
    monkeypatch.setattr(fidelity, "_load_policy_weights_from_checkpoint", lambda policy, path: weight_loads.append(str(path)))

    def _fake_eval(**kwargs):  # noqa: ANN003
        eval_calls.append(dict(kwargs))
        return float(kwargs["problem_size"]) / 10.0

    monkeypatch.setattr(fidelity, "_evaluate_rl4co_model", _fake_eval)

    cfg = {
        "problem": "jssp",
        "backend": "rl4co",
        "env_name": "jssp",
        "generator_params": {"num_jobs": 10, "num_machines": 10},
        "policy_name": "mgl_jssp",
        "policy_kwargs": {},
        "rollout_strategy": "auto",
        "objective_sign": "neg_reward",
        "train_problem_size": 10,
        "valid_problem_sizes": [10, 15],
        "train_batch_size": 8,
        "pomo_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "alpha": 1.0,
        "device": "cpu",
        "num_validation_episodes": 6,
        "validation_batch_size": 3,
    }

    by_size, aggregated = loop._stage3_pre_minitrain_eval(
        cfg_yaml=cfg,
        init_checkpoint="baseline.ckpt",
        train_problem_size=10,
        valid_problem_sizes=[10, 15],
        num_validation_episodes=6,
        train_batch_size=8,
        scratch_init_seed=123,
    )

    assert by_size == {10: 1.0, 15: 1.5}
    assert aggregated == 1.25
    assert weight_loads == ["ABS::baseline.ckpt"]
    assert cleanup_calls == ["cpu"]
    assert model.eval_called is True
    assert model.device == "cpu"
    assert [call["problem_size"] for call in eval_calls] == [10, 15]
    assert all(call["policy"] is model for call in eval_calls)
    assert all(call["rollout_strategy"] == "mgl_sampling" for call in eval_calls)
