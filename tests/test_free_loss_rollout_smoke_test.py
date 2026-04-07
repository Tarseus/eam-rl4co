from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

repo_root = Path(__file__).resolve().parents[1]
ptp_root = repo_root / "PTP"
for path in (repo_root, ptp_root):
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)

import fitness.free_loss_fidelity as fidelity


class _DummyEnv:
    def to(self, device):  # noqa: ANN001
        return self


class _DummyPolicy:
    def to(self, device):  # noqa: ANN001
        return self

    def eval(self):
        return self


def test_tsp_po4cops_rollout_smoke_test_uses_two_rollouts(monkeypatch) -> None:
    captured: dict[str, int] = {}

    monkeypatch.setattr(fidelity, "_set_seed", lambda seed: None)
    monkeypatch.setattr(fidelity, "_rl4co_build_env", lambda cfg, size: _DummyEnv())
    monkeypatch.setattr(
        fidelity,
        "_rl4co_build_policy",
        lambda cfg, env: (_DummyPolicy(), "policy_multistart"),
    )
    monkeypatch.setattr(fidelity, "_load_policy_weights_from_checkpoint", lambda policy, path: None)

    def _fake_rollout_full(
        env,
        policy,
        batch_size,
        num_rollouts,
        *,
        phase,
        rollout_strategy,
        device,
        precision="32-true",
        cfg_like=None,
        return_actions=False,
        return_entropy=False,
        return_step_logp=False,
    ):
        captured["batch_size"] = int(batch_size)
        captured["num_rollouts"] = int(num_rollouts)
        return {"reward": [0.0], "log_likelihood": [0.0]}

    monkeypatch.setattr(fidelity, "_rl4co_rollout_full", _fake_rollout_full)

    cfg = SimpleNamespace(
        seed=1234,
        device="cpu",
        precision="32-true",
        env_name="tsp",
        policy_name="pomo",
        train_problem_size=100,
        train_batch_size=64,
        pomo_size=None,
        policy_kwargs={"po4cops_compat": True},
    )

    result = fidelity.run_rl4co_rollout_smoke_test(
        cfg,
        phase="train",
        device="cpu",
        batch_size=1,
        num_rollouts=1,
    )

    assert result["ok"] is True
    assert int(result["num_rollouts"]) == 2
    assert captured["batch_size"] == 1
    assert captured["num_rollouts"] == 2
