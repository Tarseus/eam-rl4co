from pathlib import Path
import sys

import torch


def test_build_runtime_observables_can_infer_seq_len_from_step_logp_without_actions() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "PTP"))

    from fitness.free_loss_fidelity import build_runtime_observables

    reward = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    log_prob = torch.tensor([[-2.0, -3.0]], dtype=torch.float32)
    log_prob_step = torch.tensor(
        [[[-1.0, -1.0, 0.0], [-1.0, -1.0, -1.0]]],
        dtype=torch.float32,
    )

    extra = build_runtime_observables(
        reward,
        log_prob,
        observables=("seq_len", "log_prob_mean", "log_prob_step"),
        seq_len=torch.full_like(log_prob, float(log_prob_step.shape[-1])),
        log_prob_step=log_prob_step,
        entropy=None,
        seq_len_fallback=None,
    )

    assert "seq_len" in extra
    assert extra["seq_len"].shape == log_prob.shape
    assert torch.all(extra["seq_len"] == 3.0)
    assert "log_prob_step" in extra
    assert extra["log_prob_step"].shape == log_prob_step.shape
