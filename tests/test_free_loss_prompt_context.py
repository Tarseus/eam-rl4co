from pathlib import Path
import sys


def test_runtime_prompt_context_respects_configured_observables() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "PTP"))

    from ptp_discovery.free_loss_llm_ops import build_runtime_prompt_context

    ctx = build_runtime_prompt_context(
        loss_observables=("seq_len", "log_prob_mean", "advantage"),
        mode="pairwise",
    )

    available = set(ctx["available_keys"])
    blocked = set(ctx["blocked_optional_keys"])

    assert "advantage_gap" in available
    assert "log_prob_mean_gap" in available
    assert "seq_len_gap" in available
    assert "entropy_gap" in blocked
    assert "log_prob_step_w" in blocked
