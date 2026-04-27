from pathlib import Path
import sys

import torch


def _setup_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "PTP"))


def test_prepare_pairwise_loss_batch_derives_advantage_family_from_costs() -> None:
    _setup_imports()

    from fitness.free_loss_fidelity import prepare_pairwise_loss_batch

    full_batch = {
        "log_prob_w": torch.tensor([-0.5, -1.0], dtype=torch.float32),
        "log_prob_l": torch.tensor([-1.5, -2.0], dtype=torch.float32),
        "cost_a": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "cost_b": torch.tensor([3.0, 5.0], dtype=torch.float32),
    }

    batch = prepare_pairwise_loss_batch(
        full_batch,
        expects=("cost_gap", "advantage_w", "advantage_l", "advantage_gap", "weight"),
    )

    assert torch.equal(batch["cost_gap"], torch.tensor([2.0, 3.0]))
    assert torch.equal(batch["advantage_w"], torch.tensor([-1.0, -2.0]))
    assert torch.equal(batch["advantage_l"], torch.tensor([-3.0, -5.0]))
    assert torch.equal(batch["advantage_gap"], torch.tensor([-2.0, -3.0]))
    assert torch.equal(batch["advantage_gap"], -batch["cost_gap"])
    assert torch.equal(batch["weight"], torch.ones_like(batch["cost_gap"]))


def test_co_gates_do_not_raise_missing_key_for_advantage_gap_losses() -> None:
    _setup_imports()

    from ptp_discovery.free_loss_compiler import CompiledFreeLoss
    from ptp_discovery.free_loss_gates import (
        run_affine_invariance_gate,
        run_objective_sensitivity_gate,
    )
    from ptp_discovery.free_loss_ir import FreeLossIR, FreeLossImplementationHint

    ir = FreeLossIR(
        name="advantage_gap_probe",
        intuition="Probe advantage-gap access inside CO gates.",
        pseudocode="loss = mean((log_prob_w - log_prob_l) * advantage_gap)",
        hyperparams={},
        operators_used=["mul", "sub", "mean"],
        implementation_hint=FreeLossImplementationHint(
            expects=["log_prob_w", "log_prob_l", "weight", "advantage_gap"],
            returns="scalar",
            mode="pairwise",
        ),
    )
    compiled = CompiledFreeLoss(
        ir=ir,
        loss_fn=lambda batch, model_output, extra: (
            (batch["log_prob_w"] - batch["log_prob_l"]) * batch["advantage_gap"] * batch["weight"]
        ).mean(),
    )

    sensitivity = run_objective_sensitivity_gate(compiled)
    affine = run_affine_invariance_gate(compiled)

    assert not str(sensitivity.reason).startswith("missing_key:")
    assert not str(affine.reason).startswith("missing_key:")
