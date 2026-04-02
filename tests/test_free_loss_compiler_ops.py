from pathlib import Path
import sys

import pytest
import torch


repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "PTP"))

from ptp_discovery.free_loss_compiler import CompileError, compile_free_loss
from ptp_discovery.free_loss_ir import ir_from_json


def _make_ir(*, code: str, operators_used: list[str]):
    return ir_from_json(
        {
            "name": "test_loss",
            "intuition": "test",
            "pseudocode": "test",
            "hyperparams": {},
            "operators_used": operators_used,
            "implementation_hint": {
                "expects": ["log_prob_w", "log_prob_l", "weight", "advantage_gap"],
                "returns": "scalar",
                "mode": "pairwise",
            },
            "code": code,
        }
    )


def test_compile_free_loss_supports_extended_ops_namespace() -> None:
    ir = _make_ir(
        code=(
            "def generated_loss(batch, model_output, extra):\n"
            "    lpw = batch['log_prob_w']\n"
            "    lpl = batch['log_prob_l']\n"
            "    adv = batch['advantage_gap']\n"
            "    weight = batch['weight']\n"
            "    eps = 1e-6\n"
            "    den = ops.maximum(ops.abs(adv), ops.ones_like(adv) * eps)\n"
            "    scaled = ops.div(ops.sub(lpw, lpl), den)\n"
            "    hi = ops.max(scaled)\n"
            "    lo = ops.min(scaled)\n"
            "    signed = ops.sign(adv)\n"
            "    penalty = ops.norm(adv, p=2) + eps\n"
            "    loss = ops.softplus(ops.mul(signed, scaled))\n"
            "    loss = ops.add(loss, ops.mul(ops.maximum(hi - lo, hi.new_zeros(())), 0.0))\n"
            "    loss = ops.div(loss, penalty)\n"
            "    return ops.mean(ops.mul(loss, weight))\n"
        ),
        operators_used=["abs", "sub", "div", "maximum", "max", "min", "sign", "norm", "softplus", "add", "mul", "mean", "ones_like"],
    )
    compiled = compile_free_loss(ir)
    batch = {
        "log_prob_w": torch.tensor([0.5, -0.1, 0.2], requires_grad=True),
        "log_prob_l": torch.tensor([-0.2, -0.3, 0.1], requires_grad=True),
        "weight": torch.ones(3),
        "advantage_gap": torch.tensor([1.0, -2.0, 0.5]),
    }
    loss = compiled.loss_fn(batch=batch, model_output={}, extra={})
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert batch["log_prob_w"].grad is not None
    assert batch["log_prob_l"].grad is not None


def test_compile_free_loss_supports_norm_ord_and_scalar_maximum() -> None:
    ir = _make_ir(
        code=(
            "def generated_loss(batch, model_output, extra):\n"
            "    lpw = batch['log_prob_w']\n"
            "    lpl = batch['log_prob_l']\n"
            "    adv = batch['advantage_gap']\n"
            "    diff = ops.sub(lpw, lpl)\n"
            "    scale = ops.maximum(ops.norm(adv, ord=2), 1e-6)\n"
            "    return ops.mean(ops.div(diff, scale))\n"
        ),
        operators_used=["sub", "norm", "maximum", "div", "mean"],
    )
    compiled = compile_free_loss(ir)
    batch = {
        "log_prob_w": torch.tensor([0.5, -0.1, 0.2], requires_grad=True),
        "log_prob_l": torch.tensor([-0.2, -0.3, 0.1], requires_grad=True),
        "weight": torch.ones(3),
        "advantage_gap": torch.tensor([1.0, -2.0, 0.5]),
    }
    loss = compiled.loss_fn(batch=batch, model_output={}, extra={})
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert batch["log_prob_w"].grad is not None


def test_compile_free_loss_allows_safe_ones_like_alias() -> None:
    ir = _make_ir(
        code=(
            "def generated_loss(batch, model_output, extra):\n"
            "    lpw = batch['log_prob_w']\n"
            "    lpl = batch['log_prob_l']\n"
            "    weight = batch.get('weight', ones_like(lpw))\n"
            "    loss = ops.softplus(ops.sub(lpw, lpl))\n"
            "    return ops.mean(ops.mul(loss, weight))\n"
        ),
        operators_used=["softplus", "sub", "mul", "mean", "ones_like"],
    )
    compiled = compile_free_loss(ir)
    batch = {
        "log_prob_w": torch.tensor([0.5, -0.1, 0.2], requires_grad=True),
        "log_prob_l": torch.tensor([-0.2, -0.3, 0.1], requires_grad=True),
        "advantage_gap": torch.tensor([1.0, -2.0, 0.5]),
    }
    loss = compiled.loss_fn(batch=batch, model_output={}, extra={})
    assert torch.isfinite(loss)


def test_compile_free_loss_rejects_disallowed_tensor_method_ops() -> None:
    ir = _make_ir(
        code=(
            "def generated_loss(batch, model_output, extra):\n"
            "    lpw = batch['log_prob_w']\n"
            "    lpl = batch['log_prob_l']\n"
            "    adv = batch['advantage_gap']\n"
            "    return (lpw - lpl).max() + adv.norm(p=2)\n"
        ),
        operators_used=["max", "norm", "sub"],
    )
    with pytest.raises(CompileError, match=r"ops\.max|ops\.norm"):
        compile_free_loss(ir)
