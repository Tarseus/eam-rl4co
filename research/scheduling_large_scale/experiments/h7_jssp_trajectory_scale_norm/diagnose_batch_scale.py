from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_jssp_large_objectives import (
    _build_model,
    _dynamic_instance,
    _gradient_norm,
    _seed,
)


def _stats(value: torch.Tensor) -> dict[str, float]:
    detached = value.detach().float()
    return {
        "min": float(detached.min().cpu()),
        "mean": float(detached.mean().cpu()),
        "max": float(detached.max().cpu()),
        "std": float(detached.std(unbiased=False).cpu()),
    }


def main() -> None:
    device = torch.device("cuda:0")
    args = SimpleNamespace(
        num_jobs=50,
        num_machines=20,
        method="usw",
        rollouts=128,
        select_k=16,
        po_alpha=0.25,
        alpha=1.0,
        eval_rollouts=128,
        greedy=0,
        usw_pair=Path(
            "research/scheduling_large_scale/experiments/"
            "h7_jssp_trajectory_scale_norm/artifacts/"
            "usw_source_length_norm/best_pair.json"
        ),
        asw_pair=Path("unused.json"),
    )
    checkpoint = REPO_ROOT / "downloads/jssp15x15/weighting/checkpoint.ckpt"
    model, _ = _build_model(args, checkpoint)
    model = model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    original_loss = model.free_loss.loss_fn
    captured: dict[str, object] = {}

    def diagnostic_loss(batch, model_output, extra):
        advantage_gap = batch["advantage_gap"]
        cost_a = batch["cost_a"]
        cost_b = batch["cost_b"]
        mean_gap = advantage_gap.abs().mean().clamp_min(1e-6)
        mean_log_prob_diff = batch["log_prob_w_mean"] - batch["log_prob_l_mean"]
        source_diff = 100.0 * mean_log_prob_diff
        h7_logit = source_diff / mean_gap
        bopo_logit = (cost_b / cost_a.clamp_min(1e-8)) * mean_log_prob_diff
        relative_gap = (cost_b - cost_a) / cost_a.clamp_min(1e-8)
        captured.update(
            pair_count=int(advantage_gap.numel()),
            advantage_gap=_stats(advantage_gap),
            cost_a=_stats(cost_a),
            cost_b=_stats(cost_b),
            relative_gap=_stats(relative_gap),
            mean_log_prob_diff=_stats(mean_log_prob_diff),
            source_diff=_stats(source_diff),
            h7_logit=_stats(h7_logit),
            bopo_logit=_stats(bopo_logit),
            mean_abs_advantage_gap=float(mean_gap.detach().cpu()),
        )
        return original_loss(batch=batch, model_output=model_output, extra=extra)

    model.free_loss.loss_fn = diagnostic_loss
    instance = _dynamic_instance(jobs=50, machines=20, seed=12345678, instance_index=0)
    _seed(12345678, device)
    loss, _, _, pair_count = model._training_rollout([instance])
    loss.backward()
    grad_norm = _gradient_norm(model.parameters())
    captured.update(
        loss=float(loss.detach().cpu()),
        grad_norm=grad_norm,
        returned_pair_count=float(pair_count.detach().cpu()),
        finite_loss=math.isfinite(float(loss.detach().cpu())),
        finite_grad=math.isfinite(grad_norm),
        positive_grad=grad_norm > 0.0,
        peak_memory_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
    )
    print(json.dumps(captured, sort_keys=True))


if __name__ == "__main__":
    main()
