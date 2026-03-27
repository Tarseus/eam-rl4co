"""
Phase 5: Smoke tests for RL/PO/BOPO all paths with bucketed batching.
Tests:
- RL path works with bucketed batches
- PO path works with bucketed batches
- BOPO path works with bucketed batches
- All paths enforce within-instance only pairs
"""
from pathlib import Path

import pytest
import torch

from rl4co.models.zoo.mgl_jssp.data import JSSPInstanceDataset, load_instance
from rl4co.models.zoo.mgl_jssp.net import CAMEncoder3, LSTMDecoder2
from rl4co.models.zoo.mgl_jssp.sampling import (
    Solutions,
    rl_loss,
    po_loss,
    sro_loss,
    solve_jsp,
    sample_training_pair,
)


def _write_jsp_of_shape(path: Path, num_jobs: int, num_machines: int, seed: int = 0) -> None:
    """Write a JSP instance of specified shape."""
    torch.manual_seed(seed)
    lines = [f"{num_jobs} {num_machines}"]
    for _ in range(num_jobs):
        row = []
        machines = torch.randperm(num_machines).tolist()
        times = torch.randint(1, 99, (num_machines,)).tolist()
        for m, t in zip(machines, times):
            row.append(str(m))
            row.append(str(t))
        lines.append(" ".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize("baseline", ["rl", "po", "bopo"])
@pytest.mark.parametrize("shape", [(10, 10), (15, 15), (20, 20)])
def test_phase5_all_paths_all_shapes_work(tmp_path: Path, baseline: str, shape: tuple[int, int]) -> None:
    """Test RL/PO/BOPO all work with 10/15/20 shapes."""
    num_jobs, num_machines = shape

    # Create 2 instances of this shape
    instances = []
    for i in range(2):
        p = tmp_path / f"test_{num_jobs}_{num_machines}_{i}.jsp"
        _write_jsp_of_shape(p, num_jobs, num_machines, seed=i)
        instances.append(load_instance(p.as_posix()))

    encoder = CAMEncoder3(15, hidden_size=16, embed_size=32)
    decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=16, att_size=32)

    B = 4
    device = "cpu"

    if baseline == "rl":
        # RL path
        trajs, logits, makespans, _ = solve_jsp(
            instances,
            batch_size_per_instance=B,
            device=device,
            encoder=encoder,
            decoder=decoder,
            use_greedy=False,
        )

        # Reshape and compute per-instance loss
        num_steps = trajs.size(1)
        trajs_reshaped = trajs.view(2, B, num_steps)
        logits_reshaped = logits.view(2, B, num_steps, num_jobs)
        makespans_reshaped = makespans.view(2, B)

        total_loss = 0.0
        for i in range(2):
            samples_i = Solutions(
                trajs=trajs_reshaped[i],
                logits=logits_reshaped[i],
                mss=makespans_reshaped[i]
            )
            loss_i, _ = rl_loss(samples_i)
            total_loss = total_loss + loss_i

        avg_loss = total_loss / 2
        assert avg_loss.isfinite()

    elif baseline == "po":
        # PO path
        trajs, logits, makespans, _ = solve_jsp(
            instances,
            batch_size_per_instance=B,
            device=device,
            encoder=encoder,
            decoder=decoder,
            use_greedy=False,
        )

        # Reshape and compute per-instance loss
        num_steps = trajs.size(1)
        trajs_reshaped = trajs.view(2, B, num_steps)
        logits_reshaped = logits.view(2, B, num_steps, num_jobs)
        makespans_reshaped = makespans.view(2, B)

        total_loss = 0.0
        for i in range(2):
            samples_i = Solutions(
                trajs=trajs_reshaped[i],
                logits=logits_reshaped[i],
                mss=makespans_reshaped[i]
            )
            loss_i, _ = po_loss(samples_i, impl="bt")
            total_loss = total_loss + loss_i

        avg_loss = total_loss / 2
        assert avg_loss.isfinite()

    elif baseline == "bopo":
        # BOPO path
        K = 2  # Small K for test
        better, worse, best_makespan, total_pairs = sample_training_pair(
            instances,
            encoder,
            decoder,
            B=B,
            K=K,
            use_greedy=False,
            pair_mode="anchor_best",
            device=device,
        )

        loss, quality = sro_loss(better, worse)
        assert loss.isfinite()
        assert total_pairs > 0


def test_phase5_no_cross_instance_candidates_bopo(tmp_path: Path) -> None:
    """Verify BOPO does NOT mix candidates across instances."""
    num_jobs, num_machines = 10, 10

    # Create 2 instances
    instances = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jsp"
        _write_jsp_of_shape(p, num_jobs, num_machines, seed=i)
        instances.append(load_instance(p.as_posix()))

    encoder = CAMEncoder3(15, hidden_size=16, embed_size=32)
    decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=16, att_size=32)

    B = 8
    K = 4
    better, worse, best_makespan, total_pairs = sample_training_pair(
        instances,
        encoder,
        decoder,
        B=B,
        K=K,
        use_greedy=False,
        pair_mode="anchor_best",
        device="cpu",
    )

    # Each instance contributes (K-1) pairs for anchor_best
    # 2 instances * (4-1) = 6 pairs
    assert total_pairs == 2 * (K - 1)


def test_phase5_no_cross_instance_candidates_po(tmp_path: Path) -> None:
    """Verify PO does NOT mix candidates across instances (we call po_loss per-instance)."""
    num_jobs, num_machines = 10, 10

    # Create 2 instances
    instances = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jsp"
        _write_jsp_of_shape(p, num_jobs, num_machines, seed=i)
        instances.append(load_instance(p.as_posix()))

    encoder = CAMEncoder3(15, hidden_size=16, embed_size=32)
    decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=16, att_size=32)

    B = 4
    trajs, logits, makespans, _ = solve_jsp(
        instances,
        batch_size_per_instance=B,
        device="cpu",
        encoder=encoder,
        decoder=decoder,
        use_greedy=False,
    )

    # Reshape to per-instance
    num_steps = trajs.size(1)
    trajs_reshaped = trajs.view(2, B, num_steps)
    logits_reshaped = logits.view(2, B, num_steps, num_jobs)
    makespans_reshaped = makespans.view(2, B)

    # We compute po_loss SEPARATELY for each instance (no cross-instance pairs!)
    for i in range(2):
        samples_i = Solutions(
            trajs=trajs_reshaped[i],
            logits=logits_reshaped[i],
            mss=makespans_reshaped[i]
        )
        loss_i, _ = po_loss(samples_i, impl="bt")
        assert loss_i.isfinite()
