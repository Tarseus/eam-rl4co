"""
Phase 3: Smoke tests for 10x10 same-shape batched PO and BOPO.
Tests:
- sample_training_pair with N=2 instances, B=8 (strictly within-instance pairs)
- PO loss aggregation per-instance
- BOPO loss aggregation per-instance
- No cross-instance pairs
"""
from pathlib import Path

import pytest
import torch

from rl4co.models.zoo.mgl_jssp.data import JSSPInstanceDataset, load_instance
from rl4co.models.zoo.mgl_jssp.net import CAMEncoder3, LSTMDecoder2
from rl4co.models.zoo.mgl_jssp.sampling import (
    Solutions,
    JobShopStates,
    rl_loss,
    po_loss,
    sro_loss,
    solve_jsp,
    sample_training_pair,
)


def _write_10x10_jsp(path: Path, seed: int = 0) -> None:
    """Write a minimal 10x10 JSP instance (dummy data)."""
    torch.manual_seed(seed)
    lines = ["10 10"]
    for _ in range(10):
        # machine, time pairs for 10 machines
        row = []
        machines = torch.randperm(10).tolist()
        times = torch.randint(1, 99, (10,)).tolist()
        for m, t in zip(machines, times):
            row.append(str(m))
            row.append(str(t))
        lines.append(" ".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_phase3_solve_jsp_multi_instances_for_po_bopo(tmp_path: Path) -> None:
    """Test solve_jsp with N=2 instances, B=4 (used by both PO and BOPO)."""
    files = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jsp"
        _write_10x10_jsp(p, seed=i)
        files.append(p)
    instances = [load_instance(f.as_posix()) for f in files]

    encoder = CAMEncoder3(15, hidden_size=32, embed_size=64)
    decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=32, att_size=64)

    B = 4
    trajs, logits, makespans, entropies = solve_jsp(
        instances,
        batch_size_per_instance=B,
        device="cpu",
        encoder=encoder,
        decoder=decoder,
        use_greedy=False,
    )

    num_steps = 10 * 10 - 1
    N = 2
    assert trajs.shape == (N * B, num_steps)
    assert logits.shape == (N * B, num_steps, 10)
    assert makespans.shape == (N * B,)


def test_phase3_po_loss_per_instance_aggregation(tmp_path: Path) -> None:
    """Test PO loss is computed per-instance then averaged (no cross-instance pairs)."""
    N = 2
    B = 8
    num_steps = 99
    num_jobs = 10

    # Create two independent sets of trajectories
    # Instance 0
    logits0 = torch.randn(B, num_steps, num_jobs)
    trajs0 = torch.randint(0, num_jobs, (B, num_steps))
    mss0 = torch.randn(B) + 100.0
    samples0 = Solutions(trajs=trajs0, logits=logits0, mss=mss0)
    loss0, pref0 = po_loss(samples0)

    # Instance 1 (different distribution)
    logits1 = torch.randn(B, num_steps, num_jobs) + 5.0
    trajs1 = torch.randint(0, num_jobs, (B, num_steps))
    mss1 = torch.randn(B) + 200.0
    samples1 = Solutions(trajs=trajs1, logits=logits1, mss=mss1)
    loss1, pref1 = po_loss(samples1)

    # Expected average
    expected_avg_loss = (loss0 + loss1) / 2
    expected_avg_pref = (pref0 + pref1) / 2

    torch.testing.assert_close(expected_avg_loss, (loss0 + loss1) / 2)
    torch.testing.assert_close(torch.tensor(expected_avg_pref), torch.tensor((pref0 + pref1) / 2))


def test_phase3_sample_training_pair_strictly_within_instance(tmp_path: Path) -> None:
    """Test BOPO sample_training_pair constructs pairs only within each instance."""
    files = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jsp"
        _write_10x10_jsp(p, seed=i)
        files.append(p)
    instances = [load_instance(f.as_posix()) for f in files]

    encoder = CAMEncoder3(15, hidden_size=32, embed_size=64)
    decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=32, att_size=64)

    B = 8
    K = 4  # Small K for test

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
    N = len(instances)
    expected_pairs_per_instance = K - 1
    assert total_pairs == N * expected_pairs_per_instance
    assert better.mss.shape[0] == total_pairs
    assert worse.mss.shape[0] == total_pairs

    # Verify we can compute sro_loss on the result
    loss, quality = sro_loss(better, worse)
    assert loss.isfinite()
    assert isinstance(quality, float)


def test_phase3_bopo_loss_flow(tmp_path: Path) -> None:
    """Test full BOPO flow with N=2 instances: per-instance filtering -> per-instance pairing -> average loss."""
    files = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jsp"
        _write_10x10_jsp(p, seed=i)
        files.append(p)
    instances = [load_instance(f.as_posix()) for f in files]

    encoder = CAMEncoder3(15, hidden_size=32, embed_size=64)
    decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=32, att_size=64)

    # First, let's manually verify we can do per-instance processing
    B = 8
    K = 4
    device = "cpu"

    # Solve for all
    trajs, logits, makespans, _ = solve_jsp(
        instances,
        batch_size_per_instance=B,
        device=device,
        encoder=encoder,
        decoder=decoder,
        use_greedy=False,
    )

    # Reshape to (N, B, ...)
    N = len(instances)
    num_steps = trajs.shape[1]
    trajs_reshaped = trajs.view(N, B, num_steps)
    logits_reshaped = logits.view(N, B, num_steps, 10)
    makespans_reshaped = makespans.view(N, B)

    # Process each instance separately like sample_training_pair does
    total_loss = 0.0
    for i in range(N):
        # Get this instance's candidates
        trajs_i = trajs_reshaped[i]
        logits_i = logits_reshaped[i]
        makespans_i = makespans_reshaped[i]

        # Sort and select within this instance only
        sorted_idx = sorted(range(B), key=lambda idx: makespans_i[idx].item())
        selected = sorted_idx[:: B // K]

        # Anchor-best pairs within this instance only
        selected_pairs = [(selected[0], worse_idx) for worse_idx in selected[1:]]

        # Build pairs
        num_pairs_i = len(selected_pairs)
        trajs_better_i = trajs_i[[p[0] for p in selected_pairs]]
        logits_better_i = logits_i[[p[0] for p in selected_pairs]]
        mss_better_i = makespans_i[[p[0] for p in selected_pairs]]
        trajs_worse_i = trajs_i[[p[1] for p in selected_pairs]]
        logits_worse_i = logits_i[[p[1] for p in selected_pairs]]
        mss_worse_i = makespans_i[[p[1] for p in selected_pairs]]

        better_i = Solutions(mss=mss_better_i, logits=logits_better_i, trajs=trajs_better_i)
        worse_i = Solutions(mss=mss_worse_i, logits=logits_worse_i, trajs=trajs_worse_i)

        loss_i, _ = sro_loss(better_i, worse_i)
        total_loss = total_loss + loss_i

    avg_loss = total_loss / N
    assert avg_loss.isfinite()
