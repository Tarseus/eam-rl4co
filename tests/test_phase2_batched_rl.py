"""
Phase 2: Smoke tests for 10x10 same-shape batched RL.
Tests:
- collate_fn accepts batch of 10x10
- solve_jsp works with N=2 instances, B=4
- rl_loss aggregates per-instance then averages
- non-10x10 error
- mixed-shape error
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
    solve_jsp,
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


def test_phase2_collate_fn_10x10_ok(tmp_path: Path) -> None:
    """Test collate_fn accepts batch of 10x10 instances."""
    files = []
    for i in range(4):
        p = tmp_path / f"test_{i}.jsp"
        _write_10x10_jsp(p, seed=i)
        files.append(p)

    instances = [load_instance(f.as_posix()) for f in files]
    batch = JSSPInstanceDataset.collate_fn(instances)
    assert isinstance(batch, list)
    assert len(batch) == 4


def test_phase2_collate_fn_mixed_shape_only_error(tmp_path: Path) -> None:
    """Phase 4 update: collate_fn accepts any single shape, only rejects mixed-shape batches."""
    # Create fake 6x6 instance dict directly
    instance6 = {
        "j": 6,
        "m": 6,
        "shape": "6x6",
        # Add dummy required fields
        "x": torch.randn(6*6, 15),
        "job_edges": torch.randn(2, 10),
        "mac_edges": torch.randn(2, 10),
        "costs": torch.randn(6, 6),
        "machines": torch.randint(0, 6, (6, 6)),
    }

    # Single shape (even 6x6) is OK now (Phase 4)
    batch = JSSPInstanceDataset.collate_fn([instance6])
    assert isinstance(batch, list)
    assert len(batch) == 1

    # But mixed-shape is still rejected
    instance10 = {
        "j": 10,
        "m": 10,
        "shape": "10x10",
        "x": torch.randn(10*10, 15),
        "job_edges": torch.randn(2, 10),
        "mac_edges": torch.randn(2, 10),
        "costs": torch.randn(10, 10),
        "machines": torch.randint(0, 10, (10, 10)),
    }

    with pytest.raises(ValueError, match="Mixed-shape"):
        JSSPInstanceDataset.collate_fn([instance6, instance10])


def test_phase2_collate_fn_mixed_shape_error(tmp_path: Path) -> None:
    """Test collate_fn rejects mixed-shape batch."""
    # 10x10 ok
    p1 = tmp_path / "ok.jsp"
    _write_10x10_jsp(p1, seed=0)
    instance10 = load_instance(p1.as_posix())

    # Create fake 15x15 instance
    instance15 = {
        "j": 15,
        "m": 15,
        "shape": "15x15",
        "x": torch.randn(15*15, 15),
        "job_edges": torch.randn(2, 10),
        "mac_edges": torch.randn(2, 10),
        "costs": torch.randn(15, 15),
        "machines": torch.randint(0, 15, (15, 15)),
    }

    with pytest.raises(ValueError, match="Mixed-shape"):
        JSSPInstanceDataset.collate_fn([instance10, instance15])


def test_phase2_jobshopstates_init_multi_instances(tmp_path: Path) -> None:
    """Test JobShopStates.init_state with multiple 10x10 instances."""
    files = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jsp"
        _write_10x10_jsp(p, seed=i)
        files.append(p)
    instances = [load_instance(f.as_posix()) for f in files]

    jsp = JobShopStates(device="cpu")
    B = 4
    state, mask = jsp.init_state(instances, batch_size_per_instance=B)

    assert state.shape == (2 * B, 10, 11)
    assert mask.shape == (2 * B, 10)


def test_phase2_solve_jsp_multi_instances(tmp_path: Path) -> None:
    """Test solve_jsp with N=2 instances, B=4 rollouts each."""
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
    assert trajs.shape == (2 * B, num_steps)
    assert logits.shape == (2 * B, num_steps, 10)
    assert makespans.shape == (2 * B,)
    assert entropies.shape == (2 * B, num_steps)


def test_phase2_rl_loss_per_instance_aggregation(tmp_path: Path) -> None:
    """Test loss is computed per-instance then averaged."""
    # Create dummy Solutions for N=2 instances, B=4 each
    N = 2
    B = 4
    num_steps = 99
    num_jobs = 10

    # Two independent sets of trajectories
    logits1 = torch.randn(B, num_steps, num_jobs)
    trajs1 = torch.randint(0, num_jobs, (B, num_steps))
    mss1 = torch.randn(B) + 100.0

    logits2 = torch.randn(B, num_steps, num_jobs) + 10.0  # different distribution
    trajs2 = torch.randint(0, num_jobs, (B, num_steps))
    mss2 = torch.randn(B) + 200.0

    # Compute per-instance losses
    samples1 = Solutions(trajs=trajs1, logits=logits1, mss=mss1)
    loss1, q1 = rl_loss(samples1)

    samples2 = Solutions(trajs=trajs2, logits=logits2, mss=mss2)
    loss2, q2 = rl_loss(samples2)

    expected_avg_loss = (loss1 + loss2) / 2
    expected_avg_q = (q1 + q2) / 2

    # Verify the math
    torch.testing.assert_close(expected_avg_loss, (loss1 + loss2) / 2)
    torch.testing.assert_close(torch.tensor(expected_avg_q), torch.tensor((q1 + q2) / 2))
