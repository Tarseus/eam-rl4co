"""
Phase 4: Smoke tests for bucket-by-shape batching.
Tests:
- JSSPShapeBucketSampler groups by shape
- 10x10, 15x15, 20x20 all work with same-shape batching
- No cross-shape batches
"""
from pathlib import Path

import pytest
import torch

from rl4co.models.zoo.mgl_jssp.data import (
    JSSPInstanceDataset,
    JSSPShapeBucketSampler,
    load_instance,
)


def _write_jsp_of_shape(path: Path, num_jobs: int, num_machines: int, seed: int = 0) -> None:
    """Write a JSP instance of specified shape."""
    torch.manual_seed(seed)
    lines = [f"{num_jobs} {num_machines}"]
    for _ in range(num_jobs):
        # machine, time pairs for num_machines machines
        row = []
        machines = torch.randperm(num_machines).tolist()
        times = torch.randint(1, 99, (num_machines,)).tolist()
        for m, t in zip(machines, times):
            row.append(str(m))
            row.append(str(t))
        lines.append(" ".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_phase4_shape_bucket_sampler_groups_by_shape(tmp_path: Path) -> None:
    """Test that JSSPShapeBucketSampler creates same-shape batches only."""
    # Create 10x10, 15x15, 20x20 instances
    instances = []
    for shape_idx, (nj, nm) in enumerate([(10, 10), (15, 15), (20, 20)]):
        for i in range(3):
            p = tmp_path / f"inst_{nj}_{nm}_{i}.jsp"
            _write_jsp_of_shape(p, nj, nm, seed=shape_idx * 10 + i)
            instances.append(load_instance(p.as_posix()))

    dataset = JSSPInstanceDataset(instances)

    # Test bucket sampler with batch_size=2
    sampler = JSSPShapeBucketSampler(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        allowed_shapes=None,
    )

    # Check all batches are same-shape
    for batch_indices in sampler:
        shapes_in_batch = set()
        for idx in batch_indices:
            shape = dataset.get_shape(idx)
            shapes_in_batch.add(shape)
        assert len(shapes_in_batch) == 1, f"Batch has mixed shapes: {shapes_in_batch}"

    # Check bucket stats
    stats = sampler.get_bucket_stats()
    assert (10, 10) in stats
    assert (15, 15) in stats
    assert (20, 20) in stats
    assert stats[(10, 10)][0] == 3  # 3 instances
    assert stats[(15, 15)][0] == 3
    assert stats[(20, 20)][0] == 3


def test_phase4_shape_bucket_sampler_allowed_shapes(tmp_path: Path) -> None:
    """Test allowed_shapes filters correctly."""
    instances = []
    for shape_idx, (nj, nm) in enumerate([(10, 10), (15, 15), (20, 20)]):
        p = tmp_path / f"inst_{nj}_{nm}.jsp"
        _write_jsp_of_shape(p, nj, nm, seed=shape_idx)
        instances.append(load_instance(p.as_posix()))

    dataset = JSSPInstanceDataset(instances)

    # Only allow 10x10 and 15x15
    sampler = JSSPShapeBucketSampler(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        allowed_shapes=[(10, 10), (15, 15)],
    )

    stats = sampler.get_bucket_stats()
    assert (10, 10) in stats
    assert (15, 15) in stats
    assert (20, 20) not in stats


def test_phase4_10x10_15x15_20x20_same_shape_batching_work(tmp_path: Path) -> None:
    """Verify that each individual shape (10/15/20) works with same-shape batching."""
    from rl4co.models.zoo.mgl_jssp.net import CAMEncoder3, LSTMDecoder2
    from rl4co.models.zoo.mgl_jssp.sampling import solve_jsp

    for num_jobs, num_machines in [(10, 10), (15, 15), (20, 20)]:
        # Create 2 instances of this shape
        instances = []
        for i in range(2):
            p = tmp_path / f"test_{num_jobs}_{num_machines}_{i}.jsp"
            _write_jsp_of_shape(p, num_jobs, num_machines, seed=i)
            instances.append(load_instance(p.as_posix()))

        # Verify we can run solve_jsp on them
        encoder = CAMEncoder3(15, hidden_size=16, embed_size=32)
        decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=16, att_size=32)

        B = 2
        trajs, logits, makespans, entropies = solve_jsp(
            instances,
            batch_size_per_instance=B,
            device="cpu",
            encoder=encoder,
            decoder=decoder,
            use_greedy=False,
        )

        num_steps = num_jobs * num_machines - 1
        assert trajs.shape == (2 * B, num_steps)
        assert logits.shape == (2 * B, num_steps, num_jobs)
        assert makespans.shape == (2 * B,)
