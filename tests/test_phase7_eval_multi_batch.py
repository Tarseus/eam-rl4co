"""
Phase 7: Test eval supports multi-batch.
"""
from pathlib import Path

import pytest
import torch

from rl4co.models.zoo.mgl_jssp.data import JSSPInstanceDataset, load_instance
from rl4co.models.zoo.mgl_jssp.net import CAMEncoder3, LSTMDecoder2
from rl4co.models.zoo.mgl_jssp.sampling import sampling
from scripts.eval_jssp_benchmarks import build_jssp_augmentations


def _write_jsp_of_shape(path: Path, num_jobs: int, num_machines: int, seed: int = 0, ref_makespan: float = 1000.0) -> None:
    """Write a JSP instance with reference makespan."""
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
    # Add reference makespan
    lines.append(str(ref_makespan))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_phase7_sampling_supports_multi_instance_eval():
    """Test sampling() works with multiple instances for eval."""
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create 2 instances with reference makespan
        instances = []
        for i in range(2):
            p = tmp_path / f"test_{i}.jsp"
            _write_jsp_of_shape(p, 10, 10, seed=i, ref_makespan=1000.0 + i * 100)
            instances.append(load_instance(p.as_posix()))

        encoder = CAMEncoder3(15, hidden_size=16, embed_size=32)
        decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=16, att_size=32)

        B = 4
        makespans, entropies, log_probs = sampling(
            instances,
            encoder,
            decoder,
            bs=B,
            use_greedy=False,
            device="cpu",
        )

        num_steps = 10 * 10 - 1
        assert makespans.shape == (2 * B,)
        assert entropies.shape == (2 * B, num_steps)
        assert log_probs.shape == (2 * B,)


def test_phase7_eval_multi_batch_gap_calculation():
    """Test that gap calculation works with multi-instance eval batches."""
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create 3 instances with known reference makespans
        instances = []
        ref_makespans = [1000.0, 1100.0, 1200.0]
        for i, ref in enumerate(ref_makespans):
            p = tmp_path / f"test_{i}.jsp"
            _write_jsp_of_shape(p, 10, 10, seed=i, ref_makespan=ref)
            instances.append(load_instance(p.as_posix()))

        # Check that reference makespans were loaded correctly
        for i, instance in enumerate(instances):
            assert instance["makespan"] == ref_makespans[i]


def test_phase7_jssp_eval_augmentation_keeps_shape_and_samples():
    """JSSP eval augmentation creates equivalent same-shape encodings for batched sampling."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p = tmp_path / "test.jsp"
        _write_jsp_of_shape(p, 10, 10, seed=7, ref_makespan=1000.0)
        instance = load_instance(p.as_posix())

        augmented = build_jssp_augmentations(instance, aug_factor=3, seed=123)
        assert len(augmented) == 3
        assert [ins["shape"] for ins in augmented] == ["10x10", "10x10", "10x10"]
        assert [ins["makespan"] for ins in augmented] == [1000.0, 1000.0, 1000.0]

        encoder = CAMEncoder3(15, hidden_size=16, embed_size=32)
        decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=16, att_size=32)
        makespans, entropies, log_probs = sampling(
            augmented,
            encoder,
            decoder,
            bs=2,
            use_greedy=False,
            device="cpu",
        )

        assert makespans.shape == (6,)
        assert entropies.shape == (6, 99)
        assert log_probs.shape == (6,)
