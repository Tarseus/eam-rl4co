"""
Phase 6: Full validation for bucketed batching.
Tests:
- RL/PO/BOPO all work with batch_size=2 for 10x10/15x15/20x20
- Multi-shape dataset: bucket sampler groups correctly, no mixed-shape batches
- End-to-end training rollout (fake, no optimizer step)
"""
from pathlib import Path

import pytest
import torch

from rl4co.models.zoo.mgl_jssp.data import (
    JSSPInstanceDataset,
    JSSPShapeBucketSampler,
    load_instance,
)
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


def _make_fake_dataset(tmp_path: Path) -> tuple[JSSPInstanceDataset, dict]:
    """Create mixed-shape dataset: 2x10x10, 2x15x15, 2x20x20."""
    instances = []
    shape_counts = {}

    for shape_idx, (nj, nm) in enumerate([(10, 10), (15, 15), (20, 20)]):
        shape_counts[(nj, nm)] = 2
        for i in range(2):
            p = tmp_path / f"inst_{nj}_{nm}_{i}.jsp"
            _write_jsp_of_shape(p, nj, nm, seed=shape_idx * 10 + i)
            instances.append(load_instance(p.as_posix()))

    dataset = JSSPInstanceDataset(instances)
    return dataset, shape_counts


# =============================================================================
# Test 1: RL with batch_size=2 for all shapes
# =============================================================================

@pytest.mark.parametrize("shape", [(10, 10), (15, 15), (20, 20)])
def test_phase6_rl_batch_size_2_all_shapes(tmp_path: Path, shape: tuple[int, int]) -> None:
    """RL: batch_size=2 works for 10/15/20."""
    num_jobs, num_machines = shape
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

    # Reshape and compute per-instance loss
    num_steps = num_jobs * num_machines - 1
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


# =============================================================================
# Test 2: PO with batch_size=2 for all shapes
# =============================================================================

@pytest.mark.parametrize("shape", [(10, 10), (15, 15), (20, 20)])
def test_phase6_po_batch_size_2_all_shapes(tmp_path: Path, shape: tuple[int, int]) -> None:
    """PO: batch_size=2 works for 10/15/20."""
    num_jobs, num_machines = shape
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

    # Reshape and compute per-instance loss
    num_steps = num_jobs * num_machines - 1
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


# =============================================================================
# Test 3: BOPO with batch_size=2 for all shapes
# =============================================================================

@pytest.mark.parametrize("shape", [(10, 10), (15, 15), (20, 20)])
def test_phase6_bopo_batch_size_2_all_shapes(tmp_path: Path, shape: tuple[int, int]) -> None:
    """BOPO: batch_size=2 works for 10/15/20."""
    num_jobs, num_machines = shape
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

    loss, quality = sro_loss(better, worse)
    assert loss.isfinite()
    assert total_pairs == 2 * (K - 1)  # 2 instances * (K-1) pairs each


# =============================================================================
# Test 4: Multi-shape bucket sampler correctly groups, no mixed batches
# =============================================================================

def test_phase6_multi_shape_bucket_sampler_no_mixed_batches(tmp_path: Path) -> None:
    """Mixed-shape dataset: bucket sampler creates only same-shape batches."""
    dataset, shape_counts = _make_fake_dataset(tmp_path)

    sampler = JSSPShapeBucketSampler(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        allowed_shapes=None,
    )

    # Verify all batches are same-shape
    batch_shapes = []
    for batch_indices in sampler:
        shapes_in_batch = set()
        for idx in batch_indices:
            shape = dataset.get_shape(idx)
            shapes_in_batch.add(shape)
        assert len(shapes_in_batch) == 1, f"Mixed-shape batch: {shapes_in_batch}"
        batch_shapes.append(shapes_in_batch.pop())

    # Verify we have batches from all shapes
    shapes_seen = set(batch_shapes)
    assert (10, 10) in shapes_seen
    assert (15, 15) in shapes_seen
    assert (20, 20) in shapes_seen


# =============================================================================
# Test 5: End-to-end fake training rollout (RL/PO/BOPO)
# =============================================================================

@pytest.mark.parametrize("baseline", ["rl", "po", "bopo"])
def test_phase6_end_to_end_fake_training_rollout(tmp_path: Path, baseline: str) -> None:
    """End-to-end: mimic model._training_rollout for all baselines."""
    num_jobs, num_machines = 15, 15
    instances = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jsp"
        _write_jsp_of_shape(p, num_jobs, num_machines, seed=i)
        instances.append(load_instance(p.as_posix()))

    encoder = CAMEncoder3(15, hidden_size=16, embed_size=32)
    decoder = LSTMDecoder2(encoder.out_size, context_size=11, hidden_size=16, att_size=32)

    B = 4
    device = "cpu"

    if baseline == "rl":
        trajs, logits, makespans, _ = solve_jsp(
            instances, batch_size_per_instance=B, device=device,
            encoder=encoder, decoder=decoder, use_greedy=False,
        )
        num_steps = trajs.size(1)
        trajs_reshaped = trajs.view(2, B, num_steps)
        logits_reshaped = logits.view(2, B, num_steps, num_jobs)
        makespans_reshaped = makespans.view(2, B)
        total_loss = 0.0
        for i in range(2):
            samples_i = Solutions(trajs=trajs_reshaped[i], logits=logits_reshaped[i], mss=makespans_reshaped[i])
            loss_i, _ = rl_loss(samples_i)
            total_loss = total_loss + loss_i
        avg_loss = total_loss / 2
        assert avg_loss.isfinite()

    elif baseline == "po":
        trajs, logits, makespans, _ = solve_jsp(
            instances, batch_size_per_instance=B, device=device,
            encoder=encoder, decoder=decoder, use_greedy=False,
        )
        num_steps = trajs.size(1)
        trajs_reshaped = trajs.view(2, B, num_steps)
        logits_reshaped = logits.view(2, B, num_steps, num_jobs)
        makespans_reshaped = makespans.view(2, B)
        total_loss = 0.0
        for i in range(2):
            samples_i = Solutions(trajs=trajs_reshaped[i], logits=logits_reshaped[i], mss=makespans_reshaped[i])
            loss_i, _ = po_loss(samples_i, impl="bt")
            total_loss = total_loss + loss_i
        avg_loss = total_loss / 2
        assert avg_loss.isfinite()

    elif baseline == "bopo":
        K = 2
        better, worse, best_makespan, total_pairs = sample_training_pair(
            instances, encoder, decoder, B=B, K=K,
            use_greedy=False, pair_mode="anchor_best", device=device,
        )
        loss, quality = sro_loss(better, worse)
        assert loss.isfinite()
        assert total_pairs == 2 * (K - 1)
