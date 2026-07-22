"""
Phase 6: Smoke tests for bucketed batching
Covers:
- RL: batch_size=2, shapes 10x10 / 15x15 / 20x20
- PO: batch_size=2, shapes 10x10 / 15x15 / 20x20
- BOPO: batch_size=2, shapes 10x10 / 15x15 / 20x20
- Multi-shape bucketed training smoke test
"""
import pytest
import torch
from torch.utils.data import DataLoader

from rl4co.envs import JSSPEnv
from rl4co.models.rl.reinforce.reinforce import REINFORCE


class _NoBaseline:
    def eval(self, td, reward, env):
        return 0, 0


def _make_reinforce(loss_mode: str):
    model = REINFORCE.__new__(REINFORCE)
    model.baseline = _NoBaseline()
    model.advantage_scaler = lambda x: x
    model.loss_mode = loss_mode
    model.env = None
    return model


def _flatten_db(x_db: torch.Tensor) -> torch.Tensor:
    d, b = x_db.shape
    return torch.stack([x_db[:, j] for j in range(b)], dim=0).reshape(d * b)


def _test_single_shape_single_mode(shape: tuple[int, int], loss_mode: str, batch_size: int = 2):
    """Test single shape with single loss mode"""
    num_jobs, num_machines = shape

    # Create env with single shape bucket
    env = JSSPEnv(
        bucketed_training=True,
        shape_buckets=[shape],
        generator_params={"num_jobs": num_jobs, "num_machines": num_machines},
    )
    model = _make_reinforce(loss_mode=loss_mode)

    # Create dataset
    dataset = env.dataset(batch_size=batch_size * 2, phase="train")
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.get_batch_sampler(batch_size=batch_size, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    num_starts = 8 if loss_mode in ["po", "bopo"] else 1

    for batch in dataloader:
        d = batch.batch_size[0]
        assert d == batch_size

        # Verify shape
        assert batch["start_op_per_job"].shape[-1] == num_jobs
        assert batch["proc_times"].shape[-2] == num_machines
        assert batch["bucket_num_jobs"].eq(num_jobs).all()
        assert batch["bucket_num_machines"].eq(num_machines).all()

        # Simulate loss calculation
        base_reward = -torch.rand(d) * 1000
        if num_starts > 1:
            reward_db = base_reward.unsqueeze(1) - torch.arange(num_starts, dtype=torch.float32).unsqueeze(0)
            ll_db = torch.randn(d, num_starts)
        else:
            reward_db = base_reward.unsqueeze(1)
            ll_db = torch.randn(d, 1)

        out = model.calculate_loss(
            td={},
            batch=batch,
            policy_out={
                "reward": _flatten_db(reward_db),
                "log_likelihood": _flatten_db(ll_db),
                "batch_size": d,
                "num_starts": num_starts,
            },
        )

        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["reinforce_loss"])


# ========== RL tests ==========
def test_rl_10x10_batch2():
    """Test RL with 10x10, batch_size=2"""
    _test_single_shape_single_mode((10, 10), "rl", batch_size=2)


def test_rl_15x15_batch2():
    """Test RL with 15x15, batch_size=2"""
    _test_single_shape_single_mode((15, 15), "rl", batch_size=2)


def test_rl_20x20_batch2():
    """Test RL with 20x20, batch_size=2"""
    _test_single_shape_single_mode((20, 20), "rl", batch_size=2)


# ========== PO tests ==========
def test_po_10x10_batch2():
    """Test PO with 10x10, batch_size=2"""
    _test_single_shape_single_mode((10, 10), "po", batch_size=2)


def test_po_15x15_batch2():
    """Test PO with 15x15, batch_size=2"""
    _test_single_shape_single_mode((15, 15), "po", batch_size=2)


def test_po_20x20_batch2():
    """Test PO with 20x20, batch_size=2"""
    _test_single_shape_single_mode((20, 20), "po", batch_size=2)


# ========== BOPO tests ==========
def test_bopo_10x10_batch2():
    """Test BOPO with 10x10, batch_size=2"""
    _test_single_shape_single_mode((10, 10), "bopo", batch_size=2)


def test_bopo_15x15_batch2():
    """Test BOPO with 15x15, batch_size=2"""
    _test_single_shape_single_mode((15, 15), "bopo", batch_size=2)


def test_bopo_20x20_batch2():
    """Test BOPO with 20x20, batch_size=2"""
    _test_single_shape_single_mode((20, 20), "bopo", batch_size=2)


# ========== Multi-shape bucketed test ==========
def test_multi_shape_bucketed_training():
    """Test multi-shape bucketed training: dataset mixes 10x10/15x15/20x20, dataloader auto-buckets"""
    print("=" * 70)
    print("Testing multi-shape bucketed training")
    print("=" * 70)

    # Create env with all three shape buckets
    env = JSSPEnv(
        bucketed_training=True,
        shape_buckets=[[10, 10], [15, 15], [20, 20]],
    )
    model_rl = _make_reinforce(loss_mode="rl")
    model_po = _make_reinforce(loss_mode="po")
    model_bopo = _make_reinforce(loss_mode="bopo")

    # Create dataset with mixed shapes
    dataset = env.dataset(batch_size=30, phase="train")  # ~10 of each shape
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.get_batch_sampler(batch_size=3, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    # Track what we see
    seen_shapes = set()
    batch_stats = []

    for batch_idx, batch in enumerate(dataloader):
        d = batch.batch_size[0]
        num_jobs = batch["start_op_per_job"].shape[-1]
        num_machines = batch["proc_times"].shape[-2]
        shape = (num_jobs, num_machines)

        seen_shapes.add(shape)
        batch_stats.append((shape, d))

        # Verify batch shape consistency
        assert batch["bucket_num_jobs"].eq(num_jobs).all()
        assert batch["bucket_num_machines"].eq(num_machines).all()
        assert d <= 3

        print(f"  Batch {batch_idx}: shape={shape}, size={d}")

        # Test all three loss modes work with this batch
        num_starts_po_bopo = 6

        # RL mode (num_starts=1)
        reward_rl = -torch.rand(d) * 1000
        out_rl = model_rl.calculate_loss(
            td={},
            batch=batch,
            policy_out={
                "reward": reward_rl,
                "log_likelihood": torch.randn(d),
                "batch_size": d,
                "num_starts": 1,
            },
        )
        assert torch.isfinite(out_rl["loss"])

        # PO mode
        base_reward = -torch.rand(d) * 1000
        reward_db = base_reward.unsqueeze(1) - torch.arange(num_starts_po_bopo, dtype=torch.float32).unsqueeze(0)
        ll_db = torch.randn(d, num_starts_po_bopo)
        out_po = model_po.calculate_loss(
            td={},
            batch=batch,
            policy_out={
                "reward": _flatten_db(reward_db),
                "log_likelihood": _flatten_db(ll_db),
                "batch_size": d,
                "num_starts": num_starts_po_bopo,
            },
        )
        assert torch.isfinite(out_po["loss"])

        # BOPO mode
        out_bopo = model_bopo.calculate_loss(
            td={},
            batch=batch,
            policy_out={
                "reward": _flatten_db(reward_db),
                "log_likelihood": _flatten_db(ll_db),
                "batch_size": d,
                "num_starts": num_starts_po_bopo,
            },
        )
        assert torch.isfinite(out_bopo["loss"])

    # Verify we saw all shapes
    assert seen_shapes == {(10, 10), (15, 15), (20, 20)}
    print(f"\n  Seen shapes: {seen_shapes}")
    print(f"  Total batches: {len(batch_stats)}")
    print("✅ Multi-shape bucketed training works!")


if __name__ == "__main__":
    # Run all tests
    test_rl_10x10_batch2()
    print("✅ test_rl_10x10_batch2")
    test_rl_15x15_batch2()
    print("✅ test_rl_15x15_batch2")
    test_rl_20x20_batch2()
    print("✅ test_rl_20x20_batch2")

    test_po_10x10_batch2()
    print("✅ test_po_10x10_batch2")
    test_po_15x15_batch2()
    print("✅ test_po_15x15_batch2")
    test_po_20x20_batch2()
    print("✅ test_po_20x20_batch2")

    test_bopo_10x10_batch2()
    print("✅ test_bopo_10x10_batch2")
    test_bopo_15x15_batch2()
    print("✅ test_bopo_15x15_batch2")
    test_bopo_20x20_batch2()
    print("✅ test_bopo_20x20_batch2")

    test_multi_shape_bucketed_training()
    print("\n" + "=" * 70)
    print("All Phase 6 smoke tests passed! 🎉")
    print("=" * 70)
