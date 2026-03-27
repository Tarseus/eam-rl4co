"""
Phase 5: Full pipeline tests for bucketed batching with RL/PO/BOPO
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


def test_bucketed_dataloader_integration():
    """Test that bucketed dataloader integrates correctly"""
    env = JSSPEnv(
        bucketed_training=True,
        shape_buckets=[[10, 10], [15, 15], [20, 20]],
    )
    dataset = env.dataset(batch_size=30, phase="train")

    assert hasattr(dataset, "get_batch_sampler")
    assert hasattr(dataset, "collate_fn")

    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.get_batch_sampler(batch_size=4, shuffle=False),
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    seen_shapes = set()
    for batch in dataloader:
        num_jobs = batch["start_op_per_job"].shape[-1]
        num_machines = batch["proc_times"].shape[-2]
        shape = (num_jobs, num_machines)
        seen_shapes.add(shape)
        assert batch["bucket_num_jobs"].eq(num_jobs).all()
        assert batch["bucket_num_machines"].eq(num_machines).all()

    assert seen_shapes == {(10, 10), (15, 15), (20, 20)}


@pytest.mark.parametrize("loss_mode", ["rl", "po", "bopo"])
def test_e2e_bucketed_loss(loss_mode: str):
    """Test loss calculation works with bucketed batches of different shapes"""
    env = JSSPEnv(
        bucketed_training=True,
        shape_buckets=[[10, 10], [15, 15], [20, 20]],
    )
    model = _make_reinforce(loss_mode=loss_mode)
    dataset = env.dataset(batch_size=12, phase="train")
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.get_batch_sampler(batch_size=2, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    for batch in dataloader:
        d = batch.batch_size[0]
        b = 8

        base_reward = -torch.rand(d) * 1000
        reward_db = base_reward.unsqueeze(1) - torch.arange(b, dtype=torch.float32).unsqueeze(0)
        ll_db = torch.randn(d, b)

        out = model.calculate_loss(
            td={},
            batch=batch,
            policy_out={
                "reward": _flatten_db(reward_db),
                "log_likelihood": _flatten_db(ll_db),
                "batch_size": d,
                "num_starts": b,
            },
        )

        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["reinforce_loss"])


def test_per_instance_pairing():
    """Verify PO/BOPO pairing is strictly per-instance"""
    model = _make_reinforce(loss_mode="po")

    d = 2
    b = 3
    reward_db = torch.tensor([
        [10.0, 9.0, 8.0],
        [3.0, 2.0, 1.0],
    ])
    ll_db = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ])

    po_loss = REINFORCE._po_instance_loss(reward_db, ll_db)

    tri_mask = torch.triu(torch.ones(b, b, dtype=torch.bool), diagonal=1)
    instance_losses = []
    for i in range(d):
        r = reward_db[i]
        ll = ll_db[i]
        r_diff = r.unsqueeze(1) - r.unsqueeze(0)
        ll_diff = ll.unsqueeze(1) - ll.unsqueeze(0)
        pref = torch.sign(r_diff[tri_mask])
        pair_loss = -torch.nn.functional.logsigmoid(pref * ll_diff[tri_mask])
        instance_losses.append(pair_loss.mean())
    expected_loss = torch.stack(instance_losses).mean()

    assert torch.allclose(po_loss, expected_loss)

    model.loss_mode = "bopo"
    bopo_loss = REINFORCE._bopo_instance_loss(reward_db, ll_db)

    arange_b = torch.arange(b)
    instance_losses = []
    for i in range(d):
        r = reward_db[i]
        ll = ll_db[i]
        anchor_idx = torch.argmax(r)
        mask = arange_b != anchor_idx
        other_r = r[mask]
        other_ll = ll[mask]
        pref = torch.sign(r[anchor_idx] - other_r)
        loss_i = -torch.nn.functional.logsigmoid(pref * (ll[anchor_idx] - other_ll))
        instance_losses.append(loss_i.mean())
    expected_bopo_loss = torch.stack(instance_losses).mean()

    assert torch.allclose(bopo_loss, expected_bopo_loss)
