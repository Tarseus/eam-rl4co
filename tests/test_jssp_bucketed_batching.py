import pytest
import torch

from torch.utils.data import DataLoader

from rl4co.envs import JSSPEnv
from rl4co.models.rl.reinforce.reinforce import REINFORCE


def _make_bucketed_env():
    return JSSPEnv(
        bucketed_training=True,
        shape_buckets=[[10, 10], [15, 15], [20, 20]],
        generator_params={
            "min_processing_time": 1,
            "max_processing_time": 20,
        },
    )


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


def test_bucketed_sampler_keeps_single_shape_per_batch():
    env = _make_bucketed_env()
    dataset = env.dataset(batch_size=6, phase="train")
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.get_batch_sampler(batch_size=2, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    seen_shapes = set()
    for batch in dataloader:
        num_jobs = batch["start_op_per_job"].shape[-1]
        num_machines = batch["proc_times"].shape[-2]
        seen_shapes.add((num_jobs, num_machines))
        assert batch["bucket_num_jobs"].eq(num_jobs).all()
        assert batch["bucket_num_machines"].eq(num_machines).all()
        assert batch.batch_size[0] <= 2

    assert seen_shapes == {(10, 10), (15, 15), (20, 20)}


@pytest.mark.parametrize("loss_mode", ["rl", "po", "bopo"])
def test_bucketed_training_loss_path_one_batch_per_shape(loss_mode: str):
    env = _make_bucketed_env()
    model = _make_reinforce(loss_mode=loss_mode)
    dataset = env.dataset(batch_size=6, phase="train")
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.get_batch_sampler(batch_size=2, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    for batch in dataloader:
        d = batch.batch_size[0]
        b = 2
        base_reward = -batch["proc_times"].sum(dim=(1, 2))
        reward_db = torch.stack((base_reward, base_reward - 1), dim=1)
        ll_db = torch.stack(
            (
                torch.zeros_like(base_reward),
                torch.ones_like(base_reward) * 0.5,
            ),
            dim=1,
        )
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
