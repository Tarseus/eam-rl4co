import torch

from rl4co.models.rl.reinforce.reinforce import REINFORCE


class _DummyBaseline:
    def eval(self, td, reward, env):
        return 0, 0


def _make_model(loss_mode: str):
    model = REINFORCE.__new__(REINFORCE)
    model.baseline = _DummyBaseline()
    model.advantage_scaler = lambda x: x
    model.loss_mode = loss_mode
    model.env = None
    return model


def _flatten_db(x_db: torch.Tensor) -> torch.Tensor:
    # Match rl4co.utils.ops.batchify/unbatchify ordering for [D, B] <-> [D*B].
    # For D=2, B=3 this yields [x00, x10, x01, x11, x02, x12].
    d, b = x_db.shape
    return torch.stack([x_db[:, j] for j in range(b)], dim=0).reshape(d * b)


def _po_expected(reward_db: torch.Tensor, ll_db: torch.Tensor) -> torch.Tensor:
    d, b = reward_db.shape
    tri = torch.triu(torch.ones(b, b, dtype=torch.bool), diagonal=1)
    li = []
    for i in range(d):
        r = reward_db[i]
        ll = ll_db[i]
        pref = torch.sign((r.unsqueeze(1) - r.unsqueeze(0))[tri])
        ll_diff = (ll.unsqueeze(1) - ll.unsqueeze(0))[tri]
        valid = pref != 0
        if valid.any():
            li.append((-torch.nn.functional.logsigmoid(pref[valid] * ll_diff[valid])).mean())
        else:
            li.append(torch.zeros((), dtype=reward_db.dtype))
    return torch.stack(li).mean()


def _bopo_expected(reward_db: torch.Tensor, ll_db: torch.Tensor) -> torch.Tensor:
    d, b = reward_db.shape
    idx = torch.arange(b)
    li = []
    for i in range(d):
        r = reward_db[i]
        ll = ll_db[i]
        anchor = torch.argmax(r)
        mask = idx != anchor
        pref = torch.sign(r[anchor] - r[mask])
        valid = pref != 0
        if valid.any():
            li.append(
                (-torch.nn.functional.logsigmoid(pref[valid] * (ll[anchor] - ll[mask][valid]))).mean()
            )
        else:
            li.append(torch.zeros((), dtype=reward_db.dtype))
    return torch.stack(li).mean()


def test_reinforce_rl_mode_grouped_by_instance():
    model = _make_model("rl")
    reward_db = torch.tensor([[3.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    ll_db = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
    reward = _flatten_db(reward_db)
    ll = _flatten_db(ll_db)

    out = model.calculate_loss(
        td={},
        batch={},
        policy_out={"reward": reward, "log_likelihood": ll, "batch_size": 2, "num_starts": 3},
    )

    expected = -(reward_db * ll_db).mean(dim=1).mean()
    assert torch.allclose(out["reinforce_loss"], expected)


def test_reinforce_po_mode_grouped_by_instance():
    model = _make_model("po")
    reward_db = torch.tensor([[3.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    ll_db = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
    reward = _flatten_db(reward_db)
    ll = _flatten_db(ll_db)

    out = model.calculate_loss(
        td={},
        batch={},
        policy_out={"reward": reward, "log_likelihood": ll, "batch_size": 2, "num_starts": 3},
    )

    expected = _po_expected(reward_db, ll_db)
    assert torch.allclose(out["reinforce_loss"], expected)


def test_reinforce_bopo_mode_grouped_by_instance():
    model = _make_model("bopo")
    reward_db = torch.tensor([[3.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    ll_db = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
    reward = _flatten_db(reward_db)
    ll = _flatten_db(ll_db)

    out = model.calculate_loss(
        td={},
        batch={},
        policy_out={"reward": reward, "log_likelihood": ll, "batch_size": 2, "num_starts": 3},
    )

    expected = _bopo_expected(reward_db, ll_db)
    assert torch.allclose(out["reinforce_loss"], expected)
