from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "PTP"))

from fitness.free_loss_fidelity import PrefBatch


def test_pairwise_loss_batch_detaches_builder_weight() -> None:
    log_prob = torch.tensor([[0.1, 0.4, 0.7]], requires_grad=True)
    weight = (log_prob[0, 2] - log_prob[0, 0]).reshape(1)
    pref = PrefBatch(
        mode="pairwise",
        pair_idx=(torch.tensor([0]), torch.tensor([0]), torch.tensor([2])),
        weight=weight,
    )

    batch = pref.to_pairwise_loss_batch(
        {"objective": torch.tensor([[1.0, 2.0, 3.0]]), "log_prob": log_prob}
    )

    assert torch.equal(batch["weight"], weight.detach())
    assert not batch["weight"].requires_grad