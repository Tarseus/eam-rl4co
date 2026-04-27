from __future__ import annotations

from pathlib import Path


def test_offline_split_generator_preserves_extra_attrs(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "PTP"))

    import fitness.free_loss_fidelity as fidelity

    class _DummyTD:
        batch_size = [1]

    monkeypatch.setattr(fidelity, "_load_offline_tensordict", lambda path: _DummyTD())

    gen = fidelity.OfflineSplitGenerator(
        train_path="offline_data/train.pt",
        val_path="offline_data/val.pt",
        device="cpu",
        extra_attrs={"vehicle_capacity": 1.0},
    )

    assert gen.vehicle_capacity == 1.0
