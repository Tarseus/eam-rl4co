import json

import pytest
import torch

import rl4co.models.zoo.mgl_jssp.model as mgl_model_module
from rl4co.models.zoo.mgl_jssp.model import MGLJSSPModel


class _DummyEnv:
    pass


def _write_pref_pair_artifacts(tmp_path):
    builder_payload = {
        "id": "g_unit",
        "ir": {
            "name": "all_pairs",
            "intuition": "Use all strictly ordered pairs inside each instance.",
            "implementation_hint": {
                "expects": ["objective"],
                "returns": "PrefBatch",
                "mode": "pairwise",
            },
            "hyperparams": {},
            "operators_used": ["all_pairs"],
            "code": (
                "def generated_builder(feature_cache, extra):\n"
                "    objective = feature_cache['objective']\n"
                "    mask = objective[:, :, None] < objective[:, None, :]\n"
                "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)\n"
                "    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={'builder': 'all_pairs'})\n"
            ),
        },
    }
    loss_payload = {
        "id": "f_unit",
        "ir": {
            "name": "bt_loss",
            "intuition": "Stable Bradley-Terry style loss over pairwise preferences.",
            "pseudocode": "loss = -logsigmoid(alpha * (log_prob_w - log_prob_l))",
            "hyperparams": {},
            "operators_used": ["logsigmoid"],
            "implementation_hint": {
                "expects": ["log_prob_w", "log_prob_l", "weight"],
                "returns": "scalar",
                "mode": "pairwise",
            },
            "code": (
                "def generated_loss(batch, model_output, extra):\n"
                "    alpha = float(extra.get('alpha', 1.0)) if isinstance(extra, dict) else 1.0\n"
                "    margin = alpha * (batch['log_prob_w'] - batch['log_prob_l'])\n"
                "    loss = -ops.logsigmoid(margin)\n"
                "    weight = batch.get('weight', None)\n"
                "    if weight is not None:\n"
                "        loss = loss * weight\n"
                "    return loss.mean()\n"
            ),
            "theoretical_basis": "Pairwise logistic preference loss.",
        },
    }
    pair_payload = {"g_id": "g_unit", "f_id": "f_unit"}

    builder_path = tmp_path / "best_builder.json"
    loss_path = tmp_path / "best_loss.json"
    pair_path = tmp_path / "best_pair.json"
    builder_path.write_text(json.dumps(builder_payload), encoding="utf-8")
    loss_path.write_text(json.dumps(loss_payload), encoding="utf-8")
    pair_path.write_text(json.dumps(pair_payload), encoding="utf-8")
    return pair_path


def test_mgl_jssp_pref_pair_artifacts_load_for_bopo(tmp_path) -> None:
    pair_path = _write_pref_pair_artifacts(tmp_path)

    model = MGLJSSPModel(
        env=_DummyEnv(),
        baseline="bopo",
        pref_pair_json_path=str(pair_path),
        B=4,
        K=2,
    )

    assert model._free_loss_enabled is True
    assert model.pref_builder is not None
    assert model.free_loss is not None


@pytest.mark.parametrize("baseline", ["rl", "po"])
def test_mgl_jssp_pref_pair_loads_for_other_mgl_baselines(tmp_path, baseline) -> None:
    pair_path = _write_pref_pair_artifacts(tmp_path)

    model = MGLJSSPModel(
        env=_DummyEnv(),
        baseline=baseline,
        pref_pair_json_path=str(pair_path),
        B=4,
        K=2,
    )

    assert model._free_loss_enabled is True
    assert model.pref_builder is not None
    assert model.free_loss is not None


def test_mgl_jssp_pref_pair_rollout_produces_finite_loss(tmp_path, monkeypatch) -> None:
    pair_path = _write_pref_pair_artifacts(tmp_path)

    model = MGLJSSPModel(
        env=_DummyEnv(),
        baseline="bopo",
        pref_pair_json_path=str(pair_path),
        B=4,
        K=2,
        free_loss_observables=["advantage"],
    )

    def _fake_solve_jsp(
        instances,
        batch_size_per_instance,
        device,
        encoder,
        decoder,
        use_greedy=False,
    ):
        num_instances = len(instances)
        num_steps = instances[0]["j"] * instances[0]["m"] - 1
        num_jobs = instances[0]["j"]
        total = num_instances * batch_size_per_instance

        trajs = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0]],
            dtype=torch.long,
        )[:total]
        logits = torch.zeros((total, num_steps, num_jobs), dtype=torch.float32)
        logits[..., 0] = 1.0
        logits[..., 1] = -0.5
        makespans = torch.tensor([10.0, 12.0, 15.0, 18.0], dtype=torch.float32)[:total]
        entropies = torch.full((total, num_steps), 0.5, dtype=torch.float32)
        return trajs, logits, makespans, entropies

    monkeypatch.setattr(mgl_model_module, "solve_jsp", _fake_solve_jsp)

    loss, reward, aux_metric, pair_count = model._training_rollout({"j": 2, "m": 2})

    assert torch.isfinite(loss)
    assert torch.isfinite(reward)
    assert torch.isfinite(aux_metric)
    assert pair_count.item() == 6.0


def test_mgl_jssp_required_allowed_shapes_rejects_missing_allowed_shapes() -> None:
    with pytest.raises(ValueError, match="required_allowed_shapes"):
        MGLJSSPModel(
            env=_DummyEnv(),
            baseline="bopo",
            B=4,
            K=2,
            required_allowed_shapes=[[10, 10]],
        )


def test_mgl_jssp_setup_enforces_expected_dataset_size(monkeypatch) -> None:
    def _fake_load_dataset(data_dir: str, use_cached: bool = True, device: str = "cpu"):
        if "train" in data_dir:
            return [
                {"j": 10, "m": 10, "name": "train_10x10"},
                {"j": 15, "m": 15, "name": "train_15x15"},
            ]
        return [{"j": 10, "m": 10, "name": "val_10x10"}]

    monkeypatch.setattr(mgl_model_module, "load_dataset", _fake_load_dataset)

    model = MGLJSSPModel(
        env=_DummyEnv(),
        baseline="bopo",
        B=4,
        K=2,
        train_data_dir="train",
        val_data_dir="validation",
        allowed_shapes=[[10, 10]],
        required_allowed_shapes=[[10, 10]],
        expected_train_dataset_size=2,
        expected_val_dataset_size=1,
    )

    with pytest.raises(ValueError, match="train split size mismatch"):
        model.setup(stage="fit")
