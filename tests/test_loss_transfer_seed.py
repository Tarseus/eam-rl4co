from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_loss_transfer_seed_from_single_loss_artifact(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    loss_path = tmp_path / "best_loss.json"
    loss_path.write_text(
        json.dumps(
            {
                "id": "f_best",
                "score": -0.021,
                "score_history_summary": {"latest": -0.021},
                "ir": {
                    "name": "seed_loss",
                    "intuition": "seed",
                    "pseudocode": "loss = -logsigmoid(lpw-lpl)",
                    "hyperparams": {"scale": 1.0},
                    "operators_used": ["logsigmoid"],
                    "implementation_hint": {
                        "expects": ["log_prob_w", "log_prob_l", "weight"],
                        "returns": "scalar",
                        "mode": "pairwise",
                    },
                    "code": (
                        "def generated_loss(batch, model_output, extra):\n"
                        "    x = batch['log_prob_w'] - batch['log_prob_l']\n"
                        "    return (-ops.logsigmoid(x)).mean()\n"
                    ),
                    "theoretical_basis": "",
                },
            }
        ),
        encoding="utf-8",
    )

    imported = loop._load_loss_transfer_seed_entries_from_loss_path(
        str(loss_path),
        keep_source_fitness=True,
        reset_history=False,
    )

    assert len(imported) == 1
    seed = imported[0]
    assert seed["origin"] == "TRANSFER_SEED"
    assert seed["op_type"] == "TRANSFER_SEED"
    assert seed["source_loss_id"] == "f_best"
    assert seed["fitness"] == -0.021
    assert seed["source_fitness"] == -0.021
    assert seed["source_loss_path"] == str(loss_path.resolve())
    assert seed["history"][-1]["source_loss_path"] == str(loss_path.resolve())


def test_normalize_loss_transfer_seed_cfg_accepts_source_loss_path(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    cfg = loop._normalize_loss_transfer_seed_cfg(
        {
            "loss_transfer_seed": {
                "enabled": True,
                "source_loss_path": "artifacts/loss_archive/tsp100_best/best_loss.json",
            }
        }
    )

    assert cfg["enabled"] is True
    assert cfg["source_loss_path"] == "artifacts/loss_archive/tsp100_best/best_loss.json"
