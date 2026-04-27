from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts import eval_downloaded_routing_checkpoints as mod
from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_extract_metric_snapshot_tracks_last_test_values_and_max_epoch(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _write_csv(
        metrics_path,
        [
            {
                "epoch": "100",
                "test/reward": "",
                "test/max_reward": "",
                "test/max_aug_reward": "",
            },
            {
                "epoch": "220",
                "test/reward": "-15.4",
                "test/max_reward": "-15.0",
                "test/max_aug_reward": "",
            },
            {
                "epoch": "220",
                "test/reward": "-15.2",
                "test/max_reward": "-14.9",
                "test/max_aug_reward": "-14.8",
            },
        ],
    )

    snapshot = mod.extract_metric_snapshot(metrics_path)

    assert snapshot.train_max_epoch == 220
    assert snapshot.test_reward == -15.2
    assert snapshot.test_max_reward == -14.9
    assert snapshot.test_max_aug_reward == -14.8
    assert snapshot.has_max_aug_reward is True


def test_rewrite_repo_data_path_maps_cvrp_artifacts_to_local_vrp_data(tmp_path: Path) -> None:
    target = tmp_path / "data" / "vrp" / "vrp100_test_seed1234.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("placeholder", encoding="utf-8")

    rewritten = mod.rewrite_repo_data_path(
        "/data1/gushengda/eam-rl4co/data/cvrp/cvrp100_test_seed1234.npz",
        tmp_path,
    )

    assert rewritten == str(target.resolve())


def test_parse_problem_key_supports_ffsp() -> None:
    assert mod.parse_problem_key("ffsp100") == ("ffsp", 100)


def test_patch_legacy_po4cops_tsp_policy_restores_missing_start_node() -> None:
    policy = PO4COPsTSPPolicy()
    del policy.start_node
    assert not hasattr(policy, "start_node")

    patched = mod._patch_legacy_policy_object(policy)

    assert patched is policy
    assert patched.start_node == "pomo"


def test_main_skips_existing_max_aug_and_evaluates_missing_entry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    manifest_path = repo_root / "downloads" / "manifest.json"
    output_csv = repo_root / "downloads" / "routing_results.csv"

    entries = [
        {
            "problem": "tsp100",
            "method": "weighting",
            "checkpoint": "downloads/tsp100/weighting/checkpoint.ckpt",
            "source_checkpoint": "downloads/final_checkpoints/tsp100/weighting/last.ckpt",
            "moved_metadata": [],
        },
        {
            "problem": "tsp50",
            "method": "bopo",
            "checkpoint": "downloads/tsp50/bopo/checkpoint.ckpt",
            "source_checkpoint": "downloads/checkpoints_non_g53/tsp50_g51/last.ckpt",
            "moved_metadata": [],
        },
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    for rel_path in [
        "downloads/tsp100/weighting/checkpoint.ckpt",
        "downloads/tsp50/bopo/checkpoint.ckpt",
    ]:
        path = repo_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")

    _write_csv(
        repo_root / "curves" / "tsp100_best_weighting.csv",
        [
            {
                "epoch": "258",
                "test/reward": "-8.1",
                "test/max_reward": "-7.9",
                "test/max_aug_reward": "-7.8",
            }
        ],
    )
    _write_csv(
        repo_root / "curves" / "tsp50_bopo.csv",
        [
            {
                "epoch": "653",
                "test/reward": "",
                "test/max_reward": "",
                "test/max_aug_reward": "",
            }
        ],
    )

    def fake_run_single_evaluation(**kwargs):
        entry = kwargs["entry"]
        assert entry.problem_key == "tsp50"
        assert entry.method == "bopo"
        return {
            "seed": 1234,
            "num_starts": 50,
            "num_augment": 8,
            "test_data_size": 16,
            "test_batch_size": 8,
            "resolved_test_file": str(repo_root / "data" / "tsp" / "tsp50_test_seed1234.npz"),
            "test_reward": -5.9,
            "test_max_reward": -5.7,
            "test_max_aug_reward": -5.6,
            "elapsed_sec": 1.234,
        }

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "run_single_evaluation", fake_run_single_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_downloaded_routing_checkpoints.py",
            "--manifest",
            str(manifest_path),
            "--output-csv",
            str(output_csv),
            "--problems",
            "tsp100,tsp50",
            "--methods",
            "weighting,bopo",
            "--num-instances",
            "16",
            "--test-batch-size",
            "8",
        ],
    )

    exit_code = mod.main()

    assert exit_code == 0
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2

    skip_row = next(row for row in rows if row["problem_key"] == "tsp100")
    assert skip_row["status"] == "skipped_existing_metrics"
    assert skip_row["metrics_path"].endswith("curves\\tsp100_best_weighting.csv") or skip_row["metrics_path"].endswith(
        "curves/tsp100_best_weighting.csv"
    )
    assert skip_row["test_max_aug_reward"] == "-7.8"
    assert skip_row["train_max_epoch"] == "258"

    eval_row = next(row for row in rows if row["problem_key"] == "tsp50")
    assert eval_row["status"] == "evaluated"
    assert eval_row["metric_source"] == "fresh_eval"
    assert eval_row["test_max_reward"] == "-5.7"
    assert eval_row["test_max_aug_reward"] == "-5.6"
    assert eval_row["train_max_epoch"] == "653"


def test_main_records_ffsp_without_aug_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    manifest_path = repo_root / "downloads" / "manifest.json"
    output_csv = repo_root / "downloads" / "routing_results.csv"

    entries = [
        {
            "problem": "ffsp50",
            "method": "po",
            "checkpoint": "downloads/ffsp50/po/checkpoint.ckpt",
            "source_checkpoint": "downloads/final_checkpoints/ffsp50/po/last.ckpt",
            "moved_metadata": [],
        }
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    ckpt_path = repo_root / "downloads" / "ffsp50" / "po" / "checkpoint.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("placeholder", encoding="utf-8")

    def fake_run_single_evaluation(**kwargs):
        entry = kwargs["entry"]
        assert entry.problem_key == "ffsp50"
        return {
            "seed": 1234,
            "num_starts": 24,
            "num_augment": 0,
            "supports_max_aug_reward": False,
            "max_aug_reward_note": "FFSP checkpoints currently use MatNet with num_augment=0.",
            "test_data_size": 16,
            "test_batch_size": 8,
            "resolved_test_file": None,
            "test_reward": -57.0,
            "test_max_reward": -53.4,
            "test_max_aug_reward": None,
            "elapsed_sec": 1.111,
        }

    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "run_single_evaluation", fake_run_single_evaluation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_downloaded_routing_checkpoints.py",
            "--manifest",
            str(manifest_path),
            "--output-csv",
            str(output_csv),
            "--problems",
            "ffsp50",
            "--methods",
            "po",
        ],
    )

    exit_code = mod.main()

    assert exit_code == 0
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["problem_key"] == "ffsp50"
    assert row["status"] == "evaluated_without_aug_metric"
    assert row["supports_max_aug_reward"] == "False"
    assert row["test_max_reward"] == "-53.4"
    assert row["test_max_aug_reward"] == ""
    assert "num_augment=0" in row["max_aug_reward_note"]
