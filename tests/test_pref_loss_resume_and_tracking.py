from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_config_resume_latest_incomplete_prefers_newest_incomplete(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import yaml
    import ptp_discovery.run_pref_loss_coevo as launcher

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    older_incomplete = runs_dir / "20260416-111514"
    older_incomplete.mkdir()
    (older_incomplete / "checkpoint.json").write_text(
        json.dumps({"next_generation": 3}),
        encoding="utf-8",
    )
    newer_complete = runs_dir / "20260416-121514"
    newer_complete.mkdir()
    (newer_complete / "checkpoint.json").write_text(
        json.dumps({"next_generation": 10}),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(runs_dir),
                "generations": 10,
                "resume": {
                    "enabled": True,
                    "mode": "latest_incomplete",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    resume_dir, resume_mode = launcher._resolve_resume_dir_from_config(str(cfg_path))

    assert resume_mode == "latest_incomplete"
    assert resume_dir == str(older_incomplete.resolve())


def test_config_resume_latest_incomplete_allows_fresh_start_when_no_runs(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import yaml
    import ptp_discovery.run_pref_loss_coevo as launcher

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(runs_dir),
                "generations": 10,
                "resume": {
                    "enabled": True,
                    "mode": "latest_incomplete",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    resume_dir, resume_mode = launcher._resolve_resume_dir_from_config(str(cfg_path))

    assert resume_dir is None
    assert resume_mode is None


def test_persist_tracked_pair_values_writes_run_and_latest_files(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_repo_root() / "PTP"))

    import ptp_discovery.pref_loss_coevo_loop as loop

    output_root = tmp_path / "runs"
    run_dir = output_root / "20260416-111514"
    run_dir.mkdir(parents=True)
    cfg = {
        "output_root": str(output_root),
        "tracked_pair_values": {
            "enabled": True,
            "targets": [
                {
                    "name": "gen-1_pair-1",
                    "generation": -1,
                    "pair_index": -1,
                    "filename": "gen-1_pair-1_value.json",
                    "output_root_latest_filename": "latest_gen-1_pair-1_value.json",
                }
            ],
        },
    }
    rec = {
        "generation": -1,
        "pair_index": -1,
        "g_id": "g_ref",
        "f_id": "fseed_000_demo",
        "score": 0.03650093078612571,
        "stage": "high_fidelity",
        "stage_final": "transfer_seed_baseline",
        "phase": "builder",
        "pair_ok": True,
        "pair_reason": "ok_stage3_offline_minitrain",
    }

    saved = loop._persist_tracked_pair_values(
        cfg_yaml=cfg,
        run_dir=str(run_dir),
        records=[rec],
    )

    assert saved == 1
    run_value_path = run_dir / "gen-1_pair-1_value.json"
    latest_value_path = output_root / "latest_gen-1_pair-1_value.json"
    assert run_value_path.is_file()
    assert latest_value_path.is_file()

    run_payload = json.loads(run_value_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_value_path.read_text(encoding="utf-8"))
    assert run_payload["name"] == "gen-1_pair-1"
    assert run_payload["value"] == rec["score"]
    assert run_payload["score"] == rec["score"]
    assert run_payload["g_id"] == "g_ref"
    assert run_payload["f_id"] == "fseed_000_demo"
    assert latest_payload == run_payload
