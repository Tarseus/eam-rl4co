from __future__ import annotations

from pathlib import Path

import yaml


def test_ffsp100_weight_search_config_matches_ffsp_builder_search_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "PTP"
        / "configs"
        / "experiment"
        / "pref_loss_coevo"
        / "ffsp100_builder_weight_search_from_archive.yaml"
    )

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    assert cfg["search_mode"] == "builder_only"
    assert cfg["output_root"] == "runs/pref_builder_weight_search_ffsp100"
    assert cfg["resume"]["enabled"] is True
    assert cfg["resume"]["mode"] == "latest_incomplete"
    assert cfg["env_name"] == "ffsp"
    assert cfg["policy_name"] == "matnet"
    assert cfg["precision"] == "32-true"
    assert cfg["loss_observables"] == ["advantage"]
    assert cfg["aggressive_cuda_cleanup"] is True
    assert cfg["aggressive_cuda_cleanup_mode"] == "epoch"
    tracked = cfg["tracked_pair_values"]
    assert tracked["enabled"] is True
    assert tracked["targets"] == [
        {
            "name": "gen-1_pair-1",
            "generation": -1,
            "pair_index": -1,
            "filename": "gen-1_pair-1_value.json",
            "output_root_latest_filename": "latest_gen-1_pair-1_value.json",
        }
    ]

    source_loss_path = repo_root / cfg["loss_transfer_seed"]["source_loss_path"]
    assert source_loss_path.is_file()

    builder_search_space = cfg["builder_search_space"]
    assert builder_search_space["enabled"] is True
    assert builder_search_space["mode"] == "reweight_only"
    assert builder_search_space["fixed_pair_builder"] == "all_pairs"
    assert builder_search_space["require_instance_stats"] is True
    assert builder_search_space["require_instance_conditioning"] is True
    assert builder_search_space["allow_freeform_weight_family"] is False
    assert builder_search_space["allow_uniform_none"] is False

    assert cfg["builder_llm"]["enabled"] is True
    assert cfg["loss_llm"]["enabled"] is False
