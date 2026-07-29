#!/usr/bin/env python3
"""Materialize immutable worker payloads for the fixed 5x5 TSP100 pair grid.

The generated files are inputs to ``PTP/ptp_discovery/run_hf_pair_eval.py`` on
the evaluation host.  This script deliberately never evaluates, repairs, or
searches programs: every payload embeds the selected historical builder and
loss IR verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "planned_joint_grid_v1.json"
EVAL_SIGNATURE = "0b305404eaee8d33"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _config() -> dict[str, Any]:
    """The historical TSP100 stage-3 protocol, with candidate mutation disabled."""
    return {
        "seed": 1234,
        "metric_mode": "minimize",
        "eval_stages": {"stage0_gate": False, "stage1_proxy": False, "stage2_micro_unroll": False, "stage3_high_fidelity": True},
        "cheap_gate_on": False,
        "high_fidelity_on": True,
        "joint_gate_repair_enabled": False,
        "builder_gate_repair_enabled": False,
        "stage0_sandbox_gate_enabled": False,
        "stage3_early_prune": {"enabled": False},
        "hf_epochs": 10,
        "hf_instances_per_epoch": 100000,
        "f1_steps": 200,
        "train_problem_size": 100,
        "valid_problem_sizes": [100],
        "train_batch_size": 64,
        "num_validation_episodes": 10000,
        "validation_batch_size": 64,
        "pomo_size": None,
        "loss_observables": ["seq_len", "log_prob_mean", "advantage", "entropy", "log_prob_step"],
        "scratch_init_seed": 1234,
        "backend": "rl4co",
        "env_name": "tsp",
        "generator_params": {"num_loc": 100},
        "policy_name": "pomo",
        "policy_kwargs": {
            "po4cops_compat": True,
            "embed_dim": 128,
            "num_encoder_layers": 6,
            "decoder_layer_num": 1,
            "qkv_dim": 16,
            "num_heads": 8,
            "feedforward_hidden": 512,
            "tanh_clipping": 50,
            "eval_type": "argmax",
            "val_decode_type": "greedy",
            "test_decode_type": "greedy",
        },
        "rollout_strategy": "auto",
        "objective_sign": "neg_reward",
        "learning_rate": 0.0003,
        "weight_decay": 0.000001,
        "alpha": 0.05,
        "po_impl": "bt",
        "builder_pair_budget": {"per_instance": 8192, "total": 8192},
        "baseline": {
            "metrics_csv": "baseline/metrics.csv",
            "checkpoint": "baseline/tsp100_epoch_135.ckpt",
            "checkpoint_epoch": 135,
            "scratch_start_epoch": 0,
            "val_column": "val/reward",
            "mini_eval_paths": {"epoch10_inst100000": "baseline/mini_eval/baseline_minitrain_tsp100_epoch10_inst100000.json"},
            "checkpoints": ["baseline/tsp100_epoch_135.ckpt"],
            "include_scratch": True,
            "multiseed_compare_enabled": False,
        },
    }


def materialize(plan_path: Path, out_dir: Path, *, profile: str) -> dict[str, Any]:
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    records = plan.get("records")
    if not isinstance(records, list) or len(records) != 25:
        raise ValueError("Expected exactly 25 pre-registered pair records")
    if len({str(row.get("pair_id")) for row in records}) != 25:
        raise ValueError("Pair IDs must be unique")

    out_dir.mkdir(parents=True, exist_ok=False)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir()
    cfg = _config()
    if profile == "behavior_short":
        cfg.update(
            {
                "hf_epochs": 1,
                "hf_instances_per_epoch": 100000,
                "f1_steps": 32,
                "num_validation_episodes": 512,
                "policy_kwargs": {"po4cops_compat": True},
                "baseline": {
                    "mini_eval_paths": {
                        "epoch1_inst100000": "baseline/mini_eval/baseline_minitrain_tsp100_epoch1_inst100000.json"
                    },
                    "checkpoints": [],
                    "include_scratch": True,
                    "multiseed_compare_enabled": False,
                },
            }
        )
    cfg_hash = _sha256_bytes(_stable_json(cfg))
    payload_rows: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        pair_id = str(row["pair_id"])
        payload = {
            "generation": 0,
            "pair_index": index,
            "g_id": str(row["builder"]["id"]),
            "f_id": str(row["loss"]["id"]),
            "g_entry": {"id": str(row["builder"]["id"]), "ir": dict(row["builder"]["ir"])},
            "f_entry": {"id": str(row["loss"]["id"]), "ir": dict(row["loss"]["ir"])},
            "cfg_yaml": cfg,
            "device_str": "cuda:0",
            "operator_whitelist": [],
            "cheap_gate_on": False,
            "high_fidelity_on": True,
            "eval_budget_signature": EVAL_SIGNATURE,
            "joint_pair_id": pair_id,
        }
        name = f"{index:02d}_{pair_id}.json"
        raw = _stable_json(payload)
        (payload_dir / name).write_bytes(raw)
        payload_rows.append({"pair_index": index, "pair_id": pair_id, "payload": f"payloads/{name}", "sha256": _sha256_bytes(raw)})

    manifest = {
        "dataset_id": str(plan.get("dataset_id")),
        "purpose": "Fixed 5x5 real builder/loss high-fidelity evaluation; payloads contain no search or repair.",
        "profile": profile,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_bytes(plan_raw),
        "eval_budget_signature": EVAL_SIGNATURE,
        "config_sha256": cfg_hash,
        "config": cfg,
        "pairs": payload_rows,
    }
    (out_dir / "manifest.json").write_bytes(_stable_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "behavior_short"), default="full")
    args = parser.parse_args()
    manifest = materialize(args.plan.resolve(), args.out.resolve(), profile=args.profile)
    print(json.dumps({"pairs": len(manifest["pairs"]), "plan_sha256": manifest["plan_sha256"], "config_sha256": manifest["config_sha256"]}))


if __name__ == "__main__":
    main()
