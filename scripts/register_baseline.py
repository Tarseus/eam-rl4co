from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _atomic_write_json(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _resolve_effective_pomo_size(pomo_size: Any, problem_size: int) -> int:
    if pomo_size is None:
        return int(problem_size)
    try:
        value = int(pomo_size)
    except Exception:  # noqa: BLE001
        return int(problem_size)
    if value <= 0:
        return int(problem_size)
    return int(value)


def _baseline_key_from_hf_dict(hf_cfg: Dict[str, Any], *, early_eval_steps: int | None) -> str:
    train_problem_size = int(hf_cfg.get("train_problem_size", 0) or 0)
    payload: Dict[str, Any] = {
        "env_name": str(hf_cfg.get("env_name", "") or "").strip().lower(),
        "problem": str(hf_cfg.get("problem", "") or "").strip().lower(),
        "policy_name": str(hf_cfg.get("policy_name", "") or "").strip().lower(),
        "rollout_strategy": str(hf_cfg.get("rollout_strategy", "") or "").strip().lower(),
        "objective_sign": str(hf_cfg.get("objective_sign", "") or "").strip().lower(),
        "train_problem_size": train_problem_size,
        "valid_problem_sizes": [int(v) for v in (hf_cfg.get("valid_problem_sizes") or [])],
        "hf_steps": int(hf_cfg.get("hf_steps", 0) or 0),
        "hf_epochs": int(hf_cfg.get("hf_epochs", 0) or 0),
        "hf_instances_per_epoch": int(hf_cfg.get("hf_instances_per_epoch", 0) or 0),
        "train_batch_size": int(hf_cfg.get("train_batch_size", 0) or 0),
        "pomo_size": hf_cfg.get("pomo_size", None),
        "effective_pomo_size": _resolve_effective_pomo_size(hf_cfg.get("pomo_size"), train_problem_size),
        "learning_rate": float(hf_cfg.get("learning_rate", 0.0) or 0.0),
        "weight_decay": float(hf_cfg.get("weight_decay", 0.0) or 0.0),
        "alpha": float(hf_cfg.get("alpha", 0.0) or 0.0),
        "seed": int(hf_cfg.get("seed", 0) or 0),
        "num_validation_episodes": int(hf_cfg.get("num_validation_episodes", 0) or 0),
        "validation_batch_size": int(hf_cfg.get("validation_batch_size", 0) or 0),
        "early_eval_steps": int(early_eval_steps) if early_eval_steps is not None else None,
        "pool_version": str(hf_cfg.get("pool_version", "") or "").strip().lower(),
        "env_kwargs": dict(hf_cfg.get("env_kwargs", {}) or {}),
        "generator_params": dict(hf_cfg.get("generator_params", {}) or {}),
        "policy_kwargs": dict(hf_cfg.get("policy_kwargs", {}) or {}),
    }

    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(blob).hexdigest()[:16]

    env = payload["env_name"] or payload["problem"] or "unknown"
    policy = payload["policy_name"] or "auto"
    prefix = (
        f"{env}__{policy}__n{payload['train_problem_size']}__epochs{payload['hf_epochs']}__"
        f"inst{payload['hf_instances_per_epoch']}__pomo{payload['effective_pomo_size']}__"
        f"seed{payload['seed']}__{payload['objective_sign'] or 'neg_reward'}"
    )
    safe_prefix = re.sub(r"[^a-zA-Z0-9._-]+", "_", prefix).strip("_")
    return f"{safe_prefix}__{digest}"


def _paths_for_key(baseline_root: Path, key: str) -> Tuple[Path, Path]:
    base_dir = baseline_root / key
    return base_dir / "baseline.json", base_dir / "epoch_objectives.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a run_dir baseline.json into baseline/ cache.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing baseline.json")
    parser.add_argument(
        "--baseline-root",
        default=None,
        help="Baseline cache root (default: <repo_root>/baseline)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Optional explicit baseline key (overrides computed key)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    baseline_path = run_dir / "baseline.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline.json not found: {baseline_path}")

    baseline_root = (
        Path(args.baseline_root).resolve()
        if args.baseline_root is not None
        else Path(__file__).resolve().parents[1] / "baseline"
    )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise ValueError(f"Invalid baseline.json format: {baseline_path}")

    early_eval_steps = baseline.get("early_eval_steps")
    early_eval_steps_int = int(early_eval_steps) if early_eval_steps is not None else None

    key = args.key
    if not key:
        cfg = baseline.get("config") or {}
        hf_cfg = cfg.get("hf")
        if not isinstance(hf_cfg, dict):
            raise ValueError("baseline.json is missing config.hf; pass --key to override")
        key = _baseline_key_from_hf_dict(hf_cfg, early_eval_steps=early_eval_steps_int)

    out_baseline_path, out_epoch_path = _paths_for_key(baseline_root, key)
    out_baseline_path.parent.mkdir(parents=True, exist_ok=True)

    _atomic_write_json(str(out_baseline_path), baseline)

    epoch_objectives: List[float] | None = None
    epoch_eval = baseline.get("epoch_eval") or {}
    if isinstance(epoch_eval, dict):
        objs = epoch_eval.get("objectives")
        if isinstance(objs, list) and objs:
            epoch_objectives = [float(v) for v in objs]
    if epoch_objectives is not None:
        _atomic_write_json(str(out_epoch_path), epoch_objectives)

    print(f"Registered baseline key={key}")
    print(f"- {out_baseline_path}")
    if epoch_objectives is not None:
        print(f"- {out_epoch_path}")


if __name__ == "__main__":
    main()

