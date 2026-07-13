from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PTP_ROOT = REPO_ROOT / "PTP"
for path in (str(REPO_ROOT), str(PTP_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fitness.pref_loss_fidelity import eval_budget_signature  # noqa: E402
from ptp_discovery.pref_loss_coevo_loop import (  # noqa: E402
    F_REF_ID,
    G_REF_ID,
    _build_hf_cfg,
    _evaluate_pair_worker,
    _ref_builder_ir,
    _ref_loss_ir,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return dict(data)


def _artifact_ir(payload: Mapping[str, Any], *, id_key: str, ir_key: str, ref_id: str, ref_ir: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get(ir_key), Mapping):
        return dict(payload.get(ir_key) or {})
    if str(payload.get(id_key) or "") == str(ref_id):
        return dict(ref_ir)
    nested = payload.get("ir")
    if isinstance(nested, Mapping):
        return dict(nested)
    raise ValueError(f"Cannot resolve {ir_key} from best-pair artifact for id={payload.get(id_key)!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluate a preference-search best pair under a common HF budget for cross-run comparison.",
    )
    p.add_argument("--run-dir", required=True, type=Path, help="Search run directory containing best_pair.json.")
    p.add_argument("--config", required=True, type=Path, help="Base YAML config to use for evaluation.")
    p.add_argument("--best-pair", type=Path, default=None, help="Optional explicit best_pair.json path.")
    p.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hf-epochs", type=int, default=20)
    p.add_argument("--scratch-hf-epochs", type=int, default=10)
    p.add_argument("--warmstart-hf-epochs", type=int, default=10)
    p.add_argument("--hf-instances-per-epoch", type=int, default=None)
    p.add_argument("--generation", type=int, default=-20)
    p.add_argument("--pair-index", type=int, default=-20)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    best_pair_path = (args.best_pair or (run_dir / "best_pair.json")).resolve()
    out_path = (args.out or (run_dir / "posthoc_eval" / f"best_pair_hf{int(args.hf_epochs)}.json")).resolve()

    cfg = _load_yaml(args.config.resolve())
    cfg["hf_epochs"] = int(args.hf_epochs)
    cfg["scratch_hf_epochs"] = int(args.scratch_hf_epochs)
    cfg["warmstart_hf_epochs"] = int(args.warmstart_hf_epochs)
    if args.hf_instances_per_epoch is not None:
        cfg["hf_instances_per_epoch"] = int(args.hf_instances_per_epoch)
    budgets = dict(cfg.get("budgets", {}) or {})
    budgets["hf_epochs"] = int(args.hf_epochs)
    cfg["budgets"] = budgets

    best_pair = _load_json(best_pair_path)
    if not isinstance(best_pair, dict):
        raise ValueError(f"Invalid best pair artifact: {best_pair_path}")

    gid = str(best_pair.get("g_id") or best_pair.get("builder_id") or G_REF_ID)
    fid = str(best_pair.get("f_id") or best_pair.get("loss_id") or F_REF_ID)
    g_ir = _artifact_ir(best_pair, id_key="g_id", ir_key="g_ir", ref_id=G_REF_ID, ref_ir=asdict(_ref_builder_ir()))
    f_ir = _artifact_ir(best_pair, id_key="f_id", ir_key="f_ir", ref_id=F_REF_ID, ref_ir=asdict(_ref_loss_ir()))

    proxy_problem_size = int(cfg.get("proxy_problem_size", cfg.get("train_problem_size", 20)) or 20)
    proxy_batch_size = int(cfg.get("proxy_batch_size", cfg.get("train_batch_size", 64)) or 64)
    proxy_batches = int(cfg.get("proxy_batches", 1) or 1)
    proxy_weights = dict(cfg.get("proxy_weights", {}) or {})
    micro_budget = {
        "micro_unroll_steps": int(cfg.get("micro_unroll_steps", 0) or 0),
        "micro_unroll_max_pairs": int(cfg.get("micro_unroll_max_pairs", 0) or 0),
    }
    hf_cfg = _build_hf_cfg(cfg, seed=int(cfg.get("seed", 1234) or 1234), device_str=str(args.device))
    eval_sig = eval_budget_signature(
        cfg=hf_cfg,
        proxy_problem_size=proxy_problem_size,
        proxy_batch_size=proxy_batch_size,
        proxy_batches=proxy_batches,
        proxy_weights={str(k): float(v) for k, v in proxy_weights.items()},
        extra_budget=micro_budget,
    )

    eval_run_dir = out_path.parent
    payload = {
        "generation": int(args.generation),
        "pair_index": int(args.pair_index),
        "g_entry": {"id": gid, "ir": dict(g_ir)},
        "f_entry": {"id": fid, "ir": dict(f_ir)},
        "cfg_yaml": dict(cfg),
        "device_str": str(args.device),
        "operator_whitelist": list(cfg.get("operator_whitelist", [])),
        "run_dir": str(eval_run_dir),
        "cheap_gate_on": False,
        "high_fidelity_on": True,
        "eval_budget_signature": str(eval_sig),
        "proxy_record": dict(best_pair),
    }

    t0 = time.time()
    record = _evaluate_pair_worker(payload)
    record["posthoc_compare_protocol"] = {
        "source_run_dir": str(run_dir),
        "source_best_pair": str(best_pair_path),
        "source_search_score": best_pair.get("score", best_pair.get("final_score")),
        "source_search_eval_budget_signature": best_pair.get("eval_budget_signature"),
        "hf_epochs": int(args.hf_epochs),
        "scratch_hf_epochs": int(args.scratch_hf_epochs),
        "warmstart_hf_epochs": int(args.warmstart_hf_epochs),
        "hf_instances_per_epoch": int(cfg.get("hf_instances_per_epoch", 0) or 0),
        "eval_budget_signature": str(eval_sig),
        "elapsed_s": float(time.time() - t0),
        "note": "Use this score for cross-run comparison; do not compare the source search score across different HF budgets.",
    }
    _write_json(out_path, record)
    print(json.dumps({"out": str(out_path), "score": record.get("score"), "pair_ok": record.get("pair_ok")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
