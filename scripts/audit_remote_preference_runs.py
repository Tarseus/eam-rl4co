from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


METRICS = (
    "val/reward",
    "val/max_reward",
    "val/max_aug_reward",
    "test/reward",
    "test/max_reward",
    "test/max_aug_reward",
)


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first(payload: dict[str, Any], key: str) -> Any:
    return payload.get(key, _nested(payload, "model", key))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _method(pref_path: str) -> str:
    lowered = pref_path.lower()
    if "pref_builder_weight" in lowered:
        return "ASW"
    if "pref_loss" in lowered:
        return "USW"
    return "unknown"


def summarize_metrics(path: Path) -> dict[str, Any]:
    best: dict[str, dict[str, Any]] = {}
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return {"metrics_path": None, "best": best, "latest": latest}

    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = _float(row.get("epoch"))
            step = _float(row.get("step"))
            for metric in METRICS:
                value = _float(row.get(metric))
                if value is None:
                    continue
                record = {
                    "value": value,
                    "epoch": int(epoch) if epoch is not None else None,
                    "step": int(step) if step is not None else None,
                }
                latest[metric] = record
                if metric not in best or value > float(best[metric]["value"]):
                    best[metric] = record
    return {"metrics_path": str(path), "best": best, "latest": latest}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory preference-objective training runs and their validation/test metrics."
    )
    parser.add_argument("--root", type=Path, default=Path("logs/train/runs"))
    parser.add_argument("--problem", default="cvrp100")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    problem = args.problem.lower()
    routing_match = re.fullmatch(r"(?:tsp|cvrp)(\d+)", problem)
    expected_routing_starts = (
        int(routing_match.group(1)) if routing_match is not None else None
    )
    for hparams_path in sorted(args.root.rglob("hparams.yaml")):
        try:
            payload = yaml.safe_load(hparams_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        pref_path = str(_first(payload, "pref_pair_json_path") or "")
        num_starts = _first(payload, "num_starts")
        old_env = str(_first(payload, "env") or "")
        searchable = " ".join(
            (
                str(hparams_path).lower(),
                pref_path.lower(),
                old_env.lower(),
            )
        )
        if problem not in searchable and not (
            problem == "cvrp100"
            and "cvrp100" in pref_path.lower()
            and int(num_starts or 0) == 100
        ):
            continue
        if not pref_path or _method(pref_path) == "unknown":
            continue
        if (
            expected_routing_starts is not None
            and int(num_starts or 0) != expected_routing_starts
        ):
            continue

        version_dir = hparams_path.parent
        run_dir = version_dir
        while run_dir != args.root and run_dir.name not in {"checkpoints", ".hydra"}:
            if (run_dir / "checkpoints").is_dir() or (run_dir / ".hydra").is_dir():
                break
            run_dir = run_dir.parent
        metrics_path = version_dir / "metrics.csv"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints = (
            [
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(checkpoints_dir.glob("*.ckpt"))
            ]
            if checkpoints_dir.is_dir()
            else []
        )
        record = {
            "method": _method(pref_path),
            "run_dir": str(run_dir),
            "hparams_path": str(hparams_path),
            "pref_pair_json_path": pref_path,
            "num_starts": int(num_starts) if num_starts is not None else None,
            "max_epochs": _nested(payload, "trainer", "max_epochs"),
            "ckpt_path": payload.get("ckpt_path"),
            "checkpoint_monitor": _nested(
                payload, "callbacks", "model_checkpoint", "monitor"
            ),
            "checkpoints": checkpoints,
        }
        record.update(summarize_metrics(metrics_path))
        records.append(record)

    result = {"problem": args.problem, "root": str(args.root), "runs": records}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
