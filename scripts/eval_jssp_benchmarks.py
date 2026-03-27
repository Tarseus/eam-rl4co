from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from rl4co.models.zoo.mgl_jssp.data import load_instance
from rl4co.models.zoo.mgl_jssp.sampling import sampling
from scripts.prepare_bopo_jsp_data import prepare_bopo_jsp


def _parse_ood_dir(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("OOD dirs must be passed as name=path.")
    name, path = raw.split("=", 1)
    name = name.strip()
    path_obj = Path(path).expanduser()
    if not name:
        raise argparse.ArgumentTypeError("OOD dir name must be non-empty.")
    return name, path_obj


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rl4co-native JSSP MGL checkpoints on TA/LA/DMU and optional OOD directories."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the rl4co Lightning checkpoint (.ckpt).",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--B", type=int, default=128, help="Number of sampled candidates per instance.")
    parser.add_argument(
        "--greedy",
        type=int,
        default=None,
        help="Override whether to inject one greedy rollout (0/1). Defaults to the checkpoint setting.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["TA", "LA", "DMU"],
        help="Built-in BOPO/JSP benchmark sets to evaluate.",
    )
    parser.add_argument(
        "--ood-dir",
        action="append",
        default=[],
        type=_parse_ood_dir,
        help="Additional OOD set as name=path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save CSV/JSON outputs. Defaults to logs/eval/<checkpoint-stem>_<timestamp>.",
    )
    return parser.parse_args()


def _default_set_dirs(root_dir: Path) -> dict[str, Path]:
    bench_root = root_dir / "data" / "jssp_bopo"
    return {
        "validation": bench_root / "validation",
        "TA": bench_root / "TA",
        "LA": bench_root / "LA",
        "DMU": bench_root / "DMU",
    }


def _load_model(model_path: Path, device: torch.device):
    from rl4co.envs import JSSPEnv
    from rl4co.models.zoo.mgl_jssp.model import MGLJSSPModel

    env = JSSPEnv(generator_params={"num_jobs": 10, "num_machines": 10})
    model = MGLJSSPModel.load_from_checkpoint(
        model_path.as_posix(),
        env=env,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model


def _evaluate_file(
    file_path: Path,
    model,
    device: torch.device,
    num_samples: int,
    use_greedy: bool,
) -> dict[str, object]:
    instance = load_instance(file_path.as_posix(), device=str(device))
    start = time.perf_counter()
    makespans, entropies, _ = sampling(
        instance,
        model.encoder,
        model.decoder,
        bs=int(num_samples),
        use_greedy=bool(use_greedy),
        device=str(device),
    )
    elapsed = time.perf_counter() - start
    ref = float(instance["makespan"])
    gaps = (makespans / ref - 1.0) * 100.0
    return {
        "instance": instance["name"],
        "shape": instance["shape"],
        "ref_makespan": ref,
        "pred_makespan": float(makespans.min().item()),
        "gap": float(gaps.min().item()),
        "gap_avg": float(gaps.mean().item()),
        "gap_max": float(gaps.max().item()),
        "entropy_mean": float(entropies.mean().item()),
        "time_sec": float(elapsed),
    }


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    gaps = [float(row["gap"]) for row in rows]
    times = [float(row["time_sec"]) for row in rows]
    by_shape: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_shape[str(row["shape"])].append(float(row["gap"]))
    return {
        "count": len(rows),
        "avg_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "avg_time_sec": (sum(times) / len(times)) if times else 0.0,
        "shape_avg_gap": {
            shape: sum(values) / len(values) for shape, values in sorted(by_shape.items())
        },
    }


def main() -> int:
    args = _parse_args()
    prepare_bopo_jsp(REPO_ROOT)

    model_path = args.model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path.as_posix()}")

    device = torch.device(args.device)
    model = _load_model(model_path, device)
    use_greedy = model.use_greedy if args.greedy is None else bool(args.greedy)

    set_dirs = _default_set_dirs(REPO_ROOT)
    requested_sets: list[tuple[str, Path]] = []
    for name in args.sets:
        if name not in set_dirs:
            raise ValueError(f"Unknown built-in set: {name}")
        requested_sets.append((name, set_dirs[name]))
    requested_sets.extend((name, path.resolve()) for name, path in args.ood_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPO_ROOT / "logs" / "eval" / f"{model_path.stem}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "model_path": model_path.as_posix(),
        "device": str(device),
        "B": int(args.B),
        "greedy": int(use_greedy),
        "baseline": model.baseline,
        "sets": {},
    }

    for set_name, set_dir in requested_sets:
        if not set_dir.is_dir():
            raise FileNotFoundError(f"Benchmark directory not found: {set_dir.as_posix()}")

        rows = []
        for file_path in sorted(set_dir.glob("*.jsp")):
            row = _evaluate_file(
                file_path,
                model,
                device,
                num_samples=int(args.B),
                use_greedy=bool(use_greedy),
            )
            row["set"] = set_name
            rows.append(row)
            print(
                f"[{set_name}] {row['instance']} shape={row['shape']} "
                f"ms={row['pred_makespan']:.3f} gap={row['gap']:.3f} "
                f"time={row['time_sec']:.3f}s",
                flush=True,
            )

        csv_path = output_dir / f"{set_name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "set",
                    "instance",
                    "shape",
                    "ref_makespan",
                    "pred_makespan",
                    "gap",
                    "gap_avg",
                    "gap_max",
                    "entropy_mean",
                    "time_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        set_summary = _summarize_rows(rows)
        summary["sets"][set_name] = {
            "dir": set_dir.as_posix(),
            "csv_path": csv_path.as_posix(),
            **set_summary,
        }
        print(
            f"[{set_name}] avg_gap={float(set_summary['avg_gap']):.3f} "
            f"avg_time={float(set_summary['avg_time_sec']):.3f}s",
            flush=True,
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=True, indent=2)
    print(f"Saved summary to {summary_path.as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
