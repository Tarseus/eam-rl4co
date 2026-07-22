from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl4co.envs import get_env
from rl4co.models import POMO
from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy
from rl4co.utils.ops import unbatchify
from scripts.eval_downloaded_routing_checkpoints import (
    DownloadEntry,
    _finalize_model_from_payload,
    _patch_legacy_policy_object,
    checkpoint_hparams,
    load_manifest,
)


@dataclass(frozen=True)
class InstanceResult:
    target_size: int
    method: str
    instance_index: int
    seed: int
    num_starts: int
    num_augment: int
    instance_batch_size: int
    tour_length: float
    elapsed_sec: float
    peak_memory_allocated_gib: float | None
    peak_memory_reserved_gib: float | None


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_sizes(raw: str) -> list[int]:
    sizes = [int(token) for token in _parse_csv(raw)]
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("target sizes must contain integers >= 2")
    return sizes


def _autocast_context(device: torch.device, precision: str):
    normalized = str(precision).strip().lower()
    if normalized in {"32", "fp32", "float32", "32-true"}:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"precision={precision} requires a CUDA device")
    if normalized in {"16", "fp16", "float16", "16-mixed"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if normalized in {"bf16", "bfloat16", "bf16-mixed"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported precision: {precision}")


def _load_tsp_model(
    entry: DownloadEntry,
    *,
    target_size: int,
    device: torch.device,
    require_po4cops_policy: bool = True,
) -> tuple[POMO, Any]:
    raw_hparams = checkpoint_hparams(entry.checkpoint_path)
    payload = raw_hparams.pop("_checkpoint_payload")
    # Older standard POMO checkpoints may retain experimental EA settings that
    # are not accepted by the current Lightning module constructor.
    raw_hparams.pop("ea_kwargs", None)
    raw_hparams["policy"] = _patch_legacy_policy_object(raw_hparams.get("policy"))
    env = get_env(
        "tsp",
        generator_params={"num_loc": int(target_size)},
        seed=int(raw_hparams.get("seed", 1234)),
    )
    raw_hparams["env"] = env
    raw_hparams["num_starts"] = int(target_size)
    model = POMO(**raw_hparams)
    model = _finalize_model_from_payload(model=model, payload=payload)
    if require_po4cops_policy and not isinstance(model.policy, PO4COPsTSPPolicy):
        raise TypeError(
            "Scale-transfer evaluation requires PO4COPsTSPPolicy, "
            f"got {type(model.policy).__name__} for {entry.method}"
        )
    model = model.to(device)
    model.eval()
    return model, env


def _make_instances(*, target_size: int, num_instances: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(target_size) * 1_000_003)
    return torch.rand(
        (int(num_instances), int(target_size), 2),
        generator=generator,
        dtype=torch.float32,
    )


def _load_instances(
    dataset_path: Path,
    *,
    target_size: int,
    num_instances: int,
) -> torch.Tensor:
    with np.load(dataset_path) as payload:
        if "locs" not in payload.files:
            raise ValueError(f"Dataset {dataset_path} does not contain a 'locs' array")
        locs = np.asarray(payload["locs"])
    expected_tail = (int(target_size), 2)
    if locs.ndim != 3 or tuple(locs.shape[1:]) != expected_tail:
        raise ValueError(
            f"Expected dataset shape (N, {target_size}, 2), got {tuple(locs.shape)}"
        )
    if locs.shape[0] < int(num_instances):
        raise ValueError(
            f"Dataset has {locs.shape[0]} instances, fewer than requested {num_instances}"
        )
    if not np.isfinite(locs[:num_instances]).all():
        raise ValueError(f"Dataset {dataset_path} contains non-finite coordinates")
    return torch.from_numpy(locs[:num_instances].astype(np.float32, copy=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    torch.cuda.synchronize(device)
    gib = float(1024**3)
    return (
        float(torch.cuda.max_memory_allocated(device)) / gib,
        float(torch.cuda.max_memory_reserved(device)) / gib,
    )


def evaluate_instance(
    *,
    model: POMO,
    env: Any,
    locs: torch.Tensor,
    num_starts: int,
    num_augment: int,
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, float, float | None, float | None]:
    if locs.shape != (1, env.generator.num_loc, 2):
        raise ValueError(
            f"Expected locs shape (1, {env.generator.num_loc}, 2), got {tuple(locs.shape)}"
        )
    _reset_peak_memory(device)
    started_at = time.perf_counter()
    td = env.reset(TensorDict({"locs": locs.to(device)}, batch_size=[1])).to(device)
    if num_augment > 1:
        if model.augment is None or int(model.num_augment) != num_augment:
            raise ValueError(
                f"Requested num_augment={num_augment}, but checkpoint provides "
                f"num_augment={model.num_augment}"
            )
        td = model.augment(td)
    with torch.inference_mode(), _autocast_context(device, precision):
        out = model.policy(
            td,
            env,
            phase="test",
            num_starts=int(num_starts),
            return_actions=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started_at
    peak_allocated, peak_reserved = _peak_memory(device)
    rewards = unbatchify(
        out["reward"],
        (int(num_augment), int(num_starts)),
    ).detach().cpu()
    return rewards, elapsed, peak_allocated, peak_reserved


def _select_entries(manifest: Path, methods: list[str]) -> list[DownloadEntry]:
    method_set = set(methods)
    entries = [
        entry
        for entry in load_manifest(manifest, REPO_ROOT)
        if entry.problem_key == "tsp100" and entry.method in method_set
    ]
    by_method = {entry.method: entry for entry in entries}
    missing = [method for method in methods if method not in by_method]
    if missing:
        raise ValueError(f"Missing tsp100 manifest entries for methods: {missing}")
    selected = [by_method[method] for method in methods]
    for entry in selected:
        if not entry.checkpoint_path.exists():
            raise FileNotFoundError(entry.checkpoint_path)
    return selected


def _write_csv(path: Path, rows: list[InstanceResult]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _aggregate(rows: list[InstanceResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[InstanceResult]] = {}
    for row in rows:
        grouped.setdefault((row.target_size, row.method), []).append(row)

    summaries = []
    for (target_size, method), group in sorted(grouped.items()):
        lengths = np.asarray([row.tour_length for row in group], dtype=np.float64)
        elapsed = np.asarray([row.elapsed_sec for row in group], dtype=np.float64)
        peak_allocated = [
            row.peak_memory_allocated_gib
            for row in group
            if row.peak_memory_allocated_gib is not None
        ]
        peak_reserved = [
            row.peak_memory_reserved_gib
            for row in group
            if row.peak_memory_reserved_gib is not None
        ]
        summaries.append(
            {
                "target_size": target_size,
                "method": method,
                "num_instances": len(group),
                "mean_tour_length": float(lengths.mean()),
                "std_tour_length": float(lengths.std(ddof=1)) if len(group) > 1 else 0.0,
                "median_tour_length": float(np.median(lengths)),
                "mean_elapsed_sec": float(elapsed.mean()),
                "max_peak_memory_allocated_gib": max(peak_allocated, default=None),
                "max_peak_memory_reserved_gib": max(peak_reserved, default=None),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact TSP scale-transfer evaluation using one instance per forward pass."
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "downloads" / "manifest.json")
    parser.add_argument("--methods", default="po,loss_only,weighting")
    parser.add_argument("--target-sizes", default="100")
    parser.add_argument("--num-instances", type=int, default=1)
    parser.add_argument("--num-starts", type=int, default=None)
    parser.add_argument(
        "--num-augment",
        type=int,
        default=None,
        help="Augmentation count; defaults to each checkpoint's test setting (8 for TSP100).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional explicit NPZ containing locs; disables seeded instance generation.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument(
        "--allow-standard-pomo-policy",
        action="store_true",
        help="Allow the recovered standard POMO policy instead of requiring PO4COPsTSPPolicy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "tsp_scale_streaming" / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_instances < 1:
        raise ValueError("num_instances must be >= 1")
    if args.num_augment is not None and args.num_augment < 1:
        raise ValueError("num_augment must be >= 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    methods = _parse_csv(args.methods)
    target_sizes = _parse_sizes(args.target_sizes)
    entries = _select_entries(args.manifest.resolve(), methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[InstanceResult] = []
    checkpoints: dict[str, dict[str, str]] = {}
    dataset_path = args.dataset.resolve() if args.dataset is not None else None
    dataset_sha256 = _sha256(dataset_path) if dataset_path is not None else None
    for target_size in target_sizes:
        num_starts = int(args.num_starts or target_size)
        if num_starts < 1 or num_starts > target_size:
            raise ValueError(f"num_starts must be in [1, {target_size}], got {num_starts}")
        output_dataset = args.output_dir / f"tsp{target_size}_seed{args.seed}.npz"
        if dataset_path is None:
            instances = _make_instances(
                target_size=target_size,
                num_instances=args.num_instances,
                seed=args.seed,
            )
            np.savez_compressed(output_dataset, locs=instances.numpy())
        else:
            instances = _load_instances(
                dataset_path,
                target_size=target_size,
                num_instances=args.num_instances,
            )
            if dataset_path != output_dataset.resolve():
                shutil.copy2(dataset_path, output_dataset)

        for entry in entries:
            print(f"[load] method={entry.method} target_size={target_size}", flush=True)
            checkpoints[entry.method] = {
                "path": str(entry.checkpoint_path.resolve()),
                "sha256": _sha256(entry.checkpoint_path),
            }
            model, env = _load_tsp_model(
                entry,
                target_size=target_size,
                device=device,
                require_po4cops_policy=not args.allow_standard_pomo_policy,
            )
            num_augment = int(args.num_augment or model.num_augment)
            for instance_index in range(args.num_instances):
                locs = instances[instance_index : instance_index + 1]
                try:
                    rewards, elapsed, peak_allocated, peak_reserved = evaluate_instance(
                        model=model,
                        env=env,
                        locs=locs,
                        num_starts=num_starts,
                        num_augment=num_augment,
                        device=device,
                        precision=args.precision,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    raise RuntimeError(
                        "Exact B=1 evaluation ran out of CUDA memory. Try a GPU with more "
                        "free memory or reduce --num-starts, noting that fewer starts changes "
                        "the evaluation protocol."
                    ) from error
                tour_length = -float(rewards.max().item())
                row = InstanceResult(
                    target_size=target_size,
                    method=entry.method,
                    instance_index=instance_index,
                    seed=args.seed,
                    num_starts=num_starts,
                    num_augment=num_augment,
                    instance_batch_size=1,
                    tour_length=tour_length,
                    elapsed_sec=elapsed,
                    peak_memory_allocated_gib=peak_allocated,
                    peak_memory_reserved_gib=peak_reserved,
                )
                rows.append(row)
                _write_csv(args.output_dir / "per_instance.csv", rows)
                print(
                    f"[result] method={entry.method} size={target_size} instance={instance_index} "
                    f"length={tour_length:.6f} augment={num_augment} elapsed={elapsed:.2f}s "
                    f"peak_allocated={peak_allocated}GiB",
                    flush=True,
                )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "config": {
            "methods": methods,
            "target_sizes": target_sizes,
            "num_instances": args.num_instances,
            "num_starts": args.num_starts,
            "num_augment": args.num_augment,
            "instance_batch_size": 1,
            "seed": args.seed,
            "device": str(device),
            "precision": args.precision,
            "exact_full_forward": True,
            "allow_standard_pomo_policy": args.allow_standard_pomo_policy,
            "dataset": str(dataset_path) if dataset_path is not None else None,
            "dataset_sha256": dataset_sha256,
        },
        "checkpoints": checkpoints,
        "aggregates": _aggregate(rows),
        "rows": [asdict(row) for row in rows],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] output_dir={args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
