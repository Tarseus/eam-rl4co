from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = REPO_ROOT / "downloads" / "manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "downloads" / "tsp_cvrp_ffsp_checkpoint_test_results.csv"
ROUTING_PREFIXES = ("tsp", "cvrp")
SUPPORTED_PREFIXES = ("cvrp", "ffsp", "tsp")
DEFAULT_FFSP_AUG_FACTOR = 128

CURVE_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "po": ("{problem}_po.csv", "{problem}_base.csv"),
    "loss_only": ("{problem}_loss_only.csv", "{problem}_best.csv"),
    "weighting": ("{problem}_best_weighting.csv",),
    "bopo": ("{problem}_bopo.csv",),
    "sll": ("{problem}_sll.csv",),
}

CSV_COLUMNS = [
    "problem_key",
    "env_name",
    "size",
    "method",
    "checkpoint_path",
    "source_checkpoint",
    "status",
    "metric_source",
    "metrics_path",
    "resolved_test_file",
    "train_max_epoch",
    "seed",
    "num_starts",
    "num_augment",
    "supports_max_aug_reward",
    "max_aug_reward_note",
    "test_data_size",
    "test_batch_size",
    "test_reward",
    "test_max_reward",
    "test_max_aug_reward",
    "elapsed_sec",
    "error",
]


@dataclass(frozen=True)
class DownloadEntry:
    problem_key: str
    method: str
    checkpoint_path: Path
    source_checkpoint: str | None
    moved_metadata: tuple[str, ...]


@dataclass(frozen=True)
class MetricSnapshot:
    path: Path
    train_max_epoch: int | None
    test_reward: float | None
    test_max_reward: float | None
    test_max_aug_reward: float | None

    @property
    def has_max_aug_reward(self) -> bool:
        return self.test_max_aug_reward is not None


def parse_problem_key(problem_key: str) -> tuple[str, int]:
    lowered = str(problem_key).strip().lower()
    for prefix in SUPPORTED_PREFIXES:
        if lowered.startswith(prefix):
            suffix = lowered[len(prefix) :]
            if suffix.isdigit():
                return prefix, int(suffix)
    raise ValueError(f"Unsupported problem key: {problem_key}")


def load_manifest(manifest_path: Path, repo_root: Path) -> list[DownloadEntry]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[DownloadEntry] = []
    for rec in payload:
        entries.append(
            DownloadEntry(
                problem_key=str(rec["problem"]).strip().lower(),
                method=str(rec["method"]).strip().lower(),
                checkpoint_path=(repo_root / str(rec["checkpoint"])).resolve(),
                source_checkpoint=rec.get("source_checkpoint"),
                moved_metadata=tuple(rec.get("moved_metadata", []) or []),
            )
        )
    return entries


def candidate_curve_paths(problem_key: str, method: str, repo_root: Path) -> list[Path]:
    patterns = CURVE_ALIAS_MAP.get(method, ())
    return [(repo_root / "curves" / pattern.format(problem=problem_key)).resolve() for pattern in patterns]


def candidate_metrics_paths(entry: DownloadEntry, repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    sibling_metrics = entry.checkpoint_path.parent / "metrics.csv"
    if sibling_metrics.exists():
        candidates.append(sibling_metrics.resolve())
    for curve_path in candidate_curve_paths(entry.problem_key, entry.method, repo_root):
        if curve_path.exists() and curve_path not in candidates:
            candidates.append(curve_path)
    return candidates


def routing_data_dir_name(env_name: str) -> str:
    return "vrp" if env_name == "cvrp" else env_name


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metric_snapshot(csv_path: Path) -> MetricSnapshot:
    train_max_epoch: int | None = None
    test_reward: float | None = None
    test_max_reward: float | None = None
    test_max_aug_reward: float | None = None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch_raw = row.get("epoch")
            if epoch_raw not in (None, ""):
                try:
                    epoch_value = int(float(epoch_raw))
                except (TypeError, ValueError):
                    epoch_value = None
                if epoch_value is not None:
                    train_max_epoch = epoch_value if train_max_epoch is None else max(train_max_epoch, epoch_value)

            reward_val = _to_float(row.get("test/reward"))
            max_reward_val = _to_float(row.get("test/max_reward"))
            max_aug_reward_val = _to_float(row.get("test/max_aug_reward"))
            if reward_val is not None:
                test_reward = reward_val
            if max_reward_val is not None:
                test_max_reward = max_reward_val
            if max_aug_reward_val is not None:
                test_max_aug_reward = max_aug_reward_val

    return MetricSnapshot(
        path=csv_path,
        train_max_epoch=train_max_epoch,
        test_reward=test_reward,
        test_max_reward=test_max_reward,
        test_max_aug_reward=test_max_aug_reward,
    )


def find_existing_max_aug_metrics(entry: DownloadEntry, repo_root: Path) -> MetricSnapshot | None:
    for path in candidate_metrics_paths(entry, repo_root):
        snapshot = extract_metric_snapshot(path)
        if snapshot.has_max_aug_reward:
            return snapshot
    return None


def find_best_known_training_snapshot(entry: DownloadEntry, repo_root: Path) -> MetricSnapshot | None:
    snapshots: list[MetricSnapshot] = []
    for path in candidate_metrics_paths(entry, repo_root):
        snapshots.append(extract_metric_snapshot(path))
    if not snapshots:
        return None
    snapshots.sort(key=lambda item: ((item.train_max_epoch or -1), str(item.path)), reverse=True)
    return snapshots[0]


def rewrite_repo_data_path(original_path: str | None, repo_root: Path) -> str | None:
    if not original_path:
        return None
    candidate = Path(str(original_path))
    if candidate.exists():
        return str(candidate.resolve())

    normalized = str(original_path).replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        suffix_path = Path(suffix)
        candidates = [(repo_root / "data" / suffix_path).resolve()]
        if suffix_path.parts and suffix_path.parts[0] == "cvrp":
            alt_parts = list(suffix_path.parts)
            alt_parts[0] = "vrp"
            if alt_parts:
                alt_parts[-1] = alt_parts[-1].replace("cvrp", "vrp", 1)
            candidates.append((repo_root / "data" / Path(*alt_parts)).resolve())

        for rewritten in candidates:
            if rewritten.exists():
                return str(rewritten)
    return None


def resolve_existing_output_rows(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("checkpoint_path", "")).strip()
            for row in reader
            if str(row.get("checkpoint_path", "")).strip()
        }


def parse_device_spec(device: str) -> tuple[str, int | list[int]]:
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return "cpu", 1
    if normalized == "cuda":
        return "gpu", [0]
    if normalized.startswith("cuda:"):
        return "gpu", [int(normalized.split(":", 1)[1])]
    raise ValueError(f"Unsupported device spec: {device}")


def to_torch_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "cuda:0"
    if normalized.startswith("cuda:"):
        return normalized
    raise ValueError(f"Unsupported device spec: {device}")


def checkpoint_hparams(ckpt_path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hyper_parameters = dict(payload.get("hyper_parameters", {}) or {})
    hyper_parameters["_checkpoint_payload"] = payload
    return hyper_parameters


def _patch_legacy_policy_object(policy: Any) -> Any:
    if policy is None:
        return None

    policy_cls_name = policy.__class__.__name__
    policy_cls_module = policy.__class__.__module__

    if (
        policy_cls_name == "PO4COPsTSPPolicy"
        and policy_cls_module.endswith("po4cops_tsp_policy")
        and not hasattr(policy, "start_node")
    ):
        policy.start_node = "pomo"

    return policy


def build_routing_env(
    *,
    env_name: str,
    size: int,
    hparams: dict[str, Any],
    repo_root: Path,
) -> tuple[Any, str | None]:
    from rl4co.envs import get_env

    old_env = hparams.get("env")
    generator = getattr(old_env, "generator", None)

    generator_params: dict[str, Any] = {"num_loc": size}
    for attr in ("min_loc", "max_loc", "vehicle_capacity", "min_demand", "max_demand", "capacity"):
        if generator is not None and hasattr(generator, attr):
            generator_params[attr] = getattr(generator, attr)

    resolved_test_file = None
    if old_env is not None:
        resolved_test_file = rewrite_repo_data_path(getattr(old_env, "test_file", None), repo_root)

    env_kwargs: dict[str, Any] = {
        "data_dir": str((repo_root / "data" / routing_data_dir_name(env_name)).resolve()),
        "seed": int(hparams.get("seed", 1234)),
    }

    if old_env is not None:
        resolved_val_file = rewrite_repo_data_path(getattr(old_env, "val_file", None), repo_root)
        for key, value in (
            ("train_file", rewrite_repo_data_path(getattr(old_env, "train_file", None), repo_root)),
            ("val_file", resolved_val_file),
            ("test_file", resolved_test_file),
            ("val_dataloader_names", getattr(old_env, "val_dataloader_names", None)),
            ("test_dataloader_names", getattr(old_env, "test_dataloader_names", None)),
            ("check_solution", getattr(old_env, "check_solution", True)),
            ("dataset_cls", getattr(old_env, "dataset_cls", None)),
        ):
            if value is not None:
                env_kwargs[key] = value

    env = get_env(env_name, generator_params=generator_params, **env_kwargs)
    return env, resolved_test_file


def build_ffsp_env(hparams: dict[str, Any]) -> tuple[Any, str | None]:
    from rl4co.envs import FFSPEnv

    old_env = hparams.get("env")
    generator = getattr(old_env, "generator", None)
    generator_params: dict[str, Any] = {}
    for attr in ("num_stage", "num_machine", "num_job", "min_time", "max_time", "flatten_stages"):
        value = getattr(generator, attr, None) if generator is not None else None
        if value is not None:
            generator_params[attr] = value

    env = FFSPEnv(generator_params=generator_params)
    return env, None


def _finalize_model_from_payload(
    *,
    model: Any,
    payload: dict[str, Any],
    data_dir: str | None = None,
) -> Any:
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.data_cfg["generate_default_data"] = False
    if data_dir is not None:
        model.data_cfg["data_dir"] = data_dir
    return model


def build_model(
    entry: DownloadEntry,
    repo_root: Path,
) -> tuple[Any, dict[str, Any], str | None]:
    from rl4co.models import MatNet, POMO

    env_name, size = parse_problem_key(entry.problem_key)
    raw_hparams = checkpoint_hparams(entry.checkpoint_path)
    payload = raw_hparams.pop("_checkpoint_payload")
    raw_hparams["policy"] = _patch_legacy_policy_object(raw_hparams.get("policy"))

    if env_name in ROUTING_PREFIXES:
        env, resolved_test_file = build_routing_env(
            env_name=env_name,
            size=size,
            hparams=raw_hparams,
            repo_root=repo_root,
        )
        raw_hparams["env"] = env
        model = POMO(**raw_hparams)
        model = _finalize_model_from_payload(
            model=model,
            payload=payload,
            data_dir=str((repo_root / "data" / routing_data_dir_name(env_name)).resolve()),
        )
        return model, raw_hparams, resolved_test_file

    if env_name == "ffsp":
        env, resolved_test_file = build_ffsp_env(raw_hparams)
        raw_hparams["env"] = env
        model = MatNet(**raw_hparams)
        model = _finalize_model_from_payload(model=model, payload=payload)
        return model, raw_hparams, resolved_test_file

    raise ValueError(f"Unsupported environment: {env_name}")


def run_ffsp_augmented_evaluation(
    *,
    model: Any,
    hparams: dict[str, Any],
    device: str,
    ffsp_aug_factor: int,
    ffsp_aug_batch_size: int,
) -> dict[str, Any]:
    import torch

    from rl4co.utils.ops import batchify, unbatchify

    if ffsp_aug_factor < 1:
        raise ValueError(f"ffsp_aug_factor must be >= 1, got {ffsp_aug_factor}")
    if ffsp_aug_batch_size < 1:
        raise ValueError(f"ffsp_aug_batch_size must be >= 1, got {ffsp_aug_batch_size}")

    torch_device = torch.device(to_torch_device(device))
    model = model.to(torch_device)
    model.eval()
    model.setup(stage="test")

    reward_sum = 0.0
    reward_count = 0
    max_reward_sum = 0.0
    max_aug_reward_sum = 0.0
    instance_count = 0

    with torch.no_grad():
        dataloader = model.test_dataloader()
        for batch in dataloader:
            batch = batch.to(torch_device)
            td = model.env.reset(batch)
            n_start = model.num_starts
            if n_start is None or n_start <= 0:
                n_start = model.env.get_num_starts(td)

            best_aug_reward = None
            base_max_reward = None
            remaining = int(ffsp_aug_factor)

            while remaining > 0:
                aug_chunk = min(int(ffsp_aug_batch_size), remaining)
                td_aug = batchify(td.clone(), aug_chunk)
                out = model.policy(
                    td_aug,
                    model.env,
                    phase="test",
                    num_starts=n_start,
                    return_actions=False,
                )
                reward_flat = out["reward"]
                reward_ms = unbatchify(reward_flat, (0, n_start))
                reward_aug = unbatchify(reward_ms, aug_chunk)
                max_reward_aug = reward_aug.max(dim=-1).values

                if base_max_reward is None:
                    reward_sum += float(reward_aug[:, 0, :].sum().item())
                    reward_count += int(reward_aug[:, 0, :].numel())
                    base_max_reward = max_reward_aug[:, 0]
                    best_aug_reward = max_reward_aug.max(dim=1).values
                else:
                    best_aug_reward = torch.maximum(best_aug_reward, max_reward_aug.max(dim=1).values)

                remaining -= aug_chunk

            assert base_max_reward is not None
            assert best_aug_reward is not None
            max_reward_sum += float(base_max_reward.sum().item())
            max_aug_reward_sum += float(best_aug_reward.sum().item())
            instance_count += int(base_max_reward.numel())

    supports_max_aug_reward = ffsp_aug_factor > 1
    return {
        "seed": int(hparams.get("seed", 1234)),
        "num_starts": int(hparams.get("num_starts") or 0),
        "num_augment": int(ffsp_aug_factor),
        "supports_max_aug_reward": supports_max_aug_reward,
        "max_aug_reward_note": (
            f"FFSP max_aug_reward computed via {ffsp_aug_factor} RandomOneHot inference passes "
            f"batched in chunks of {ffsp_aug_batch_size}."
            if supports_max_aug_reward
            else "FFSP evaluated with a single RandomOneHot inference pass."
        ),
        "test_data_size": int(model.data_cfg["test_data_size"]),
        "test_batch_size": int(model.test_batch_size),
        "resolved_test_file": None,
        "test_reward": (reward_sum / reward_count) if reward_count > 0 else None,
        "test_max_reward": (max_reward_sum / instance_count) if instance_count > 0 else None,
        "test_max_aug_reward": (
            (max_aug_reward_sum / instance_count) if supports_max_aug_reward and instance_count > 0 else None
        ),
    }


def run_single_evaluation(
    *,
    entry: DownloadEntry,
    repo_root: Path,
    device: str,
    precision: str,
    num_instances: int | None,
    test_batch_size: int | None,
    ffsp_aug_factor: int,
    ffsp_aug_batch_size: int,
) -> dict[str, Any]:
    import lightning as L
    from rl4co.utils.trainer import RL4COTrainer

    env_name, _ = parse_problem_key(entry.problem_key)
    model, hparams, resolved_test_file = build_model(entry, repo_root)
    metrics_cfg = list(dict.fromkeys(["reward", "max_reward", "max_aug_reward"]))
    model.test_metrics = metrics_cfg

    if num_instances is not None:
        model.data_cfg["test_data_size"] = int(num_instances)
    if test_batch_size is not None:
        model.data_cfg["test_batch_size"] = int(test_batch_size)

    seed_value = int(hparams.get("seed", 1234))
    L.seed_everything(seed_value, workers=True)

    accelerator, devices = parse_device_spec(device)
    trainer = RL4COTrainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        precision=precision,
        gradient_clip_val=None,
    )

    started_at = time.time()
    if env_name == "ffsp":
        result = run_ffsp_augmented_evaluation(
            model=model,
            hparams=hparams,
            device=device,
            ffsp_aug_factor=ffsp_aug_factor,
            ffsp_aug_batch_size=ffsp_aug_batch_size,
        )
    else:
        trainer.test(model=model, verbose=False)
        callback_metrics = dict(trainer.callback_metrics)
        num_augment = int(getattr(model, "num_augment", hparams.get("num_augment") or 0) or 0)
        supports_max_aug_reward = num_augment > 1
        max_aug_reward_note = ""
        if not supports_max_aug_reward:
            max_aug_reward_note = "Checkpoint num_augment<=1, so no augmentation metric is emitted."
        result = {
            "seed": seed_value,
            "num_starts": int(hparams.get("num_starts") or 0),
            "num_augment": num_augment,
            "supports_max_aug_reward": supports_max_aug_reward,
            "max_aug_reward_note": max_aug_reward_note,
            "test_data_size": int(model.data_cfg["test_data_size"]),
            "test_batch_size": int(model.test_batch_size),
            "resolved_test_file": resolved_test_file,
            "test_reward": _tensor_to_float(callback_metrics.get("test/reward")),
            "test_max_reward": _tensor_to_float(callback_metrics.get("test/max_reward")),
            "test_max_aug_reward": _tensor_to_float(callback_metrics.get("test/max_aug_reward")),
        }
    elapsed = time.time() - started_at
    result["elapsed_sec"] = round(elapsed, 3)
    return result


def _tensor_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def append_row(output_csv: Path, row: dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def build_base_row(entry: DownloadEntry) -> dict[str, Any]:
    env_name, size = parse_problem_key(entry.problem_key)
    return {
        "problem_key": entry.problem_key,
        "env_name": env_name,
        "size": size,
        "method": entry.method,
        "checkpoint_path": str(entry.checkpoint_path),
        "source_checkpoint": entry.source_checkpoint or "",
        "status": "",
        "metric_source": "",
        "metrics_path": "",
        "resolved_test_file": "",
        "train_max_epoch": "",
        "seed": "",
        "num_starts": "",
        "num_augment": "",
        "supports_max_aug_reward": "",
        "max_aug_reward_note": "",
        "test_data_size": "",
        "test_batch_size": "",
        "test_reward": "",
        "test_max_reward": "",
        "test_max_aug_reward": "",
        "elapsed_sec": "",
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate downloaded tsp/cvrp/ffsp checkpoints under downloads/ and write a unified CSV. "
            "Entries that already have test/max_aug_reward are copied into the CSV and skipped."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--problems",
        type=str,
        default="tsp50,tsp100,cvrp50,cvrp100,ffsp50,ffsp100",
        help="Comma-separated problem keys to consider.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Optional comma-separated method filter, e.g. bopo,sll,weighting.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Override test_data_size for a smaller or larger evaluation run.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Override test batch size used by the LightningModule dataloader.",
    )
    parser.add_argument(
        "--ffsp-aug-factor",
        type=int,
        default=DEFAULT_FFSP_AUG_FACTOR,
        help="Number of repeated RandomOneHot inference passes used to compute FFSP test/max_aug_reward.",
    )
    parser.add_argument(
        "--ffsp-aug-batch-size",
        type=int,
        default=DEFAULT_FFSP_AUG_FACTOR,
        help="How many FFSP augmentation passes to batch together in one forward call.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Evaluate even if existing metrics already contain test/max_aug_reward.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints that already have a row in the output CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many manifest entries to process after filtering.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem_filter = {
        token.strip().lower()
        for token in str(args.problems).split(",")
        if token.strip()
    }
    method_filter = {
        token.strip().lower()
        for token in str(args.methods).split(",")
        if token.strip()
    }
    seen_rows = resolve_existing_output_rows(args.output_csv) if args.resume else set()
    entries = load_manifest(args.manifest, REPO_ROOT)
    selected = [
        entry
        for entry in entries
        if entry.problem_key in problem_filter and (not method_filter or entry.method in method_filter)
    ]
    if args.limit is not None:
        selected = selected[: int(args.limit)]

    processed = 0
    skipped_existing = 0
    skipped_resume = 0
    evaluated = 0
    errored = 0

    for entry in selected:
        checkpoint_key = str(entry.checkpoint_path)
        if checkpoint_key in seen_rows:
            skipped_resume += 1
            print(f"[resume] skipping already recorded checkpoint: {checkpoint_key}")
            continue

        row = build_base_row(entry)
        known_training = find_best_known_training_snapshot(entry, REPO_ROOT)
        if known_training is not None and known_training.train_max_epoch is not None:
            row["train_max_epoch"] = known_training.train_max_epoch

        existing_metrics = find_existing_max_aug_metrics(entry, REPO_ROOT)
        if existing_metrics is not None and not args.force:
            row.update(
                {
                    "status": "skipped_existing_metrics",
                    "metric_source": "existing_metrics",
                    "metrics_path": str(existing_metrics.path),
                    "supports_max_aug_reward": True,
                    "test_reward": existing_metrics.test_reward,
                    "test_max_reward": existing_metrics.test_max_reward,
                    "test_max_aug_reward": existing_metrics.test_max_aug_reward,
                }
            )
            append_row(args.output_csv, row)
            processed += 1
            skipped_existing += 1
            print(f"[skip] {entry.problem_key}/{entry.method} already has test/max_aug_reward in {existing_metrics.path}")
            continue

        try:
            result = run_single_evaluation(
                entry=entry,
                repo_root=REPO_ROOT,
                device=args.device,
                precision=args.precision,
                num_instances=args.num_instances,
                test_batch_size=args.test_batch_size,
                ffsp_aug_factor=args.ffsp_aug_factor,
                ffsp_aug_batch_size=args.ffsp_aug_batch_size,
            )
            supports_max_aug_reward = bool(
                result.get("supports_max_aug_reward", result.get("test_max_aug_reward") is not None)
            )
            max_aug_reward_note = str(result.get("max_aug_reward_note", "") or "")
            row.update(
                {
                    "status": (
                        "evaluated"
                        if supports_max_aug_reward or result["test_max_aug_reward"] is not None
                        else "evaluated_without_aug_metric"
                    ),
                    "metric_source": "fresh_eval",
                    "resolved_test_file": result["resolved_test_file"] or "",
                    "seed": result["seed"],
                    "num_starts": result["num_starts"],
                    "num_augment": result["num_augment"],
                    "supports_max_aug_reward": supports_max_aug_reward,
                    "max_aug_reward_note": max_aug_reward_note,
                    "test_data_size": result["test_data_size"],
                    "test_batch_size": result["test_batch_size"],
                    "test_reward": result["test_reward"],
                    "test_max_reward": result["test_max_reward"],
                    "test_max_aug_reward": result["test_max_aug_reward"],
                    "elapsed_sec": result["elapsed_sec"],
                }
            )
            append_row(args.output_csv, row)
            processed += 1
            evaluated += 1
            print(
                f"[eval] {entry.problem_key}/{entry.method} "
                f"max_reward={result['test_max_reward']} max_aug_reward={result['test_max_aug_reward']}"
            )
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "status": "error",
                    "metric_source": "fresh_eval",
                    "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                }
            )
            append_row(args.output_csv, row)
            processed += 1
            errored += 1
            print(f"[error] {entry.problem_key}/{entry.method}: {exc}")

    print(
        "done:",
        json.dumps(
            {
                "selected": len(selected),
                "processed": processed,
                "evaluated": evaluated,
                "skipped_existing": skipped_existing,
                "skipped_resume": skipped_resume,
                "errored": errored,
                "output_csv": str(args.output_csv),
            },
            ensure_ascii=False,
        ),
    )
    return 0 if errored == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
