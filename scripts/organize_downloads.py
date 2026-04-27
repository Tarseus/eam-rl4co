from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_ROOT = REPO_ROOT / "downloads"
ARCHIVE_ROOT = REPO_ROOT / "artifacts" / "downloads_archive"

REQUESTED_PROBLEMS = [
    "tsp50",
    "tsp100",
    "cvrp50",
    "cvrp100",
    "ffsp50",
    "ffsp100",
    "jssp10x10",
    "jssp15x15",
]


@dataclass(frozen=True)
class MethodSpec:
    problem: str
    method: str
    checkpoint_rel: str
    metadata_dir_rel: str | None = None


def _method_dir_spec(problem: str, method: str) -> MethodSpec:
    base = f"downloads/final_checkpoints/{problem}/{method}"
    return MethodSpec(problem=problem, method=method, checkpoint_rel=base, metadata_dir_rel=base)


SPECS: list[MethodSpec] = [
    MethodSpec("tsp50", "bopo", "downloads/checkpoints_non_g53/tsp50_g51/last.ckpt"),
    MethodSpec(
        "tsp50",
        "sll",
        "downloads/final_checkpoints/g51/runs/tsp50_cvrp50_sll_20260413-025121__tsp50_sll_seed1234__epoch_405.ckpt",
    ),
    MethodSpec("tsp50", "weighting", "downloads/final_checkpoints/g51/runs/loss_weighting_tsp50__epoch_904.ckpt"),
    _method_dir_spec("tsp100", "po"),
    _method_dir_spec("tsp100", "sll"),
    _method_dir_spec("tsp100", "bopo"),
    _method_dir_spec("tsp100", "loss_only"),
    _method_dir_spec("tsp100", "weighting"),
    MethodSpec("cvrp50", "bopo", "downloads/checkpoints_non_g53/cvrp50_g51/last.ckpt"),
    MethodSpec(
        "cvrp50",
        "sll",
        "downloads/final_checkpoints/g51/runs/tsp50_cvrp50_sll_20260413-025121__cvrp50_sll_seed1234__epoch_315.ckpt",
    ),
    _method_dir_spec("cvrp100", "po"),
    _method_dir_spec("cvrp100", "sll"),
    _method_dir_spec("cvrp100", "bopo"),
    _method_dir_spec("cvrp100", "loss_only"),
    _method_dir_spec("cvrp100", "weighting"),
    _method_dir_spec("ffsp50", "rl"),
    _method_dir_spec("ffsp50", "po"),
    _method_dir_spec("ffsp50", "bopo"),
    MethodSpec("ffsp100", "po", "downloads/final_checkpoints/g49/baseline/ffsp100_epoch100.ckpt"),
    MethodSpec("ffsp100", "bopo", "downloads/checkpoints_non_g53/ffsp100_g52/last.ckpt"),
    _method_dir_spec("jssp10x10", "rl"),
    _method_dir_spec("jssp10x10", "po"),
    _method_dir_spec("jssp10x10", "bopo"),
    _method_dir_spec("jssp10x10", "sll"),
    _method_dir_spec("jssp10x10", "loss_only"),
    _method_dir_spec("jssp10x10", "weighting"),
    _method_dir_spec("jssp15x15", "rl"),
    _method_dir_spec("jssp15x15", "po"),
    _method_dir_spec("jssp15x15", "bopo"),
    _method_dir_spec("jssp15x15", "sll"),
    _method_dir_spec("jssp15x15", "weighting"),
]

LEGACY_DOWNLOAD_DIRS = [
    "all_checkpoints",
    "checkpoints_non_g53",
    "final_checkpoints",
    "final_checkpoints_dryrun",
    "jssp_only_dryrun",
]


def resolve_checkpoint_source(spec: MethodSpec) -> Path:
    source = REPO_ROOT / spec.checkpoint_rel
    if source.is_file():
        return source
    if source.is_dir():
        ckpts = sorted(source.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in directory: {source}")
        for candidate in ckpts:
            if candidate.name == "last.ckpt":
                return candidate
        if len(ckpts) == 1:
            return ckpts[0]
        raise RuntimeError(f"Multiple checkpoints in {source}, but no last.ckpt to disambiguate: {ckpts}")
    raise FileNotFoundError(f"Checkpoint source not found: {source}")


def move_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Target already exists: {dst}")
    shutil.move(str(src), str(dst))
    return True


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = ARCHIVE_ROOT / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []

    for problem in REQUESTED_PROBLEMS:
        (DOWNLOADS_ROOT / problem).mkdir(parents=True, exist_ok=True)

    for spec in SPECS:
        src_ckpt = resolve_checkpoint_source(spec)
        target_dir = DOWNLOADS_ROOT / spec.problem / spec.method
        target_ckpt = target_dir / "checkpoint.ckpt"
        target_dir.mkdir(parents=True, exist_ok=True)

        if target_ckpt.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {target_ckpt}")

        shutil.move(str(src_ckpt), str(target_ckpt))

        moved_metadata: list[str] = []
        if spec.metadata_dir_rel:
            metadata_dir = REPO_ROOT / spec.metadata_dir_rel
            for name in ("metrics.csv", "hparams.yaml"):
                src_meta = metadata_dir / name
                dst_meta = target_dir / name
                if move_if_exists(src_meta, dst_meta):
                    moved_metadata.append(name)

        manifest.append(
            {
                "problem": spec.problem,
                "method": spec.method,
                "checkpoint": str(target_ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
                "source_checkpoint": str(src_ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
                "moved_metadata": moved_metadata,
            }
        )

    for legacy_name in LEGACY_DOWNLOAD_DIRS:
        legacy_path = DOWNLOADS_ROOT / legacy_name
        if legacy_path.exists():
            shutil.move(str(legacy_path), str(archive_dir / legacy_name))

    manifest_path = DOWNLOADS_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "archive_dir": str(archive_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "problems": REQUESTED_PROBLEMS,
        "entries": len(manifest),
    }
    (DOWNLOADS_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
