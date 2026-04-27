from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "baseline"
MINI_EVAL_DIR = BASELINE_DIR / "mini_eval"
TEXT_SCAN_DIRS = ("PTP", "tests", "configs", "scripts")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".sh", ".ps1", ".json"}


@dataclass(frozen=True)
class ScenarioHints:
    max_epoch: int
    hparams_paths: tuple[str, ...] = ()


SCENARIO_HINTS: dict[str, ScenarioHints] = {
    "cvrp50": ScenarioHints(
        max_epoch=100,
        hparams_paths=(
            "logs/train/runs/cvrp50_baseline_20260324-083129/cvrp50_baseline/version_0/hparams.yaml",
        ),
    ),
    "cvrp100": ScenarioHints(
        max_epoch=200,
        hparams_paths=(
            "logs/train/runs/cvrp100_baseline_20260322-062025/cvrp100_baseline/version_0/hparams.yaml",
            "logs/train/runs/cvrp100_baseline_20260322-062019/cvrp100_baseline/version_0/hparams.yaml",
            "logs/train/runs/cvrp100_baseline_20260322-043941/cvrp100_baseline/version_0/hparams.yaml",
        ),
    ),
    "tsp50": ScenarioHints(max_epoch=100),
    "tsp100": ScenarioHints(max_epoch=200),
}


def _iter_text_files() -> Iterable[Path]:
    for rel_dir in TEXT_SCAN_DIRS:
        root = REPO_ROOT / rel_dir
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            yield path


def _collect_expected_artifacts(scenario: str) -> tuple[list[Path], list[Path]]:
    ckpt_pattern = re.compile(rf"baseline/{re.escape(scenario)}_epoch_(\d+)\.ckpt")
    mini_pattern = re.compile(
        rf"baseline/mini_eval/(baseline_minitrain_{re.escape(scenario)}_[A-Za-z0-9_]+\.json)"
    )
    ckpts: set[Path] = set()
    mini_eval: set[Path] = set()

    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in ckpt_pattern.finditer(text):
            ckpts.add(REPO_ROOT.joinpath(*match.group(0).split("/")))
        for match in mini_pattern.finditer(text):
            mini_eval.add(MINI_EVAL_DIR / match.group(1).split("/")[-1])

    def _ckpt_epoch_key(path: Path) -> tuple[int, str]:
        found = re.search(r"_epoch_(\d+)\.ckpt$", path.name)
        epoch = int(found.group(1)) if found else -1
        return (epoch, path.name)

    return sorted(ckpts, key=_ckpt_epoch_key), sorted(mini_eval, key=lambda p: p.name)


def _hparams_checkpoint_dirs(hparams_path: Path) -> list[Path]:
    dirs: list[Path] = []
    if not hparams_path.is_file():
        return dirs
    try:
        payload = yaml.safe_load(hparams_path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    if isinstance(payload, dict):
        callbacks = payload.get("callbacks")
        if isinstance(callbacks, dict):
            ckpt_cfg = callbacks.get("model_checkpoint")
            if isinstance(ckpt_cfg, dict):
                raw = ckpt_cfg.get("dirpath")
                if isinstance(raw, str) and raw.strip():
                    dirs.append(Path(raw.strip()))

    # Common local run layouts:
    # .../<run>/<name>/version_0/hparams.yaml -> .../<run>/checkpoints or .../<name>/checkpoints
    for parent_idx in (3, 2):
        if len(hparams_path.parents) > parent_idx:
            dirs.append(hparams_path.parents[parent_idx] / "checkpoints")
    dirs.append(hparams_path.parent / "checkpoints")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_checkpoint_names(alias_epoch: int, *, max_epoch: int) -> list[str]:
    names: list[str] = []
    for epoch in (alias_epoch, alias_epoch - 1):
        if epoch < 0:
            continue
        names.append(f"epoch_{epoch}.ckpt")
        names.append(f"epoch_{epoch:03d}.ckpt")
    if alias_epoch == max_epoch:
        names.append("last.ckpt")
    return names


def _find_checkpoint_source(search_roots: list[Path], alias_epoch: int, *, max_epoch: int) -> Path | None:
    candidate_names = _candidate_checkpoint_names(alias_epoch, max_epoch=max_epoch)

    for name in candidate_names:
        for root in search_roots:
            direct = root / name
            if direct.is_file():
                return direct

    for name in candidate_names:
        for root in search_roots:
            if not root.exists():
                continue
            try:
                for match in root.rglob(name):
                    if match.is_file():
                        return match
            except OSError:
                continue
    return None


def _checkpoint_epoch_from_target(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)\.ckpt$", path.name)
    if not match:
        raise ValueError(f"Could not parse epoch from checkpoint target: {path}")
    return int(match.group(1))


def _default_search_roots_for_scenario(scenario: str) -> list[Path]:
    roots: list[Path] = []
    hints = SCENARIO_HINTS.get(scenario)
    if hints is None:
        return roots
    for rel in hints.hparams_paths:
        roots.extend(_hparams_checkpoint_dirs(REPO_ROOT / rel))
    return roots


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _scan_scenario(
    scenario: str,
    *,
    extra_search_roots: list[Path],
    dry_run: bool,
) -> dict[str, object]:
    expected_ckpts, expected_mini_eval = _collect_expected_artifacts(scenario)
    hints = SCENARIO_HINTS.get(scenario, ScenarioHints(max_epoch=0))
    search_roots = _default_search_roots_for_scenario(scenario) + list(extra_search_roots)

    checkpoint_status: list[dict[str, object]] = []
    for target in expected_ckpts:
        alias_epoch = _checkpoint_epoch_from_target(target)
        if target.is_file():
            checkpoint_status.append(
                {
                    "target": str(target.relative_to(REPO_ROOT)),
                    "status": "present",
                    "source": None,
                }
            )
            continue

        source = _find_checkpoint_source(search_roots, alias_epoch, max_epoch=hints.max_epoch)
        if source is None:
            checkpoint_status.append(
                {
                    "target": str(target.relative_to(REPO_ROOT)),
                    "status": "missing_source",
                    "source": None,
                }
            )
            continue

        _copy_file(source, target, dry_run=dry_run)
        checkpoint_status.append(
            {
                "target": str(target.relative_to(REPO_ROOT)),
                "status": "copied" if not dry_run else "would_copy",
                "source": str(source),
            }
        )

    mini_eval_status: list[dict[str, object]] = []
    for target in expected_mini_eval:
        mini_eval_status.append(
            {
                "target": str(target.relative_to(REPO_ROOT)),
                "status": "present" if target.is_file() else "missing",
            }
        )

    return {
        "scenario": scenario,
        "search_roots": [str(p) for p in search_roots],
        "checkpoints": checkpoint_status,
        "mini_eval": mini_eval_status,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supplement stage3 baseline artifacts by copying expected checkpoint baselines "
            "from known run/checkpoint directories into baseline/."
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario to supplement, e.g. cvrp50 or tsp50. Repeatable. Default: cvrp50",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional checkpoint search root. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files; only report what would be supplemented.",
    )
    parser.add_argument(
        "--write-report",
        type=str,
        default="baseline/stage3_baseline_supplement_report.json",
        help="JSON report path relative to repo root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scenarios = args.scenario or ["cvrp50"]
    extra_search_roots = []
    for raw in args.search_root:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        extra_search_roots.append(path)

    report = {
        "repo_root": str(REPO_ROOT),
        "dry_run": bool(args.dry_run),
        "scenarios": [],
    }

    for scenario in scenarios:
        report["scenarios"].append(
            _scan_scenario(
                scenario,
                extra_search_roots=extra_search_roots,
                dry_run=bool(args.dry_run),
            )
        )

    report_path = Path(args.write_report).expanduser()
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not args.dry_run:
        print(f"\nWrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
