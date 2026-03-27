from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


def _shape_counts(target_dir: Path) -> dict[str, int]:
    counts = Counter(path.stem.split("_")[0] for path in target_dir.glob("*.jsp"))
    return {shape: int(count) for shape, count in sorted(counts.items())}


def _copy_jsp_files(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(source_dir.glob("*.jsp")):
        dst = target_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def _extract_zip_if_needed(zip_path: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob("*.jsp"))
    if existing:
        return 0
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return len(sorted(target_dir.glob("*.jsp")))


def _sync_split(
    *,
    name: str,
    target_dir: Path,
    source_dir: Path | None = None,
    source_zip: Path | None = None,
) -> dict[str, object]:
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob("*.jsp"))
    copied = 0
    extracted = False

    if not existing:
        if source_zip is not None and source_zip.is_file():
            _extract_zip_if_needed(source_zip, target_dir)
            extracted = True
        elif source_dir is not None and source_dir.is_dir():
            copied = _copy_jsp_files(source_dir, target_dir)
        else:
            raise FileNotFoundError(
                f"Could not prepare {name}: neither source_dir nor source_zip is available."
            )
        existing = sorted(target_dir.glob("*.jsp"))

    return {
        "target_dir": target_dir.as_posix(),
        "count": len(existing),
        "copied": copied,
        "extracted": extracted,
        "shape_counts": _shape_counts(target_dir),
        "source_dir": source_dir.as_posix() if source_dir is not None else None,
        "source_zip": source_zip.as_posix() if source_zip is not None else None,
    }


def prepare_bopo_jsp(
    root_dir: Path,
    *,
    source_root: Path | None = None,
    target_root: Path | None = None,
) -> dict[str, object]:
    root_dir = root_dir.resolve()
    source_root = (root_dir / "BOPO" / "JSP") if source_root is None else source_root.resolve()
    target_root = (root_dir / "data" / "jssp_bopo") if target_root is None else target_root.resolve()

    results = {
        "train": _sync_split(
            name="train",
            target_dir=target_root / "train",
            source_dir=source_root / "dataset5k",
            source_zip=source_root / "dataset5k" / "dataset5k.zip",
        ),
        "validation": _sync_split(
            name="validation",
            target_dir=target_root / "validation",
            source_dir=source_root / "benchmarks" / "validation",
            source_zip=source_root / "benchmarks" / "validation" / "validation.zip",
        ),
    }

    for name in ("TA", "LA", "DMU"):
        results[name] = _sync_split(
            name=name,
            target_dir=target_root / name,
            source_dir=source_root / "benchmarks" / name,
            source_zip=None,
        )

    results["root"] = {
        "source_root": source_root.as_posix(),
        "target_root": target_root.as_posix(),
    }
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync BOPO/JSP protocol data into this repository's own data/jssp_bopo directory."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional BOPO/JSP source root. Defaults to <root>/BOPO/JSP.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Optional repo-owned target root. Defaults to <root>/data/jssp_bopo.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = prepare_bopo_jsp(
        args.root_dir.resolve(),
        source_root=args.source_root,
        target_root=args.target_root,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=True))
        return 0

    print(
        f"source_root={summary['root']['source_root']} | target_root={summary['root']['target_root']}",
        flush=True,
    )
    for name, payload in summary.items():
        if name == "root":
            continue
        pieces = [f"{name}: dir={payload['target_dir']}", f"count={payload['count']}"]
        if payload.get("source_zip"):
            pieces.append(f"zip={payload['source_zip']}")
        if payload.get("source_dir"):
            pieces.append(f"source={payload['source_dir']}")
        pieces.append(f"copied={payload['copied']}")
        pieces.append(f"extracted={payload['extracted']}")
        print(" | ".join(pieces), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
