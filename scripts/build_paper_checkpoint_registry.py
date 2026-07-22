from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SPEC = REPO_ROOT / "paper_materials" / "statistical_analysis" / "checkpoint_spec.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_materials" / "statistical_analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    hparams = dict(payload.get("hyper_parameters", {}) or {})
    optimizer_kwargs = hparams.get("optimizer_kwargs")
    if not isinstance(optimizer_kwargs, dict):
        optimizer_kwargs = {}
    policy = hparams.get("policy")
    return {
        "checkpoint_epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "loss_type": hparams.get("loss_type"),
        "pref_pair_json_path": hparams.get("pref_pair_json_path"),
        "num_starts": hparams.get("num_starts"),
        "num_augment": hparams.get("num_augment"),
        "optimizer_lr": optimizer_kwargs.get("lr"),
        "policy_class": (
            f"{policy.__class__.__module__}.{policy.__class__.__name__}"
            if policy is not None
            else None
        ),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "problem",
        "paper_method",
        "legacy_method",
        "checkpoint_epoch",
        "main_tex_cost",
        "checkpoint",
        "sha256",
    ]
    lines = [
        "# Paper checkpoint registry",
        "",
        "`loss_only` maps to USW and `weighting` maps to ASW throughout the legacy artifacts.",
        "",
        "| Problem | Paper method | Legacy name | Epoch | Pre-audit draft cost | Checkpoint | SHA256 |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        values = {column: row.get(column) for column in columns}
        lines.append(
            "| {problem} | {paper_method} | {legacy_method} | {checkpoint_epoch} | "
            "{main_tex_cost} | `{checkpoint}` | `{sha256}` |".format(**values)
        )
    lines.extend(
        [
            "",
            "The `Pre-audit draft cost` column preserves the value found before this audit. "
            "It is a provenance field, not a validation that the value is correct. Fresh "
            "paired evaluation results in `paper_result_registry.md` supersede it.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible paper checkpoint registry.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = args.spec.resolve()
    output_dir = args.output_dir.resolve()
    records = json.loads(spec_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for record in records:
        checkpoint = (REPO_ROOT / record["checkpoint"]).resolve()
        row = dict(record)
        row["checkpoint"] = checkpoint.relative_to(REPO_ROOT).as_posix()
        row["exists"] = checkpoint.is_file()
        if not checkpoint.is_file():
            missing.append(row["checkpoint"])
            rows.append(row)
            continue
        row["size_bytes"] = checkpoint.stat().st_size
        row["sha256"] = sha256(checkpoint)
        row.update(checkpoint_metadata(checkpoint))
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_registry.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "checkpoint_registry.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "checkpoint_registry.md").write_text(
        markdown_table(rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": len(rows),
                "missing": missing,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
