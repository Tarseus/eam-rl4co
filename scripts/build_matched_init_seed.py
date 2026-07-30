#!/usr/bin/env python3
"""Build a strict generation-0 matched-init seed and gate replay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PTP.ptp_discovery import pref_loss_coevo_loop as loop


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _runtime_loss_signature(ir: dict[str, Any]) -> str:
    return loop._sig_free_loss(loop.free_loss_ir_from_json(ir))


def build_seed(
    *,
    losses_path: Path,
    pairs_path: Path,
    output_losses_path: Path,
    output_replay_path: Path,
    source_run: str,
    source_commit: str,
    source_losses_label: str,
    source_pairs_label: str,
    expected_count: int,
) -> None:
    loss_rows = sorted(
        (row for row in _read_jsonl(losses_path) if int(row.get("generation", -1)) == 0),
        key=lambda row: (int(row.get("index", -1)), str(row.get("id", ""))),
    )
    pair_rows = sorted(
        (row for row in _read_jsonl(pairs_path) if int(row.get("generation", -1)) == 0),
        key=lambda row: int(row.get("pair_index", -1)),
    )
    if len(loss_rows) != expected_count:
        raise ValueError(f"generation-0 loss count mismatch: {len(loss_rows)} != {expected_count}")
    if len(pair_rows) != expected_count:
        raise ValueError(f"generation-0 pair count mismatch: {len(pair_rows)} != {expected_count}")
    if [int(row.get("index", -1)) for row in loss_rows] != list(range(expected_count)):
        raise ValueError("generation-0 loss indices are not a complete zero-based range")
    if [int(row.get("pair_index", -1)) for row in pair_rows] != list(range(expected_count)):
        raise ValueError("generation-0 pair indices are not a complete zero-based range")

    loss_by_id = {str(row.get("id", "")): row for row in loss_rows}
    if len(loss_by_id) != expected_count:
        raise ValueError("generation-0 losses contain duplicate ids")
    if set(str(row.get("f_id", "")) for row in pair_rows) != set(loss_by_id):
        raise ValueError("generation-0 pair loss ids differ from the loss population")
    if any(str(row.get("g_id", "")) != "g_ref" for row in pair_rows):
        raise ValueError("generation-0 pairs do not all use g_ref")

    replay_entries: list[dict[str, Any]] = []
    for pair in pair_rows:
        hf_eligible = str(pair.get("stage", "")) == "high_fidelity"
        entry: dict[str, Any] = {
            "pair_index": int(pair["pair_index"]),
            "source_loss_id": str(pair["f_id"]),
            "hf_eligible": hf_eligible,
            "original_pair_reason": str(pair.get("pair_reason") or "matched_replay_gate_rejected"),
            "original_stage": str(pair.get("stage") or "none"),
            "original_stage_final": str(pair.get("stage_final") or "none"),
        }
        if hf_eligible:
            runtime_ir = pair.get("f_ir")
            if not isinstance(runtime_ir, dict):
                raise ValueError(f"HF-eligible pair {pair['pair_index']} lacks runtime f_ir")
            entry["runtime_loss_signature"] = _runtime_loss_signature(runtime_ir)
            entry["runtime_loss_ir"] = runtime_ir
        replay_entries.append(entry)

    output_losses_path.parent.mkdir(parents=True, exist_ok=True)
    output_replay_path.parent.mkdir(parents=True, exist_ok=True)
    with output_losses_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in loss_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    replay_payload = {
        "schema_version": 1,
        "source_run": source_run,
        "source_commit": source_commit,
        "source_artifacts": {
            "losses_path": source_losses_label,
            "losses_sha256": _sha256(losses_path),
            "pairs_path": source_pairs_label,
            "pairs_sha256": _sha256(pairs_path),
        },
        "entries": replay_entries,
    }
    output_replay_path.write_text(
        json.dumps(replay_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-losses", type=Path, required=True)
    parser.add_argument("--output-replay", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--source-losses-label", default="")
    parser.add_argument("--source-pairs-label", default="")
    parser.add_argument("--expected-count", type=int, default=16)
    args = parser.parse_args()
    build_seed(
        losses_path=args.losses.resolve(),
        pairs_path=args.pairs.resolve(),
        output_losses_path=args.output_losses.resolve(),
        output_replay_path=args.output_replay.resolve(),
        source_run=args.source_run,
        source_commit=args.source_commit,
        source_losses_label=args.source_losses_label or str(args.losses),
        source_pairs_label=args.source_pairs_label or str(args.pairs),
        expected_count=args.expected_count,
    )


if __name__ == "__main__":
    main()