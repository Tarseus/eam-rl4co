#!/usr/bin/env python3
"""Deterministically sample search candidates across fitness quantiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def candidate_fitness(row: Mapping[str, Any]) -> float | None:
    if not bool(row.get("pair_ok")):
        return None
    if str(row.get("stage_final") or "") != "high_fidelity":
        return None
    if not isinstance(row.get("g_ir"), Mapping) or not isinstance(row.get("f_ir"), Mapping):
        return None
    raw = row.get("final_score", row.get("score"))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def eligible_candidates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        fitness = candidate_fitness(row)
        if fitness is None:
            continue
        key = (str(row.get("g_id") or ""), str(row.get("f_id") or ""))
        if not all(key):
            continue
        prior = best_by_pair.get(key)
        if prior is None or fitness < float(prior["_selection_fitness"]):
            item = dict(row)
            item["_selection_fitness"] = fitness
            best_by_pair[key] = item
    return sorted(
        best_by_pair.values(),
        key=lambda row: (
            float(row["_selection_fitness"]),
            str(row.get("g_id") or ""),
            str(row.get("f_id") or ""),
        ),
    )


def quantile_indices(population_size: int, sample_size: int) -> list[int]:
    if population_size < 1:
        raise ValueError("no eligible candidates")
    if sample_size < 2:
        raise ValueError("num_candidates must be at least 2")
    sample_size = min(sample_size, population_size)
    if sample_size == population_size:
        return list(range(population_size))
    indices = [round(i * (population_size - 1) / (sample_size - 1)) for i in range(sample_size)]
    if len(set(indices)) != len(indices):
        raise AssertionError("quantile selection produced duplicate indices")
    return indices


def materialize(pairs_jsonl: Path, output_dir: Path, num_candidates: int) -> list[dict[str, Any]]:
    population = eligible_candidates(iter_jsonl(pairs_jsonl))
    indices = quantile_indices(len(population), num_candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for sample_index, population_index in enumerate(indices):
        row = dict(population[population_index])
        fitness = float(row.pop("_selection_fitness"))
        quantile = population_index / max(1, len(population) - 1)
        candidate_id = f"q{sample_index:02d}_{row['g_id']}__{row['f_id']}"
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        pair_path = candidate_dir / "best_pair.json"
        builder_path = candidate_dir / "best_builder.json"
        loss_path = candidate_dir / "best_loss.json"
        pair_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        builder_path.write_text(
            json.dumps({"id": row["g_id"], "ir": row["g_ir"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        loss_path.write_text(
            json.dumps({"id": row["f_id"], "ir": row["f_ir"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest.append(
            {
                "candidate_id": candidate_id,
                "selection_index": sample_index,
                "population_rank": population_index,
                "population_size": len(population),
                "fitness_quantile": quantile,
                "g_id": row["g_id"],
                "f_id": row["f_id"],
                "search_fitness": fitness,
                "pair_json": pair_path.as_posix(),
                "final_cost": "",
                "final_result_source": "",
            }
        )
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (output_dir / "selection.json").write_text(
        json.dumps(
            {
                "source": pairs_jsonl.as_posix(),
                "eligible_population_size": len(population),
                "requested_candidates": num_candidates,
                "selected_indices": indices,
                "selection_rule": "even ranks after ascending minimization fitness sort",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = materialize(args.pairs_jsonl, args.output_dir, args.num_candidates)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
