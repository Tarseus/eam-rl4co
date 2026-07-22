#!/usr/bin/env python3
"""Build a best-so-far trajectory for direct joint search.

The default mode uses ``final_score`` for completed high-fidelity pairs and
``fitness.delta_mean`` for Stage-3 early-pruned pairs.  Checkpoint-only mode
instead aggregates non-scratch entries from ``fitness.per_init`` for both kinds
of records.  Gate-only and failed-gate records have no fitness and are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _checkpoint_only_fitness(row: Mapping[str, Any]) -> float | None:
    fitness = row.get("fitness")
    if not isinstance(fitness, Mapping):
        return None
    per_init = fitness.get("per_init")
    if not isinstance(per_init, Mapping):
        return None
    checkpoint_deltas: list[float] = []
    for init_name, init_record in per_init.items():
        if str(init_name) == "scratch" or not isinstance(init_record, Mapping):
            continue
        delta = _finite_float(init_record.get("delta"))
        if delta is not None:
            checkpoint_deltas.append(delta)
    if not checkpoint_deltas:
        return None
    return sum(checkpoint_deltas) / len(checkpoint_deltas)


def _candidate_fitness(row: Mapping[str, Any], *, fitness_mode: str) -> tuple[float, str] | None:
    """Convert a search record to its comparable minimization fitness."""
    reason = str(row.get("pair_reason") or "")
    if fitness_mode == "checkpoint_only":
        score = _checkpoint_only_fitness(row)
        if score is None:
            return None
        if reason == "stage3_early_pruned":
            return score, "early_pruned_checkpoint_only"
        if (
            bool(row.get("pair_ok"))
            and str(row.get("stage_final") or "") == "high_fidelity"
            and reason == "ok_stage3_offline_minitrain"
        ):
            return score, "completed_high_fidelity_checkpoint_only"
        return None
    if fitness_mode != "aggregate_delta_mean":
        raise ValueError(f"unsupported fitness_mode: {fitness_mode}")
    if reason == "stage3_early_pruned":
        fitness = row.get("fitness")
        if isinstance(fitness, Mapping):
            score = _finite_float(fitness.get("delta_mean"))
            if score is not None:
                return score, "early_pruned_delta_mean"
        return None
    if (
        bool(row.get("pair_ok"))
        and str(row.get("stage_final") or "") == "high_fidelity"
        and reason == "ok_stage3_offline_minitrain"
    ):
        score = _finite_float(row.get("final_score"))
        if score is not None:
            return score, "completed_high_fidelity"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partial_delta(row: Mapping[str, Any]) -> float | None:
    fitness = row.get("fitness")
    if not isinstance(fitness, Mapping):
        return None
    return _finite_float(fitness.get("delta_mean"))


def build_trajectory(
    records: list[dict[str, Any]],
    *,
    generations: int,
    metric_mode: str,
    fitness_mode: str = "aggregate_delta_mean",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if metric_mode != "minimize":
        raise ValueError("this trajectory currently supports metric_mode=minimize only")
    if generations <= 0:
        raise ValueError("generations must be positive")

    by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    for row in records:
        reason_counts[str(row.get("pair_reason") or "unknown")] += 1
        generation = row.get("generation")
        if isinstance(generation, int) and 0 <= generation < generations:
            by_generation[generation].append(row)

    incumbent_score: float | None = None
    incumbent_builder_id = ""
    incumbent_loss_id = ""
    incumbent_source = ""
    trajectory: list[dict[str, Any]] = []
    completed_hf_scores: list[float] = []

    for generation in range(generations):
        generation_records = by_generation.get(generation, [])
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for row in generation_records:
            converted = _candidate_fitness(row, fitness_mode=fitness_mode)
            if converted is None:
                continue
            score, source = converted
            candidates.append((score, source, row))
            if source.startswith("completed_high_fidelity"):
                completed_hf_scores.append(score)

        generation_best_score: float | str = ""
        generation_best_builder_id = ""
        generation_best_loss_id = ""
        generation_best_source = ""
        improved = False
        if candidates:
            best_score, best_source, best_row = min(
                candidates,
                key=lambda item: (
                    item[0],
                    str(item[2].get("g_id") or ""),
                    str(item[2].get("f_id") or ""),
                ),
            )
            generation_best_score = best_score
            generation_best_builder_id = str(best_row.get("g_id") or "")
            generation_best_loss_id = str(best_row.get("f_id") or "")
            generation_best_source = best_source
            if incumbent_score is None or best_score < incumbent_score:
                incumbent_score = best_score
                incumbent_builder_id = generation_best_builder_id
                incumbent_loss_id = generation_best_loss_id
                incumbent_source = best_source
                improved = True

        generation_reasons = Counter(str(row.get("pair_reason") or "unknown") for row in generation_records)
        trajectory.append(
            {
                "problem": "",
                "search_mode": "joint_pair",
                "generation": generation,
                "evaluated_pairs": len(generation_records),
                "fitness_pairs": len(candidates),
                "completed_high_fidelity_pairs": sum(1 for _, source, _ in candidates if source.startswith("completed_high_fidelity")),
                "early_pruned_converted": sum(1 for _, source, _ in candidates if source.startswith("early_pruned_")),
                "gate_only_excluded": generation_reasons.get("ok_gate_only", 0),
                "gate_failed_excluded": sum(
                    count
                    for reason, count in generation_reasons.items()
                    if reason.endswith("gate_failed") or reason in {"cheap_gate_failed", "co_gate_failed"}
                ),
                "generation_best_fitness": generation_best_score,
                "generation_best_builder_id": generation_best_builder_id,
                "generation_best_loss_id": generation_best_loss_id,
                "generation_best_source": generation_best_source,
                "best_so_far_fitness": incumbent_score if incumbent_score is not None else "",
                "incumbent_builder_id": incumbent_builder_id,
                "incumbent_loss_id": incumbent_loss_id,
                "incumbent_source": incumbent_source,
                "improved_best_so_far": improved,
            }
        )

    early_partial = [
        delta
        for row in records
        if str(row.get("pair_reason") or "") == "stage3_early_pruned"
        for delta in [_partial_delta(row)]
        if delta is not None
    ]
    selected_early_pruned = [
        converted[0]
        for row in records
        if str(row.get("pair_reason") or "") == "stage3_early_pruned"
        for converted in [_candidate_fitness(row, fitness_mode=fitness_mode)]
        if converted is not None
    ]
    audit = {
        "record_count": len(records),
        "fitness_mode": fitness_mode,
        "reason_counts": dict(sorted(reason_counts.items())),
        "completed_high_fidelity_count": len(completed_hf_scores),
        "completed_high_fidelity_score_min": min(completed_hf_scores) if completed_hf_scores else None,
        "completed_high_fidelity_score_max": max(completed_hf_scores) if completed_hf_scores else None,
        "early_pruned_partial_fitness_count": len(early_partial),
        "early_pruned_partial_fitness_min": min(early_partial) if early_partial else None,
        "early_pruned_partial_fitness_max": max(early_partial) if early_partial else None,
        "early_pruned_selected_fitness_count": len(selected_early_pruned),
        "early_pruned_selected_fitness_min": min(selected_early_pruned) if selected_early_pruned else None,
        "early_pruned_selected_fitness_max": max(selected_early_pruned) if selected_early_pruned else None,
        "final_incumbent_fitness": incumbent_score,
        "final_incumbent_builder_id": incumbent_builder_id,
        "final_incumbent_loss_id": incumbent_loss_id,
        "final_incumbent_source": incumbent_source,
    }
    return trajectory, audit


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("trajectory is empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, Any]], problem: str, audit: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[1]
    plotting_dir = repo_root / "paper_plotting"
    if plotting_dir.is_dir():
        sys.path.insert(0, str(plotting_dir))
    try:
        from style import FIGSIZE_SEARCH, paper_style
    except ImportError:
        from contextlib import nullcontext

        FIGSIZE_SEARCH = (8.2, 4.65)

        def paper_style(**_: Any):  # type: ignore[misc]
            return nullcontext()

    generations = [int(row["generation"]) for row in rows]
    fitness = [float(row["best_so_far_fitness"]) for row in rows if row["best_so_far_fitness"] != ""]
    if len(fitness) != len(rows):
        raise ValueError("at least one generation has no candidate fitness or prior incumbent")
    with paper_style(figsize=FIGSIZE_SEARCH, suppress_titles=False, extra_save_formats=("pdf",)):
        fig, ax = plt.subplots()
        ax.step(
            generations,
            fitness,
            where="post",
            color="#287271",
            linewidth=2.2,
            label="Joint-search best-so-far",
        )
        ax.scatter(generations, fitness, s=14, color="#287271", zorder=3)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best-so-far fitness (lower is better)")
        ax.set_title(f"{problem} direct joint-search trajectory")
        tick_step = max(1, len(generations) // 10)
        ax.set_xticks(sorted(set(generations[::tick_step] + [generations[-1]])))
        ax.grid(axis="y")
        ax.legend(loc="best")
        y_min = min(fitness)
        y_max = max(fitness)
        if y_min == y_max:
            ax.set_ylim(y_min - 0.012, y_max + 0.022)
        else:
            span = y_max - y_min
            ax.set_ylim(y_min - 0.18 * span, y_max + 0.32 * span)
        fig.tight_layout()
        fig.savefig(path, dpi=220, bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="Raw pairs.jsonl from the joint-search run.")
    parser.add_argument("--summary", type=Path, help="Optional raw summary.json for run provenance.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--problem", default="FFSP100")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--metric-mode", choices=("minimize",), default="minimize")
    parser.add_argument(
        "--fitness-mode",
        choices=("aggregate_delta_mean", "checkpoint_only"),
        default="aggregate_delta_mean",
        help="How to aggregate per-init fitness for each evaluated pair.",
    )
    args = parser.parse_args()

    records = list(_iter_jsonl(args.pairs))
    trajectory, audit = build_trajectory(
        records,
        generations=args.generations,
        metric_mode=args.metric_mode,
        fitness_mode=args.fitness_mode,
    )
    for row in trajectory:
        row["problem"] = args.problem

    raw_summary: dict[str, Any] = {}
    if args.summary:
        raw_summary = json.loads(args.summary.read_text(encoding="utf-8"))

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "search_trajectory.csv"
    json_path = args.outdir / "search_trajectory.json"
    summary_path = args.outdir / "fitness_summary.json"
    plot_path = args.outdir / "search_trajectory.png"
    _write_csv(csv_path, trajectory)
    json_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "problem": args.problem,
        "search_mode": "joint_pair",
        "metric_mode": args.metric_mode,
        "fitness_mode": args.fitness_mode,
        "fitness_policy": {
            "completed_high_fidelity": (
                "mean(fitness.per_init[non-scratch].delta)"
                if args.fitness_mode == "checkpoint_only"
                else "final_score"
            ),
            "stage3_early_pruned": (
                "mean(fitness.per_init[non-scratch].delta)"
                if args.fitness_mode == "checkpoint_only"
                else "fitness.delta_mean"
            ),
            "gate_only_policy": "excluded_no_fitness",
            "gate_failed_policy": "excluded_no_fitness",
            "best_so_far_update": "strictly_lower_fitness_for_minimize",
        },
        "raw_input": {
            "pairs_path": str(args.pairs.resolve()),
            "pairs_sha256": _sha256(args.pairs),
            "summary_path": str(args.summary.resolve()) if args.summary else None,
            "remote_run_dir": raw_summary.get("run_dir"),
            "raw_summary_best_score": raw_summary.get("best_score"),
            "raw_summary_best_so_far": raw_summary.get("best_so_far"),
        },
        "audit": audit,
    }
    summary_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot(plot_path, trajectory, args.problem, audit)
    print(json.dumps({"outputs": [str(csv_path), str(json_path), str(summary_path), str(plot_path)], "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
