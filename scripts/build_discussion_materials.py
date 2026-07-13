from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "discussion_materials"


LOSS_RUNS = {
    "TSP100": ROOT / "runs/pref_loss_tsp100_discovery/20260317-131507",
    "CVRP100": ROOT / "runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008",
    "FFSP100": ROOT / "runs/pref_loss_ffsp100_discovery/20260403-142801",
    "JSSP10x10": ROOT / "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409",
}

WEIGHT_RUNS = {
    "TSP100": ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757",
    "CVRP100": ROOT / "runs/pref_builder_weight_search_cvrp100/20260416-093909",
    "FFSP100": ROOT / "runs/pref_builder_weight_search_ffsp100/20260416-111514",
    "JSSP10x10": ROOT / "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def maybe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def ensure_dirs() -> None:
    for rel in [
        "",
        "5.1_objective_discovery",
        "5.1_objective_discovery/figures",
        "5.1_objective_discovery/raw",
        "5.2_surviving_design_patterns",
        "5.2_surviving_design_patterns/figures",
        "5.2_surviving_design_patterns/raw",
        "5.3_loss_side_by_problem",
        "5.3_loss_side_by_problem/figures",
        "5.4_weighting_transfer",
        "5.4_weighting_transfer/figures",
        "5.4_weighting_transfer/raw",
        "5.5_implications_limitations",
        "tables",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst_rel: str, manifest: list[dict[str, str]], note: str) -> None:
    if not src.is_file():
        return
    dst = OUT / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "material_path": str(dst.relative_to(OUT)).replace("\\", "/"),
            "source_path": str(src.relative_to(ROOT)).replace("\\", "/"),
            "note": note,
        }
    )


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[Mapping[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def candidate_meta_from_loss_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(run_dir / "losses.jsonl"):
        cid = row.get("id")
        if cid is not None:
            meta[str(cid)] = row
    return meta


def candidate_meta_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}

    def add(entry: Any) -> None:
        if not isinstance(entry, Mapping):
            return
        cid = entry.get("id") or entry.get("builder_id") or entry.get("g_id")
        if cid is not None:
            meta.setdefault(str(cid), dict(entry))

    for key in ("hof_g", "resident_pop_g", "archive_g", "elites_g", "diverse_elites_g"):
        value = checkpoint.get(key)
        if isinstance(value, list):
            for item in value:
                add(item)
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, list):
                    for sub in item:
                        add(sub)
                else:
                    add(item)
    return meta


def best_series(run_dir: Path, phase: str, meta_lookup: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint = read_json(run_dir / "checkpoint.json")
    entries: list[dict[str, Any]] = []
    for key, history in (checkpoint.get("pair_score_history_map") or {}).items():
        if not isinstance(history, list):
            continue
        key_s = str(key)
        cand_id = key_s.split("::", 1)[-1] if phase == "loss" else key_s.split("::", 1)[0]
        for rec in history:
            if not isinstance(rec, Mapping):
                continue
            gen = rec.get("generation")
            score = rec.get("score")
            if isinstance(gen, int) and finite(score):
                entries.append(
                    {
                        "generation": int(gen),
                        "score": float(score),
                        "candidate_id": cand_id,
                        "reference_score": rec.get("reference_score", ""),
                        "stage_final": rec.get("stage_final", ""),
                    }
                )
    by_gen: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in entries:
        by_gen[int(rec["generation"])].append(rec)

    rows: list[dict[str, Any]] = []
    milestones: list[dict[str, Any]] = []
    current: float | None = None
    for gen in sorted(by_gen):
        scores = sorted(float(item["score"]) for item in by_gen[gen])
        best = min(by_gen[gen], key=lambda item: (float(item["score"]), item["candidate_id"]))
        improved = current is None or float(best["score"]) < current
        if improved:
            current = float(best["score"])
            meta = meta_lookup.get(str(best["candidate_id"]), {})
            ir = meta.get("ir") if isinstance(meta, Mapping) else {}
            milestones.append(
                {
                    "phase": phase,
                    "generation": gen,
                    "candidate_id": best["candidate_id"],
                    "score": best["score"],
                    "family_signature": meta.get("family_signature") or meta.get("family") or "",
                    "op_type": meta.get("op_type", ""),
                    "intuition": (ir or {}).get("intuition", "") if isinstance(ir, Mapping) else "",
                }
            )
        rows.append(
            {
                "phase": phase,
                "generation": gen,
                "evaluated_candidates": len(scores),
                "generation_best_score": best["score"],
                "generation_mean_score": sum(scores) / len(scores),
                "best_so_far_score": current if current is not None else "",
                "best_candidate_id": best["candidate_id"],
                "improved_best_so_far": improved,
            }
        )
    return rows, milestones


def gate_counts(run_dir: Path, problem: str, phase: str) -> dict[str, Any]:
    rows = list(iter_jsonl(run_dir / "gate_reports.jsonl"))
    generated = len(rows)
    static_ok = 0
    semantic_ok = 0
    numeric_ok = 0
    high_fidelity = 0
    top_elites = 0
    pair_reason = Counter()
    for row in rows:
        if bool(row.get("static_ok")) or bool(row.get("builder_gate_ok")):
            static_ok += 1
        sem = row.get("pair_ok")
        if sem is None:
            sem = row.get("pref_semantic_ok")
        if bool(sem):
            semantic_ok += 1
        if bool(row.get("co_ok")) or bool(row.get("joint_gate_ok")):
            numeric_ok += 1
        entered = str(row.get("stage_final") or "") == "high_fidelity"
        if entered:
            high_fidelity += 1
        if entered and bool(row.get("better_than_incumbent")):
            top_elites += 1
        pair_reason[str(row.get("pair_reason") or row.get("stage_final") or "unknown")] += 1
    return {
        "problem": problem,
        "phase": phase,
        "generated_candidates": generated,
        "static_or_compile_ok": static_ok,
        "semantic_pair_ok": semantic_ok,
        "numeric_gate_ok": numeric_ok,
        "entered_high_fidelity": high_fidelity,
        "top_elites": top_elites,
        "top_rejection_or_stage_reason": "; ".join(f"{k}:{v}" for k, v in pair_reason.most_common(5)),
    }


def family_diversity_loss(run_dir: Path, problem: str) -> list[dict[str, Any]]:
    by_gen: dict[int, list[str]] = defaultdict(list)
    for row in iter_jsonl(run_dir / "losses.jsonl"):
        gen = row.get("generation")
        if not isinstance(gen, int):
            continue
        family = str(row.get("family") or row.get("family_signature") or "unknown")
        by_gen[gen].append(family)
    return family_rows(problem, "loss", by_gen)


def family_diversity_builder(run_dir: Path, problem: str) -> list[dict[str, Any]]:
    checkpoint = read_json(run_dir / "checkpoint.json")
    entries = candidate_meta_from_checkpoint(checkpoint)
    by_gen: dict[int, list[str]] = defaultdict(list)
    for row in entries.values():
        gen = row.get("generation")
        if not isinstance(gen, int):
            continue
        family = str(row.get("family_signature") or row.get("family") or "unknown")
        by_gen[gen].append(family)
    return family_rows(problem, "weighting", by_gen)


def family_rows(problem: str, phase: str, by_gen: Mapping[int, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gen in sorted(by_gen):
        vals = by_gen[gen]
        counts = Counter(vals)
        total = len(vals)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            entropy -= p * math.log(p, 2)
        rows.append(
            {
                "problem": problem,
                "phase": phase,
                "generation": gen,
                "candidate_count": total,
                "unique_families": len(counts),
                "top_family_share": max(counts.values()) / total if total else "",
                "family_entropy_bits": entropy,
                "top_families": "; ".join(f"{k}:{v}" for k, v in counts.most_common(4)),
            }
        )
    return rows


def best_loss_motifs() -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    md: list[str] = ["# Best discovered loss and weighting rules\n"]
    for problem, run_dir in LOSS_RUNS.items():
        best = read_json(run_dir / "best_loss.json")
        ir = best.get("ir") or {}
        operators = ir.get("operators_used") or []
        pseudocode = str(ir.get("pseudocode") or "").strip()
        code = str(ir.get("code") or "")
        motifs = []
        for token, label in [
            ("logsigmoid", "pairwise preference backbone"),
            ("softplus", "pairwise preference backbone"),
            ("advantage", "advantage/statistical normalization"),
            ("delta_rank", "rank statistics"),
            ("rank", "rank statistics"),
            ("std", "instance dispersion"),
            ("mad", "instance dispersion"),
            ("clamp", "bounded transform/clipping"),
            ("tanh", "bounded transform/clipping"),
            ("temperature", "temperature/calibration"),
            ("margin", "policy margin/hardness"),
            ("log_prob", "policy margin/hardness"),
        ]:
            hay = " ".join([pseudocode, code, " ".join(map(str, operators))]).lower()
            if token in hay and label not in motifs:
                motifs.append(label)
        rows.append(
            {
                "problem": problem,
                "loss_id": best.get("id") or best.get("loss_id") or "",
                "score": best.get("score", ""),
                "family_signature": best.get("family_signature")
                or best.get("family")
                or "|".join(f"{k}={v}" for k, v in (ir.get("hyperparams") or {}).items() if str(k).endswith("_family")),
                "operators": "; ".join(map(str, operators)),
                "interpretable_motifs": "; ".join(motifs),
            }
        )
        md.append(f"## {problem} loss\n")
        md.append(f"- score: `{best.get('score', '')}`")
        md.append(f"- family: `{best.get('family_signature') or best.get('family') or ''}`")
        md.append(f"- motifs: {', '.join(motifs) if motifs else 'inspect pseudocode'}")
        md.append("\n```text\n" + (pseudocode or "(no pseudocode)") + "\n```\n")

        wdir = WEIGHT_RUNS.get(problem)
        if wdir and (wdir / "best_builder.json").is_file():
            builder = read_json(wdir / "best_builder.json")
            bir = builder.get("ir") or {}
            md.append(f"## {problem} weighting\n")
            md.append(f"- score: `{builder.get('score', '')}`")
            md.append(f"- family: `{builder.get('family_signature') or builder.get('family') or ''}`")
            md.append("\n```text\n" + str(bir.get("pseudocode") or bir.get("intuition") or "(no pseudocode)") + "\n```\n")
    return rows, "\n".join(md)


def evaluated_loss_frequencies() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    operator_rows: list[dict[str, Any]] = []
    family_rows_out: list[dict[str, Any]] = []
    for problem, run_dir in LOSS_RUNS.items():
        checkpoint = read_json(run_dir / "checkpoint.json")
        meta = candidate_meta_from_loss_run(run_dir)
        evaluated: set[str] = set()
        for key, history in (checkpoint.get("pair_score_history_map") or {}).items():
            if not isinstance(history, list):
                continue
            cand_id = str(key).split("::", 1)[-1]
            if any(isinstance(rec, Mapping) and finite(rec.get("score")) for rec in history):
                evaluated.add(cand_id)

        op_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        for cand_id in evaluated:
            row = meta.get(cand_id, {})
            ir = row.get("ir") if isinstance(row, Mapping) else {}
            if isinstance(ir, Mapping):
                for op in ir.get("operators_used") or []:
                    op_counts[str(op)] += 1
            family = str(row.get("family_signature") or row.get("family") or "unknown")
            family_counts[family] += 1

        denom = max(1, len(evaluated))
        for op, count in op_counts.most_common():
            operator_rows.append(
                {
                    "problem": problem,
                    "operator": op,
                    "evaluated_loss_count": len(evaluated),
                    "candidate_count_with_operator": count,
                    "share": count / denom,
                }
            )
        for family, count in family_counts.most_common():
            family_rows_out.append(
                {
                    "problem": problem,
                    "family_signature": family,
                    "evaluated_loss_count": len(evaluated),
                    "candidate_count_in_family": count,
                    "share": count / denom,
                }
            )
    return operator_rows, family_rows_out


def selected_performance_summary() -> list[dict[str, Any]]:
    src = ROOT / "完整结果.csv"
    rows: list[dict[str, Any]] = []
    if not src.is_file():
        return rows
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            solver = row.get("Solver", "")
            if solver not in {"RL, aug_max", "PO, aug_max", "SLL, aug_max", "BOPO, aug_max", "Loss, aug_max", "Loss-weighting, aug_max"}:
                continue
            out: dict[str, Any] = {"Solver": solver}
            for key in row:
                if "Gap" in key or "Gap(%)" in key:
                    out[key] = row.get(key, "")
            rows.append(out)
    return rows


def write_clean_replay_summary() -> None:
    src = ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/04_replay_diagnosis_summary_table.csv"
    if not src.is_file():
        return
    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return
        for raw in reader:
            if len(raw) < 7:
                continue
            rows.append(
                {
                    "problem": raw[0],
                    "transfer": raw[1],
                    "method": raw[2],
                    "consistency": raw[3],
                    "effective_pair_ratio": raw[4],
                    "top10_weight_mass": raw[5],
                    "weight_gini": raw[6],
                }
            )
    write_csv(
        OUT / "5.4_weighting_transfer/04_replay_diagnosis_summary_clean.csv",
        rows,
        ["problem", "transfer", "method", "consistency", "effective_pair_ratio", "top10_weight_mass", "weight_gini"],
    )


def write_notes() -> None:
    (OUT / "5.1_objective_discovery/notes.md").write_text(
        """# 5.1 Objective Discovery as Evaluated Design Instinct

Core sentence: LLM supplies objective-design instincts; semantic gates and training-based evaluation decide which instincts survive.

Use these materials:
- `search_trajectory.csv`: best-so-far curves for Stage I loss search and Stage II weighting search.
- `filtering_funnel.csv`: generated -> semantic/numeric gate -> high-fidelity -> elite survival.
- `family_diversity_by_generation.csv`: initial and later family diversity; supports the claim that search keeps multiple hypotheses alive.
- `best_candidate_milestones.csv`: concrete improvement events with candidate ids, families, and intuitions.
- `figures/01_search_trajectory*.png`, `figures/02_filtering_funnel.png`, `figures/03_stagewise_contribution.png`, `figures/05_tsp_family_diversity_timeline.png`.

Writing angle:
- Do not claim the LLM directly knows the optimal objective.
- Claim the LLM proposes broad objective hypotheses, and evaluation turns them into empirically selected design rules.
- Stage I and Stage II should be framed as complementary: comparison semantics first, pair allocation second.
""",
        encoding="utf-8",
    )

    (OUT / "5.2_surviving_design_patterns/notes.md").write_text(
        """# 5.2 What Design Patterns Survive the Search?

Core sentence: The discovered losses are empirical design hypotheses that survived semantic filtering and training-based selection.

Surviving motifs to discuss:
- Pairwise preference backbone is retained.
- Objective gap is often used as calibration/temperature rather than BOPO-style linear amplification.
- Rank, advantage, and instance-dispersion statistics recur, suggesting instance-internal normalization matters.
- Clipping, bounded transforms, and temperatures recur, suggesting stable gradients beat stronger raw supervision.
- Policy margin/hardness appears in the objective, meaning a pair's value depends on whether the current policy already separates it.

Use `02_all_methods_gap_matched_residual_surface.*` and `surface.csv` to argue that searched losses are not reducible to an objective-gap response. The residuals still depend on margin hardness, rank, or instance statistics.
""",
        encoding="utf-8",
    )

    (OUT / "5.3_loss_side_by_problem/problem_takeaways.md").write_text(
        """# 5.3 Loss-Side Insights by Problem

TSP: Large-gap pairs are not automatically more valuable; easy pairs should be mildly suppressed once the policy already separates them.

CVRP: Raw cost gaps are unstable across instances; rank/dispersion behaves more like scale-free calibration.

FFSP: Makespan gap works better as a temperature/calibration signal than as a direct pair multiplier.

JSSP: The complex task does not necessarily need a more complex discovered loss; global advantage or dispersion normalization can already stabilize comparison.

Recommended evidence:
- `per_problem_selected_claims.csv` for compact claim wording.
- `per_problem_loss_signature_summary.csv` for nearest baseline, residual features, and scale ratios.
- Per-problem PNG/PDF signatures for visual support.
""",
        encoding="utf-8",
    )

    (OUT / "5.4_weighting_transfer/notes.md").write_text(
        """# 5.4 What Does Weighting Learn, and Why Does It Transfer Poorly?

Core sentence: Weighting rules encode allocation heuristics over the candidate pool, whereas loss rules encode local comparison semantics.

TSP/CVRP: Weighting changes pair concentration. Under transfer, pair feature and weight distributions shift; the effective pair ratio drops.

FFSP: The failure mode is not concentration collapse. The clamp-window shifts: at the search scale the pre-clamp scores are mostly saturated, so weighting is nearly constant; at transfer scale more pairs enter the clamp window, which downweights high-gap/high-margin strong pairs.

JSSP: The shift is milder than TSP/CVRP but still allocation-dependent. Weighting is not universally bad; its gain depends on whether the new-scale candidate distribution matches the searched allocation rule.

Use:
- `04_replay_diagnosis_summary_clean.csv`
- `pair_feature_shift.csv`
- `weight_concentration.csv`
- `weight_informativeness_alignment.csv`
- `ffsp_weight_clamp_release.csv`
- `ffsp_rank_signal.csv`
""",
        encoding="utf-8",
    )

    (OUT / "5.5_implications_limitations/discussion_outline_materials.md").write_text(
        """# 5.5 Broader Implications and Limitations

Implications:
- For preference-based NCO, the bottleneck is not merely constructing more pairs; it is calibrating the preference gradient.
- For objective discovery, LLM-guided search is valuable because it proposes evaluable design hypotheses, not because the LLM directly outputs the final formula.
- The two-stage design has a useful division of labor: Stage I learns stable comparison semantics; Stage II learns locally useful but distribution-sensitive allocation.

Limitations:
- The analysis is interpretive and does not prove every term is individually necessary.
- Gap-matched residual and replay probes support mechanisms but are not causal ablations of each operator.
- Weighting has weaker cross-scale transfer; allocation rules depend on the candidate-pool distribution.
- Current evidence is strongest for the searched scales and transfer directions represented in the materials.
""",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    manifest: list[dict[str, str]] = []

    trajectory_rows: list[dict[str, Any]] = []
    milestone_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []

    for problem, run_dir in LOSS_RUNS.items():
        meta = candidate_meta_from_loss_run(run_dir)
        rows, milestones = best_series(run_dir, "loss", meta)
        for row in rows:
            row["problem"] = problem
        for row in milestones:
            row["problem"] = problem
        trajectory_rows.extend(rows)
        milestone_rows.extend(milestones)
        funnel_rows.append(gate_counts(run_dir, problem, "loss"))
        diversity_rows.extend(family_diversity_loss(run_dir, problem))
        copy_file(run_dir / "best_loss.json", f"5.2_surviving_design_patterns/raw/{problem.lower()}_best_loss.json", manifest, "Best discovered loss raw JSON.")
        copy_file(run_dir / "summary.json", f"5.1_objective_discovery/raw/{problem.lower()}_loss_summary.json", manifest, "Loss-search run summary.")

    for problem, run_dir in WEIGHT_RUNS.items():
        checkpoint = read_json(run_dir / "checkpoint.json")
        meta = candidate_meta_from_checkpoint(checkpoint)
        rows, milestones = best_series(run_dir, "weighting", meta)
        for row in rows:
            row["problem"] = problem
        for row in milestones:
            row["problem"] = problem
        trajectory_rows.extend(rows)
        milestone_rows.extend(milestones)
        funnel_rows.append(gate_counts(run_dir, problem, "weighting"))
        diversity_rows.extend(family_diversity_builder(run_dir, problem))
        copy_file(run_dir / "best_builder.json", f"5.4_weighting_transfer/raw/{problem.lower()}_best_builder.json", manifest, "Best weighting/builder raw JSON.")
        copy_file(run_dir / "summary.json", f"5.1_objective_discovery/raw/{problem.lower()}_weighting_summary.json", manifest, "Weighting-search run summary.")

    write_csv(
        OUT / "5.1_objective_discovery/search_trajectory.csv",
        trajectory_rows,
        [
            "problem",
            "phase",
            "generation",
            "evaluated_candidates",
            "generation_best_score",
            "generation_mean_score",
            "best_so_far_score",
            "best_candidate_id",
            "improved_best_so_far",
        ],
    )
    write_csv(
        OUT / "5.1_objective_discovery/best_candidate_milestones.csv",
        milestone_rows,
        ["problem", "phase", "generation", "candidate_id", "score", "family_signature", "op_type", "intuition"],
    )
    write_csv(
        OUT / "5.1_objective_discovery/filtering_funnel.csv",
        funnel_rows,
        [
            "problem",
            "phase",
            "generated_candidates",
            "static_or_compile_ok",
            "semantic_pair_ok",
            "numeric_gate_ok",
            "entered_high_fidelity",
            "top_elites",
            "top_rejection_or_stage_reason",
        ],
    )
    write_csv(
        OUT / "5.1_objective_discovery/family_diversity_by_generation.csv",
        diversity_rows,
        ["problem", "phase", "generation", "candidate_count", "unique_families", "top_family_share", "family_entropy_bits", "top_families"],
    )

    motifs, motif_md = best_loss_motifs()
    op_freq_rows, family_freq_rows = evaluated_loss_frequencies()
    write_csv(
        OUT / "5.2_surviving_design_patterns/best_loss_motifs.csv",
        motifs,
        ["problem", "loss_id", "score", "family_signature", "operators", "interpretable_motifs"],
    )
    write_csv(
        OUT / "5.2_surviving_design_patterns/evaluated_loss_operator_frequency.csv",
        op_freq_rows,
        ["problem", "operator", "evaluated_loss_count", "candidate_count_with_operator", "share"],
    )
    write_csv(
        OUT / "5.2_surviving_design_patterns/evaluated_loss_family_frequency.csv",
        family_freq_rows,
        ["problem", "family_signature", "evaluated_loss_count", "candidate_count_in_family", "share"],
    )
    (OUT / "5.2_surviving_design_patterns/best_loss_pseudocode.md").write_text(motif_md, encoding="utf-8")

    perf = selected_performance_summary()
    if perf:
        keys = ["Solver"]
        for row in perf:
            for key in row:
                if key not in keys:
                    keys.append(key)
        write_csv(OUT / "tables/performance_selected_gaps.csv", perf, keys)

    figure_copies = [
        (ROOT / "figures/objective_search/01_search_trajectory.png", "5.1_objective_discovery/figures/01_search_trajectory.png", "Stage I/II search trajectory."),
        (ROOT / "figures/objective_search/01_search_trajectory_nogate.png", "5.1_objective_discovery/figures/01_search_trajectory_nogate.png", "No-gate replay trajectory."),
        (ROOT / "figures/objective_search/02_filtering_funnel.png", "5.1_objective_discovery/figures/02_filtering_funnel.png", "Candidate filtering funnel."),
        (ROOT / "figures/objective_search/03_stagewise_contribution.png", "5.1_objective_discovery/figures/03_stagewise_contribution.png", "Stagewise contribution."),
        (ROOT / "figures/objective_search/05_tsp_family_diversity_timeline.png", "5.1_objective_discovery/figures/05_tsp_family_diversity_timeline.png", "Family diversity timeline."),
        (ROOT / "figures/loss_fine_grained_signature/core_02_06/02_all_methods_gap_matched_residual_surface.png", "5.2_surviving_design_patterns/figures/02_all_methods_gap_matched_residual_surface.png", "Gap-matched residual surface."),
        (ROOT / "figures/loss_fine_grained_signature/core_02_06/06_counterfactual_objective_scale_sensitivity.png", "5.2_surviving_design_patterns/figures/06_counterfactual_objective_scale_sensitivity.png", "Objective scale sensitivity."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/07_fine_grained_signature_summary.png", "5.2_surviving_design_patterns/figures/07_fine_grained_signature_summary.png", "Fine-grained signature summary."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/per_problem/tsp_per_problem_signature.png", "5.3_loss_side_by_problem/figures/tsp_per_problem_signature.png", "TSP loss-side signature."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/per_problem/cvrp_per_problem_signature.png", "5.3_loss_side_by_problem/figures/cvrp_per_problem_signature.png", "CVRP loss-side signature."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/per_problem/ffsp_per_problem_signature.png", "5.3_loss_side_by_problem/figures/ffsp_per_problem_signature.png", "FFSP loss-side signature."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/per_problem/jssp_per_problem_signature.png", "5.3_loss_side_by_problem/figures/jssp_per_problem_signature.png", "JSSP loss-side signature."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/01_pair_weight_distribution_shift.png", "5.4_weighting_transfer/figures/01_pair_weight_distribution_shift.png", "Pair/weight distribution shift."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/03_effective_signal_concentration.png", "5.4_weighting_transfer/figures/03_effective_signal_concentration.png", "Effective signal concentration."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/05_weight_informativeness_alignment.png", "5.4_weighting_transfer/figures/05_weight_informativeness_alignment.png", "Weight-informativeness alignment."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/ffsp_failure_probe/ffsp_weighting_transfer_failure_probe.png", "5.4_weighting_transfer/figures/ffsp_weighting_transfer_failure_probe.png", "FFSP clamp-window shift probe."),
    ]
    for src, dst, note in figure_copies:
        copy_file(src, dst, manifest, note)

    data_copies = [
        (ROOT / "完整结果.csv", "tables/final_results_full.csv", "Final result table currently open in IDE."),
        (ROOT / "all_problem_results_no_eam_max_aug.csv", "tables/all_problem_results_no_eam_max_aug.csv", "Comparable all-problem result source."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/06_fine_grained_signature_summary.csv", "5.2_surviving_design_patterns/fine_grained_signature_summary.csv", "Summary of gap-matched residual diagnostics."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/surface.csv", "5.2_surviving_design_patterns/surface.csv", "Gap-matched residual surface data."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/dependency.csv", "5.2_surviving_design_patterns/dependency.csv", "Residual feature-dependency data."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/instance.csv", "5.2_surviving_design_patterns/instance.csv", "Instance-adaptive normalization data."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/per_problem/per_problem_loss_signature_summary.csv", "5.3_loss_side_by_problem/per_problem_loss_signature_summary.csv", "Per-problem loss-side diagnostic summary."),
        (ROOT / "figures/loss_fine_grained_signature/20260501-final-all-methods/per_problem/per_problem_selected_claims.csv", "5.3_loss_side_by_problem/per_problem_selected_claims.csv", "Compact per-problem claims."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/04_replay_diagnosis_summary_table.csv", "5.4_weighting_transfer/04_replay_diagnosis_summary_table.csv", "Transfer replay diagnosis summary."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/pair_feature_shift.csv", "5.4_weighting_transfer/pair_feature_shift.csv", "Pair-feature distribution shift data."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/weight_concentration.csv", "5.4_weighting_transfer/weight_concentration.csv", "Weight concentration and effective pair ratio data."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/weight_informativeness_alignment.csv", "5.4_weighting_transfer/weight_informativeness_alignment.csv", "Weight informativeness alignment data."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/20260501-final-alignment/policy_consistency.csv", "5.4_weighting_transfer/policy_consistency.csv", "Policy preference consistency under transfer."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/ffsp_failure_probe/ffsp_weight_clamp_release.csv", "5.4_weighting_transfer/ffsp_weight_clamp_release.csv", "FFSP clamp-window shift data."),
        (ROOT / "figures/scale_transfer_replay_diagnosis/ffsp_failure_probe/ffsp_rank_signal.csv", "5.4_weighting_transfer/ffsp_rank_signal.csv", "FFSP rank signal data."),
    ]
    for src, dst, note in data_copies:
        copy_file(src, dst, manifest, note)
    write_clean_replay_summary()

    write_notes()

    claim_rows = [
        {
            "Section": "5.1",
            "Claim": "LLM proposes objective-design instincts; gates and training decide survival.",
            "Primary material": "5.1_objective_discovery/search_trajectory.csv; filtering_funnel.csv; figures/01_search_trajectory.png",
            "Use carefully": "Frame as evaluated hypothesis generation, not direct formula knowledge.",
        },
        {
            "Section": "5.1",
            "Claim": "Family diversity prevents early collapse.",
            "Primary material": "5.1_objective_discovery/family_diversity_by_generation.csv; figures/05_tsp_family_diversity_timeline.png",
            "Use carefully": "Discuss as search behavior evidence, not a formal guarantee.",
        },
        {
            "Section": "5.2",
            "Claim": "Searched losses cannot be reduced to objective-gap response alone.",
            "Primary material": "5.2_surviving_design_patterns/surface.csv; dependency.csv; figures/02_all_methods_gap_matched_residual_surface.png",
            "Use carefully": "Call these residual signatures, not proof of causal necessity.",
        },
        {
            "Section": "5.3",
            "Claim": "Each problem has a compact loss-side takeaway.",
            "Primary material": "5.3_loss_side_by_problem/per_problem_selected_claims.csv; problem_takeaways.md",
            "Use carefully": "Keep formulas short and make the section reader-insight oriented.",
        },
        {
            "Section": "5.4",
            "Claim": "Weighting transfers poorly because allocation rules depend on candidate-pool distributions.",
            "Primary material": "5.4_weighting_transfer/weight_concentration.csv; pair_feature_shift.csv; 04_replay_diagnosis_summary_table.csv",
            "Use carefully": "Separate TSP/CVRP concentration shift from FFSP clamp-window shift.",
        },
        {
            "Section": "5.5",
            "Claim": "Objective discovery is useful as evaluated design-hypothesis generation.",
            "Primary material": "5.5_implications_limitations/discussion_outline_materials.md",
            "Use carefully": "State limitations explicitly: interpretive, not term-by-term ablation.",
        },
    ]
    (OUT / "claim_evidence_map.md").write_text(
        "# Discussion claim-evidence map\n\n"
        + markdown_table(claim_rows, ["Section", "Claim", "Primary material", "Use carefully"])
        + "\n",
        encoding="utf-8",
    )

    write_csv(OUT / "source_manifest.csv", manifest, ["material_path", "source_path", "note"])
    (OUT / "README.md").write_text(
        """# Discussion Materials

This folder packages the data needed to write the proposed Discussion section.

Directory map:
- `5.1_objective_discovery/`: search trajectory, gates, family diversity, and milestone evidence for LLM-guided objective discovery.
- `5.2_surviving_design_patterns/`: best-loss motifs and gap-matched residual diagnostics.
- `5.3_loss_side_by_problem/`: per-problem loss-side takeaways and supporting signatures.
- `5.4_weighting_transfer/`: replay diagnostics for weighting behavior and transfer failure.
- `5.5_implications_limitations/`: high-level implications and limitation notes.
- `tables/`: final result tables and selected gap summaries.

Start with `claim_evidence_map.md`, then use each subfolder's `notes.md` or takeaway file when drafting the corresponding subsection.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
