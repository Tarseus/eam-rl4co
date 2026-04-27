from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_mplconfigdir = (Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


LOSS_RUNS = {
    "TSP100": REPO_ROOT / "runs/pref_loss_tsp100_discovery/20260317-131507",
}

BUILDER_RUNS = {
    "TSP100": REPO_ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757",
}

SELECTED_TASKS = ("TSP100",)
SEARCH_TRAJECTORY_MAX_GENERATION = 9
SEARCH_TRAJECTORY_ELITE_FRACTION = 0.25


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return dict(payload)


def _is_finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _aggregate_by_generation(rows: Sequence[Mapping[str, Any]]) -> Dict[int, List[float]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        gen = row.get("generation")
        score = row.get("score")
        if not isinstance(gen, int):
            continue
        if not _is_finite(score):
            continue
        grouped[int(gen)].append(float(score))
    return dict(grouped)


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    for name in ("summary.json", "checkpoint.json"):
        path = run_dir / name
        if path.is_file():
            return _load_json(path)
    raise FileNotFoundError(f"No summary-like JSON found in {run_dir}")


def _load_best_reference(run_dir: Path) -> Tuple[float | None, float | None]:
    candidates = ["best_loss.json", "best_pair.json", "best_builder.json"]
    for name in candidates:
        path = run_dir / name
        if not path.is_file():
            continue
        payload = _load_json(path)
        score = _maybe_float(payload.get("score"))
        ref = _maybe_float(payload.get("reference_score"))
        if ref is None:
            history = payload.get("score_history")
            if isinstance(history, list) and history:
                first = history[0]
                if isinstance(first, Mapping):
                    ref = _maybe_float(first.get("reference_score"))
        if ref is None:
            history = payload.get("best_pair_score_history")
            if isinstance(history, list) and history:
                first = history[0]
                if isinstance(first, Mapping):
                    ref = _maybe_float(first.get("reference_score"))
        if score is not None or ref is not None:
            return ref, score
    return None, None


def _load_checkpoint(run_dir: Path) -> Dict[str, Any]:
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint.json: {checkpoint_path}")
    payload = _load_json(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict checkpoint at {checkpoint_path}")
    return payload


def _candidate_meta_index(checkpoint: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}

    def _add(entry: Mapping[str, Any]) -> None:
        cand_id = entry.get("id") or entry.get("builder_id") or entry.get("g_id")
        if not cand_id:
            return
        cid = str(cand_id)
        if cid not in index:
            index[cid] = dict(entry)

    for key in ("hof_g", "resident_pop_g", "archive_g", "elites_g", "diverse_elites_g", "seen_g"):
        value = checkpoint.get(key)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, Mapping):
                    _add(entry)
        elif isinstance(value, dict):
            for entry in value.values():
                if isinstance(entry, Mapping):
                    _add(entry)
    return index


def _shorten_id(candidate_id: str, limit: int = 18) -> str:
    if len(candidate_id) <= limit:
        return candidate_id
    return candidate_id[: max(8, limit - 3)] + "..."


def _short_parent_list(parents: Any, limit: int = 3) -> str:
    if not isinstance(parents, list) or not parents:
        return "-"
    parent_ids = [str(p) for p in parents if p]
    if len(parent_ids) <= limit:
        return ", ".join(_shorten_id(pid, 16) for pid in parent_ids)
    kept = ", ".join(_shorten_id(pid, 16) for pid in parent_ids[:limit])
    return f"{kept}, +{len(parent_ids) - limit}"


def _format_milestone_label(meta: Mapping[str, Any] | None, generation: int, candidate_id: str) -> str:
    if not meta:
        return f"g{generation} {_shorten_id(candidate_id)}"

    origin = str(meta.get("origin") or "?")
    origin_base = str(meta.get("origin_base") or "?")
    op_type = str(meta.get("op_type") or "?")
    parents = _short_parent_list(meta.get("parents"))
    prompt = Path(str(meta.get("prompt_path") or "")).name or "-"
    llm_seed = meta.get("llm_seed")
    seed_text = f" seed={llm_seed}" if llm_seed is not None else ""

    lines = [
        f"g{generation} {_shorten_id(candidate_id)}",
        f"{op_type} / {origin_base}",
        f"parents: {parents}",
        f"{prompt}{seed_text}",
    ]
    return "\n".join(lines)


def _trace_lineage(meta_index: Mapping[str, Mapping[str, Any]], candidate_id: str) -> List[Dict[str, Any]]:
    lineage: List[Dict[str, Any]] = []
    seen: set[str] = set()
    current_id = candidate_id
    while current_id and current_id not in seen:
        seen.add(current_id)
        meta = meta_index.get(current_id)
        if meta is None:
            break
        lineage.append(dict(meta))
        parents = meta.get("parents")
        if not isinstance(parents, list) or not parents:
            break
        parent_ids = [str(p) for p in parents if p]
        first_parent = next((pid for pid in parent_ids if not pid.startswith("bootstrap_")), None)
        if first_parent is None:
            break
        current_id = first_parent
    lineage.reverse()
    return lineage


def _format_lineage_card(milestone: Mapping[str, Any], lineage: Sequence[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(
        f"final best: g{int(milestone['generation'])} {_shorten_id(str(milestone['candidate_id']))}"
    )
    lines.append(f"score: {float(milestone['score']):.6f}")
    lines.append("")

    if lineage:
        root = lineage[0]
        root_parents = root.get("parents")
        if isinstance(root_parents, list) and root_parents:
            lines.append("root parents: " + ", ".join(_shorten_id(str(p), 18) for p in root_parents))

        for meta in lineage:
            cid = _shorten_id(str(meta.get("id") or meta.get("builder_id") or meta.get("g_id") or "?"))
            generation = meta.get("generation")
            origin = str(meta.get("origin") or "?")
            origin_base = str(meta.get("origin_base") or "?")
            op_type = str(meta.get("op_type") or "?")
            prompt = Path(str(meta.get("prompt_path") or "")).name or "-"
            parents = meta.get("parents")
            if isinstance(parents, list) and parents:
                parent_text = ", ".join(_shorten_id(str(p), 16) for p in parents)
            else:
                parent_text = "-"
            lines.append(f"g{generation} {cid}: {op_type} / {origin_base}")
            lines.append(f"  parents: {parent_text}")
            lines.append(f"  prompt: {prompt}")
            if meta is not lineage[-1]:
                lines.append("")
    return "\n".join(lines)


def _summarize_representative_formula(meta: Mapping[str, Any] | None) -> str:
    if not isinstance(meta, Mapping):
        return "-"
    ir = meta.get("ir")
    if not isinstance(ir, Mapping):
        return "-"

    pseudocode = str(ir.get("pseudocode") or "")
    if pseudocode:
        lines = [line.strip() for line in pseudocode.splitlines() if line.strip()]
        if lines:
            text = "; ".join(lines[:2])
            return textwrap.shorten(" ".join(text.split()), width=42, placeholder="...")

    name = str(ir.get("name") or "")
    intuition = str(ir.get("intuition") or "")
    text = name.replace("_", " ").strip() or intuition.split(".")[0].strip() or "-"
    return textwrap.shorten(text, width=42, placeholder="...")


def _internal_parents(meta: Mapping[str, Any], meta_index: Mapping[str, Mapping[str, Any]]) -> List[str]:
    parents = meta.get("parents")
    if not isinstance(parents, list):
        return []
    return [str(p) for p in parents if p and str(p) in meta_index]


def _draw_ancestry_tree(
    ax: plt.Axes,
    meta_index: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
    x: float,
    y: float,
    visited: set[str] | None = None,
    is_final: bool = False,
) -> None:
    visited = visited or set()
    if candidate_id in visited:
        return
    visited.add(candidate_id)

    meta = meta_index.get(candidate_id)
    if meta is None:
        return

    generation = meta.get("generation")
    origin = str(meta.get("origin") or "?")
    origin_base = str(meta.get("origin_base") or "?")
    op_type = str(meta.get("op_type") or "?")
    short_id = _shorten_id(candidate_id, 18)

    node_y = y
    node_x = x
    node_color = "#d62728" if is_final else "#111111"
    node_size = 72 if is_final else 48
    ax.scatter([node_x], [node_y], s=node_size, color=node_color, zorder=4 if is_final else 3)
    label = f"{'FINAL BEST' if is_final else f'g{generation}'}\n{short_id}\n{op_type} / {origin_base}"
    ax.text(
        node_x + 0.045,
        node_y,
        label,
        ha="left",
        va="center",
        fontsize=7.6 if is_final else 7.0,
        bbox={
            "boxstyle": "round,pad=0.20",
            "fc": "white",
            "ec": "#bbbbbb" if not is_final else "#d62728",
            "alpha": 0.96,
        },
    )

    parents = _internal_parents(meta, meta_index)
    all_parents = [str(p) for p in meta.get("parents") or [] if p]
    if len(parents) == 1:
        parent_id = parents[0]
        next_y = y - 1.0
        ax.annotate(
            "",
            xy=(x, next_y + 0.18),
            xytext=(x, node_y - 0.18),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#666666"},
        )
        _draw_ancestry_tree(ax, meta_index, parent_id, x, next_y, visited, False)
    elif len(parents) == 2:
        offsets = (-0.22, 0.22)
        next_y = y - 1.0
        for child_x, parent_id in zip((x + offsets[0], x + offsets[1]), parents):
            ax.annotate(
                "",
                xy=(child_x, next_y + 0.18),
                xytext=(x, node_y - 0.18),
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#666666"},
            )
            _draw_ancestry_tree(ax, meta_index, parent_id, child_x, next_y, visited, False)
    elif all_parents:
        bundle_y = y - 1.0
        bundle_text = "parents: " + ", ".join(_shorten_id(p, 16) for p in all_parents)
        ax.annotate(
            "",
            xy=(x, bundle_y + 0.18),
            xytext=(x, node_y - 0.18),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#666666"},
        )
        ax.text(
            x,
            bundle_y,
            bundle_text,
            ha="center",
            va="center",
            fontsize=7.0,
            bbox={"boxstyle": "round,pad=0.18", "fc": "#f5f5f5", "ec": "#cccccc", "alpha": 0.98},
        )


def _pair_history_best_series(
    run_dir: Path,
    meta_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    checkpoint = _load_checkpoint(run_dir)
    entries: List[Dict[str, Any]] = []
    for key, hist_list in (checkpoint.get("pair_score_history_map") or {}).items():
        if not isinstance(hist_list, list):
            continue
        cand_id = str(key).split("::", 1)[-1]
        meta = meta_lookup.get(cand_id) if meta_lookup is not None else None
        for rec in hist_list:
            if not isinstance(rec, Mapping):
                continue
            gen = rec.get("generation")
            score = rec.get("score")
            if not isinstance(gen, int) or not _is_finite(score):
                continue
            entries.append(
                {
                    "generation": int(gen),
                    "score": float(score),
                    "candidate_id": cand_id,
                    "meta": meta,
                }
            )

    if not entries:
        return {"generations": [], "best_so_far": [], "cumulative_elite_mean": [], "milestones": [], "best_records": []}

    by_gen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in entries:
        by_gen[int(rec["generation"])].append(rec)

    generations = sorted(by_gen)
    gen_to_scores = _aggregate_by_generation(entries)
    best_per_gen: Dict[int, Dict[str, Any]] = {}
    for gen in generations:
        best_per_gen[gen] = min(by_gen[gen], key=lambda r: (float(r["score"]), r["candidate_id"]))

    best_so_far: List[float] = []
    cumulative_elite_mean: List[float] = []
    milestones: List[Dict[str, Any]] = []
    best_records: List[Dict[str, Any]] = []
    current_best = None
    cumulative_scores: List[float] = []
    for gen in generations:
        rec = best_per_gen[gen]
        score = float(rec["score"])
        scores = sorted(gen_to_scores[gen])
        cumulative_scores.extend(scores)
        cumulative_sorted = sorted(cumulative_scores)
        elite_count = max(1, int(math.ceil(len(cumulative_sorted) * SEARCH_TRAJECTORY_ELITE_FRACTION)))
        cumulative_elite_mean.append(mean(cumulative_sorted[:elite_count]))
        if current_best is None or score < current_best:
            current_best = score
            milestones.append(
                {
                    "generation": gen,
                    "score": score,
                    "candidate_id": rec["candidate_id"],
                    "meta": rec["meta"],
                }
            )
        best_so_far.append(float(current_best))
        best_records.append(
            {
                "generation": gen,
                "score": score,
                "candidate_id": rec["candidate_id"],
                "meta": rec["meta"],
            }
        )

    return {
        "generations": generations,
        "best_so_far": best_so_far,
        "cumulative_elite_mean": cumulative_elite_mean,
        "milestones": milestones,
        "best_records": best_records,
    }


def _loss_best_so_far_series(run_dir: Path) -> Dict[str, Any]:
    meta_lookup: Dict[str, Dict[str, Any]] = {}
    losses_path = run_dir / "losses.jsonl"
    if losses_path.is_file():
        for row in _iter_jsonl(losses_path):
            loss_id = row.get("id")
            if loss_id is not None and str(loss_id) not in meta_lookup:
                meta_lookup[str(loss_id)] = dict(row)
    return _pair_history_best_series(run_dir, meta_lookup)


def _weighting_best_so_far_series(run_dir: Path) -> Dict[str, Any]:
    checkpoint = _load_checkpoint(run_dir)
    meta_lookup = _candidate_meta_index(checkpoint)
    return _pair_history_best_series(run_dir, meta_lookup)


def _improvement_points(
    series: Mapping[str, Any],
    baseline_score: float,
    phase: str,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    prev_value: float | None = None
    for gen, score, record in zip(series["generations"], series["best_so_far"], series["best_records"]):
        value = float(baseline_score) - float(score)
        if prev_value is None or value > prev_value + 1e-12:
            points.append(
                {
                    "generation": int(gen),
                    "value": value,
                    "phase": phase,
                    "meta": record.get("meta"),
                }
            )
            prev_value = value
    return points


def _improvement_formulas(
    series: Mapping[str, Any],
    phase_label: str,
    generations: Sequence[int],
    x_positions: Sequence[int],
) -> List[Dict[str, Any]]:
    x_lookup = {int(g): float(x) for g, x in zip(generations, x_positions)}
    formulas: List[Dict[str, Any]] = []
    prev_score: float | None = None
    prev_gen: int | None = None
    for gen, score in zip(series["generations"], series["best_so_far"]):
        gen = int(gen)
        score = float(score)
        if prev_score is None:
            prev_score = score
            prev_gen = gen
            continue
        if score < prev_score - 1e-12:
            x = x_lookup.get(gen)
            if x is not None:
                delta = prev_score - score
                formulas.append(
                    {
                        "generation": gen,
                        "x": x,
                        "text": rf"$\Delta_{{{phase_label},{gen}}}={delta:.5f}$",
                    }
                )
            prev_score = score
            prev_gen = gen
    return formulas


def _load_funnel_counts(run_dir: Path) -> Dict[str, int]:
    path = run_dir / "gate_reports.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gate_reports.jsonl: {path}")

    rows = list(_iter_jsonl(path))
    total = len(rows)
    compile_ok = sum(1 for r in rows if bool(r.get("g_compile_ok")) and bool(r.get("f_compile_ok")))
    pair_gate_ok = sum(
        1
        for r in rows
        if bool(r.get("builder_gate_ok")) and bool(r.get("joint_gate_ok")) and bool(r.get("co_ok"))
    )
    hf_ok = sum(
        1
        for r in rows
        if str(r.get("stage_final") or "") == "high_fidelity"
        or str(r.get("stage") or "") == "high_fidelity"
        or bool(r.get("high_fidelity_on"))
    )
    elite = sum(1 for r in rows if bool(r.get("better_than_incumbent")))
    return {
        "generated": total,
        "compile_ok": compile_ok,
        "pair_gate_ok": pair_gate_ok,
        "hf_ok": hf_ok,
        "elite": elite,
    }


def _load_diagnostic_points(run_dir: Path) -> List[Dict[str, float | str]]:
    path = run_dir / "gate_reports.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gate_reports.jsonl: {path}")

    points: List[Dict[str, float | str]] = []
    for row in _iter_jsonl(path):
        x = _maybe_float(row.get("co_sensitivity_visible_rel_delta"))
        y = None
        joint = row.get("joint_gate_trace")
        if isinstance(joint, Mapping):
            observed = joint.get("observed")
            if isinstance(observed, Mapping):
                y = _maybe_float(observed.get("effective_grad_ratio"))
        if x is None or y is None:
            continue
        points.append(
            {
                "x": x,
                "y": y,
                "pair_reason": str(row.get("pair_reason") or ""),
                "stage": str(row.get("stage") or ""),
            }
        )
    return points


def _plot_search_trajectory(outdir: Path) -> Path:
    fig, ax_gain = plt.subplots(figsize=(10.6, 5.8))
    title = "TSP100"
    loss_run_dir = LOSS_RUNS[title]
    weight_run_dir = BUILDER_RUNS[title]

    loss_series = _loss_best_so_far_series(loss_run_dir)
    weight_series = _weighting_best_so_far_series(weight_run_dir)

    loss_gens = [g for g in loss_series["generations"] if g <= SEARCH_TRAJECTORY_MAX_GENERATION]
    weight_gens = [g for g in weight_series["generations"] if g <= SEARCH_TRAJECTORY_MAX_GENERATION]

    display_loss_gens = list(dict.fromkeys(loss_gens[:6] + loss_gens[-1:]))
    display_weight_gens = list(dict.fromkeys(weight_gens[:6] + weight_gens[-1:]))

    if display_loss_gens and display_weight_gens:
        loss_best_lookup = dict(zip(loss_series["generations"], loss_series["best_so_far"]))
        weight_best_lookup = dict(zip(weight_series["generations"], weight_series["best_so_far"]))
        loss_x = list(range(min(6, len(display_loss_gens))))
        if len(display_loss_gens) > 6:
            loss_x.append(loss_x[-1] + 2)
        weight_start_x = (loss_x[-1] + 1) if loss_x else 0
        weight_x = list(range(weight_start_x, weight_start_x + min(6, len(display_weight_gens))))
        if len(display_weight_gens) > 6:
            weight_x.append(weight_x[-1] + 2)
        x_positions = loss_x + weight_x

        loss_y = [-float(loss_best_lookup[g]) for g in display_loss_gens]
        if loss_y:
            loss_y[0] = 0.0
        loss_end_gain = float(loss_y[-1]) if loss_y else 0.0
        weight_start_gain = -float(weight_best_lookup[display_weight_gens[0]])
        weight_y = [loss_end_gain + (-float(weight_best_lookup[g]) - weight_start_gain) for g in display_weight_gens]
        y_values = loss_y + weight_y

        ax_gain.plot(x_positions, y_values, color="#111111", lw=2.5, marker="o", ms=4.5)
        ax_gain.fill_between(loss_x, loss_y, color="#111111", alpha=0.08)
        ax_gain.fill_between(weight_x, weight_y, color="#111111", alpha=0.08)

        ax_gain.set_title("TSP100 loss then weighting")
        ax_gain.set_ylabel("score gain")
        ax_gain.grid(True, alpha=0.3)
        ax_gain.axhline(0.0, color="#888888", lw=1.0, ls=":")

        xtick_positions = list(loss_x[:6])
        xtick_labels = [f"L{g}" for g in display_loss_gens[:6]]
        if len(display_loss_gens) > 6:
            xtick_positions += [loss_x[5] + 1, loss_x[-1]]
            xtick_labels += ["...", f"L{display_loss_gens[-1]}"]
        else:
            xtick_positions += loss_x[6:]
            xtick_labels += [f"L{g}" for g in display_loss_gens[6:]]
        xtick_positions += list(weight_x[:6])
        xtick_labels += [f"W{g}" for g in display_weight_gens[:6]]
        if len(display_weight_gens) > 6:
            xtick_positions += [weight_x[5] + 1, weight_x[-1]]
            xtick_labels += ["...", f"W{display_weight_gens[-1]}"]
        else:
            xtick_positions += weight_x[6:]
            xtick_labels += [f"W{g}" for g in display_weight_gens[6:]]
        ax_gain.set_xticks(xtick_positions)
        ax_gain.set_xticklabels(xtick_labels, fontsize=7.4)
        ax_gain.tick_params(axis="x", pad=10)
        phase_boundary_x = (loss_x[-1] + 0.5) if loss_x and weight_x else 0.0
        ax_gain.axvline(phase_boundary_x, color="#bdbdbd", lw=1.0, ls="--")
        ax_gain.text(
            phase_boundary_x + 0.25,
            0.96,
            "weighting",
            transform=ax_gain.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8.0,
            color="#666666",
        )

        all_values = loss_y + weight_y
        y_min = min(all_values)
        y_max = max(all_values)
        y_pad = max(0.001, (y_max - y_min) * 0.12)
        ax_gain.set_ylim(bottom=max(-0.001, y_min - y_pad))
        ax_gain.set_xlim(-0.4, weight_x[-1] + 0.8)

        loss_formulas = _improvement_formulas(loss_series, "L", display_loss_gens, loss_x)
        weight_formulas = _improvement_formulas(weight_series, "W", display_weight_gens, weight_x)
        for item in loss_formulas:
            ax_gain.text(
                item["x"],
                -0.28,
                item["text"],
                transform=ax_gain.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.1,
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#cfcfcf", "alpha": 0.97},
            )
        for idx, item in enumerate(weight_formulas):
            y_pos = -0.40 if idx % 2 == 0 else -0.49
            ax_gain.text(
                item["x"],
                y_pos,
                item["text"],
                transform=ax_gain.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.1,
                clip_on=False,
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#cfcfcf", "alpha": 0.97},
            )
        ax_gain.text(
            0.02,
            0.92,
            f"L6-L8 omitted; W6-W8 omitted. loss baseline = g0; weighting anchored at loss endpoint ({loss_end_gain:.6f}).  "
            "Formula labels show Δ = previous best - new best.",
            transform=ax_gain.transAxes,
            fontsize=8.0,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#dddddd", "alpha": 0.95},
        )
    else:
        ax_gain.set_title(title)
        ax_gain.text(0.5, 0.5, "no data", ha="center", va="center")

    fig.suptitle("Search trajectory", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.18, 1, 0.98))
    path = outdir / "01_search_trajectory.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_funnel(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.9, 4.4))
    title = "TSP100"
    counts = _load_funnel_counts(BUILDER_RUNS[title])
    labels = ["generated", "compile ok", "pair gate ok", "HF eval", "elite"]
    values = [counts["generated"], counts["compile_ok"], counts["pair_gate_ok"], counts["hf_ok"], counts["elite"]]
    maxv = max(values) if values else 1
    bars = ax.bar(labels, values, color=["#6c757d", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, maxv * 1.15 + 1e-9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(maxv * 0.03, 0.25), str(val), ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Candidate filtering funnel", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = outdir / "02_filtering_funnel.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stagewise_bars(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    task_rows = []
    for task in SELECTED_TASKS:
        ref, _ = _load_best_reference(LOSS_RUNS[task])
        loss_summary = _load_summary(LOSS_RUNS[task])
        builder_summary = _load_summary(BUILDER_RUNS[task])
        stage1 = _maybe_float(loss_summary.get("best_score"))
        stage2 = _maybe_float(builder_summary.get("best_score"))
        if ref is None or stage1 is None or stage2 is None:
            continue
        task_rows.append(
            {
                "task": task,
                "reference": 0.0,
                "stage1": float(ref) - float(stage1),
                "stage2": float(ref) - float(stage2),
                "raw_ref": float(ref),
                "raw_stage1": float(stage1),
                "raw_stage2": float(stage2),
            }
        )

    x = list(range(len(task_rows)))
    width = 0.24
    colors = {"reference": "#6c757d", "stage1": "#1f77b4", "stage2": "#2ca02c"}
    for idx, key, label in [
        (0, "reference", "Reference"),
        (1, "stage1", "Stage I"),
        (2, "stage2", "Stage I+II"),
    ]:
        vals = [row[key] for row in task_rows]
        offsets = [i + (idx - 1) * width for i in x]
        bars = ax.bar(offsets, vals, width=width, label=label, color=colors[key])
        for bar, row in zip(bars, task_rows):
            if key == "reference":
                text = "0"
            else:
                raw = row["raw_stage1"] if key == "stage1" else row["raw_stage2"]
                text = f"{raw:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, text, ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([row["task"] for row in task_rows])
    ax.set_ylabel("improvement over reference score\n(higher is better)")
    ax.set_title("Stage-wise contribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    fig.tight_layout()
    path = outdir / "03_stagewise_contribution.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_diagnostic(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.9, 4.4))
    title, run_dir = ("TSP100 builder", BUILDER_RUNS["TSP100"])
    palette = {
        "cheap_gate_failed": "#d62728",
        "co_gate_failed": "#1f77b4",
        "pref_semantic_failed": "#ff7f0e",
        "": "#7f7f7f",
    }

    pts = _load_diagnostic_points(run_dir)
    if not pts:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
    else:
        for reason in sorted({str(p["pair_reason"]) for p in pts}):
            xs = [float(p["x"]) for p in pts if p["pair_reason"] == reason]
            ys = [float(p["y"]) for p in pts if p["pair_reason"] == reason]
            ax.scatter(xs, ys, s=18, alpha=0.72, label=reason or "other", color=palette.get(reason, "#7f7f7f"))
        ax.set_title(title)
        ax.set_xlabel("co-sensitivity rel. delta")
        ax.set_ylabel("effective grad ratio")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Objective sensitivity vs gradient response", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = outdir / "04_diagnostic_scatter.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot core objective-search figures from existing runs.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "figures/objective_search"), type=str)
    args = parser.parse_args()

    outdir = Path(args.output_dir).resolve()
    _ensure_dir(outdir)

    manifest: Dict[str, Any] = {"output_dir": str(outdir), "figures": []}
    for maker in (
        _plot_search_trajectory,
        _plot_funnel,
        _plot_stagewise_bars,
        _plot_diagnostic,
    ):
        path = maker(outdir)
        manifest["figures"].append(str(path))

    manifest_path = outdir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
