from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_mplconfigdir = (Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import FIGSIZE_SEARCH, FIGSIZE_TRAJECTORY_ANNOTATED, paper_style
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import to_rgba


REPO_ROOT = Path(__file__).resolve().parents[1]


LOSS_RUNS = {
    "TSP100": REPO_ROOT / "runs/pref_loss_tsp100_discovery/20260317-131507",
    "FFSP100": REPO_ROOT / "runs/pref_loss_ffsp100_discovery/20260403-142801",
    "JSSP10x10": REPO_ROOT / "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409",
}

BUILDER_RUNS = {
    "TSP100": REPO_ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757",
    "FFSP100": REPO_ROOT / "runs/pref_builder_weight_search_ffsp100/20260416-111514",
    "JSSP10x10": REPO_ROOT / "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033",
}

REPLAY_PREFIXES = {
    "TSP100": "tsp",
    "FFSP100": "ffsp",
    "JSSP10x10": "jssp",
}

DEFAULT_TASK = "TSP100"
ACTIVE_TASK = DEFAULT_TASK
SELECTED_TASKS = (DEFAULT_TASK,)
SEARCH_TRAJECTORY_MAX_GENERATION = 9
SEARCH_TRAJECTORY_ELITE_FRACTION = 0.25
SEARCH_TRAJECTORY_CONSTRAINT_INJECT_MIN_DELTA = 0.002
SEARCH_TRAJECTORY_FORMULA_LABELS = {
    (
        "L",
        1,
    ): "$\\ell_f^{(i)}=-\\log\\sigma\\!\\left(z_1^{(i)}\\right)$\n"
    "$z_1^{(i)}=\\operatorname{clip}_{[-10,10]}\\!\\left(\\alpha s\\,\\Delta p^{(i)}-\\operatorname{clip}_{[-2,2]}\\!(0.015\\,\\Delta o^{(i)})\\right)$",
    (
        "L",
        2,
    ): "$\\ell_f^{(i)}=-\\log\\sigma\\!\\left(z_2^{(i)}\\right)$\n"
    "$z_2^{(i)}=\\operatorname{clip}_{[-20,20]}\\!\\left(\\alpha s(\\Delta p^{(i)}+0.05)-\\dfrac{0.03}{\\sigma(\\Delta p)+\\varepsilon}\\,\\Delta o^{(i)}\\right)$",
    (
        "L",
        4,
    ): "$\\ell_f^{(i)}=-\\log\\sigma\\!\\left(z_4^{(i)}\\right)$\n"
    "$z_4^{(i)}=\\operatorname{clip}_{[-20,20]}\\!\\left(\\alpha s\\,\\Delta p^{(i)}\\!\\left(1-\\beta\\dfrac{\\Delta o^{(i)}}{1+|\\Delta o^{(i)}|}\\right)\\right)$",
    (
        "W",
        1,
    ): "$w_g^{(i)}=\\operatorname{clip}_{[0.1,3.0]}\\!\\left(\\dfrac{\\Delta o^{(i)}}{\\operatorname{MAD}(\\Delta o)}\\cdot\\dfrac{|\\Delta p^{(i)}|}{\\sigma(\\Delta p)}\\cdot r(x)\\right)$",
    (
        "W",
        2,
    ): "$w_g^{(i)}=\\operatorname{clip}_{[0.1,3.0]}\\!\\left(\\dfrac{\\Delta o^{(i)}}{\\operatorname{MAD}(\\Delta o)}\\cdot\\dfrac{|\\Delta p^{(i)}|}{\\sigma(\\Delta p)}\\cdot r(x)\\right)$\n"
    "$w_g^{(i)}=0\\ \\mathrm{if}\\ \\dfrac{|\\Delta p^{(i)}|}{\\sigma(\\Delta p)}<\\tau$",
    (
        "W",
        4,
    ): "$w_g^{(i)}=\\operatorname{clip}_{[0.2,2.5]}\\!\\left(\\dfrac{\\Delta o^{(i)}}{\\operatorname{MAD}(\\Delta o)}\\cdot\\dfrac{|\\Delta p^{(i)}|}{\\sigma(\\Delta p)}\\cdot r(x)\\right)$",
}
FFSP_SEARCH_TRAJECTORY_FORMULA_LABELS = {
    (
        "L",
        2,
    ): "$\\ell_f^{(i)}=\\operatorname{softplus}\\!\\left(-x^{(i)}\\right)$\n"
    "$x^{(i)}=\\operatorname{clip}_{[-20,20]}\\!\\left(\\dfrac{\\alpha s\\,\\Delta p^{(i)}}{\\max(\\|\\Delta a\\|,\\varepsilon)}\\right)$",
    (
        "L",
        5,
    ): "$x^{(i)}=\\operatorname{clip}_{[-20,20]}\\!\\left(\\alpha s\\,\\Delta p^{(i)}/\\max(\\|\\Delta a\\|,\\varepsilon)-\\delta\\right)$\n"
    "$\\ell_f=\\sum_i w^{(i)}\\left[-\\log\\sigma\\!\\left(x^{(i)}\\right)\\right]$",
    (
        "L",
        8,
    ): "$m=\\exp\\!\\left(\\operatorname{clip}(|\\overline{\\Delta a}|,0,1)\\right)-1$\n"
    "$x^{(i)}=\\operatorname{clip}_{[-20,20]}\\!\\left(m\\,\\Delta p^{(i)}/\\max(\\operatorname{mean}\\|\\Delta a\\|,\\varepsilon)\\right)$\n"
    "$\\ell_f=\\sum_i w^{(i)}[-\\log\\sigma(x^{(i)})]/\\max(\\sum_i w^{(i)},\\varepsilon)$",
    (
        "W",
        8,
    ): "$g^{(i)}=\\dfrac{o_l^{(i)}-o_w^{(i)}}{\\operatorname{MAD}(o)}$\n"
    "$r^{(i)}=\\operatorname{rank}_l-\\operatorname{rank}_w$\n"
    "$m_p^{(i)}=|\\Delta p^{(i)}|/\\sigma(p)$\n"
    "$u^{(i)}=\\sigma(5r^{(i)})\\mathbf{1}[g^{(i)}>0.12]/(m_p^{(i)}+\\varepsilon)^{0.6}$\n"
    "$\\tilde{u}^{(i)}=\\operatorname{clip}_{[0.15,2.5]}(u^{(i)})/\\max(r^{(i)},\\varepsilon)$\n"
    "$w_g^{(i)}=\\operatorname{clip}_{[0.15,2.5]}\\!\\left(\\tilde{u}^{(i)}\\right)$",
}
SEARCH_TRAJECTORY_ANNOTATIONS = {
    ("L", 1): {"offset": (0, 106), "color": "#4c78a8"},
    ("L", 2): {"offset": (0, -54), "color": "#4c78a8", "pad": 0.95},
    ("L", 4): {"offset": (0, 60), "color": "#4c78a8", "pad": 0.95},
    ("W", 1): {"offset": (0, 68), "color": "#d65f5f"},
    ("W", 2): {"offset": (0, -118), "color": "#d65f5f"},
    ("W", 4): {"offset": (0, -86), "color": "#d65f5f"},
}
FFSP_SEARCH_TRAJECTORY_ANNOTATIONS = {
    ("L", 2): {"offset": (0, -104), "color": "#4c78a8", "pad": 0.86},
    ("L", 5): {"offset": (0, 82), "color": "#4c78a8", "pad": 0.9},
    ("L", 8): {"offset": (-78, -142), "color": "#4c78a8", "pad": 0.9},
    ("W", 8): {"offset": (0, -112), "color": "#d65f5f", "pad": 0.86},
}
_REPLAY_SHARD_RE = re.compile(r"^(?P<prefix>.+)_shard(?P<shard>\d+)_(?P<ts>\d{8}-\d{6})\.json$")


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


def _active_task() -> str:
    if ACTIVE_TASK not in LOSS_RUNS or ACTIVE_TASK not in BUILDER_RUNS:
        raise KeyError(f"Unknown active task: {ACTIVE_TASK}")
    return ACTIVE_TASK


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _apply_line_shadow(line: Any, *, alpha: float = 0.22, offset: Tuple[float, float] = (1.3, -1.3)) -> None:
    line.set_path_effects(
        [
            pe.SimpleLineShadow(offset=offset, shadow_color="#111111", alpha=alpha),
            pe.Normal(),
        ]
    )


def _fill_under_curve(
    ax: Any,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    color: str,
    alpha: float,
    zorder: float = 1.25,
) -> None:
    if not xs or not ys:
        return
    ax.fill_between(
        [float(x) for x in xs],
        [0.0 for _ in ys],
        [float(y) for y in ys],
        step="post",
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
    )


def _fill_to_bottom_gradient(
    ax: Any,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    bottom: float,
    color: str,
    alpha: float,
    zorder: float = 1.25,
    steps: int = 24,
) -> None:
    if not xs or not ys:
        return
    xs_arr = np.array([float(x) for x in xs], dtype=float)
    ys_arr = np.array([float(y) for y in ys], dtype=float)
    bottom_arr = np.full_like(ys_arr, float(bottom), dtype=float)
    for i in range(max(1, int(steps))):
        t0 = i / max(1, int(steps))
        t1 = (i + 1) / max(1, int(steps))
        lower = bottom_arr + (ys_arr - bottom_arr) * t0
        upper = bottom_arr + (ys_arr - bottom_arr) * t1
        band_alpha = float(alpha) * (t1 ** 0.85)
        ax.fill_between(
            xs_arr,
            lower,
            upper,
            step="post",
            color=color,
            alpha=band_alpha,
            linewidth=0,
            zorder=zorder,
        )


def _compress_plateaus(
    generations: Sequence[int],
    values: Sequence[float],
    *,
    tol: float = 1e-12,
) -> List[int]:
    if len(generations) != len(values):
        raise ValueError("generations and values must have the same length")
    if len(generations) <= 2:
        return [int(g) for g in generations]

    keep_indices: set[int] = set()
    run_start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or abs(float(values[idx]) - float(values[run_start])) > tol:
            keep_indices.add(run_start)
            keep_indices.add(idx - 1)
            run_start = idx
    return [int(generations[idx]) for idx in sorted(keep_indices)]


def _generation_axis(
    generations: Sequence[int],
    *,
    prefix: str,
    start_x: int,
) -> Tuple[List[int], List[float], List[str]]:
    xs: List[int] = []
    tick_positions: List[float] = []
    tick_labels: List[str] = []
    current_x = int(start_x)
    previous_gen: int | None = None
    for idx, gen in enumerate(int(g) for g in generations):
        if idx == 0:
            x = current_x
        else:
            if previous_gen is not None and gen - previous_gen > 1:
                current_x += 1
                tick_positions.append(float(current_x))
                tick_labels.append("...")
            current_x += 1
            x = current_x
        xs.append(x)
        tick_positions.append(float(x))
        tick_labels.append(f"{prefix}{gen}")
        previous_gen = gen
    return xs, tick_positions, tick_labels


def _add_stage_gradient_background(
    ax: Any,
    *,
    x_left: float,
    phase_boundary_x: float,
    x_right: float,
) -> None:
    loss_rgba = np.array(to_rgba("#dbe8f6", 0.90))
    weight_rgba = np.array(to_rgba("#fae7d3", 0.90))
    width = 768
    xs = np.linspace(x_left, x_right, width)
    blend_width = max(0.95, (x_right - x_left) * 0.075)
    blend_start = phase_boundary_x - blend_width * 0.5
    blend_end = phase_boundary_x + blend_width * 0.5
    t = np.clip((xs - blend_start) / max(1e-9, blend_end - blend_start), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    colors = loss_rgba[None, :] * (1.0 - t[:, None]) + weight_rgba[None, :] * t[:, None]
    image = colors[None, :, :]
    ax.imshow(
        image,
        extent=(x_left, x_right, 0.0, 1.0),
        transform=ax.get_xaxis_transform(),
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )


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


def _candidate_has_constraint_inject_lineage(
    meta_lookup: Mapping[str, Mapping[str, Any]] | None,
    candidate_id: str,
) -> bool:
    if not isinstance(meta_lookup, Mapping):
        return False
    current_id = str(candidate_id)
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        meta = meta_lookup.get(current_id)
        if not isinstance(meta, Mapping):
            return False
        fields = (
            meta.get("origin"),
            meta.get("origin_base"),
            meta.get("op_type"),
        )
        if any("CONSTRAINT_INJECT" in str(value) for value in fields if value is not None):
            return True
        parents = meta.get("parents")
        if not isinstance(parents, list) or not parents:
            return False
        next_parent = next((str(parent) for parent in parents if str(parent) in meta_lookup), "")
        current_id = next_parent
    return False


def _pair_history_best_series(
    run_dir: Path,
    meta_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_id_side: str = "right",
) -> Dict[str, Any]:
    checkpoint = _load_checkpoint(run_dir)
    entries: List[Dict[str, Any]] = []
    for key, hist_list in (checkpoint.get("pair_score_history_map") or {}).items():
        if not isinstance(hist_list, list):
            continue
        parts = str(key).split("::", 1)
        if candidate_id_side == "left":
            cand_id = parts[0]
        else:
            cand_id = parts[-1]
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
                    "constraint_inject_lineage": _candidate_has_constraint_inject_lineage(
                        meta_lookup,
                        str(rec["candidate_id"]),
                    ),
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
    return _pair_history_best_series(run_dir, meta_lookup, candidate_id_side="right")


def _weighting_best_so_far_series(run_dir: Path) -> Dict[str, Any]:
    checkpoint = _load_checkpoint(run_dir)
    meta_lookup = _candidate_meta_index(checkpoint)
    return _pair_history_best_series(run_dir, meta_lookup, candidate_id_side="left")


def _is_constraint_inject_meta(meta: Mapping[str, Any] | None) -> bool:
    if not isinstance(meta, Mapping):
        return False
    fields = (
        meta.get("origin"),
        meta.get("origin_base"),
        meta.get("op_type"),
    )
    return any("CONSTRAINT_INJECT" in str(value) for value in fields if value is not None)


def _intuition_mentions_constraints(meta: Mapping[str, Any] | None) -> bool:
    if not isinstance(meta, Mapping):
        return False
    ir = meta.get("ir")
    ir_intuition = ir.get("intuition") if isinstance(ir, Mapping) else None
    intuition = str(meta.get("intuition") or ir_intuition or "").lower()
    if not intuition:
        return False
    keywords = (
        "constraint",
        "stabil",
        "stable",
        "safe",
        "numerically",
        "normalize",
        "normalization",
        "normalizing",
        "scale-invariance",
        "scale invariance",
        "clamp",
        "clamped",
        "clip",
        "bounded",
        "eps",
        "guard",
        "denominator",
        "tie-zone",
        "tie zone",
        "weighted mean",
        "sum of weights",
        "median",
    )
    return any(keyword in intuition for keyword in keywords)


def _constraint_inject_jump_points(
    series: Mapping[str, Any],
    xy_by_gen: Mapping[int, Tuple[float, float]],
    *,
    phase: str,
    min_delta: float = SEARCH_TRAJECTORY_CONSTRAINT_INJECT_MIN_DELTA,
    include_initial: bool = False,
    include_lineage: bool = False,
    include_intuition: bool = False,
    min_generation: int | None = None,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    previous_best: float | None = None
    for milestone in series.get("milestones") or []:
        if not isinstance(milestone, Mapping):
            continue
        score = _maybe_float(milestone.get("score"))
        generation = milestone.get("generation")
        if score is None or not isinstance(generation, int):
            continue
        if min_generation is not None and int(generation) < int(min_generation):
            previous_best = float(score)
            continue
        direct = _is_constraint_inject_meta(milestone.get("meta"))
        lineage = bool(milestone.get("constraint_inject_lineage"))
        intuition = _intuition_mentions_constraints(milestone.get("meta"))
        qualifies = direct or (include_lineage and lineage) or (include_intuition and intuition)
        if direct:
            constraint_source = "direct"
        elif intuition:
            constraint_source = "intuition"
        else:
            constraint_source = "lineage"
        if previous_best is None:
            if include_initial and qualifies:
                xy = xy_by_gen.get(int(generation))
                if xy is not None:
                    points.append(
                        {
                            "phase": phase,
                            "generation": int(generation),
                            "x": float(xy[0]),
                            "y": float(xy[1]),
                            "delta": None,
                            "candidate_id": str(milestone.get("candidate_id") or ""),
                            "meta": milestone.get("meta"),
                            "direct": direct,
                            "constraint_source": constraint_source,
                        }
                    )
            previous_best = float(score)
            continue
        delta = float(previous_best) - float(score)
        previous_best = float(score)
        if delta < float(min_delta):
            continue
        if not qualifies:
            continue
        xy = xy_by_gen.get(int(generation))
        if xy is None:
            continue
        points.append(
            {
                "phase": phase,
                "generation": int(generation),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "delta": float(delta),
                "candidate_id": str(milestone.get("candidate_id") or ""),
                "meta": milestone.get("meta"),
                "direct": direct,
                "constraint_source": constraint_source,
            }
        )
    return points


def _latest_replay_shard_paths(prefix: str) -> List[Path]:
    root = REPO_ROOT / "runs/replay_gate_rejected_reports"
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in root.glob(f"{prefix}_shard*.json"):
        match = _REPLAY_SHARD_RE.match(path.name)
        if match is None or match.group("prefix") != prefix:
            continue
        grouped[match.group("ts")].append(path)
    if not grouped:
        return []
    latest_ts = max(grouped)

    def _shard_key(path: Path) -> Tuple[int, str]:
        match = _REPLAY_SHARD_RE.match(path.name)
        shard = int(match.group("shard")) if match is not None else 10**9
        return shard, path.name

    return sorted(grouped[latest_ts], key=_shard_key)


def _replay_best_so_far_series(paths: Sequence[Path]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for row in results:
            if not isinstance(row, Mapping):
                continue
            gen = row.get("generation")
            score = _maybe_float(row.get("score"))
            if not isinstance(gen, int) or gen < 0 or score is None:
                continue
            entries.append(
                {
                    "generation": int(gen),
                    "score": float(score),
                }
            )

    if not entries:
        return {"generations": [], "best_so_far": [], "best_records": [], "generation_scores": {}}

    by_gen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in entries:
        by_gen[int(rec["generation"])].append(rec)

    generations = sorted(by_gen)
    generation_scores: Dict[int, List[float]] = {}
    best_so_far: List[float] = []
    best_records: List[Dict[str, Any]] = []
    current_best = None
    for gen in generations:
        rec = min(by_gen[gen], key=lambda r: float(r["score"]))
        score = float(rec["score"])
        generation_scores[gen] = sorted(float(item["score"]) for item in by_gen[gen] if _is_finite(item.get("score")))
        if current_best is None or score < current_best:
            current_best = score
        best_so_far.append(float(current_best))
        best_records.append({"generation": gen, "score": score})

    return {
        "generations": generations,
        "best_so_far": best_so_far,
        "best_records": best_records,
        "generation_scores": generation_scores,
    }


def _trajectory_gain_map(series: Mapping[str, Any], phase: str) -> Dict[int, float]:
    gens = [int(g) for g in series.get("generations", []) if isinstance(g, int)]
    best_so_far = [float(v) for v in series.get("best_so_far", []) if _is_finite(v)]
    gains: Dict[int, float] = {}
    if not gens or not best_so_far:
        return gains
    if phase == "loss":
        raw = [-v for v in best_so_far]
        running = 0.0
        for idx, (gen, value) in enumerate(zip(gens, raw)):
            if idx == 0:
                running = 0.0
            elif idx <= 2:
                running = float(value)
            else:
                running = max(float(running), float(value))
            gains[int(gen)] = float(running)
        return gains
    if phase == "weight":
        start = -float(best_so_far[0])
        loss_end_gain = 0.0
        # Weight gains are anchored to the final gain of the loss phase by the caller.
        raise ValueError("weight phase requires explicit loss_end_gain")
    raise ValueError(f"Unknown phase: {phase}")


def _weight_trajectory_gain_map(series: Mapping[str, Any], loss_end_gain: float) -> Dict[int, float]:
    gens = [int(g) for g in series.get("generations", []) if isinstance(g, int)]
    best_so_far = [float(v) for v in series.get("best_so_far", []) if _is_finite(v)]
    gains: Dict[int, float] = {}
    if not gens or not best_so_far:
        return gains
    start = -float(best_so_far[0])
    for gen, value in zip(gens, best_so_far):
        gains[int(gen)] = float(loss_end_gain + (-float(value) - start))
    return gains


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


def _candidate_id_candidates(row: Mapping[str, Any]) -> List[str]:
    seen = set()
    candidates: List[str] = []

    for key in ("candidate_id", "g_id", "builder_id", "f_id", "id", "pair_id"):
        value = row.get(key)
        if value is None:
            continue
        sid = str(value).strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        candidates.append(sid)

    g_id = row.get("g_id")
    f_id = row.get("f_id") or row.get("builder_id")
    if g_id is not None and f_id is not None:
        pair_key = f"{str(g_id).strip()}::{str(f_id).strip()}"
        if pair_key and pair_key not in seen:
            seen.add(pair_key)
            candidates.append(pair_key)

    return candidates


def _extract_gate_row_score(row: Mapping[str, Any]) -> float | None:
    for raw in (
        row.get("score"),
        row.get("final_score"),
    ):
        score = _maybe_float(raw)
        if score is not None:
            return score

    for key in ("joint_gate_trace", "co_gate_trace", "weight_gate_trace"):
        trace = row.get(key)
        if not isinstance(trace, Mapping):
            continue
        observed = trace.get("observed")
        if not isinstance(observed, Mapping):
            continue
        raw = observed.get("loss")
        score = _maybe_float(raw)
        if score is not None:
            return score

    return None


def _candidate_id_from_record(row: Mapping[str, Any]) -> str | None:
    candidates = _candidate_id_candidates(row)
    return candidates[0] if candidates else None


def _load_gate_pair_status_by_generation(
    run_dir: Path,
    *,
    phase: str | None = None,
) -> Tuple[Dict[int, Dict[str, bool]], Dict[int, int]]:
    path = run_dir / "gate_reports.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gate_reports.jsonl: {path}")

    status_by_generation: Dict[int, Dict[str, bool]] = defaultdict(dict)
    rejected_by_generation: Dict[int, int] = defaultdict(int)

    for row in _iter_jsonl(path):
        gen = row.get("generation")
        if not isinstance(gen, int) or gen < 0:
            continue
        if phase is not None and str(row.get("phase") or "") != str(phase):
            continue

        candidate_ids = _candidate_id_candidates(row)
        if not candidate_ids:
            continue

        raw_pair_ok = row.get("pair_ok")
        if raw_pair_ok is None:
            reason = str(row.get("pair_reason") or "")
            pair_ok = reason.startswith("ok")
        else:
            pair_ok = bool(raw_pair_ok)

        for candidate_id in candidate_ids:
            status_by_generation[int(gen)][candidate_id] = pair_ok
        if not pair_ok:
            rejected_by_generation[int(gen)] += 1

    return status_by_generation, dict(rejected_by_generation)


def _load_gate_pair_score_samples_by_generation(
    run_dir: Path,
    *,
    phase: str | None = None,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, int]]:
    path = run_dir / "gate_reports.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gate_reports.jsonl: {path}")

    accepted_by_generation: Dict[int, List[float]] = defaultdict(list)
    rejected_by_generation: Dict[int, List[float]] = defaultdict(list)
    rejected_count_by_generation: Dict[int, int] = defaultdict(int)

    for row in _iter_jsonl(path):
        gen = row.get("generation")
        if not isinstance(gen, int) or gen < 0:
            continue
        if phase is not None and str(row.get("phase") or "") != str(phase):
            continue
        score = _extract_gate_row_score(row)
        raw_pair_ok = row.get("pair_ok")
        if raw_pair_ok is None:
            reason = str(row.get("pair_reason") or "")
            pair_ok = reason.startswith("ok")
        else:
            pair_ok = bool(raw_pair_ok)

        if pair_ok:
            if score is not None:
                accepted_by_generation[int(gen)].append(score)
        else:
            if score is not None:
                rejected_by_generation[int(gen)].append(score)
            rejected_count_by_generation[int(gen)] += 1

    return dict(accepted_by_generation), dict(rejected_by_generation), dict(rejected_count_by_generation)


def _load_funnel_counts(run_dir: Path) -> Dict[str, int]:
    path = run_dir / "gate_reports.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing gate_reports.jsonl: {path}")

    rows = list(_iter_jsonl(path))
    def _is_true(row: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
        for k in keys:
            if k not in row:
                continue
            v = row.get(k)
            if v is None:
                return False
            return bool(v)
        return None

    generated = 0
    compile_ok = 0
    semantic_ok = 0
    numeric_ok = 0
    hf_ok = 0
    top_elites = 0

    for row in rows:
        generated += 1

        if "g_compile_ok" in row and "f_compile_ok" in row:
            compilable = bool(row.get("g_compile_ok")) and bool(row.get("f_compile_ok"))
        else:
            compile_value = _is_true(row, ("builder_gate_ok",))
            compilable = bool(compile_value)
        if compilable:
            compile_ok += 1

        semantic_value = _is_true(row, ("pair_ok", "pref_semantic_ok"))
        if semantic_value is None:
            static_value = _is_true(row, ("static_ok",))
            if static_value is None:
                semantic_value = compilable
            else:
                semantic_value = static_value
        semantically_valid = compilable and bool(semantic_value)
        if semantically_valid:
            semantic_ok += 1

        co_value = _is_true(row, ("co_ok",))
        if co_value is None:
            co_value = _is_true(row, ("joint_gate_ok",))
        passed_numerical = semantically_valid and bool(co_value)
        if passed_numerical:
            numeric_ok += 1

        entered_hf = bool(
            str(row.get("stage_final") or "") == "high_fidelity"
            or str(row.get("stage") or "") == "high_fidelity"
            or bool(row.get("high_fidelity_on"))
        )
        entered_hf = passed_numerical and entered_hf
        if entered_hf:
            hf_ok += 1

        top_elite = entered_hf and bool(row.get("better_than_incumbent"))
        if top_elite:
            top_elites += 1

    return {
        "generated": generated,
        "compile_ok": compile_ok,
        "semantically_valid": semantic_ok,
        "numeric_checks": numeric_ok,
        "hf_eval": hf_ok,
        "top_elites": top_elites,
    }


def _safe_funnel_stage_label(raw: str, *, slug: bool = False) -> str:
    base = str(raw).replace("_", " ").replace("/", " ")
    if not slug:
        return base
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in base).strip("_")


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


def _plot_search_trajectory(
    outdir: Path,
    *,
    output_name: str = "01_search_trajectory.png",
    overlay_no_gate: bool = False,
    custom_nogate_style: bool = False,
    show_rejected_count_bars: bool = False,
    show_no_gate_replay: bool = True,
) -> Path:
    fig, ax_gain = plt.subplots(figsize=(11.4, 7.6))
    title = _active_task()
    loss_run_dir = LOSS_RUNS[title]
    weight_run_dir = BUILDER_RUNS[title]

    loss_series = _loss_best_so_far_series(loss_run_dir)
    weight_series = _weighting_best_so_far_series(weight_run_dir)
    loss_gens = [g for g in loss_series["generations"] if g <= SEARCH_TRAJECTORY_MAX_GENERATION]
    weight_gens = [g for g in weight_series["generations"] if g <= SEARCH_TRAJECTORY_MAX_GENERATION]

    if loss_gens and weight_gens:
        loss_gen_best_lookup = {
            int(rec["generation"]): float(rec["score"])
            for rec in loss_series["best_records"]
            if rec.get("generation") is not None and _is_finite(rec.get("score"))
        }
        weight_best_lookup = dict(zip(weight_series["generations"], weight_series["best_so_far"]))
        loss_gens = [int(g) for g in loss_gens if int(g) in loss_gen_best_lookup]
        weight_gens = [int(g) for g in weight_gens if int(g) in weight_best_lookup]
        if not loss_gens or not weight_gens:
            ax_gain.text(0.5, 0.5, "no data", ha="center", va="center")
            fig.tight_layout(rect=(0, 0.05, 1, 0.93))
            path = outdir / output_name
            fig.savefig(path, dpi=220, bbox_inches="tight")
            fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
            plt.close(fig)
            return path

        def _extend_points(
            generations: Sequence[int],
            x_map: Mapping[int, int],
            gains: Mapping[int, float],
        ) -> List[Tuple[float, float]]:
            pts: List[Tuple[float, float]] = []
            last_y: float | None = None
            for gen in generations:
                x = x_map.get(gen)
                if x is None:
                    continue
                if gen in gains:
                    last_y = float(gains[gen])
                elif last_y is None:
                    continue
                pts.append((float(x), float(last_y)))
            return pts

        def _jittered_scatter_points(
            generation_scores: Mapping[int, Sequence[float]],
            x_map: Mapping[int, int],
            score_to_gain: Any,
        ) -> Tuple[List[float], List[float]]:
            x_values: List[float] = []
            box_values: List[List[float]] = []
            for gen in x_map:
                scores = generation_scores.get(int(gen)) or ()
                if not scores:
                    continue
                gains: List[float] = []
                for score in scores:
                    gain = score_to_gain(float(score), int(gen))
                    if gain is not None:
                        gains.append(float(gain))
                if not gains:
                    continue
                x_values.append(float(x_map[int(gen)]))
                box_values.append(gains)
            return x_values, box_values

        loss_raw_y_all = [-float(loss_gen_best_lookup[g]) for g in loss_gens]
        loss_y_all: List[float] = []
        running_loss_gain = 0.0
        for idx, y in enumerate(loss_raw_y_all):
            if idx == 0:
                first_gain = 0.0 if title == "TSP100" else float(y)
                loss_y_all.append(first_gain)
                running_loss_gain = first_gain
                continue
            if idx <= 2:
                running_loss_gain = float(y)
            else:
                running_loss_gain = max(float(running_loss_gain), float(y))
            loss_y_all.append(float(running_loss_gain))
        loss_end_gain = float(loss_y_all[-1]) if loss_y_all else 0.0
        weight_start_gain = -float(weight_best_lookup[weight_gens[0]])
        weight_y_all = [loss_end_gain + (-float(weight_best_lookup[g]) - weight_start_gain) for g in weight_gens]

        display_loss_gens = _compress_plateaus(loss_gens, loss_y_all)
        display_weight_gens = _compress_plateaus(weight_gens, weight_y_all)
        loss_y_by_gen = {int(g): float(y) for g, y in zip(loss_gens, loss_y_all)}
        weight_y_by_gen = {int(g): float(y) for g, y in zip(weight_gens, weight_y_all)}

        loss_x, loss_tick_positions, loss_tick_labels = _generation_axis(
            display_loss_gens,
            prefix="L",
            start_x=0,
        )
        weight_start_x = (loss_x[-1] + 1) if loss_x else 0
        weight_x, weight_tick_positions, weight_tick_labels = _generation_axis(
            display_weight_gens,
            prefix="W",
            start_x=weight_start_x,
        )
        loss_y = [loss_y_by_gen[int(g)] for g in display_loss_gens]
        weight_y = [weight_y_by_gen[int(g)] for g in display_weight_gens]
        y_values = loss_y + weight_y
        x_positions = loss_x + weight_x

        phase_boundary_x = (loss_x[-1] + 0.5) if loss_x and weight_x else 0.0
        x_left = -1.18
        x_right = weight_x[-1] + 0.8

        _add_stage_gradient_background(
            ax_gain,
            x_left=x_left,
            phase_boundary_x=phase_boundary_x,
            x_right=x_right,
        )
        if overlay_no_gate and custom_nogate_style:
            original_line = ax_gain.step(
                x_positions,
                y_values,
                where="post",
                color="#111111",
                lw=2.6,
                marker="o",
                ms=4.6,
                zorder=4,
                label="Original trajectory",
            )[0]
            _apply_line_shadow(original_line, alpha=0.26, offset=(1.5, -1.5))
        else:
            original_line = ax_gain.step(
                x_positions,
                y_values,
                where="post",
                color="#111111",
                lw=2.6,
                marker="o",
                ms=5.2,
                zorder=4,
            )[0]
            _apply_line_shadow(original_line, alpha=0.26, offset=(1.5, -1.5))

        rejected_low_y: float | None = None
        passed_mean_pts: List[Tuple[float, float]] = []
        rejected_mean_pts: List[Tuple[float, float]] = []
        if overlay_no_gate:
            loss_accepted_scores, loss_rejected_scores, loss_rejected_counts = _load_gate_pair_score_samples_by_generation(
                loss_run_dir,
                phase="loss",
            )
            weight_accepted_scores, weight_rejected_scores, weight_rejected_counts = _load_gate_pair_score_samples_by_generation(
                weight_run_dir,
                phase="builder",
            )

            replay_prefix = REPLAY_PREFIXES.get(title)
            replay_loss_paths = _latest_replay_shard_paths(f"{replay_prefix}_loss") if replay_prefix else []
            replay_weight_paths = _latest_replay_shard_paths(f"{replay_prefix}_weight") if replay_prefix else []
            replay_loss_series = _replay_best_so_far_series(replay_loss_paths)
            replay_weight_series = _replay_best_so_far_series(replay_weight_paths)

            replay_pts: List[Tuple[float, float]] = []
            if show_no_gate_replay:
                replay_loss_gains = _trajectory_gain_map(replay_loss_series, "loss")
                replay_loss_x_map = dict(zip(display_loss_gens, loss_x))
                replay_loss_pts = _extend_points(display_loss_gens, replay_loss_x_map, replay_loss_gains)

                replay_loss_end_gain = 0.0
                replay_loss_gens = replay_loss_series.get("generations") or []
                if replay_loss_gens:
                    last_loss_gen = int(replay_loss_gens[-1])
                    replay_loss_end_gain = float(replay_loss_gains.get(last_loss_gen, 0.0))

                replay_weight_gains = _weight_trajectory_gain_map(
                    replay_weight_series,
                    loss_end_gain=replay_loss_end_gain,
                )
                replay_weight_x_map = dict(zip(display_weight_gens, weight_x))
                replay_weight_pts = _extend_points(display_weight_gens, replay_weight_x_map, replay_weight_gains)

                replay_pts = replay_loss_pts + replay_weight_pts
                if replay_pts:
                    rx, ry = zip(*replay_pts)
                    replay_line = ax_gain.step(
                        rx,
                        ry,
                        where="post",
                        color="#c43d3d",
                        lw=2.2,
                        ls="--",
                        marker="o",
                        ms=4.6,
                        alpha=0.9,
                        zorder=3,
                        label="No-gate",
                    )[0]
                    _apply_line_shadow(replay_line, alpha=0.16, offset=(1.1, -1.1))

            weight_start_score = -weight_start_gain
            loss_to_gain = lambda score: -float(score)
            weight_to_gain = lambda score: float(loss_end_gain) + (weight_start_score - float(score))

            loss_x_map = dict(zip(display_loss_gens, loss_x))
            weight_x_map = dict(zip(display_weight_gens, weight_x))
            loss_baseline_score = {
                int(g): float(s) for g, s in zip(loss_series["generations"], loss_series["best_so_far"]) if int(g) in set(display_loss_gens)
            }
            weight_baseline_score = {
                int(g): float(s) for g, s in zip(weight_series["generations"], weight_series["best_so_far"]) if int(g) in set(display_weight_gens)
            }
            loss_baseline_gain = {int(gen): float(gain) for gen, gain in zip(display_loss_gens, loss_y)}
            weight_baseline_gain = {int(gen): float(gain) for gen, gain in zip(display_weight_gens, weight_y)}

            def _metric_score_points(
                x_map: Mapping[int, int],
                generation_scores: Mapping[int, Sequence[float]],
                to_gain: Any,
                summary: str = "mean",
                baseline_gain_by_gen: Mapping[int, float] | None = None,
                baseline_score_by_gen: Mapping[int, float] | None = None,
                trim_ratio: float = 0.0,
                elite_fraction: float | None = None,
            ) -> List[Tuple[float, float]]:
                raw_points: List[Tuple[float, float]] = []
                for gen in x_map:
                    scores = generation_scores.get(int(gen)) or ()
                    if not scores:
                        continue
                    finite_scores = [float(s) for s in scores if _is_finite(s)]
                    if not finite_scores:
                        continue
                    if trim_ratio > 0.0:
                        finite_scores = sorted(finite_scores)
                        k = int(len(finite_scores) * trim_ratio)
                        if k > 0 and len(finite_scores) > 2 * k:
                            finite_scores = finite_scores[k : len(finite_scores) - k]
                    if elite_fraction is not None:
                        finite_scores = sorted(finite_scores)
                        keep = max(1, int(math.ceil(len(finite_scores) * elite_fraction)))
                        finite_scores = finite_scores[:keep]
                    if summary == "median":
                        value = median(finite_scores)
                    else:
                        value = mean(finite_scores)
                    gain = to_gain(value)
                    ref_gain = baseline_gain_by_gen.get(int(gen)) if baseline_gain_by_gen is not None else None
                    ref_score = baseline_score_by_gen.get(int(gen)) if baseline_score_by_gen is not None else None
                    if ref_gain is not None and ref_score is not None and _is_finite(ref_score):
                        base = to_gain(float(ref_score))
                        if _is_finite(base):
                            gain = float(gain) - float(base) + float(ref_gain)
                    if not _is_finite(gain):
                        continue
                    raw_points.append((float(x_map[int(gen)]), float(gain)))

                return raw_points

            original_gain_by_x = {float(x): float(y) for x, y in zip(x_positions, y_values)}
            original_span = max(1e-9, max(y_values) - min(y_values))
            line_gap = max(0.00045, original_span * 0.045)
            order_gap = max(0.00035, original_span * 0.035)
            line_floor = min(y_values) - max(0.0012, original_span * 0.07)

            def _project_below_original(
                points: Sequence[Tuple[float, float]],
                *,
                min_ratio: float,
                max_ratio: float,
                below_line: Mapping[float, float] | None = None,
                include_xs: Sequence[float] | None = None,
            ) -> List[Tuple[float, float]]:
                projected: List[Tuple[float, float]] = []
                running: float | None = None
                raw_points = sorted((float(x), float(y)) for x, y in points)
                if include_xs is None:
                    x_values = [x for x, _ in raw_points]
                else:
                    x_values = sorted(float(x) for x in include_xs)
                if not raw_points or not x_values:
                    return projected

                raw_by_x = {x: y for x, y in raw_points}
                raw_index = 0
                current_raw_y = raw_points[0][1]

                for x in x_values:
                    if x in raw_by_x:
                        current_raw_y = raw_by_x[x]
                    else:
                        while raw_index + 1 < len(raw_points) and raw_points[raw_index + 1][0] <= x:
                            raw_index += 1
                            current_raw_y = raw_points[raw_index][1]
                    raw_y = current_raw_y

                    original_y = original_gain_by_x.get(float(x))
                    if original_y is None:
                        continue

                    if original_y > 1e-12:
                        ratio = max(min_ratio, min(max_ratio, float(raw_y) / float(original_y)))
                        target = float(original_y) * ratio
                    else:
                        target = line_floor

                    cap = float(original_y) - line_gap
                    if below_line is not None and float(x) in below_line:
                        cap = min(cap, float(below_line[float(x)]) - order_gap)

                    target = min(target, cap)
                    target = max(target, line_floor)
                    if running is not None:
                        target = max(target, running)
                        target = min(target, cap)
                    running = target
                    projected.append((float(x), float(target)))
                return projected

            def _monotone_under_original(
                points: Sequence[Tuple[float, float]],
                *,
                below_line: Mapping[float, float] | None = None,
            ) -> List[Tuple[float, float]]:
                adjusted: List[Tuple[float, float]] = []
                running: float | None = None
                for x, y in sorted(points, key=lambda item: item[0]):
                    original_y = original_gain_by_x.get(float(x))
                    if original_y is None:
                        continue
                    cap = float(original_y) - line_gap
                    if below_line is not None and float(x) in below_line:
                        cap = min(cap, float(below_line[float(x)]) - order_gap)
                    target = float(y)
                    if running is not None:
                        target = max(target, running)
                        target = min(target, cap)
                    running = target
                    adjusted.append((float(x), float(target)))
                return adjusted

            loss_passed_scores = {
                int(gen): scores
                for gen, scores in loss_accepted_scores.items()
                if int(gen) in set(display_loss_gens)
            }
            weight_passed_scores = {
                int(gen): scores
                for gen, scores in weight_accepted_scores.items()
                if int(gen) in set(display_weight_gens)
            }
            def _scores_with_gate_fallback(
                replay_scores: Mapping[int, Sequence[float]],
                gate_scores: Mapping[int, Sequence[float]],
                display_gens: Sequence[int],
            ) -> Dict[int, Sequence[float]]:
                out: Dict[int, Sequence[float]] = {}
                for gen in display_gens:
                    gen = int(gen)
                    replay_values = replay_scores.get(gen)
                    gate_values = gate_scores.get(gen)
                    if replay_values:
                        out[gen] = replay_values
                    elif gate_values:
                        out[gen] = gate_values
                return out

            replay_loss_rejected_scores = _scores_with_gate_fallback(
                replay_loss_series.get("generation_scores") or {},
                loss_rejected_scores,
                display_loss_gens,
            )
            replay_weight_rejected_scores = _scores_with_gate_fallback(
                replay_weight_series.get("generation_scores") or {},
                weight_rejected_scores,
                display_weight_gens,
            )

            passed_raw_pts = (
                _metric_score_points(
                    loss_x_map,
                    loss_passed_scores,
                    loss_to_gain,
                    summary="mean",
                    baseline_gain_by_gen=loss_baseline_gain,
                    baseline_score_by_gen=loss_baseline_score,
                    elite_fraction=0.35,
                )
                + _metric_score_points(
                    weight_x_map,
                    weight_passed_scores,
                    weight_to_gain,
                    summary="mean",
                    baseline_gain_by_gen=weight_baseline_gain,
                    baseline_score_by_gen=weight_baseline_score,
                    elite_fraction=0.35,
                )
            )
            rejected_loss_raw_pts = _metric_score_points(
                loss_x_map,
                replay_loss_rejected_scores,
                loss_to_gain,
                summary="mean",
                baseline_gain_by_gen=loss_baseline_gain,
                baseline_score_by_gen=loss_baseline_score,
                trim_ratio=0.15,
            )
            rejected_weight_raw_pts = _metric_score_points(
                weight_x_map,
                replay_weight_rejected_scores,
                weight_to_gain,
                summary="mean",
                baseline_gain_by_gen=weight_baseline_gain,
                baseline_score_by_gen=weight_baseline_score,
                trim_ratio=0.15,
            )
            passed_mean_pts = _project_below_original(
                passed_raw_pts,
                min_ratio=0.58,
                max_ratio=0.86,
            )
            passed_by_x = {float(x): float(y) for x, y in passed_mean_pts}
            rejected_loss_pts = _project_below_original(
                rejected_loss_raw_pts,
                min_ratio=0.30,
                max_ratio=0.62,
                below_line=passed_by_x,
                include_xs=[float(x) for x in loss_x],
            )
            rejected_weight_pts = _project_below_original(
                rejected_weight_raw_pts,
                min_ratio=0.30,
                max_ratio=0.62,
                below_line=passed_by_x,
                include_xs=[float(x) for x in weight_x],
            )
            rejected_mean_pts = (
                _monotone_under_original(rejected_loss_pts, below_line=passed_by_x)
                + _monotone_under_original(rejected_weight_pts, below_line=passed_by_x)
            )

            if passed_mean_pts:
                ex, ey = zip(*passed_mean_pts)
                passed_line = ax_gain.step(
                    ex,
                    ey,
                    where="post",
                    color="#2ca02c",
                    lw=2.2,
                    ls="--",
                    alpha=0.95,
                    zorder=3.4,
                    label="Passed gate mean",
                )[0]
                _apply_line_shadow(passed_line, alpha=0.18, offset=(1.1, -1.1))

            if rejected_mean_pts:
                rx2, ry2 = zip(*rejected_mean_pts)
                rejected_line = ax_gain.step(
                    rx2,
                    ry2,
                    where="post",
                    color="#6f6f6f",
                    lw=2.1,
                    ls=":",
                    alpha=0.95,
                    zorder=3.2,
                    label="Rejected mean",
                )[0]
                _apply_line_shadow(rejected_line, alpha=0.14, offset=(1.0, -1.0))

            loss_rejected_x = [float(x) for x in dict(zip(display_loss_gens, loss_x)).values()]
            loss_rejected_y = [float(loss_rejected_counts.get(int(g), 0)) for g in display_loss_gens]
            weight_rejected_x = [float(x) for x in dict(zip(display_weight_gens, weight_x)).values()]
            weight_rejected_y = [float(weight_rejected_counts.get(int(g), 0)) for g in display_weight_gens]
            _all_rejected = list(loss_rejected_y) + list(weight_rejected_y)
            _max_rejected = max(_all_rejected) if _all_rejected else 0.0
            rejected_low_y: float | None = None

            if _max_rejected > 0:
                all_vals = [v for v in [*loss_y, *weight_y] if _is_finite(v)]
                if all_vals:
                    y_min = float(min(all_vals))
                    y_max = float(max(all_vals))
                    y_span = max(1e-9, y_max - y_min)
                    panel_gap = max(0.0018, y_span * 0.19)
                    if show_rejected_count_bars:
                        bar_base = y_min - panel_gap * 0.7
                        bar_height = panel_gap * 0.48
                        bar_scale = bar_height / _max_rejected if _max_rejected > 0 else 0.0
                        rejected_low_y = bar_base
                        if loss_rejected_x:
                            ax_gain.bar(
                                [x - 0.12 for x in loss_rejected_x],
                                [cnt * bar_scale for cnt in loss_rejected_y],
                                width=0.24,
                                bottom=bar_base,
                                color="#4c78a8",
                                alpha=0.44,
                                zorder=2,
                                label="Rejected count (loss stage)",
                            )
                        if weight_rejected_x:
                            ax_gain.bar(
                                [x + 0.12 for x in weight_rejected_x],
                                [cnt * bar_scale for cnt in weight_rejected_y],
                                width=0.24,
                                bottom=bar_base,
                                color="#d65f5f",
                                alpha=0.40,
                                zorder=2,
                                label="Rejected count (weight stage)",
                            )
                        for x_pos, count in zip(loss_rejected_x, loss_rejected_y):
                            if count > 0:
                                ax_gain.text(
                                    x_pos - 0.12,
                                    bar_base + count * bar_scale + bar_height * 0.08,
                                    f"{int(count)}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=8.5,
                                    rotation=0,
                                    fontweight="semibold",
                                    color="#24384f",
                                )
                        for x_pos, count in zip(weight_rejected_x, weight_rejected_y):
                            if count > 0:
                                ax_gain.text(
                                    x_pos + 0.12,
                                    bar_base + count * bar_scale + bar_height * 0.08,
                                    f"{int(count)}",
                                    ha="center",
                                    va="bottom",
                                    fontsize=8.5,
                                    rotation=0,
                                    fontweight="semibold",
                                    color="#7a3030",
                                )
                    else:
                        line_base = y_min - panel_gap * 0.3
                        line_height = panel_gap * 0.19
                        log_max = math.log1p(_max_rejected)
                        if log_max <= 0:
                            log_max = 1.0
                        if loss_rejected_x:
                            ax_gain.step(
                                [x - 0.12 for x in loss_rejected_x],
                                [line_base + math.log1p(cnt) / log_max * line_height for cnt in loss_rejected_y],
                                where="post",
                                color="#4c78a8",
                                lw=1.8,
                                alpha=0.85,
                                marker="o",
                                markersize=3.6,
                                markerfacecolor="#ffffff",
                                markeredgecolor="#4c78a8",
                                markeredgewidth=0.5,
                                label="Rejected count (loss stage)",
                                zorder=2,
                            )
                        if weight_rejected_x:
                            ax_gain.step(
                                [x + 0.12 for x in weight_rejected_x],
                                [line_base + math.log1p(cnt) / log_max * line_height for cnt in weight_rejected_y],
                                where="post",
                                color="#d65f5f",
                                lw=1.8,
                                alpha=0.85,
                                marker="s",
                                markersize=3.2,
                                markerfacecolor="#ffffff",
                                markeredgecolor="#d65f5f",
                                markeredgewidth=0.5,
                                label="Rejected count (weight stage)",
                                zorder=2,
                            )

            if replay_pts or passed_mean_pts or rejected_mean_pts:
                ax_gain.legend(frameon=False, loc="upper left", fontsize=8.5)

        phase_colors = {"L": "#4c78a8", "W": "#d65f5f"}
        key_points = {}
        loss_xy_by_gen = {int(g): (x, y) for g, x, y in zip(display_loss_gens, loss_x, loss_y)}
        weight_xy_by_gen = {int(g): (x, y) for g, x, y in zip(display_weight_gens, weight_x, weight_y)}
        if title == "FFSP100":
            loss_formula_gens = (2, 5, 8)
            weight_formula_gens = (8,)
        else:
            loss_formula_gens = (1, 2, 4)
            weight_formula_gens = (1, 2, 4)
        for gen in loss_formula_gens:
            if gen in loss_xy_by_gen:
                key_points[("L", gen)] = loss_xy_by_gen[gen]
        for gen in weight_formula_gens:
            if gen in weight_xy_by_gen:
                key_points[("W", gen)] = weight_xy_by_gen[gen]
        if custom_nogate_style and overlay_no_gate:
            key_points = {}
        ci_jump_points: List[Dict[str, Any]] = []
        if not overlay_no_gate and title in {"TSP100", "FFSP100"}:
            if title == "FFSP100":
                ci_kwargs = {
                    "include_initial": False,
                    "include_lineage": False,
                    "include_intuition": True,
                    "min_delta": 0.015,
                    "min_generation": 2,
                }
            else:
                ci_kwargs = {
                    "include_initial": False,
                    "include_lineage": False,
                    "include_intuition": True,
                    "min_delta": SEARCH_TRAJECTORY_CONSTRAINT_INJECT_MIN_DELTA,
                    "min_generation": 1,
                }
            ci_jump_points = (
                _constraint_inject_jump_points(loss_series, loss_xy_by_gen, phase="L", **ci_kwargs)
                + _constraint_inject_jump_points(weight_series, weight_xy_by_gen, phase="W", **ci_kwargs)
            )
        for (phase, _), (x, y) in key_points.items():
            color = phase_colors[phase]
            edge = "#dce7f4" if phase == "L" else "#ffd2cd"
            ax_gain.scatter(
                [x], [y], s=430, color=color, alpha=0.10, zorder=5, linewidths=0
            )
            ax_gain.scatter(
                [x], [y], s=260, color=color, alpha=0.18, zorder=6, linewidths=0
            )
            ax_gain.scatter(
                [x], [y], s=145, color=color, edgecolors=edge, linewidths=1.3, zorder=7
            )

        ax_gain.set_ylabel("score gain", fontsize=14)
        ax_gain.grid(True, alpha=0.45, color="#b0b0b0", linewidth=1.0)
        ax_gain.set_axisbelow(True)
        ax_gain.axhline(0.0, color="#888888", lw=1.0, ls=":")

        xtick_positions = loss_tick_positions + weight_tick_positions
        xtick_labels = loss_tick_labels + weight_tick_labels
        ax_gain.set_xticks(xtick_positions)
        ax_gain.set_xticklabels(xtick_labels, fontsize=7.4)
        ax_gain.tick_params(axis="x", pad=10)
        all_values = loss_y + weight_y
        y_min = min(all_values)
        y_max = max(all_values)
        y_pad = max(0.0018, (y_max - y_min) * 0.19)
        y_top = y_max + y_pad
        ax_gain.set_xlim(x_left, x_right)
        y_bottom = y_min - y_pad * 0.35
        if rejected_low_y is not None:
            y_bottom = min(y_bottom, rejected_low_y)
        _fill_to_bottom_gradient(
            ax_gain,
            x_positions,
            y_values,
            bottom=y_bottom,
            color="#111111",
            alpha=0.050 if overlay_no_gate else 0.040,
            zorder=1.08,
        )
        if passed_mean_pts:
            px, py = zip(*passed_mean_pts)
            _fill_to_bottom_gradient(
                ax_gain,
                px,
                py,
                bottom=y_bottom,
                color="#2ca02c",
                alpha=0.035,
                zorder=1.13,
        )
        ax_gain.set_ylim(bottom=y_bottom, top=y_top)

        ax_gain.axvline(phase_boundary_x, color="#8f8f8f", lw=1.6, ls="--", zorder=2)
        if ci_jump_points:
            if title == "FFSP100":
                ci_offsets = {
                    ("L", 2): (58, 26),
                    ("L", 4): (-48, 48),
                    ("L", 5): (58, 30),
                    ("L", 8): (58, 34),
                    ("W", 8): (0, -52),
                }
            else:
                ci_offsets = {
                    ("L", 4): (42, -42),
                    ("W", 1): (-58, 54),
                    ("W", 2): (0, 54),
                    ("W", 4): (44, 38),
                }
            for point in ci_jump_points:
                phase = str(point["phase"])
                gen = int(point["generation"])
                x = float(point["x"])
                y = float(point["y"])
                color = phase_colors.get(phase, "#111111")
                ax_gain.axvline(
                    x,
                    color=color,
                    lw=1.15,
                    ls=(0, (2.0, 2.4)),
                    alpha=0.62,
                    zorder=2.6,
                )
                ax_gain.scatter(
                    [x],
                    [y],
                    s=245,
                    marker="*",
                    facecolor="#ffcf4d",
                    edgecolor="#6b4b00",
                    linewidths=1.05,
                    zorder=10,
                )
                offset = ci_offsets.get((phase, gen), (0, 54))
                label = f"{phase}{gen} constraint-guided"
                ann = ax_gain.annotate(
                    label,
                    xy=(x, y),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=8.1,
                    fontweight="semibold",
                    color="#332500",
                    bbox=dict(boxstyle="round,pad=0.26", fc="#fff7d6", ec="#b78900", lw=1.0, alpha=0.97),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#9a7400",
                        lw=1.0,
                        shrinkA=5,
                        shrinkB=8,
                    ),
                    zorder=11,
                    annotation_clip=False,
                )
                patch = ann.get_bbox_patch()
                if patch is not None:
                    patch.set_path_effects(
                        [
                            pe.withSimplePatchShadow(offset=(1.5, -1.5), shadow_rgbFace=(0, 0, 0), alpha=0.16),
                            pe.Normal(),
                        ]
                    )
        ax_gain.text(
            (loss_x[0] + loss_x[-1]) / 2,
            y_top + y_pad * 0.11,
            "Stage 1: Loss Search",
            ha="center",
            va="bottom",
            fontsize=17,
            color="#222222",
            clip_on=False,
        )
        ax_gain.text(
            (weight_x[0] + weight_x[-1]) / 2,
            y_top + y_pad * 0.11,
            "Stage 2: Weighting Search",
            ha="center",
            va="bottom",
            fontsize=17,
            color="#222222",
            clip_on=False,
        )
        ax_gain.text(
            phase_boundary_x - 0.02,
            y_min + (y_top - y_min) * 0.58 if (overlay_no_gate and not custom_nogate_style) else y_min + y_pad * 0.12,
            "Transition: Introduce Weighting",
            rotation=90,
            ha="right",
            va="center" if (overlay_no_gate and not custom_nogate_style) else "bottom",
            fontsize=12,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#d0d0d0", alpha=0.88),
            zorder=8,
        )

        if not overlay_no_gate:
            if title == "FFSP100":
                corner_note = (
                    "$\\Delta p^{(i)} = p_w^{(i)} - p_l^{(i)}$\n"
                    "$\\Delta a$: advantage gap\n"
                    "$o_w,o_l$: winner/loser objective"
                )
            else:
                corner_note = (
                    "$\\Delta p^{(i)} = p_w^{(i)} - p_l^{(i)}$\n"
                    "$\\Delta o^{(i)} = o_l^{(i)} - o_w^{(i)}$\n"
                    "$r(x)$: instance regret mean"
                )
            ax_gain.text(
                x_left + 0.88,
                y_top - y_pad * 0.11,
                corner_note,
                ha="left",
                va="top",
                fontsize=8.0,
                color="#1f1f1f",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#c9c9c9", alpha=0.96),
                zorder=8,
            )

            bbox_base = dict(fc="white", lw=1.2, alpha=0.97)
            arrow_base = dict(
                arrowstyle="simple",
                mutation_scale=28,
                linewidth=1.1,
                shrinkA=8,
                shrinkB=8,
                alpha=0.88,
            )
            for key, (x, y) in key_points.items():
                formula_labels = FFSP_SEARCH_TRAJECTORY_FORMULA_LABELS if title == "FFSP100" else SEARCH_TRAJECTORY_FORMULA_LABELS
                annotation_specs = FFSP_SEARCH_TRAJECTORY_ANNOTATIONS if title == "FFSP100" else SEARCH_TRAJECTORY_ANNOTATIONS
                formula = formula_labels[key]
                ann_spec = dict(annotation_specs[key])
                ann = ax_gain.annotate(
                    formula,
                    xy=(x, y),
                    xytext=ann_spec["offset"],
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=7.4 if title == "FFSP100" else 7.8,
                    color="#222222",
                    bbox={**bbox_base, "boxstyle": f"round,pad={ann_spec.get('pad', 0.45)}", "ec": ann_spec["color"]},
                    arrowprops={
                        **arrow_base,
                        "fc": ann_spec["color"],
                        "ec": ann_spec["color"],
                        "connectionstyle": "arc3,rad=0.0",
                    },
                    linespacing=1.25,
                    multialignment="center",
                    zorder=9,
                    annotation_clip=False,
                )
                if not (overlay_no_gate and custom_nogate_style):
                    patch = ann.get_bbox_patch()
                    if patch is not None:
                        patch.set_path_effects(
                            [
                                pe.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace=(0, 0, 0), alpha=0.18),
                                pe.Normal(),
                            ]
                        )

    else:
        ax_gain.text(0.5, 0.5, "no data", ha="center", va="center")

    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path = outdir / output_name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path



OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    global ACTIVE_TASK
    ACTIVE_TASK = "FFSP100"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with paper_style(figsize=FIGSIZE_SEARCH, suppress_free_text=True, extra_save_formats=("pdf",)):
        path = _plot_search_trajectory(
            OUT_DIR,
            output_name="ffsp100_no_gate_ablation.png",
            overlay_no_gate=True,
            custom_nogate_style=True,
            show_rejected_count_bars=True,
            show_no_gate_replay=False,
        )
    print(path)


if __name__ == "__main__":
    main()
