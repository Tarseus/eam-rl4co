from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

_mplconfigdir = (Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from style import FIGSIZE_ABLATION, paper_style
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]

FAMILY_AWARE_RUN = ROOT / "runs/pref_loss_ffsp100_discovery/20260403-142801"
FAMILY_OFF_RUN = ROOT / "runs/pref_loss_ffsp100_discovery_family_off/20260503-030418"
WEIGHT_ONLY_RUN = ROOT / "runs/pref_builder_weight_search_ffsp100/20260416-111514"
GEOMETRY_WEIGHT_RUN = ROOT / "runs/pref_builder_geometry_weight_search_ffsp100_ablation/20260504-200255"

OBJECTIVE_OUT = Path(__file__).resolve().parent
BUILDER_OUT = Path(__file__).resolve().parent

FULL_PAIR_COUNT = 960.0
FFSP_FAMILY_DISCOVERY_THRESHOLD = -1.25
FFSP_FAMILY_MISSING_VALUE = -8.0
FFSP_FAMILY_PLOT_FLOOR = -4.0

COLORS = {
    "family_aware": "#1d4ed8",
    "family_off": "#d97706",
    "weight_only": "#D55E00",
    "geometry_full": "#E69F00",
    "geometry_nonfull": "#CC79A7",
    "grid": "#ded8cf",
    "bg": "#fbfaf7",
    "text": "#2f2d2a",
}

FAMILY_ORDER = [
    "Cost / Pairwise margin",
    "Rank / Pairwise margin",
    "Rank / Rank-weighted",
    "Rank / Rank-weighted norm",
    "Rank / Rank-weighted scale",
    "Rank / Adv-weighted norm",
    "Rank / Margin clipped",
    "Regret / Pairwise margin",
    "Advantage / Pairwise margin",
    "Advantage / Logistic",
    "Prob / Rank-prob",
    "Other / Pairwise margin",
]
FAMILY_COLORS = {
    "Cost / Pairwise margin": "#7c8798",
    "Rank / Pairwise margin": "#0072b2",
    "Rank / Rank-weighted": "#009e73",
    "Rank / Rank-weighted norm": "#8b5cf6",
    "Rank / Rank-weighted scale": "#56b4e9",
    "Rank / Adv-weighted norm": "#2ca02c",
    "Rank / Margin clipped": "#b79f00",
    "Regret / Pairwise margin": "#cc79a7",
    "Advantage / Pairwise margin": "#e69f00",
    "Advantage / Logistic": "#8c564b",
    "Prob / Rank-prob": "#d55e00",
    "Other / Pairwise margin": "#4b5563",
}
EXTRA_FAMILY_COLORS = [
    "#00a6a6",
    "#6a3d9a",
    "#f781bf",
    "#a6761d",
    "#66a61e",
    "#17becf",
    "#e7298a",
    "#1b9e77",
    "#7570b3",
    "#bcbd22",
    "#fb8072",
    "#80b1d3",
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.4,
        "axes.titlesize": 12.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.8,
        "legend.fontsize": 8.8,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict at {path}")
    return payload


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _score(row: Mapping[str, Any], *, prefer_estimated: bool = False) -> float | None:
    keys = ("score_hf20_estimated", "score", "final_score") if prefer_estimated else ("score", "final_score")
    for key in keys:
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _stage(row: Mapping[str, Any]) -> str:
    return str(row.get("stage_final") or row.get("stage") or "")


def _pair_count(row: Mapping[str, Any]) -> float | None:
    trace = row.get("builder_gate_trace")
    if not isinstance(trace, Mapping):
        return None
    checks = trace.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, Mapping) and check.get("metric_name") == "pair_count":
            return _finite_float(check.get("observed_value"))
    return None


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.replace("-", "_").replace(" ", "_").replace(":", "_")


def _parse_signature(signature: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(signature or "").split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _primary_signal(*values: Any) -> str:
    text = " ".join(_normalize_text(v) for v in values)
    if any(key in text for key in ("rankprob", "rank_prob", "probabilistic", "probability", "bayesian")):
        return "Prob"
    if "rank" in text:
        return "Rank"
    if "regret" in text:
        return "Regret"
    if "cost" in text:
        return "Cost"
    if "advantage" in text:
        return "Advantage"
    return "Other"


def _paradigm_variant(value: Any) -> str:
    raw = _normalize_text(value)
    if "rankprob" in raw or "rank_prob" in raw:
        return "Rank-prob"
    if "logsigmoid" in raw or "logistic" in raw:
        return "Logistic"
    if "advantage_weighted" in raw:
        return "Adv-weighted norm" if "normal" in raw else "Adv-weighted"
    if "rank_weight" in raw or "ranked" in raw or "rank_gap" in raw:
        if "scale" in raw:
            return "Rank-weighted scale"
        if "normal" in raw:
            return "Rank-weighted norm"
        return "Rank-weighted"
    if "with_clipping" in raw or "clipping" in raw:
        return "Margin clipped"
    if "pairwise_margin" in raw or "pairwise_comparison" in raw:
        return "Pairwise margin"
    return raw.replace("pairwise_", "").replace("_", " ").title()


def _loss_family_from_ir(ir: Mapping[str, Any] | None, signature: str | None = None) -> str:
    hp = (ir or {}).get("hyperparams") if isinstance(ir, Mapping) else {}
    if not isinstance(hp, Mapping):
        hp = {}
    sig = _parse_signature(signature)
    paradigm = hp.get("paradigm_family") or sig.get("paradigm")
    signal = hp.get("signal_family") or sig.get("signal")
    link = hp.get("link_family") or sig.get("link")
    constraint = hp.get("constraint_family") or sig.get("constraint")
    name = (ir or {}).get("name") if isinstance(ir, Mapping) else None
    return f"{_primary_signal(signal, paradigm, name, link, constraint)} / {_paradigm_variant(paradigm)}"


def _loss_meta(run_dir: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in _iter_jsonl(run_dir / "losses.jsonl"):
        fid = row.get("id")
        if not fid:
            continue
        signature = str(row.get("family_signature") or row.get("family") or fid)
        out[str(fid)] = {
            "family": _loss_family_from_ir(row.get("ir"), signature),
            "signature": signature,
        }
    return out


def _loss_pair_rows(run_dir: Path) -> list[dict[str, Any]]:
    meta = _loss_meta(run_dir)
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(run_dir / "pairs.jsonl"):
        if _stage(row) != "high_fidelity":
            continue
        score = _score(row)
        gen = row.get("generation")
        if score is None or not isinstance(gen, int):
            continue
        fid = row.get("f_id") or row.get("f_id_after_repair") or row.get("f_id_before_repair") or ""
        family = meta.get(str(fid), {}).get("family") or _loss_family_from_ir(row.get("f_ir"))
        rows.append(
            {
                "generation": int(gen),
                "score": float(score),
                "improvement": -float(score),
                "id": str(fid),
                "family": family,
            }
        )
    return rows


def _best_loss_row(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "best_loss.json"
    if not path.exists():
        return None
    meta = _loss_meta(run_dir)
    row = _load_json(path)
    score = _finite_float(row.get("score"))
    if score is None:
        return None
    fid = str(row.get("id") or "")
    gen = row.get("generation", row.get("best_pair_generation", -1))
    return {
        "generation": int(gen) if isinstance(gen, int) else -1,
        "score": score,
        "improvement": -score,
        "id": fid,
        "family": meta.get(fid, {}).get("family") or _loss_family_from_ir(row.get("ir")),
    }


def _best_so_far_improvement(rows: list[Mapping[str, Any]], max_gen: int) -> tuple[np.ndarray, np.ndarray]:
    by_gen: dict[int, list[float]] = {g: [] for g in range(max_gen + 1)}
    for row in rows:
        gen = int(row["generation"])
        if 0 <= gen <= max_gen:
            by_gen[gen].append(float(row["improvement"]))
    xs = np.asarray(list(range(max_gen + 1)), dtype=float)
    ys: list[float] = []
    running = -math.inf
    for gen in range(max_gen + 1):
        vals = by_gen.get(gen) or []
        if vals:
            running = max(running, max(vals))
        ys.append(float(running) if math.isfinite(running) else np.nan)
    return xs, np.asarray(ys, dtype=float)


def _best_so_far_with_missing_value(
    rows: list[Mapping[str, Any]],
    max_gen: int,
    *,
    missing_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = _best_so_far_improvement(rows, max_gen)
    missing = np.isnan(ys)
    ys_plot = ys.copy()
    ys_plot[missing] = float(missing_value)
    return xs, ys_plot, missing


def _best_so_far_score(rows: list[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["generation"])].append(float(row["score"]))
    if not grouped:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    xs = list(range(min(grouped), max(grouped) + 1))
    running = math.inf
    ys: list[float] = []
    for gen in xs:
        vals = grouped.get(gen) or []
        if vals:
            running = min(running, min(vals))
        ys.append(float(running) if math.isfinite(running) else np.nan)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _line_shadow(line: Any, *, alpha: float = 0.18) -> None:
    line.set_path_effects(
        [
            pe.SimpleLineShadow(offset=(1.2, -1.2), shadow_color="#111111", alpha=alpha),
            pe.Normal(),
        ]
    )


def _jitter(xs: list[float], *, seed: int, scale: float = 0.055) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x + rng.normal(0.0, scale)) for x in xs]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_ffsp_family_ablation() -> tuple[Path, Path, Path]:
    OBJECTIVE_OUT.mkdir(parents=True, exist_ok=True)
    records = {
        "Family-aware": {"run": FAMILY_AWARE_RUN, "color": COLORS["family_aware"], "marker": "o", "jitter": -0.055},
        "Family-disabled": {"run": FAMILY_OFF_RUN, "color": COLORS["family_off"], "marker": "D", "jitter": 0.055},
    }
    max_gen = 0
    summary_rows: list[dict[str, Any]] = []
    for label, rec in records.items():
        pairs = _loss_pair_rows(rec["run"])
        best = _best_loss_row(rec["run"])
        if best is not None and all(
            not (
                p["generation"] == best["generation"]
                and p["id"] == best["id"]
                and abs(float(p["score"]) - float(best["score"])) < 1e-12
            )
            for p in pairs
        ):
            pairs.append(best)
        rec["pairs"] = pairs
        rec["best"] = best
        max_gen = max(max_gen, max((int(p["generation"]) for p in pairs), default=0))
        positive = [p for p in pairs if p["generation"] > 0 and p["improvement"] >= FFSP_FAMILY_DISCOVERY_THRESHOLD]
        summary_rows.append(
            {
                "method": label,
                "best_score": "" if best is None else f"{best['score']:.12g}",
                "best_discovery_score": "" if best is None else f"{best['improvement']:.12g}",
                "best_generation": "" if best is None else best["generation"],
                "discovery_ge_threshold_candidates": len(positive),
                "discovery_ge_threshold_families": len({p["family"] for p in positive}),
                "discovery_threshold": FFSP_FAMILY_DISCOVERY_THRESHOLD,
                "discovery_score_definition": "-raw_score",
                "hf_candidates": len(pairs),
            }
        )

    fig, ax = plt.subplots(figsize=(8.9, 4.95))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.72, alpha=0.58)
    ax.grid(axis="x", color="#e8e1d3", lw=0.52, alpha=0.32)

    method_handles = []
    present_families = sorted(
        {
            p["family"]
            for rec in records.values()
            for p in rec["pairs"]
            if p["generation"] > 0 and p["improvement"] >= FFSP_FAMILY_DISCOVERY_THRESHOLD
        }
    )
    plot_family_order = [family for family in FAMILY_ORDER if family in present_families]
    overflow_families = [family for family in present_families if family not in plot_family_order]
    plot_family_order.extend(overflow_families)
    plot_family_colors = dict(FAMILY_COLORS)
    for idx, family in enumerate(overflow_families):
        plot_family_colors[family] = EXTRA_FAMILY_COLORS[idx % len(EXTRA_FAMILY_COLORS)]
    rng = np.random.default_rng(19)
    for label, rec in records.items():
        color = rec["color"]
        marker = rec["marker"]
        pairs = rec["pairs"]
        display = [p for p in pairs if p["generation"] > 0]
        if display:
            xs = np.asarray([p["generation"] for p in display], dtype=float)
            ys = np.asarray([p["improvement"] for p in display], dtype=float)
            jitter = float(rec["jitter"]) + rng.normal(0, 0.032, size=len(xs))
            for family in plot_family_order:
                idx = np.asarray([p["family"] == family for p in display], dtype=bool)
                if not np.any(idx):
                    continue
                ax.scatter(
                    xs[idx] + jitter[idx],
                    ys[idx],
                    s=52 if label == "Family-aware" else 38,
                    marker=marker,
                    facecolor=plot_family_colors.get(family, "#94a3b8"),
                    edgecolor=color,
                    linewidth=0.7,
                    alpha=0.78 if label == "Family-aware" else 0.42,
                    zorder=6,
                )
        xs, ys, missing_best = _best_so_far_with_missing_value(
            pairs,
            max_gen,
            missing_value=FFSP_FAMILY_MISSING_VALUE,
        )
        ax.step(xs, ys, where="post", color="white", lw=6.2, alpha=0.94, zorder=2, solid_capstyle="round")
        line = ax.step(
            xs,
            ys,
            where="post",
            color=color,
            lw=2.75,
            marker=marker,
            markersize=4.9,
            markerfacecolor="white",
            markeredgewidth=1.35,
            label=label,
            zorder=3,
            solid_capstyle="round",
        )[0]
        _line_shadow(line, alpha=0.14)
        method_handles.append(line)
        best = rec["best"]
        if best is not None:
            ax.scatter(
                [best["generation"]],
                [best["improvement"]],
                s=92,
                marker=marker,
                facecolor=plot_family_colors.get(best["family"], "#d6d3d1"),
                edgecolor=color,
                linewidth=1.5,
                zorder=7,
            )

    ax.axhline(FFSP_FAMILY_DISCOVERY_THRESHOLD, color="#2d2a24", lw=0.85, alpha=0.58)
    ax.set_title("FFSP100 family ablation", loc="left", pad=7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Discovery score (-raw score)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.45, max_gen + 0.45)
    best_improvements = [
        float(rec["best"]["improvement"])
        for rec in records.values()
        if rec.get("best") is not None and rec["best"]["improvement"] is not None
    ]
    upper = max([0.001] + [v for v in best_improvements if v > 0]) * 1.15
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.75)
    ax.set_ylim(FFSP_FAMILY_PLOT_FLOOR, upper)
    ax.set_yticks([-4, -2, -1, -0.5, 0, 0.25, 0.75, 1.25])
    ax.set_yticklabels(["-4", "-2", "-1", "-0.5", "0", "0.25", "0.75", "1.25"])
    method_legend = ax.legend(
        handles=method_handles,
        loc="upper left",
        frameon=True,
        facecolor=COLORS["bg"],
        edgecolor=COLORS["grid"],
        title="Best-so-far",
    )
    ax.add_artist(method_legend)
    family_handles = [
        Patch(facecolor=plot_family_colors[family], edgecolor="none", label=family)
        for family in plot_family_order
    ]
    if family_handles:
        ax.legend(
            handles=family_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(4, len(family_handles)),
            frameon=False,
            columnspacing=1.0,
            handlelength=1.0,
            fontsize=8.2,
        )
    fig.tight_layout(rect=(0.02, 0.14, 0.995, 0.98))
    png = OBJECTIVE_OUT / "ffsp100_no_family_ablation.png"
    pdf = OBJECTIVE_OUT / "ffsp100_no_family_ablation.pdf"
    csv_path = OBJECTIVE_OUT / "ffsp100_no_family_ablation_summary.csv"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    _write_csv(csv_path, summary_rows)
    return png, pdf, csv_path


def _builder_rows(run_dir: Path, *, method: str, prefer_estimated: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = run_dir / ("pairs_hf20_estimated_from_hf3.jsonl" if prefer_estimated else "pairs.jsonl")
    if not path.exists():
        path = run_dir / "pairs.jsonl"
    for row in _iter_jsonl(path):
        if _stage(row) != "high_fidelity":
            continue
        score = _score(row, prefer_estimated=prefer_estimated)
        gen = row.get("generation")
        if score is None or not isinstance(gen, int):
            continue
        pair_count = _pair_count(row)
        family = "full_pair" if pair_count is not None and abs(pair_count - FULL_PAIR_COUNT) <= 1e-6 else "non_full_pair"
        rows.append(
            {
                "method": method,
                "generation": int(gen),
                "score": float(score),
                "discovery_score": -float(score),
                "pair_count": pair_count,
                "family": family,
                "g_id": row.get("g_id") or "",
                "pair_index": row.get("pair_index"),
            }
        )
    return rows


def _summary_by_generation(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["generation"])].append(float(row["discovery_score"]))
    out: list[dict[str, Any]] = []
    running = -math.inf
    for gen in sorted(grouped):
        vals = sorted(grouped[gen])
        running = max(running, max(vals))
        out.append(
            {
                "method": method,
                "generation": gen,
                "n": len(vals),
                "max_discovery_score": f"{max(vals):.12g}",
                "median_discovery_score": f"{median(vals):.12g}",
                "mean_discovery_score": f"{mean(vals):.12g}",
                "q25_discovery_score": f"{np.quantile(vals, 0.25):.12g}",
                "q75_discovery_score": f"{np.quantile(vals, 0.75):.12g}",
                "best_so_far": f"{running:.12g}",
            }
        )
    return out


def _best_so_far_discovery(rows: list[Mapping[str, Any]], *, max_gen: int = 9) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["generation"])].append(float(row["discovery_score"]))
    xs = list(range(0, max_gen + 1))
    running = -math.inf
    ys: list[float] = []
    for gen in xs:
        vals = grouped.get(gen) or []
        if vals:
            running = max(running, max(vals))
        ys.append(float(running) if math.isfinite(running) else np.nan)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plot_ffsp_builder_geometry_weight_ablation() -> tuple[Path, Path, Path]:
    BUILDER_OUT.mkdir(parents=True, exist_ok=True)
    weight_rows = _builder_rows(WEIGHT_ONLY_RUN, method="full_pair_reweight_only", prefer_estimated=False)
    geometry_rows = _builder_rows(GEOMETRY_WEIGHT_RUN, method="geometry_plus_weight", prefer_estimated=True)
    geometry_full = [row for row in geometry_rows if row["family"] == "full_pair"]
    geometry_nonfull = [row for row in geometry_rows if row["family"] != "full_pair"]
    wx, wy = _best_so_far_discovery(weight_rows, max_gen=9)
    gfx, gfy = _best_so_far_discovery(geometry_full, max_gen=9)
    gnfx, gnfy = _best_so_far_discovery(geometry_nonfull, max_gen=9)

    summary = []
    summary.extend(_summary_by_generation("full_pair_reweight_only", weight_rows))
    summary.extend(_summary_by_generation("geometry_plus_weight_full_pair", geometry_full))
    summary.extend(_summary_by_generation("geometry_plus_weight_non_full_pair", geometry_nonfull))
    summary_path = BUILDER_OUT / "ffsp_builder_geometry_weight_ablation_summary.csv"
    _write_csv(summary_path, summary)

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.82)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55, alpha=0.42)
    ax.axhline(0.0, color="#77736b", lw=1.0, ls=":", zorder=1)
    ax.fill_between([-0.5, 9.5], -0.23, 0.0, color=COLORS["weight_only"], alpha=0.05, zorder=0)
    ax.fill_between([-0.5, 9.5], 0.0, 14.0, color=COLORS["geometry_nonfull"], alpha=0.04, zorder=0)

    def scatter(rows: list[dict[str, Any]], color: str, marker: str, seed: int, label: str) -> None:
        xs = [float(row["generation"]) for row in rows]
        ys = [float(row["discovery_score"]) for row in rows]
        ax.scatter(
            _jitter(xs, seed=seed),
            ys,
            s=18,
            color=color,
            alpha=0.28,
            marker=marker,
            linewidths=0,
            label=label,
            zorder=2,
        )

    scatter(weight_rows, COLORS["weight_only"], "D", 13, "Full-pair reweight-only candidates")
    scatter(geometry_full, COLORS["geometry_full"], "s", 17, "Geometry+weight full-pair candidates")
    scatter(geometry_nonfull, COLORS["geometry_nonfull"], "o", 19, "Geometry+weight non-full-pair candidates")

    if len(gnfx):
        line = ax.step(
            gnfx,
            gnfy,
            where="post",
            color=COLORS["geometry_nonfull"],
            lw=2.35,
            ls="--",
            label="Best-so-far: geometry+weight non-full-pair",
            zorder=4,
        )[0]
        _line_shadow(line, alpha=0.14)
    if len(gfx):
        line = ax.step(
            gfx,
            gfy,
            where="post",
            color=COLORS["geometry_full"],
            lw=2.45,
            ls="-.",
            label="Best-so-far: geometry+weight full-pair",
            zorder=5,
        )[0]
        _line_shadow(line, alpha=0.14)
    if len(wx):
        line = ax.step(
            wx,
            wy,
            where="post",
            color=COLORS["weight_only"],
            lw=2.8,
            label="Best-so-far: full-pair reweight-only",
            zorder=6,
        )[0]
        _line_shadow(line, alpha=0.2)

    all_scores = [row["discovery_score"] for row in weight_rows + geometry_rows]
    lower = min(all_scores + [-1.0])
    upper = max(all_scores + [0.2])
    ax.set_xlim(-0.45, 9.45)
    ax.set_ylim(max(-14.0, lower * 1.08), upper * 1.12)
    ax.set_yscale("symlog", linthresh=0.055, linscale=0.75)
    ax.set_yticks([-10.0, -5.0, -1.0, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])
    ax.set_yticklabels(["-10", "-5", "-1", "-0.1", "-0.05", "0", "0.05", "0.1", "0.2"])
    ax.set_xticks(range(10))
    ax.set_xlabel("Builder-branch generation")
    ax.set_ylabel("Discovery score (-score)", labelpad=6)
    ax.set_title("FFSP100: opening pair-geometry search hurts compared with fixed all-pairs")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        facecolor=COLORS["bg"],
        edgecolor=COLORS["grid"],
    )
    pair_counts = Counter(row["pair_count"] for row in geometry_rows)
    fig.text(
        0.01,
        0.012,
        (
            "Geometry+weight uses hf3 scores calibrated to hf20 with offset -0.0004995. "
            f"Geometry pair-count distribution: {dict(pair_counts)}."
        ),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#5d5850",
    )
    fig.subplots_adjust(left=0.155, right=0.985, top=0.88, bottom=0.34)
    png = BUILDER_OUT / "ffsp_builder_geometry_weight_ablation.png"
    pdf = BUILDER_OUT / "ffsp_builder_geometry_weight_ablation.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return png, pdf, summary_path



def main() -> None:
    with paper_style(figsize=FIGSIZE_ABLATION, extra_save_formats=("svg",)):
        png, pdf, summary = plot_ffsp_family_ablation()
    print(png)
    print(pdf)
    print(summary)


if __name__ == "__main__":
    main()
