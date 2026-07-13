from __future__ import annotations

import json
import math
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

_mplconfigdir = (Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "interpretability_four_problems"

TASKS = OrderedDict(
    [
        (
            "TSP100",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_tsp100_discovery/20260317-131507",
                "pair_run": REPO_ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757",
            },
        ),
        (
            "CVRP100",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008",
                "pair_run": REPO_ROOT / "runs/pref_builder_weight_search_cvrp100/20260416-093909",
            },
        ),
        (
            "FFSP100",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_ffsp100_discovery/20260403-142801",
                "pair_run": REPO_ROOT / "runs/pref_builder_weight_search_ffsp100/20260416-111514",
            },
        ),
        (
            "JSSP10x10",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409",
                "pair_run": REPO_ROOT
                / "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033",
            },
        ),
    ]
)

COLORS = {
    "loss": "#0072B2",
    "pair": "#D55E00",
    "accepted": "#009E73",
    "rejected": "#CC79A7",
    "neutral": "#6B7280",
    "bg": "#fbfaf7",
    "grid": "#e5dfd2",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.3,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _best_artifact(run_dir: Path, candidates: tuple[str, ...]) -> dict[str, Any]:
    for name in candidates:
        path = run_dir / name
        if path.is_file():
            payload = _read_json(path)
            score = _finite_float(payload.get("score"))
            ref = _finite_float(payload.get("reference_score"))
            histories = (
                payload.get("score_history"),
                payload.get("best_pair_score_history"),
            )
            if ref is None:
                for history in histories:
                    if isinstance(history, list) and history and isinstance(history[0], Mapping):
                        ref = _finite_float(history[0].get("reference_score"))
                        if ref is not None:
                            break
            ir = payload.get("ir") if isinstance(payload.get("ir"), Mapping) else {}
            return {
                "path": path,
                "score": score,
                "reference": ref,
                "gain": (ref - score) if ref is not None and score is not None else None,
                "id": str(payload.get("id") or ""),
                "name": str(ir.get("name") or payload.get("name") or ""),
                "intuition": str(ir.get("intuition") or ""),
                "hyperparams": dict(ir.get("hyperparams") or {}),
                "operators": [str(x) for x in (ir.get("operators_used") or [])],
            }
    raise FileNotFoundError(f"No best artifact found in {run_dir}")


def _load_hf_points(run_dir: Path, phase: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in _iter_jsonl(run_dir / "pairs.jsonl") or []:
        if str(row.get("phase") or phase) != phase:
            continue
        stage = str(row.get("stage_final") or row.get("stage") or "")
        if stage != "high_fidelity":
            continue
        score = _finite_float(row.get("score") if row.get("score") is not None else row.get("final_score"))
        ref = _finite_float(row.get("reference_score"))
        gen = row.get("generation")
        if score is None or ref is None or not isinstance(gen, int):
            continue
        points.append(
            {
                "generation": int(gen),
                "score": score,
                "reference": ref,
                "gain": ref - score,
                "better": bool(row.get("better_than_incumbent") or row.get("better_than_last_phase")),
            }
        )
    return points


def _funnel_counts(run_dir: Path, phase: str) -> dict[str, int]:
    counts = {
        "generated": 0,
        "compiled": 0,
        "semantic": 0,
        "co_sensitive": 0,
        "high_fidelity": 0,
        "better": 0,
    }
    for row in _iter_jsonl(run_dir / "pairs.jsonl") or []:
        if str(row.get("phase") or phase) != phase:
            continue
        if not isinstance(row.get("generation"), int) or int(row.get("generation")) < 0:
            continue
        counts["generated"] += 1
        g_compile = row.get("g_compile_ok")
        f_compile = row.get("f_compile_ok")
        if g_compile is False or f_compile is False:
            continue
        counts["compiled"] += 1
        builder_ok = row.get("builder_gate_ok")
        joint_ok = row.get("joint_gate_ok")
        sandbox_ok = row.get("sandbox_gate_ok")
        semantic_ok = bool(sandbox_ok) or (builder_ok is not False and joint_ok is not False)
        if not semantic_ok:
            continue
        counts["semantic"] += 1
        co_ok = row.get("co_ok")
        if co_ok is False:
            continue
        counts["co_sensitive"] += 1
        stage = str(row.get("stage_final") or row.get("stage") or "")
        if stage == "high_fidelity":
            counts["high_fidelity"] += 1
        if bool(row.get("better_than_incumbent") or row.get("better_than_last_phase")):
            counts["better"] += 1
    return counts


def _best_so_far(points: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for point in points:
        grouped[int(point["generation"])].append(float(point["gain"]))
    xs: list[int] = []
    ys: list[float] = []
    best: float | None = None
    for gen in range(max(grouped.keys(), default=-1) + 1):
        vals = grouped.get(gen) or []
        if vals:
            current = max(vals)
            best = current if best is None else max(best, current)
        if best is not None:
            xs.append(gen)
            ys.append(best)
    return xs, ys


def _collect() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for task, paths in TASKS.items():
        loss_run = paths["loss_run"]
        pair_run = paths["pair_run"]
        data[task] = {
            "loss": _best_artifact(loss_run, ("best_loss.json", "best_pair.json")),
            "pair": _best_artifact(pair_run, ("best_builder.json", "best_pair.json", "best_elite_builder.json")),
            "loss_points": _load_hf_points(loss_run, "loss"),
            "pair_points": _load_hf_points(pair_run, "builder"),
            "loss_funnel": _funnel_counts(loss_run, "loss"),
            "pair_funnel": _funnel_counts(pair_run, "builder"),
        }
    return data


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.75, alpha=0.75)


def _plot_stage_gain(data: Mapping[str, Mapping[str, Any]]) -> Path:
    tasks = list(data)
    x = np.arange(len(tasks))
    width = 0.35
    loss_vals = [float(data[t]["loss"]["gain"] or 0.0) for t in tasks]
    pair_vals = [float(data[t]["pair"]["gain"] or 0.0) for t in tasks]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_axis(ax)
    bars_loss = ax.bar(x - width / 2, loss_vals, width, color=COLORS["loss"], label="Best loss vs loss baseline")
    bars_pair = ax.bar(x + width / 2, pair_vals, width, color=COLORS["pair"], label="Best pair builder vs pair baseline")
    positive_vals = [v for v in loss_vals + pair_vals if v > 0]
    ymin = max(min(positive_vals) * 0.45, 1e-5) if positive_vals else 1e-5
    ymax = max(positive_vals) * 1.85 if positive_vals else 1.0
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("Reference score - candidate score\nlog scale, higher is better")
    ax.set_title("Stage gain across four problems")
    ax.legend(frameon=True, facecolor=COLORS["bg"], edgecolor=COLORS["grid"], loc="upper left")

    for bars in (bars_loss, bars_pair):
        for bar in bars:
            h = bar.get_height()
            if h <= 0:
                continue
            va = "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.09,
                f"{h:.3g}",
                ha="center",
                va=va,
                fontsize=8.8,
            )
    fig.text(
        0.5,
        -0.02,
        "Scores are minimized in the logs; positive bars mean the discovered loss or pair builder beats its recorded reference.",
        ha="center",
        fontsize=9.2,
        color="#55524c",
    )
    fig.tight_layout()
    path = OUT_DIR / "01_stage_gain_four_problems.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_search_distributions(data: Mapping[str, Mapping[str, Any]]) -> Path:
    fig, axes = plt.subplots(4, 2, figsize=(11.2, 12.0), sharey=False)
    fig.patch.set_facecolor(COLORS["bg"])
    rng = np.random.default_rng(11)

    for row_idx, task in enumerate(data):
        for col_idx, (phase, label, color) in enumerate(
            [("loss_points", "Loss candidates", COLORS["loss"]), ("pair_points", "Pair-builder candidates", COLORS["pair"])]
        ):
            ax = axes[row_idx, col_idx]
            _style_axis(ax)
            points = data[task][phase]
            if not points:
                ax.text(0.5, 0.5, "no high-fidelity points", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{task}: {label}")
                continue
            generations = sorted({int(p["generation"]) for p in points if int(p["generation"]) >= 0})
            values_by_gen = [[float(p["gain"]) for p in points if int(p["generation"]) == gen] for gen in generations]
            positions = np.arange(len(generations))
            ax.boxplot(
                values_by_gen,
                positions=positions,
                widths=0.56,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "lw": 1.3},
                boxprops={"facecolor": color, "edgecolor": "#333333", "alpha": 0.28},
                whiskerprops={"color": "#666666"},
                capprops={"color": "#666666"},
            )
            for pos, vals in zip(positions, values_by_gen):
                jitter = rng.normal(0, 0.055, size=len(vals))
                ax.scatter(pos + jitter, vals, s=18, color=color, alpha=0.45, linewidths=0)
            xs, ys = _best_so_far(points)
            pos_lookup = {gen: i for i, gen in enumerate(generations)}
            line_x = [pos_lookup[g] for g in xs if g in pos_lookup]
            line_y = [y for g, y in zip(xs, ys) if g in pos_lookup]
            ax.plot(line_x, line_y, color="#111111", lw=2.0, marker="o", ms=3.8, label="best so far")
            ax.axhline(0, color="#555555", lw=0.85, alpha=0.75)
            ax.set_xticks(positions, [str(g) for g in generations])
            ax.set_title(f"{task}: {label}")
            ax.set_xlabel("generation")
            if col_idx == 0:
                ax.set_ylabel("improvement")
            ax.legend(frameon=False, loc="best")
    fig.suptitle("High-fidelity candidate distributions reveal why the selected designs are better", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = OUT_DIR / "02_hf_candidate_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_funnel(data: Mapping[str, Mapping[str, Any]]) -> Path:
    stages = ["generated", "compiled", "semantic", "co_sensitive", "high_fidelity", "better"]
    labels = ["Generated", "Compiled", "Semantic", "CO-sensitive", "HF eval", "Better"]
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.6), sharey=True)
    fig.patch.set_facecolor(COLORS["bg"])
    axes = axes.ravel()
    for idx, task in enumerate(data):
        for offset, (phase_key, title, color) in enumerate(
            [("loss_funnel", "loss", COLORS["loss"]), ("pair_funnel", "pair", COLORS["pair"])]
        ):
            ax = axes[idx + offset * 4]
            _style_axis(ax)
            vals = [data[task][phase_key].get(stage, 0) for stage in stages]
            ax.plot(range(len(stages)), vals, marker="o", color=color, lw=2.2)
            ax.fill_between(range(len(stages)), vals, color=color, alpha=0.13)
            ax.set_yscale("log")
            ax.set_xticks(range(len(stages)), labels, rotation=35, ha="right")
            ax.set_title(f"{task} {title}")
            for x, y in enumerate(vals):
                if y > 0:
                    ax.text(x, y * 1.12, str(y), ha="center", va="bottom", fontsize=8.2)
            if idx == 0:
                ax.set_ylabel("candidate count (log)")
    fig.suptitle("Filtering funnels: useful improvements survive compile, semantic, CO, and HF checks", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = OUT_DIR / "03_filtering_funnels.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _short(value: Any, max_len: int = 24) -> str:
    text = str(value or "-").replace("_", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _plot_design_table(data: Mapping[str, Mapping[str, Any]]) -> Path:
    columns = [
        "Loss signal",
        "Loss link",
        "Loss aggregation",
        "Loss constraint",
        "Pair geometry",
        "Pair weight",
        "Pair constraint",
    ]
    rows = []
    for task in data:
        lhp = data[task]["loss"]["hyperparams"]
        php = data[task]["pair"]["hyperparams"]
        rows.append(
            [
                _short(lhp.get("signal_family")),
                _short(lhp.get("link_family")),
                _short(lhp.get("agg_family")),
                _short(lhp.get("constraint_family")),
                _short(php.get("geometry_family")),
                _short(php.get("weight_family")),
                _short(php.get("constraint_family")),
            ]
        )

    fig, ax = plt.subplots(figsize=(13.2, 4.4))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=list(data),
        colLabels=columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.85)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d0c4")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#ece7df")
            cell.set_text_props(weight="bold", color="#2d2a24")
        elif c == -1:
            cell.set_facecolor("#f4efe6")
            cell.set_text_props(weight="bold", color="#2d2a24")
        elif c <= 3:
            cell.set_facecolor("#e8f1fb")
        else:
            cell.set_facecolor("#fff0df")
    ax.set_title(
        "Best designs share the same explanation pattern: pairwise margin loss + dense instance-local pair reweighting",
        pad=18,
        fontweight="bold",
    )
    legend = [
        Patch(facecolor="#e8f1fb", edgecolor="#d7d0c4", label="loss design"),
        Patch(facecolor="#fff0df", edgecolor="#d7d0c4", label="pair-builder design"),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    path = OUT_DIR / "04_best_design_table.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(data: Mapping[str, Mapping[str, Any]], figures: list[Path]) -> Path:
    lines = [
        "# Four-problem interpretability visual analysis",
        "",
        "This report explains why the discovered loss and pair builders outperform their recorded baselines.",
        "All scores in the source logs are minimized, so the plots use `reference_score - score`; larger is better.",
        "",
        "## Figures",
    ]
    for path in figures:
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines.append(f"- `{rel}`")
    lines.extend(["", "## Main reading"])
    for task, row in data.items():
        loss_gain = row["loss"]["gain"]
        pair_gain = row["pair"]["gain"]
        lhp = row["loss"]["hyperparams"]
        php = row["pair"]["hyperparams"]
        lines.append(
            f"- {task}: loss gain={loss_gain:.6g}, pair gain={pair_gain:.6g}. "
            f"Loss uses {lhp.get('signal_family')} with {lhp.get('link_family')} and {lhp.get('constraint_family')}; "
            f"pair builder uses {php.get('geometry_family')} with {php.get('weight_family')} and {php.get('constraint_family')}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- The best losses are not generic scalar losses. They preserve pairwise preference direction and add objective-aware signals such as cost gap, delta rank, or advantage gap.",
            "- The best pair builders keep candidate pools dense and instance-local, then reweight pairs by gap, rank, margin, or regret so the trainer sees informative comparisons instead of uniform pairs.",
            "- The funnel plots show that improvements are not from arbitrary code generation. Candidates must pass compilation, semantic preference checks, CO sensitivity checks, and high-fidelity evaluation.",
            "- The distribution plots show that the selected designs are upper-tail candidates in high-fidelity evaluation, not isolated unverified claims.",
        ]
    )
    path = OUT_DIR / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _collect()
    figures = [
        _plot_stage_gain(data),
        _plot_search_distributions(data),
        _plot_funnel(data),
        _plot_design_table(data),
    ]
    report = _write_report(data, figures)
    manifest = {
        "output_dir": str(OUT_DIR),
        "figures": [str(p) for p in figures],
        "report": str(report),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
