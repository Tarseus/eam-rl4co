from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

_mplconfigdir = (Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "final_loss_weighting_analysis"

TASKS = OrderedDict(
    [
        (
            "TSP100",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_tsp100_discovery/20260317-131507",
                "weight_run": REPO_ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757",
            },
        ),
        (
            "CVRP100",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008",
                "weight_run": REPO_ROOT / "runs/pref_builder_weight_search_cvrp100/20260416-093909",
            },
        ),
        (
            "FFSP100",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_ffsp100_discovery/20260403-142801",
                "weight_run": REPO_ROOT / "runs/pref_builder_weight_search_ffsp100/20260416-111514",
            },
        ),
        (
            "JSSP10x10",
            {
                "loss_run": REPO_ROOT / "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409",
                "weight_run": REPO_ROOT
                / "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033",
            },
        ),
    ]
)

COLORS = {
    "loss": "#0072B2",
    "weight": "#D55E00",
    "po": "#009E73",
    "bopo": "#CC79A7",
    "bg": "#fbfaf7",
    "grid": "#e5dfd2",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.8,
        "axes.titlesize": 12.8,
        "axes.labelsize": 10.8,
        "legend.fontsize": 9.4,
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
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


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _best_artifact(run_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        path = run_dir / name
        if not path.is_file():
            continue
        payload = _read_json(path)
        score = _finite_float(payload.get("score"))
        reference = _finite_float(payload.get("reference_score"))
        for history_key in ("score_history", "best_pair_score_history"):
            history = payload.get(history_key)
            if reference is None and isinstance(history, list) and history and isinstance(history[0], Mapping):
                reference = _finite_float(history[0].get("reference_score"))
        ir = payload.get("ir") if isinstance(payload.get("ir"), Mapping) else {}
        hp = ir.get("hyperparams") if isinstance(ir.get("hyperparams"), Mapping) else {}
        return {
            "path": path,
            "id": str(payload.get("id") or ""),
            "name": str(ir.get("name") or payload.get("name") or ""),
            "score": score,
            "reference": reference,
            "gain": reference - score if score is not None and reference is not None else None,
            "hyperparams": dict(hp),
            "intuition": str(ir.get("intuition") or ""),
            "pseudocode": str(ir.get("pseudocode") or ""),
            "operators": [str(op) for op in (ir.get("operators_used") or [])],
        }
    raise FileNotFoundError(f"Missing best artifact in {run_dir}")


def _collect() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task, paths in TASKS.items():
        out[task] = {
            "loss": _best_artifact(paths["loss_run"], ("best_loss.json", "best_pair.json")),
            "weight": _best_artifact(paths["weight_run"], ("best_builder.json", "best_pair.json", "best_elite_builder.json")),
        }
    return out


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.75, alpha=0.75)


def _plot_final_gains(data: Mapping[str, Mapping[str, Any]]) -> Path:
    tasks = list(data)
    x = np.arange(len(tasks))
    width = 0.34
    loss_gain = [float(data[t]["loss"]["gain"] or 0.0) for t in tasks]
    weight_gain = [float(data[t]["weight"]["gain"] or 0.0) for t in tasks]
    vals = [v for v in loss_gain + weight_gain if v > 0]

    fig, ax = plt.subplots(figsize=(10.8, 4.7))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(max(min(vals) * 0.45, 1e-5), max(vals) * 1.8)
    bars_loss = ax.bar(x - width / 2, loss_gain, width, color=COLORS["loss"], label="Final loss")
    bars_weight = ax.bar(x + width / 2, weight_gain, width, color=COLORS["weight"], label="Final weighting")
    ax.set_xticks(x, tasks)
    ax.set_ylabel("reference score - final score\nlog scale, higher is better")
    ax.set_title("Final-only comparison: loss and weighting both improve over their baselines")
    ax.legend(frameon=True, facecolor=COLORS["bg"], edgecolor=COLORS["grid"], loc="upper left")
    for bars in (bars_loss, bars_weight):
        for bar in bars:
            h = bar.get_height()
            if h <= 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.09, f"{h:.3g}", ha="center", va="bottom", fontsize=9.2)
    fig.text(
        0.5,
        -0.02,
        "Only final artifacts are shown. Scores in logs are minimized, so positive gain means better than the recorded reference.",
        ha="center",
        fontsize=9.3,
        color="#55524c",
    )
    fig.tight_layout()
    path = OUT_DIR / "01_final_loss_weighting_gain.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _short(text: Any, limit: int = 22) -> str:
    value = str(text or "-").replace("_", " ")
    return value if len(value) <= limit else value[: limit - 1] + "..."


def _plot_final_designs(data: Mapping[str, Mapping[str, Any]]) -> Path:
    columns = [
        "Loss signal",
        "Loss link",
        "Loss aggregation",
        "Loss constraint",
        "Weight geometry",
        "Weight rule",
        "Weight constraint",
    ]
    rows = []
    for task in data:
        lhp = data[task]["loss"]["hyperparams"]
        whp = data[task]["weight"]["hyperparams"]
        rows.append(
            [
                _short(lhp.get("signal_family")),
                _short(lhp.get("link_family")),
                _short(lhp.get("agg_family")),
                _short(lhp.get("constraint_family")),
                _short(whp.get("geometry_family")),
                _short(whp.get("weight_family")),
                _short(whp.get("constraint_family")),
            ]
        )

    fig, ax = plt.subplots(figsize=(13.2, 4.35))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.axis("off")
    table = ax.table(cellText=rows, rowLabels=list(data), colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    table.scale(1.0, 1.8)
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
    ax.set_title("Final mechanism: objective-aware pairwise loss + dense pair weighting", pad=17, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="#e8f1fb", edgecolor="#d7d0c4", label="loss"),
            Patch(facecolor="#fff0df", edgecolor="#d7d0c4", label="weighting"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    path = OUT_DIR / "02_final_loss_weighting_designs.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_paper_style_comparison() -> Path:
    columns = ["PO4COPs", "BOPO", "This final analysis"]
    rows = [
        [
            "All sampled-solution pairs\nby reward preference",
            "Best-anchored pairs\nafter hybrid rollout + filtering",
            "Keep final dense pairs;\nanalyze final weighting only",
        ],
        [
            "BT-style preference loss\non log-prob gaps",
            "Objective-guided pairwise loss\nwith adaptive scaling",
            "Compare final discovered loss\nagainst its reference loss",
        ],
        [
            "No learned weighting;\npair mask is preference indicator",
            "Implicit emphasis via best anchor\nand objective scale",
            "Explicit final weighting:\ngap/rank/margin/regret rules",
        ],
        [
            "Final quality + convergence\nvs RL/SLL baselines",
            "Final benchmark tables;\ncomponent and loss ablations;\ngradient analysis",
            "Final loss gain;\nfinal weighting gain;\nmechanism table",
        ],
    ]
    row_labels = ["Pair construction", "Loss", "Weighting", "Evidence"]

    fig, ax = plt.subplots(figsize=(12.4, 5.25))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.axis("off")
    table = ax.table(cellText=rows, rowLabels=row_labels, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 2.15)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d0c4")
        cell.set_linewidth(0.75)
        if r == 0:
            cell.set_facecolor("#ece7df")
            cell.set_text_props(weight="bold", color="#2d2a24")
        elif c == -1:
            cell.set_facecolor("#f4efe6")
            cell.set_text_props(weight="bold", color="#2d2a24")
        elif c == 0:
            cell.set_facecolor("#e8f5ef")
        elif c == 1:
            cell.set_facecolor("#f8eaf4")
        else:
            cell.set_facecolor("#fff7e8")
    ax.set_title("How to justify the final result, following PO4COPs and BOPO", pad=18, fontweight="bold")
    path = OUT_DIR / "03_paper_style_evidence_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(data: Mapping[str, Mapping[str, Any]], figures: list[Path]) -> Path:
    lines = [
        "# Final loss and weighting analysis",
        "",
        "This is a final-only analysis. It does not visualize candidate search, filtering history, or generation trajectories.",
        "The structure follows the evidence style used by PO4COPs and BOPO: final performance, component effect, and mechanism explanation.",
        "",
        "## Figures",
    ]
    for path in figures:
        lines.append(f"- `{path.relative_to(REPO_ROOT).as_posix()}`")
    lines.extend(["", "## Final results"])
    for task, row in data.items():
        lhp = row["loss"]["hyperparams"]
        whp = row["weight"]["hyperparams"]
        lines.append(
            f"- {task}: final loss gain={row['loss']['gain']:.6g}; final weighting gain={row['weight']['gain']:.6g}. "
            f"Loss={lhp.get('signal_family')} + {lhp.get('link_family')} + {lhp.get('constraint_family')}; "
            f"Weighting={whp.get('geometry_family')} + {whp.get('weight_family')} + {whp.get('constraint_family')}."
        )
    lines.extend(
        [
            "",
            "## How to state the conclusion",
            "- Like PO4COPs, report final objective/gap and convergence against RL/SLL-style baselines, but do not claim search-process evidence.",
            "- Like BOPO, separate two mechanisms: the loss decides how winner-loser likelihood gaps are penalized; weighting decides which final pairs receive larger gradient mass.",
            "- For this project, the final explanation is: the discovered loss adds objective-aware preference signal, while the final weighting keeps dense pair coverage and redistributes importance toward informative objective/rank/regret gaps.",
        ]
    )
    path = OUT_DIR / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _collect()
    figures = [
        _plot_final_gains(data),
        _plot_final_designs(data),
        _plot_paper_style_comparison(),
    ]
    report = _write_report(data, figures)
    manifest = {"output_dir": str(OUT_DIR), "figures": [str(p) for p in figures], "report": str(report)}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
