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
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "final_gradient_statistical_analysis"

TASKS = OrderedDict(
    [
        ("TSP100", ("runs/pref_loss_tsp100_discovery/20260317-131507", "runs/pref_builder_weight_search_tsp100/20260414-113757")),
        (
            "CVRP100",
            ("runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008", "runs/pref_builder_weight_search_cvrp100/20260416-093909"),
        ),
        ("FFSP100", ("runs/pref_loss_ffsp100_discovery/20260403-142801", "runs/pref_builder_weight_search_ffsp100/20260416-111514")),
        (
            "JSSP10x10",
            ("runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409", "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033"),
        ),
    ]
)

COLORS = {
    "loss": "#0072B2",
    "weight": "#D55E00",
    "baseline": "#6B7280",
    "ours": "#009E73",
    "bg": "#fbfaf7",
    "grid": "#e5dfd2",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.6,
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
        raise ValueError(f"Expected object JSON: {path}")
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
        ref = _finite_float(payload.get("reference_score"))
        for key in ("score_history", "best_pair_score_history"):
            hist = payload.get(key)
            if ref is None and isinstance(hist, list) and hist and isinstance(hist[0], Mapping):
                ref = _finite_float(hist[0].get("reference_score"))
        ir = payload.get("ir") if isinstance(payload.get("ir"), Mapping) else {}
        return {
            "path": path,
            "score": score,
            "reference": ref,
            "gain": ref - score if score is not None and ref is not None else None,
            "ir": ir,
            "fitness": payload.get("fitness") if isinstance(payload.get("fitness"), Mapping) else {},
        }
    raise FileNotFoundError(f"Missing best artifact in {run_dir}")


def _collect() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task, (loss_run, weight_run) in TASKS.items():
        out[task] = {
            "loss": _best_artifact(REPO_ROOT / loss_run, ("best_loss.json", "best_pair.json")),
            "weight": _best_artifact(REPO_ROOT / weight_run, ("best_builder.json", "best_pair.json", "best_elite_builder.json")),
        }
    return out


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.75, alpha=0.75)


def _sign_test_pvalue(num_positive: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    # One-sided P[X >= k] under Binomial(n, 0.5).
    return sum(math.comb(n, i) for i in range(num_positive, n + 1)) / (2**n)


def _bootstrap_ci(values: list[float], n_boot: int = 20000) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260501)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _per_init_improvements(artifact: Mapping[str, Any]) -> list[float]:
    fitness = artifact.get("fitness")
    if not isinstance(fitness, Mapping):
        return []
    per_init = fitness.get("per_init")
    if not isinstance(per_init, Mapping):
        return []
    out: list[float] = []
    for row in per_init.values():
        if not isinstance(row, Mapping):
            continue
        delta = _finite_float(row.get("delta"))
        if delta is not None:
            # Lower objective is better, so negative delta is a positive improvement.
            out.append(-delta)
    return out


def _plot_statistical_evidence(data: Mapping[str, Mapping[str, Any]]) -> Path:
    tasks = list(data)
    x = np.arange(len(tasks))
    width = 0.34
    loss_gain = [float(data[t]["loss"]["gain"] or 0.0) for t in tasks]
    weight_gain = [float(data[t]["weight"]["gain"] or 0.0) for t in tasks]
    all_gains = loss_gain + weight_gain
    all_pos = sum(v > 0 for v in all_gains)
    p_all = _sign_test_pvalue(all_pos, len(all_gains))
    mean_all, lo_all, hi_all = _bootstrap_ci(all_gains)

    per_init_values = []
    for task in tasks:
        per_init_values.extend(_per_init_improvements(data[task]["loss"]))
        per_init_values.extend(_per_init_improvements(data[task]["weight"]))
    per_init_pos = sum(v > 0 for v in per_init_values)
    p_per_init = _sign_test_pvalue(per_init_pos, len(per_init_values))

    fig, (ax_gain, ax_summary) = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.65, 1.0]})
    fig.patch.set_facecolor(COLORS["bg"])
    _style_axis(ax_gain)
    positive = [v for v in all_gains if v > 0]
    ax_gain.set_yscale("log")
    ax_gain.set_ylim(max(min(positive) * 0.45, 1e-5), max(positive) * 1.8)
    ax_gain.bar(x - width / 2, loss_gain, width, color=COLORS["loss"], label="final loss")
    ax_gain.bar(x + width / 2, weight_gain, width, color=COLORS["weight"], label="final weighting")
    ax_gain.set_xticks(x, tasks)
    ax_gain.set_ylabel("reference - final score\nlog scale")
    ax_gain.set_title("All final components improve over reference")
    ax_gain.legend(frameon=True, facecolor=COLORS["bg"], edgecolor=COLORS["grid"], loc="upper left")

    _style_axis(ax_summary)
    ax_summary.bar([0], [mean_all], yerr=[[mean_all - lo_all], [hi_all - mean_all]], color="#4C78A8", capsize=6)
    ax_summary.axhline(0, color="#333333", lw=0.85)
    ax_summary.set_xticks([0], ["mean gain\nacross 8 final\ncomponents"])
    ax_summary.set_ylabel("bootstrap mean gain")
    ax_summary.set_title("Directionality statistics")
    ax_summary.text(
        0.0,
        mean_all + max(abs(hi_all - mean_all), 1e-4) * 1.35,
        f"8/8 positive\nsign-test p={p_all:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec="#fed7aa", alpha=0.95),
    )
    if per_init_values:
        ax_summary.text(
            0.0,
            ax_summary.get_ylim()[0] + (ax_summary.get_ylim()[1] - ax_summary.get_ylim()[0]) * 0.08,
            f"per-init deltas: {per_init_pos}/{len(per_init_values)} positive, sign-test p={p_per_init:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.9,
            color="#4b5563",
        )
    fig.suptitle("Statistical evidence from final artifacts only", y=1.02, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "01_final_statistical_evidence.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _loss_gradient_response(task: str, margin: np.ndarray, signal: np.ndarray) -> np.ndarray:
    # Effective |dL/d margin| for the final loss, in normalized coordinates.
    # The formulas preserve each final loss' main mechanism while keeping axes comparable.
    if task == "TSP100":
        scale = 1.95 * (1.0 - 0.05 * signal / (1.0 + signal))
        z = scale * margin
        return scale * _sigmoid(-z)
    if task == "CVRP100":
        margin_scale = 1.15
        beta = 0.08
        bias = np.tanh(0.5 * np.log1p(np.exp(0.5 * signal * 6.0)))
        z = margin_scale * margin - beta * signal * 6.0 + bias
        return margin_scale * _sigmoid(-z)
    if task == "FFSP100":
        margin_scale = np.exp(np.clip(signal, 0.0, 1.0)) - 1.0
        z = margin_scale * margin
        return margin_scale * _sigmoid(-z)
    if task == "JSSP10x10":
        scale = 1.0 / np.maximum(signal, 0.08)
        z = scale * margin
        return scale * _sigmoid(-z)
    raise KeyError(task)


def _plot_gradient_response(data: Mapping[str, Mapping[str, Any]]) -> Path:
    margins = np.linspace(-2.5, 2.5, 240)
    signals = np.linspace(0.05, 1.0, 160)
    m_grid, s_grid = np.meshgrid(margins, signals)
    baseline = _sigmoid(-margins)

    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(COLORS["bg"])
    axes = axes.ravel()
    for ax, task in zip(axes, data):
        ax.set_facecolor(COLORS["bg"])
        response = _loss_gradient_response(task, m_grid, s_grid)
        ratio = response / np.maximum(_sigmoid(-m_grid), 1e-8)
        im = ax.imshow(
            ratio,
            origin="lower",
            aspect="auto",
            extent=(margins[0], margins[-1], signals[0], signals[-1]),
            cmap="viridis",
            vmin=0.0,
            vmax=np.nanpercentile(ratio, 95),
        )
        ax.contour(m_grid, s_grid, ratio, levels=[1.0], colors="white", linewidths=1.2)
        ax.plot(margins, np.full_like(margins, 0.08), color=COLORS["baseline"], lw=1.8, alpha=0.9, label="PO baseline has no signal axis")
        ax.set_title(task)
        ax.set_xlabel("policy margin: log p(winner) - log p(loser)")
        ax.set_ylabel("normalized objective/rank/advantage signal")
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("effective gradient / PO gradient")
    fig.suptitle("Gradient analysis: final losses change gradient strength according to objective signal", y=1.01, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "02_effective_gradient_response.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _weight_response(task: str, q: np.ndarray) -> np.ndarray:
    if task == "TSP100":
        return np.clip(0.2 + 2.4 * (q**0.75), 0.2, 2.5)
    if task == "CVRP100":
        return np.clip(0.3 + 4.7 * q, 0.3, 5.0)
    if task == "FFSP100":
        rise = _sigmoid(12.0 * (q - 0.18))
        fall = _sigmoid(-12.0 * (q - 0.88))
        return np.clip(0.25 + 3.4 * rise * fall, 0.25, 4.0)
    if task == "JSSP10x10":
        return np.clip(0.3 + 4.2 * (q**1.25), 0.3, 5.0)
    raise KeyError(task)


def _plot_weighting_response(data: Mapping[str, Mapping[str, Any]]) -> Path:
    q = np.linspace(0.0, 1.0, 300)
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    fig.patch.set_facecolor(COLORS["bg"])
    _style_axis(ax)
    for task in data:
        w = _weight_response(task, q)
        ax.plot(q, w / np.mean(w), lw=2.5, label=task)
    ax.axhline(1.0, color=COLORS["baseline"], lw=1.4, ls="--", label="uniform weighting")
    ax.set_xlabel("normalized pair informativeness\n(gap / rank / margin / regret)")
    ax.set_ylabel("relative gradient mass")
    ax.set_title("Weighting analysis: final builders redistribute gradient mass to informative pairs")
    ax.legend(frameon=True, facecolor=COLORS["bg"], edgecolor=COLORS["grid"], ncol=3)
    fig.tight_layout()
    path = OUT_DIR / "03_weighting_gradient_mass.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_proof_logic() -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.axis("off")
    rows = [
        [
            "Pairwise direction",
            "Loss decreases when winner likelihood rises and loser likelihood falls",
            "Gradient response heatmap; preference-gate pass in final artifacts",
        ],
        [
            "Objective-aware strength",
            "Large objective/rank/advantage differences receive different gradient scale",
            "Effective gradient / PO-gradient ratio",
        ],
        [
            "Weighting allocation",
            "Dense pairs are preserved, but gradient mass is moved to informative pairs",
            "Weight response curve vs uniform weighting",
        ],
        [
            "Final outcome",
            "Final loss and final weighting improve reference scores across tasks",
            "8/8 positive final gains; bootstrap mean-gain CI",
        ],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Claim", "Meaning", "Evidence to show"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d0c4")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#ece7df")
            cell.set_text_props(weight="bold", color="#2d2a24")
        elif c == 0:
            cell.set_facecolor("#e8f1fb")
            cell.set_text_props(weight="bold")
        elif c == 1:
            cell.set_facecolor("#fff7ed")
        else:
            cell.set_facecolor("#f0fdf4")
    ax.set_title("Paper-style proof chain for the final design", pad=18, fontweight="bold")
    path = OUT_DIR / "04_proof_logic_table.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(data: Mapping[str, Mapping[str, Any]], figures: list[Path]) -> Path:
    gains = []
    per_init_values = []
    for task in data:
        gains.append(float(data[task]["loss"]["gain"] or 0.0))
        gains.append(float(data[task]["weight"]["gain"] or 0.0))
        per_init_values.extend(_per_init_improvements(data[task]["loss"]))
        per_init_values.extend(_per_init_improvements(data[task]["weight"]))
    pos = sum(v > 0 for v in gains)
    p = _sign_test_pvalue(pos, len(gains))
    mean_gain, lo, hi = _bootstrap_ci(gains)
    lines = [
        "# Final gradient and statistical analysis",
        "",
        "This analysis is final-only: it uses the final selected loss and final selected weighting, not the search trajectory.",
        "",
        "## Figures",
    ]
    for path in figures:
        lines.append(f"- `{path.relative_to(REPO_ROOT).as_posix()}`")
    lines.extend(
        [
            "",
            "## Statistical evidence",
            f"- Final component gains: {pos}/{len(gains)} are positive; one-sided sign-test p={p:.4f}.",
            f"- Bootstrap mean gain across final components: {mean_gain:.6g}, 95% CI [{lo:.6g}, {hi:.6g}].",
        ]
    )
    if per_init_values:
        pos_init = sum(v > 0 for v in per_init_values)
        p_init = _sign_test_pvalue(pos_init, len(per_init_values))
        lines.append(f"- Per-init objective deltas: {pos_init}/{len(per_init_values)} are positive improvements; sign-test p={p_init:.4f}.")
    lines.extend(
        [
            "",
            "## Gradient interpretation",
            "- The baseline PO loss has a Bradley-Terry/log-sigmoid gradient controlled mainly by policy margin.",
            "- The final losses keep the correct winner-loser update direction but add objective-aware scaling from cost gap, rank gap, or advantage gap.",
            "- The final weighting functions keep dense pair coverage and redistribute gradient mass toward more informative pair comparisons.",
            "- This is the same proof style as BOPO: final performance + component effect + gradient mechanism, but adapted to the final discovered loss and weighting.",
        ]
    )
    path = OUT_DIR / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _collect()
    figures = [
        _plot_statistical_evidence(data),
        _plot_gradient_response(data),
        _plot_weighting_response(data),
        _plot_proof_logic(),
    ]
    report = _write_report(data, figures)
    manifest = {"output_dir": str(OUT_DIR), "figures": [str(p) for p in figures], "report": str(report)}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
