from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
_mplconfigdir = REPO_ROOT / ".cache" / "matplotlib"
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

PROBLEM_ORDER = ["TSP100", "CVRP100", "FFSP100", "JSSP"]
BASELINE_ORDER = ["RL", "PO/BT", "BOPO-style", "SLL"]
COLORS = {
    "RL": "#8C8C8C",
    "PO/BT": "#0072B2",
    "BOPO-style": "#D55E00",
    "SLL": "#F58518",
    "Loss-only": "#009E73",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def _mean(rows: list[dict[str, Any]], key: str, **conds: Any) -> float:
    vals = []
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            val = _f(row, key)
            if np.isfinite(val):
                vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def _problem_key_to_label(problem: str) -> str:
    return {"tsp100": "TSP100", "cvrp100": "CVRP100", "ffsp100": "FFSP100", "jssp10x10": "JSSP"}.get(problem, problem)


def _label_to_problem_key(label: str) -> str:
    return {"TSP100": "tsp100", "CVRP100": "cvrp100", "FFSP100": "ffsp100", "JSSP": "jssp10x10"}[label]


def _nearest_for_problem(nearest_rows: list[dict[str, Any]], problem_key: str) -> str:
    for row in nearest_rows:
        if row["problem"] == problem_key and str(row["is_nearest"]).lower() == "true":
            return str(row["method"])
    vals = [row for row in nearest_rows if row["problem"] == problem_key]
    return min(vals, key=lambda r: _f(r, "gap_curve_rmse_to_loss_only"))["method"]


def _scale_ratio(scale_rows: list[dict[str, Any]], problem_key: str, method: str) -> float:
    lo = _mean(scale_rows, "mean_coefficient", problem=problem_key, method=method, objective_scale="0.25")
    hi = _mean(scale_rows, "mean_coefficient", problem=problem_key, method=method, objective_scale="4.0")
    return hi / max(lo, 1e-12)


def _best_dependency(dependency_rows: list[dict[str, Any]], problem_key: str, baseline: str) -> tuple[str, float, float]:
    candidates = [r for r in dependency_rows if r["problem"] == problem_key and r["baseline"] == baseline]
    finite = [r for r in candidates if np.isfinite(_f(r, "abs_residual_corr"))]
    if not finite:
        return "", float("nan"), float("nan")
    best = max(finite, key=lambda r: _f(r, "abs_residual_corr"))
    return str(best["feature"]), _f(best, "residual_corr"), _f(best, "abs_residual_corr")


def _derive_summary(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = {
        "nearest": _read_csv(input_dir / "nearest.csv"),
        "hardness": _read_csv(input_dir / "hardness.csv"),
        "dependency": _read_csv(input_dir / "dependency.csv"),
        "instance": _read_csv(input_dir / "instance.csv"),
        "scale": _read_csv(input_dir / "scale_sensitivity.csv"),
        "surface": _read_csv(input_dir / "surface.csv"),
    }
    summary: list[dict[str, Any]] = []
    for label in PROBLEM_ORDER:
        problem_key = _label_to_problem_key(label)
        nearest = _nearest_for_problem(rows["nearest"], problem_key)
        lo_scale = _scale_ratio(rows["scale"], problem_key, "Loss-only")
        baseline_scale_ratios = {b: _scale_ratio(rows["scale"], problem_key, b) for b in BASELINE_ORDER}
        scale_uniqueness = abs(math.log(max(lo_scale, 1e-12))) - max(abs(math.log(max(v, 1e-12))) for v in baseline_scale_ratios.values())
        for baseline in BASELINE_ORDER:
            hard = _mean(rows["hardness"], "mean_log2_loss_over_matched_baseline", problem=problem_key, baseline=baseline, hardness_group="hard")
            easy = _mean(rows["hardness"], "mean_log2_loss_over_matched_baseline", problem=problem_key, baseline=baseline, hardness_group="easy")
            hard_easy = hard - easy
            feature, corr, abs_corr = _best_dependency(rows["dependency"], problem_key, baseline)
            inst_lo = _mean(rows["instance"], "mean_coeff_vs_pool_std_corr", problem=problem_key, method="Loss-only")
            inst_base = _mean(rows["instance"], "mean_coeff_vs_pool_std_corr", problem=problem_key, method=baseline)
            summary.append(
                {
                    "Problem": label,
                    "Baseline": baseline,
                    "Nearest": baseline == nearest,
                    "Gap curve RMSE": _mean(rows["nearest"], "gap_curve_rmse_to_loss_only", problem=problem_key, method=baseline),
                    "Hard residual": hard,
                    "Easy residual": easy,
                    "Hard-Easy residual": hard_easy,
                    "Best residual feature": feature,
                    "Best residual corr": corr,
                    "Best residual abs corr": abs_corr,
                    "Loss-only scale ratio": lo_scale,
                    "Baseline scale ratio": baseline_scale_ratios[baseline],
                    "Scale uniqueness margin": scale_uniqueness,
                    "Loss-only pool-std corr": inst_lo,
                    "Baseline pool-std corr": inst_base,
                }
            )
    return summary, rows


def _select_problem_claims(summary: list[dict[str, Any]]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for problem in PROBLEM_ORDER:
        rows = [r for r in summary if r["Problem"] == problem]
        nearest = next(r["Baseline"] for r in rows if r["Nearest"])
        min_hard_easy = min(_f(r, "Hard-Easy residual") for r in rows)
        max_abs_dep = max(_f(r, "Best residual abs corr") for r in rows)
        lo_scale = _f(rows[0], "Loss-only scale ratio")
        max_base_scale_dev = max(abs(math.log(max(_f(r, "Baseline scale ratio"), 1e-12))) for r in rows)
        lo_scale_dev = abs(math.log(max(lo_scale, 1e-12)))
        scale_margin = lo_scale_dev - max_base_scale_dev
        if problem in {"TSP100", "CVRP100"} and min_hard_easy > 0.15:
            claim = "gap-controlled policy-hardness selectivity"
            evidence = f"hard-easy residual is positive against every baseline; minimum={min_hard_easy:.2f}, nearest={nearest}"
        elif scale_margin > 1.0:
            claim = "objective-dispersion-adaptive temperature"
            evidence = f"loss-only scale ratio={lo_scale:.3f} while all baselines stay near their own fixed response; nearest={nearest}"
        elif max_abs_dep > 0.5:
            claim = "secondary-feature residual dependence"
            evidence = f"max residual-feature correlation={max_abs_dep:.2f}; nearest={nearest}"
        else:
            claim = "weak or baseline-local signature"
            evidence = f"nearest={nearest}; no all-baseline residual criterion is strong"
        claims.append({"Problem": problem, "Selected signature": claim, "Evidence": evidence, "Nearest baseline": nearest})
    return claims


def _plot_problem_panels(input_dir: Path, out_dir: Path, summary: list[dict[str, Any]], rows: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
        }
    )
    for label in PROBLEM_ORDER:
        problem_key = _label_to_problem_key(label)
        nearest = _nearest_for_problem(rows["nearest"], problem_key)
        fig = plt.figure(figsize=(11.6, 7.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])
        vmax = 2.0
        image = None
        for ci, baseline in enumerate(BASELINE_ORDER):
            ax = fig.add_subplot(gs[0, ci])
            mat = np.full((8, 8), np.nan, dtype=np.float64)
            for row in rows["surface"]:
                if row["problem"] == problem_key and row["baseline"] == baseline:
                    mat[int(row["gap_bin"]), int(row["margin_bin"])] = _f(row, "mean_log2_loss_over_matched_baseline")
            image = ax.imshow(mat, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            suffix = " (nearest)" if baseline == nearest else ""
            ax.set_title(f"{baseline}{suffix}")
            ax.set_xlabel("margin quantile")
            if ci == 0:
                ax.set_ylabel("gap quantile")
            ax.set_xticks([0, 7], ["low", "high"])
            ax.set_yticks([0, 7], ["low", "high"])
        if image is not None:
            fig.colorbar(image, ax=[fig.axes[i] for i in range(4)], shrink=0.75, label="log2 loss-only / gap-matched baseline")

        ax_hard = fig.add_subplot(gs[1, 0])
        vals = []
        for baseline in BASELINE_ORDER:
            row = next(r for r in summary if r["Problem"] == label and r["Baseline"] == baseline)
            vals.append(_f(row, "Hard-Easy residual"))
        ax_hard.bar(BASELINE_ORDER, vals, color=[COLORS[b] for b in BASELINE_ORDER], width=0.66)
        ax_hard.axhline(0.0, color="#333333", lw=0.8)
        ax_hard.set_title("hard-easy residual")
        ax_hard.set_xticks(np.arange(len(BASELINE_ORDER)), BASELINE_ORDER, rotation=35, ha="right")
        ax_hard.set_ylabel("log2 residual gap")

        ax_dep = fig.add_subplot(gs[1, 1])
        vals = [_f(next(r for r in summary if r["Problem"] == label and r["Baseline"] == b), "Best residual abs corr") for b in BASELINE_ORDER]
        ax_dep.bar(BASELINE_ORDER, vals, color=[COLORS[b] for b in BASELINE_ORDER], width=0.66)
        ax_dep.set_title("strongest residual dependency")
        ax_dep.set_xticks(np.arange(len(BASELINE_ORDER)), BASELINE_ORDER, rotation=35, ha="right")
        ax_dep.set_ylabel("max |corr|")

        ax_scale = fig.add_subplot(gs[1, 2])
        for method in ["Loss-only", *BASELINE_ORDER]:
            xs = sorted({float(r["objective_scale"]) for r in rows["scale"] if r["problem"] == problem_key and r["method"] == method})
            ys_raw = [_mean(rows["scale"], "mean_coefficient", problem=problem_key, method=method, objective_scale=str(x)) for x in xs]
            if not xs:
                continue
            base = ys_raw[xs.index(1.0)] if 1.0 in xs else float(np.nanmean(ys_raw))
            ys = [v / max(base, 1e-12) for v in ys_raw]
            ax_scale.plot(xs, ys, marker="o", ms=3.0, lw=1.25, color=COLORS.get(method, "#222222"), label=method)
        ax_scale.set_xscale("log", base=2)
        ax_scale.axhline(1.0, color="#333333", lw=0.8, ls="--")
        ax_scale.set_title("objective-dispersion counterfactual")
        ax_scale.set_xlabel("objective dispersion scale")
        ax_scale.set_ylabel("relative coefficient")

        ax_table = fig.add_subplot(gs[1, 3])
        ax_table.axis("off")
        display_rows = []
        for baseline in BASELINE_ORDER:
            row = next(r for r in summary if r["Problem"] == label and r["Baseline"] == baseline)
            display_rows.append(
                [
                    baseline + ("*" if baseline == nearest else ""),
                    f"{_f(row, 'Hard-Easy residual'):.2f}",
                    str(row["Best residual feature"]),
                    f"{_f(row, 'Best residual corr'):.2f}",
                    f"{_f(row, 'Baseline scale ratio'):.2f}",
                ]
            )
        table = ax_table.table(
            cellText=display_rows,
            colLabels=["Base", "H-E", "Feature", "Corr", "Scale"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1.0, 1.35)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#d5d5d5")
            cell.set_linewidth(0.55)
            if r == 0:
                cell.set_facecolor("#eef1f5")
                cell.set_text_props(weight="bold")
        ax_table.set_title("per-baseline summary", pad=8)

        fig.suptitle(f"{label}: per-problem fine-grained loss signature", fontweight="bold")
        stem = label.lower().replace("100", "").replace("jssp", "jssp")
        fig.savefig(out_dir / f"{stem}_per_problem_signature.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}_per_problem_signature.pdf", bbox_inches="tight")
        plt.close(fig)


def _write_report(out_dir: Path, claims: list[dict[str, str]], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Per-problem fine-grained loss signatures",
        "",
        "This report does not force a single mechanism across all problems. For each discovered loss, it compares loss-only against RL, PO/BT, BOPO-style, and SLL after matching each baseline's marginal objective-gap response.",
        "",
        "## Selected Signatures",
        "",
    ]
    for claim in claims:
        lines.append(f"### {claim['Problem']}")
        lines.append(f"- Selected signature: **{claim['Selected signature']}**")
        lines.append(f"- Nearest baseline: `{claim['Nearest baseline']}`")
        lines.append(f"- Evidence: {claim['Evidence']}")
        lines.append("")
    lines.append("## Baseline-Level Metrics")
    lines.append("")
    lines.append("| Problem | Baseline | Nearest | Hard-Easy | Best feature | Corr | LO scale | Baseline scale |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for row in summary:
        lines.append(
            "| {Problem} | {Baseline} | {Nearest} | {he:.3f} | {feat} | {corr:.3f} | {los:.3f} | {bs:.3f} |".format(
                Problem=row["Problem"],
                Baseline=row["Baseline"],
                Nearest="yes" if row["Nearest"] else "",
                he=_f(row, "Hard-Easy residual"),
                feat=row["Best residual feature"],
                corr=_f(row, "Best residual corr"),
                los=_f(row, "Loss-only scale ratio"),
                bs=_f(row, "Baseline scale ratio"),
            )
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(REPO_ROOT / "figures" / "loss_fine_grained_signature" / "20260501-final-all-methods"))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    stamp = "per_problem"
    out_dir = Path(args.out_dir or (input_dir / stamp))
    summary, rows = _derive_summary(input_dir)
    claims = _select_problem_claims(summary)
    _write_csv(out_dir / "per_problem_loss_signature_summary.csv", summary)
    _write_csv(out_dir / "per_problem_selected_claims.csv", claims)
    _plot_problem_panels(input_dir, out_dir, summary, rows)
    _write_report(out_dir, claims, summary)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
