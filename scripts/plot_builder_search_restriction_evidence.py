from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

_mplconfigdir = (Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "search_space_restriction_evidence"
HIST_COMMIT = "b99658c8c"

RUNS = {
    "loss_only_tsp100": REPO_ROOT / "runs/pref_loss_tsp100_discovery/20260317-131507",
    "reweight_only_tsp100": REPO_ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757",
}

HIST_RUNS = {
    "free_round1": "runs/pref_loss_alternating_simple/20260301-084801",
    "free_calibrated": "runs/pref_loss_alternating_simple/20260305-092503",
}

COLORS = {
    "free1": "#7A7A7A",
    "free2": "#E69F00",
    "loss": "#0072B2",
    "weight": "#D55E00",
    "sparse": "#CC79A7",
    "grid": "#ddd7cd",
    "bg": "#fbfaf7",
    "text": "#2f2d2a",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.2,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.2,
        "legend.fontsize": 8.6,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
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
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _git_json(commit: str, rel_path: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{rel_path}"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {commit}:{rel_path}")
    return payload


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _score_from_record(record: Mapping[str, Any]) -> float | None:
    for key in ("score", "final_score", "fitness"):
        score = _finite_float(record.get(key))
        if score is not None:
            return score
    return None


def _records_from_checkpoint(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, float, str]] = set()

    def add(record: Mapping[str, Any], source: str) -> None:
        score = _score_from_record(record)
        gen = record.get("generation")
        if score is None or gen is None:
            return
        try:
            gen_i = int(gen)
        except (TypeError, ValueError):
            return
        stage = str(record.get("stage_final") or record.get("stage") or "")
        key = (gen_i, round(score, 12), source)
        if key in seen:
            return
        seen.add(key)
        records.append({"generation": gen_i, "score": score, "stage": stage, "source": source})

    history_map = checkpoint.get("pair_score_history_map")
    if isinstance(history_map, Mapping):
        for pair_id, history in history_map.items():
            if not isinstance(history, list):
                continue
            for record in history:
                if isinstance(record, Mapping):
                    add(record, str(pair_id))

    for key in ("best_pair_score_history", "score_history"):
        history = checkpoint.get(key)
        if not isinstance(history, list):
            continue
        for record in history:
            if isinstance(record, Mapping):
                add(record, key)

    best = checkpoint.get("best_so_far")
    if isinstance(best, Mapping):
        add(best, "best_so_far")
    return records


def _best_curve(
    records: Iterable[Mapping[str, Any]],
    *,
    last_generation: int | None = None,
    start_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    by_gen: dict[int, list[float]] = {}
    max_gen = 0 if last_generation is None else int(last_generation)
    for record in records:
        score = _score_from_record(record)
        if score is None:
            continue
        gen = int(record["generation"])
        by_gen.setdefault(gen, []).append(score)
        max_gen = max(max_gen, gen)
    if not by_gen:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    xs: list[int] = []
    ys: list[float] = []
    best = start_score
    for gen in range(0, max_gen + 1):
        if gen in by_gen:
            best = min(best, min(by_gen[gen]))
        xs.append(gen)
        ys.append(best)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _manual_curve(best_generation: int, last_generation: int, score: float) -> tuple[np.ndarray, np.ndarray]:
    records = [{"generation": best_generation, "score": score}]
    return _best_curve(records, last_generation=last_generation)


def _static_metric(payload: Mapping[str, Any], metric_name: str) -> float | None:
    trace = payload.get("builder_static_trace")
    if not isinstance(trace, Mapping):
        return None
    checks = trace.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if check.get("metric_name") == metric_name:
            return _finite_float(check.get("observed_value"))
    return None


def _builder_family(payload: Mapping[str, Any]) -> str:
    ir = payload.get("ir") if isinstance(payload.get("ir"), Mapping) else {}
    hp = ir.get("hyperparams") if isinstance(ir.get("hyperparams"), Mapping) else {}
    geometry = str(hp.get("geometry_family") or "")
    cap = str(hp.get("cap_family") or "")
    weight = str(hp.get("weight_family") or "")
    return "|".join(part for part in (geometry, cap, weight) if part)


def _read_tsp100_aug_gaps() -> dict[str, float]:
    table_path = REPO_ROOT / "all_problem_results_no_eam_max_aug.csv"
    out: dict[str, float] = {}
    with table_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            solver = str(row.get("Solver") or "")
            if solver in {"Loss-only, aug_max", "Loss-weighting, aug_max"}:
                gap = _finite_float(row.get("TSP100 Gap"))
                if gap is not None:
                    out[solver] = gap
    required = {"Loss-only, aug_max", "Loss-weighting, aug_max"}
    missing = required.difference(out)
    if missing:
        raise ValueError(f"Missing TSP100 gaps in {table_path}: {sorted(missing)}")
    return out


def _linear_gap_estimator(loss_score: float, loss_gap: float, weight_score: float, weight_gap: float):
    loss_utility = -loss_score
    weight_utility = -weight_score
    slope = (weight_gap - loss_gap) / (weight_utility - loss_utility)
    intercept = loss_gap - slope * loss_utility

    def estimate(score: float) -> float:
        return intercept + slope * (-score)

    return estimate, slope, intercept


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _style_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis=grid_axis, color=COLORS["grid"], lw=0.75, alpha=0.78)


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.1,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12.4,
        fontweight="bold",
        color=COLORS["text"],
        va="top",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    free1_summary = _git_json(HIST_COMMIT, f"{HIST_RUNS['free_round1']}/summary.json")
    free1_builder = _git_json(HIST_COMMIT, f"{HIST_RUNS['free_round1']}/best_builder.json")
    free2_summary = _git_json(HIST_COMMIT, f"{HIST_RUNS['free_calibrated']}/summary.json")
    free2_builder = _git_json(HIST_COMMIT, f"{HIST_RUNS['free_calibrated']}/best_builder.json")

    loss_ckpt = _read_json(RUNS["loss_only_tsp100"] / "checkpoint.json")
    weight_ckpt = _read_json(RUNS["reweight_only_tsp100"] / "checkpoint.json")
    weight_builder = _read_json(RUNS["reweight_only_tsp100"] / "best_builder.json")

    free1_best = free1_summary["best_so_far"]
    free2_best = free2_summary["best_so_far"]
    loss_best = loss_ckpt["best_so_far"]
    weight_best = weight_ckpt["best_so_far"]

    free1_x, free1_y = _manual_curve(
        int(free1_best["generation"]),
        int(free1_summary["last_generation"]),
        float(free1_best["score"]),
    )
    free2_x, free2_y = _manual_curve(
        int(free2_best["generation"]),
        int(free2_summary["last_generation"]),
        float(free2_best["score"]),
    )
    loss_x, loss_y = _best_curve(
        _records_from_checkpoint(loss_ckpt),
        last_generation=max(int(loss_ckpt.get("next_generation", 1)) - 1, int(loss_best["generation"])),
    )
    weight_x, weight_y = _best_curve(
        _records_from_checkpoint(weight_ckpt),
        last_generation=max(int(weight_ckpt.get("next_generation", 1)) - 1, int(weight_best["generation"])),
    )

    full_pair_count = 960.0
    sparse_pair_count = _static_metric(free1_builder, "pair_count") or 568.0
    free2_pair_count = _static_metric(free2_builder, "pair_count") or full_pair_count
    weight_pair_count = full_pair_count
    sparse_ratio = sparse_pair_count / full_pair_count
    free2_ratio = free2_pair_count / full_pair_count
    weight_ratio = weight_pair_count / full_pair_count

    gaps = _read_tsp100_aug_gaps()
    loss_gap = gaps["Loss-only, aug_max"]
    weight_gap = gaps["Loss-weighting, aug_max"]
    estimate_gap, est_slope, est_intercept = _linear_gap_estimator(
        float(loss_best["score"]),
        loss_gap,
        float(weight_best["score"]),
        weight_gap,
    )

    sparse_score = float(free1_builder["fitness"])
    free1_score = float(free1_best["score"])
    free2_score = float(free2_best["score"])
    loss_score = float(loss_best["score"])
    weight_score = float(weight_best["score"])

    summary_rows = [
        {
            "method": "early_free_sparse_builder",
            "source": f"{HIST_COMMIT}:{HIST_RUNS['free_round1']}/best_builder.json",
            "search_score_lower_better": f"{sparse_score:.12g}",
            "search_utility_minus_score": f"{-sparse_score:.12g}",
            "pair_count": f"{sparse_pair_count:.0f}",
            "pair_ratio_vs_full": f"{sparse_ratio:.6f}",
            "builder_family": _builder_family(free1_builder),
            "tsp100_aug_gap": f"{estimate_gap(sparse_score):.9f}",
            "gap_source": "linear_estimate_from_loss_only_and_weighting_actual_gaps",
            "note": "Sparse rank-band candidate from unrestricted builder search; worse than the run best-so-far.",
        },
        {
            "method": "early_free_run_best",
            "source": f"{HIST_COMMIT}:{HIST_RUNS['free_round1']}/summary.json",
            "search_score_lower_better": f"{free1_score:.12g}",
            "search_utility_minus_score": f"{-free1_score:.12g}",
            "pair_count": "",
            "pair_ratio_vs_full": "",
            "builder_family": "best_so_far_kept_g_ref",
            "tsp100_aug_gap": f"{estimate_gap(free1_score):.9f}",
            "gap_source": "linear_estimate_from_loss_only_and_weighting_actual_gaps",
            "note": "Round-1 unrestricted search did not beat its reference builder.",
        },
        {
            "method": "early_free_calibrated_best",
            "source": f"{HIST_COMMIT}:{HIST_RUNS['free_calibrated']}/summary.json",
            "search_score_lower_better": f"{free2_score:.12g}",
            "search_utility_minus_score": f"{-free2_score:.12g}",
            "pair_count": f"{free2_pair_count:.0f}",
            "pair_ratio_vs_full": f"{free2_ratio:.6f}",
            "builder_family": _builder_family(free2_builder),
            "tsp100_aug_gap": f"{estimate_gap(free2_score):.9f}",
            "gap_source": "linear_estimate_from_loss_only_and_weighting_actual_gaps",
            "note": "Best unrestricted builder collapsed back to dense all-pairs with deduplication.",
        },
        {
            "method": "loss_only_full_pair",
            "source": str(RUNS["loss_only_tsp100"] / "checkpoint.json"),
            "search_score_lower_better": f"{loss_score:.12g}",
            "search_utility_minus_score": f"{-loss_score:.12g}",
            "pair_count": "960",
            "pair_ratio_vs_full": "1.000000",
            "builder_family": "g_ref_full_pair",
            "tsp100_aug_gap": f"{loss_gap:.9f}",
            "gap_source": "actual_final_table",
            "note": "Fixed full-pair builder; optimize loss only.",
        },
        {
            "method": "full_pair_reweight_only",
            "source": str(RUNS["reweight_only_tsp100"] / "checkpoint.json"),
            "search_score_lower_better": f"{weight_score:.12g}",
            "search_utility_minus_score": f"{-weight_score:.12g}",
            "pair_count": "960",
            "pair_ratio_vs_full": f"{weight_ratio:.6f}",
            "builder_family": _builder_family(weight_builder),
            "tsp100_aug_gap": f"{weight_gap:.9f}",
            "gap_source": "actual_final_table",
            "note": "Search restricted to all-pairs topology and nonnegative reweighting.",
        },
    ]
    summary_path = OUT_DIR / "search_space_restriction_summary.csv"
    _write_summary(summary_rows, summary_path)

    fig = plt.figure(figsize=(12.6, 5.25), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["bg"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.62, 0.9, 1.0], wspace=0.36)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    _style_axis(ax0, grid_axis="both")
    _add_panel_label(ax0, "A")
    ax0.step(free1_x, free1_y, where="post", color=COLORS["free1"], lw=1.9, ls="--", label="Free search round-1")
    ax0.step(free2_x, free2_y, where="post", color=COLORS["free2"], lw=2.0, ls="-.", label="Free search calibrated")
    ax0.step(loss_x, loss_y, where="post", color=COLORS["loss"], lw=2.2, label="Full-pair loss-only")
    ax0.step(weight_x, weight_y, where="post", color=COLORS["weight"], lw=2.5, label="Full-pair reweight-only")
    ax0.scatter(
        [float(free1_builder["generation"])],
        [sparse_score],
        s=48,
        facecolor=COLORS["bg"],
        edgecolor=COLORS["sparse"],
        lw=1.8,
        marker="D",
        zorder=5,
    )
    ax0.annotate(
        "sparse rank-band\ncandidate is worse",
        xy=(float(free1_builder["generation"]), sparse_score),
        xytext=(11.5, -0.0015),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["sparse"]},
        fontsize=8.4,
        color=COLORS["sparse"],
    )
    ax0.annotate(
        "best free builder\n= all-pairs dedup",
        xy=(float(free2_best["generation"]), free2_score),
        xytext=(20.6, -0.031),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["free2"]},
        fontsize=8.4,
        color="#7a5300",
    )
    ax0.set_title("Search evidence from best-so-far curves")
    ax0.set_xlabel("Search generation")
    ax0.set_ylabel("Validation score (lower is better)")
    ax0.set_ylim(-0.052, 0.004)
    ax0.legend(loc="lower left", ncol=1, frameon=True, facecolor=COLORS["bg"], edgecolor=COLORS["grid"])

    _style_axis(ax1)
    _add_panel_label(ax1, "B")
    topology_labels = ["Sparse\nfree", "All-pair\nfree", "All-pair\nweight"]
    topology_values = [sparse_ratio, free2_ratio, weight_ratio]
    topology_colors = [COLORS["sparse"], COLORS["free2"], COLORS["weight"]]
    bars1 = ax1.bar(np.arange(len(topology_labels)), topology_values, color=topology_colors, width=0.62)
    ax1.axhline(1.0, color="#5f5a51", lw=0.9, ls=":", zorder=1)
    ax1.set_ylim(0.0, 1.18)
    ax1.set_xticks(np.arange(len(topology_labels)), topology_labels)
    ax1.set_ylabel("Pair count / full-pair count")
    ax1.set_title("Supervision retained")
    for bar, value in zip(bars1, topology_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=COLORS["text"],
        )

    _style_axis(ax2)
    _add_panel_label(ax2, "C")
    final_labels = ["Sparse\nest.", "Free best\nest.", "Loss-only\nactual", "Reweight\nactual"]
    final_values = [
        estimate_gap(sparse_score),
        estimate_gap(free2_score),
        loss_gap,
        weight_gap,
    ]
    final_colors = [COLORS["sparse"], COLORS["free2"], COLORS["loss"], COLORS["weight"]]
    bars2 = ax2.bar(np.arange(len(final_labels)), final_values, color=final_colors, width=0.62)
    for i in (0, 1):
        bars2[i].set_hatch("//")
        bars2[i].set_alpha(0.62)
    ax2.set_xticks(np.arange(len(final_labels)), final_labels)
    ax2.set_ylabel("TSP100 aug-max gap")
    ax2.set_title("Downstream support")
    upper = max(final_values) * 1.22
    ax2.set_ylim(0.0, upper)
    for bar, value in zip(bars2, final_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.025,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["text"],
        )

    fig.text(
        0.012,
        0.012,
        (
            "Estimated bars use a transparent 2-point calibration from actual TSP100 aug-max gaps "
            f"(gap = {est_intercept:.5f} {est_slope:+.5f} * utility, utility=-score)."
        ),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#5d5850",
    )
    fig.suptitle(
        "Why restrict builder search to full-pair + weighting-only?",
        x=0.52,
        y=0.975,
        fontsize=12.9,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.subplots_adjust(left=0.075, right=0.988, bottom=0.21, top=0.86, wspace=0.34)

    png_path = OUT_DIR / "search_space_restriction_evidence.png"
    pdf_path = OUT_DIR / "search_space_restriction_evidence.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
