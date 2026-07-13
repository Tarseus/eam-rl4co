from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

_mplconfigdir = (Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "weighting_only_branch_ablation"

HIST_COMMIT = "b99658c8c"
HIST_RUN = "runs/pref_loss_alternating_simple/20260305-092503"
CURRENT_RUN = REPO_ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757"

FULL_PAIR_COUNT = 960.0

COLORS = {
    "nonfull": "#CC79A7",
    "free_full": "#E69F00",
    "reweight": "#D55E00",
    "grid": "#ded8cf",
    "bg": "#fbfaf7",
    "text": "#2f2d2a",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.8,
        "legend.fontsize": 9.2,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    }
)


def _iter_jsonl_text(text: str) -> Iterable[dict[str, Any]]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def _git_text(commit: str, rel_path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{rel_path}"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
    )


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _score(row: Mapping[str, Any]) -> float | None:
    for key in ("final_score", "score"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _pair_count(row: Mapping[str, Any]) -> float | None:
    trace = row.get("builder_gate_trace")
    if not isinstance(trace, Mapping):
        return None
    checks = trace.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if check.get("metric_name") == "pair_count":
            return _finite_float(check.get("observed_value"))
    return None


def _family_from_pair_count(pair_count: float | None) -> str:
    if pair_count is None:
        return "unknown"
    return "full_pair" if abs(float(pair_count) - FULL_PAIR_COUNT) <= 1e-6 else "non_full_pair"


def _load_historical_branch_rows() -> list[dict[str, Any]]:
    text = _git_text(HIST_COMMIT, f"{HIST_RUN}/gate_reports.jsonl")
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl_text(text):
        if str(row.get("phase") or "") != "builder":
            continue
        gen = row.get("generation")
        score = _score(row)
        if not isinstance(gen, int) or score is None:
            continue
        pair_count = _pair_count(row)
        family = _family_from_pair_count(pair_count)
        if family == "unknown":
            continue
        rows.append(
            {
                "source": "unrestricted_stage_branch",
                "generation": int(gen) - 10,
                "raw_generation": int(gen),
                "family": family,
                "score": score,
                "pair_count": pair_count,
            }
        )
    return [row for row in rows if 0 <= int(row["generation"]) <= 9]


def _load_current_reweight_rows() -> list[dict[str, Any]]:
    path = CURRENT_RUN / "gate_reports.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for row in _iter_jsonl_text(f.read()):
            if str(row.get("phase") or "") != "builder":
                continue
            gen = row.get("generation")
            score = _score(row)
            if not isinstance(gen, int) or score is None:
                continue
            pair_count = _pair_count(row)
            rows.append(
                {
                    "source": "full_pair_reweight_only",
                    "generation": int(gen),
                    "raw_generation": int(gen),
                    "family": "full_pair_reweight",
                    "score": score,
                    "pair_count": pair_count,
                }
            )
    return [row for row in rows if 0 <= int(row["generation"]) <= 9]


def _discovery_score(row: Mapping[str, Any]) -> float | None:
    score = _finite_float(row.get("score"))
    return None if score is None else -score


def _group_discovery_scores(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        score = _discovery_score(row)
        if score is None:
            continue
        grouped[int(row["generation"])].append(score)
    return dict(grouped)


def _best_so_far_by_generation(grouped: Mapping[int, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not grouped:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    xs = list(range(min(grouped), max(grouped) + 1))
    ys: list[float] = []
    running = -math.inf
    for gen in xs:
        scores = grouped.get(gen)
        if scores:
            running = max(running, max(scores))
        ys.append(float(running) if math.isfinite(running) else np.nan)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _summary_rows(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_discovery_scores(rows)
    out: list[dict[str, Any]] = []
    running = -math.inf
    for gen in sorted(grouped):
        values = sorted(grouped[gen])
        running = max(running, max(values))
        out.append(
            {
                "method": method,
                "generation": gen,
                "n": len(values),
                "max_discovery_score": f"{max(values):.12g}",
                "median_discovery_score": f"{median(values):.12g}",
                "mean_discovery_score": f"{mean(values):.12g}",
                "q25_discovery_score": f"{np.quantile(values, 0.25):.12g}",
                "q75_discovery_score": f"{np.quantile(values, 0.75):.12g}",
                "best_so_far": f"{running:.12g}",
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jitter_points(xs: list[float], *, scale: float, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(x + rng.normal(0.0, scale)) for x in xs]


def _line_shadow(line: Any, *, alpha: float = 0.22) -> None:
    line.set_path_effects(
        [
            pe.SimpleLineShadow(offset=(1.2, -1.2), shadow_color="#111111", alpha=alpha),
            pe.Normal(),
        ]
    )


def _plot() -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hist_rows = _load_historical_branch_rows()
    current_rows = _load_current_reweight_rows()

    nonfull_rows = [row for row in hist_rows if row["family"] == "non_full_pair"]
    free_full_rows = [row for row in hist_rows if row["family"] == "full_pair"]
    reweight_rows = list(current_rows)

    summary = []
    summary.extend(_summary_rows("unrestricted_non_full_pair", nonfull_rows))
    summary.extend(_summary_rows("unrestricted_full_pair", free_full_rows))
    summary.extend(_summary_rows("full_pair_reweight_only", reweight_rows))
    summary_path = OUT_DIR / "weighting_only_branch_ablation_summary.csv"
    _write_csv(summary_path, summary)

    nonfull_x, nonfull_y = _best_so_far_by_generation(_group_discovery_scores(nonfull_rows))
    free_full_x, free_full_y = _best_so_far_by_generation(_group_discovery_scores(free_full_rows))
    reweight_x, reweight_y = _best_so_far_by_generation(_group_discovery_scores(reweight_rows))

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.82)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55, alpha=0.42)

    ax.axhline(0.0, color="#77736b", lw=1.0, ls=":", zorder=1)
    ax.fill_between([-0.5, 9.5], -8.2, 0.0, color=COLORS["nonfull"], alpha=0.045, zorder=0)
    ax.fill_between([-0.5, 9.5], 0.0, 0.055, color=COLORS["reweight"], alpha=0.045, zorder=0)

    def scatter(rows: list[dict[str, Any]], color: str, marker: str, seed: int, label: str) -> None:
        filtered = [(float(row["generation"]), _discovery_score(row)) for row in rows]
        filtered = [(x, y) for x, y in filtered if y is not None]
        xs = [x for x, _ in filtered]
        ys = [float(y) for _, y in filtered]
        ax.scatter(
            _jitter_points(xs, scale=0.055, seed=seed),
            ys,
            s=18,
            color=color,
            alpha=0.26,
            marker=marker,
            linewidths=0,
            label=label,
            zorder=2,
        )

    scatter(nonfull_rows, COLORS["nonfull"], "o", 7, "Non-full-pair candidates")
    scatter(free_full_rows, COLORS["free_full"], "s", 11, "Full-pair candidates in free search")
    scatter(reweight_rows, COLORS["reweight"], "D", 13, "Full-pair reweight candidates")

    line = ax.step(
        nonfull_x,
        nonfull_y,
        where="post",
        color=COLORS["nonfull"],
        lw=2.35,
        ls="--",
        label="Best-so-far: non-full-pair branch",
        zorder=4,
    )[0]
    _line_shadow(line, alpha=0.14)
    line = ax.step(
        free_full_x,
        free_full_y,
        where="post",
        color=COLORS["free_full"],
        lw=2.45,
        ls="-.",
        label="Best-so-far: full-pair branch in free search",
        zorder=5,
    )[0]
    _line_shadow(line, alpha=0.14)
    line = ax.step(
        reweight_x,
        reweight_y,
        where="post",
        color=COLORS["reweight"],
        lw=2.8,
        label="Best-so-far: full-pair reweight-only",
        zorder=6,
    )[0]
    _line_shadow(line, alpha=0.2)

    ax.set_xlim(-0.45, 9.45)
    ax.set_ylim(-8.1, 0.052)
    ax.set_yscale("symlog", linthresh=0.055, linscale=0.75)
    ax.set_yticks([-5.0, -1.0, -0.1, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_yticklabels(["-5", "-1", "-0.1", "0", "0.01", "0.02", "0.03", "0.04", "0.05"])
    ax.set_xticks(range(10))
    ax.set_xlabel("Builder-branch generation (aligned to branch start)")
    ax.set_ylabel("Discovery score (-validation score)", labelpad=6)
    ax.set_title("Opening pair-topology search mainly discovers bad non-full-pair builders (higher is better)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        facecolor=COLORS["bg"],
        edgecolor=COLORS["grid"],
    )
    fig.text(
        0.01,
        0.012,
        (
            f"Unrestricted branch data: {HIST_COMMIT}:{HIST_RUN}/gate_reports.jsonl. "
            "Full-pair is identified by pair_count=960; non-full-pair uses any other pair count."
        ),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#5d5850",
    )
    fig.subplots_adjust(left=0.155, right=0.985, top=0.88, bottom=0.34)

    png_path = OUT_DIR / "weighting_only_branch_ablation.png"
    pdf_path = OUT_DIR / "weighting_only_branch_ablation.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return png_path, pdf_path, summary_path


def main() -> None:
    png_path, pdf_path, summary_path = _plot()
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
