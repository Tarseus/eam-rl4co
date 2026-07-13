from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PTP_ROOT = REPO_ROOT / "PTP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PTP_ROOT) not in sys.path:
    sys.path.insert(0, str(PTP_ROOT))

_mplconfigdir = REPO_ROOT / ".cache" / "matplotlib"
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

from scripts.final_gradient_behavior_analysis import build_problem_specs, rollout_feature_caches  # noqa: E402
from scripts.plot_scale_generalization_loss_weighting import (  # noqa: E402
    WEIGHTING,
    _ensure_jssp_data,
    _load_pair,
    _replace_objective,
    _state_log_prob,
    _target_spec,
)


PROBLEMS = ["tsp100", "cvrp100", "ffsp100", "jssp10x10"]
LABELS = {
    "tsp100": "TSP100 -> TSP50",
    "cvrp100": "CVRP100 -> CVRP50",
    "ffsp100": "FFSP100 -> FFSP50",
    "jssp10x10": "JSSP10x10 -> JSSP15x15",
}
SCALE_PLAN = {
    "tsp100": (1.0, 0.5),
    "cvrp100": (1.0, 0.5),
    "ffsp100": (1.0, 0.5),
    "jssp10x10": (1.0, 1.5),
}
CLAMPS = {
    "tsp100": (0.2, 2.5),
    "cvrp100": (0.3, 5.0),
    "ffsp100": (0.15, 2.5),
    "jssp10x10": (0.1, 3.0),
}
FORMULA_SHORT = {
    "tsp100": "gap/MAD * margin/std * regret",
    "cvrp100": "absolute objective gap",
    "ffsp100": "tie(gap/MAD) * margin^-0.6 / rank span",
    "jssp10x10": "gap/MAD * regret^0.9 * sigmoid(rank)",
}
COLORS = {"search": "#0072B2", "transfer": "#D55E00"}


def _as_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy().reshape(-1)


def _subsample_np(arr: np.ndarray, max_items: int, seed: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size <= max_items:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=max_items, replace=False)
    return arr[idx]


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    vals = np.sort(np.unique(np.concatenate([a, b])))
    ca = np.searchsorted(np.sort(a), vals, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), vals, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    arr = np.sort(arr)
    n = len(arr)
    return float((2.0 * np.arange(1, n + 1).dot(arr) / (n * arr.sum())) - (n + 1.0) / n)


def _eff_ratio(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if len(arr) == 0 or float(arr.sum()) <= 1e-12:
        return float("nan")
    return float((arr.sum() ** 2) / (np.square(arr).sum() * len(arr)))


def _factor_arrays(problem: str, fc: Mapping[str, torch.Tensor], pref: Any) -> dict[str, np.ndarray]:
    objective = fc["objective"]
    log_prob = fc["log_prob"]
    b, w, l = pref.pair_idx
    eps = 1e-6
    gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
    rank = fc.get("rank")
    if not isinstance(rank, torch.Tensor):
        sorted_idx = objective.argsort(dim=1, descending=False)
        rank = torch.empty_like(sorted_idx)
        rank.scatter_(1, sorted_idx, torch.arange(objective.shape[1], device=objective.device)[None, :].expand_as(sorted_idx))
    rank_diff = (rank[b, l] - rank[b, w]).float().clamp_min(0.0)
    obj_mad = fc.get("instance_obj_mad", objective.std(dim=1))
    regret_mean = fc.get("instance_regret_mean", (objective - objective.min(dim=1, keepdim=True).values).mean(dim=1))
    logp_std = fc.get("instance_log_prob_std", log_prob.std(dim=1))
    gap_mad = gap / obj_mad[b].clamp_min(eps)
    margin_abs = (log_prob[b, w] - log_prob[b, l]).abs()
    margin_norm = margin_abs / logp_std[b].clamp_min(eps)

    if isinstance(pref.weight, torch.Tensor):
        final_weight = pref.weight.detach().float().reshape(-1)
    else:
        final_weight = torch.ones_like(gap)

    tie_pass = torch.ones_like(gap, dtype=torch.bool)
    if problem == "tsp100":
        raw = gap_mad * margin_norm * regret_mean[b].clamp_min(eps)
        primary = raw
    elif problem == "cvrp100":
        raw = gap
        primary = raw
    elif problem == "ffsp100":
        raw_margin_rank = torch.sigmoid(5.0 * rank_diff) / (margin_norm + eps).pow(0.6)
        tie_pass = gap_mad > 0.12
        after_tie = raw_margin_rank * tie_pass.to(raw_margin_rank.dtype)
        first_clamp = after_tie.clamp(0.15, 2.5)
        raw = first_clamp / rank_diff.clamp_min(eps)
        primary = raw
    elif problem == "jssp10x10":
        inst_regret = regret_mean[b].clamp_min(eps)
        base = gap_mad * inst_regret.pow(0.9)
        blend = torch.sigmoid(3.5 * rank_diff)
        raw = base * blend + 0.05 * inst_regret.pow(0.75)
        primary = raw
    else:
        raise KeyError(problem)

    lo, hi = CLAMPS[problem]
    final_np = _as_np(final_weight)
    return {
        "pre_clamp_pressure": np.log10(np.clip(_as_np(primary) / hi, 1e-12, None)),
        "final_weight": final_np,
        "normalized_final_weight": np.clip((final_np - lo) / max(hi - lo, 1e-12), 0.0, 1.0),
        "gap_mad": _as_np(gap_mad),
        "margin_norm": _as_np(margin_norm),
        "rank_diff": _as_np(rank_diff),
        "tie_pass": _as_np(tie_pass.float()),
    }


def _summary(problem: str, scale_name: str, target_size: int, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    lo, hi = CLAMPS[problem]
    weight = arrays["final_weight"]
    pre = arrays["pre_clamp_pressure"]
    tie = arrays["tie_pass"]
    return {
        "problem": problem,
        "transfer": LABELS[problem],
        "scale_name": scale_name,
        "target_size": target_size,
        "formula": FORMULA_SHORT[problem],
        "num_pairs": int(weight.size),
        "pre_pressure_q10": float(np.nanquantile(pre, 0.10)),
        "pre_pressure_median": float(np.nanmedian(pre)),
        "pre_pressure_q90": float(np.nanquantile(pre, 0.90)),
        "weight_mean": float(np.nanmean(weight)),
        "weight_std": float(np.nanstd(weight)),
        "weight_q10": float(np.nanquantile(weight, 0.10)),
        "weight_median": float(np.nanmedian(weight)),
        "weight_q90": float(np.nanquantile(weight, 0.90)),
        "low_clamp_share": float(np.nanmean(weight <= lo + 1e-6)),
        "interior_share": float(np.nanmean((weight > lo + 1e-6) & (weight < hi - 1e-6))),
        "high_clamp_share": float(np.nanmean(weight >= hi - 1e-6)),
        "tie_pass_share": float(np.nanmean(tie)) if problem == "ffsp100" else float("nan"),
        "effective_pair_ratio": _eff_ratio(weight),
        "weight_gini": _gini(weight),
    }


def collect(*, batches: int, seed: int, device: str, max_pairs: int, state: str, sharpness: float) -> dict[str, Any]:
    if "jssp10x10" in PROBLEMS:
        _ensure_jssp_data(REPO_ROOT)
    specs = build_problem_specs(device, batches)
    pairs = {problem: _load_pair(WEIGHTING[problem]) for problem in PROBLEMS}
    samples: dict[tuple[str, str, str], list[np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    for problem in PROBLEMS:
        for scale_name, scale in [("search", SCALE_PLAN[problem][0]), ("transfer", SCALE_PLAN[problem][1])]:
            target_spec = _target_spec(specs[problem], problem, scale)
            target_size = int(target_spec.hf.train_problem_size)
            print(f"[collect] {problem} {scale_name} target_size={target_size}", flush=True)
            caches = rollout_feature_caches(target_spec, seed=seed + int(round(scale * 1000)), device=torch.device(device))
            per_cache: dict[str, list[np.ndarray]] = {}
            for cache_id, raw_fc in enumerate(caches):
                log_prob = _state_log_prob(raw_fc, state, sharpness).detach()
                fc = _replace_objective(raw_fc, raw_fc["objective"].detach(), log_prob)
                pref = pairs[problem][0](fc)
                if pref.pair_idx is None or pref.num_examples() <= 0:
                    continue
                arrays = _factor_arrays(problem, fc, pref)
                for key, value in arrays.items():
                    value = _subsample_np(value, max_pairs, seed + 101 * cache_id)
                    samples.setdefault((problem, scale_name, key), []).append(value)
                    per_cache.setdefault(key, []).append(value)
            merged = {k: np.concatenate(v) for k, v in per_cache.items() if v}
            if merged:
                rows.append(_summary(problem, scale_name, target_size, merged))

    shift_rows: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for key in ["pre_clamp_pressure", "final_weight", "normalized_final_weight", "gap_mad", "margin_norm", "rank_diff"]:
            src = np.concatenate(samples.get((problem, "search", key), [np.asarray([], dtype=float)]))
            tr = np.concatenate(samples.get((problem, "transfer", key), [np.asarray([], dtype=float)]))
            shift_rows.append(
                {
                    "problem": problem,
                    "transfer": LABELS[problem],
                    "feature": key,
                    "ks_search_vs_transfer": _ks_stat(src, tr),
                    "search_mean": float(np.nanmean(src)) if src.size else float("nan"),
                    "transfer_mean": float(np.nanmean(tr)) if tr.size else float("nan"),
                    "search_median": float(np.nanmedian(src)) if src.size else float("nan"),
                    "transfer_median": float(np.nanmedian(tr)) if tr.size else float("nan"),
                }
            )
    return {"samples": samples, "summary": rows, "shift": shift_rows, "state": state, "sharpness": sharpness}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.asarray([]), np.asarray([])
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.6,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    samples: dict[tuple[str, str, str], list[np.ndarray]] = results["samples"]
    summary = results["summary"]
    row_lookup = {(r["problem"], r["scale_name"]): r for r in summary}

    fig, axes = plt.subplots(len(PROBLEMS), 3, figsize=(10.8, 9.2), constrained_layout=True)
    for row, problem in enumerate(PROBLEMS):
        label = LABELS[problem]
        ax = axes[row, 0]
        for scale_name in ["search", "transfer"]:
            arr = np.concatenate(samples.get((problem, scale_name, "pre_clamp_pressure"), [np.asarray([], dtype=float)]))
            x, y = _ecdf(np.clip(arr, -5.0, 2.0))
            ax.plot(x, y, color=COLORS[scale_name], lw=1.7, label=scale_name)
        ax.axvline(0.0, color="#333333", lw=0.8, ls="--")
        ax.set_xlim(-5.0, 2.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{label}\n{FORMULA_SHORT[problem]}")
        ax.set_xlabel("log10(pre-clamp score / upper clamp)")
        ax.set_ylabel("ECDF")
        if row == 0:
            ax.legend(frameon=False, loc="lower right")

        ax = axes[row, 1]
        for scale_name in ["search", "transfer"]:
            arr = np.concatenate(samples.get((problem, scale_name, "normalized_final_weight"), [np.asarray([], dtype=float)]))
            x, y = _ecdf(arr)
            ax.plot(x, y, color=COLORS[scale_name], lw=1.7, label=scale_name)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("final pair measure after clamp")
        ax.set_xlabel("normalized final weight")
        ax.set_ylabel("ECDF")

        ax = axes[row, 2]
        xloc = np.arange(2)
        bottom = np.zeros(2)
        parts = [
            ("low_clamp_share", "low clamp", "#8DA0CB"),
            ("interior_share", "interior", "#66C2A5"),
            ("high_clamp_share", "high clamp", "#FC8D62"),
        ]
        for metric, part_label, color in parts:
            vals = [float(row_lookup[(problem, s)][metric]) for s in ["search", "transfer"]]
            ax.bar(xloc, vals, bottom=bottom, width=0.55, color=color, label=part_label)
            bottom += np.asarray(vals)
        ax.set_xticks(xloc, ["search", "transfer"])
        ax.set_ylim(0.0, 1.0)
        ax.set_title("where the formula puts mass")
        ax.set_ylabel("pair share")
        if problem == "ffsp100":
            for xi, scale_name in zip(xloc, ["search", "transfer"]):
                tie = row_lookup[(problem, scale_name)]["tie_pass_share"]
                ax.text(xi, 1.02, f"tie pass {tie:.2f}", ha="center", va="bottom", fontsize=7.2)
        if row == 0:
            ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3)

    fig.suptitle(
        "Discovered weighting rules are scale-calibrated pair measures, not invariant loss semantics",
        fontweight="bold",
    )
    fig.savefig(out_dir / "weighting_formula_scale_shift.png", bbox_inches="tight")
    fig.savefig(out_dir / "weighting_formula_scale_shift.pdf", bbox_inches="tight")
    plt.close(fig)

    shift = results["shift"]
    mat = []
    for problem in PROBLEMS:
        row = []
        for feature in ["pre_clamp_pressure", "final_weight", "gap_mad", "margin_norm", "rank_diff"]:
            val = next(float(r["ks_search_vs_transfer"]) for r in shift if r["problem"] == problem and r["feature"] == feature)
            row.append(val)
        mat.append(row)
    fig, ax = plt.subplots(figsize=(7.4, 2.8), constrained_layout=True)
    image = ax.imshow(np.asarray(mat), cmap="YlOrRd", vmin=0.0, vmax=max(0.65, float(np.nanmax(mat))))
    ax.set_yticks(np.arange(len(PROBLEMS)), [LABELS[p] for p in PROBLEMS])
    ax.set_xticks(
        np.arange(5),
        ["pre-clamp\npressure", "final\nweight", "gap/MAD", "margin/std", "rank\nspan"],
    )
    for i in range(len(PROBLEMS)):
        for j in range(5):
            ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.82, label="KS distance")
    ax.set_title("Which formula operands move when scale changes?", fontweight="bold")
    fig.savefig(out_dir / "weighting_operand_shift_heatmap.png", bbox_inches="tight")
    fig.savefig(out_dir / "weighting_operand_shift_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--state", choices=["sampled", "aligned", "misaligned"], default="aligned")
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "paper_materials" / "weighting_generalization"))
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    out_dir = Path(args.out_dir)
    results = collect(
        batches=max(1, int(args.batches)),
        seed=int(args.seed),
        device=device,
        max_pairs=max(1000, int(args.max_pairs)),
        state=str(args.state),
        sharpness=float(args.sharpness),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "weighting_formula_distribution_summary.csv", results["summary"])
    _write_csv(out_dir / "weighting_operand_shift.csv", results["shift"])
    plot(results, out_dir)
    (out_dir / "README.md").write_text(
        "# Weighting Formula Scale Shift\n\n"
        "CPU replay analysis for the discovered best weighting builders. The script applies the same weighting formula at the search scale and a transfer scale, then records the pre-clamp formula score, final pair weight, clamp shares, and operand-level distribution shifts.\n\n"
        f"State: `{results['state']}`; sharpness: `{results['sharpness']}`.\n\n"
        "This is not a new training run. It diagnoses how second-stage weighting changes the pair measure induced over the candidate pool when the problem scale changes.\n",
        encoding="utf-8",
    )
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
