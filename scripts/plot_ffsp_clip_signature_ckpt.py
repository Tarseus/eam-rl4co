from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

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

from scripts.final_gradient_behavior_analysis import build_problem_specs  # noqa: E402
from scripts.plot_weighting_delta_cost_ckpt import _make_spec, _rollout_routing_or_ffsp  # noqa: E402


CONDITIONS = [
    ("source", "FFSP100 ckpt @ FFSP100", 1.0, "downloads/ffsp100/weighting.ckpt"),
    ("transfer", "FFSP100 ckpt @ FFSP50", 0.5, "downloads/ffsp100/weighting.ckpt"),
    ("target_trained", "FFSP50 ckpt @ FFSP50", 0.5, "downloads/ffsp50/weighting.ckpt"),
]
COLORS = {"source": "#0072B2", "transfer": "#D55E00", "target_trained": "#009E73"}
CLAMP_LO = 0.15
CLAMP_HI = 2.5


def _as_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy().reshape(-1)


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
    if arr.size == 0:
        return np.asarray([]), np.asarray([])
    arr = np.sort(arr)
    return arr, np.arange(1, arr.size + 1) / arr.size


def _binned(values: np.ndarray, y: np.ndarray, bins: int) -> list[dict[str, float]]:
    values = np.asarray(values, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(values) & np.isfinite(y)
    values = values[mask]
    y = y[mask]
    if values.size == 0:
        return []
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    rows: list[dict[str, float]] = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        pick = (values >= lo) & (values <= hi) if i == bins - 1 else (values >= lo) & (values < hi)
        if not np.any(pick):
            continue
        xv = values[pick]
        yv = y[pick]
        rows.append(
            {
                "bin": float(i),
                "x_mid": float(np.median(xv)),
                "y_mean": float(np.mean(yv)),
                "y_q25": float(np.quantile(yv, 0.25)),
                "y_q75": float(np.quantile(yv, 0.75)),
                "count": float(xv.size),
            }
        )
    return rows


def _ffsp_components(fc: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    objective = fc["objective"]
    log_prob = fc["log_prob"]
    mask = objective[:, :, None] < objective[:, None, :]
    b, w, l = mask.nonzero(as_tuple=True)
    gap = objective[b, l] - objective[b, w]
    gap_mad = gap / fc["instance_obj_mad"][b].clamp_min(1e-6)
    rank_span = (fc["rank"][b, l] - fc["rank"][b, w]).float()
    margin_norm = (log_prob[b, w] - log_prob[b, l]).abs() / fc["instance_log_prob_std"][b].clamp_min(1e-6)
    raw_margin_rank = torch.sigmoid(5.0 * rank_span) / (margin_norm + 1e-6).pow(0.6)
    tie_pass = gap_mad > 0.12
    after_tie = raw_margin_rank * tie_pass.float()
    first_clamp = after_tie.clamp(CLAMP_LO, CLAMP_HI)
    pre_final = first_clamp / rank_span.clamp_min(1e-6)
    final_weight = pre_final.clamp(CLAMP_LO, CLAMP_HI)
    upper_clip = final_weight >= CLAMP_HI - 1e-6
    return {
        "delta_cost": _as_np(gap),
        "gap_mad": _as_np(gap_mad),
        "rank_span": _as_np(rank_span),
        "margin_norm": _as_np(margin_norm),
        "raw_margin_rank": _as_np(raw_margin_rank),
        "first_clamp": _as_np(first_clamp),
        "pre_final": _as_np(pre_final),
        "final_weight": _as_np(final_weight),
        "log_pre_over_upper": np.log2(np.clip(_as_np(pre_final) / CLAMP_HI, 1e-12, None)),
        "upper_clip": _as_np(upper_clip.float()),
    }


def collect(*, batches: int, seed: int, device: str, max_pairs: int, bins: int) -> dict[str, Any]:
    specs = build_problem_specs(device, batches)
    torch_device = torch.device(device)
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for condition, label, scale, ckpt in CONDITIONS:
        spec = _make_spec(specs["ffsp100"], "ffsp100", scale, None)
        print(f"[rollout] {label}", flush=True)
        caches = _rollout_routing_or_ffsp(
            spec,
            (REPO_ROOT / ckpt).resolve(),
            batches=batches,
            seed=seed + int(scale * 1000),
            device=torch_device,
        )
        merged: dict[str, list[np.ndarray]] = {}
        for fc in caches:
            comp = _ffsp_components(fc)
            for k, v in comp.items():
                merged.setdefault(k, []).append(v)
        arrays = {k: np.concatenate(v) for k, v in merged.items()}
        n = len(arrays["final_weight"])
        if n > max_pairs:
            rng = np.random.default_rng(seed + len(condition))
            idx = rng.choice(n, size=max_pairs, replace=False)
            arrays = {k: v[idx] for k, v in arrays.items()}
            n = max_pairs
        for i in range(n):
            pair_rows.append({"condition": condition, "condition_label": label, **{k: float(v[i]) for k, v in arrays.items()}})
        fw = arrays["final_weight"]
        pre = arrays["pre_final"]
        summary_rows.append(
            {
                "condition": condition,
                "condition_label": label,
                "pair_count": int(n),
                "upper_clip_share": float(np.mean(arrays["upper_clip"] > 0.5)),
                "interior_share": float(np.mean((fw > CLAMP_LO + 1e-6) & (fw < CLAMP_HI - 1e-6))),
                "pre_final_median": float(np.median(pre)),
                "pre_final_q90": float(np.quantile(pre, 0.90)),
                "final_weight_mean": float(np.mean(fw)),
                "final_weight_median": float(np.median(fw)),
                "margin_norm_median": float(np.median(arrays["margin_norm"])),
                "rank_span_median": float(np.median(arrays["rank_span"])),
            }
        )
        for axis in ["margin_norm", "rank_span", "delta_cost"]:
            for row in _binned(arrays[axis], arrays["upper_clip"], bins):
                curve_rows.append({"condition": condition, "condition_label": label, "axis": axis, **row})
    return {"pairs": pair_rows, "summary": summary_rows, "curves": curve_rows}


def plot(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
        }
    )
    pairs = results["pairs"]
    summary = results["summary"]
    curves = results["curves"]
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 6.2), constrained_layout=True)

    ax = axes[0, 0]
    for condition, label, _, _ in CONDITIONS:
        vals = np.asarray([r["log_pre_over_upper"] for r in pairs if r["condition"] == condition], dtype=float)
        x, y = _ecdf(np.clip(vals, -4, 3))
        ax.plot(x, y, color=COLORS[condition], lw=2.0, label=label)
    ax.axvline(0.0, color="#333333", lw=0.9, ls="--")
    ax.set_xlabel("log2(pre-final score / upper clamp)")
    ax.set_ylabel("ECDF")
    ax.set_title("Hidden pre-clamp scores are often above the cap")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    x = np.arange(len(summary))
    upper = np.asarray([float(r["upper_clip_share"]) for r in summary])
    interior = np.asarray([float(r["interior_share"]) for r in summary])
    ax.bar(x, interior, color="#66C2A5", label="interior")
    ax.bar(x, upper, bottom=interior, color="#FC8D62", label="upper clipped")
    ax.set_xticks(x, [str(r["condition"]).replace("_", "\n") for r in summary])
    ax.set_ylim(0, 1)
    ax.set_ylabel("pair share")
    ax.set_title("Final weights hide about half of the dynamic range")
    for xi, val in zip(x, upper):
        ax.text(xi, interior[xi] + val / 2, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    for condition, label, _, _ in CONDITIONS:
        curve = [r for r in curves if r["condition"] == condition and r["axis"] == "margin_norm"]
        cx = np.asarray([r["x_mid"] for r in curve])
        cy = np.asarray([r["y_mean"] for r in curve])
        order = np.argsort(cx)
        ax.plot(cx[order], cy[order], color=COLORS[condition], lw=2.0, marker="o", ms=3.0, label=label)
    ax.set_xlabel(r"normalized policy margin $|\ell_w-\ell_l|/\sigma_\ell$")
    ax.set_ylabel("upper-clipped pair share")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Clipping is concentrated on low-margin pairs")

    ax = axes[1, 1]
    for condition, label, _, _ in CONDITIONS:
        curve = [r for r in curves if r["condition"] == condition and r["axis"] == "rank_span"]
        cx = np.asarray([r["x_mid"] for r in curve])
        cy = np.asarray([r["y_mean"] for r in curve])
        order = np.argsort(cx)
        ax.plot(cx[order], cy[order], color=COLORS[condition], lw=2.0, marker="o", ms=3.0, label=label)
    ax.set_xlabel("normalized rank span")
    ax.set_ylabel("upper-clipped pair share")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Clipping also targets local rank-neighbor pairs")

    fig.suptitle("FFSP weighting transfer issue: saturation masks pair-measure differences", fontweight="bold")
    fig.savefig(out_dir / "ffsp_clip_signature_ckpt.png", bbox_inches="tight")
    fig.savefig(out_dir / "ffsp_clip_signature_ckpt.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--max-pairs", type=int, default=80000)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "paper_materials" / "ffsp_clip_signature_ckpt"))
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
        bins=max(4, int(args.bins)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ffsp_clip_pairs.csv", results["pairs"])
    _write_csv(out_dir / "ffsp_clip_summary.csv", results["summary"])
    _write_csv(out_dir / "ffsp_clip_curves.csv", results["curves"])
    plot(results, out_dir)
    (out_dir / "README.md").write_text(
        "# FFSP Clip Signature on Checkpoint Rollouts\n\n"
        "This figure diagnoses the FFSP weighting rule's clipping behavior. It compares FFSP100 checkpoint rollouts at the source size, the same FFSP100 checkpoint at FFSP50, and the FFSP50 weighting checkpoint at FFSP50.\n\n"
        "Main figure: `ffsp_clip_signature_ckpt.png`.\n\n"
        "The key point is saturation: roughly half of the pairwise pre-final scores exceed the upper clamp, so final weights can look similar even when the rule depends on low-margin, local rank-neighbor regions.\n",
        encoding="utf-8",
    )
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
