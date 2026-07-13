from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "PTP") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PTP"))

from fitness.free_loss_fidelity import PrefBatch, extract_feature_cache  # noqa: E402
from ptp_discovery.free_loss_compiler import compile_free_loss  # noqa: E402
from ptp_discovery.free_loss_ir import ir_from_json as free_loss_ir_from_json  # noqa: E402
from ptp_discovery.pref_builder_compiler import compile_preference_builder  # noqa: E402
from ptp_discovery.pref_builder_ir import ir_from_json as pref_builder_ir_from_json  # noqa: E402


PROBLEMS = ["tsp100", "cvrp100", "ffsp100", "jssp10x10"]
BASELINE = {"tsp100": "PO", "cvrp100": "PO", "ffsp100": "PO", "jssp10x10": "BOPO"}
LOSS_ONLY = {
    "tsp100": "runs/pref_loss_tsp100_discovery/20260317-131507/best_pair.json",
    "cvrp100": "runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008/best_pair.json",
    "ffsp100": "runs/pref_loss_ffsp100_discovery/20260403-142801/best_pair.json",
    "jssp10x10": "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json",
}
WEIGHTING = {
    "tsp100": "runs/pref_builder_weight_search_tsp100/20260414-113757/best_pair.json",
    "cvrp100": "runs/pref_builder_weight_search_cvrp100/20260416-093909/best_pair.json",
    "ffsp100": "runs/pref_builder_weight_search_ffsp100/20260416-111514/best_pair.json",
    "jssp10x10": "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json",
}
COLORS = {"baseline": "#4C78A8", "loss_only": "#F58518", "weighting": "#54A24B"}


def _load_pair(path: str) -> tuple[Any, Any, Mapping[str, Any]]:
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    g = compile_preference_builder(pref_builder_ir_from_json(payload["g_ir"]))
    f = compile_free_loss(free_loss_ir_from_json(payload["f_ir"]))
    return g, f, payload


def _all_pairs_builder(fc: Mapping[str, torch.Tensor]) -> PrefBatch:
    objective = fc["objective"]
    mask = objective[:, :, None] < objective[:, None, :]
    b_idx, w_idx, l_idx = mask.nonzero(as_tuple=True)
    return PrefBatch(mode="pairwise", pair_idx=(b_idx, w_idx, l_idx), weight=None)


def _bopo_builder(fc: Mapping[str, torch.Tensor], select_k: int = 4) -> PrefBatch:
    objective = fc["objective"]
    bsz, k_total = objective.shape
    sorted_idx = objective.argsort(dim=1, descending=False)
    stride = max(k_total // select_k, 1)
    selected = sorted_idx[:, ::stride][:, :select_k]
    b_idx = torch.arange(bsz, device=objective.device)[:, None].expand(bsz, select_k - 1).reshape(-1)
    w_idx = selected[:, :1].expand(bsz, select_k - 1).reshape(-1)
    l_idx = selected[:, 1:].reshape(-1)
    obj_w = objective[b_idx, w_idx]
    obj_l = objective[b_idx, l_idx]
    weight = (obj_l + 1e-6) / (obj_w + 1e-6)
    return PrefBatch(mode="pairwise", pair_idx=(b_idx, w_idx, l_idx), weight=weight)


def _loss_call(compiled_f: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    expects = [str(x) for x in getattr(compiled_f.ir.implementation_hint, "expects", [])]
    sub = {k: batch[k] for k in expects if k in batch} if expects else dict(batch)
    return compiled_f.loss_fn(sub, {}, {"alpha": 1.0})


def _baseline_update(problem: str, margin: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
    if problem == "jssp10x10":
        ratio = 1.0 + signal.clamp_min(0.0)
        return ratio * torch.sigmoid(-ratio * margin)
    if problem == "ffsp100":
        return torch.ones_like(margin)
    alpha = 0.05
    return alpha * torch.sigmoid(-alpha * margin)


def _kernel_update(compiled_f: Any, problem: str, margin_grid: np.ndarray, signal_grid: np.ndarray) -> np.ndarray:
    m = torch.tensor(margin_grid.reshape(-1), dtype=torch.float32, requires_grad=True)
    s = torch.tensor(signal_grid.reshape(-1), dtype=torch.float32)
    batch = {
        "log_prob_w": m,
        "log_prob_l": torch.zeros_like(m),
        "weight": torch.ones_like(m),
        "cost_a": torch.ones_like(m),
        "cost_b": torch.ones_like(m) + s.clamp_min(0.0),
        "cost_gap": s,
        "delta_rank": s,
        "advantage_gap": s,
    }
    loss = _loss_call(compiled_f, batch)
    loss.backward()
    update = (-m.grad.detach()).reshape(margin_grid.shape).numpy()
    scale = np.nanmax(np.abs(update))
    return update / scale if np.isfinite(scale) and scale > 1e-12 else update


def make_synthetic_cache(seed: int, *, k: int = 64, corr: float = 0.0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    objective = np.linspace(0.2, 1.8, k, dtype=np.float32)[None, :]
    noise = rng.normal(0.0, 0.55, size=(1, k)).astype(np.float32)
    log_prob = (-corr * objective + noise).astype(np.float32)
    return extract_feature_cache(
        torch.tensor(objective, dtype=torch.float32),
        torch.tensor(log_prob, dtype=torch.float32),
        extra={"advantage": torch.tensor(-objective + objective.mean(axis=1, keepdims=True), dtype=torch.float32)},
    )


def pair_budget_records(problem: str, compiled_f: Any, builder_fn: Callable[[Mapping[str, torch.Tensor]], PrefBatch], *, seeds: int = 12) -> dict[str, np.ndarray]:
    xs: list[float] = []
    masses: list[float] = []
    weights: list[float] = []
    margins: list[float] = []
    for seed in range(seeds):
        fc = make_synthetic_cache(seed, corr=(-0.4 + 0.8 * (seed % 3) / 2.0))
        pref = builder_fn(fc)
        if pref.num_examples() <= 0:
            continue
        batch0 = pref.to_pairwise_loss_batch(fc)
        lpw = batch0["log_prob_w"].detach().clone().requires_grad_(True)
        lpl = batch0["log_prob_l"].detach().clone().requires_grad_(True)
        batch = dict(batch0)
        batch["log_prob_w"] = lpw
        batch["log_prob_l"] = lpl
        if isinstance(batch.get("weight"), torch.Tensor):
            batch["weight"] = batch["weight"].detach()
        loss = _loss_call(compiled_f, batch)
        loss.backward()
        assert pref.pair_idx is not None
        b, w, l = pref.pair_idx
        rank = fc["rank"]
        span = (rank[b, l] - rank[b, w]).detach().float()
        span = (span / span.max().clamp_min(1.0)).cpu().numpy()
        mass = (lpw.grad.detach().abs() + lpl.grad.detach().abs()).cpu().numpy()
        weight = batch0["weight"].detach().cpu().numpy()
        margin = (batch0["log_prob_w"] - batch0["log_prob_l"]).detach().abs().cpu().numpy()
        xs.extend(span.tolist())
        masses.extend(mass.tolist())
        weights.extend(weight.tolist())
        margins.extend(margin.tolist())
    return {
        "rank_span": np.asarray(xs, dtype=np.float64),
        "mass": np.asarray(masses, dtype=np.float64),
        "weight": np.asarray(weights, dtype=np.float64),
        "margin_abs": np.asarray(margins, dtype=np.float64),
    }


def _binned_share(x: np.ndarray, mass: np.ndarray, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.zeros(bins, dtype=np.float64)
    total = float(np.nansum(mass))
    if total <= 1e-12:
        return centers, out
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1] if i < bins - 1 else x <= edges[i + 1])
        out[i] = float(np.nansum(mass[mask]) / total)
    return centers, out


def build_all() -> dict[str, Any]:
    out: dict[str, Any] = {}
    margins = np.linspace(-4.0, 4.0, 121)
    signals = np.linspace(0.0, 3.0, 81)
    m_grid, s_grid = np.meshgrid(margins, signals)
    for problem in PROBLEMS:
        g_loss, f_loss, payload_loss = _load_pair(LOSS_ONLY[problem])
        g_weight, f_weight, payload_weight = _load_pair(WEIGHTING[problem])
        baseline = _baseline_update(
            problem,
            torch.tensor(m_grid, dtype=torch.float32),
            torch.tensor(s_grid, dtype=torch.float32),
        ).numpy()
        baseline = baseline / np.nanmax(np.abs(baseline))
        loss_kernel = _kernel_update(f_loss, problem, m_grid, s_grid)
        weight_kernel = _kernel_update(f_weight, problem, m_grid, s_grid)
        baseline_builder = (lambda fc: _bopo_builder(fc, 4)) if problem == "jssp10x10" else _all_pairs_builder
        loss_records = pair_budget_records(problem, f_loss, _all_pairs_builder)
        weight_records = pair_budget_records(problem, f_weight, lambda fc, gw=g_weight: gw.build_fn(fc, {}))
        base_records = pair_budget_records(
            problem,
            f_loss,
            baseline_builder,
        )
        out[problem] = {
            "margin_grid": m_grid,
            "signal_grid": s_grid,
            "baseline_kernel": baseline,
            "loss_kernel": loss_kernel,
            "weight_kernel": weight_kernel,
            "loss_payload": payload_loss,
            "weight_payload": payload_weight,
            "base_records": base_records,
            "loss_records": loss_records,
            "weight_records": weight_records,
        }
    return out


def plot(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25), sharey=True, constrained_layout=True)
    signal_levels = [0.2, 1.2, 2.6]
    linestyles = ["-", "--", ":"]
    for ax, problem in zip(axes, PROBLEMS):
        r = results[problem]
        margins = r["margin_grid"][0]
        signals = r["signal_grid"][:, 0]
        for sig, ls in zip(signal_levels, linestyles):
            idx = int(np.argmin(np.abs(signals - sig)))
            ax.plot(margins, r["baseline_kernel"][idx], color=COLORS["baseline"], lw=1.3, ls=ls)
            ax.plot(margins, r["loss_kernel"][idx], color=COLORS["loss_only"], lw=1.7, ls=ls)
            ax.plot(margins, r["weight_kernel"][idx], color=COLORS["weighting"], lw=1.7, ls=ls)
        ax.axvline(0, color="#333333", lw=0.7)
        ax.set_title(problem)
        ax.set_xlabel(r"log-prob margin $\Delta=\log\pi_w-\log\pi_l$")
    axes[0].set_ylabel(r"normalized winner update $-\partial L/\partial \log\pi_w$")
    axes[-1].plot([], [], color=COLORS["baseline"], label="baseline loss")
    axes[-1].plot([], [], color=COLORS["loss_only"], label="final loss")
    axes[-1].plot([], [], color=COLORS["weighting"], label="final loss + weighting")
    axes[-1].plot([], [], color="#555", ls="-", label="low signal")
    axes[-1].plot([], [], color="#555", ls="--", label="mid signal")
    axes[-1].plot([], [], color="#555", ls=":", label="high signal")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=6.7)
    fig.suptitle("Loss kernel: exact policy-update response of the final loss", fontweight="bold", y=1.05)
    fig.savefig(out_dir / "fig1_loss_kernel_update.pdf")
    fig.savefig(out_dir / "fig1_loss_kernel_update.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25), constrained_layout=True)
    for ax, problem in zip(axes, PROBLEMS):
        rec = results[problem]["weight_records"]
        x = rec["rank_span"]
        y = rec["margin_abs"]
        c = rec["weight"]
        if len(x) > 0:
            sc = ax.scatter(x, y, c=c, s=4, cmap="viridis", alpha=0.55, linewidths=0)
        ax.set_title(problem)
        ax.set_xlabel("rank span (near → far)")
        ax.set_ylabel(r"$|\Delta \log\pi|$")
        ax.grid(False)
    fig.colorbar(sc, ax=axes, shrink=0.75, label="final weighting value")
    fig.suptitle("Weighting kernel: what the final builder upweights before training", fontweight="bold", y=1.05)
    fig.savefig(out_dir / "fig2_weighting_kernel_scatter.pdf")
    fig.savefig(out_dir / "fig2_weighting_kernel_scatter.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25), sharey=True, constrained_layout=True)
    for ax, problem in zip(axes, PROBLEMS):
        for key, label in [("base_records", BASELINE[problem]), ("loss_records", "Loss only"), ("weight_records", "Loss + weighting")]:
            rec = results[problem][key]
            x, share = _binned_share(rec["rank_span"], rec["mass"], bins=12)
            color = COLORS["baseline"] if key == "base_records" else COLORS["loss_only"] if key == "loss_records" else COLORS["weighting"]
            ax.plot(x, share, marker="o", ms=3.2, lw=1.7, color=color, label=label)
        ax.set_title(problem)
        ax.set_xlabel("pair rank span")
    axes[0].set_ylabel("share of total pair-gradient budget")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=7)
    fig.suptitle("Combined budget: exact loss gradient multiplied by final weighting", fontweight="bold", y=1.05)
    fig.savefig(out_dir / "fig3_combined_gradient_budget.pdf")
    fig.savefig(out_dir / "fig3_combined_gradient_budget.png")
    plt.close(fig)

    summary = []
    for problem in PROBLEMS:
        for key, label in [("base_records", BASELINE[problem]), ("loss_records", "loss_only"), ("weight_records", "weighting")]:
            rec = results[problem][key]
            span = rec["rank_span"]
            mass = rec["mass"]
            total = float(np.nansum(mass))
            high = float(np.nansum(mass[span >= 0.67]) / total) if total > 0 else float("nan")
            low = float(np.nansum(mass[span <= 0.33]) / total) if total > 0 else float("nan")
            summary.append({"problem": problem, "method": label, "low_span_budget": low, "high_span_budget": high, "pairs": int(len(span))})
    import csv

    with (out_dir / "kernel_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem", "method", "low_span_budget", "high_span_budget", "pairs"])
        writer.writeheader()
        writer.writerows(summary)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Loss and Weighting Kernel Analysis\n\n"
            "This analysis does not use checkpoints. It directly executes the final loss and final preference builder.\n\n"
            "- `fig1_loss_kernel_update`: exact winner update coefficient from the loss as a function of log-prob margin and objective/rank/advantage signal.\n"
            "- `fig2_weighting_kernel_scatter`: exact final builder weights on synthetic all-pair feature caches.\n"
            "- `fig3_combined_gradient_budget`: how the product of loss gradient and weight redistributes total gradient budget over pair rank spans.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures" / "loss_weighting_kernel_analysis"))
    args = parser.parse_args()
    results = build_all()
    plot(results, Path(args.out_dir))
    print(f"[done] outputs={args.out_dir}")


if __name__ == "__main__":
    main()
