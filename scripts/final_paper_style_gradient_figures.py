from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "PTP") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PTP"))

from scripts.final_gradient_behavior_analysis import (  # noqa: E402
    AllPairsBuilder,
    BopoAnchorBestBuilder,
    ProblemSpec,
    VariantSpec,
    VARIANTS,
    bopo_loss_from_batch,
    build_problem_specs,
    load_final_pair,
    po_loss_from_batch,
    rollout_feature_caches,
)


METHOD_LABEL = {
    "po": "PO",
    "bopo": "BOPO",
    "loss_only": "Loss only",
    "weighting": "Loss + weighting",
}
METHOD_COLOR = {
    "po": "#4C78A8",
    "bopo": "#4C78A8",
    "loss_only": "#F58518",
    "weighting": "#54A24B",
}
PROBLEM_ORDER = ["tsp100", "cvrp100", "ffsp100", "jssp10x10"]


def _refresh_cache(problem: ProblemSpec, base_fc: Mapping[str, torch.Tensor], objective: torch.Tensor, log_prob: torch.Tensor) -> dict[str, torch.Tensor]:
    from fitness.free_loss_fidelity import extract_feature_cache

    seq_len = base_fc.get("seq_len")
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.full_like(log_prob, float(max(int(problem.hf.train_problem_size), 1)))
    seq_len = seq_len.to(device=log_prob.device, dtype=log_prob.dtype)
    reward_like = -objective
    extra: dict[str, torch.Tensor] = {
        "advantage": reward_like - reward_like.mean(dim=1, keepdim=True),
        "seq_len": seq_len,
        "log_prob_mean": log_prob / seq_len.clamp_min(1.0),
    }
    for key in ("log_prob_step", "entropy", "entropy_mean"):
        value = base_fc.get(key)
        if isinstance(value, torch.Tensor):
            extra[key] = value
    return extract_feature_cache(objective, log_prob, extra=extra)


def _variant_components(problem: ProblemSpec, variant: VariantSpec) -> tuple[Callable[[Mapping[str, torch.Tensor]], Any], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]:
    if variant.kind == "po":
        builder = AllPairsBuilder()
        loss_fn = lambda batch: po_loss_from_batch(batch, alpha=problem.hf.alpha, impl=variant.po_impl or problem.po_impl)
        return builder, loss_fn, "baseline_po"
    if variant.kind == "bopo":
        builder = BopoAnchorBestBuilder(problem.bopo_select_k)
        loss_fn = lambda batch: bopo_loss_from_batch(batch, alpha=problem.hf.alpha)
        return builder, loss_fn, "baseline_bopo"
    assert variant.pair_path is not None
    builder, loss_fn, _ = load_final_pair(variant.pair_path)
    return builder, loss_fn, variant.pair_path


def _safe_standardize(x: np.ndarray) -> np.ndarray:
    scale = float(np.nanmean(np.abs(x)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return x * 0.0
    return x / scale


def _bin_mean(x: np.ndarray, y: np.ndarray, bins: int, lo: float = 0.0, hi: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(bins, np.nan, dtype=np.float64)
    for i in range(bins):
        if i == bins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(mask):
            vals[i] = float(np.nanmean(y[mask]))
    return centers, vals


def _rank_positions(objective: torch.Tensor) -> torch.Tensor:
    order = objective.argsort(dim=1, descending=False)
    ranks = torch.empty_like(order, dtype=torch.long)
    base = torch.arange(objective.size(1), device=objective.device, dtype=torch.long).expand_as(order)
    ranks.scatter_(1, order, base)
    return ranks


def analyze_problem(problem: ProblemSpec, variants: list[VariantSpec], caches: list[dict[str, torch.Tensor]], *, rank_bins: int, pair_bins: int) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Any]]:
    rows: list[dict[str, Any]] = []
    payload: dict[tuple[str, str], Any] = {}

    for variant in variants:
        builder, loss_fn, source = _variant_components(problem, variant)
        update_rank_x: list[np.ndarray] = []
        update_rank_y: list[np.ndarray] = []
        gap_x: list[np.ndarray] = []
        gap_sens_y: list[np.ndarray] = []
        pair_matrix = np.zeros((pair_bins, pair_bins), dtype=np.float64)
        pair_total = 0.0
        pair_count_total = 0
        positive_best_mass = []
        negative_worst_mass = []

        for fc in caches:
            objective = fc["objective"].detach()
            lp = fc["log_prob"].detach().clone().requires_grad_(True)
            fc_grad = _refresh_cache(problem, fc, objective, lp)
            pref = builder(fc_grad)
            if pref.num_examples() <= 0:
                continue
            batch = pref.to_pairwise_loss_batch(fc_grad)
            loss = loss_fn(batch)
            loss.backward()
            update = (-lp.grad.detach()).float()
            ranks = _rank_positions(objective)
            denom = max(int(objective.size(1)) - 1, 1)
            rank_pos = (ranks.float() / float(denom)).detach().cpu().numpy().reshape(-1)
            update_np = _safe_standardize(update.detach().cpu().numpy()).reshape(-1)
            update_rank_x.append(rank_pos)
            update_rank_y.append(update_np)

            order = objective.argsort(dim=1, descending=False)
            best_idx = order[:, : max(1, int(math.ceil(0.1 * objective.size(1))))]
            worst_idx = order[:, -max(1, int(math.ceil(0.1 * objective.size(1)))) :]
            best_update = update.gather(1, best_idx)
            worst_update = update.gather(1, worst_idx)
            positive_best_mass.append(float(torch.clamp(best_update, min=0.0).sum().item()))
            negative_worst_mass.append(float(torch.clamp(-worst_update, min=0.0).sum().item()))

            pref_detached = type(pref)(
                mode=pref.mode,
                pair_idx=tuple(t.detach() for t in pref.pair_idx) if pref.pair_idx is not None else None,
                list_idx=pref.list_idx.detach() if isinstance(pref.list_idx, torch.Tensor) else pref.list_idx,
                weight=pref.weight.detach() if isinstance(pref.weight, torch.Tensor) else pref.weight,
                meta=dict(pref.meta or {}),
            )
            pair_batch0 = pref_detached.to_pairwise_loss_batch(fc)
            lpw = pair_batch0["log_prob_w"].detach().clone().requires_grad_(True)
            lpl = pair_batch0["log_prob_l"].detach().clone().requires_grad_(True)
            pair_batch = dict(pair_batch0)
            pair_batch["log_prob_w"] = lpw
            pair_batch["log_prob_l"] = lpl
            if isinstance(pair_batch.get("weight"), torch.Tensor):
                pair_batch["weight"] = pair_batch["weight"].detach()
            pair_loss = loss_fn(pair_batch)
            pair_loss.backward()
            sens = (lpw.grad.detach().abs() + lpl.grad.detach().abs()).float()
            if sens.numel() <= 0:
                continue
            sens_np = sens.cpu().numpy()
            gap = pair_batch0["cost_gap"].detach().float()
            gap_scale = fc["instance_obj_mad"].detach().float().clamp_min(1e-8)
            assert pref_detached.pair_idx is not None
            b_idx, w_idx, l_idx = pref_detached.pair_idx
            gap_norm = (gap / gap_scale[b_idx]).cpu().numpy()
            gap_norm = np.clip(gap_norm, 0.0, np.nanpercentile(gap_norm, 99) if np.isfinite(gap_norm).any() else 1.0)
            max_gap = float(np.nanmax(gap_norm)) if gap_norm.size else 1.0
            if max_gap <= 1e-12:
                max_gap = 1.0
            gap_x.append(gap_norm / max_gap)
            gap_sens_y.append(_safe_standardize(sens_np))

            rank = _rank_positions(objective)
            wr = rank[b_idx, w_idx].detach().cpu().numpy() / float(denom)
            lr = rank[b_idx, l_idx].detach().cpu().numpy() / float(denom)
            wi = np.clip((wr * pair_bins).astype(int), 0, pair_bins - 1)
            li = np.clip((lr * pair_bins).astype(int), 0, pair_bins - 1)
            for a, b, mass in zip(wi, li, sens_np):
                pair_matrix[a, b] += float(mass)
                pair_total += float(mass)
            pair_count_total += int(sens.numel())

        rx = np.concatenate(update_rank_x) if update_rank_x else np.array([])
        ry = np.concatenate(update_rank_y) if update_rank_y else np.array([])
        gx = np.concatenate(gap_x) if gap_x else np.array([])
        gy = np.concatenate(gap_sens_y) if gap_sens_y else np.array([])
        rank_centers, rank_curve = _bin_mean(rx, ry, rank_bins) if rx.size else (np.linspace(0, 1, rank_bins), np.full(rank_bins, np.nan))
        gap_centers, gap_curve = _bin_mean(gx, gy, rank_bins) if gx.size else (np.linspace(0, 1, rank_bins), np.full(rank_bins, np.nan))
        if pair_total > 0:
            pair_matrix = pair_matrix / pair_total

        rows.append(
            {
                "problem": problem.key,
                "variant": variant.key,
                "source": source,
                "top10_positive_update_mass": float(np.mean(positive_best_mass)) if positive_best_mass else float("nan"),
                "bottom10_negative_update_mass": float(np.mean(negative_worst_mass)) if negative_worst_mass else float("nan"),
                "pair_count": int(pair_count_total),
                "pair_mass_best_anchor_col": float(pair_matrix[0, :].sum()),
                "pair_mass_large_gap_triangle": float(np.triu(pair_matrix, k=max(1, pair_bins // 3)).sum()),
            }
        )
        payload[(problem.key, variant.key)] = {
            "rank_centers": rank_centers,
            "rank_curve": rank_curve,
            "gap_centers": gap_centers,
            "gap_curve": gap_curve,
            "pair_matrix": pair_matrix,
        }

    return rows, payload


def plot_outputs(rows: list[dict[str, Any]], payload: dict[tuple[str, str], Any], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

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

    row_map = {(r["problem"], r["variant"]): r for r in rows}
    problem_variants = {
        "tsp100": ["po", "loss_only", "weighting"],
        "cvrp100": ["po", "loss_only", "weighting"],
        "ffsp100": ["po", "loss_only", "weighting"],
        "jssp10x10": ["bopo", "loss_only", "weighting"],
    }

    present_problems = [p for p in PROBLEM_ORDER if any(k[0] == p for k in payload.keys())]
    fig, axes = plt.subplots(1, len(present_problems), figsize=(max(2.0 * len(present_problems), 3.2), 2.15), sharey=True, constrained_layout=True)
    if len(present_problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, present_problems):
        for variant in problem_variants[problem]:
            if (problem, variant) not in payload:
                continue
            data = payload[(problem, variant)]
            ax.plot(
                data["rank_centers"],
                data["rank_curve"],
                label=METHOD_LABEL[variant],
                color=METHOD_COLOR[variant],
                linewidth=1.8,
            )
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(problem)
        ax.set_xlabel("sorted solution rank\n(best → worst)")
        ax.grid(True, axis="y")
    axes[0].set_ylabel(r"policy update coefficient $-\partial L/\partial \log \pi(y)$")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("PO-style update separation from exact loss gradients", fontweight="bold", y=1.04)
    fig.savefig(out_dir / "fig1_update_separation.pdf")
    fig.savefig(out_dir / "fig1_update_separation.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(present_problems), figsize=(max(2.0 * len(present_problems), 3.2), 2.15), sharey=True, constrained_layout=True)
    if len(present_problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, present_problems):
        for variant in problem_variants[problem]:
            if (problem, variant) not in payload:
                continue
            data = payload[(problem, variant)]
            ax.plot(
                data["gap_centers"],
                data["gap_curve"],
                label=METHOD_LABEL[variant],
                color=METHOD_COLOR[variant],
                linewidth=1.8,
            )
        ax.set_title(problem)
        ax.set_xlabel("normalized objective gap")
        ax.grid(True, axis="y")
    axes[0].set_ylabel("pair gradient sensitivity")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("BOPO-style objective-guided scaling from exact pair gradients", fontweight="bold", y=1.04)
    fig.savefig(out_dir / "fig2_objective_guided_scaling.pdf")
    fig.savefig(out_dir / "fig2_objective_guided_scaling.png")
    plt.close(fig)

    fig, axes = plt.subplots(len(present_problems), 3, figsize=(6.8, max(1.9 * len(present_problems), 2.4)), constrained_layout=True)
    if len(present_problems) == 1:
        axes = np.asarray([axes])
    for r, problem in enumerate(present_problems):
        variants = problem_variants[problem]
        for c, variant in enumerate(variants):
            ax = axes[r, c]
            if (problem, variant) not in payload:
                ax.axis("off")
                continue
            mat = payload[(problem, variant)]["pair_matrix"]
            mat_plot = np.where(mat > 0, mat, np.nan)
            vmax = float(np.nanmax(mat_plot)) if np.isfinite(mat_plot).any() else 1.0
            vmin = max(float(np.nanmin(mat_plot)) if np.isfinite(mat_plot).any() else 1e-8, vmax * 1e-4)
            im = ax.imshow(mat_plot.T, origin="lower", cmap="YlGnBu", norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
            ax.set_xticks([0, mat.shape[0] - 1], ["best", "worst"])
            ax.set_yticks([0, mat.shape[1] - 1], ["best", "worst"])
            if r == 0:
                ax.set_title(METHOD_LABEL[variant])
            if c == 0:
                ax.set_ylabel(f"{problem}\nloser rank")
            if r == len(present_problems) - 1:
                ax.set_xlabel("winner rank")
            ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.6, label="normalized pair gradient mass")
    fig.suptitle("Pair-contribution geometry: where the loss actually spends gradient", fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig3_pair_contribution_geometry.pdf")
    fig.savefig(out_dir / "fig3_pair_contribution_geometry.png")
    plt.close(fig)

    fields = sorted({k for r in rows for k in r.keys()})
    with open(out_dir / "paper_style_gradient_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    serializable = {f"{k[0]}/{k[1]}": {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv for kk, vv in v.items()} for k, v in payload.items()}
    with open(out_dir / "paper_style_gradient_curves.json", "w", encoding="utf-8") as f:
        json.dump({"summary": rows, "curves": serializable}, f, indent=2)
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(
            "# Paper-style Gradient Figures\n\n"
            "These figures follow PO4COPs and BOPO analysis logic instead of ad-hoc proxy metrics.\n\n"
            "- `fig1_update_separation`: exact trajectory update coefficient `-dL/dlogpi(y)` sorted by objective rank, matching PO4COPs' advantage/update separation argument.\n"
            "- `fig2_objective_guided_scaling`: exact pair gradient sensitivity as a function of objective gap, matching BOPO's adaptive-scaling gradient analysis.\n"
            "- `fig3_pair_contribution_geometry`: normalized pair gradient mass over winner/loser ranks, showing whether the final pair construction spends gradient on informative comparisons.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rank-bins", type=int, default=20)
    parser.add_argument("--pair-bins", type=int, default=14)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "final_paper_style_gradient" / stamp))
    specs = build_problem_specs(str(device), max(int(args.batches), 1))

    all_rows: list[dict[str, Any]] = []
    all_payload: dict[tuple[str, str], Any] = {}
    for problem_key in [p.strip() for p in str(args.problems).split(",") if p.strip()]:
        print(f"[problem] {problem_key}", flush=True)
        if problem_key == "jssp10x10":
            try:
                from scripts.prepare_bopo_jsp_data import prepare_bopo_jsp

                prepare_bopo_jsp(REPO_ROOT)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] prepare_bopo_jsp_data failed or unnecessary: {exc}", flush=True)
        spec = specs[problem_key]
        caches = rollout_feature_caches(spec, seed=int(args.seed), device=device)
        rows, payload = analyze_problem(spec, VARIANTS[problem_key], caches, rank_bins=int(args.rank_bins), pair_bins=int(args.pair_bins))
        all_rows.extend(rows)
        all_payload.update(payload)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    plot_outputs(all_rows, all_payload, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
