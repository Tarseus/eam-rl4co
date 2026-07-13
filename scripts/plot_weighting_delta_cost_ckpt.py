from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
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

from fitness.free_loss_fidelity import (  # noqa: E402
    _build_mgl_jssp_model,
    _load_policy_weights_from_checkpoint,
    _mgl_jssp_selected_log_prob_step,
    _rl4co_build_env,
    _rl4co_build_policy,
    _rl4co_objective_from_reward,
    _rl4co_rollout,
    extract_feature_cache,
)
from fitness.ptp_high_fidelity import _set_seed, resolve_pomo_size  # noqa: E402
from scripts.final_gradient_behavior_analysis import build_problem_specs  # noqa: E402
from scripts.plot_scale_generalization_loss_weighting import _ensure_jssp_data, _target_spec, _load_pair  # noqa: E402


PROBLEM_ORDER = ["tsp100", "cvrp100", "ffsp100", "jssp10x10"]
PAIR_PATHS = {
    "tsp100": "runs/pref_builder_weight_search_tsp100/20260414-113757/best_pair.json",
    "cvrp100": "runs/pref_builder_weight_search_cvrp100/20260416-093909/best_pair.json",
    "ffsp100": "runs/pref_builder_weight_search_ffsp100/20260416-111514/best_pair.json",
    "jssp10x10": "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json",
}
CONDITIONS = {
    "tsp100": [
        ("search", "TSP100 ckpt @ TSP100", 1.0, "downloads/tsp100/weighting/checkpoint.ckpt", None),
        ("transfer", "TSP50 ckpt @ TSP50", 0.5, "downloads/tsp50/weighting/checkpoint.ckpt", None),
    ],
    "cvrp100": [
        ("search", "CVRP100 ckpt @ CVRP100", 1.0, "downloads/cvrp100/weighting/checkpoint.ckpt", 50),
        ("transfer", "CVRP100 ckpt @ CVRP50", 0.5, "downloads/cvrp100/weighting/checkpoint.ckpt", 50),
    ],
    "ffsp100": [
        ("search", "FFSP100 ckpt @ FFSP100", 1.0, "downloads/ffsp100/weighting.ckpt", None),
        ("transfer", "FFSP50 ckpt @ FFSP50", 0.5, "downloads/ffsp50/weighting.ckpt", None),
    ],
    "jssp10x10": [
        ("search", "JSSP10x10 ckpt @ 10x10", 1.0, "downloads/jssp10x10/weighting/checkpoint.ckpt", None),
        ("transfer", "JSSP15x15 ckpt @ 15x15", 1.5, "downloads/jssp15x15/weighting/checkpoint.ckpt", None),
    ],
}
PROBLEM_LABELS = {
    "tsp100": "TSP",
    "cvrp100": "CVRP",
    "ffsp100": "FFSP",
    "jssp10x10": "JSSP",
}
COLORS = {"search": "#0072B2", "transfer": "#D55E00"}


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


def _subsample_pair_arrays(arrays: dict[str, np.ndarray], max_pairs: int, seed: int) -> dict[str, np.ndarray]:
    n = min(len(v) for v in arrays.values())
    if n <= max_pairs:
        return {k: np.asarray(v[:n]) for k, v in arrays.items()}
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pairs, replace=False)
    return {k: np.asarray(v[:n])[idx] for k, v in arrays.items()}


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in ["mean", "std", "q10", "median", "q90"]}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def _binned_curve(delta: np.ndarray, weight: np.ndarray, bins: int) -> list[dict[str, float]]:
    delta = np.asarray(delta, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mask = np.isfinite(delta) & np.isfinite(weight)
    delta = delta[mask]
    weight = weight[mask]
    if delta.size == 0:
        return []
    edges = np.quantile(delta, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    rows: list[dict[str, float]] = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            pick = (delta >= lo) & (delta <= hi)
        else:
            pick = (delta >= lo) & (delta < hi)
        if not np.any(pick):
            continue
        d = delta[pick]
        w = weight[pick]
        rows.append(
            {
                "bin": float(i),
                "delta_cost_min": float(np.min(d)),
                "delta_cost_max": float(np.max(d)),
                "delta_cost_mid": float(np.median(d)),
                "weight_mean": float(np.mean(w)),
                "weight_q25": float(np.quantile(w, 0.25)),
                "weight_q75": float(np.quantile(w, 0.75)),
                "count": float(d.size),
            }
        )
    return rows


def _binned_xy_curve(x_values: np.ndarray, weight: np.ndarray, bins: int) -> list[dict[str, float]]:
    x_values = np.asarray(x_values, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mask = np.isfinite(x_values) & np.isfinite(weight)
    x_values = x_values[mask]
    weight = weight[mask]
    if x_values.size == 0:
        return []
    edges = np.quantile(x_values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    rows: list[dict[str, float]] = []
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            pick = (x_values >= lo) & (x_values <= hi)
        else:
            pick = (x_values >= lo) & (x_values < hi)
        if not np.any(pick):
            continue
        x = x_values[pick]
        w = weight[pick]
        rows.append(
            {
                "bin": float(i),
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "x_mid": float(np.median(x)),
                "weight_mean": float(np.mean(w)),
                "weight_q25": float(np.quantile(w, 0.25)),
                "weight_q75": float(np.quantile(w, 0.75)),
                "count": float(x.size),
            }
        )
    return rows


def _rollout_routing_or_ffsp(
    spec: Any,
    ckpt_path: Path,
    *,
    batches: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    env = _rl4co_build_env(spec.hf, spec.hf.train_problem_size).to(device)
    policy, rollout_strategy = _rl4co_build_policy(spec.hf, env)
    _load_policy_weights_from_checkpoint(policy, str(ckpt_path))
    policy = policy.to(device)
    policy.eval()
    num_rollouts = resolve_pomo_size(spec.hf.pomo_size, spec.hf.train_problem_size)
    caches: list[dict[str, torch.Tensor]] = []
    for batch_id in range(batches):
        _set_seed(seed + 1009 * batch_id)
        with torch.no_grad():
            reward, log_prob = _rl4co_rollout(
                env,
                policy,
                spec.batch_size,
                num_rollouts,
                phase="train",
                rollout_strategy=rollout_strategy,
                device=device,
                precision=spec.hf.precision,
                cfg_like=spec.hf,
            )
        objective = _rl4co_objective_from_reward(reward.float(), spec.hf)
        log_prob_f = log_prob.float()
        reward_f = reward.float()
        seq_len = torch.full_like(log_prob_f, float(max(int(spec.hf.train_problem_size), 1)))
        extra = {
            "advantage": reward_f - reward_f.mean(dim=1, keepdim=True),
            "seq_len": seq_len,
            "log_prob_mean": log_prob_f / seq_len.clamp_min(1.0),
        }
        caches.append(extract_feature_cache(objective, log_prob_f, extra=extra))
    return caches


def _rollout_jssp(
    spec: Any,
    ckpt_path: Path,
    *,
    batches: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    from rl4co.models.zoo.mgl_jssp.sampling import solve_jsp

    model = _build_mgl_jssp_model(spec.hf, problem_size=spec.hf.train_problem_size)
    _load_policy_weights_from_checkpoint(model, str(ckpt_path))
    model = model.to(device)
    model.eval()
    model.setup("fit")
    loader = model.train_dataloader()
    loader_iter = iter(loader)
    batch_rollouts = int(getattr(model, "B", resolve_pomo_size(spec.hf.pomo_size, spec.hf.train_problem_size)) or 1)
    caches: list[dict[str, torch.Tensor]] = []
    for batch_id in range(batches):
        _set_seed(seed + 1009 * batch_id)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        instances = batch if isinstance(batch, list) else [batch]
        instances = list(instances[: max(int(spec.batch_size), 1)])
        with torch.no_grad():
            trajs, logits, makespans, entropies = solve_jsp(
                instances,
                batch_size_per_instance=batch_rollouts,
                device=str(device),
                encoder=model.encoder,
                decoder=model.decoder,
                use_greedy=bool(getattr(model, "use_greedy", False)),
            )
        num_instances = len(instances)
        num_steps = int(trajs.size(1))
        step_log_prob = _mgl_jssp_selected_log_prob_step(logits, trajs)
        log_prob_f = step_log_prob.sum(dim=-1).view(num_instances, batch_rollouts).float()
        reward_f = (-makespans.view(num_instances, batch_rollouts)).float()
        objective = _rl4co_objective_from_reward(reward_f, spec.hf)
        seq_len = torch.full_like(log_prob_f, float(num_steps))
        entropy_total = entropies.view(num_instances, batch_rollouts, num_steps).sum(dim=-1).float()
        extra = {
            "advantage": reward_f - reward_f.mean(dim=1, keepdim=True),
            "seq_len": seq_len,
            "log_prob_mean": log_prob_f / seq_len.clamp_min(1.0),
            "entropy": entropy_total,
            "entropy_mean": entropy_total / seq_len.clamp_min(1.0),
        }
        caches.append(extract_feature_cache(objective, log_prob_f, extra=extra))
    return caches


def _make_spec(base_spec: Any, problem: str, scale: float, pomo_size: int | None) -> Any:
    spec = _target_spec(base_spec, problem, scale)
    if pomo_size is None:
        return spec
    hf = replace(spec.hf, pomo_size=int(pomo_size))
    return replace(spec, hf=hf)


def collect(*, problems: list[str], batches: int, seed: int, device: str, max_pairs: int, bins: int) -> dict[str, Any]:
    if "jssp10x10" in problems:
        _ensure_jssp_data(REPO_ROOT)
    torch_device = torch.device(device)
    specs = build_problem_specs(device, batches)
    pair_builders = {problem: _load_pair(PAIR_PATHS[problem])[0] for problem in problems}
    pair_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    rank_curve_rows: list[dict[str, Any]] = []
    margin_curve_rows: list[dict[str, Any]] = []
    raw_margin_curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for problem in problems:
        for condition, label, scale, ckpt_rel, pomo_size in CONDITIONS[problem]:
            ckpt_path = (REPO_ROOT / ckpt_rel).resolve()
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint for {label}: {ckpt_path}")
            spec = _make_spec(specs[problem], problem, scale, pomo_size)
            target_size = int(spec.hf.train_problem_size)
            print(f"[rollout] {label} target_size={target_size} ckpt={ckpt_rel}", flush=True)
            if problem == "jssp10x10":
                caches = _rollout_jssp(spec, ckpt_path, batches=batches, seed=seed + int(scale * 1000), device=torch_device)
            else:
                caches = _rollout_routing_or_ffsp(spec, ckpt_path, batches=batches, seed=seed + int(scale * 1000), device=torch_device)

            all_delta: list[np.ndarray] = []
            all_weight: list[np.ndarray] = []
            all_logp_gap: list[np.ndarray] = []
            for cache_id, fc in enumerate(caches):
                pref = pair_builders[problem](fc)
                if pref.pair_idx is None or pref.num_examples() <= 0:
                    continue
                b, w, l = pref.pair_idx
                objective = fc["objective"]
                log_prob = fc["log_prob"]
                batch_size, k = objective.shape
                sorted_idx = objective.argsort(dim=1, descending=False)
                ranks = torch.empty_like(sorted_idx)
                rank_values = torch.arange(k, device=objective.device)[None, :].expand(batch_size, k)
                ranks.scatter_(1, sorted_idx, rank_values)
                delta = objective[b, l] - objective[b, w]
                rank_span = (ranks[b, l] - ranks[b, w]).float()
                rank_span_norm = rank_span / max(k - 1, 1)
                logp_gap = log_prob[b, w] - log_prob[b, l]
                logp_std = fc["instance_log_prob_std"][b].clamp_min(1e-6)
                margin_norm = logp_gap.abs() / logp_std
                if isinstance(pref.weight, torch.Tensor):
                    weight = pref.weight.detach().float().reshape(-1)
                else:
                    weight = torch.ones_like(delta, dtype=objective.dtype)
                arrays = _subsample_pair_arrays(
                    {
                        "delta_cost": _as_np(delta),
                        "weight": _as_np(weight),
                        "logp_gap": _as_np(logp_gap),
                        "rank_span": _as_np(rank_span),
                        "rank_span_norm": _as_np(rank_span_norm),
                        "margin_norm": _as_np(margin_norm),
                    },
                    max_pairs=max_pairs,
                    seed=seed + 97 * cache_id,
                )
                all_delta.append(arrays["delta_cost"])
                all_weight.append(arrays["weight"])
                all_logp_gap.append(arrays["logp_gap"])
                # Reuse the sampled arrays below through per-cache pair rows.
                for d, wv, lg, rs, rsn, mn in zip(
                    arrays["delta_cost"],
                    arrays["weight"],
                    arrays["logp_gap"],
                    arrays["rank_span"],
                    arrays["rank_span_norm"],
                    arrays["margin_norm"],
                ):
                    pair_rows.append(
                        {
                            "problem": problem,
                            "problem_label": PROBLEM_LABELS[problem],
                            "condition": condition,
                            "condition_label": label,
                            "target_size": target_size,
                            "checkpoint": ckpt_rel,
                            "delta_cost": float(d),
                            "weight": float(wv),
                            "logp_gap": float(lg),
                            "rank_span": float(rs),
                            "rank_span_norm": float(rsn),
                            "margin_norm": float(mn),
                        }
                    )

            delta_np = np.concatenate(all_delta) if all_delta else np.asarray([], dtype=float)
            weight_np = np.concatenate(all_weight) if all_weight else np.asarray([], dtype=float)
            logp_np = np.concatenate(all_logp_gap) if all_logp_gap else np.asarray([], dtype=float)
            subset_rows = [r for r in pair_rows if r["problem"] == problem and r["condition"] == condition]
            rank_np = np.asarray([r["rank_span_norm"] for r in subset_rows], dtype=float)
            margin_np = np.asarray([r["margin_norm"] for r in subset_rows], dtype=float)
            raw_margin_np = np.asarray([abs(float(r["logp_gap"])) for r in subset_rows], dtype=float)
            subset_weight_np = np.asarray([r["weight"] for r in subset_rows], dtype=float)
            for row in _binned_curve(delta_np, weight_np, bins):
                curve_rows.append(
                    {
                        "problem": problem,
                        "problem_label": PROBLEM_LABELS[problem],
                        "condition": condition,
                        "condition_label": label,
                        "target_size": target_size,
                        **row,
                    }
                )
            for row in _binned_xy_curve(rank_np, subset_weight_np, bins):
                rank_curve_rows.append(
                    {
                        "problem": problem,
                        "problem_label": PROBLEM_LABELS[problem],
                        "condition": condition,
                        "condition_label": label,
                        "target_size": target_size,
                        **row,
                    }
                )
            for row in _binned_xy_curve(margin_np, subset_weight_np, bins):
                margin_curve_rows.append(
                    {
                        "problem": problem,
                        "problem_label": PROBLEM_LABELS[problem],
                        "condition": condition,
                        "condition_label": label,
                        "target_size": target_size,
                        **row,
                    }
                )
            for row in _binned_xy_curve(raw_margin_np, subset_weight_np, bins):
                raw_margin_curve_rows.append(
                    {
                        "problem": problem,
                        "problem_label": PROBLEM_LABELS[problem],
                        "condition": condition,
                        "condition_label": label,
                        "target_size": target_size,
                        **row,
                    }
                )
            delta_stats = _summary(delta_np)
            weight_stats = _summary(weight_np)
            corr = float(np.corrcoef(delta_np, weight_np)[0, 1]) if delta_np.size > 2 and np.std(delta_np) > 1e-12 and np.std(weight_np) > 1e-12 else float("nan")
            rank_corr = float(np.corrcoef(rank_np, subset_weight_np)[0, 1]) if rank_np.size > 2 and np.std(rank_np) > 1e-12 and np.std(subset_weight_np) > 1e-12 else float("nan")
            margin_corr = float(np.corrcoef(margin_np, subset_weight_np)[0, 1]) if margin_np.size > 2 and np.std(margin_np) > 1e-12 and np.std(subset_weight_np) > 1e-12 else float("nan")
            raw_margin_corr = float(np.corrcoef(raw_margin_np, subset_weight_np)[0, 1]) if raw_margin_np.size > 2 and np.std(raw_margin_np) > 1e-12 and np.std(subset_weight_np) > 1e-12 else float("nan")
            summary_rows.append(
                {
                    "problem": problem,
                    "condition": condition,
                    "condition_label": label,
                    "target_size": target_size,
                    "checkpoint": ckpt_rel,
                    "pair_count_sampled": int(delta_np.size),
                    "delta_cost_mean": delta_stats["mean"],
                    "delta_cost_median": delta_stats["median"],
                    "delta_cost_q10": delta_stats["q10"],
                    "delta_cost_q90": delta_stats["q90"],
                    "weight_mean": weight_stats["mean"],
                    "weight_median": weight_stats["median"],
                    "weight_q10": weight_stats["q10"],
                    "weight_q90": weight_stats["q90"],
                    "corr_delta_cost_weight": corr,
                    "corr_rank_span_weight": rank_corr,
                    "corr_margin_norm_weight": margin_corr,
                    "corr_raw_margin_weight": raw_margin_corr,
                    "raw_margin_median": float(np.median(raw_margin_np)) if raw_margin_np.size else float("nan"),
                    "raw_margin_q90": float(np.quantile(raw_margin_np, 0.90)) if raw_margin_np.size else float("nan"),
                }
            )
    return {
        "pairs": pair_rows,
        "curves": curve_rows,
        "rank_curves": rank_curve_rows,
        "margin_curves": margin_curve_rows,
        "raw_margin_curves": raw_margin_curve_rows,
        "summary": summary_rows,
        "problems": problems,
    }


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
    pair_rows = results["pairs"]
    curve_rows = results["curves"]
    rank_curve_rows = results["rank_curves"]
    margin_curve_rows = results["margin_curves"]
    raw_margin_curve_rows = results["raw_margin_curves"]
    problems = results["problems"]
    fig, axes = plt.subplots(len(problems), 1, figsize=(7.8, 2.35 * len(problems)), constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        for condition in ["search", "transfer"]:
            pts = [r for r in pair_rows if r["problem"] == problem and r["condition"] == condition]
            if not pts:
                continue
            rng = np.random.default_rng(100 + len(pts))
            show_n = min(len(pts), 4500)
            idx = rng.choice(len(pts), size=show_n, replace=False) if len(pts) > show_n else np.arange(len(pts))
            x = np.asarray([pts[i]["delta_cost"] for i in idx], dtype=float)
            y = np.asarray([pts[i]["weight"] for i in idx], dtype=float)
            ax.scatter(x, y, s=5, alpha=0.10, color=COLORS[condition], linewidths=0)
            curve = [r for r in curve_rows if r["problem"] == problem and r["condition"] == condition]
            cx = np.asarray([r["delta_cost_mid"] for r in curve], dtype=float)
            cy = np.asarray([r["weight_mean"] for r in curve], dtype=float)
            lo = np.asarray([r["weight_q25"] for r in curve], dtype=float)
            hi = np.asarray([r["weight_q75"] for r in curve], dtype=float)
            order = np.argsort(cx)
            label = next((r["condition_label"] for r in curve), condition)
            ax.plot(cx[order], cy[order], color=COLORS[condition], lw=2.0, marker="o", ms=2.8, label=label)
            ax.fill_between(cx[order], lo[order], hi[order], color=COLORS[condition], alpha=0.14, linewidth=0)
        ax.set_title(f"{PROBLEM_LABELS[problem]}: discovered weight over checkpoint-generated trajectory pairs")
        ax.set_xlabel(r"$\Delta$ cost = cost(loser trajectory) - cost(winner trajectory)")
        ax.set_ylabel("weight")
        ax.legend(frameon=False, loc="best")
    fig.suptitle("The same weighting stage induces a scale-dependent measure over actual checkpoint rollouts", fontweight="bold")
    fig.savefig(out_dir / "delta_cost_vs_weight_ckpt.png", bbox_inches="tight")
    fig.savefig(out_dir / "delta_cost_vs_weight_ckpt.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(problems), 2, figsize=(11.2, 2.45 * len(problems)), constrained_layout=True)
    if len(problems) == 1:
        axes = np.asarray([axes])
    for row_idx, problem in enumerate(problems):
        for col_idx, (feature, curves, xlabel, title_suffix) in enumerate(
            [
                ("delta_cost", curve_rows, r"$\Delta$ cost = cost(loser) - cost(winner)", "cost gap"),
                ("raw_abs_logp_gap", raw_margin_curve_rows, r"raw policy margin $|\ell_w-\ell_l|$", "raw policy margin"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            if feature == "raw_abs_logp_gap":
                row_values = [abs(float(r["logp_gap"])) for r in pair_rows if r["problem"] == problem and np.isfinite(float(r["logp_gap"]))]
            else:
                row_values = [float(r[feature]) for r in pair_rows if r["problem"] == problem and np.isfinite(float(r[feature]))]
            x_hi = float(np.quantile(row_values, 0.995)) if row_values else 1.0
            for condition in ["search", "transfer"]:
                pts = [r for r in pair_rows if r["problem"] == problem and r["condition"] == condition]
                if not pts:
                    continue
                rng = np.random.default_rng(1700 + row_idx * 31 + col_idx * 7 + len(pts))
                show_n = min(len(pts), 3000)
                idx = rng.choice(len(pts), size=show_n, replace=False) if len(pts) > show_n else np.arange(len(pts))
                if feature == "raw_abs_logp_gap":
                    x = np.asarray([abs(float(pts[i]["logp_gap"])) for i in idx], dtype=float)
                else:
                    x = np.asarray([pts[i][feature] for i in idx], dtype=float)
                y = np.asarray([pts[i]["weight"] for i in idx], dtype=float)
                ax.scatter(x, y, s=4.5, alpha=0.08, color=COLORS[condition], linewidths=0)
                curve = [r for r in curves if r["problem"] == problem and r["condition"] == condition]
                if feature == "delta_cost":
                    cx = np.asarray([r["delta_cost_mid"] for r in curve], dtype=float)
                else:
                    cx = np.asarray([r["x_mid"] for r in curve], dtype=float)
                cy = np.asarray([r["weight_mean"] for r in curve], dtype=float)
                lo = np.asarray([r["weight_q25"] for r in curve], dtype=float)
                hi = np.asarray([r["weight_q75"] for r in curve], dtype=float)
                order = np.argsort(cx)
                label = next((r["condition_label"] for r in curve), condition)
                ax.plot(cx[order], cy[order], color=COLORS[condition], lw=2.0, marker="o", ms=2.7, label=label)
                ax.fill_between(cx[order], lo[order], hi[order], color=COLORS[condition], alpha=0.13, linewidth=0)
            ax.set_xlim(left=0.0, right=max(x_hi * 1.05, 1e-6))
            ax.set_title(f"{PROBLEM_LABELS[problem]} over {title_suffix}")
            ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel("weight")
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, loc="best")
    fig.suptitle("Checkpoint-generated pair measure: objective gap and raw policy-margin axes", fontweight="bold")
    fig.savefig(out_dir / "delta_cost_rawmargin_vs_weight_4x2_ckpt.png", bbox_inches="tight")
    fig.savefig(out_dir / "delta_cost_rawmargin_vs_weight_4x2_ckpt.pdf", bbox_inches="tight")
    plt.close(fig)

    compare = [p for p in ["cvrp100", "ffsp100"] if p in problems]
    if compare:
        fig, axes = plt.subplots(1, len(compare), figsize=(4.4 * len(compare), 3.15), sharey=True, constrained_layout=True)
        if len(compare) == 1:
            axes = [axes]
        for ax, problem in zip(axes, compare):
            for condition in ["search", "transfer"]:
                pts = [r for r in pair_rows if r["problem"] == problem and r["condition"] == condition]
                if not pts:
                    continue
                rng = np.random.default_rng(900 + len(pts))
                show_n = min(len(pts), 4500)
                idx = rng.choice(len(pts), size=show_n, replace=False) if len(pts) > show_n else np.arange(len(pts))
                x = np.asarray([pts[i]["rank_span_norm"] for i in idx], dtype=float)
                y = np.asarray([pts[i]["weight"] for i in idx], dtype=float)
                ax.scatter(x, y, s=5, alpha=0.10, color=COLORS[condition], linewidths=0)
                curve = [r for r in rank_curve_rows if r["problem"] == problem and r["condition"] == condition]
                cx = np.asarray([r["x_mid"] for r in curve], dtype=float)
                cy = np.asarray([r["weight_mean"] for r in curve], dtype=float)
                lo = np.asarray([r["weight_q25"] for r in curve], dtype=float)
                hi = np.asarray([r["weight_q75"] for r in curve], dtype=float)
                order = np.argsort(cx)
                label = next((r["condition_label"] for r in curve), condition)
                ax.plot(cx[order], cy[order], color=COLORS[condition], lw=2.1, marker="o", ms=3.0, label=label)
                ax.fill_between(cx[order], lo[order], hi[order], color=COLORS[condition], alpha=0.14, linewidth=0)
            ax.set_title(f"{PROBLEM_LABELS[problem]}: weight over rank span")
            ax.set_xlabel("normalized rank span = (rank_loser - rank_winner) / (K - 1)")
            ax.grid(True, alpha=0.24)
        axes[0].set_ylabel("weight")
        axes[-1].legend(frameon=False, loc="best")
        fig.suptitle("A cost-free axis separates CVRP and FFSP weighting behavior", fontweight="bold")
        fig.savefig(out_dir / "rank_span_vs_weight_cvrp_ffsp_ckpt.png", bbox_inches="tight")
        fig.savefig(out_dir / "rank_span_vs_weight_cvrp_ffsp_ckpt.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--max-pairs", type=int, default=60000)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--problems", nargs="*", default=PROBLEM_ORDER)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "paper_materials" / "weighting_delta_cost_ckpt"))
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p for p in args.problems if p in PROBLEM_ORDER]
    out_dir = Path(args.out_dir)
    results = collect(
        problems=problems,
        batches=max(1, int(args.batches)),
        seed=int(args.seed),
        device=device,
        max_pairs=max(1000, int(args.max_pairs)),
        bins=max(4, int(args.bins)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "delta_cost_weight_pairs.csv", results["pairs"])
    _write_csv(out_dir / "delta_cost_weight_binned.csv", results["curves"])
    _write_csv(out_dir / "rank_span_weight_binned.csv", results["rank_curves"])
    _write_csv(out_dir / "margin_norm_weight_binned.csv", results["margin_curves"])
    _write_csv(out_dir / "raw_margin_weight_binned.csv", results["raw_margin_curves"])
    _write_csv(out_dir / "delta_cost_weight_summary.csv", results["summary"])
    plot(results, out_dir)
    (out_dir / "README.md").write_text(
        "# Delta-Cost vs Weight on Checkpoint Rollouts\n\n"
        "This analysis loads downloaded weighting checkpoints, rolls out candidate trajectories, applies the discovered weighting builder, and plots pair weight against the actual pair-level cost gap produced by the checkpoint policy.\n\n"
        "Main figure: `delta_cost_vs_weight_ckpt.png`.\n\n"
        "Combined 4x2 figure: `delta_cost_rawmargin_vs_weight_4x2_ckpt.png`.\n\n"
        "CVRP-vs-FFSP cost-free comparison: `rank_span_vs_weight_cvrp_ffsp_ckpt.png`.\n\n"
        "The columns are raw pair-level `delta_cost = cost_loser - cost_winner` and raw policy margin `abs(logp_w - logp_l)`. The y-axis is the discovered builder's pair weight. Lines are quantile-binned means; shaded regions are interquartile bands; faint points are sampled pairs. Normalized-margin data are still exported as `margin_norm_weight_binned.csv`, but are not used in the main figure.\n",
        encoding="utf-8",
    )
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
