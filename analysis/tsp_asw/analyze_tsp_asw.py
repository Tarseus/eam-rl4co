from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path(__file__).resolve().parents[2] / ".cache/matplotlib").resolve())
)
import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


REPO_ROOT = Path(__file__).resolve().parents[2]
PTP_ROOT = REPO_ROOT / "PTP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PTP_ROOT) not in sys.path:
    sys.path.insert(0, str(PTP_ROOT))

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts/downloads_archive/20260425-154028/final_checkpoints/g49/baseline/tsp100_epoch_135.ckpt"
)
DEFAULT_LOSS = REPO_ROOT / "artifacts/loss_archive/tsp100_best_current/best_loss.json"
DEFAULT_BUILDER = (
    REPO_ROOT / "runs/pref_builder_weight_search_tsp100/20260414-113757/best_builder.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class LossParameters:
    alpha: float
    scale: float
    beta: float
    clamp_abs: float


@dataclass(frozen=True)
class BuilderParameters:
    clamp_lo: float
    clamp_hi: float


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_parameters(loss_path: Path, builder_path: Path, alpha: float) -> tuple[LossParameters, BuilderParameters]:
    loss_payload = _load_json(loss_path)
    loss_code = str(loss_payload.get("ir", loss_payload).get("code", ""))
    builder_payload = _load_json(builder_path)
    builder_code = str(builder_payload.get("ir", builder_payload).get("code", ""))

    scale = 1.9496803035382082
    beta = 0.05
    clamp_abs = 20.0
    clamp_lo = 0.2
    clamp_hi = 2.5

    expected_loss_fragments = (
        "gap / (1.0 + ops.abs(gap))",
        "margin = (lpw - lpl) * (1.0 - beta * norm_gap)",
        "ops.clamp(x, -20.0, 20.0)",
    )
    expected_builder_fragments = (
        "gap_scaled = gap / instance_obj_mad[b_idx]",
        "margin_norm = margin_abs / instance_log_prob_std[b_idx]",
        "raw_weight = gap_scaled * margin_norm * regret_scale",
        "clamp(min=clamp_lo, max=clamp_hi)",
    )
    if not all(fragment in loss_code for fragment in expected_loss_fragments):
        raise ValueError(f"Unexpected TSP loss formula in {loss_path}")
    if not all(fragment in builder_code for fragment in expected_builder_fragments):
        raise ValueError(f"Unexpected TSP builder formula in {builder_path}")

    return (
        LossParameters(alpha=alpha, scale=scale, beta=beta, clamp_abs=clamp_abs),
        BuilderParameters(clamp_lo=clamp_lo, clamp_hi=clamp_hi),
    )


def _rollout_config() -> SimpleNamespace:
    return SimpleNamespace(
        env_name="tsp",
        problem="tsp",
        generator_params={},
        env_kwargs={},
        policy_name="pomo",
        policy_kwargs={
            "po4cops_compat": True,
            "embed_dim": 128,
            "num_encoder_layers": 6,
            "decoder_layer_num": 1,
            "qkv_dim": 16,
            "num_heads": 8,
            "feedforward_hidden": 512,
            "tanh_clipping": 50,
            "eval_type": "argmax",
        },
        rollout_strategy="auto",
        objective_sign="neg_reward",
    )


@torch.no_grad()
def _generate_candidate_pools(
    checkpoint: Path,
    *,
    num_instances: int,
    rollout_batch_size: int,
    num_rollouts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from fitness.free_loss_fidelity import (
        _load_policy_weights_from_checkpoint,
        _rl4co_build_env,
        _rl4co_build_policy,
        _rl4co_objective_from_reward,
        _rl4co_rollout,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _rollout_config()
    env = _rl4co_build_env(config, 100).to(device)
    policy, rollout_strategy = _rl4co_build_policy(config, env)
    _load_policy_weights_from_checkpoint(policy, str(checkpoint))
    policy = policy.to(device).eval()

    objectives: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    generated = 0
    while generated < num_instances:
        current_batch = min(rollout_batch_size, num_instances - generated)
        torch.manual_seed(seed + generated)
        reward, log_likelihood = _rl4co_rollout(
            env,
            policy,
            current_batch,
            num_rollouts,
            phase="test",
            rollout_strategy=rollout_strategy,
            device=device,
        )
        objective = _rl4co_objective_from_reward(reward, config)
        objectives.append(objective.detach().cpu().numpy().astype(np.float64))
        log_probs.append(log_likelihood.detach().cpu().numpy().astype(np.float64))
        generated += current_batch
        print(f"Generated {generated}/{num_instances} TSP100 candidate pools", flush=True)

    return np.concatenate(objectives, axis=0), np.concatenate(log_probs, axis=0)


def _load_or_generate_pools(
    cache_path: Path,
    checkpoint: Path,
    *,
    num_instances: int,
    rollout_batch_size: int,
    num_rollouts: int,
    seed: int,
    force_rollout: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file() and not force_rollout:
        payload = np.load(cache_path)
        objective = payload["objective"]
        log_prob = payload["log_prob"]
        if objective.shape != (num_instances, num_rollouts) or log_prob.shape != objective.shape:
            raise ValueError(
                f"Pool cache shape mismatch: objective={objective.shape}, expected={(num_instances, num_rollouts)}. "
                "Use --force-rollout or matching arguments."
            )
        return objective.astype(np.float64), log_prob.astype(np.float64)

    objective, log_prob = _generate_candidate_pools(
        checkpoint,
        num_instances=num_instances,
        rollout_batch_size=rollout_batch_size,
        num_rollouts=num_rollouts,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        objective=objective,
        log_prob=log_prob,
        checkpoint=str(checkpoint),
        seed=seed,
    )
    return objective, log_prob


def _mad(values: np.ndarray, axis: int = 1) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - median), axis=axis)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _build_pair_records(
    objective: np.ndarray,
    log_prob: np.ndarray,
    loss_parameters: LossParameters,
    builder_parameters: BuilderParameters,
) -> dict[str, np.ndarray]:
    eps = 1e-8
    num_instances = objective.shape[0]
    objective_mad = np.maximum(_mad(objective), eps)
    log_prob_std = np.maximum(np.std(log_prob, axis=1, ddof=1), eps)
    robust_scale = np.maximum(1.4826 * _mad(objective), eps)
    best_objective = np.min(objective, axis=1, keepdims=True)
    regret = (objective - best_objective) / robust_scale[:, None]
    pool_regret = np.mean(regret, axis=1)

    records: dict[str, list[np.ndarray]] = {
        "pool": [],
        "delta_o": [],
        "delta_p": [],
        "normalized_gap": [],
        "normalized_margin": [],
        "pool_regret": [],
        "omega": [],
        "gradient_uniform": [],
        "gradient_usw": [],
        "gradient_asw": [],
    }

    base_scale = loss_parameters.alpha
    for pool_index in range(num_instances):
        better, worse = np.nonzero(objective[pool_index, :, None] < objective[pool_index, None, :])
        delta_o = objective[pool_index, worse] - objective[pool_index, better]
        delta_p = log_prob[pool_index, better] - log_prob[pool_index, worse]
        normalized_gap = delta_o / objective_mad[pool_index]
        normalized_margin = np.abs(delta_p) / log_prob_std[pool_index]
        raw_weight = normalized_gap * normalized_margin * pool_regret[pool_index]
        omega = np.clip(raw_weight, builder_parameters.clamp_lo, builder_parameters.clamp_hi)

        bounded_gap = delta_o / (1.0 + np.abs(delta_o))
        discovered_scale = (
            loss_parameters.alpha
            * loss_parameters.scale
            * (1.0 - loss_parameters.beta * bounded_gap)
        )
        score = np.clip(
            discovered_scale * delta_p,
            -loss_parameters.clamp_abs,
            loss_parameters.clamp_abs,
        )
        base_score = base_scale * delta_p

        gradient_uniform = base_scale * _sigmoid(-base_score)
        gradient_usw = discovered_scale * _sigmoid(-score)
        gradient_asw = omega * gradient_usw

        gradient_uniform /= max(float(gradient_uniform.size), 1.0)
        gradient_usw /= max(float(gradient_usw.size), 1.0)
        gradient_asw /= max(float(np.sum(omega)), eps)
        gradient_uniform /= num_instances
        gradient_usw /= num_instances
        gradient_asw /= num_instances

        pair_count = delta_o.size
        records["pool"].append(np.full(pair_count, pool_index, dtype=np.int32))
        records["delta_o"].append(delta_o)
        records["delta_p"].append(delta_p)
        records["normalized_gap"].append(normalized_gap)
        records["normalized_margin"].append(normalized_margin)
        records["pool_regret"].append(np.full(pair_count, pool_regret[pool_index]))
        records["omega"].append(omega)
        records["gradient_uniform"].append(gradient_uniform)
        records["gradient_usw"].append(gradient_usw)
        records["gradient_asw"].append(gradient_asw)

    return {key: np.concatenate(value) for key, value in records.items()}


def _write_pair_records(path: Path, records: dict[str, np.ndarray]) -> None:
    fields = list(records)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for row in zip(*(records[field] for field in fields)):
            writer.writerow(row)


def _pool_weight_statistics(
    records: dict[str, np.ndarray], builder_parameters: BuilderParameters
) -> tuple[dict[str, float], list[dict[str, float]]]:
    pool_rows: list[dict[str, float]] = []
    pools = np.unique(records["pool"])
    for pool in pools:
        mask = records["pool"] == pool
        omega = records["omega"][mask]
        top_count = max(1, int(math.ceil(0.1 * omega.size)))
        top_mass = float(np.sum(np.partition(omega, -top_count)[-top_count:]) / np.sum(omega))
        ess = float(np.sum(omega) ** 2 / np.sum(omega**2))
        pool_rows.append(
            {
                "pool": int(pool),
                "pair_count": int(omega.size),
                "top10_weight_mass": top_mass,
                "ess": ess,
                "ess_ratio": ess / omega.size,
                "clip_lower_rate": float(np.mean(np.isclose(omega, builder_parameters.clamp_lo))),
                "clip_upper_rate": float(np.mean(np.isclose(omega, builder_parameters.clamp_hi))),
            }
        )

    omega = records["omega"]
    top_count = max(1, int(math.ceil(0.1 * omega.size)))
    ess = float(np.sum(omega) ** 2 / np.sum(omega**2))
    ess_ratios = [row["ess_ratio"] for row in pool_rows]
    pooled = {
        "num_pools": int(pools.size),
        "num_pairs": int(omega.size),
        "top10_weight_mass": float(np.sum(np.partition(omega, -top_count)[-top_count:]) / np.sum(omega)),
        "ess": ess,
        "ess_ratio": ess / omega.size,
        "clip_lower_rate": float(np.mean(np.isclose(omega, builder_parameters.clamp_lo))),
        "clip_upper_rate": float(np.mean(np.isclose(omega, builder_parameters.clamp_hi))),
        "rho_normalized_gap": _spearman(omega, records["normalized_gap"]),
        "rho_normalized_margin": _spearman(omega, records["normalized_margin"]),
        "rho_pool_regret": _spearman(omega, records["pool_regret"]),
        "pool_top10_weight_mass_mean": float(np.mean([row["top10_weight_mass"] for row in pool_rows])),
        "pool_ess_ratio_mean": float(np.mean(ess_ratios)),
        "pool_ess_ratio_std": float(np.std(ess_ratios, ddof=1)) if len(ess_ratios) > 1 else 0.0,
    }
    return pooled, pool_rows


def _write_dict_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile_allocation(
    records: dict[str, np.ndarray], feature: str, bins: int = 6
) -> list[dict[str, float | int | str]]:
    values = records[feature]
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.maximum.accumulate(edges)
    edges[-1] += np.finfo(np.float64).eps * max(1.0, abs(edges[-1]))
    indices = np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)
    rows: list[dict[str, float | int | str]] = []
    methods = (
        ("Uniform BT", "gradient_uniform"),
        ("USW", "gradient_usw"),
        ("ASW", "gradient_asw"),
    )
    for method, gradient_key in methods:
        gradient = records[gradient_key]
        total = float(np.sum(gradient))
        for bin_index in range(bins):
            mask = indices == bin_index
            rows.append(
                {
                    "feature": feature,
                    "method": method,
                    "bin": bin_index + 1,
                    "lower": float(edges[bin_index]),
                    "upper": float(edges[bin_index + 1]),
                    "pair_fraction": float(np.mean(mask)),
                    "gradient_mass_fraction": float(np.sum(gradient[mask]) / total),
                }
            )
    return rows


def _binned_pair_mass_heatmap(
    delta_p: np.ndarray,
    normalized_gap: np.ndarray,
    omega: np.ndarray,
    *,
    x_bins: int = 24,
    y_bins: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x_limit = max(
        abs(float(np.quantile(delta_p, 0.01))),
        abs(float(np.quantile(delta_p, 0.99))),
    )
    y_limit = float(np.quantile(normalized_gap, 0.99))
    visible = (
        (delta_p >= -x_limit)
        & (delta_p <= x_limit)
        & (normalized_gap >= 0.0)
        & (normalized_gap <= y_limit)
    )
    x_edges = np.linspace(-x_limit, x_limit, x_bins + 1)
    y_edges = np.linspace(0.0, y_limit, y_bins + 1)
    x_idx = np.clip(np.digitize(delta_p[visible], x_edges[1:-1]), 0, x_bins - 1)
    y_idx = np.clip(np.digitize(normalized_gap[visible], y_edges[1:-1]), 0, y_bins - 1)
    mass = np.zeros((y_bins, x_bins), dtype=np.float64)
    counts = np.zeros((y_bins, x_bins), dtype=np.int64)
    np.add.at(mass, (y_idx, x_idx), omega[visible])
    np.add.at(counts, (y_idx, x_idx), 1)
    mass_percent = 100.0 * mass / np.sum(omega)
    visible_mass_fraction = float(np.sum(omega[visible]) / np.sum(omega))
    return mass_percent, counts, x_edges, y_edges, visible_mass_fraction


def _response_curves(
    delta_p: np.ndarray,
    gap_values: list[tuple[str, float]],
    parameters: LossParameters,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    margin_limit = max(2.0, float(np.quantile(np.abs(delta_p), 0.99)))
    grid = np.linspace(-margin_limit, margin_limit, 500)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, gap in gap_values:
        bounded_gap = gap / (1.0 + abs(gap))
        slope = parameters.alpha * parameters.scale * (1.0 - parameters.beta * bounded_gap)
        raw_score = slope * grid
        clipped_score = np.clip(raw_score, -parameters.clamp_abs, parameters.clamp_abs)
        semantic_loss = np.logaddexp(0.0, -clipped_score)
        derivative = slope * _sigmoid(-clipped_score)
        derivative[np.abs(raw_score) >= parameters.clamp_abs] = 0.0
        curves[label] = (semantic_loss, np.abs(derivative))
    return grid, curves


def _plot_figure(
    output_path: Path,
    records: dict[str, np.ndarray],
    loss_parameters: LossParameters,
    gap_allocations: list[dict],
    margin_allocations: list[dict],
    usw_gap: float,
    asw_gap: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "legend.fontsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
    method_colors = {"Uniform BT": "#8C8C8C", "USW": "#56B4E9", "ASW": "#D55E00"}

    gap_quantiles = [
        ("0", 0.0),
        ("median", float(np.median(records["delta_o"]))),
        ("P90", float(np.quantile(records["delta_o"], 0.90))),
        ("max", float(np.max(records["delta_o"]))),
    ]
    grid, curves = _response_curves(records["delta_p"], gap_quantiles, loss_parameters)

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 5.25))
    fig.subplots_adjust(wspace=0.42, hspace=0.48)

    ax = axes[0, 0]
    for color, (label, gap) in zip(colors, gap_quantiles):
        ax.plot(grid, curves[label][0], color=color, label=f"{label}: {gap:.3g}")
    ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel(r"Policy margin $\Delta p$")
    ax.set_ylabel(r"Semantic loss $s_{\mathrm{T}}=-\log\sigma(z)$")
    ax.set_title("(a) Actual semantic loss")
    ax.legend(title=r"Objective gap $\Delta o$", title_fontsize=7.6, frameon=False)

    ax = axes[0, 1]
    for color, (label, _) in zip(colors, gap_quantiles):
        ax.plot(grid, curves[label][1], color=color, label=label)
    ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel(r"Policy margin $\Delta p$")
    ax.set_ylabel(r"$|\partial s_{\mathrm{T}}/\partial\Delta p|$")
    ax.set_title("(b) Actual loss gradient")

    ax = axes[0, 2]
    pair_mass, counts, x_edges, y_edges, visible_mass_fraction = _binned_pair_mass_heatmap(
        records["delta_p"], records["normalized_gap"], records["omega"]
    )
    masked = np.ma.masked_where(counts == 0, pair_mass)
    positive_mass = pair_mass[pair_mass > 0]
    color_max = float(np.quantile(positive_mass, 0.98)) if positive_mass.size else 1.0
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        masked,
        cmap="YlOrRd",
        norm=Normalize(vmin=0.0, vmax=color_max),
        shading="auto",
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    log_counts = np.log1p(counts.astype(np.float64))
    positive_counts = log_counts[log_counts > 0]
    if positive_counts.size:
        contour_levels = np.unique(np.quantile(positive_counts, [0.50, 0.75, 0.90]))
        if contour_levels.size:
            ax.contour(
                x_centers,
                y_centers,
                log_counts,
                levels=contour_levels,
                colors="#4D4D4D",
                linewidths=0.7,
                alpha=0.85,
            )
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, label="ASW mass per bin (%)")
    ax.axvline(0.0, color="white", lw=0.8, ls=":", alpha=0.9)
    ax.set_xlabel(r"Policy margin $\Delta p$")
    ax.set_ylabel(r"MAD-normalized gap $\Delta o/m_o$")
    ax.set_title("(c) Real pairs and ASW mass")
    ax.text(
        0.03,
        0.97,
        f"contours: pair density\nshown mass: {100 * visible_mass_fraction:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#3F3F3F",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )

    def plot_allocations(ax: plt.Axes, rows: list[dict], title: str, xlabel: str) -> None:
        for method in ("Uniform BT", "USW", "ASW"):
            method_rows = [row for row in rows if row["method"] == method]
            x = [int(row["bin"]) for row in method_rows]
            y = [100.0 * float(row["gradient_mass_fraction"]) for row in method_rows]
            ax.plot(x, y, marker="o", ms=3.5, color=method_colors[method], label=method)
        ax.axhline(100.0 / 6.0, color="#999999", lw=0.8, ls=":")
        ax.set_xticks(range(1, 7))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Gradient mass (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)

    plot_allocations(
        axes[1, 0], gap_allocations, "(d) Gradient by objective gap", "Normalized-gap quantile bin"
    )
    axes[1, 0].legend(frameon=False, ncol=1)
    plot_allocations(
        axes[1, 1], margin_allocations, "(e) Gradient by policy margin", "Normalized-margin quantile bin"
    )

    ax = axes[1, 2]
    bars = ax.bar(
        ["USW", "ASW"],
        [usw_gap, asw_gap],
        color=[method_colors["USW"], method_colors["ASW"]],
        width=0.62,
    )
    for bar, value in zip(bars, [usw_gap, asw_gap]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}%", ha="center", fontsize=8.5)
    improvement = 100.0 * (usw_gap - asw_gap) / usw_gap
    ax.text(0.5, max(usw_gap, asw_gap) * 0.58, f"{improvement:.1f}% relative\nreduction", ha="center")
    ax.set_ylabel("TSP100 optimality gap (%)")
    ax.set_title("(f) Final performance bridge")
    ax.set_ylim(0.0, max(usw_gap, asw_gap) * 1.28)
    ax.grid(axis="y", alpha=0.18)
    ax.text(
        0.5,
        0.02,
        "empirical association; not a causal estimate",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#555555",
    )

    fig.suptitle("TSP100: actual loss landscape and pool-aware gradient allocation", y=1.02, fontsize=11.4)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def _write_latex_table(path: Path, stats: dict[str, float]) -> None:
    text = "\n".join(
        [
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Top-10\% mass & ESS/N & Clip-low & Clip-high & $\rho_{\Delta o/m_o}$ & $\rho_{|\Delta p|/\sigma_p}$ & $\rho_{\bar q_o}$ & Pairs \\",
            r"\midrule",
            (
                f"{100 * stats['top10_weight_mass']:.1f}\\% & "
                f"{stats['ess_ratio']:.3f} & "
                f"{100 * stats['clip_lower_rate']:.1f}\\% & "
                f"{100 * stats['clip_upper_rate']:.1f}\\% & "
                f"{stats['rho_normalized_gap']:.3f} & "
                f"{stats['rho_normalized_margin']:.3f} & "
                f"{stats['rho_pool_regret']:.3f} & "
                f"{int(stats['num_pairs']):,} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _write_report(
    path: Path,
    stats: dict[str, float],
    checkpoint: Path,
    usw_gap: float,
    asw_gap: float,
) -> None:
    relative_gain = 100.0 * (usw_gap - asw_gap) / usw_gap
    lines = [
        "# TSP100 ASW analysis",
        "",
        "## Protocol",
        "",
        f"- Candidate pools: {int(stats['num_pools'])} real TSP100 instances, 100 policy multistarts each.",
        f"- Snapshot checkpoint: `{checkpoint.relative_to(REPO_ROOT).as_posix()}`.",
        f"- Valid ordered pairs: {int(stats['num_pairs']):,}.",
        "- Pair construction, normalization, clipping, and gradient aggregation are all instance-local.",
        "",
        "## Exact analyzed rules",
        "",
        "Panels (a-b) plot the actual per-pair semantic loss used for training and its true policy-margin gradient:",
        "",
        r"`z = clip(alpha * 1.94968 * Delta p * (1 - 0.05 * Delta o / (1 + |Delta o|)), -20, 20)`.",
        "",
        r"`s_T = -log(sigmoid(z)) = softplus(-z)`.",
        "",
        r"`|d s_T / d Delta p| = alpha * 1.94968 * (1 - 0.05 * Delta o / (1 + |Delta o|)) * sigmoid(-z)` "
        "inside the unclipped region.",
        "",
        "The ASW builder keeps all objective-ordered pairs and applies:",
        "",
        r"`omega_i = clip((Delta o_i / MAD_o) * (|Delta p_i| / sigma_p) * mean_pool_regret, 0.2, 2.5)`.",
        "",
        "## Main observations",
        "",
        "- The actual semantic loss decreases monotonically with the winner-minus-loser policy margin for every observed cost gap.",
        "- The objective gap only changes the local slope through `gap / (1 + abs(gap))`; its influence therefore saturates. "
        "For nonnegative ordered-pair gaps, the multiplicative slope factor stays in `(0.95, 1]`, so preference direction cannot flip.",
        "- Panel (c) projects real training pairs into `(Delta p, Delta o / MAD_o)` space: contours show empirical pair density, "
        "while color shows the fraction of total ASW weight assigned to each bin.",
        f"- The top 10% of pairs receive {100 * stats['top10_weight_mass']:.1f}% of total ASW mass.",
        f"- ESS/N is {stats['ess_ratio']:.3f}; lower/upper clipping rates are "
        f"{100 * stats['clip_lower_rate']:.1f}%/{100 * stats['clip_upper_rate']:.1f}%.",
        f"- Spearman correlations of weight with normalized gap, normalized margin, and pool regret are "
        f"{stats['rho_normalized_gap']:.3f}, {stats['rho_normalized_margin']:.3f}, and {stats['rho_pool_regret']:.3f}.",
        "- The gradient panels normalize each method to its own total gradient mass, so they show redistribution rather than global-norm inflation.",
        f"- The supplied final TSP100 gaps are USW {usw_gap:.3f}% and ASW {asw_gap:.3f}% "
        f"({relative_gain:.1f}% relative reduction). The empirical redistribution of pairwise gradient mass is consistent "
        "with this improvement, but the post-hoc analysis alone does not establish causality.",
        "",
        "## Evidence chain",
        "",
        "`discovered formula -> loss landscape -> gradient landscape -> real-pair projection -> empirical performance consistency`",
        "",
        "## Artifacts",
        "",
        "- `tsp100_asw_analysis.pdf` and `.png`: six-panel paper figure.",
        "- `tsp100_weight_statistics.csv` and `.tex`: compact concentration table.",
        "- `tsp100_gradient_allocation.csv`: gradient-mass fractions by normalized-gap and normalized-margin quantiles.",
        "- `tsp100_pair_records.csv.gz`: pair-level normalized gaps, margins, pool regret, weights, and gradients.",
        "- `tsp100_candidate_pools.npz`: cached objective and log-probability pools.",
        "",
        "The checkpoint is a fixed Stage-2 starting-policy snapshot, not a time average over training. The analysis therefore identifies where ASW allocates mass on representative real pools without requiring retraining.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the discovered TSP100 loss and ASW rule.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--loss", type=Path, default=DEFAULT_LOSS)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-instances", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--usw-gap", type=float, default=0.125)
    parser.add_argument("--asw-gap", type=float, default=0.088)
    parser.add_argument("--force-rollout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_instances <= 0 or args.rollout_batch_size <= 0 or args.num_rollouts <= 1:
        raise ValueError("num-instances and rollout-batch-size must be positive; num-rollouts must exceed one")
    if args.num_rollouts != 100:
        raise ValueError("This TSP100 analysis requires 100 multistarts to match the training candidate-pool semantics")

    checkpoint = args.checkpoint.resolve()
    loss_path = args.loss.resolve()
    builder_path = args.builder.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (checkpoint, loss_path, builder_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    loss_parameters, builder_parameters = _load_parameters(loss_path, builder_path, args.alpha)
    cache_path = output_dir / "tsp100_candidate_pools.npz"
    objective, log_prob = _load_or_generate_pools(
        cache_path,
        checkpoint,
        num_instances=args.num_instances,
        rollout_batch_size=args.rollout_batch_size,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
        force_rollout=args.force_rollout,
    )
    if objective.ndim != 2 or objective.shape != log_prob.shape:
        raise ValueError(f"Expected matching [instance, rollout] arrays, got {objective.shape} and {log_prob.shape}")
    if not np.isfinite(objective).all() or not np.isfinite(log_prob).all():
        raise ValueError("Candidate pools contain non-finite values")

    records = _build_pair_records(objective, log_prob, loss_parameters, builder_parameters)
    _write_pair_records(output_dir / "tsp100_pair_records.csv.gz", records)

    pooled_stats, pool_stats = _pool_weight_statistics(records, builder_parameters)
    pooled_stats.update(
        {
            "checkpoint": str(checkpoint),
            "loss_artifact": str(loss_path),
            "builder_artifact": str(builder_path),
            "alpha": loss_parameters.alpha,
            "scale": loss_parameters.scale,
            "beta": loss_parameters.beta,
            "clamp_lo": builder_parameters.clamp_lo,
            "clamp_hi": builder_parameters.clamp_hi,
        }
    )
    _write_dict_csv(output_dir / "tsp100_weight_statistics.csv", [pooled_stats])
    _write_dict_csv(output_dir / "tsp100_pool_statistics.csv", pool_stats)
    _write_latex_table(output_dir / "tsp100_weight_statistics.tex", pooled_stats)

    gap_allocations = _quantile_allocation(records, "normalized_gap")
    margin_allocations = _quantile_allocation(records, "normalized_margin")
    gradient_rows = gap_allocations + margin_allocations
    for rows in (gap_allocations, margin_allocations):
        for method in ("Uniform BT", "USW", "ASW"):
            total_mass = sum(
                float(row["gradient_mass_fraction"]) for row in rows if row["method"] == method
            )
            if not math.isclose(total_mass, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(f"Gradient allocation for {method} sums to {total_mass}, expected 1")
    _write_dict_csv(output_dir / "tsp100_gradient_allocation.csv", gradient_rows)

    _plot_figure(
        output_dir / "tsp100_asw_analysis",
        records,
        loss_parameters,
        gap_allocations,
        margin_allocations,
        args.usw_gap,
        args.asw_gap,
    )
    summary = {
        "protocol": {
            "checkpoint": str(checkpoint),
            "num_instances": args.num_instances,
            "num_rollouts": args.num_rollouts,
            "seed": args.seed,
        },
        "loss_parameters": loss_parameters.__dict__,
        "builder_parameters": builder_parameters.__dict__,
        "weight_statistics": pooled_stats,
        "performance": {
            "usw_tsp100_gap_percent": args.usw_gap,
            "asw_tsp100_gap_percent": args.asw_gap,
            "relative_reduction_percent": 100.0 * (args.usw_gap - args.asw_gap) / args.usw_gap,
            "interpretation": (
                "The empirical redistribution of pairwise gradient mass is consistent with the improvement "
                "from USW to ASW; this post-hoc analysis is not a causal estimate."
            ),
        },
    }
    (output_dir / "tsp100_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(
        output_dir / "README.md",
        pooled_stats,
        checkpoint,
        args.usw_gap,
        args.asw_gap,
    )
    print(f"Wrote TSP100 ASW analysis to {output_dir}")


if __name__ == "__main__":
    main()
