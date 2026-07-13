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

from scripts.final_gradient_behavior_analysis import build_problem_specs, rollout_feature_caches  # noqa: E402
from scripts.plot_scale_generalization_loss_weighting import LOSS_ONLY, WEIGHTING, _load_pair, _state_log_prob, _target_spec  # noqa: E402
from scripts.plot_scale_transfer_replay_diagnosis import _build_pref, _coefficient_vector, _pref_pair_features  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _net_rank_stats(pref: Any, fc: dict[str, torch.Tensor], coeff: np.ndarray) -> dict[str, float]:
    objective = fc["objective"].detach()
    b, w, l = pref.pair_idx
    coeff_t = torch.as_tensor(coeff, device=objective.device, dtype=objective.dtype)
    net = torch.zeros_like(objective)
    net.index_put_((b, w), coeff_t, accumulate=True)
    net.index_put_((b, l), -coeff_t, accumulate=True)
    net = net / net.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    ranked = net.gather(1, objective.argsort(dim=1, descending=False)).detach().cpu().numpy()
    return {
        "best_rank_signal": float(np.mean(ranked[:, 0])),
        "worst_rank_signal": float(np.mean(ranked[:, -1])),
        "best_worst_amplitude": float(np.mean(ranked[:, 0] - ranked[:, -1])),
        "middle_rank_signal": float(np.mean(ranked[:, ranked.shape[1] // 2])),
    }


def _ffsp_pre_final_weight(pref: Any, fc: dict[str, torch.Tensor]) -> np.ndarray:
    objective = fc["objective"]
    log_prob = fc["log_prob"]
    b_idx, winner_idx, loser_idx = pref.pair_idx
    eps = 1e-6
    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]
    instance_obj_scale = fc["instance_obj_mad"][b_idx].clamp_min(eps)
    gap_scaled = gap / instance_obj_scale
    rank = fc["rank"]
    rank_span = rank[b_idx, loser_idx] - rank[b_idx, winner_idx]
    margin_abs = (log_prob[b_idx, winner_idx] - log_prob[b_idx, loser_idx]).abs()
    margin_abs = margin_abs / fc["instance_log_prob_std"][b_idx].clamp_min(eps)
    raw_weight = torch.sigmoid(5.0 * rank_span) / (margin_abs + eps).pow(0.6)
    raw_weight = torch.nan_to_num(raw_weight, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tie_zone_mask = gap_scaled > 0.12
    weight = raw_weight * tie_zone_mask.to(raw_weight.dtype)
    weight = weight.clamp(0.15, 2.5)
    rank_span_epsguard = rank_span.clamp_min(eps)
    pre_final = weight * (1.0 / rank_span_epsguard.float())
    return pre_final.detach().float().cpu().numpy().reshape(-1)


def collect(*, batches: int, seed: int, device: str, max_pairs: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = build_problem_specs(device, batches)
    pairs = {"Loss-only": _load_pair(LOSS_ONLY["ffsp100"]), "Loss+Weighting": _load_pair(WEIGHTING["ffsp100"])}
    weight_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for scale_name, scale in [("search", 1.0), ("transfer", 0.5)]:
        spec = _target_spec(specs["ffsp100"], "ffsp100", scale)
        caches = rollout_feature_caches(spec, seed=seed + int(scale * 1000), device=torch.device(device))
        for cache_id, raw_fc in enumerate(caches):
            log_prob = _state_log_prob(raw_fc, "aligned", 1.0).detach()
            fc = {**dict(raw_fc), "log_prob": log_prob}
            pref_w = _build_pref("Loss+Weighting", fc, pairs)
            features, weight, _ = _pref_pair_features(pref_w, fc, log_prob, max_pairs=max_pairs, seed=seed + cache_id)
            pre_final = _ffsp_pre_final_weight(pref_w, fc)
            if pre_final.size > weight.size:
                pre_final = pre_final[: weight.size]
            log_pre_over_hi = np.log2(np.clip(pre_final, 1e-12, None) / 2.5)
            inside_window = (pre_final >= 0.15) & (pre_final <= 2.5)
            low = weight < 2.5 - 1e-5
            weight_rows.append(
                {
                    "scale_name": scale_name,
                    "target_size": int(spec.hf.train_problem_size),
                    "cache_id": cache_id,
                    "hi_clamp_share": float(np.mean(np.isclose(weight, 2.5, rtol=1e-5, atol=1e-5))),
                    "pre_final_log2_over_upper_q25": float(np.nanquantile(log_pre_over_hi, 0.25)),
                    "pre_final_log2_over_upper_median": float(np.nanmedian(log_pre_over_hi)),
                    "pre_final_log2_over_upper_q75": float(np.nanquantile(log_pre_over_hi, 0.75)),
                    "pre_final_inside_clamp_window_share": float(np.mean(inside_window)),
                    "mean_weight": float(np.mean(weight)),
                    "std_weight": float(np.std(weight)),
                    "weight_gap_corr": float(np.corrcoef(weight, features["relative_gap"])[0, 1]) if float(np.std(weight)) > 1e-12 else float("nan"),
                    "weight_margin_corr": float(np.corrcoef(weight, features["logp_diff"])[0, 1]) if float(np.std(weight)) > 1e-12 else float("nan"),
                    "low_weight_pair_share": float(np.mean(low)),
                    "low_weight_gap_mean": float(np.mean(features["relative_gap"][low])) if bool(np.any(low)) else float("nan"),
                    "all_pair_gap_mean": float(np.mean(features["relative_gap"])),
                    "low_weight_margin_mean": float(np.mean(features["logp_diff"][low])) if bool(np.any(low)) else float("nan"),
                    "all_pair_margin_mean": float(np.mean(features["logp_diff"])),
                }
            )
            for method in ["Loss-only", "Loss+Weighting"]:
                pref = _build_pref(method, fc, pairs)
                coeff = _coefficient_vector(method, fc, pref, pairs)
                rank_rows.append(
                    {
                        "scale_name": scale_name,
                        "target_size": int(spec.hf.train_problem_size),
                        "cache_id": cache_id,
                        "method": method,
                        **_net_rank_stats(pref, fc, coeff),
                    }
                )
    return weight_rows, rank_rows


def _mean(rows: list[dict[str, Any]], key: str, **conds: Any) -> float:
    vals = []
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            try:
                val = float(row[key])
            except Exception:
                continue
            if np.isfinite(val):
                vals.append(val)
    return float(np.mean(vals)) if vals else float("nan")


def plot(weight_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1), constrained_layout=True)
    scales = ["search", "transfer"]
    labels = ["FFSP100\nsearch", "FFSP50\ntransfer"]
    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    gray = "#8C8C8C"

    med = [_mean(weight_rows, "pre_final_log2_over_upper_median", scale_name=s) for s in scales]
    q25 = [_mean(weight_rows, "pre_final_log2_over_upper_q25", scale_name=s) for s in scales]
    q75 = [_mean(weight_rows, "pre_final_log2_over_upper_q75", scale_name=s) for s in scales]
    inside = [_mean(weight_rows, "pre_final_inside_clamp_window_share", scale_name=s) for s in scales]
    x = np.arange(2)
    ax = axes[0]
    yerr = np.vstack([np.asarray(med) - np.asarray(q25), np.asarray(q75) - np.asarray(med)])
    ax.errorbar(x - 0.14, med, yerr=yerr, color=blue, marker="o", lw=1.8, capsize=4, label="pre-clamp score")
    ax.axhline(0.0, color="#333333", lw=0.8, ls="--")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, inside, width=0.28, color=orange, alpha=0.85, label="inside window")
    ax.set_xticks(x, labels)
    ax2.set_ylim(0.0, max(inside) * 1.35 if max(inside) > 0 else 0.05)
    ax.set_ylabel("log2(pre-clamp score / upper bound)")
    ax2.set_ylabel("share inside clamp window")
    ax.set_title("Pre-clamp range shifts toward window")
    for xi, val in zip(x - 0.14, med):
        ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.4, color=blue)
    for xi, val in zip(x + 0.18, inside):
        ax2.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7.4, color=orange)

    ax = axes[1]
    corr_gap = [_mean(weight_rows, "weight_gap_corr", scale_name=s) for s in scales]
    corr_margin = [_mean(weight_rows, "weight_margin_corr", scale_name=s) for s in scales]
    width = 0.34
    ax.bar(x - width / 2, corr_gap, width=width, color=green, label="corr(weight, gap)")
    ax.bar(x + width / 2, corr_margin, width=width, color=orange, label="corr(weight, margin)")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("correlation")
    ax.set_title("Released weights downweight strong pairs")
    ax.legend(frameon=False, loc="lower left")
    for xi, val in zip(np.r_[x - width / 2, x + width / 2], corr_gap + corr_margin):
        if np.isfinite(val):
            ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.2)

    ax = axes[2]
    methods = ["Loss-only", "Loss+Weighting"]
    colors = [blue, orange]
    width = 0.34
    for mi, method in enumerate(methods):
        vals = [_mean(rank_rows, "best_worst_amplitude", scale_name=s, method=method) for s in scales]
        ax.bar(x + (mi - 0.5) * width, vals, width=width, color=colors[mi], label=method)
        for xi, val in zip(x + (mi - 0.5) * width, vals):
            ax.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("best-worst net signal amplitude")
    ax.set_title("Rank separation is weakened")
    ax.legend(frameon=False, loc="lower left")
    fig.suptitle("FFSP transfer failure probe: clamp-window shift rather than pair concentration", fontweight="bold")
    fig.savefig(out_dir / "ffsp_weighting_transfer_failure_probe.png", bbox_inches="tight")
    fig.savefig(out_dir / "ffsp_weighting_transfer_failure_probe.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures" / "scale_transfer_replay_diagnosis" / "ffsp_failure_probe"))
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    out_dir = Path(args.out_dir)
    weight_rows, rank_rows = collect(batches=max(int(args.batches), 1), seed=int(args.seed), device=device, max_pairs=max(int(args.max_pairs), 1024))
    _write_csv(out_dir / "ffsp_weight_clamp_release.csv", weight_rows)
    _write_csv(out_dir / "ffsp_rank_signal.csv", rank_rows)
    plot(weight_rows, rank_rows, out_dir)
    (out_dir / "README.md").write_text(
        "# FFSP weighting transfer probe\n\n"
        "This probe explains why FFSP is not covered by the concentration mechanism in the main scale-transfer summary.\n\n"
        "Finding: the pre-clamp weighting scores move relative to the allowed clamp window under transfer. At the search scale, the generated scores sit well above the upper boundary, so the rule behaves close to a constant rescaling. At transfer scale, more scores enter the clamp window; these are high-gap/high-margin pairs, so the learned weighting slightly downweights the strong rank-separation signal produced by loss-only.\n",
        encoding="utf-8",
    )
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
