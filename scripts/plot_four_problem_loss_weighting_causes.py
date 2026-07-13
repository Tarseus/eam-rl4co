from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

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

from fitness.free_loss_fidelity import PrefBatch, extract_feature_cache  # noqa: E402
from ptp_discovery.free_loss_compiler import compile_free_loss  # noqa: E402
from ptp_discovery.free_loss_ir import ir_from_json as free_loss_ir_from_json  # noqa: E402
from ptp_discovery.pref_builder_compiler import compile_preference_builder  # noqa: E402
from ptp_discovery.pref_builder_ir import ir_from_json as pref_builder_ir_from_json  # noqa: E402
from scripts.final_gradient_behavior_analysis import build_problem_specs, rollout_feature_caches  # noqa: E402


PROBLEMS = ["tsp100", "cvrp100", "ffsp100", "jssp10x10"]
LABELS = {"tsp100": "TSP100", "cvrp100": "CVRP100", "ffsp100": "FFSP100", "jssp10x10": "JSSP10x10"}
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
METHODS = ["Loss-only", "Loss+Weighting"]
COLORS = {"Loss-only": "#4C78A8", "Loss+Weighting": "#54A24B"}


def _write_random_jsp(path: Path, *, num_jobs: int, num_machines: int, rng: np.random.Generator) -> None:
    lines = [f"{num_jobs} {num_machines}"]
    for _ in range(num_jobs):
        machines = rng.permutation(num_machines)
        durations = rng.integers(1, 100, size=num_machines)
        tokens: list[str] = []
        for machine, duration in zip(machines, durations):
            tokens.extend([str(int(machine)), str(int(duration))])
        lines.append(" ".join(tokens))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_jssp_data(root: Path) -> None:
    try:
        from scripts.prepare_bopo_jsp_data import prepare_bopo_jsp

        prepare_bopo_jsp(root)
        return
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] BOPO JSP source data unavailable, generating random probe set: {exc}", flush=True)
    rng = np.random.default_rng(12345678)
    for split, count in {"train": 96, "validation": 24}.items():
        split_dir = root / "data" / "jssp_bopo" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            path = split_dir / f"10x10_random_{idx:05d}.jsp"
            if not path.exists():
                _write_random_jsp(path, num_jobs=10, num_machines=10, rng=rng)


def _load_pair(path: str) -> tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]:
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    builder = compile_preference_builder(pref_builder_ir_from_json(payload["g_ir"]))
    loss = compile_free_loss(free_loss_ir_from_json(payload["f_ir"]))

    def build(feature_cache: Mapping[str, torch.Tensor]) -> PrefBatch:
        return builder.build_fn(feature_cache, {"analysis": True})

    def loss_fn(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # Some discovered losses declare a narrow `expects` list but still use
        # `batch.get("weight", ...)` for optional weighting. Passing the full
        # pair batch preserves the actual loss+weighting behavior.
        return loss.loss_fn(dict(batch), {}, {"alpha": 1.0, "analysis": True})

    return build, loss_fn


def _replace_log_prob(base_fc: Mapping[str, torch.Tensor], log_prob: torch.Tensor) -> dict[str, torch.Tensor]:
    objective = base_fc["objective"].detach()
    reward = -objective
    seq_len = base_fc.get("seq_len")
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.full_like(log_prob, float(objective.shape[1]))
    extra: dict[str, torch.Tensor] = {
        "advantage": reward - reward.mean(dim=1, keepdim=True),
        "seq_len": seq_len.to(device=log_prob.device, dtype=log_prob.dtype),
        "log_prob_mean": log_prob / seq_len.to(device=log_prob.device, dtype=log_prob.dtype).clamp_min(1.0),
    }
    for key in ("entropy", "entropy_mean", "log_prob_step"):
        value = base_fc.get(key)
        if isinstance(value, torch.Tensor):
            extra[key] = value
    return extract_feature_cache(objective, log_prob, extra=extra)


def _oracle_aligned_log_prob(fc: Mapping[str, torch.Tensor], sharpness: float) -> torch.Tensor:
    objective = fc["objective"].detach()
    quality = -(objective - objective.mean(dim=1, keepdim=True))
    quality = quality / quality.std(dim=1, keepdim=True).clamp_min(1e-6)
    base = fc["log_prob"].detach()
    residual = base - base.mean(dim=1, keepdim=True)
    residual = residual / residual.std(dim=1, keepdim=True).clamp_min(1e-6)
    return float(sharpness) * quality + 0.05 * residual


def _pair_grad_record(
    fc: Mapping[str, torch.Tensor],
    builder: Callable[[Mapping[str, torch.Tensor]], PrefBatch],
    loss_fn: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
) -> dict[str, np.ndarray | float]:
    pref_raw = builder(fc)
    if pref_raw.num_examples() <= 0 or pref_raw.pair_idx is None:
        return {}
    pref = PrefBatch(
        mode=pref_raw.mode,
        pair_idx=tuple(t.detach() for t in pref_raw.pair_idx),
        list_idx=pref_raw.list_idx.detach() if isinstance(pref_raw.list_idx, torch.Tensor) else pref_raw.list_idx,
        weight=pref_raw.weight.detach() if isinstance(pref_raw.weight, torch.Tensor) else pref_raw.weight,
        meta=dict(pref_raw.meta or {}),
    )
    batch0 = pref.to_pairwise_loss_batch(fc)
    lpw = batch0["log_prob_w"].detach().clone().requires_grad_(True)
    lpl = batch0["log_prob_l"].detach().clone().requires_grad_(True)
    batch = dict(batch0)
    batch["log_prob_w"] = lpw
    batch["log_prob_l"] = lpl
    if isinstance(batch.get("weight"), torch.Tensor):
        batch["weight"] = batch["weight"].detach()
    loss = loss_fn(batch)
    loss.backward()
    gw = lpw.grad.detach()
    gl = lpl.grad.detach()
    b, w, l = pref.pair_idx
    objective = fc["objective"].detach()
    gap = (objective[b, l] - objective[b, w]).clamp_min(0.0)
    inst_gap = (objective.max(dim=1).values - objective.min(dim=1).values).clamp_min(1e-8)
    gap_norm = gap / inst_gap[b]
    margin = (batch0["log_prob_w"] - batch0["log_prob_l"]).detach()
    pair_mass = (gw.abs() + gl.abs()).detach()
    update_by_solution = torch.zeros_like(objective)
    update_by_solution.index_put_((b, w), -gw, accumulate=True)
    update_by_solution.index_put_((b, l), -gl, accumulate=True)
    quality = -(objective - objective.mean(dim=1, keepdim=True))
    quality = quality / quality.std(dim=1, keepdim=True).clamp_min(1e-8)
    align_num = (update_by_solution * quality).sum(dim=1)
    align_den = update_by_solution.abs().sum(dim=1).clamp_min(1e-12)
    return {
        "loss": float(loss.detach().item()),
        "gap_norm": gap_norm.cpu().numpy(),
        "margin": margin.cpu().numpy(),
        "pair_mass": pair_mass.cpu().numpy(),
        "alignment_eff": (align_num / align_den).detach().cpu().numpy(),
        "total_mass": float(pair_mass.sum().item()),
    }


def _concat(records: list[dict[str, np.ndarray | float]], key: str) -> np.ndarray:
    vals = [np.asarray(r[key], dtype=np.float64).reshape(-1) for r in records if key in r]
    return np.concatenate(vals, axis=0) if vals else np.asarray([], dtype=np.float64)


def collect(*, problems: list[str], batches: int, seed: int, device: str, sharpness: list[float]) -> dict[str, Any]:
    if "jssp10x10" in problems:
        _ensure_jssp_data(REPO_ROOT)
    specs = build_problem_specs(device, max(int(batches), 1))
    pairs = {
        problem: {
            "Loss-only": _load_pair(LOSS_ONLY[problem]),
            "Loss+Weighting": _load_pair(WEIGHTING[problem]),
        }
        for problem in problems
    }
    out: dict[str, Any] = {"problems": problems, "records": {}, "sharpness": {}}
    for problem in problems:
        print(f"[problem] {problem}", flush=True)
        caches = rollout_feature_caches(specs[problem], seed=seed, device=torch.device(device))
        out["records"][problem] = {m: [] for m in METHODS}
        out["sharpness"][problem] = {m: [] for m in METHODS}
        for fc in caches:
            for method in METHODS:
                builder, loss_fn = pairs[problem][method]
                rec = _pair_grad_record(fc, builder, loss_fn)
                if rec:
                    out["records"][problem][method].append(rec)
                for sharp in sharpness:
                    fc_sharp = _replace_log_prob(fc, _oracle_aligned_log_prob(fc, sharp))
                    srec = _pair_grad_record(fc_sharp, builder, loss_fn)
                    if not srec:
                        continue
                    gap = np.asarray(srec["gap_norm"], dtype=np.float64)
                    mass = np.asarray(srec["pair_mass"], dtype=np.float64)
                    low = float(np.nansum(mass[gap <= 0.10]))
                    total = float(np.nansum(mass))
                    out["sharpness"][problem][method].append(
                        {
                            "sharpness": float(sharp),
                            "low_gap_mass": low,
                            "total_mass": total,
                            "low_gap_share": low / (total + 1e-12),
                            "alignment_eff": float(np.nanmean(np.asarray(srec["alignment_eff"], dtype=np.float64))),
                        }
                    )
    return out


def _cum_mass_by_gap(records: list[dict[str, np.ndarray | float]], thresholds: np.ndarray) -> np.ndarray:
    gap = _concat(records, "gap_norm")
    mass = _concat(records, "pair_mass")
    total = float(np.nansum(mass))
    if total <= 1e-12:
        return np.zeros_like(thresholds)
    return np.asarray([float(np.nansum(mass[gap <= t]) / total) for t in thresholds], dtype=np.float64)


def _mean_metric(records: list[dict[str, np.ndarray | float]], fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float]) -> float:
    gap = _concat(records, "gap_norm")
    margin = _concat(records, "margin")
    mass = _concat(records, "pair_mass")
    if len(gap) == 0:
        return float("nan")
    return fn(gap, margin, mass)


def write_tables(results: dict[str, Any], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for problem in results["problems"]:
        for method in METHODS:
            records = results["records"][problem][method]
            align = _concat(records, "alignment_eff")
            rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "objective_alignment_eff": float(np.nanmean(align)) if len(align) else float("nan"),
                    "low_gap_mass_share_0p10": _mean_metric(
                        records, lambda gap, _margin, mass: float(np.nansum(mass[gap <= 0.10]) / (np.nansum(mass) + 1e-12))
                    ),
                    "corrective_high_gap_share": _mean_metric(
                        records,
                        lambda gap, margin, mass: float(
                            np.nansum(mass[(gap >= 0.25) & (margin < 0)]) / (np.nansum(mass) + 1e-12)
                        ),
                    ),
                    "already_solved_high_gap_share": _mean_metric(
                        records,
                        lambda gap, margin, mass: float(
                            np.nansum(mass[(gap >= 0.25) & (margin > 0)]) / (np.nansum(mass) + 1e-12)
                        ),
                    ),
                }
            )
    fields = list(rows[0].keys()) if rows else []
    with (out_dir / "cause_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(results: dict[str, Any], out_dir: Path) -> None:
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
    problems = results["problems"]
    thresholds = np.linspace(0.0, 0.35, 80)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 2.75), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        for method in METHODS:
            y = _cum_mass_by_gap(results["records"][problem][method], thresholds)
            ax.plot(thresholds, y, color=COLORS[method], lw=2.0, label=method)
        ax.set_title(LABELS[problem])
        ax.set_xlabel("normalized objective gap threshold")
    axes[0].set_ylabel("gradient mass from near-tie pairs")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Cause 1: low-information pair pollution", fontweight="bold")
    fig.savefig(out_dir / "01_low_information_pair_pollution.png", bbox_inches="tight")
    fig.savefig(out_dir / "01_low_information_pair_pollution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 2.85), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    cats = ["ambiguous", "corrective", "solved"]
    hatches = ["", "//", ".."]
    for ax, problem in zip(axes, problems):
        x = np.arange(len(METHODS))
        bottoms = np.zeros(len(METHODS), dtype=np.float64)
        for ci, cat in enumerate(cats):
            vals = []
            for method in METHODS:
                records = results["records"][problem][method]
                if cat == "ambiguous":
                    val = _mean_metric(records, lambda gap, _margin, mass: float(np.nansum(mass[gap <= 0.10]) / (np.nansum(mass) + 1e-12)))
                elif cat == "corrective":
                    val = _mean_metric(
                        records,
                        lambda gap, margin, mass: float(np.nansum(mass[(gap >= 0.25) & (margin < 0)]) / (np.nansum(mass) + 1e-12)),
                    )
                else:
                    val = _mean_metric(
                        records,
                        lambda gap, margin, mass: float(np.nansum(mass[(gap >= 0.25) & (margin > 0)]) / (np.nansum(mass) + 1e-12)),
                    )
                vals.append(val)
            ax.bar(x, vals, bottom=bottoms, width=0.62, color=["#9ecae1", "#74c476"], edgecolor="#333333", linewidth=0.4, hatch=hatches[ci], label=cat)
            bottoms += np.asarray(vals)
        ax.set_title(LABELS[problem])
        ax.set_xticks(x, METHODS, rotation=20, ha="right")
    axes[0].set_ylabel("share of gradient mass")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Cause 2: noisy vs corrective gradient pressure", fontweight="bold")
    fig.savefig(out_dir / "02_noisy_vs_corrective_pressure.png", bbox_inches="tight")
    fig.savefig(out_dir / "02_noisy_vs_corrective_pressure.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 3.05), constrained_layout=True)
    x = np.arange(len(problems))
    width = 0.34
    for mi, method in enumerate(METHODS):
        vals = []
        for problem in problems:
            align = _concat(results["records"][problem][method], "alignment_eff")
            vals.append(float(np.nanmean(align)) if len(align) else float("nan"))
        ax.bar(x + (mi - 0.5) * width, vals, width=width, color=COLORS[method], label=method)
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, [LABELS[p] for p in problems])
    ax.set_ylabel("objective-aligned update efficiency")
    ax.set_title("Cause 3: gradient direction quality after pair aggregation")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "03_objective_alignment_efficiency.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_objective_alignment_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 2.85), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        for method in METHODS:
            rows = results["sharpness"][problem][method]
            xs = sorted({float(r["sharpness"]) for r in rows})
            ys = []
            for xval in xs:
                same = [r for r in rows if abs(float(r["sharpness"]) - xval) < 1e-12]
                ys.append(float(np.mean([r["low_gap_mass"] for r in same])))
            arr = np.asarray(ys, dtype=np.float64)
            arr = arr / max(float(arr[0]), 1e-12)
            ax.plot(xs, arr, marker="o", ms=3.0, lw=1.8, color=COLORS[method], label=method)
        ax.set_xscale("log")
        ax.set_title(LABELS[problem])
        ax.set_xlabel("oracle policy sharpness")
    axes[0].set_ylabel("relative near-tie gradient pressure")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Cause 4: late-stage near-tie pressure under aligned policies", fontweight="bold")
    fig.savefig(out_dir / "04_late_stage_near_tie_pressure.png", bbox_inches="tight")
    fig.savefig(out_dir / "04_late_stage_near_tie_pressure.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Four-Problem Loss/Weighting Cause Analysis\n\n"
            "This compares final loss-only vs final loss+weighting on TSP100, CVRP100, FFSP100, and JSSP10x10.\n"
            "The plots avoid topology-only facts and instead measure non-obvious optimization pressure: near-tie pollution, corrective pressure, aggregate objective alignment, and late-stage near-tie pressure.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sharpness", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    sharpness = [float(x) for x in str(args.sharpness).split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "four_problem_loss_weighting_causes" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect(problems=problems, batches=max(int(args.batches), 1), seed=int(args.seed), device=device, sharpness=sharpness)
    plot(results, out_dir)
    write_tables(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
