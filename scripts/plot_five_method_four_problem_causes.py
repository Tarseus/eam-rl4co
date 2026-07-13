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
import torch.nn.functional as F


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
PROBLEM_LABELS = {"tsp100": "TSP100", "cvrp100": "CVRP100", "ffsp100": "FFSP100", "jssp10x10": "JSSP10x10"}
METHODS = ["PO", "BOPO", "SLL", "Loss-only", "Loss+Weighting"]
COLORS = {
    "PO": "#4C78A8",
    "BOPO": "#B279A2",
    "SLL": "#F58518",
    "Loss-only": "#72B7B2",
    "Loss+Weighting": "#54A24B",
}
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


def _write_random_jsp(path: Path, *, rng: np.random.Generator) -> None:
    lines = ["10 10"]
    for _ in range(10):
        machines = rng.permutation(10)
        durations = rng.integers(1, 100, size=10)
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
                _write_random_jsp(path, rng=rng)


def _load_pair(path: str) -> tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]:
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    builder = compile_preference_builder(pref_builder_ir_from_json(payload["g_ir"]))
    loss = compile_free_loss(free_loss_ir_from_json(payload["f_ir"]))

    def build(feature_cache: Mapping[str, torch.Tensor]) -> PrefBatch:
        return builder.build_fn(feature_cache, {"analysis": True})

    def loss_fn(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
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


def _po_loss(reward: torch.Tensor, log_prob: torch.Tensor, *, alpha: float, impl: str) -> torch.Tensor:
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp_pair = alpha * (log_prob[:, :, None] - log_prob[:, None, :])
    if impl == "exponential":
        pf_log = logp_pair
    else:
        pf_log = F.logsigmoid(logp_pair)
    return -(pf_log * preference).mean()


def _bopo_loss(reward: torch.Tensor, log_prob: torch.Tensor, *, seq_len: torch.Tensor | None, select_k: int = 4) -> torch.Tensor:
    objective = -reward
    score = log_prob if seq_len is None else log_prob / seq_len.clamp_min(1.0)
    batch_size, k_total = reward.shape
    if k_total % select_k != 0:
        select_k = max(2, min(select_k, k_total))
    stride = max(k_total // select_k, 1)
    losses: list[torch.Tensor] = []
    eps = 1e-8
    for i in range(batch_size):
        selected = reward[i].sort(descending=True).indices[::stride][:select_k]
        obj = objective[i, selected]
        logp = score[i, selected]
        factor = (obj[1:] + eps) / (obj[:1] + eps)
        losses.append(-F.logsigmoid(factor * (logp[:1] - logp[1:])))
    return torch.cat(losses).mean()


def _sll_loss(reward: torch.Tensor, log_prob: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    sorted_idx = reward.sort(dim=1, descending=True).indices
    logp_rank = (log_prob / float(temperature)).gather(1, sorted_idx)
    suffix = torch.logcumsumexp(logp_rank.flip(dims=[1]), dim=1).flip(dims=[1])
    return -(logp_rank - suffix).mean()


def _method_loss(
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    log_prob: torch.Tensor,
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]],
) -> torch.Tensor:
    objective = fc["objective"].detach()
    reward = -objective
    if method == "PO":
        return _po_loss(reward, log_prob, alpha=1.0 if problem in {"ffsp100", "jssp10x10"} else 0.05, impl="exponential" if problem == "ffsp100" else "bt")
    if method == "BOPO":
        seq_len = fc.get("seq_len")
        return _bopo_loss(reward, log_prob, seq_len=seq_len if isinstance(seq_len, torch.Tensor) else None)
    if method == "SLL":
        return _sll_loss(reward, log_prob)
    builder, loss_fn = pairs[method]
    live_fc = _replace_log_prob(fc, log_prob)
    pref_raw = builder(live_fc)
    if pref_raw.num_examples() <= 0:
        return log_prob.sum() * 0.0
    pref = PrefBatch(
        mode=pref_raw.mode,
        pair_idx=pref_raw.pair_idx,
        list_idx=pref_raw.list_idx,
        weight=pref_raw.weight.detach() if isinstance(pref_raw.weight, torch.Tensor) else pref_raw.weight,
        meta=dict(pref_raw.meta or {}),
    )
    return loss_fn(pref.to_pairwise_loss_batch(live_fc))


def _oracle_log_prob(fc: Mapping[str, torch.Tensor], sharpness: float) -> torch.Tensor:
    objective = fc["objective"].detach()
    quality = -(objective - objective.mean(dim=1, keepdim=True))
    quality = quality / quality.std(dim=1, keepdim=True).clamp_min(1e-6)
    residual = fc["log_prob"].detach() - fc["log_prob"].detach().mean(dim=1, keepdim=True)
    residual = residual / residual.std(dim=1, keepdim=True).clamp_min(1e-6)
    return float(sharpness) * quality + 0.05 * residual


def _grad_record(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]],
    log_prob_override: torch.Tensor | None = None,
) -> dict[str, np.ndarray | float]:
    lp0 = fc["log_prob"].detach() if log_prob_override is None else log_prob_override.detach()
    lp = lp0.clone().requires_grad_(True)
    loss = _method_loss(method, problem, fc, lp, pairs)
    loss.backward()
    grad = lp.grad.detach()
    update = -grad
    objective = fc["objective"].detach()
    quality = -(objective - objective.mean(dim=1, keepdim=True))
    quality_z = quality / quality.std(dim=1, keepdim=True).clamp_min(1e-8)
    regret = objective - objective.min(dim=1, keepdim=True).values
    regret_norm = regret / regret.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    mass = grad.abs()
    alignment = (update * quality_z).sum(dim=1) / mass.sum(dim=1).clamp_min(1e-12)
    bad = regret_norm >= 0.70
    good = regret_norm <= 0.10
    harmful_bad_up = torch.clamp(update[bad], min=0.0).sum() / update.abs().sum().clamp_min(1e-12)
    helpful_good_up = torch.clamp(update[good], min=0.0).sum() / update.abs().sum().clamp_min(1e-12)
    near_best_mass = mass[good].sum() / mass.sum().clamp_min(1e-12)
    high_regret_mass = mass[bad].sum() / mass.sum().clamp_min(1e-12)
    return {
        "loss": float(loss.detach().item()),
        "alignment_eff": alignment.cpu().numpy(),
        "harmful_bad_up": float(harmful_bad_up.item()),
        "helpful_good_up": float(helpful_good_up.item()),
        "near_best_mass": float(near_best_mass.item()),
        "high_regret_mass": float(high_regret_mass.item()),
        "total_mass": float(mass.sum().item()),
        "regret_norm": regret_norm.cpu().numpy().reshape(-1),
        "signed_update": update.cpu().numpy().reshape(-1),
        "mass": mass.cpu().numpy().reshape(-1),
    }


def collect(*, problems: list[str], batches: int, seed: int, device: str, sharpness: list[float]) -> dict[str, Any]:
    if "jssp10x10" in problems:
        _ensure_jssp_data(REPO_ROOT)
    specs = build_problem_specs(device, batches)
    pair_cache = {
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
                out["records"][problem][method].append(
                    _grad_record(method=method, problem=problem, fc=fc, pairs=pair_cache[problem])
                )
                for sharp in sharpness:
                    rec = _grad_record(
                        method=method,
                        problem=problem,
                        fc=fc,
                        pairs=pair_cache[problem],
                        log_prob_override=_oracle_log_prob(fc, sharp),
                    )
                    out["sharpness"][problem][method].append(
                        {
                            "sharpness": float(sharp),
                            "alignment_eff": float(np.nanmean(np.asarray(rec["alignment_eff"], dtype=np.float64))),
                            "harmful_bad_up": float(rec["harmful_bad_up"]),
                            "total_mass": float(rec["total_mass"]),
                        }
                    )
    return out


def _mean(records: list[dict[str, Any]], key: str) -> float:
    vals = []
    for r in records:
        value = r[key]
        vals.append(float(np.nanmean(np.asarray(value, dtype=np.float64))))
    return float(np.nanmean(vals)) if vals else float("nan")


def _concat(records: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = [np.asarray(r[key], dtype=np.float64).reshape(-1) for r in records]
    return np.concatenate(vals) if vals else np.asarray([], dtype=np.float64)


def write_tables(results: dict[str, Any], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for problem in results["problems"]:
        for method in METHODS:
            recs = results["records"][problem][method]
            rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "alignment_eff": _mean(recs, "alignment_eff"),
                    "harmful_bad_up": _mean(recs, "harmful_bad_up"),
                    "helpful_good_up": _mean(recs, "helpful_good_up"),
                    "near_best_mass": _mean(recs, "near_best_mass"),
                    "high_regret_mass": _mean(recs, "high_regret_mass"),
                    "total_mass": _mean(recs, "total_mass"),
                    "loss": _mean(recs, "loss"),
                }
            )
    fields = list(rows[0]) if rows else []
    with (out_dir / "five_method_cause_metrics.csv").open("w", newline="", encoding="utf-8") as f:
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
            "font.size": 8.6,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.6,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    problems = results["problems"]
    x = np.arange(len(METHODS))

    def bar_grid(metric: str, ylabel: str, title: str, filename: str) -> None:
        fig, axes = plt.subplots(1, len(problems), figsize=(3.1 * len(problems), 2.9), sharey=True, constrained_layout=True)
        if len(problems) == 1:
            axes = [axes]
        for ax, problem in zip(axes, problems):
            vals = [_mean(results["records"][problem][m], metric) for m in METHODS]
            ax.bar(x, vals, color=[COLORS[m] for m in METHODS], width=0.72)
            ax.set_title(PROBLEM_LABELS[problem])
            ax.set_xticks(x, METHODS, rotation=30, ha="right")
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{filename}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    bar_grid("harmful_bad_up", "share of signed update", "Cause 1: bad-solution promotion", "01_bad_solution_promotion")
    bar_grid("alignment_eff", "objective-aligned update efficiency", "Cause 2: aggregate update alignment", "02_objective_alignment")
    bar_grid("near_best_mass", "share of |gradient|", "Cause 3: pressure on near-best / near-tie candidates", "03_near_best_pressure")

    fig, axes = plt.subplots(1, len(problems), figsize=(3.1 * len(problems), 2.9), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        for method in METHODS:
            rows = results["sharpness"][problem][method]
            xs = sorted({float(r["sharpness"]) for r in rows})
            ys = []
            for sx in xs:
                same = [r for r in rows if abs(float(r["sharpness"]) - sx) < 1e-12]
                ys.append(float(np.mean([r["alignment_eff"] for r in same])))
            ax.plot(xs, ys, marker="o", ms=2.8, lw=1.55, color=COLORS[method], label=method)
        ax.set_xscale("log")
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xlabel("oracle policy sharpness")
    axes[0].set_ylabel("alignment efficiency")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Cause 4: late-stage behavior as policy becomes sharper", fontweight="bold")
    fig.savefig(out_dir / "04_late_stage_alignment.png", bbox_inches="tight")
    fig.savefig(out_dir / "04_late_stage_alignment.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.1 * len(problems), 2.9), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    bins = np.linspace(0.0, 1.0, 13)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for ax, problem in zip(axes, problems):
        for method in METHODS:
            regret = _concat(results["records"][problem][method], "regret_norm")
            update = _concat(results["records"][problem][method], "signed_update")
            scale = max(float(np.nanmax(np.abs(update))) if len(update) else 0.0, 1e-12)
            ys = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (regret >= lo) & (regret < hi if hi < 1.0 else regret <= hi)
                ys.append(float(np.nanmean(update[mask] / scale)) if np.any(mask) else np.nan)
            ax.plot(centers, ys, lw=1.45, marker="o", ms=2.5, color=COLORS[method], label=method)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xlabel("normalized regret")
    axes[0].set_ylabel("normalized signed update")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Cause 5: update shape over solution quality", fontweight="bold")
    fig.savefig(out_dir / "05_update_shape_by_regret.png", bbox_inches="tight")
    fig.savefig(out_dir / "05_update_shape_by_regret.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Five-Method Four-Problem Cause Analysis\n\n"
            "Every figure compares PO, BOPO, SLL, Loss-only, and Loss+Weighting on TSP100, CVRP100, FFSP100, and JSSP10x10.\n"
            "All methods are reduced to their induced gradient/update on rollout log-probabilities, so pairwise, listwise, and discovered weighted losses share the same measurement space.\n"
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
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "five_method_four_problem_causes" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect(problems=problems, batches=max(int(args.batches), 1), seed=int(args.seed), device=device, sharpness=sharpness)
    plot(results, out_dir)
    write_tables(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
