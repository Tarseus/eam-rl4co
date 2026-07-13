from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
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
PROBLEM_LABELS = {
    "tsp100": "TSP100",
    "cvrp100": "CVRP100",
    "ffsp100": "FFSP100",
    "jssp10x10": "JSSP10x10",
}
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


def _ensure_jssp_shape_data(root: Path, *, num_jobs: int, num_machines: int, count_train: int = 96, count_val: int = 24) -> tuple[str, str]:
    rng = np.random.default_rng(12345678 + 1009 * int(num_jobs) + int(num_machines))
    base = root / "data" / "jssp_scale_probe" / f"{num_jobs}x{num_machines}"
    for split, count in {"train": count_train, "validation": count_val}.items():
        split_dir = base / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            path = split_dir / f"{num_jobs}x{num_machines}_random_{idx:05d}.jsp"
            if not path.exists():
                _write_random_jsp(path, num_jobs=num_jobs, num_machines=num_machines, rng=rng)
    train_rel = str((base / "train").relative_to(root)).replace("\\", "/")
    val_rel = str((base / "validation").relative_to(root)).replace("\\", "/")
    return train_rel, val_rel


def _load_pair(path: str) -> tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]:
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    builder = compile_preference_builder(pref_builder_ir_from_json(payload["g_ir"]))
    loss = compile_free_loss(free_loss_ir_from_json(payload["f_ir"]))

    def build(feature_cache: Mapping[str, torch.Tensor]) -> PrefBatch:
        return builder.build_fn(feature_cache, {"analysis": True})

    def loss_fn(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # Some discovered losses declare a narrow expects list but still read
        # optional `weight`; pass the full batch to preserve actual behavior.
        return loss.loss_fn(dict(batch), {}, {"alpha": 1.0, "analysis": True})

    return build, loss_fn, str(payload.get("f_ir", {}).get("name", ""))


def _replace_objective(base_fc: Mapping[str, torch.Tensor], objective: torch.Tensor, log_prob: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    lp = base_fc["log_prob"].detach() if log_prob is None else log_prob
    reward = -objective
    seq_len = base_fc.get("seq_len")
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.full_like(lp, float(objective.shape[1]))
    extra: dict[str, torch.Tensor] = {
        "advantage": reward - reward.mean(dim=1, keepdim=True),
        "seq_len": seq_len.to(device=lp.device, dtype=lp.dtype),
        "log_prob_mean": lp / seq_len.to(device=lp.device, dtype=lp.dtype).clamp_min(1.0),
    }
    for key in ("entropy", "entropy_mean", "log_prob_step"):
        value = base_fc.get(key)
        if isinstance(value, torch.Tensor):
            extra[key] = value
    return extract_feature_cache(objective, lp, extra=extra)


def _scale_objective(base_fc: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    objective = base_fc["objective"].detach()
    centered = objective - objective.mean(dim=1, keepdim=True)
    scaled = objective.mean(dim=1, keepdim=True) + float(scale) * centered
    return _replace_objective(base_fc, scaled, base_fc["log_prob"].detach())


def _po_loss(reward: torch.Tensor, log_prob: torch.Tensor, *, problem: str) -> torch.Tensor:
    alpha = 1.0 if problem in {"ffsp100", "jssp10x10"} else 0.05
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp_pair = alpha * (log_prob[:, :, None] - log_prob[:, None, :])
    pf_log = logp_pair if problem == "ffsp100" else F.logsigmoid(logp_pair)
    return -(pf_log * preference).mean()


def _bopo_loss(reward: torch.Tensor, log_prob: torch.Tensor, seq_len: torch.Tensor | None, select_k: int = 4) -> torch.Tensor:
    objective = -reward
    score = log_prob if seq_len is None else log_prob / seq_len.clamp_min(1.0)
    batch_size, k_total = reward.shape
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


def _sll_loss(reward: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    sorted_idx = reward.sort(dim=1, descending=True).indices
    logp_rank = log_prob.gather(1, sorted_idx)
    suffix = torch.logcumsumexp(logp_rank.flip(dims=[1]), dim=1).flip(dims=[1])
    return -(logp_rank - suffix).mean()


def _method_loss(
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    log_prob: torch.Tensor,
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]],
) -> torch.Tensor:
    objective = fc["objective"].detach()
    reward = -objective
    if method == "PO":
        return _po_loss(reward, log_prob, problem=problem)
    if method == "BOPO":
        seq_len = fc.get("seq_len")
        return _bopo_loss(reward, log_prob, seq_len=seq_len if isinstance(seq_len, torch.Tensor) else None)
    if method == "SLL":
        return _sll_loss(reward, log_prob)
    builder, loss_fn, _ = pairs[method]
    live_fc = _replace_objective(fc, objective, log_prob)
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


def _state_log_prob(fc: Mapping[str, torch.Tensor], state: str, sharpness: float) -> torch.Tensor:
    objective = fc["objective"].detach()
    quality = -(objective - objective.mean(dim=1, keepdim=True))
    quality = quality / quality.std(dim=1, keepdim=True).clamp_min(1e-6)
    raw = fc["log_prob"].detach()
    residual = raw - raw.mean(dim=1, keepdim=True)
    residual = residual / residual.std(dim=1, keepdim=True).clamp_min(1e-6)
    if state == "sampled":
        return raw
    if state == "aligned":
        return float(sharpness) * quality + 0.05 * residual
    if state == "misaligned":
        return -float(sharpness) * quality + 0.05 * residual
    raise KeyError(state)


def _policy_metrics(objective: torch.Tensor, log_prob: torch.Tensor) -> dict[str, torch.Tensor]:
    regret = objective - objective.min(dim=1, keepdim=True).values
    regret_norm = regret / regret.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    probs = torch.softmax(log_prob - log_prob.max(dim=1, keepdim=True).values, dim=1)
    return {
        "expected_regret": (probs * regret_norm).sum(dim=1),
        "good_mass": (probs * (regret_norm <= 0.10).float()).sum(dim=1),
        "bad_mass": (probs * (regret_norm >= 0.70).float()).sum(dim=1),
    }


def _weight_stats_for_pref(pref: PrefBatch, objective: torch.Tensor, method: str) -> dict[str, float]:
    if pref.num_examples() <= 0:
        return {"weight_ess": float("nan"), "top10_mass": float("nan"), "hi_clip_share": float("nan"), "lo_clip_share": float("nan")}
    weight = pref.weight
    if not isinstance(weight, torch.Tensor):
        if pref.pair_idx is None:
            n = objective.numel()
        else:
            n = int(pref.pair_idx[0].numel())
        w = torch.ones((n,), device=objective.device, dtype=objective.dtype)
    else:
        w = weight.detach().float().reshape(-1).clamp_min(0.0)
    n = int(w.numel())
    if n <= 0 or float(w.sum().item()) <= 1e-12:
        return {"weight_ess": float("nan"), "top10_mass": float("nan"), "hi_clip_share": float("nan"), "lo_clip_share": float("nan")}
    ess = (w.sum() ** 2 / (w.square().sum().clamp_min(1e-12) * n)).item()
    k = max(int(np.ceil(0.10 * n)), 1)
    top_mass = float(w.topk(k).values.sum().item() / w.sum().item())
    lo = float(w.min().item())
    hi = float(w.max().item())
    tol = 1e-6
    lo_share = float((w <= lo + tol).float().mean().item())
    hi_share = float((w >= hi - tol).float().mean().item())
    # Uniform methods have all weights equal; do not call that clipping.
    if method in {"PO", "SLL", "Loss-only"} or abs(hi - lo) <= 1e-8:
        lo_share = 0.0
        hi_share = 0.0
    return {"weight_ess": float(ess), "top10_mass": top_mass, "hi_clip_share": hi_share, "lo_clip_share": lo_share}


def _implicit_pref_for_method(
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]],
) -> PrefBatch:
    objective = fc["objective"].detach()
    if method in {"PO", "Loss-only", "Loss+Weighting"}:
        if method in {"Loss-only", "Loss+Weighting"}:
            return pairs[method][0](fc)
        mask = objective[:, :, None] < objective[:, None, :]
        b, w, l = mask.nonzero(as_tuple=True)
        return PrefBatch(mode="pairwise", pair_idx=(b, w, l), weight=None, meta={"method": method})
    if method == "BOPO":
        reward = -objective
        batch_size, k_total = reward.shape
        select_k = 4
        stride = max(k_total // select_k, 1)
        b_all: list[torch.Tensor] = []
        w_all: list[torch.Tensor] = []
        l_all: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        eps = torch.as_tensor(1e-8, device=objective.device, dtype=objective.dtype)
        for i in range(batch_size):
            selected = reward[i].sort(descending=True).indices[::stride][:select_k]
            b_all.append(torch.full((select_k - 1,), i, device=objective.device, dtype=torch.long))
            w_all.append(selected[:1].expand(select_k - 1))
            l_all.append(selected[1:])
            weights.append((objective[i, selected[1:]] + eps) / (objective[i, selected[:1]] + eps))
        return PrefBatch(mode="pairwise", pair_idx=(torch.cat(b_all), torch.cat(w_all), torch.cat(l_all)), weight=torch.cat(weights), meta={"method": method})
    if method == "SLL":
        # Treat SLL ranking as an implicit uniform pair topology for concentration diagnostics.
        mask = objective[:, :, None] < objective[:, None, :]
        b, w, l = mask.nonzero(as_tuple=True)
        return PrefBatch(mode="pairwise", pair_idx=(b, w, l), weight=None, meta={"method": method})
    raise KeyError(method)


def _one_step_record(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], str]],
    scale: float,
    state: str,
    sharpness: float,
    step_size: float,
) -> dict[str, float | str]:
    scaled_fc = dict(fc)
    objective = scaled_fc["objective"].detach()
    lp0 = _state_log_prob(scaled_fc, state, sharpness).detach().clone().requires_grad_(True)
    loss = _method_loss(method, problem, scaled_fc, lp0, pairs)
    loss.backward()
    update = -lp0.grad.detach()
    update = update - update.mean(dim=1, keepdim=True)
    update = update / update.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    lp1 = lp0.detach() + float(step_size) * update
    before = _policy_metrics(objective, lp0.detach())
    after = _policy_metrics(objective, lp1.detach())
    regret_reduction = before["expected_regret"] - after["expected_regret"]
    good_gain = after["good_mass"] - before["good_mass"]
    bad_reduction = before["bad_mass"] - after["bad_mass"]
    pref = _implicit_pref_for_method(method, problem, scaled_fc, pairs)
    weight_stats = _weight_stats_for_pref(pref, objective, method)
    return {
        "problem": problem,
        "method": method,
        "scale": float(scale),
        "state": state,
        "sharpness": float(sharpness),
        "loss": float(loss.detach().item()),
        "regret_reduction": float(regret_reduction.mean().item()),
        "good_mass_gain": float(good_gain.mean().item()),
        "bad_mass_reduction": float(bad_reduction.mean().item()),
        "negative_instance_rate": float((regret_reduction < 0).float().mean().item()),
        "clip_share": float(weight_stats["hi_clip_share"] + weight_stats["lo_clip_share"]),
        **weight_stats,
    }


def _target_spec(source_spec: Any, problem: str, scale: float) -> Any:
    if problem == "tsp100":
        size = max(int(round(100 * float(scale))), 5)
        hf = replace(
            source_spec.hf,
            generator_params={"num_loc": size},
            train_problem_size=size,
            valid_problem_sizes=(size,),
            pomo_size=None,
        )
        return replace(source_spec, hf=hf, batch_size=max(2, min(int(source_spec.batch_size), 8)))
    if problem == "cvrp100":
        size = max(int(round(100 * float(scale))), 5)
        hf = replace(
            source_spec.hf,
            generator_params={"num_loc": size},
            train_problem_size=size,
            valid_problem_sizes=(size,),
            pomo_size=None,
        )
        return replace(source_spec, hf=hf, batch_size=max(2, min(int(source_spec.batch_size), 8)))
    if problem == "ffsp100":
        size = max(int(round(100 * float(scale))), 5)
        params = dict(source_spec.hf.generator_params)
        params["num_job"] = size
        hf = replace(
            source_spec.hf,
            generator_params=params,
            train_problem_size=size,
            valid_problem_sizes=(size,),
        )
        return replace(source_spec, hf=hf, batch_size=max(2, min(int(source_spec.batch_size), 4)))
    if problem == "jssp10x10":
        size = max(int(round(10 * float(scale))), 5)
        train_dir, val_dir = _ensure_jssp_shape_data(REPO_ROOT, num_jobs=size, num_machines=size)
        policy_kwargs = dict(source_spec.hf.policy_kwargs)
        policy_kwargs.update(
            {
                "train_data_dir": train_dir,
                "val_data_dir": val_dir,
                "allowed_shapes": [[size, size]],
            }
        )
        hf = replace(
            source_spec.hf,
            generator_params={"num_jobs": size, "num_machines": size, "min_processing_time": 1, "max_processing_time": 99},
            policy_kwargs=policy_kwargs,
            train_problem_size=size,
            valid_problem_sizes=(size,),
            pomo_size=128,
        )
        return replace(source_spec, hf=hf, batch_size=1)
    raise KeyError(problem)


def collect(*, problems: list[str], batches: int, seed: int, device: str, scales: list[float], step_size: float, state: str, sharpness: float) -> dict[str, Any]:
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
    rows: list[dict[str, Any]] = []
    for problem in problems:
        print(f"[problem] {problem}", flush=True)
        for scale in scales:
            target_spec = _target_spec(specs[problem], problem, scale)
            target_size = int(target_spec.hf.train_problem_size)
            source_size = 10 if problem == "jssp10x10" else 100
            print(f"[scale] {problem} target_size={target_size} ratio={scale}", flush=True)
            caches = rollout_feature_caches(target_spec, seed=seed + int(round(float(scale) * 1000)), device=torch.device(device))
            for fc in caches:
                for method in METHODS:
                    rec = _one_step_record(
                        method=method,
                        problem=problem,
                        fc=fc,
                        pairs=pair_cache[problem],
                        scale=scale,
                        state=state,
                        sharpness=sharpness,
                        step_size=step_size,
                    )
                    rec["target_size"] = target_size
                    rec["source_size"] = source_size
                    rows.append(rec)
    return {"problems": problems, "scales": scales, "rows": rows, "state": state, "sharpness": sharpness}


def _mean(rows: list[dict[str, Any]], problem: str, method: str, scale: float, metric: str) -> float:
    vals = [
        float(r[metric])
        for r in rows
        if r["problem"] == problem and r["method"] == method and abs(float(r["scale"]) - float(scale)) < 1e-12
    ]
    return float(np.nanmean(vals)) if vals else float("nan")


def _write_table(results: dict[str, Any], out_dir: Path) -> None:
    rows = results["rows"]
    fields = list(rows[0]) if rows else []
    with (out_dir / "scale_generalization_cause_metrics.csv").open("w", encoding="utf-8", newline="") as f:
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
            "legend.fontsize": 7.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    rows = results["rows"]
    problems = results["problems"]
    scales = [float(s) for s in results["scales"]]

    def line_grid(metric: str, ylabel: str, title: str, fname: str, *, hline: float | None = None) -> None:
        fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 2.95), sharey=True, constrained_layout=True)
        if len(problems) == 1:
            axes = [axes]
        for ax, problem in zip(axes, problems):
            for method in METHODS:
                ys = [_mean(rows, problem, method, s, metric) for s in scales]
                ax.plot(scales, ys, marker="o", ms=2.8, lw=1.45, color=COLORS[method], label=method)
            ax.axvline(1.0, color="#333333", lw=0.8, ls="--", alpha=0.55)
            if hline is not None:
                ax.axhline(hline, color="#333333", lw=0.8)
            ax.set_title(PROBLEM_LABELS[problem])
            ax.set_xlabel("target/source problem size")
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False, loc="best")
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    line_grid(
        "regret_reduction",
        "expected regret reduction",
        "Scale-shift stress: loss-only remains useful, weighting is source-scale sensitive",
        "01_scale_shift_regret_reduction",
        hline=0.0,
    )
    line_grid(
        "negative_instance_rate",
        "fraction of instances",
        "Failure under scale shift: one step makes the policy distribution worse",
        "02_scale_shift_failure_rate",
        hline=0.0,
    )
    line_grid(
        "weight_ess",
        "weight ESS / pairs",
        "Why weighting does not generalize: effective pair sample size changes with scale",
        "03_weight_ess_collapse",
        hline=None,
    )
    line_grid(
        "top10_mass",
        "mass on top 10% weighted pairs",
        "Weight concentration: scale shift moves optimization budget into fewer pairs",
        "04_top10_weight_mass",
        hline=0.10,
    )
    line_grid(
        "clip_share",
        "fraction of pair weights at a clamp boundary",
        "Weighting fragility: learned weights often saturate instead of adapting smoothly",
        "05_weight_clip_saturation",
        hline=0.0,
    )

    fig, axes = plt.subplots(1, len(problems), figsize=(3.15 * len(problems), 2.95), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    x = np.arange(len(METHODS))
    for ax, problem in zip(axes, problems):
        source = np.asarray([_mean(rows, problem, m, 1.0, "regret_reduction") for m in METHODS], dtype=np.float64)
        worst_shift: list[float] = []
        for method in METHODS:
            vals = [_mean(rows, problem, method, s, "regret_reduction") for s in scales if abs(s - 1.0) > 1e-12]
            worst_shift.append(float(np.nanmin(vals)) if vals else float("nan"))
        worst = np.asarray(worst_shift, dtype=np.float64)
        width = 0.36
        ax.bar(x - width / 2, source, width=width, color=[COLORS[m] for m in METHODS], alpha=0.95, label="source scale")
        ax.bar(x + width / 2, worst, width=width, color=[COLORS[m] for m in METHODS], alpha=0.45, hatch="//", label="worst shifted scale")
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xticks(x, METHODS, rotation=30, ha="right")
    axes[0].set_ylabel("expected regret reduction")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Same-scale behavior versus scale-shift behavior", fontweight="bold")
    fig.savefig(out_dir / "06_same_scale_vs_shift_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "06_same_scale_vs_shift_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Scale-Generalization Cause Analysis\n\n"
            "All figures compare PO, BOPO, SLL, Loss-only, and Loss+Weighting on the same candidate pools.\n"
            "The x-axis changes the rollout problem size while preserving the source loss/weighting found at 100-node or 10x10 source scale.\n"
            "A good loss should still reduce normalized expected regret after one equalized update. Loss-only has no learned pair weighting layer;\n"
            "Loss+Weighting uses the discovered weighting builder and therefore can concentrate or saturate its optimization budget when scale statistics move.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scales", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--state", default="aligned", choices=["sampled", "aligned", "misaligned"])
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    scales = [float(x) for x in str(args.scales).split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "scale_generalization_loss_weighting" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect(
        problems=problems,
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        scales=scales,
        step_size=float(args.step_size),
        state=str(args.state),
        sharpness=float(args.sharpness),
    )
    plot(results, out_dir)
    _write_table(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
