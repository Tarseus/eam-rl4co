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
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]],
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
    top_mask = regret_norm <= 0.10
    bad_mask = regret_norm >= 0.70
    return {
        "expected_regret": (probs * regret_norm).sum(dim=1),
        "top_mass": (probs * top_mask.float()).sum(dim=1),
        "bad_mass": (probs * bad_mask.float()).sum(dim=1),
    }


def _one_step_record(
    *,
    method: str,
    problem: str,
    fc: Mapping[str, torch.Tensor],
    pairs: Mapping[str, tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]],
    state: str,
    sharpness: float,
    step_size: float,
) -> dict[str, float]:
    objective = fc["objective"].detach()
    lp0 = _state_log_prob(fc, state, sharpness).detach().clone().requires_grad_(True)
    loss = _method_loss(method, problem, fc, lp0, pairs)
    loss.backward()
    update = -lp0.grad.detach()
    update = update - update.mean(dim=1, keepdim=True)
    update = update / update.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    lp1 = lp0.detach() + float(step_size) * update
    before = _policy_metrics(objective, lp0.detach())
    after = _policy_metrics(objective, lp1.detach())
    regret_reduction = before["expected_regret"] - after["expected_regret"]
    top_gain = after["top_mass"] - before["top_mass"]
    bad_reduction = before["bad_mass"] - after["bad_mass"]
    return {
        "loss": float(loss.detach().item()),
        "regret_reduction": float(regret_reduction.mean().item()),
        "top_mass_gain": float(top_gain.mean().item()),
        "bad_mass_reduction": float(bad_reduction.mean().item()),
        "negative_instance_rate": float((regret_reduction < 0).float().mean().item()),
        "state": state,
        "sharpness": float(sharpness),
    }


def collect(*, problems: list[str], batches: int, seed: int, device: str, sharpness: list[float], step_size: float) -> dict[str, Any]:
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
    out: dict[str, Any] = {"problems": problems, "records": {p: {m: [] for m in METHODS} for p in problems}}
    for problem in problems:
        print(f"[problem] {problem}", flush=True)
        caches = rollout_feature_caches(specs[problem], seed=seed, device=torch.device(device))
        for fc in caches:
            for method in METHODS:
                out["records"][problem][method].append(
                    _one_step_record(
                        method=method,
                        problem=problem,
                        fc=fc,
                        pairs=pair_cache[problem],
                        state="sampled",
                        sharpness=1.0,
                        step_size=step_size,
                    )
                )
                for s in sharpness:
                    out["records"][problem][method].append(
                        _one_step_record(
                            method=method,
                            problem=problem,
                            fc=fc,
                            pairs=pair_cache[problem],
                            state="aligned",
                            sharpness=s,
                            step_size=step_size,
                        )
                    )
                    out["records"][problem][method].append(
                        _one_step_record(
                            method=method,
                            problem=problem,
                            fc=fc,
                            pairs=pair_cache[problem],
                            state="misaligned",
                            sharpness=s,
                            step_size=step_size,
                        )
                    )
    return out


def _mean(records: list[dict[str, Any]], metric: str, *, state: str, sharpness: float | None = None) -> float:
    vals = []
    for r in records:
        if r["state"] != state:
            continue
        if sharpness is not None and abs(float(r["sharpness"]) - float(sharpness)) > 1e-12:
            continue
        vals.append(float(r[metric]))
    return float(np.nanmean(vals)) if vals else float("nan")


def write_tables(results: dict[str, Any], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for problem in results["problems"]:
        for method in METHODS:
            recs = results["records"][problem][method]
            for state in sorted({str(r["state"]) for r in recs}):
                for sharpness in sorted({float(r["sharpness"]) for r in recs if r["state"] == state}):
                    same = [r for r in recs if r["state"] == state and abs(float(r["sharpness"]) - sharpness) < 1e-12]
                    rows.append(
                        {
                            "problem": problem,
                            "method": method,
                            "state": state,
                            "sharpness": sharpness,
                            "regret_reduction": float(np.mean([r["regret_reduction"] for r in same])),
                            "top_mass_gain": float(np.mean([r["top_mass_gain"] for r in same])),
                            "bad_mass_reduction": float(np.mean([r["bad_mass_reduction"] for r in same])),
                            "negative_instance_rate": float(np.mean([r["negative_instance_rate"] for r in same])),
                        }
                    )
    fields = list(rows[0]) if rows else []
    with (out_dir / "policy_shift_cause_metrics.csv").open("w", newline="", encoding="utf-8") as f:
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
    problems = results["problems"]
    x = np.arange(len(METHODS))

    def bar_grid(metric: str, ylabel: str, title: str, fname: str, *, state: str = "sampled") -> None:
        fig, axes = plt.subplots(1, len(problems), figsize=(3.1 * len(problems), 2.95), sharey=True, constrained_layout=True)
        if len(problems) == 1:
            axes = [axes]
        for ax, problem in zip(axes, problems):
            vals = [_mean(results["records"][problem][m], metric, state=state, sharpness=1.0) for m in METHODS]
            ax.bar(x, vals, color=[COLORS[m] for m in METHODS], width=0.72)
            ax.axhline(0.0, color="#333333", lw=0.8)
            ax.set_title(PROBLEM_LABELS[problem])
            ax.set_xticks(x, METHODS, rotation=30, ha="right")
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / f"{fname}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)

    bar_grid("regret_reduction", "expected regret reduction", "One equalized loss step: expected quality improvement", "01_equal_step_regret_reduction")
    bar_grid("top_mass_gain", "top-solution probability gain", "One equalized loss step: probability mass moved to good candidates", "02_top_mass_gain")
    bar_grid("bad_mass_reduction", "bad-solution probability reduction", "One equalized loss step: probability mass removed from bad candidates", "03_bad_mass_reduction")
    bar_grid("negative_instance_rate", "fraction of instances", "Failure rate: one loss step makes expected regret worse", "04_negative_instance_rate")

    fig, axes = plt.subplots(1, len(problems), figsize=(3.1 * len(problems), 2.95), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        for method in METHODS:
            recs = results["records"][problem][method]
            xs = sorted({float(r["sharpness"]) for r in recs if r["state"] == "aligned"})
            ys = [_mean(recs, "regret_reduction", state="aligned", sharpness=s) for s in xs]
            ax.plot(xs, ys, marker="o", ms=2.7, lw=1.45, color=COLORS[method], label=method)
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xlabel("aligned policy sharpness")
    axes[0].set_ylabel("expected regret reduction")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Late-stage test: does the loss still improve an already aligned policy?", fontweight="bold")
    fig.savefig(out_dir / "05_late_stage_aligned_policy.png", bbox_inches="tight")
    fig.savefig(out_dir / "05_late_stage_aligned_policy.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(3.1 * len(problems), 2.95), sharey=True, constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]
    for ax, problem in zip(axes, problems):
        for method in METHODS:
            recs = results["records"][problem][method]
            xs = sorted({float(r["sharpness"]) for r in recs if r["state"] == "misaligned"})
            ys = [_mean(recs, "regret_reduction", state="misaligned", sharpness=s) for s in xs]
            ax.plot(xs, ys, marker="o", ms=2.7, lw=1.45, color=COLORS[method], label=method)
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(PROBLEM_LABELS[problem])
        ax.set_xlabel("misalignment sharpness")
    axes[0].set_ylabel("expected regret reduction")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Correction test: does the loss recover when policy ranks bad solutions too high?", fontweight="bold")
    fig.savefig(out_dir / "06_misaligned_policy_correction.png", bbox_inches="tight")
    fig.savefig(out_dir / "06_misaligned_policy_correction.pdf", bbox_inches="tight")
    plt.close(fig)

    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Five-Method Policy-Shift Cause Analysis\n\n"
            "Every figure compares PO, BOPO, SLL, Loss-only, and Loss+Weighting on all four problems.\n"
            "Each method receives the same candidate pool and is converted into a one-step equal-norm update on rollout log-probabilities.\n"
            "The plots report actual policy-distribution consequences: expected regret reduction, top-solution mass gain, bad-solution mass reduction, failure rate, and late-stage behavior.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sharpness", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--step-size", type=float, default=0.08)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    problems = [p.strip() for p in str(args.problems).split(",") if p.strip()]
    sharpness = [float(x) for x in str(args.sharpness).split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "five_method_policy_shift_causes" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = collect(
        problems=problems,
        batches=max(int(args.batches), 1),
        seed=int(args.seed),
        device=device,
        sharpness=sharpness,
        step_size=float(args.step_size),
    )
    plot(results, out_dir)
    write_tables(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
