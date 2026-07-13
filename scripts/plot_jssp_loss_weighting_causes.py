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
from scripts.final_gradient_behavior_analysis import (  # noqa: E402
    build_problem_specs,
    rollout_feature_caches,
)


WEIGHTING_PAIR_PATH = "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json"
METHODS = ["PO", "BOPO", "SLL", "Loss+Weighting"]
COLORS = {
    "PO": "#4C78A8",
    "BOPO": "#B279A2",
    "SLL": "#F58518",
    "Loss+Weighting": "#54A24B",
}


def _load_weighting_pair(path: str) -> tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor]]:
    payload = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    builder = compile_preference_builder(pref_builder_ir_from_json(payload["g_ir"]))
    loss = compile_free_loss(free_loss_ir_from_json(payload["f_ir"]))

    def build(feature_cache: Mapping[str, torch.Tensor]) -> PrefBatch:
        return builder.build_fn(feature_cache, {"analysis": True})

    def loss_fn(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        expects = [str(x) for x in getattr(loss.ir.implementation_hint, "expects", [])]
        sub = {k: batch[k] for k in expects if k in batch} if expects else dict(batch)
        return loss.loss_fn(sub, {}, {"alpha": 1.0, "analysis": True})

    return build, loss_fn


def _po_loss(reward: torch.Tensor, log_prob: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    preference = (reward[:, :, None] > reward[:, None, :]).float()
    logp_pair = alpha * (log_prob[:, :, None] - log_prob[:, None, :])
    return -(F.logsigmoid(logp_pair) * preference).mean()


def _bopo_loss(
    reward: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    sequence_length: torch.Tensor | None,
    alpha: float = 1.0,
    select_k: int = 4,
) -> torch.Tensor:
    objective = -reward
    score = log_prob
    if sequence_length is not None:
        score = score / sequence_length.clamp_min(1.0)
    batch_size, num_rollouts = reward.shape
    if num_rollouts % select_k != 0:
        raise ValueError(f"BOPO paper selection needs num_rollouts % select_k == 0, got {num_rollouts} and {select_k}")
    losses: list[torch.Tensor] = []
    eps = 1e-8
    stride = num_rollouts // select_k
    for i in range(batch_size):
        selected = reward[i].sort(descending=True).indices[::stride][:select_k]
        obj = objective[i, selected]
        logp = alpha * score[i, selected]
        factor = (obj[1:] + eps) / (obj[:1] + eps)
        losses.append(-F.logsigmoid(factor * (logp[:1] - logp[1:])))
    return torch.cat([x.reshape(-1) for x in losses]).mean()


def _sll_pseudo_label_loss(reward: torch.Tensor, log_prob: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    best = reward.argmax(dim=1)
    chosen = log_prob.gather(1, best[:, None]).squeeze(1)
    return -(alpha * chosen).mean()


def _refresh_cache(base_fc: Mapping[str, torch.Tensor], log_prob: torch.Tensor) -> dict[str, torch.Tensor]:
    objective = base_fc["objective"].detach()
    seq_len = base_fc.get("seq_len")
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.full_like(log_prob, float(objective.shape[1]))
    reward = -objective
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


def _method_loss(
    method: str,
    fc: Mapping[str, torch.Tensor],
    log_prob: torch.Tensor,
    weighting_builder: Callable[[Mapping[str, torch.Tensor]], PrefBatch],
    weighting_loss: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
) -> torch.Tensor:
    objective = fc["objective"].detach()
    reward = -objective
    if method == "PO":
        return _po_loss(reward, log_prob)
    if method == "BOPO":
        seq_len = fc.get("seq_len")
        seq = seq_len if isinstance(seq_len, torch.Tensor) else None
        return _bopo_loss(reward, log_prob, sequence_length=seq)
    if method == "SLL":
        return _sll_pseudo_label_loss(reward, log_prob)
    if method == "Loss+Weighting":
        fc_live = _refresh_cache(fc, log_prob)
        pref_raw = weighting_builder(fc_live)
        pref = PrefBatch(
            mode=pref_raw.mode,
            pair_idx=pref_raw.pair_idx,
            list_idx=pref_raw.list_idx,
            weight=pref_raw.weight.detach() if isinstance(pref_raw.weight, torch.Tensor) else pref_raw.weight,
            meta=dict(pref_raw.meta or {}),
        )
        if pref.num_examples() <= 0:
            return log_prob.sum() * 0.0
        return weighting_loss(pref.to_pairwise_loss_batch(fc_live))
    raise KeyError(method)


def _grad_record(
    method: str,
    fc: Mapping[str, torch.Tensor],
    weighting_builder: Callable[[Mapping[str, torch.Tensor]], PrefBatch],
    weighting_loss: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    *,
    sharpen: float = 1.0,
) -> dict[str, np.ndarray | float]:
    base_logp = fc["log_prob"].detach()
    centered = base_logp - base_logp.mean(dim=1, keepdim=True)
    lp = (base_logp.mean(dim=1, keepdim=True) + float(sharpen) * centered).clone().detach().requires_grad_(True)
    loss = _method_loss(method, fc, lp, weighting_builder, weighting_loss)
    loss.backward()
    grad = lp.grad.detach()
    update = -grad
    mass = grad.abs()
    objective = fc["objective"].detach()
    rank = objective.argsort(dim=1, descending=False).argsort(dim=1).float()
    max_rank = max(int(objective.shape[1]) - 1, 1)
    rank_norm = (rank / float(max_rank)).cpu().numpy()
    regret = (objective - objective.min(dim=1, keepdim=True).values)
    regret_norm = regret / regret.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    return {
        "loss": float(loss.detach().item()),
        "rank_norm": rank_norm.reshape(-1),
        "regret_norm": regret_norm.cpu().numpy().reshape(-1),
        "mass": mass.cpu().numpy().reshape(-1),
        "update": update.cpu().numpy().reshape(-1),
    }


def _bin_mean(x: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(bins, np.nan, dtype=np.float64)
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1] if i < bins - 1 else x <= edges[i + 1])
        if np.any(mask):
            out[i] = float(np.nanmean(y[mask]))
    return centers, out


def _bin_share(x: np.ndarray, mass: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    centers, summed = _bin_mean(x, mass, bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = np.zeros(bins, dtype=np.float64)
    total = float(np.nansum(mass))
    if total <= 1e-12:
        return centers, out
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1] if i < bins - 1 else x <= edges[i + 1])
        out[i] = float(np.nansum(mass[mask]) / total)
    return centers, out


def collect(
    *,
    batches: int,
    seed: int,
    device: str,
    sharpen_grid: list[float],
) -> dict[str, Any]:
    weighting_builder, weighting_loss = _load_weighting_pair(WEIGHTING_PAIR_PATH)
    specs = build_problem_specs(device, batches)
    spec = specs["jssp10x10"]
    _ensure_jssp_data(REPO_ROOT)
    caches = rollout_feature_caches(spec, seed=seed, device=torch.device(device))
    records: dict[str, list[dict[str, np.ndarray | float]]] = {m: [] for m in METHODS}
    sharpen_records: dict[str, list[dict[str, float]]] = {m: [] for m in METHODS}
    for fc in caches:
        for method in METHODS:
            records[method].append(_grad_record(method, fc, weighting_builder, weighting_loss))
            for sharpen in sharpen_grid:
                rec = _grad_record(method, fc, weighting_builder, weighting_loss, sharpen=sharpen)
                mass = np.asarray(rec["mass"], dtype=np.float64)
                rank = np.asarray(rec["rank_norm"], dtype=np.float64)
                update = np.asarray(rec["update"], dtype=np.float64)
                total = float(np.nansum(mass))
                top_mask = rank <= 0.2
                bottom_mask = rank >= 0.8
                sharpen_records[method].append(
                    {
                        "sharpen": float(sharpen),
                        "loss": float(rec["loss"]),
                        "total_mass": total,
                        "top20_mass_share": float(np.nansum(mass[top_mask]) / (total + 1e-12)),
                        "bottom20_positive_share": float(
                            np.nansum(np.maximum(update[bottom_mask], 0.0)) / (np.nansum(np.abs(update)) + 1e-12)
                        ),
                    }
                )
    return {"records": records, "sharpen_records": sharpen_records, "batches": len(caches)}


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

    target = root / "data" / "jssp_bopo"
    splits = {"train": 96, "validation": 24}
    rng = np.random.default_rng(12345678)
    for split, count in splits.items():
        split_dir = target / split
        split_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(split_dir.glob("10x10_random_*.jsp"))
        if len(existing) >= count:
            continue
        for idx in range(count):
            path = split_dir / f"10x10_random_{idx:05d}.jsp"
            if not path.exists():
                _write_random_jsp(path, num_jobs=10, num_machines=10, rng=rng)


def summarize(results: dict[str, Any]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method, recs in results["records"].items():
        for rec in recs:
            mass = np.asarray(rec["mass"], dtype=np.float64)
            rank = np.asarray(rec["rank_norm"], dtype=np.float64)
            regret = np.asarray(rec["regret_norm"], dtype=np.float64)
            update = np.asarray(rec["update"], dtype=np.float64)
            total = float(np.nansum(mass))
            rows.append(
                {
                    "method": method,
                    "loss": float(rec["loss"]),
                    "top1_mass_share": float(np.nansum(mass[rank <= 1.0 / 127.0]) / (total + 1e-12)),
                    "top20_mass_share": float(np.nansum(mass[rank <= 0.2]) / (total + 1e-12)),
                    "bottom20_mass_share": float(np.nansum(mass[rank >= 0.8]) / (total + 1e-12)),
                    "high_regret_mass_share": float(np.nansum(mass[regret >= 0.8]) / (total + 1e-12)),
                    "bad_positive_update_share": float(
                        np.nansum(np.maximum(update[rank >= 0.8], 0.0)) / (np.nansum(np.abs(update)) + 1e-12)
                    ),
                }
            )
    return rows


def plot(results: dict[str, Any], out_dir: Path, bins: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
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

    records: dict[str, list[dict[str, np.ndarray | float]]] = results["records"]

    def cat(method: str, key: str) -> np.ndarray:
        return np.concatenate([np.asarray(r[key], dtype=np.float64) for r in records[method]], axis=0)

    fig, ax = plt.subplots(figsize=(5.6, 3.35), constrained_layout=True)
    for method in METHODS:
        x, y = _bin_share(cat(method, "rank_norm"), cat(method, "mass"), bins)
        ax.plot(x, y, marker="o", lw=1.9, ms=3.5, color=COLORS[method], label=method)
    ax.set_xlabel("rollout rank (0=best schedule, 1=worst)")
    ax.set_ylabel("share of total |gradient|")
    ax.set_title("Cause 1: gradient budget by rank")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "01_gradient_budget_by_rank.png", bbox_inches="tight")
    fig.savefig(out_dir / "01_gradient_budget_by_rank.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.35), constrained_layout=True)
    for method in METHODS:
        mass = cat(method, "mass")
        update = cat(method, "update")
        denom = np.nanmax(np.abs(update))
        yval = update / denom if np.isfinite(denom) and denom > 1e-12 else update
        x, y = _bin_mean(cat(method, "rank_norm"), yval, bins)
        ax.plot(x, y, marker="o", lw=1.9, ms=3.5, color=COLORS[method], label=method)
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xlabel("rollout rank (0=best schedule, 1=worst)")
    ax.set_ylabel("normalized signed log-prob update")
    ax.set_title("Cause 2: signed update by rank")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "02_signed_update_by_rank.png", bbox_inches="tight")
    fig.savefig(out_dir / "02_signed_update_by_rank.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.35), constrained_layout=True)
    for method in METHODS:
        x, y = _bin_share(cat(method, "regret_norm"), cat(method, "mass"), bins)
        ax.plot(x, y, marker="o", lw=1.9, ms=3.5, color=COLORS[method], label=method)
    ax.set_xlabel("normalized regret from instance best")
    ax.set_ylabel("share of total |gradient|")
    ax.set_title("Cause 3: gradient budget by regret")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "03_gradient_budget_by_regret.png", bbox_inches="tight")
    fig.savefig(out_dir / "03_gradient_budget_by_regret.pdf", bbox_inches="tight")
    plt.close(fig)

    sharpen_records: dict[str, list[dict[str, float]]] = results["sharpen_records"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.25), constrained_layout=True)
    for method in METHODS:
        rows = sharpen_records[method]
        xs = sorted({float(r["sharpen"]) for r in rows})
        total_mass = []
        top20 = []
        for x in xs:
            same = [r for r in rows if abs(float(r["sharpen"]) - x) < 1e-12]
            total_mass.append(float(np.mean([r["total_mass"] for r in same])))
            top20.append(float(np.mean([r["top20_mass_share"] for r in same])))
        norm_mass = np.asarray(total_mass, dtype=np.float64)
        norm_mass = norm_mass / max(float(norm_mass[0]), 1e-12)
        axes[0].plot(xs, norm_mass, marker="o", lw=1.9, ms=3.5, color=COLORS[method], label=method)
        axes[1].plot(xs, top20, marker="o", lw=1.9, ms=3.5, color=COLORS[method], label=method)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("policy sharpening multiplier")
    axes[1].set_xlabel("policy sharpening multiplier")
    axes[0].set_ylabel("relative total |gradient|")
    axes[1].set_ylabel("top-20% rank budget share")
    axes[0].set_title("late-stage signal saturation")
    axes[1].set_title("late-stage target concentration")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Cause 4: late-stage sharpening sensitivity", fontweight="bold")
    fig.savefig(out_dir / "04_late_stage_sharpening_sensitivity.png", bbox_inches="tight")
    fig.savefig(out_dir / "04_late_stage_sharpening_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def write_tables(results: dict[str, Any], out_dir: Path) -> None:
    summary_rows = summarize(results)
    fields = list(summary_rows[0].keys()) if summary_rows else []
    with (out_dir / "cause_summary_per_cache.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    agg: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [r for r in summary_rows if r["method"] == method]
        out: dict[str, Any] = {"method": method}
        for key in fields:
            if key == "method":
                continue
            vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.nanmean(vals))
            out[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg.append(out)
    agg_fields = sorted({k for row in agg for k in row})
    with (out_dir / "cause_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg)
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write(
            "# JSSP Loss/Weighting Cause Analysis\n\n"
            "Every figure compares PO, BOPO, SLL, and Loss+Weighting on the same JSSP10x10 rollout caches.\n"
            "The plots intentionally show cause metrics only: gradient budget allocation, signed update direction, regret allocation, and late-stage sharpening sensitivity.\n\n"
            "- `01_gradient_budget_by_rank`: shows sample utilization across rollout ranks.\n"
            "- `02_signed_update_by_rank`: shows whether better schedules are increased and worse schedules are suppressed.\n"
            "- `03_gradient_budget_by_regret`: shows whether gradient mass is spent on high-regret/noisy candidates.\n"
            "- `04_late_stage_sharpening_sensitivity`: probes PO-style late-stage saturation/concentration by sharpening policy log-probs without changing objective ranks.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--sharpen-grid", default="0.25,0.5,1,2,4,8")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "jssp_loss_weighting_causes" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    sharpen_grid = [float(x) for x in str(args.sharpen_grid).split(",") if x.strip()]
    results = collect(batches=max(int(args.batches), 1), seed=int(args.seed), device=device, sharpen_grid=sharpen_grid)
    plot(results, out_dir, bins=max(int(args.bins), 4))
    write_tables(results, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
