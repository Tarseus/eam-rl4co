from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "PTP") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "PTP"))

from eval_downloaded_routing_checkpoints import (  # noqa: E402
    DownloadEntry,
    build_model,
    load_manifest,
)
from final_gradient_behavior_analysis import (  # noqa: E402
    ProblemSpec,
    VARIANTS,
    analyze_variant,
    build_problem_specs,
    rollout_feature_caches,
)


PROBLEM_METHODS = {
    "tsp100": ["po", "loss_only", "weighting"],
    "cvrp100": ["po", "loss_only", "weighting"],
    "ffsp100": ["po", "loss_only", "weighting"],
    "jssp10x10": ["bopo", "loss_only", "weighting"],
}
BASELINE_METHOD = {"tsp100": "po", "cvrp100": "po", "ffsp100": "po", "jssp10x10": "bopo"}
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
PAIR_PATHS = {
    ("tsp100", "loss_only"): "runs/pref_loss_tsp100_discovery/20260317-131507/best_pair.json",
    ("tsp100", "weighting"): "runs/pref_builder_weight_search_tsp100/20260414-113757/best_pair.json",
    ("cvrp100", "loss_only"): "runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008/best_pair.json",
    ("cvrp100", "weighting"): "runs/pref_builder_weight_search_cvrp100/20260416-093909/best_pair.json",
    ("ffsp100", "loss_only"): "runs/pref_loss_ffsp100_discovery/20260403-142801/best_pair.json",
    ("ffsp100", "weighting"): "runs/pref_builder_weight_search_ffsp100/20260416-111514/best_pair.json",
    ("jssp10x10", "loss_only"): "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json",
    ("jssp10x10", "weighting"): "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json",
}


def _find_entry(entries: list[DownloadEntry], problem: str, method: str) -> DownloadEntry:
    for entry in entries:
        if entry.problem_key == problem and entry.method == method:
            return entry
    raise KeyError(f"Missing manifest entry for {problem}/{method}")


def _tensor_to_rows(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().float().cpu().reshape(-1).tolist()]


def _routing_or_ffsp_instance_scores(entry: DownloadEntry, *, limit: int, batch_size: int, device: str) -> list[float]:
    from rl4co.utils.ops import unbatchify

    model, _, _ = build_model(entry, REPO_ROOT)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    model.setup(stage="test")
    model.data_cfg["test_data_size"] = int(limit)
    model.data_cfg["test_batch_size"] = int(batch_size)
    scores: list[float] = []
    with torch.no_grad():
        for batch in model.test_dataloader():
            batch = batch.to(torch_device)
            td = model.env.reset(batch)
            n_start = model.num_starts
            if n_start is None or int(n_start) <= 0:
                n_start = model.env.get_num_starts(td)
            out = model.policy(td, model.env, phase="test", num_starts=int(n_start), return_actions=False)
            reward = unbatchify(out["reward"], (0, int(n_start)))
            best_reward = reward.max(dim=-1).values
            scores.extend(_tensor_to_rows(best_reward))
            if len(scores) >= limit:
                break
    return scores[:limit]


def _jssp_instance_scores(entry: DownloadEntry, *, limit: int, device: str) -> list[float]:
    from rl4co.models.zoo.mgl_jssp.data import load_instance
    from rl4co.models.zoo.mgl_jssp.model import MGLJSSPModel
    from rl4co.models.zoo.mgl_jssp.sampling import sampling
    from scripts.prepare_bopo_jsp_data import prepare_bopo_jsp

    prepare_bopo_jsp(REPO_ROOT)
    torch_device = torch.device(device)
    env = __import__("rl4co.envs", fromlist=["JSSPEnv"]).JSSPEnv(generator_params={"num_jobs": 10, "num_machines": 10})
    model = MGLJSSPModel.load_from_checkpoint(str(entry.checkpoint_path), env=env, map_location=torch_device)
    model = model.to(torch_device)
    model.eval()
    b = int(getattr(model, "test_B", getattr(model, "val_B", 128)) or 128)
    use_greedy = bool(getattr(model, "use_greedy", False))
    files = sorted((REPO_ROOT / "data" / "jssp_bopo" / "validation").glob("*.jsp"))[:limit]
    scores: list[float] = []
    with torch.no_grad():
        for file_path in files:
            instance = load_instance(str(file_path), device=str(torch_device))
            makespans, _, _ = sampling(
                instance,
                model.encoder,
                model.decoder,
                bs=b,
                use_greedy=use_greedy,
                device=str(torch_device),
            )
            scores.append(float(-makespans.min().item()))
    return scores


def paired_final_rows(entries: list[DownloadEntry], *, limit: int, batch_size: int, device: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for problem, methods in PROBLEM_METHODS.items():
        method_scores: dict[str, list[float]] = {}
        for method in methods:
            entry = _find_entry(entries, problem, method)
            print(f"[paired] {problem}/{method}", flush=True)
            if problem == "jssp10x10":
                scores = _jssp_instance_scores(entry, limit=min(limit, 100), device=device)
            else:
                scores = _routing_or_ffsp_instance_scores(entry, limit=limit, batch_size=batch_size, device=device)
            method_scores[method] = scores
            for idx, score in enumerate(scores):
                rows.append({"problem": problem, "method": method, "instance": idx, "score": score})
        base = BASELINE_METHOD[problem]
        n = min(len(method_scores[base]), *(len(method_scores[m]) for m in methods))
        for method in methods:
            if method == base:
                continue
            for idx in range(n):
                rows.append(
                    {
                        "problem": problem,
                        "method": f"{method}_minus_{base}",
                        "instance": idx,
                        "score": method_scores[method][idx] - method_scores[base][idx],
                    }
                )
    return rows


def _configure_counterfactual_loss(model: Any, problem: str, method: str) -> None:
    if method == "po":
        model.loss_type = "po_loss"
        if problem == "ffsp100":
            model.po_impl = "exponential"
        return
    if method == "bopo":
        if hasattr(model, "baseline"):
            model.baseline = "bopo"
        if hasattr(model, "loss_type"):
            model.loss_type = "bopo_loss"
        return
    pair_path = PAIR_PATHS[(problem, method)]
    model.loss_type = "free_loss"
    model.pref_pair_json_path = pair_path
    model.free_loss_ir_json_path = None
    model.pref_builder_ir_json_path = None
    model.free_loss = None
    model.pref_builder = None
    model._resolve_pref_pair_artifacts()
    model._load_free_loss()
    model._load_pref_builder()


def _eval_model_score(model: Any, *, batches: list[Any], device: torch.device) -> float:
    from rl4co.utils.ops import unbatchify

    vals: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            td = model.env.reset(batch)
            n_start = model.num_starts
            if n_start is None or int(n_start) <= 0:
                n_start = model.env.get_num_starts(td)
            out = model.policy(td, model.env, phase="test", num_starts=int(n_start), return_actions=False)
            reward = unbatchify(out["reward"], (0, int(n_start)))
            vals.extend(_tensor_to_rows(reward.max(dim=-1).values))
    return float(np.mean(vals)) if vals else float("nan")


def counterfactual_rows(entries: list[DownloadEntry], *, device: str, eval_batches: int, batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    torch_device = torch.device(device)
    for problem in ("tsp100", "cvrp100", "ffsp100"):
        base_method = BASELINE_METHOD[problem]
        base_entry = _find_entry(entries, problem, base_method)
        base_model, _, _ = build_model(base_entry, REPO_ROOT)
        base_model = base_model.to(torch_device)
        base_model.setup(stage="test")
        base_model.data_cfg["test_batch_size"] = int(batch_size)
        eval_data = []
        for batch in base_model.test_dataloader():
            eval_data.append(batch)
            if len(eval_data) >= int(eval_batches):
                break
        train_batch = copy.deepcopy(eval_data[0]).to(torch_device)
        before = _eval_model_score(base_model, batches=eval_data, device=torch_device)
        for method in PROBLEM_METHODS[problem]:
            print(f"[one-step] {problem}/{method}", flush=True)
            model, _, _ = build_model(base_entry, REPO_ROOT)
            model = model.to(torch_device)
            model.setup(stage="test")
            model.data_cfg["test_batch_size"] = int(batch_size)
            _configure_counterfactual_loss(model, problem, method)
            opt = torch.optim.Adam(model.parameters(), lr=float(model.optimizer_kwargs.get("lr", 1e-4)))
            model.train()
            opt.zero_grad(set_to_none=True)
            out = model.shared_step(train_batch, 0, "train")
            loss = out["loss"]
            if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss).all():
                raise RuntimeError(f"non-finite one-step loss for {problem}/{method}: {loss}")
            loss.backward()
            opt.step()
            after = _eval_model_score(model, batches=eval_data, device=torch_device)
            rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "before_score": before,
                    "after_score": after,
                    "delta_score": after - before,
                    "train_loss": float(loss.detach().item()),
                }
            )
    return rows


def alignment_rows(*, device: str, batches: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = build_problem_specs(device, max(int(batches), 1))
    for problem, spec in specs.items():
        if problem not in PROBLEM_METHODS:
            continue
        print(f"[alignment] {problem}", flush=True)
        caches = rollout_feature_caches(spec, seed=1234, device=torch.device(device))
        for variant in VARIANTS[problem]:
            if variant.key not in PROBLEM_METHODS[problem]:
                continue
            row = analyze_variant(problem=spec, variant=variant, feature_caches=caches, micro_steps=0, micro_lr=0.0)
            row["method"] = row.pop("variant")
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(vals: np.ndarray, seed: int = 0) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(vals, size=vals.size, replace=True))) for _ in range(2000)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_all(out_dir: Path, paired: list[dict[str, Any]], counter: list[dict[str, Any]], align: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

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
            "grid.alpha": 0.2,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), constrained_layout=True)
    for ax, problem in zip(axes, PROBLEM_METHODS.keys()):
        base = BASELINE_METHOD[problem]
        for method in PROBLEM_METHODS[problem]:
            if method == base:
                continue
            vals = np.array([float(r["score"]) for r in paired if r["problem"] == problem and r["method"] == f"{method}_minus_{base}"])
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            xs = np.sort(vals)
            ys = np.arange(1, xs.size + 1) / xs.size
            lo, hi = _bootstrap_ci(vals)
            ax.plot(xs, ys, label=f"{METHOD_LABEL[method]} CI[{lo:.3g},{hi:.3g}]", color=METHOD_COLOR[method], lw=1.8)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_title(problem)
        ax.set_xlabel(f"score improvement over {METHOD_LABEL[base]}")
    axes[0].set_ylabel("fraction of instances")
    axes[-1].legend(frameon=False, fontsize=6.8)
    fig.suptitle("Paired final improvement on identical instances", fontweight="bold", y=1.05)
    fig.savefig(out_dir / "fig1_paired_final_improvement.pdf")
    fig.savefig(out_dir / "fig1_paired_final_improvement.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 2.4), constrained_layout=True)
    xlabels = []
    vals = []
    colors = []
    for r in counter:
        xlabels.append(f"{r['problem']}\n{METHOD_LABEL[str(r['method'])]}")
        vals.append(float(r["delta_score"]))
        colors.append(METHOD_COLOR[str(r["method"])])
    ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(np.arange(len(vals)), xlabels, rotation=35, ha="right")
    ax.set_ylabel("validation score after one update - before")
    ax.set_title("One-step counterfactual update from the same checkpoint", fontweight="bold")
    fig.savefig(out_dir / "fig2_one_step_counterfactual.pdf")
    fig.savefig(out_dir / "fig2_one_step_counterfactual.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), constrained_layout=True)
    for ax, problem in zip(axes, PROBLEM_METHODS.keys()):
        methods = PROBLEM_METHODS[problem]
        vals = []
        labels = []
        colors = []
        for method in methods:
            rec = next((r for r in align if r["problem"] == problem and r["method"] == method), None)
            if rec is None:
                continue
            vals.append(float(rec.get("gap_grad_spearman", np.nan)))
            labels.append(METHOD_LABEL[method])
            colors.append(METHOD_COLOR[method])
        ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=30, ha="right")
        ax.set_title(problem)
    axes[0].set_ylabel("Spearman: objective gap vs gradient mass")
    fig.suptitle("Gradient alignment with objective-improving direction", fontweight="bold", y=1.05)
    fig.savefig(out_dir / "fig3_gradient_alignment.pdf")
    fig.savefig(out_dir / "fig3_gradient_alignment.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--paired-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--counter-eval-batches", type=int, default=2)
    parser.add_argument("--alignment-batches", type=int, default=4)
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "stronger_interpretability" / time.strftime("%Y%m%d-%H%M%S")))
    entries = load_manifest(REPO_ROOT / "downloads" / "manifest.json", REPO_ROOT)
    paired = paired_final_rows(entries, limit=int(args.paired_limit), batch_size=int(args.batch_size), device=str(args.device))
    counter = [] if args.skip_counterfactual else counterfactual_rows(entries, device=str(args.device), eval_batches=int(args.counter_eval_batches), batch_size=int(args.batch_size))
    align = alignment_rows(device=str(args.device), batches=int(args.alignment_batches))
    _write_csv(out_dir / "paired_final_rows.csv", paired)
    _write_csv(out_dir / "one_step_counterfactual.csv", counter)
    _write_csv(out_dir / "gradient_alignment_rows.csv", align)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"paired_rows": len(paired), "counterfactual_rows": len(counter), "alignment_rows": len(align)}, f, indent=2)
    plot_all(out_dir, paired, counter, align)
    print(f"[done] outputs={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
