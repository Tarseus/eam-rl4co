from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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

from fitness.free_loss_fidelity import (  # noqa: E402
    PrefBatch,
    extract_feature_cache,
    _build_mgl_jssp_model,
    _mgl_jssp_selected_log_prob_step,
    _rl4co_build_env,
    _rl4co_build_policy,
    _rl4co_objective_from_reward,
    _rl4co_rollout,
)
from fitness.ptp_high_fidelity import HighFidelityConfig, resolve_pomo_size, _set_seed  # noqa: E402
from ptp_discovery.free_loss_compiler import compile_free_loss  # noqa: E402
from ptp_discovery.free_loss_ir import ir_from_json as free_loss_ir_from_json  # noqa: E402
from ptp_discovery.pref_builder_compiler import compile_preference_builder  # noqa: E402
from ptp_discovery.pref_builder_ir import ir_from_json as pref_builder_ir_from_json  # noqa: E402


@dataclass(frozen=True)
class ProblemSpec:
    key: str
    baseline: str
    hf: HighFidelityConfig
    batch_size: int
    batches: int
    po_impl: str = "bt"
    bopo_select_k: int = 4


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    kind: str
    pair_path: str | None = None
    po_impl: str = "bt"


class AllPairsBuilder:
    def __call__(self, feature_cache: Mapping[str, torch.Tensor]) -> PrefBatch:
        objective = feature_cache["objective"]
        mask = objective[:, :, None] < objective[:, None, :]
        b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
        return PrefBatch(
            mode="pairwise",
            pair_idx=(b_idx, winner_idx, loser_idx),
            weight=None,
            meta={"builder": "all_pairs"},
        )


class BopoAnchorBestBuilder:
    def __init__(self, select_k: int) -> None:
        self.select_k = int(select_k)

    def __call__(self, feature_cache: Mapping[str, torch.Tensor]) -> PrefBatch:
        objective = feature_cache["objective"]
        batch_size, num_pomo = objective.shape
        if num_pomo % self.select_k != 0:
            raise ValueError(f"BOPO paper selection requires K % select_k == 0, got K={num_pomo}, k={self.select_k}")
        sorted_idx = objective.sort(dim=1, descending=False).indices
        stride = num_pomo // self.select_k
        selected = sorted_idx[:, ::stride][:, : self.select_k]
        b_grid = torch.arange(batch_size, device=objective.device)[:, None].expand(batch_size, self.select_k - 1)
        winner_idx = selected[:, :1].expand(batch_size, self.select_k - 1)
        loser_idx = selected[:, 1:]
        b_flat = b_grid.reshape(-1)
        w_flat = winner_idx.reshape(-1)
        l_flat = loser_idx.reshape(-1)
        eps = torch.as_tensor(1e-8, device=objective.device, dtype=objective.dtype)
        weight = (objective[b_flat, l_flat] + eps) / (objective[b_flat, w_flat] + eps)
        return PrefBatch(
            mode="pairwise",
            pair_idx=(b_flat, w_flat, l_flat),
            weight=weight.detach(),
            meta={"builder": "bopo_anchor_best", "select_k": self.select_k},
        )


def po_loss_from_batch(batch: Mapping[str, torch.Tensor], *, alpha: float, impl: str) -> torch.Tensor:
    logit = float(alpha) * (batch["log_prob_w"] - batch["log_prob_l"])
    if impl == "exponential":
        return -logit.mean()
    return -F.logsigmoid(logit).mean()


def bopo_loss_from_batch(batch: Mapping[str, torch.Tensor], *, alpha: float) -> torch.Tensor:
    logit = float(alpha) * (batch["log_prob_w"] - batch["log_prob_l"])
    weight = batch.get("weight")
    if not isinstance(weight, torch.Tensor):
        weight = torch.ones_like(logit)
    return -F.logsigmoid(weight.detach() * logit).mean()


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def load_final_pair(path: str) -> tuple[Callable[[Mapping[str, torch.Tensor]], PrefBatch], Callable[[Mapping[str, torch.Tensor]], torch.Tensor], dict[str, Any]]:
    with open(REPO_ROOT / path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    builder_ir = pref_builder_ir_from_json(payload["g_ir"])
    loss_ir = free_loss_ir_from_json(payload["f_ir"])
    compiled_g = compile_preference_builder(builder_ir)
    compiled_f = compile_free_loss(loss_ir)

    def build(feature_cache: Mapping[str, torch.Tensor]) -> PrefBatch:
        return compiled_g.build_fn(feature_cache, {"analysis": True})

    def loss(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        expects = [str(x) for x in getattr(compiled_f.ir.implementation_hint, "expects", [])]
        loss_batch = {k: batch[k] for k in expects if k in batch} if expects else dict(batch)
        return compiled_f.loss_fn(loss_batch, {}, {"alpha": 1.0, "analysis": True})

    return build, loss, payload


def build_problem_specs(device: str, batches: int) -> dict[str, ProblemSpec]:
    common_pomo = {
        "policy_name": "pomo",
        "rollout_strategy": "auto",
        "objective_sign": "neg_reward",
        "pomo_size": None,
        "policy_kwargs": {
            "po4cops_compat": True,
            "embed_dim": 128,
            "num_encoder_layers": 6,
            "decoder_layer_num": 1,
            "qkv_dim": 16,
            "num_heads": 8,
            "feedforward_hidden": 512,
            "tanh_clipping": 50,
            "eval_type": "argmax",
            "val_decode_type": "greedy",
            "test_decode_type": "greedy",
        },
    }
    ffsp_policy = {
        "embed_dim": 256,
        "num_encoder_layers": 3,
        "num_heads": 16,
        "normalization": "instance",
        "use_graph_context": False,
        "feedforward_hidden": 512,
        "train_decode_type": "sampling",
        "val_decode_type": "greedy",
        "test_decode_type": "greedy",
    }
    jssp_policy = {
        "baseline": "bopo",
        "train_data_dir": "data/jssp_bopo/train",
        "val_data_dir": "data/jssp_bopo/validation",
        "use_cached": True,
        "batch_size": 1,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "dataloader_num_workers": 0,
        "enc_hidden": 64,
        "enc_out": 128,
        "mem_hidden": 64,
        "mem_out": 128,
        "clf_hidden": 128,
        "B": 128,
        "val_B": 128,
        "test_B": 128,
        "K": 16,
        "D": 1,
        "pair_mode": "anchor_best",
        "po_impl": "bt",
        "greedy": 0,
        "use_shape_buckets": True,
        "bucket_drop_last": False,
        "allowed_shapes": [[10, 10]],
    }
    return {
        "tsp100": ProblemSpec(
            key="tsp100",
            baseline="po",
            batch_size=8,
            batches=batches,
            hf=HighFidelityConfig(
                problem="tsp",
                env_name="tsp",
                generator_params={"num_loc": 100},
                train_problem_size=100,
                alpha=0.05,
                device=device,
                **common_pomo,
            ),
        ),
        "cvrp100": ProblemSpec(
            key="cvrp100",
            baseline="po",
            batch_size=8,
            batches=batches,
            hf=HighFidelityConfig(
                problem="cvrp",
                env_name="cvrp",
                generator_params={"num_loc": 100},
                train_problem_size=100,
                alpha=0.05,
                device=device,
                **common_pomo,
            ),
        ),
        "ffsp100": ProblemSpec(
            key="ffsp100",
            baseline="po",
            batch_size=4,
            batches=batches,
            po_impl="exponential",
            hf=HighFidelityConfig(
                problem="ffsp",
                env_name="ffsp",
                generator_params={"num_stage": 3, "num_machine": 4, "num_job": 100, "flatten_stages": False},
                policy_name="matnet",
                policy_kwargs=ffsp_policy,
                rollout_strategy="auto",
                objective_sign="neg_reward",
                train_problem_size=100,
                pomo_size=24,
                alpha=1.0,
                train_batch_size=4,
                precision="32-true",
                device=device,
            ),
        ),
        "jssp10x10": ProblemSpec(
            key="jssp10x10",
            baseline="bopo",
            batch_size=1,
            batches=batches,
            bopo_select_k=4,
            hf=HighFidelityConfig(
                problem="jssp",
                env_name="jssp",
                generator_params={"num_jobs": 10, "num_machines": 10, "min_processing_time": 1, "max_processing_time": 99},
                policy_name="mgl_jssp",
                policy_kwargs=jssp_policy,
                rollout_strategy="mgl_sampling",
                objective_sign="neg_reward",
                train_problem_size=10,
                valid_problem_sizes=(10,),
                pomo_size=128,
                train_batch_size=1,
                alpha=1.0,
                learning_rate=2e-4,
                precision="32-true",
                device=device,
            ),
        ),
    }


VARIANTS: dict[str, list[VariantSpec]] = {
    "tsp100": [
        VariantSpec("po", "PO", "po"),
        VariantSpec("loss_only", "Loss only", "pair", "runs/pref_loss_tsp100_discovery/20260317-131507/best_pair.json"),
        VariantSpec("weighting", "Loss + weighting", "pair", "runs/pref_builder_weight_search_tsp100/20260414-113757/best_pair.json"),
    ],
    "cvrp100": [
        VariantSpec("po", "PO", "po"),
        VariantSpec("loss_only", "Loss only", "pair", "runs/pref_loss_cvrp100_from_tsp100_elite/20260320-224008/best_pair.json"),
        VariantSpec("weighting", "Loss + weighting", "pair", "runs/pref_builder_weight_search_cvrp100/20260416-093909/best_pair.json"),
    ],
    "ffsp100": [
        VariantSpec("po", "PO", "po", po_impl="exponential"),
        VariantSpec("loss_only", "Loss only", "pair", "runs/pref_loss_ffsp100_discovery/20260403-142801/best_pair.json"),
        VariantSpec("weighting", "Loss + weighting", "pair", "runs/pref_builder_weight_search_ffsp100/20260416-111514/best_pair.json"),
    ],
    "jssp10x10": [
        VariantSpec("bopo", "BOPO", "bopo"),
        VariantSpec("loss_only", "Loss only", "pair", "runs/pref_loss_jssp10x10_from_ffsp100_elite/20260416-113409/best_pair.json"),
        VariantSpec("weighting", "Loss + weighting", "pair", "runs/pref_builder_weight_search_jssp10x10_from_best_loss/20260417-123033/best_pair.json"),
    ],
}


def rollout_feature_caches(spec: ProblemSpec, *, seed: int, device: torch.device) -> list[dict[str, torch.Tensor]]:
    caches: list[dict[str, torch.Tensor]] = []
    _set_seed(seed)
    if spec.key == "jssp10x10":
        from rl4co.models.zoo.mgl_jssp.sampling import solve_jsp

        model = _build_mgl_jssp_model(spec.hf, problem_size=spec.hf.train_problem_size)
        model = model.to(device)
        model.eval()
        model.setup("fit")
        loader = model.train_dataloader()
        loader_iter = iter(loader)
        batch_rollouts = int(getattr(model, "B", resolve_pomo_size(spec.hf.pomo_size, spec.hf.train_problem_size)) or 1)
        for batch_id in range(spec.batches):
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

    env = _rl4co_build_env(spec.hf, spec.hf.train_problem_size).to(device)
    policy, rollout_strategy = _rl4co_build_policy(spec.hf, env)
    policy = policy.to(device)
    policy.eval()
    num_rollouts = resolve_pomo_size(spec.hf.pomo_size, spec.hf.train_problem_size)
    for batch_id in range(spec.batches):
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


def analyze_variant(
    *,
    problem: ProblemSpec,
    variant: VariantSpec,
    feature_caches: list[dict[str, torch.Tensor]],
    micro_steps: int,
    micro_lr: float,
) -> dict[str, Any]:
    if variant.kind == "po":
        builder = AllPairsBuilder()
        loss_fn = lambda batch: po_loss_from_batch(batch, alpha=problem.hf.alpha, impl=variant.po_impl or problem.po_impl)
        source = "baseline_po"
    elif variant.kind == "bopo":
        builder = BopoAnchorBestBuilder(problem.bopo_select_k)
        loss_fn = lambda batch: bopo_loss_from_batch(batch, alpha=problem.hf.alpha)
        source = "baseline_bopo"
    else:
        assert variant.pair_path is not None
        builder, loss_fn, payload = load_final_pair(variant.pair_path)
        source = variant.pair_path

    def refresh_cache(base_fc: Mapping[str, torch.Tensor], objective: torch.Tensor, log_prob: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = base_fc.get("seq_len")
        if not isinstance(seq_len, torch.Tensor):
            seq_len = torch.full_like(log_prob, float(max(int(problem.hf.train_problem_size), 1)))
        reward_like = -objective
        extra: dict[str, torch.Tensor] = {
            "advantage": reward_like - reward_like.mean(dim=1, keepdim=True),
            "seq_len": seq_len.to(device=log_prob.device, dtype=log_prob.dtype),
            "log_prob_mean": log_prob / seq_len.to(device=log_prob.device, dtype=log_prob.dtype).clamp_min(1.0),
        }
        for key in ("log_prob_step", "entropy", "entropy_mean"):
            value = base_fc.get(key)
            if isinstance(value, torch.Tensor):
                extra[key] = value
        return extract_feature_cache(objective, log_prob, extra=extra)

    losses: list[float] = []
    pair_counts: list[float] = []
    eff_ratios: list[float] = []
    winner_pass_rates: list[float] = []
    loser_pass_rates: list[float] = []
    grad_ess: list[float] = []
    weight_ess: list[float] = []
    top10_mass: list[float] = []
    gap_grad_corrs: list[float] = []
    gap_weight_corrs: list[float] = []
    micro_deltas: list[float] = []

    for fc in feature_caches:
        pref = builder(fc)
        if pref.num_examples() <= 0:
            continue
        detached_pref = PrefBatch(
            mode=pref.mode,
            pair_idx=tuple(t.detach() for t in pref.pair_idx) if pref.pair_idx is not None else None,
            list_idx=pref.list_idx.detach() if isinstance(pref.list_idx, torch.Tensor) else pref.list_idx,
            weight=pref.weight.detach() if isinstance(pref.weight, torch.Tensor) else pref.weight,
            meta=dict(pref.meta or {}),
        )
        batch0 = detached_pref.to_pairwise_loss_batch(fc)
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
        finite = torch.isfinite(gw) & torch.isfinite(gl)
        active = finite & ((gw.abs() + gl.abs()) > 1e-12)
        pair_count = int(active.sum().item())
        if pair_count <= 0:
            continue
        winner_ok = active & (gw < 0)
        loser_ok = active & (gl > 0)
        both_ok = winner_ok & loser_ok
        mass = (gw.abs() + gl.abs()).detach()[active]
        cost_gap = batch0["cost_gap"].detach()[active]
        weight = batch0.get("weight")
        weight_active = weight.detach()[active] if isinstance(weight, torch.Tensor) and weight.ndim == 1 else torch.ones_like(mass)

        mass_sum = float(mass.sum().item())
        grad_ess_val = float((mass_sum * mass_sum) / (float(pair_count) * float((mass * mass).sum().item()) + 1e-12))
        weight_sum = float(weight_active.clamp_min(0).sum().item())
        weight_ess_val = float(
            (weight_sum * weight_sum) / (float(pair_count) * float((weight_active.clamp_min(0) ** 2).sum().item()) + 1e-12)
        )
        k = max(1, int(math.ceil(0.1 * pair_count)))
        top_mass = float(torch.topk(mass, k=k).values.sum().item() / (mass_sum + 1e-12))

        losses.append(float(loss.detach().item()))
        pair_counts.append(float(pair_count))
        winner_pass_rates.append(float(winner_ok.sum().item() / pair_count))
        loser_pass_rates.append(float(loser_ok.sum().item() / pair_count))
        eff_ratios.append(float(both_ok.sum().item() / pair_count))
        grad_ess.append(grad_ess_val)
        weight_ess.append(weight_ess_val)
        top10_mass.append(top_mass)
        gap_np = cost_gap.float().cpu().numpy()
        mass_np = mass.float().cpu().numpy()
        weight_np = weight_active.float().cpu().numpy()
        gap_grad_corrs.append(_corr(_rankdata(gap_np), _rankdata(mass_np)))
        gap_weight_corrs.append(_corr(_rankdata(gap_np), _rankdata(weight_np)))

        if micro_steps > 0:
            lp = fc["log_prob"].detach().clone().requires_grad_(True)
            objective = fc["objective"].detach()
            opt = torch.optim.SGD([lp], lr=float(micro_lr))
            with torch.no_grad():
                fc_init = refresh_cache(fc, objective, lp.detach())
                pref_init = builder(fc_init)
                init_loss = float(loss_fn(pref_init.to_pairwise_loss_batch(fc_init)).detach().item())
            for _ in range(micro_steps):
                opt.zero_grad(set_to_none=True)
                fc_step = refresh_cache(fc, objective, lp)
                pref_step_raw = builder(fc_step)
                pref_step = PrefBatch(
                    mode=pref_step_raw.mode,
                    pair_idx=pref_step_raw.pair_idx,
                    list_idx=pref_step_raw.list_idx,
                    weight=pref_step_raw.weight.detach() if isinstance(pref_step_raw.weight, torch.Tensor) else pref_step_raw.weight,
                    meta=dict(pref_step_raw.meta or {}),
                )
                step_loss = loss_fn(pref_step.to_pairwise_loss_batch(fc_step))
                step_loss.backward()
                opt.step()
            with torch.no_grad():
                fc_final = refresh_cache(fc, objective, lp.detach())
                pref_final = builder(fc_final)
                final_loss = float(loss_fn(pref_final.to_pairwise_loss_batch(fc_final)).detach().item())
            micro_deltas.append(final_loss - init_loss)

    def mean_std(name: str, vals: list[float], out: dict[str, Any]) -> None:
        mean, std = _summarize([v for v in vals if np.isfinite(v)])
        out[name] = mean
        out[f"{name}_std"] = std

    result: dict[str, Any] = {
        "problem": problem.key,
        "variant": variant.key,
        "label": variant.label,
        "source": source,
        "batches": len(feature_caches),
    }
    for name, vals in [
        ("loss", losses),
        ("pair_count", pair_counts),
        ("effective_grad_ratio", eff_ratios),
        ("winner_grad_pass_rate", winner_pass_rates),
        ("loser_grad_pass_rate", loser_pass_rates),
        ("grad_ess_ratio", grad_ess),
        ("weight_ess_ratio", weight_ess),
        ("top10_grad_mass", top10_mass),
        ("gap_grad_spearman", gap_grad_corrs),
        ("gap_weight_spearman", gap_weight_corrs),
        ("micro_unroll_delta", micro_deltas),
    ]:
        mean_std(name, vals, result)
    return result


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with open(out_dir / "gradient_behavior_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "gradient_behavior_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable, skip plots: {exc}", flush=True)
        return

    colors = {"po": "#4C78A8", "bopo": "#4C78A8", "loss_only": "#F58518", "weighting": "#54A24B"}
    preferred_problems = ["tsp100", "cvrp100", "ffsp100", "jssp10x10"]
    present = {str(r["problem"]) for r in rows}
    problems = [p for p in preferred_problems if p in present]
    variant_order = {
        "tsp100": ["po", "loss_only", "weighting"],
        "cvrp100": ["po", "loss_only", "weighting"],
        "ffsp100": ["po", "loss_only", "weighting"],
        "jssp10x10": ["bopo", "loss_only", "weighting"],
    }
    row_map = {(r["problem"], r["variant"]): r for r in rows}

    def grouped_plot(metric: str, title: str, ylabel: str, path: str) -> None:
        fig, axes = plt.subplots(1, len(problems), figsize=(max(4.2 * len(problems), 5.0), 3.8), constrained_layout=True)
        if len(problems) == 1:
            axes = [axes]
        for ax, problem in zip(axes, problems):
            variants = [v for v in variant_order[problem] if (problem, v) in row_map]
            vals = [row_map[(problem, v)].get(metric, np.nan) for v in variants]
            labels = [row_map[(problem, v)]["label"] for v in variants]
            ax.bar(range(len(vals)), vals, color=[colors[v] for v in variants], width=0.72)
            ax.set_title(problem)
            ax.set_xticks(range(len(vals)), labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylabel(ylabel)
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_dir / path, dpi=220)
        plt.close(fig)

    grouped_plot(
        "effective_grad_ratio",
        "Winner-loser gradient direction is more consistently correct",
        "fraction of active pairs",
        "01_effective_gradient_ratio.png",
    )
    grouped_plot(
        "micro_unroll_delta",
        "Offline log-prob micro-unroll: lower delta means easier local optimization",
        "final loss - initial loss",
        "02_micro_unroll_delta.png",
    )
    grouped_plot(
        "grad_ess_ratio",
        "Gradient mass ESS: high means broad signal, low means concentrated signal",
        "normalized ESS",
        "03_gradient_ess.png",
    )
    grouped_plot(
        "top10_grad_mass",
        "Top-10% gradient mass: weighting should focus updates on informative pairs",
        "mass share",
        "04_top10_gradient_mass.png",
    )
    grouped_plot(
        "gap_grad_spearman",
        "Gradient mass tracks objective gap",
        "Spearman rho",
        "05_gap_gradient_correlation.png",
    )

    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(
            "# Final Gradient Behavior Analysis\n\n"
            "This analysis compares each problem against its own baseline on the same rollout caches.\n"
            "TSP/CVRP/FFSP use PO vs loss-only vs loss+weighting; JSSP uses BOPO vs loss-only vs loss+weighting.\n\n"
            "Key files:\n"
            "- `gradient_behavior_metrics.csv`: numeric table for the paper.\n"
            "- `01_effective_gradient_ratio.png`: whether gradients push winners up and losers down.\n"
            "- `02_micro_unroll_delta.png`: local optimization response on cached rollouts.\n"
            "- `03_gradient_ess.png` and `04_top10_gradient_mass.png`: gradient mass distribution.\n"
            "- `05_gap_gradient_correlation.png`: whether gradient mass follows objective gaps.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="tsp100,cvrp100,ffsp100,jssp10x10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--micro-steps", type=int, default=5)
    parser.add_argument("--micro-lr", type=float, default=0.5)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (REPO_ROOT / "figures" / "final_gradient_behavior_remote" / stamp))

    specs = build_problem_specs(str(device), max(int(args.batches), 1))
    rows: list[dict[str, Any]] = []
    for problem_key in [p.strip() for p in str(args.problems).split(",") if p.strip()]:
        spec = specs[problem_key]
        print(f"[problem] {problem_key}: rollout caches on {device}", flush=True)
        if problem_key == "jssp10x10":
            try:
                from scripts.prepare_bopo_jsp_data import prepare_bopo_jsp

                prepare_bopo_jsp(REPO_ROOT)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] prepare_bopo_jsp_data failed or unnecessary: {exc}", flush=True)
        caches = rollout_feature_caches(spec, seed=int(args.seed), device=device)
        for variant in VARIANTS[problem_key]:
            print(f"[variant] {problem_key}/{variant.key}", flush=True)
            row = analyze_variant(
                problem=spec,
                variant=variant,
                feature_caches=caches,
                micro_steps=int(args.micro_steps),
                micro_lr=float(args.micro_lr),
            )
            print(json.dumps({k: row[k] for k in ("problem", "variant", "effective_grad_ratio", "grad_ess_ratio", "top10_grad_mass", "micro_unroll_delta")}, ensure_ascii=False), flush=True)
            rows.append(row)
    write_outputs(rows, out_dir)
    print(f"[done] outputs={out_dir}", flush=True)


if __name__ == "__main__":
    main()
