from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest_sha256(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _finite_grad_norm(model: torch.nn.Module) -> tuple[float, int]:
    total_sq = 0.0
    nonzero = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError("non-finite gradient detected")
        total_sq += float(torch.sum(grad.float() ** 2).item())
        nonzero += int(torch.count_nonzero(grad).item())
    return math.sqrt(total_sq), nonzero


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path.as_posix(), map_location="cpu", weights_only=False)
    hparams = dict(payload.get("hyper_parameters", {}) or {})
    return {
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "optimizer_name": str(hparams.get("optimizer_name", hparams.get("optimizer", "unknown"))),
        "checkpoint_loss_type": str(hparams.get("loss_type", "unknown")),
        "checkpoint_pref_pair_json_path": hparams.get("pref_pair_json_path"),
    }


def _restore(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state restore mismatch: missing={missing}, unexpected={unexpected}")


def _configure_pair(model: Any, pair_json: Path) -> None:
    model.loss_type = "free_loss"
    model.pref_pair_json_path = pair_json.as_posix()
    model.free_loss_ir_json_path = None
    model.pref_builder_ir_json_path = None
    model.free_loss = None
    model.pref_builder = None
    model._resolve_pref_pair_artifacts()
    model._load_free_loss()
    model._load_pref_builder()
    if getattr(model, "free_loss", None) is None or getattr(model, "pref_builder", None) is None:
        raise RuntimeError("frozen preference pair did not compile")


def _jssp_eval(model: Any, instance: dict[str, Any], *, device: torch.device, b: int, seed: int) -> float:
    from rl4co.models.zoo.mgl_jssp.sampling import sampling

    _seed_everything(seed, device)
    model.eval()
    with torch.inference_mode():
        makespans, _, _ = sampling(
            instance,
            model.encoder,
            model.decoder,
            bs=int(b),
            use_greedy=False,
            device=str(device),
        )
    return float(makespans.min().item())


def _jssp_loss(model: Any, instance: dict[str, Any], *, device: torch.device, seed: int) -> tuple[torch.Tensor, float]:
    _seed_everything(seed, device)
    model.train()
    loss, _, _, pair_count = model._training_rollout([instance])
    return loss, float(pair_count.detach().item())


def _run_jssp(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from rl4co.envs import JSSPEnv
    from rl4co.models.zoo.mgl_jssp.data import load_instance
    from rl4co.models.zoo.mgl_jssp.model import MGLJSSPModel

    shape = tuple(int(x) for x in args.problem.removeprefix("jssp").split("x"))
    files = sorted(args.data.glob("*.jsp"))
    if len(files) < args.instances:
        raise ValueError(f"only {len(files)} JSSP files found, need {args.instances}")
    selected = files[: args.instances]
    for path in selected:
        instance = load_instance(path.as_posix(), device="cpu")
        got = (int(instance["j"]), int(instance["m"]))
        if got != shape:
            raise ValueError(f"mixed/unsupported JSSP shape at {path}: expected {shape}, got {got}")

    env = JSSPEnv(generator_params={"num_jobs": shape[0], "num_machines": shape[1]})
    model = MGLJSSPModel.load_from_checkpoint(args.checkpoint.as_posix(), env=env, map_location=device)
    model = model.to(device)
    model.B = int(args.rollouts)
    model.D = 1
    model.use_greedy = False
    _configure_pair(model, args.pair_json)
    baseline = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(selected):
        _restore(model, baseline)
        instance = load_instance(path.as_posix(), device="cpu")
        eval_seed = int(args.eval_seed + index)
        train_seed = int(args.train_seed + index)
        before_cost = _jssp_eval(model, instance, device=device, b=args.rollouts, seed=eval_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
        optimizer.zero_grad(set_to_none=True)
        loss_before, pair_count_before = _jssp_loss(model, instance, device=device, seed=train_seed)
        if not torch.isfinite(loss_before).all() or pair_count_before <= 0:
            raise RuntimeError(f"invalid loss/pair count for instance {index}")
        loss_before.backward()
        grad_norm, grad_nonzero = _finite_grad_norm(model)
        if grad_norm <= 0 or grad_nonzero <= 0:
            raise RuntimeError(f"zero gradient for instance {index}")
        optimizer.step()
        with torch.no_grad():
            loss_after, pair_count_after = _jssp_loss(model, instance, device=device, seed=train_seed)
        after_cost = _jssp_eval(model, instance, device=device, b=args.rollouts, seed=eval_seed)
        row = {
            "instance_index": index,
            "instance_id": path.name,
            "shape": f"{shape[0]}x{shape[1]}",
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "preference_loss_before": float(loss_before.detach().item()),
            "preference_loss_after": float(loss_after.detach().item()),
            "preference_loss_reduction": float(loss_before.detach().item() - loss_after.detach().item()),
            "pair_count_before": pair_count_before,
            "pair_count_after": pair_count_after,
            "grad_norm": grad_norm,
            "grad_nonzero": grad_nonzero,
            "makespan_before": before_cost,
            "makespan_after": after_cost,
            "makespan_improvement": before_cost - after_cost,
        }
        rows.append(row)
        print(f"[instance] {index + 1}/{len(selected)} cost={before_cost}->{after_cost} loss={row['preference_loss_before']:.6g}->{row['preference_loss_after']:.6g}", flush=True)

    protocol = {
        "data_manifest_sha256": _directory_manifest_sha256(args.data, selected),
        "selected_files": [path.name for path in selected],
        "rollouts_per_instance": int(args.rollouts),
        "augmentation_factor": 1,
        "greedy": False,
        "physical_batch_size": 1,
        "same_shape_only": True,
    }
    return rows, protocol


def _ffsp_eval(model: Any, batch: Any, *, device: torch.device, augment_factor: int, seed: int) -> float:
    from rl4co.utils.ops import batchify, unbatchify

    _seed_everything(seed, device)
    model.eval()
    best = None
    with torch.inference_mode():
        for offset in range(0, int(augment_factor), int(augment_factor)):
            chunk = min(int(augment_factor), int(augment_factor) - offset)
            augmented = batchify(batch, chunk)
            td = model.env.reset(augmented)
            num_starts = int(model.num_starts or model.env.get_num_starts(td))
            out = model.policy(td.clone(), model.env, phase="test", num_starts=num_starts, return_actions=False)
            reward_starts = unbatchify(out["reward"], (0, num_starts))
            reward_aug = unbatchify(reward_starts, chunk)
            chunk_best = reward_aug.amax(dim=(-1, -2))
            best = chunk_best if best is None else torch.maximum(best, chunk_best)
    if best is None or best.numel() != 1:
        raise RuntimeError("FFSP evaluation did not produce one instance cost")
    return float(-best.item())


def _ffsp_loss(model: Any, batch: Any, *, device: torch.device, seed: int) -> tuple[torch.Tensor, float]:
    _seed_everything(seed, device)
    model.train()
    if "free_loss_pair_count" not in model.train_metrics:
        model.train_metrics = [*model.train_metrics, "free_loss_pair_count"]
    out = model.shared_step(batch, 0, "train")
    loss = out["loss"]
    pair_count = out.get("free_loss_pair_count")
    if pair_count is None:
        pair_count = out.get("train/free_loss_pair_count")
    if pair_count is None:
        pair_count = out.get("pair_count", torch.tensor(float("nan"), device=device))
    if isinstance(pair_count, torch.Tensor):
        pair_count_value = float(pair_count.detach().float().mean().item())
    else:
        pair_count_value = float(pair_count)
    return loss, pair_count_value


def _run_ffsp(args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.eval_downloaded_routing_checkpoints import DownloadEntry, build_model

    entry = DownloadEntry(
        problem_key=args.problem,
        method=args.method,
        checkpoint_path=args.checkpoint,
        source_checkpoint=None,
        moved_metadata=(),
    )
    model, _, _ = build_model(entry, REPO_ROOT)
    model.env.test_file = args.data.as_posix()
    with np.load(args.data) as payload:
        data_count = int(payload["run_time"].shape[0])
    if data_count < args.instances:
        raise ValueError(f"only {data_count} FFSP instances found, need {args.instances}")
    model.data_cfg["test_data_size"] = data_count
    model.data_cfg["test_batch_size"] = min(max(args.instances, 1), data_count)
    model = model.to(device)
    _configure_pair(model, args.pair_json)
    model.setup(stage="test")
    selected: list[Any] = []
    for batch in model.test_dataloader():
        batch = batch.to(device)
        for index in range(int(batch.batch_size[0])):
            selected.append(batch[index : index + 1].clone())
            if len(selected) >= args.instances:
                break
        if len(selected) >= args.instances:
            break
    if len(selected) != args.instances:
        raise RuntimeError(f"loaded {len(selected)} FFSP instances, expected {args.instances}")
    baseline = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    rows: list[dict[str, Any]] = []
    for index, batch in enumerate(selected):
        _restore(model, baseline)
        eval_seed = int(args.eval_seed + index)
        train_seed = int(args.train_seed + index)
        before_cost = _ffsp_eval(model, batch, device=device, augment_factor=args.augment_factor, seed=eval_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
        optimizer.zero_grad(set_to_none=True)
        loss_before, pair_count_before = _ffsp_loss(model, batch, device=device, seed=train_seed)
        if not torch.isfinite(loss_before).all() or not math.isfinite(pair_count_before) or pair_count_before <= 0:
            raise RuntimeError(f"invalid loss/pair count for instance {index}")
        loss_before.backward()
        grad_norm, grad_nonzero = _finite_grad_norm(model)
        if grad_norm <= 0 or grad_nonzero <= 0:
            raise RuntimeError(f"zero gradient for instance {index}")
        optimizer.step()
        with torch.no_grad():
            loss_after, pair_count_after = _ffsp_loss(model, batch, device=device, seed=train_seed)
        after_cost = _ffsp_eval(model, batch, device=device, augment_factor=args.augment_factor, seed=eval_seed)
        row = {
            "instance_index": index,
            "instance_id": f"{args.problem}_{index:05d}",
            "shape": args.problem,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "preference_loss_before": float(loss_before.detach().item()),
            "preference_loss_after": float(loss_after.detach().item()),
            "preference_loss_reduction": float(loss_before.detach().item() - loss_after.detach().item()),
            "pair_count_before": pair_count_before,
            "pair_count_after": pair_count_after,
            "grad_norm": grad_norm,
            "grad_nonzero": grad_nonzero,
            "makespan_before": before_cost,
            "makespan_after": after_cost,
            "makespan_improvement": before_cost - after_cost,
        }
        rows.append(row)
        print(f"[instance] {index + 1}/{len(selected)} cost={before_cost}->{after_cost} loss={row['preference_loss_before']:.6g}->{row['preference_loss_after']:.6g}", flush=True)

    protocol = {
        "data_file_sha256": _sha256(args.data),
        "data_count": data_count,
        "selected_instance_indices": list(range(args.instances)),
        "num_starts": int(model.num_starts or 0),
        "augmentation_factor": int(args.augment_factor),
        "physical_batch_size": 1,
    }
    return rows, protocol


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr, wilcoxon

    loss_reduction = np.asarray([row["preference_loss_reduction"] for row in rows], dtype=np.float64)
    improvement = np.asarray([row["makespan_improvement"] for row in rows], dtype=np.float64)
    before = np.asarray([row["makespan_before"] for row in rows], dtype=np.float64)
    after = np.asarray([row["makespan_after"] for row in rows], dtype=np.float64)
    if np.allclose(loss_reduction, loss_reduction[0]) or np.allclose(improvement, improvement[0]):
        spear_r, spear_p = float("nan"), float("nan")
        pear_r, pear_p = float("nan"), float("nan")
    else:
        spear = spearmanr(loss_reduction, improvement)
        pear = pearsonr(loss_reduction, improvement)
        spear_r, spear_p = float(spear.statistic), float(spear.pvalue)
        pear_r, pear_p = float(pear.statistic), float(pear.pvalue)
    if np.allclose(before, after):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(after, before, alternative="two-sided", zero_method="wilcox").pvalue)
    wins = int(np.sum(improvement > 0))
    ties = int(np.sum(improvement == 0))
    losses = int(np.sum(improvement < 0))
    if float(np.median(improvement)) > 0 and wins > losses and wilcoxon_p < 0.05 and spear_r > 0 and spear_p < 0.05:
        verdict = "objective_aligned_diagnostic"
    elif (float(np.median(improvement)) < 0 and losses > wins and wilcoxon_p < 0.05) or (spear_r < 0 and spear_p < 0.05):
        verdict = "objective_mismatch_diagnostic"
    else:
        verdict = "inconclusive_alignment_diagnostic"
    return {
        "instances": len(rows),
        "mean_makespan_before": float(np.mean(before)),
        "mean_makespan_after": float(np.mean(after)),
        "mean_makespan_improvement": float(np.mean(improvement)),
        "median_makespan_improvement": float(np.median(improvement)),
        "wins_ties_losses": [wins, ties, losses],
        "wilcoxon_pre_post_two_sided_p": wilcoxon_p,
        "loss_reduction_vs_makespan_improvement_spearman_rho": spear_r,
        "loss_reduction_vs_makespan_improvement_spearman_p": spear_p,
        "loss_reduction_vs_makespan_improvement_pearson_r": pear_r,
        "loss_reduction_vs_makespan_improvement_pearson_p": pear_p,
        "loss_decreased_instances": int(np.sum(loss_reduction > 0)),
        "finite_gradients_all": bool(all(math.isfinite(float(row["grad_norm"])) and float(row["grad_norm"]) > 0 for row in rows)),
        "nonzero_pairs_all": bool(all(float(row["pair_count_before"]) > 0 for row in rows)),
        "verdict": verdict,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded per-instance preference-update/makespan alignment audit")
    parser.add_argument("--problem", required=True, choices=["jssp10x10", "jssp15x15", "ffsp50", "ffsp100"])
    parser.add_argument("--method", required=True, choices=["loss_only", "weighting"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-json", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--train-seed", type=int, default=31000000)
    parser.add_argument("--eval-seed", type=int, default=41000000)
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--augment-factor", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.pair_json = args.pair_json.resolve()
    args.data = args.data.resolve()
    args.output_dir = args.output_dir.resolve()
    for path in (args.checkpoint, args.pair_json, args.data):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"collision-new output required: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.time()
    checkpoint_meta = _checkpoint_metadata(args.checkpoint)
    rows, protocol = (_run_jssp(args, device) if args.problem.startswith("jssp") else _run_ffsp(args, device))
    if not rows:
        raise RuntimeError("audit produced no rows")
    per_instance = args.output_dir / "per_instance.csv"
    _write_csv(per_instance, rows)
    summary = {
        "evidence_class": "test_trained_same_set_evaluated_leakage_diagnostic_objective_alignment_audit",
        "inferential_limit": "One-step in-sample counterfactual diagnostic only; not paper identity, confirmatory significance, or generalization evidence.",
        "problem": args.problem,
        "method": args.method,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_metadata": checkpoint_meta,
        "pair_json": args.pair_json.as_posix(),
        "pair_json_sha256": _sha256(args.pair_json),
        "optimizer": "fresh_torch_adam_single_step_per_instance_reset_to_checkpoint",
        "learning_rate": float(args.learning_rate),
        "train_seed_base": int(args.train_seed),
        "eval_seed_base": int(args.eval_seed),
        "protocol": protocol,
        "statistics": _stats(rows),
        "per_instance_csv": per_instance.as_posix(),
        "per_instance_csv_sha256": _sha256(per_instance),
        "elapsed_sec": time.time() - started,
        "errors": [],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] output={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
