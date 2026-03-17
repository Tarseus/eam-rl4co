from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from hashlib import sha1
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import yaml


def _ensure_paths() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ptp_root = os.path.join(repo_root, "PTP")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if ptp_root not in sys.path:
        sys.path.insert(0, ptp_root)


def _repo_root_dir() -> str:
    # This file lives at scripts/eval_baseline_minitrain.py.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_from_repo_root(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_repo_root_dir(), path))


def _file_sha1(path: str, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(int(chunk_size))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _mean(xs: Sequence[float]) -> float:
    values = [float(v) for v in xs]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _objective_to_reward(obj: float, *, objective_sign: str) -> float:
    sign = str(objective_sign or "neg_reward").strip().lower()
    if sign == "neg_reward":
        return -float(obj)
    return float(obj)


def _baseline_minitrain_eval_mode(cfg_yaml: Mapping[str, Any]) -> str:
    env_name = str(cfg_yaml.get("env_name") or cfg_yaml.get("problem") or "tsp").strip().lower()
    if env_name == "cvrp":
        return "native_po_loss"
    return "ref_free_loss"


@torch.no_grad()
def _pre_minitrain_eval(
    *,
    cfg_yaml: Mapping[str, Any],
    init_checkpoint: str | None,
    train_problem_size: int,
    valid_problem_sizes: Sequence[int],
    num_validation_episodes: int,
    train_batch_size: int,
    scratch_init_seed: int,
    offline_train: str,
    offline_val_by_size: Mapping[int, str],
) -> Tuple[Dict[int, float], float]:
    from fitness.free_loss_fidelity import (
        _evaluate_rl4co_model,
        _load_policy_weights_from_checkpoint,
        _rl4co_build_env,
        _rl4co_build_policy,
    )
    from fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed

    generator_params = dict(cfg_yaml.get("generator_params", {}) or {})
    generator_params["offline_train_path"] = str(offline_train)
    generator_params["offline_val_paths"] = {str(int(k)): str(v) for k, v in offline_val_by_size.items()}

    hf_cfg = HighFidelityConfig(
        problem=str(cfg_yaml.get("problem", "tsp")),
        backend=str(cfg_yaml.get("backend", "rl4co") or "rl4co"),
        env_name=str(cfg_yaml.get("env_name") or cfg_yaml.get("problem", "tsp")),
        env_kwargs=dict(cfg_yaml.get("env_kwargs", {}) or {}),
        generator_params=generator_params,
        policy_name=str(cfg_yaml.get("policy_name", "") or ""),
        policy_kwargs=dict(cfg_yaml.get("policy_kwargs", {}) or {}),
        rollout_strategy=str(cfg_yaml.get("rollout_strategy", "auto") or "auto"),
        objective_sign=str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward"),
        hf_steps=1,
        hf_epochs=0,
        hf_instances_per_epoch=0,
        train_problem_size=int(train_problem_size),
        valid_problem_sizes=tuple(int(x) for x in valid_problem_sizes),
        train_batch_size=int(train_batch_size),
        pomo_size=(int(cfg_yaml.get("pomo_size")) if cfg_yaml.get("pomo_size", None) is not None else None),
        learning_rate=float(cfg_yaml.get("learning_rate", 3e-4) or 3e-4),
        weight_decay=float(cfg_yaml.get("weight_decay", 1e-6) or 1e-6),
        alpha=float(cfg_yaml.get("alpha", 0.05) or 0.05),
        device=str(cfg_yaml.get("device", "cuda") or "cuda"),
        seed=int(scratch_init_seed),
        num_validation_episodes=int(num_validation_episodes),
        validation_batch_size=int(cfg_yaml.get("validation_batch_size", 64) or 64),
        generalization_penalty_weight=float(cfg_yaml.get("generalization_penalty_weight", 1.0) or 1.0),
        size_aggregation=str(cfg_yaml.get("size_aggregation", "mean") or "mean"),
        size_cvar_alpha=float(cfg_yaml.get("size_cvar_alpha", 0.2) or 0.2),
        pool_version=str(cfg_yaml.get("pool_version", "v0") or "v0"),
    )

    _set_seed(int(hf_cfg.seed))
    device_str = str(hf_cfg.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    env = _rl4co_build_env(hf_cfg, int(train_problem_size)).to(device)
    policy, rollout_strategy = _rl4co_build_policy(hf_cfg, env)
    if init_checkpoint:
        _load_policy_weights_from_checkpoint(policy, _abs_from_repo_root(str(init_checkpoint)))
    policy = policy.to(device)
    policy.eval()

    by_size: Dict[int, float] = {}
    for sz in valid_problem_sizes:
        obj = _evaluate_rl4co_model(
            policy=policy,
            cfg=hf_cfg,
            problem_size=int(sz),
            device=device,
            num_episodes=int(num_validation_episodes),
            batch_size=int(hf_cfg.validation_batch_size),
            rollout_strategy=str(rollout_strategy),
        )
        by_size[int(sz)] = float(obj)

    aggregated = _mean([by_size[int(sz)] for sz in valid_problem_sizes])
    return by_size, float(aggregated)


class _CompiledBuilderAdapter:
    def __init__(self, compiled_builder) -> None:
        self._compiled = compiled_builder

    def build(self, feature_cache: Mapping[str, torch.Tensor], *, meta: Mapping[str, Any] | None = None):
        extra = dict(meta or {})
        return self._compiled.build_fn(feature_cache, extra)


def _parse_offline_val_map(args: argparse.Namespace) -> Dict[int, str]:
    out: Dict[int, str] = {}
    # Preferred: repeated --offline_val <size> <path>
    if isinstance(getattr(args, "offline_val", None), list):
        for pair in args.offline_val:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            size_s, path = pair
            out[int(size_s)] = str(path)

    # Compatibility: accept --offline_val_<size> <path> via parse_known_args().
    if isinstance(getattr(args, "_unknown_offline_val", None), list):
        out.update(dict(args._unknown_offline_val))

    return out


def _load_cfg(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config (expected dict): {config_path}")
    return dict(cfg)


def _build_eval_signature(
    *,
    cfg_yaml: Mapping[str, Any],
    K: int,
    train_problem_size: int,
    valid_problem_sizes: Sequence[int],
    num_validation_episodes: int,
    train_batch_size: int,
    scratch_init_seed: int,
    offline_train: str,
    offline_val_by_size: Mapping[int, str],
    ckpt_135: str,
    ckpt_409: str,
) -> Dict[str, Any]:
    env_name = str(cfg_yaml.get("env_name") or cfg_yaml.get("problem") or "tsp")
    baseline_eval_mode = _baseline_minitrain_eval_mode(cfg_yaml)
    policy_name = str(cfg_yaml.get("policy_name") or "")
    policy_kwargs = dict(cfg_yaml.get("policy_kwargs", {}) or {})
    env_kwargs = dict(cfg_yaml.get("env_kwargs", {}) or {})
    rollout_strategy = str(cfg_yaml.get("rollout_strategy", "auto") or "auto")
    objective_sign = str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward")

    pomo_size = cfg_yaml.get("pomo_size", 64)
    pomo_size_out = int(pomo_size) if pomo_size is not None else None

    validation_batch_size = int(cfg_yaml.get("validation_batch_size", 64) or 64)
    alpha = float(cfg_yaml.get("alpha", 0.05) or 0.05)
    lr = float(cfg_yaml.get("learning_rate", 3e-4) or 3e-4)
    wd = float(cfg_yaml.get("weight_decay", 1e-6) or 1e-6)
    size_aggregation = str(cfg_yaml.get("size_aggregation", "mean") or "mean")
    size_cvar_alpha = float(cfg_yaml.get("size_cvar_alpha", 0.2) or 0.2)

    offline_train_abs = _abs_from_repo_root(str(offline_train))
    offline_train_sha1 = _file_sha1(offline_train_abs)
    offline_val_sig: Dict[str, Any] = {}
    for size, path in sorted((int(k), str(v)) for k, v in offline_val_by_size.items()):
        p_abs = _abs_from_repo_root(path)
        offline_val_sig[str(size)] = {"path": str(path), "sha1": _file_sha1(p_abs)}

    ckpt_135_abs = _abs_from_repo_root(str(ckpt_135))
    ckpt_409_abs = _abs_from_repo_root(str(ckpt_409))

    return {
        "protocol": "stage3_offline_minitrain_v1",
        "baseline_eval_mode": str(baseline_eval_mode),
        "env_name": env_name,
        "policy_name": policy_name,
        "policy_kwargs": policy_kwargs,
        "env_kwargs": env_kwargs,
        "rollout_strategy": rollout_strategy,
        "objective_sign": objective_sign,
        "alpha": alpha,
        "K": int(K),
        "train_problem_size": int(train_problem_size),
        "valid_problem_sizes": [int(x) for x in valid_problem_sizes],
        "train_batch_size": int(train_batch_size),
        "num_validation_episodes": int(num_validation_episodes),
        "validation_batch_size": int(validation_batch_size),
        "pomo_size": pomo_size_out,
        "learning_rate": lr,
        "weight_decay": wd,
        "size_aggregation": size_aggregation,
        "size_cvar_alpha": size_cvar_alpha,
        "scratch_init_seed": int(scratch_init_seed),
        "offline": {
            "train": {"path": str(offline_train), "sha1": offline_train_sha1},
            "val": offline_val_sig,
        },
        "checkpoints": {
            "ckpt_135": {"path": str(ckpt_135), "sha1": _file_sha1(ckpt_135_abs)},
            "ckpt_409": {"path": str(ckpt_409), "sha1": _file_sha1(ckpt_409_abs)},
        },
    }


def _evaluate_one_init_native_po_loss(
    *,
    cfg_yaml: Mapping[str, Any],
    init_checkpoint: str | None,
    K: int,
    train_problem_size: int,
    valid_problem_sizes: Sequence[int],
    num_validation_episodes: int,
    train_batch_size: int,
    scratch_init_seed: int,
    offline_train: str,
    offline_val_by_size: Mapping[int, str],
) -> Tuple[Dict[int, float], float]:
    from torch.optim import Adam

    from PTP.fitness.free_loss_fidelity import (
        _evaluate_rl4co_model,
        _load_policy_weights_from_checkpoint,
        _rl4co_build_env,
        _rl4co_build_policy,
        _rl4co_rollout,
    )
    from PTP.fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed, resolve_pomo_size
    from rl4co.models.rl.reinforce.preference_losses import po_loss

    generator_params = dict(cfg_yaml.get("generator_params", {}) or {})
    generator_params["offline_train_path"] = str(offline_train)
    generator_params["offline_val_paths"] = {str(int(k)): str(v) for k, v in offline_val_by_size.items()}

    hf_cfg = HighFidelityConfig(
        problem=str(cfg_yaml.get("problem", "cvrp")),
        backend=str(cfg_yaml.get("backend", "rl4co") or "rl4co"),
        env_name=str(cfg_yaml.get("env_name") or cfg_yaml.get("problem", "cvrp")),
        env_kwargs=dict(cfg_yaml.get("env_kwargs", {}) or {}),
        generator_params=generator_params,
        policy_name=str(cfg_yaml.get("policy_name", "") or ""),
        policy_kwargs=dict(cfg_yaml.get("policy_kwargs", {}) or {}),
        rollout_strategy=str(cfg_yaml.get("rollout_strategy", "auto") or "auto"),
        objective_sign=str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward"),
        hf_steps=int(K),
        hf_epochs=0,
        hf_instances_per_epoch=0,
        train_problem_size=int(train_problem_size),
        valid_problem_sizes=tuple(int(x) for x in valid_problem_sizes),
        train_batch_size=int(train_batch_size),
        pomo_size=(int(cfg_yaml.get("pomo_size")) if cfg_yaml.get("pomo_size", None) is not None else None),
        learning_rate=float(cfg_yaml.get("learning_rate", 3e-4) or 3e-4),
        weight_decay=float(cfg_yaml.get("weight_decay", 1e-6) or 1e-6),
        alpha=float(cfg_yaml.get("alpha", 0.05) or 0.05),
        device=str(cfg_yaml.get("device", "cuda") or "cuda"),
        seed=int(scratch_init_seed),
        num_validation_episodes=int(num_validation_episodes),
        validation_batch_size=int(cfg_yaml.get("validation_batch_size", 64) or 64),
        generalization_penalty_weight=float(cfg_yaml.get("generalization_penalty_weight", 1.0) or 1.0),
        size_aggregation=str(cfg_yaml.get("size_aggregation", "mean") or "mean"),
        size_cvar_alpha=float(cfg_yaml.get("size_cvar_alpha", 0.2) or 0.2),
        pool_version=str(cfg_yaml.get("pool_version", "v0") or "v0"),
    )

    _set_seed(int(hf_cfg.seed))
    device_str = str(hf_cfg.device)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    env = _rl4co_build_env(hf_cfg, int(train_problem_size)).to(device)
    policy, rollout_strategy = _rl4co_build_policy(hf_cfg, env)
    if init_checkpoint:
        _load_policy_weights_from_checkpoint(policy, _abs_from_repo_root(str(init_checkpoint)))
    policy = policy.to(device)
    optimizer = Adam(
        policy.parameters(),
        lr=float(hf_cfg.learning_rate),
        weight_decay=float(hf_cfg.weight_decay),
    )

    num_rollouts = resolve_pomo_size(hf_cfg.pomo_size, hf_cfg.train_problem_size)
    steps = max(int(K), 0)
    for _ in range(steps):
        policy.train()
        reward, log_likelihood = _rl4co_rollout(
            env,
            policy,
            hf_cfg.train_batch_size,
            num_rollouts,
            phase="train",
            rollout_strategy=str(rollout_strategy),
            device=device,
        )
        loss, _ = po_loss(reward, log_likelihood, alpha=float(hf_cfg.alpha))
        if not torch.isfinite(loss).all():
            raise RuntimeError("Non-finite po_loss encountered during CVRP baseline mini-train")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    by_size: Dict[int, float] = {}
    for sz in valid_problem_sizes:
        obj = _evaluate_rl4co_model(
            policy=policy,
            cfg=hf_cfg,
            problem_size=int(sz),
            device=device,
            num_episodes=int(num_validation_episodes),
            batch_size=int(hf_cfg.validation_batch_size),
            rollout_strategy=str(rollout_strategy),
        )
        by_size[int(sz)] = float(obj)

    aggregated = _mean([by_size[int(sz)] for sz in valid_problem_sizes])
    return by_size, float(aggregated)


def _evaluate_one_init(
    *,
    cfg_yaml: Mapping[str, Any],
    compiled_builder,
    compiled_loss,
    init_checkpoint: str | None,
    K: int,
    train_problem_size: int,
    valid_problem_sizes: Sequence[int],
    num_validation_episodes: int,
    train_batch_size: int,
    scratch_init_seed: int,
    offline_train: str,
    offline_val_by_size: Mapping[int, str],
) -> Tuple[Dict[int, float], float]:
    eval_mode = _baseline_minitrain_eval_mode(cfg_yaml)
    if str(eval_mode) == "native_po_loss":
        return _evaluate_one_init_native_po_loss(
            cfg_yaml=cfg_yaml,
            init_checkpoint=init_checkpoint,
            K=K,
            train_problem_size=train_problem_size,
            valid_problem_sizes=valid_problem_sizes,
            num_validation_episodes=num_validation_episodes,
            train_batch_size=train_batch_size,
            scratch_init_seed=scratch_init_seed,
            offline_train=offline_train,
            offline_val_by_size=offline_val_by_size,
        )

    from fitness.free_loss_fidelity import FreeLossFidelityConfig, evaluate_free_loss_candidate
    from fitness.ptp_high_fidelity import HighFidelityConfig

    generator_params = dict(cfg_yaml.get("generator_params", {}) or {})
    generator_params["offline_train_path"] = str(offline_train)
    generator_params["offline_val_paths"] = {str(int(k)): str(v) for k, v in offline_val_by_size.items()}

    hf_cfg = HighFidelityConfig(
        problem=str(cfg_yaml.get("problem", "tsp")),
        backend=str(cfg_yaml.get("backend", "rl4co") or "rl4co"),
        env_name=str(cfg_yaml.get("env_name") or cfg_yaml.get("problem", "tsp")),
        env_kwargs=dict(cfg_yaml.get("env_kwargs", {}) or {}),
        generator_params=generator_params,
        policy_name=str(cfg_yaml.get("policy_name", "") or ""),
        policy_kwargs=dict(cfg_yaml.get("policy_kwargs", {}) or {}),
        rollout_strategy=str(cfg_yaml.get("rollout_strategy", "auto") or "auto"),
        objective_sign=str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward"),
        hf_steps=int(K),
        hf_epochs=0,
        hf_instances_per_epoch=0,
        train_problem_size=int(train_problem_size),
        valid_problem_sizes=tuple(int(x) for x in valid_problem_sizes),
        train_batch_size=int(train_batch_size),
        pomo_size=(int(cfg_yaml.get("pomo_size")) if cfg_yaml.get("pomo_size", None) is not None else None),
        learning_rate=float(cfg_yaml.get("learning_rate", 3e-4) or 3e-4),
        weight_decay=float(cfg_yaml.get("weight_decay", 1e-6) or 1e-6),
        alpha=float(cfg_yaml.get("alpha", 0.05) or 0.05),
        device=str(cfg_yaml.get("device", "cuda") or "cuda"),
        seed=int(scratch_init_seed),
        num_validation_episodes=int(num_validation_episodes),
        validation_batch_size=int(cfg_yaml.get("validation_batch_size", 64) or 64),
        generalization_penalty_weight=float(cfg_yaml.get("generalization_penalty_weight", 1.0) or 1.0),
        size_aggregation=str(cfg_yaml.get("size_aggregation", "mean") or "mean"),
        size_cvar_alpha=float(cfg_yaml.get("size_cvar_alpha", 0.2) or 0.2),
        pool_version=str(cfg_yaml.get("pool_version", "v0") or "v0"),
    )

    free_cfg = FreeLossFidelityConfig(
        hf=hf_cfg,
        f1_steps=int(K),
        f2_steps=0,
        f3_enabled=False,
        init_checkpoint_path=_abs_from_repo_root(str(init_checkpoint)) if init_checkpoint else None,
        init_checkpoint_epoch=None,
    )
    adapter = _CompiledBuilderAdapter(compiled_builder)
    fitness = evaluate_free_loss_candidate(compiled_loss, free_cfg, pref_builder=adapter)

    size_objectives_raw = fitness.get("size_objectives", {})
    size_objectives: Dict[int, float] = {}
    if isinstance(size_objectives_raw, dict):
        for k, v in size_objectives_raw.items():
            try:
                size_objectives[int(k)] = float(v)
            except Exception:  # noqa: BLE001
                continue

    by_size: Dict[int, float] = {}
    for sz in valid_problem_sizes:
        if int(sz) not in size_objectives:
            raise RuntimeError(f"Missing size_objectives[{int(sz)}] in fitness output")
        by_size[int(sz)] = float(size_objectives[int(sz)])
    aggregated = _mean([by_size[int(sz)] for sz in valid_problem_sizes])
    return by_size, float(aggregated)


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline mini-train (offline) and write JSON")
    p.add_argument("--config", required=True, type=str, help="Experiment YAML (used for env/policy/hparams)")
    p.add_argument("--K", required=True, type=int, help="Mini-train step budget (steps)")
    p.add_argument("--train_problem_size", required=True, type=int)
    p.add_argument("--valid_problem_sizes", required=True, nargs="+", type=int)
    p.add_argument("--num_validation_episodes", required=True, type=int)
    p.add_argument("--train_batch_size", required=True, type=int)
    p.add_argument("--offline_train", required=True, type=str)
    p.add_argument("--offline_val", action="append", nargs=2, metavar=("SIZE", "PATH"), default=[])
    p.add_argument("--scratch_init_seed", required=True, type=int)
    p.add_argument("--ckpt_135", required=True, type=str)
    p.add_argument("--ckpt_409", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    return p.parse_args(argv)


def _parse_known_offline_val_compat(argv: Sequence[str]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if isinstance(tok, str) and tok.startswith("--offline_val_"):
            size_s = tok[len("--offline_val_") :]
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for {tok}")
            path = argv[i + 1]
            out.append((int(size_s), str(path)))
            i += 2
            continue
        i += 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_paths()
    if argv is None:
        argv = sys.argv[1:]

    # Two-pass parse: keep compatibility with --offline_val_<size> flags.
    compat_vals = _parse_known_offline_val_compat(list(argv))
    filtered_argv: List[str] = []
    skip_next = False
    for i, tok in enumerate(list(argv)):
        if skip_next:
            skip_next = False
            continue
        if isinstance(tok, str) and tok.startswith("--offline_val_"):
            skip_next = True
            continue
        filtered_argv.append(tok)

    args = _parse_args(filtered_argv)
    args._unknown_offline_val = compat_vals  # type: ignore[attr-defined]

    cfg_yaml = _load_cfg(str(args.config))
    offline_val_by_size = _parse_offline_val_map(args)
    if not offline_val_by_size:
        raise ValueError("No offline val paths provided (use --offline_val <size> <path> or --offline_val_<size>)")

    for p in [str(args.offline_train)] + [str(v) for v in offline_val_by_size.values()]:
        p_abs = _abs_from_repo_root(p)
        if not os.path.exists(p_abs):
            raise FileNotFoundError(f"Offline data file not found: {p} (abs={p_abs})")
    for p in [str(args.ckpt_135), str(args.ckpt_409)]:
        p_abs = _abs_from_repo_root(p)
        if not os.path.exists(p_abs):
            raise FileNotFoundError(f"Checkpoint not found: {p} (abs={p_abs})")

    # Compile reference builder/loss.
    from ptp_discovery.free_loss_compiler import compile_free_loss
    from ptp_discovery.free_loss_gates import run_static_gates
    from ptp_discovery.pref_builder_compiler import compile_preference_builder
    from ptp_discovery.pref_loss_coevo_loop import _ref_builder_ir, _ref_loss_ir

    compiled_builder = compile_preference_builder(_ref_builder_ir(), operator_whitelist=[])
    ref_loss_ir = _ref_loss_ir()
    static_ref = run_static_gates(ref_loss_ir, operator_whitelist=[])
    if not static_ref.ok:
        raise RuntimeError(f"Reference loss failed static gates: {static_ref.reason}")
    compiled_loss = compile_free_loss(ref_loss_ir, operator_whitelist=[])

    valid_sizes = [int(x) for x in args.valid_problem_sizes]
    signature = _build_eval_signature(
        cfg_yaml=cfg_yaml,
        K=int(args.K),
        train_problem_size=int(args.train_problem_size),
        valid_problem_sizes=valid_sizes,
        num_validation_episodes=int(args.num_validation_episodes),
        train_batch_size=int(args.train_batch_size),
        scratch_init_seed=int(args.scratch_init_seed),
        offline_train=str(args.offline_train),
        offline_val_by_size=offline_val_by_size,
        ckpt_135=str(args.ckpt_135),
        ckpt_409=str(args.ckpt_409),
    )

    per_init: Dict[str, Any] = {}
    for name, ckpt in [
        ("scratch", None),
        ("ckpt_135", str(args.ckpt_135)),
        ("ckpt_409", str(args.ckpt_409)),
    ]:
        t0 = time.time()
        objective_sign = str(cfg_yaml.get("objective_sign", "neg_reward") or "neg_reward")

        t_pre0 = time.time()
        pre_by_size, pre_agg = _pre_minitrain_eval(
            cfg_yaml=cfg_yaml,
            init_checkpoint=ckpt,
            train_problem_size=int(args.train_problem_size),
            valid_problem_sizes=valid_sizes,
            num_validation_episodes=int(args.num_validation_episodes),
            train_batch_size=int(args.train_batch_size),
            scratch_init_seed=int(args.scratch_init_seed),
            offline_train=str(args.offline_train),
            offline_val_by_size=offline_val_by_size,
        )
        pre_elapsed_s = float(time.time() - t_pre0)

        t_post0 = time.time()
        by_size, agg = _evaluate_one_init(
            cfg_yaml=cfg_yaml,
            compiled_builder=compiled_builder,
            compiled_loss=compiled_loss,
            init_checkpoint=ckpt,
            K=int(args.K),
            train_problem_size=int(args.train_problem_size),
            valid_problem_sizes=valid_sizes,
            num_validation_episodes=int(args.num_validation_episodes),
            train_batch_size=int(args.train_batch_size),
            scratch_init_seed=int(args.scratch_init_seed),
            offline_train=str(args.offline_train),
            offline_val_by_size=offline_val_by_size,
        )
        post_elapsed_s = float(time.time() - t_post0)
        per_init[name] = {
            "pre_val_objective_by_size": {str(int(k)): float(v) for k, v in pre_by_size.items()},
            "pre_val_reward_by_size": {
                str(int(k)): float(_objective_to_reward(v, objective_sign=objective_sign))
                for k, v in pre_by_size.items()
            },
            "pre_aggregated_objective": float(pre_agg),
            "pre_aggregated_reward": float(_objective_to_reward(pre_agg, objective_sign=objective_sign)),
            "val_objective_by_size": {str(int(k)): float(v) for k, v in by_size.items()},
            "val_reward_by_size": {
                str(int(k)): float(_objective_to_reward(v, objective_sign=objective_sign))
                for k, v in by_size.items()
            },
            "aggregated_objective": float(agg),
            "aggregated_reward": float(_objective_to_reward(agg, objective_sign=objective_sign)),
            "delta_objective_post_minus_pre": float(float(agg) - float(pre_agg)),
            "delta_reward_post_minus_pre": float(
                _objective_to_reward(agg, objective_sign=objective_sign)
                - _objective_to_reward(pre_agg, objective_sign=objective_sign)
            ),
            "elapsed_s": float(time.time() - t0),
            "elapsed_pre_s": float(pre_elapsed_s),
            "elapsed_post_s": float(post_elapsed_s),
            "init_checkpoint": str(ckpt) if ckpt else None,
        }
        print(
            f"[baseline] init={name} pre_obj={float(pre_agg):.6f} post_obj={float(agg):.6f} "
            f"pre_reward={float(_objective_to_reward(pre_agg, objective_sign=objective_sign)):.6f} "
            f"post_reward={float(_objective_to_reward(agg, objective_sign=objective_sign)):.6f} "
            f"delta_obj={float(agg - pre_agg):+.6f} elapsed_s={float(time.time()-t0):.1f}",
            flush=True,
        )

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(args.config),
        "eval_signature": signature,
        "baseline_eval_mode": str(_baseline_minitrain_eval_mode(cfg_yaml)),
        "per_init": per_init,
    }
    if str(_baseline_minitrain_eval_mode(cfg_yaml)) == "ref_free_loss":
        payload["reference"] = {
            "builder_ir": asdict(_ref_builder_ir()),
            "loss_ir": asdict(_ref_loss_ir()),
        }
    else:
        payload["reference"] = {
            "loss_type": "po_loss",
            "source": "native_rl4co_preference_loss",
        }

    _atomic_write_json(str(args.out), payload)
    print(f"[baseline] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
