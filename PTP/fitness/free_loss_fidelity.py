from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import logging
import torch
from torch.optim import Adam

from .co_features import build_model_output, gather_pairwise_deltas
from .ptp_high_fidelity import (
    HighFidelityConfig,
    _set_seed,
    resolve_pomo_size,
    aggregate_objectives_by_size,
    get_hf_epoch_plan,
    get_total_hf_train_steps,
)
from ptp_discovery.free_loss_compiler import CompiledFreeLoss


logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        value = float(val)
        self.val = value
        self.sum += value * n
        self.count += int(n)
        if self.count > 0:
            self.avg = self.sum / self.count


@dataclass
class FreeLossFidelityConfig:
    hf: HighFidelityConfig
    f1_steps: int = 32
    f2_steps: int = 0
    f3_enabled: bool = False
    baseline_epoch_violation_weight: float = 1.0


def _build_preference_pairs(
    objective: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Vectorized construction of winner/loser indices.

    For each instance in the batch, we consider all pairs (i, j) such that
    objective[i] < objective[j] (i is better than j). This yields three
    index tensors (batch_idx, winner_idx, loser_idx) plus the total pair
    count. We intentionally do not compute structural features here to keep
    the free-loss evaluation lightweight.
    """

    # objective: (batch, pomo)
    # mask[b, i, j] = True if i is better (lower cost) than j for instance b.
    mask = objective[:, :, None] < objective[:, None, :]
    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
    pair_count = int(b_idx.numel())
    return (b_idx, winner_idx, loser_idx), pair_count


def _rl4co_env_name(cfg: HighFidelityConfig) -> str:
    env_name = getattr(cfg, "env_name", "") or getattr(cfg, "problem", "tsp")
    return str(env_name).strip().lower()


def _rl4co_size_key(env_name: str) -> str | None:
    return {
        "tsp": "num_loc",
        "cvrp": "num_loc",
        "jssp": "num_jobs",
        "fjsp": "num_jobs",
        "ffsp": "num_job",
    }.get(env_name)


def _rl4co_policy_name(cfg: HighFidelityConfig, env_name: str) -> str:
    policy_name = str(getattr(cfg, "policy_name", "") or "").strip().lower()
    if policy_name:
        return policy_name
    return {
        "tsp": "pomo",
        "cvrp": "pomo",
        "jssp": "l2d",
        "fjsp": "l2d",
        "ffsp": "matnet",
    }.get(env_name, "pomo")


def _rl4co_rollout_strategy(cfg: HighFidelityConfig, policy_name: str) -> str:
    strategy = str(getattr(cfg, "rollout_strategy", "auto") or "auto").strip().lower()
    if strategy and strategy != "auto":
        return strategy
    if policy_name in {"pomo", "matnet"}:
        return "policy_multistart"
    return "batchify_sampling"


def _rl4co_set_multistart_decode(policy) -> None:
    for phase in ("train", "val", "test"):
        attr = f"{phase}_decode_type"
        val = getattr(policy, attr, None)
        if val is None:
            continue
        if "multistart" in str(val):
            continue
        setattr(policy, attr, f"multistart_{val}")


def _rl4co_build_env(
    cfg: HighFidelityConfig,
    problem_size: int,
):
    from rl4co.envs import CVRPEnv, FJSPEnv, JSSPEnv, TSPEnv
    from rl4co.envs.scheduling.ffsp.env import FFSPEnv

    env_name = _rl4co_env_name(cfg)
    env_kwargs = dict(getattr(cfg, "env_kwargs", {}) or {})
    generator_params = dict(getattr(cfg, "generator_params", {}) or {})
    size_key = _rl4co_size_key(env_name)
    if size_key is not None:
        generator_params[size_key] = int(problem_size)

    env_map = {
        "tsp": TSPEnv,
        "cvrp": CVRPEnv,
        "jssp": JSSPEnv,
        "fjsp": FJSPEnv,
        "ffsp": FFSPEnv,
    }
    if env_name not in env_map:
        raise ValueError(f"Unsupported env_name for RL4CO backend: {env_name}")

    return env_map[env_name](generator_params=generator_params, **env_kwargs)


def _rl4co_build_policy(cfg: HighFidelityConfig, env):
    from rl4co.models.zoo.am import AttentionModelPolicy
    from rl4co.models.zoo.l2d.policy import L2DPolicy
    from rl4co.models.zoo.matnet.model import select_matnet_policy

    env_name = _rl4co_env_name(cfg)
    policy_name = _rl4co_policy_name(cfg, env_name)
    policy_kwargs = dict(getattr(cfg, "policy_kwargs", {}) or {})

    if policy_name in {"pomo", "am"}:
        policy_defaults = {
            "num_encoder_layers": 6,
            "normalization": "instance",
            "use_graph_context": False,
        }
        policy_defaults.update(policy_kwargs)
        policy = AttentionModelPolicy(env_name=env.name, **policy_defaults)
    elif policy_name == "l2d":
        policy_kwargs.setdefault("test_decode_type", "greedy")
        policy = L2DPolicy(env_name=env.name, **policy_kwargs)
    elif policy_name == "matnet":
        policy = select_matnet_policy(env=env, **policy_kwargs)
    else:
        raise ValueError(f"Unsupported policy_name for RL4CO backend: {policy_name}")

    rollout_strategy = _rl4co_rollout_strategy(cfg, policy_name)
    if rollout_strategy == "policy_multistart":
        _rl4co_set_multistart_decode(policy)
    return policy, rollout_strategy


def _rl4co_objective_from_reward(reward: torch.Tensor, cfg: HighFidelityConfig) -> torch.Tensor:
    sign = str(getattr(cfg, "objective_sign", "neg_reward") or "neg_reward").strip().lower()
    if sign == "reward":
        return reward
    return -reward


def _rl4co_rollout(
    env,
    policy,
    batch_size: int,
    num_rollouts: int,
    *,
    phase: str,
    rollout_strategy: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from rl4co.utils.ops import batchify, unbatchify

    batch = env.generator(batch_size)
    batch = batch.to(device)
    td = env.reset(batch)

    if rollout_strategy == "policy_multistart":
        out = policy(td, env, phase=phase, num_starts=num_rollouts)
        reward = unbatchify(out["reward"], num_rollouts)
        log_likelihood = unbatchify(out["log_likelihood"], num_rollouts)
    else:
        td_rep = batchify(td, num_rollouts) if num_rollouts > 1 else td
        out = policy(td_rep, env, phase=phase)
        reward = unbatchify(out["reward"], num_rollouts)
        log_likelihood = unbatchify(out["log_likelihood"], num_rollouts)

    return reward, log_likelihood


def _train_one_batch_with_free_loss_rl4co(
    env,
    policy,
    optimizer: Adam,
    compiled_loss: CompiledFreeLoss,
    hf_cfg: HighFidelityConfig,
    rollout_strategy: str,
    device: torch.device,
) -> Tuple[float, float, int]:
    batch_size = hf_cfg.train_batch_size
    num_rollouts = resolve_pomo_size(hf_cfg.pomo_size, hf_cfg.train_problem_size)

    policy.train()
    reward, log_likelihood = _rl4co_rollout(
        env,
        policy,
        batch_size,
        num_rollouts,
        phase="train",
        rollout_strategy=rollout_strategy,
        device=device,
    )

    objective = _rl4co_objective_from_reward(reward, hf_cfg)
    log_prob = log_likelihood

    model_output, co_features = build_model_output(objective=objective, log_prob=log_prob)
    (b_idx, winner_idx, loser_idx), pair_count = _build_preference_pairs(objective)

    mode = getattr(compiled_loss.ir.implementation_hint, "mode", "pairwise")
    mode = str(mode or "pairwise").strip().lower()

    if mode == "setwise":
        loss = compiled_loss.loss_fn(
            batch={},
            model_output=model_output,
            extra={"alpha": hf_cfg.alpha},
        )
    elif pair_count == 0:
        advantage = reward - reward.mean(dim=1, keepdim=True)
        loss = -(advantage * log_prob).mean()
    else:
        cost_a_tensor = objective[b_idx, winner_idx]
        cost_b_tensor = objective[b_idx, loser_idx]
        logp_w_tensor = log_prob[b_idx, winner_idx]
        logp_l_tensor = log_prob[b_idx, loser_idx]

        pairwise_deltas = gather_pairwise_deltas(
            co_features, b_idx=b_idx, winner_idx=winner_idx, loser_idx=loser_idx
        )
        batch = {
            "cost_a": cost_a_tensor,
            "cost_b": cost_b_tensor,
            "log_prob_w": logp_w_tensor,
            "log_prob_l": logp_l_tensor,
            **pairwise_deltas,
            "weight": torch.ones_like(logp_w_tensor),
        }
        loss = compiled_loss.loss_fn(
            batch=batch,
            model_output=model_output,
            extra={"alpha": hf_cfg.alpha},
        )

    max_reward, _ = reward.max(dim=1)
    score_mean = _rl4co_objective_from_reward(max_reward, hf_cfg).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return score_mean.item(), float(loss.item()), pair_count


@torch.no_grad()
def _evaluate_rl4co_model(
    *,
    policy,
    cfg: HighFidelityConfig,
    problem_size: int,
    device: torch.device,
    num_episodes: int,
    batch_size: int,
    rollout_strategy: str,
) -> float:
    env = _rl4co_build_env(cfg, problem_size)
    env = env.to(device)
    policy.eval()

    num_rollouts = resolve_pomo_size(cfg.pomo_size, problem_size)
    score_meter = AverageMeter()
    episodes_done = 0

    while episodes_done < num_episodes:
        remaining = num_episodes - episodes_done
        current_batch = min(batch_size, remaining)

        reward, _ = _rl4co_rollout(
            env,
            policy,
            current_batch,
            num_rollouts,
            phase="test",
            rollout_strategy=rollout_strategy,
            device=device,
        )
        max_reward, _ = reward.max(dim=1)
        score = _rl4co_objective_from_reward(max_reward, cfg).float().mean().item()
        score_meter.update(score, n=current_batch)
        episodes_done += current_batch

    return float(score_meter.avg)


def _evaluate_free_loss_candidate_rl4co(
    compiled_loss: CompiledFreeLoss,
    cfg: FreeLossFidelityConfig,
    *,
    baseline_early_valid: float | None = None,
    early_eval_steps: int = 0,
    baseline_epoch_objectives: Sequence[float] | None = None,
) -> Dict[str, Any]:
    _set_seed(cfg.hf.seed)

    device_str = cfg.hf.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    env = _rl4co_build_env(cfg.hf, cfg.hf.train_problem_size)
    env = env.to(device)
    policy, rollout_strategy = _rl4co_build_policy(cfg.hf, env)
    policy = policy.to(device)
    optimizer = Adam(
        policy.parameters(),
        lr=float(cfg.hf.learning_rate),
        weight_decay=float(cfg.hf.weight_decay),
    )

    score_meter = AverageMeter()
    loss_meter = AverageMeter()
    total_pairs = 0

    steps_f1 = get_total_hf_train_steps(cfg.hf)
    steps_f2 = max(int(cfg.f2_steps), 0)
    steps = steps_f1 + steps_f2
    steps_per_epoch, epochs_total = get_hf_epoch_plan(cfg.hf)
    epoch_validation_objectives: List[float] = []

    score_meter_f1 = AverageMeter()
    loss_meter_f1 = AverageMeter()
    total_pairs_f1 = 0

    logger.info(
        "RL4CO free-loss training: f1_steps=%d, f2_steps=%d, total_steps=%d, train_problem_size=%d, "
        "rollouts=%d, batch_size=%d, device=%s, env=%s",
        steps_f1,
        steps_f2,
        steps,
        cfg.hf.train_problem_size,
        resolve_pomo_size(cfg.hf.pomo_size, cfg.hf.train_problem_size),
        cfg.hf.train_batch_size,
        str(device),
        _rl4co_env_name(cfg.hf),
    )

    log_interval = max(steps // 10, 1)

    early_eval_steps = max(int(early_eval_steps or 0), 0)
    early_eval_effective = min(early_eval_steps, steps) if early_eval_steps > 0 else 0
    early_validation_objective: float | None = None
    early_stopped = False

    for step in range(steps):
        score, loss, pair_count = _train_one_batch_with_free_loss_rl4co(
            env=env,
            policy=policy,
            optimizer=optimizer,
            compiled_loss=compiled_loss,
            hf_cfg=cfg.hf,
            rollout_strategy=rollout_strategy,
            device=device,
        )
        score_meter.update(score)
        loss_meter.update(loss)
        total_pairs += int(pair_count)

        if step < steps_f1:
            score_meter_f1.update(score)
            loss_meter_f1.update(loss)
            total_pairs_f1 += int(pair_count)

        if (step + 1) % log_interval == 0 or step == 0:
            logger.info(
                "RL4CO free-loss step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f), pairs_step=%d, pairs_total=%d",
                step + 1,
                steps,
                score,
                float(score_meter.avg),
                loss,
                float(loss_meter.avg),
                int(pair_count),
                total_pairs,
            )

        if steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0:
            epoch_idx = (step + 1) // steps_per_epoch
            if epoch_idx <= epochs_total:
                epoch_valid_obj = _evaluate_rl4co_model(
                    policy=policy,
                    cfg=cfg.hf,
                    problem_size=cfg.hf.train_problem_size,
                    device=device,
                    num_episodes=cfg.hf.num_validation_episodes,
                    batch_size=cfg.hf.validation_batch_size,
                    rollout_strategy=rollout_strategy,
                )
                epoch_validation_objectives.append(epoch_valid_obj)
                logger.info(
                    "RL4CO free-loss epoch %d/%d: validation_objective=%.6f",
                    epoch_idx,
                    epochs_total,
                    epoch_valid_obj,
                )

        if early_eval_effective > 0 and (step + 1) == early_eval_effective:
            early_validation_objective = _evaluate_rl4co_model(
                policy=policy,
                cfg=cfg.hf,
                problem_size=cfg.hf.train_problem_size,
                device=device,
                num_episodes=cfg.hf.num_validation_episodes,
                batch_size=cfg.hf.validation_batch_size,
                rollout_strategy=rollout_strategy,
            )
            if (
                baseline_early_valid is not None
                and early_validation_objective > baseline_early_valid
            ):
                early_stopped = True
                logger.info(
                    "RL4CO early stop at step %d: candidate early_valid=%.6f baseline_early=%.6f",
                    step + 1,
                    early_validation_objective,
                    baseline_early_valid,
                )
                break

    if early_stopped and early_validation_objective is not None:
        main_valid_obj = float(early_validation_objective)
    else:
        main_valid_obj = _evaluate_rl4co_model(
            policy=policy,
            cfg=cfg.hf,
            problem_size=cfg.hf.train_problem_size,
            device=device,
            num_episodes=cfg.hf.num_validation_episodes,
            batch_size=cfg.hf.validation_batch_size,
            rollout_strategy=rollout_strategy,
        )

    size_objectives: Dict[int, float] = {int(cfg.hf.train_problem_size): float(main_valid_obj)}
    for size in cfg.hf.valid_problem_sizes:
        size_int = int(size)
        if size_int in size_objectives:
            continue
        size_objectives[size_int] = _evaluate_rl4co_model(
            policy=policy,
            cfg=cfg.hf,
            problem_size=size_int,
            device=device,
            num_episodes=cfg.hf.num_validation_episodes,
            batch_size=cfg.hf.validation_batch_size,
            rollout_strategy=rollout_strategy,
        )
    gen_objectives = {
        k: v for k, v in size_objectives.items() if k != int(cfg.hf.train_problem_size)
    }

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, max_gen_obj - main_valid_obj)

    epoch_objective_mean: float | None = None
    if epoch_validation_objectives:
        epoch_objective_mean = float(
            sum(epoch_validation_objectives) / len(epoch_validation_objectives)
        )

    epoch_baseline_violations: int | None = None
    epoch_better_than_baseline: bool | None = None
    epoch_baseline_margins: List[float] | None = None
    if baseline_epoch_objectives:
        baseline_list = [float(v) for v in baseline_epoch_objectives]
        compare_len = min(len(epoch_validation_objectives), len(baseline_list))
        epoch_baseline_margins = []
        for i in range(compare_len):
            margin = float(epoch_validation_objectives[i]) - baseline_list[i]
            epoch_baseline_margins.append(margin)
        violations = sum(1 for m in epoch_baseline_margins if m > 0.0)
        epoch_baseline_violations = int(violations)
        epoch_better_than_baseline = epoch_baseline_violations == 0

    agg_method = str(cfg.hf.size_aggregation or "legacy").strip().lower()
    base_objective = (
        epoch_objective_mean if epoch_objective_mean is not None else float(main_valid_obj)
    )
    if agg_method == "legacy":
        hf_like_score = base_objective + cfg.hf.generalization_penalty_weight * generalization_penalty
    else:
        hf_like_score = aggregate_objectives_by_size(
            size_objectives,
            method=agg_method,
            cvar_alpha=float(cfg.hf.size_cvar_alpha),
        )
    if epoch_baseline_violations is not None:
        hf_like_score += cfg.baseline_epoch_violation_weight * float(epoch_baseline_violations)

    return {
        "hf_like_score": hf_like_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "size_objectives": size_objectives,
        "size_aggregation": agg_method,
        "size_cvar_alpha": float(cfg.hf.size_cvar_alpha),
        "epoch_objective_mean": epoch_objective_mean,
        "epoch_baseline_violations": epoch_baseline_violations,
        "epoch_better_than_baseline": epoch_better_than_baseline,
        "epoch_eval": {
            "enabled": bool(steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch > 0 else None,
            "epochs_total": int(epochs_total),
            "objectives": epoch_validation_objectives,
            "objective_mean": epoch_objective_mean,
            "baseline_margins": epoch_baseline_margins,
            "baseline_violations": epoch_baseline_violations,
            "better_than_baseline": epoch_better_than_baseline,
        },
        "train_score_mean": float(score_meter.avg),
        "train_loss_mean": float(loss_meter.avg),
        "pair_count": int(total_pairs),
        "early_eval": {
            "enabled": bool(early_eval_effective),
            "steps": int(early_eval_effective),
            "baseline_validation_objective": baseline_early_valid,
            "candidate_validation_objective": early_validation_objective,
            "early_stopped": early_stopped,
        },
        "phases": {
            "f1": {
                "steps": int(steps_f1),
                "train_score_mean": float(score_meter_f1.avg) if steps_f1 > 0 else None,
                "train_loss_mean": float(loss_meter_f1.avg) if steps_f1 > 0 else None,
                "pair_count": int(total_pairs_f1),
            },
            "f2": {
                "steps": int(steps_f2),
                "train_score_mean": float(score_meter.avg) if steps_f2 > 0 else None,
                "train_loss_mean": float(loss_meter.avg) if steps_f2 > 0 else None,
                "pair_count": int(total_pairs - total_pairs_f1),
            },
        },
        "config": {
            "hf": asdict(cfg.hf),
            "free_loss": {
                "f1_steps": cfg.f1_steps,
                "total_train_steps": steps,
                "f2_steps": cfg.f2_steps,
                "f3_enabled": cfg.f3_enabled,
                "baseline_epoch_violation_weight": cfg.baseline_epoch_violation_weight,
            },
        },
        "loss_ir": {
            "name": compiled_loss.ir.name,
            "intuition": compiled_loss.ir.intuition,
            "hyperparams": compiled_loss.ir.hyperparams,
            "operators_used": compiled_loss.ir.operators_used,
        },
    }


def evaluate_po_baseline_rl4co(
    cfg: HighFidelityConfig,
    *,
    early_eval_steps: int | None = None,
) -> Dict[str, Any]:
    _set_seed(cfg.seed)

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    env = _rl4co_build_env(cfg, cfg.train_problem_size)
    env = env.to(device)
    policy, rollout_strategy = _rl4co_build_policy(cfg, env)
    policy = policy.to(device)
    optimizer = Adam(
        policy.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    score_meter = AverageMeter()
    loss_meter = AverageMeter()
    total_steps = get_total_hf_train_steps(cfg)
    steps_per_epoch, epochs_total = get_hf_epoch_plan(cfg)
    epoch_validation_objectives: List[float] = []

    if early_eval_steps is None:
        early_eval_steps = min(100, total_steps)
    else:
        early_eval_steps = min(max(int(early_eval_steps), 0), total_steps)
    early_validation_objective: float | None = None

    log_interval = max(total_steps // 20, 1)

    for step in range(total_steps):
        num_rollouts = resolve_pomo_size(cfg.pomo_size, cfg.train_problem_size)
        reward, log_likelihood = _rl4co_rollout(
            env,
            policy,
            cfg.train_batch_size,
            num_rollouts,
            phase="train",
            rollout_strategy=rollout_strategy,
            device=device,
        )
        preference = reward[:, :, None] > reward[:, None, :]
        log_prob_pair = log_likelihood[:, :, None] - log_likelihood[:, None, :]
        alpha = float(cfg.alpha)
        pf_log = torch.log(torch.sigmoid(alpha * log_prob_pair))
        loss = -torch.mean(pf_log * preference)

        max_reward, _ = reward.max(dim=1)
        score = _rl4co_objective_from_reward(max_reward, cfg).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_meter.update(score.item())
        loss_meter.update(float(loss.item()))

        if (step + 1) % log_interval == 0 or step == 0:
            logger.info(
                "RL4CO baseline PO step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f)",
                step + 1,
                total_steps,
                score.item(),
                float(score_meter.avg),
                float(loss.item()),
                float(loss_meter.avg),
            )

        if steps_per_epoch > 0 and (step + 1) % steps_per_epoch == 0:
            epoch_idx = (step + 1) // steps_per_epoch
            epoch_valid_obj = _evaluate_rl4co_model(
                policy=policy,
                cfg=cfg,
                problem_size=cfg.train_problem_size,
                device=device,
                num_episodes=cfg.num_validation_episodes,
                batch_size=cfg.validation_batch_size,
                rollout_strategy=rollout_strategy,
            )
            epoch_validation_objectives.append(epoch_valid_obj)
            logger.info(
                "RL4CO baseline PO epoch %d/%d: validation_objective=%.6f",
                epoch_idx,
                epochs_total,
                epoch_valid_obj,
            )

        if (step + 1) == early_eval_steps:
            early_validation_objective = _evaluate_rl4co_model(
                policy=policy,
                cfg=cfg,
                problem_size=cfg.train_problem_size,
                device=device,
                num_episodes=cfg.num_validation_episodes,
                batch_size=cfg.validation_batch_size,
                rollout_strategy=rollout_strategy,
            )

    main_valid_obj = _evaluate_rl4co_model(
        policy=policy,
        cfg=cfg,
        problem_size=cfg.train_problem_size,
        device=device,
        num_episodes=cfg.num_validation_episodes,
        batch_size=cfg.validation_batch_size,
        rollout_strategy=rollout_strategy,
    )

    gen_objectives: Dict[int, float] = {}
    for size in cfg.valid_problem_sizes:
        size_int = int(size)
        gen_objectives[size_int] = _evaluate_rl4co_model(
            policy=policy,
            cfg=cfg,
            problem_size=size_int,
            device=device,
            num_episodes=cfg.num_validation_episodes,
            batch_size=cfg.validation_batch_size,
            rollout_strategy=rollout_strategy,
        )

    size_objectives: Dict[int, float] = {int(cfg.train_problem_size): float(main_valid_obj)}
    for size_int, obj in gen_objectives.items():
        size_objectives[int(size_int)] = float(obj)

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, float(max_gen_obj) - float(main_valid_obj))

    epoch_objective_mean: float | None = None
    if epoch_validation_objectives:
        epoch_objective_mean = float(
            sum(epoch_validation_objectives) / len(epoch_validation_objectives)
        )

    agg_method = str(getattr(cfg, "size_aggregation", "legacy") or "legacy").strip().lower()
    base_objective = epoch_objective_mean if epoch_objective_mean is not None else float(main_valid_obj)
    if agg_method == "legacy":
        hf_score = float(main_valid_obj) + cfg.generalization_penalty_weight * generalization_penalty
        fitness_score = base_objective + cfg.generalization_penalty_weight * generalization_penalty
    else:
        hf_score = aggregate_objectives_by_size(
            size_objectives,
            method=agg_method,
            cvar_alpha=float(getattr(cfg, "size_cvar_alpha", 0.2)),
        )
        fitness_score = float(hf_score)

    return {
        "hf_score": hf_score,
        "fitness_score": fitness_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "size_objectives": size_objectives,
        "size_aggregation": agg_method,
        "size_cvar_alpha": float(getattr(cfg, "size_cvar_alpha", 0.2)),
        "train_score_mean": float(score_meter.avg),
        "train_loss_mean": float(loss_meter.avg),
        "early_validation_objective": early_validation_objective,
        "early_eval_steps": early_eval_steps,
        "epoch_eval": {
            "enabled": bool(steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch > 0 else None,
            "epochs_total": int(epochs_total),
            "objectives": epoch_validation_objectives,
            "objective_mean": epoch_objective_mean,
        },
        "config": {
            "hf": cfg.__dict__,
            "baseline_type": "po_loss",
        },
    }


def evaluate_free_loss_candidate(
    compiled_loss: CompiledFreeLoss,
    cfg: FreeLossFidelityConfig,
    *,
    baseline_early_valid: float | None = None,
    early_eval_steps: int = 0,
    baseline_epoch_objectives: Sequence[float] | None = None,
) -> Dict[str, Any]:
    backend = str(getattr(cfg.hf, "backend", "rl4co") or "rl4co").strip().lower()
    if backend != "rl4co":
        raise NotImplementedError(
            "PTP POMO training backend has been removed; only backend='rl4co' is supported."
        )
    return _evaluate_free_loss_candidate_rl4co(
        compiled_loss,
        cfg,
        baseline_early_valid=baseline_early_valid,
        early_eval_steps=early_eval_steps,
        baseline_epoch_objectives=baseline_epoch_objectives,
    )
