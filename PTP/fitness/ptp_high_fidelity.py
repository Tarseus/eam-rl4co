from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class HighFidelityConfig:
    """Configuration for short-run high-fidelity evaluation.

    Note: the original PTP POMO training backend has been removed.
    Use the RL4CO backend for fitness evaluation.
    """

    problem: str = "tsp"
    # Backend: "rl4co" (RL4CO env/policy). The original "ptp" backend is removed.
    backend: str = "rl4co"
    # RL4CO-specific fields.
    env_name: str = ""
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    generator_params: Dict[str, Any] = field(default_factory=dict)
    policy_name: str = ""
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    rollout_strategy: str = "auto"  # auto | policy_multistart | batchify_sampling
    objective_sign: str = "neg_reward"  # neg_reward | reward
    hf_steps: int = 200
    # Optional epoch-style configuration. When both hf_epochs and
    # hf_instances_per_epoch are > 0, they are used to derive the
    # total number of training steps as:
    #   total_steps = hf_epochs * ceil(hf_instances_per_epoch / train_batch_size)
    # and hf_steps is treated as a fallback/legacy setting.
    hf_epochs: int = 0
    hf_instances_per_epoch: int = 0
    train_problem_size: int = 20
    valid_problem_sizes: Sequence[int] = (100,)
    train_batch_size: int = 64
    # If None, align POMO rollout count to the current problem size for all
    # training/evaluation calls. If set, the value overrides across sizes.
    pomo_size: int | None = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    alpha: float = 0.05  # preference sharpness
    device: str = "cuda"
    seed: int = 0
    num_validation_episodes: int = 128
    validation_batch_size: int = 64
    generalization_penalty_weight: float = 1.0
    size_aggregation: str = "cvar"  # one of: legacy, mean, cvar, worst
    size_cvar_alpha: float = 0.2
    pool_version: str = "v0"


def resolve_pomo_size(pomo_size: int | None, problem_size: int) -> int:
    """Return the effective pomo_size for a given problem_size.

    If pomo_size is not provided (None), align it to problem_size.
    """

    if pomo_size is None:
        return int(problem_size)
    value = int(pomo_size)
    if value <= 0:
        logger.warning(
            "Invalid pomo_size=%s; falling back to problem_size=%d", pomo_size, problem_size
        )
        return int(problem_size)
    return value


def aggregate_objectives_by_size(
    size_objectives: Mapping[int, float],
    *,
    method: str = "cvar",
    cvar_alpha: float = 0.2,
) -> float:
    values = [float(v) for v in size_objectives.values()]
    if not values:
        return float("inf")

    m = str(method or "cvar").strip().lower()
    if m in {"legacy", "mean"}:
        return float(sum(values) / len(values))
    if m in {"worst", "max"}:
        return float(max(values))
    if m == "cvar":
        alpha = float(cvar_alpha)
        if not (0.0 < alpha <= 1.0):
            alpha = 0.2
        values_sorted = sorted(values, reverse=True)  # worst objectives first
        k = max(1, int(math.ceil(alpha * len(values_sorted))))
        return float(sum(values_sorted[:k]) / k)

    raise ValueError(f"Unknown size aggregation method: {method}")


def get_total_hf_train_steps(config: HighFidelityConfig) -> int:
    """Return total training *steps* given an HF config.

    - If both hf_epochs and hf_instances_per_epoch are > 0, derive the total
      number of steps from an epoch-style configuration.
    - Otherwise, fall back to hf_steps (legacy behaviour).
    """

    if config.hf_epochs > 0 and config.hf_instances_per_epoch > 0:
        batch_size = max(int(config.train_batch_size), 1)
        steps_per_epoch = math.ceil(config.hf_instances_per_epoch / batch_size)
        total_steps = config.hf_epochs * steps_per_epoch
        return max(int(total_steps), 1)

    return max(int(config.hf_steps), 1)


def get_hf_epoch_plan(config: HighFidelityConfig) -> Tuple[int, int]:
    """Return (steps_per_epoch, epochs_total) for epoch-style HF configs.

    When hf_epochs and hf_instances_per_epoch are set, we derive the number
    of training steps per epoch; otherwise, return (0, 0).
    """

    if config.hf_epochs > 0 and config.hf_instances_per_epoch > 0:
        batch_size = max(int(config.train_batch_size), 1)
        steps_per_epoch = math.ceil(config.hf_instances_per_epoch / batch_size)
        return max(int(steps_per_epoch), 1), int(config.hf_epochs)

    return 0, 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_ptp_dsl_high_fidelity(*_args, **_kwargs) -> Dict[str, Any]:
    """PTP POMO training backend has been removed.

    This stub keeps the public API but prevents accidental use.
    """

    raise NotImplementedError(
        "PTP POMO high-fidelity evaluation has been removed. "
        "Use the RL4CO backend in fitness.free_loss_fidelity instead."
    )
