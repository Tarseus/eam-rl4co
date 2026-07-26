"""Freeze NCO branch probes with exact policy-parameter tangent influence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from fitness.free_loss_fidelity import (
    _load_policy_weights_from_checkpoint,
    _rl4co_build_env,
    _rl4co_build_policy,
    _rl4co_objective_from_reward,
)
from fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed
from ptp_discovery.matched_branch_probe import (
    _config_fingerprint,
    _file_fingerprint,
)
from rl4co.utils.ops import unbatchify


SCHEMA = "nco-matched-branch-tangent-probes-v2"


@dataclass(frozen=True)
class TangentBranchProbe:
    policy_state: str
    depth: int
    local_logp: torch.Tensor
    terminal_logp: torch.Tensor
    objective: torch.Tensor
    local_target_influence: torch.Tensor
    terminal_target_influence: torch.Tensor
    target_gradient_norm: float

    def validate(self) -> None:
        tensors = (
            self.local_logp,
            self.terminal_logp,
            self.objective,
            self.local_target_influence,
            self.terminal_target_influence,
        )
        if any(tensor.ndim != 2 for tensor in tensors):
            raise ValueError("tangent probe tensors must have shape [instance, branch]")
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError("tangent probe tensors must share shape")
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            raise ValueError("tangent probe tensors must be finite")
        if not torch.isfinite(torch.tensor(self.target_gradient_norm)):
            raise ValueError("target gradient norm must be finite")
        if self.target_gradient_norm <= 0.0:
            raise ValueError("target gradient norm must be positive")


def parameter_tangent_influence(
    observables: Sequence[torch.Tensor],
    target: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    """Return ``J_observable @ grad_parameter(target)`` exactly.

    The implementation uses one reverse-over-reverse product, independent of
    how many downstream source programs will later be screened.
    """

    influence, _ = _parameter_tangent_influence_and_norm(
        observables, target, parameters
    )
    return influence


def _parameter_tangent_influence_and_norm(
    observables: Sequence[torch.Tensor],
    target: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[tuple[torch.Tensor, ...], float]:
    if not observables:
        raise ValueError("at least one observable tensor is required")
    if target.numel() != 1:
        raise ValueError("target must be scalar")
    active_parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not active_parameters:
        raise ValueError("no trainable policy parameter is available")
    target_gradient = torch.autograd.grad(
        target,
        active_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    target_norm_squared = sum(
        float(part.detach().double().square().sum())
        for part in target_gradient
        if part is not None
    )
    target_gradient_norm = target_norm_squared**0.5
    if target_gradient_norm <= 0.0:
        raise RuntimeError("branch target has zero parameter gradient")
    flat_observable = torch.cat([tensor.reshape(-1) for tensor in observables])
    cotangent = torch.ones_like(flat_observable, requires_grad=True)
    observable_probe = (flat_observable * cotangent).sum()
    probe_gradient = torch.autograd.grad(
        observable_probe,
        active_parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    parameter_inner_product = None
    for target_part, probe_part in zip(target_gradient, probe_gradient):
        if target_part is None or probe_part is None:
            continue
        term = (target_part.detach() * probe_part).sum()
        parameter_inner_product = (
            term
            if parameter_inner_product is None
            else parameter_inner_product + term
        )
    if parameter_inner_product is None:
        raise RuntimeError("target and observables share no trainable parameter")
    flat_influence = torch.autograd.grad(parameter_inner_product, cotangent)[0]
    outputs = []
    offset = 0
    for tensor in observables:
        count = tensor.numel()
        outputs.append(flat_influence[offset : offset + count].reshape_as(tensor))
        offset += count
    return tuple(outputs), target_gradient_norm


def _normalize_objective(objective: torch.Tensor) -> torch.Tensor:
    minimum = objective.min(dim=-1, keepdim=True).values
    maximum = objective.max(dim=-1, keepdim=True).values
    scale = (maximum - minimum).clamp_min(1e-12)
    return ((objective - minimum) / scale).detach()


def _collect_tangent_policy_state(
    hf_cfg: HighFidelityConfig,
    *,
    policy_state: str,
    checkpoint_path: str | None,
    instances: int,
    branch_count: int,
    depths: Sequence[int],
    device: torch.device | str,
) -> list[TangentBranchProbe]:
    if policy_state not in {"early", "late"}:
        raise ValueError("policy_state must be 'early' or 'late'")
    _set_seed(int(hf_cfg.seed))
    env = _rl4co_build_env(hf_cfg, int(hf_cfg.train_problem_size)).to(device)
    policy, _ = _rl4co_build_policy(hf_cfg, env)
    if checkpoint_path:
        _load_policy_weights_from_checkpoint(policy, checkpoint_path)
    policy = policy.to(device)
    policy.eval()
    policy.zero_grad(set_to_none=True)

    generator = getattr(env, "generator", None)
    if generator is None:
        raise RuntimeError("TSP environment has no generator")
    batch = generator(int(instances)).to(device)
    td = env.reset(batch)
    problem_size = int(td["locs"].shape[-2])
    normalized_depths = tuple(int(depth) for depth in depths)
    if branch_count < 2 or branch_count > problem_size:
        raise ValueError("branch_count must lie in [2, problem_size]")
    if not normalized_depths:
        raise ValueError("at least one depth is required")
    if any(
        depth < 1 or depth + branch_count > problem_size
        for depth in normalized_depths
    ):
        raise ValueError("each depth must leave enough distinct branch actions")

    with torch.inference_mode():
        base = policy(
            td,
            env,
            phase="val",
            num_starts=max(branch_count, 2),
            return_actions=True,
        )
        base_actions = unbatchify(
            base["actions"], (0, max(branch_count, 2))
        )[:, 0]

    local_observables: list[torch.Tensor] = []
    terminal_observables: list[torch.Tensor] = []
    objectives: list[torch.Tensor] = []
    target_terms: list[torch.Tensor] = []
    for depth in normalized_depths:
        shared = base_actions[:, None, :depth].expand(-1, branch_count, -1)
        candidates = base_actions[:, depth : depth + branch_count]
        forced_prefix = torch.cat((shared, candidates.unsqueeze(-1)), dim=-1)
        output = policy(
            td,
            env,
            phase="val",
            num_starts=branch_count,
            return_actions=True,
            return_sum_log_likelihood=False,
            forced_prefix_actions=forced_prefix,
        )
        actions = unbatchify(output["actions"], (0, branch_count))
        if not torch.equal(actions[:, :, :depth], shared):
            raise RuntimeError("tangent branches do not share the forced prefix")
        if not torch.equal(actions[:, :, depth], candidates):
            raise RuntimeError("tangent branches did not take the forced action")
        step_logp = unbatchify(output["log_likelihood"], (0, branch_count))
        reward = unbatchify(output["reward"], (0, branch_count))
        local_logp = step_logp[:, :, depth]
        terminal_logp = step_logp.sum(dim=-1)
        objective = _rl4co_objective_from_reward(reward, hf_cfg).detach()
        normalized_objective = _normalize_objective(objective)
        local_observables.append(local_logp)
        terminal_observables.append(terminal_logp)
        objectives.append(objective)
        target_terms.append(
            (torch.softmax(local_logp, dim=-1) * normalized_objective)
            .sum(dim=-1)
            .mean()
        )

    target = torch.stack(target_terms).mean()
    influences, target_gradient_norm = _parameter_tangent_influence_and_norm(
        tuple(local_observables) + tuple(terminal_observables),
        target,
        tuple(policy.parameters()),
    )
    split = len(local_observables)
    local_influences = influences[:split]
    terminal_influences = influences[split:]
    probes = []
    for index, depth in enumerate(normalized_depths):
        probe = TangentBranchProbe(
            policy_state=policy_state,
            depth=depth,
            local_logp=local_observables[index].detach().cpu(),
            terminal_logp=terminal_observables[index].detach().cpu(),
            objective=objectives[index].detach().cpu(),
            local_target_influence=local_influences[index].detach().cpu(),
            terminal_target_influence=terminal_influences[index].detach().cpu(),
            target_gradient_norm=target_gradient_norm,
        )
        probe.validate()
        probes.append(probe)
    return probes


def _probe_record(probe: TangentBranchProbe) -> dict[str, Any]:
    probe.validate()
    return {
        "policy_state": probe.policy_state,
        "depth": probe.depth,
        "local_logp": probe.local_logp.double().tolist(),
        "terminal_logp": probe.terminal_logp.double().tolist(),
        "objective": probe.objective.double().tolist(),
        "local_target_influence": probe.local_target_influence.double().tolist(),
        "terminal_target_influence": probe.terminal_target_influence.double().tolist(),
        "target_gradient_norm": probe.target_gradient_norm,
    }


def freeze_tangent_branch_probe_suite(
    config_path: str | Path,
    output_path: str | Path,
    *,
    late_checkpoint: str,
    instances: int = 8,
    branch_count: int = 8,
    depths: Sequence[int] = (1, 25, 50, 75),
    device: str = "cuda",
) -> Path:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError("config root must be a mapping")
    if str(config.get("env_name", config.get("problem", ""))).lower() != "tsp":
        raise ValueError("tangent pilot currently supports only TSP")
    from ptp_discovery.pref_loss_coevo_loop import _build_hf_cfg

    seed = int(config.get("scratch_init_seed", config.get("seed", 1234)))
    hf_cfg = _build_hf_cfg(config, seed=seed, device_str=device)
    probes = _collect_tangent_policy_state(
        hf_cfg,
        policy_state="early",
        checkpoint_path=None,
        instances=instances,
        branch_count=branch_count,
        depths=depths,
        device=device,
    )
    probes.extend(
        _collect_tangent_policy_state(
            hf_cfg,
            policy_state="late",
            checkpoint_path=late_checkpoint,
            instances=instances,
            branch_count=branch_count,
            depths=depths,
            device=device,
        )
    )
    payload = {
        "schema": SCHEMA,
        "source": "real_same_prefix_policy_tangent",
        "task_name": "tsp",
        "problem_size": int(hf_cfg.train_problem_size),
        "branch_count": int(branch_count),
        "policy_states": ["early", "late"],
        "config_sha256": _config_fingerprint(config),
        "late_checkpoint": str(late_checkpoint),
        "late_checkpoint_sha256": _file_fingerprint(late_checkpoint),
        "probes": [_probe_record(probe) for probe in probes],
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination
