"""Freeze same-prefix forced-action probes from real TSP policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from fitness.free_loss_fidelity import (
    _load_policy_weights_from_checkpoint,
    _rl4co_build_env,
    _rl4co_objective_from_reward,
    _rl4co_build_policy,
)
from fitness.ptp_high_fidelity import HighFidelityConfig, _set_seed
from rl4co.utils.ops import unbatchify


SCHEMA = "nco-matched-branch-probes-v1"


@dataclass(frozen=True)
class MatchedBranchProbe:
    policy_state: str
    depth: int
    local_logp: torch.Tensor
    terminal_logp: torch.Tensor
    objective: torch.Tensor

    def validate(self) -> None:
        values = (self.local_logp, self.terminal_logp, self.objective)
        if any(value.ndim != 2 for value in values):
            raise ValueError("branch probe tensors must have shape [instance, branch]")
        if len({tuple(value.shape) for value in values}) != 1:
            raise ValueError("branch probe tensors must share shape")
        if not all(torch.isfinite(value).all().item() for value in values):
            raise ValueError("branch probe tensors must be finite")
        if self.depth < 1:
            raise ValueError("depth must retain at least one shared prefix action")


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_fingerprint(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_record(probe: MatchedBranchProbe) -> dict[str, Any]:
    probe.validate()
    return {
        "policy_state": probe.policy_state,
        "depth": int(probe.depth),
        "local_logp": probe.local_logp.detach().double().cpu().tolist(),
        "terminal_logp": probe.terminal_logp.detach().double().cpu().tolist(),
        "objective": probe.objective.detach().double().cpu().tolist(),
    }


def save_matched_branch_probes(
    path: str | Path,
    probes: Sequence[MatchedBranchProbe],
    *,
    config: Mapping[str, Any],
    late_checkpoint: str | None,
    problem_size: int,
    branch_count: int,
) -> Path:
    states = {probe.policy_state for probe in probes}
    if states != {"early", "late"}:
        raise ValueError("probe suite requires early and late policy states")
    payload = {
        "schema": SCHEMA,
        "source": "real_same_prefix_forced_action_rollout",
        "frozen": True,
        "task_name": "tsp",
        "problem_size": int(problem_size),
        "branch_count": int(branch_count),
        "policy_states": sorted(states),
        "config_sha256": _config_fingerprint(config),
        "late_checkpoint": str(late_checkpoint) if late_checkpoint else None,
        "late_checkpoint_sha256": _file_fingerprint(late_checkpoint),
        "probes": [_probe_record(probe) for probe in probes],
    }
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def _collect_policy_state(
    hf_cfg: HighFidelityConfig,
    *,
    policy_state: str,
    checkpoint_path: str | None,
    instances: int,
    branch_count: int,
    depths: Sequence[int],
    device: torch.device | str,
) -> list[MatchedBranchProbe]:
    if policy_state not in {"early", "late"}:
        raise ValueError("policy_state must be 'early' or 'late'")
    if instances < 1:
        raise ValueError("instances must be positive")
    _set_seed(int(hf_cfg.seed))
    env = _rl4co_build_env(hf_cfg, int(hf_cfg.train_problem_size)).to(device)
    policy, _ = _rl4co_build_policy(hf_cfg, env)
    if checkpoint_path:
        _load_policy_weights_from_checkpoint(policy, checkpoint_path)
    policy = policy.to(device)
    policy.eval()

    if not hasattr(policy, "forward"):
        raise TypeError("matched branch collection requires a constructive policy")
    generator = getattr(env, "generator", None)
    if generator is None:
        raise RuntimeError("TSP environment has no generator")
    batch = generator(int(instances)).to(device)
    td = env.reset(batch)
    problem_size = int(td["locs"].shape[-2])
    if branch_count < 2 or branch_count > problem_size:
        raise ValueError("branch_count must lie in [2, problem_size]")
    normalized_depths = tuple(int(depth) for depth in depths)
    if not normalized_depths:
        raise ValueError("at least one anchor depth is required")
    if any(
        depth < 1 or depth + branch_count > problem_size
        for depth in normalized_depths
    ):
        raise ValueError(
            "each depth must retain one shared action and enough unvisited branch actions"
        )

    with torch.inference_mode():
        base = policy(
            td,
            env,
            phase="val",
            num_starts=max(int(branch_count), 2),
            return_actions=True,
        )
        base_actions_all = unbatchify(
            base["actions"], (0, max(int(branch_count), 2))
        )
        base_actions = base_actions_all[:, 0]

        probes: list[MatchedBranchProbe] = []
        for depth in normalized_depths:
            shared = base_actions[:, None, :depth].expand(
                -1, branch_count, -1
            )
            candidates = base_actions[:, depth : depth + branch_count]
            forced_prefix = torch.cat(
                (shared, candidates.unsqueeze(-1)), dim=-1
            )
            branched = policy(
                td,
                env,
                phase="val",
                num_starts=int(branch_count),
                return_actions=True,
                return_sum_log_likelihood=False,
                forced_prefix_actions=forced_prefix,
            )
            actions = unbatchify(branched["actions"], (0, branch_count))
            step_logp = unbatchify(
                branched["log_likelihood"], (0, branch_count)
            )
            reward = unbatchify(branched["reward"], (0, branch_count))
            if not torch.equal(actions[:, :, :depth], shared):
                raise RuntimeError("branched trajectories do not share the anchor prefix")
            if not torch.equal(actions[:, :, depth], candidates):
                raise RuntimeError("branched trajectories did not take forced actions")
            objective = _rl4co_objective_from_reward(reward, hf_cfg)
            probes.append(
                MatchedBranchProbe(
                    policy_state=policy_state,
                    depth=depth,
                    local_logp=step_logp[:, :, depth].detach().cpu(),
                    terminal_logp=step_logp.sum(dim=-1).detach().cpu(),
                    objective=objective.detach().cpu(),
                )
            )
    for probe in probes:
        probe.validate()
    return probes


def freeze_matched_branch_probe_suite(
    config_path: str | Path,
    output_path: str | Path,
    *,
    late_checkpoint: str,
    instances: int = 16,
    branch_count: int = 8,
    depths: Sequence[int] = (1, 25, 50, 75),
    device: str = "cuda",
) -> Path:
    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError("config root must be a mapping")
    if str(config.get("env_name", config.get("problem", ""))).lower() != "tsp":
        raise ValueError("matched branch pilot currently supports only TSP")
    from ptp_discovery.pref_loss_coevo_loop import _build_hf_cfg

    seed = int(config.get("scratch_init_seed", config.get("seed", 1234)))
    hf_cfg = _build_hf_cfg(config, seed=seed, device_str=str(device))
    target_device = torch.device(str(device))
    probes = _collect_policy_state(
        hf_cfg,
        policy_state="early",
        checkpoint_path=None,
        instances=instances,
        branch_count=branch_count,
        depths=depths,
        device=target_device,
    )
    probes.extend(
        _collect_policy_state(
            hf_cfg,
            policy_state="late",
            checkpoint_path=str(late_checkpoint),
            instances=instances,
            branch_count=branch_count,
            depths=depths,
            device=target_device,
        )
    )
    return save_matched_branch_probes(
        output_path,
        probes,
        config=config,
        late_checkpoint=late_checkpoint,
        problem_size=int(hf_cfg.train_problem_size),
        branch_count=branch_count,
    )
