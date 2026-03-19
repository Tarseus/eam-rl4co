from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch


DEPOT_ENVS = {
    "cvrp",
    "cvrptw",
    "cvrpmvc",
    "sdvrp",
    "mtsp",
    "op",
    "pctsp",
    "spctsp",
    "knapsack",
}

VARIANT_ALIASES = {
    "ga": "eam",
    "eam": "eam",
    "random_2opt": "random_only",
    "random_only": "random_only",
    "local_search": "ls_only",
    "ls_only": "ls_only",
    "resample": "resample",
}


def infer_problem_size(env: Any) -> Optional[int]:
    generator = getattr(env, "generator", None)
    if generator is None:
        return None
    for attr in ("num_loc", "num_items"):
        value = getattr(generator, attr, None)
        if value is not None:
            return int(value)
    return None


def infer_objective_sense(task_name: str) -> str:
    return "max" if task_name == "op" else "min"


def infer_diversity_metric(task_name: str) -> str:
    return "node_jaccard" if task_name == "op" else "edge_jaccard"


def infer_backbone_name(baseline: str) -> str:
    return "pomo" if baseline == "shared" else "am"


def normalize_variant_name(variant: str) -> str:
    return VARIANT_ALIASES.get(variant, variant)


def reshape_actions(
    actions: Optional[torch.Tensor], batch_size: int, n_traj: Optional[int] = None
) -> Optional[torch.Tensor]:
    if actions is None or batch_size <= 0:
        return None
    if actions.dim() == 3:
        if actions.shape[0] != batch_size:
            return None
        if n_traj is not None and actions.shape[1] != n_traj:
            return None
        return actions
    if actions.dim() != 2:
        return None
    total = actions.shape[0]
    if total % batch_size != 0:
        return None
    inferred = total // batch_size
    if n_traj is not None and inferred != n_traj:
        return None
    return actions.view(batch_size, inferred, actions.shape[-1])


def infer_num_traj(actions: Optional[torch.Tensor], batch_size: int) -> int:
    actions_b = reshape_actions(actions, batch_size, None)
    if actions_b is None:
        return 1
    return int(actions_b.shape[1])


def take_first_trajectories(
    tensor: Optional[torch.Tensor], batch_size: int, count: Optional[int]
) -> Optional[torch.Tensor]:
    if tensor is None or count is None or batch_size <= 0:
        return tensor
    if count <= 0:
        return tensor
    total = tensor.shape[0]
    if total % batch_size != 0:
        return tensor
    n_traj = total // batch_size
    if count >= n_traj:
        return tensor
    new_shape = (batch_size, n_traj, *tensor.shape[1:])
    return tensor.view(new_shape)[:, :count].reshape(batch_size * count, *tensor.shape[1:])


def actions_to_numpy(actions: Optional[torch.Tensor], batch_size: int) -> Optional[np.ndarray]:
    actions_b = reshape_actions(actions, batch_size, None)
    if actions_b is None:
        return None
    return actions_b.detach().cpu().numpy()


def align_improved_actions(
    improved_actions: Optional[torch.Tensor], original_actions: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if improved_actions is None or original_actions is None:
        return improved_actions
    if improved_actions.dim() != original_actions.dim():
        return improved_actions
    if improved_actions.shape[-1] == original_actions.shape[-1]:
        return improved_actions
    if improved_actions.shape[-1] + 1 == original_actions.shape[-1]:
        return torch.cat([original_actions[..., :1], improved_actions], dim=-1)
    return improved_actions


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _jaccard_distance(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - (len(set_a & set_b) / len(union))


def _undirected_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def tsp_solution_to_edge_set(solution: np.ndarray) -> set[tuple[int, int]]:
    route = [int(node) for node in solution.tolist()]
    if len(route) < 2:
        return set()
    edges = set()
    for idx in range(len(route)):
        src = route[idx]
        dst = route[(idx + 1) % len(route)]
        if src != dst:
            edges.add(_undirected_edge(src, dst))
    return edges


def cvrp_solution_to_edge_set(solution: np.ndarray, depot: int = 0) -> set[tuple[int, int]]:
    edges = set()
    prev = depot
    for node in solution.tolist():
        node = int(node)
        if node == depot:
            if prev != depot:
                edges.add(_undirected_edge(prev, depot))
            prev = depot
            continue
        edges.add(_undirected_edge(prev, node))
        prev = node
    if prev != depot:
        edges.add(_undirected_edge(prev, depot))
    return edges


def op_solution_to_edge_set(solution: np.ndarray, depot: int = 0) -> set[tuple[int, int]]:
    route = [int(node) for node in solution.tolist() if int(node) != depot]
    if not route:
        return set()
    edges = {_undirected_edge(depot, route[0]), _undirected_edge(route[-1], depot)}
    for src, dst in zip(route[:-1], route[1:]):
        if src != dst:
            edges.add(_undirected_edge(src, dst))
    return edges


def op_solution_to_node_set(solution: np.ndarray, depot: int = 0) -> set[int]:
    return {int(node) for node in solution.tolist() if int(node) != depot}


class MechanismTaskAdapter:
    def __init__(
        self,
        task_name: str,
        *,
        objective_sense: Optional[str] = None,
        diversity_metric: Optional[str] = None,
    ) -> None:
        self.task_name = task_name
        self.objective_sense = objective_sense or infer_objective_sense(task_name)
        self.diversity_metric = diversity_metric or infer_diversity_metric(task_name)

    def reward_to_score(self, reward: torch.Tensor) -> torch.Tensor:
        if self.objective_sense == "min":
            return -reward
        if self.objective_sense == "max":
            return reward
        raise ValueError(f"Unknown objective_sense: {self.objective_sense}")

    def gain_metric_name(self) -> str:
        return "cost_drop" if self.objective_sense == "min" else "objective_gain"

    def pair_gain(self, score0: torch.Tensor, scorek: torch.Tensor) -> torch.Tensor:
        if self.objective_sense == "min":
            return score0 - scorek
        return scorek - score0

    def solution_to_repr(self, solution: np.ndarray) -> set:
        if self.task_name == "tsp":
            return tsp_solution_to_edge_set(solution)
        if self.task_name == "cvrp":
            return cvrp_solution_to_edge_set(solution)
        if self.task_name == "op":
            if self.diversity_metric == "node_jaccard":
                return op_solution_to_node_set(solution)
            return op_solution_to_edge_set(solution)
        raise NotImplementedError(f"Unsupported task for mechanism probe: {self.task_name}")

    def compute_diversity(self, population: Optional[torch.Tensor], batch_size: int) -> float:
        population_np = actions_to_numpy(population, batch_size)
        if population_np is None:
            return 0.0
        batch_diversity = []
        for batch_actions in population_np:
            n_traj = batch_actions.shape[0]
            if n_traj < 2:
                continue
            reprs = [self.solution_to_repr(solution) for solution in batch_actions]
            total = 0.0
            count = 0
            for idx in range(n_traj - 1):
                for jdx in range(idx + 1, n_traj):
                    total += _jaccard_distance(reprs[idx], reprs[jdx])
                    count += 1
            if count > 0:
                batch_diversity.append(total / count)
        return _average(batch_diversity)

    def compute_paired_diversity(
        self,
        tau0: Optional[torch.Tensor],
        tauk: Optional[torch.Tensor],
        batch_size: int,
        pair_count: Optional[int] = None,
    ) -> float:
        if tau0 is None or tauk is None:
            return 0.0
        tau0_np = actions_to_numpy(take_first_trajectories(tau0, batch_size, pair_count), batch_size)
        tauk_np = actions_to_numpy(take_first_trajectories(tauk, batch_size, pair_count), batch_size)
        if tau0_np is None or tauk_np is None:
            return 0.0
        per_pair = []
        for batch_idx in range(tau0_np.shape[0]):
            n_pairs = min(tau0_np.shape[1], tauk_np.shape[1])
            if n_pairs == 0:
                continue
            distances = []
            for pair_idx in range(n_pairs):
                repr0 = self.solution_to_repr(tau0_np[batch_idx, pair_idx])
                reprk = self.solution_to_repr(tauk_np[batch_idx, pair_idx])
                distances.append(_jaccard_distance(repr0, reprk))
            per_pair.append(_average(distances))
        return _average(per_pair)


@dataclass
class MechanismConfig:
    enabled: bool = False
    variant: str = "eam"
    base_traj: Optional[int] = None
    extra_traj: Optional[int] = None
    refine_budget: Optional[int] = None
    trigger_every: int = 1
    log_every: int = 10
    save_root: str = "outputs/mechanism"
    diversity_metric: Optional[str] = None
    objective_sense: Optional[str] = None
    paired_gain: bool = True
    compute_delta_nll: bool = True
    seed: int = 0
    backbone: Optional[str] = None


def build_mechanism_config(
    env: Any,
    baseline: str,
    mechanism: Optional[dict[str, Any]],
    ea_kwargs: Optional[dict[str, Any]],
    num_starts: Optional[int],
) -> MechanismConfig:
    mechanism = mechanism or {}
    ea_kwargs = ea_kwargs or {}
    legacy_variant = normalize_variant_name(ea_kwargs.get("improve_mode", "ga"))
    variant = normalize_variant_name(mechanism.get("variant", legacy_variant))
    task_name = getattr(env, "name", "unknown")
    config = MechanismConfig(
        enabled=bool(mechanism.get("enabled", True)),
        variant=variant,
        base_traj=mechanism.get("base_traj", num_starts),
        extra_traj=mechanism.get("extra_traj", num_starts),
        refine_budget=mechanism.get("refine_budget", ea_kwargs.get("num_generations")),
        trigger_every=int(mechanism.get("trigger_every", 1) or 1),
        log_every=int(mechanism.get("log_every", 10) or 1),
        save_root=str(mechanism.get("save_root", "outputs/mechanism")),
        diversity_metric=mechanism.get("diversity_metric", infer_diversity_metric(task_name)),
        objective_sense=mechanism.get("objective_sense", infer_objective_sense(task_name)),
        paired_gain=bool(mechanism.get("paired_gain", True)),
        compute_delta_nll=bool(mechanism.get("compute_delta_nll", True)),
        seed=int(mechanism.get("seed", 0) or 0),
        backbone=mechanism.get("backbone", infer_backbone_name(baseline)),
    )
    if config.variant in {"random_only", "ls_only"} and config.refine_budget is None:
        config.refine_budget = 1
    return config


class AugmentController:
    def __init__(self, config: MechanismConfig) -> None:
        self.config = config
        self.variant = config.variant
        self.base_traj = config.base_traj
        self.extra_traj = config.extra_traj
        self.refine_budget = config.refine_budget
        self.trigger_every = max(1, int(config.trigger_every))

    def resolve_base_traj(self, fallback: int) -> int:
        return max(1, int(self.base_traj if self.base_traj is not None else fallback))

    def resolve_extra_traj(self, fallback: int) -> int:
        return max(1, int(self.extra_traj if self.extra_traj is not None else fallback))

    def resolve_refine_budget(self, fallback: Optional[int] = None) -> Optional[int]:
        budget = self.refine_budget if self.refine_budget is not None else fallback
        if budget is None:
            return None
        return max(1, int(budget))

    def triggered(self, step: int, improve_prob: Optional[float] = None) -> bool:
        if not self.config.enabled:
            return False
        if (step + 1) % self.trigger_every != 0:
            return False
        if improve_prob is None or improve_prob >= 1.0:
            return True
        return bool(np.random.random() <= improve_prob)

    def augment(
        self,
        *,
        tau0: torch.Tensor,
        batch_size: int,
        sample_fn: Callable[[int], dict[str, Any]],
        ga_fn: Callable[[torch.Tensor, Optional[int]], tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        random_only_fn: Callable[[torch.Tensor, int], Optional[torch.Tensor]],
        ls_only_fn: Callable[[torch.Tensor, int], Optional[torch.Tensor]],
    ) -> Optional[dict[str, Any]]:
        pair_count = infer_num_traj(tau0, batch_size)
        pair_ids = torch.arange(pair_count, device=tau0.device)

        if self.variant == "resample":
            # Keep paired training/evaluation shapes aligned in the current EAM path.
            extra_traj = min(self.resolve_extra_traj(pair_count), pair_count)
            resample_out = sample_fn(extra_traj)
            tauk = resample_out.get("actions", None)
            if tauk is None:
                return None
            pair_count = min(pair_count, infer_num_traj(tauk, batch_size))
            return {
                "tau0": tau0,
                "tauk": tauk,
                "pair_count": pair_count,
                "pair_ids": pair_ids[:pair_count],
                "precomputed_out": resample_out,
                "population_actions": None,
            }

        budget = self.resolve_refine_budget(pair_count)
        if self.variant == "random_only":
            tauk = random_only_fn(tau0, budget or 1)
            population_actions = None
        elif self.variant == "ls_only":
            tauk = ls_only_fn(tau0, budget or 1)
            population_actions = None
        elif self.variant == "eam":
            tauk, population_actions = ga_fn(tau0, budget)
        else:
            raise ValueError(f"Unknown mechanism variant: {self.variant}")

        tauk = align_improved_actions(tauk, tau0)
        if tauk is None:
            return None
        pair_count = min(pair_count, infer_num_traj(tauk, batch_size))
        return {
            "tau0": tau0,
            "tauk": tauk,
            "pair_count": pair_count,
            "pair_ids": pair_ids[:pair_count],
            "precomputed_out": None,
            "population_actions": population_actions,
        }


class MechanismProbe:
    def __init__(
        self,
        *,
        task_name: str,
        size: Optional[int],
        backbone: str,
        config: MechanismConfig,
    ) -> None:
        self.task = task_name
        self.size = size
        self.backbone = backbone
        self.config = config
        self.adapter = MechanismTaskAdapter(
            task_name,
            objective_sense=config.objective_sense,
            diversity_metric=config.diversity_metric,
        )

    def should_log(self, step: int) -> bool:
        return self.config.enabled and (step + 1) % max(1, self.config.log_every) == 0

    def compute(
        self,
        *,
        variant: str,
        batch_size: int,
        tau0: Optional[torch.Tensor],
        tauk: Optional[torch.Tensor],
        pair_count: Optional[int],
        score0: Optional[torch.Tensor],
        scorek: Optional[torch.Tensor],
        log_likelihood0: Optional[torch.Tensor],
        log_likelihoodk: Optional[torch.Tensor],
        population_actions: Optional[torch.Tensor],
        step: int,
        epoch: int,
        val_metric: Optional[float],
    ) -> dict[str, Any]:
        diversity = self.adapter.compute_diversity(
            population_actions if population_actions is not None else tauk,
            batch_size,
        )
        paired_diversity = self.adapter.compute_paired_diversity(
            tau0, tauk, batch_size, pair_count
        )

        gain = 0.0
        relative_gain = 0.0
        if score0 is not None and scorek is not None:
            if self.config.paired_gain:
                score0_eval = take_first_trajectories(score0, batch_size, pair_count)
                scorek_eval = take_first_trajectories(scorek, batch_size, pair_count)
                gain_tensor = self.adapter.pair_gain(score0_eval, scorek_eval)
            else:
                score0_eval = score0
                scorek_eval = scorek
                score0_mean = reshape_actions(score0_eval, batch_size, None)
                scorek_mean = reshape_actions(scorek_eval, batch_size, None)
                if score0_mean is not None:
                    score0_eval = score0_mean.mean(dim=1)
                if scorek_mean is not None:
                    scorek_eval = scorek_mean.mean(dim=1)
                gain_tensor = self.adapter.pair_gain(score0_eval, scorek_eval)
            gain = float(gain_tensor.mean().item())
            denom = float(score0_eval.abs().mean().item()) + 1e-8
            relative_gain = gain / denom

        delta_nll = 0.0
        if (
            self.config.compute_delta_nll
            and log_likelihood0 is not None
            and log_likelihoodk is not None
        ):
            ll0 = take_first_trajectories(log_likelihood0, batch_size, pair_count)
            llk = take_first_trajectories(log_likelihoodk, batch_size, pair_count)
            delta_nll = float(((-llk).mean() - (-ll0).mean()).item())

        return {
            "task": self.task,
            "size": self.size,
            "backbone": self.backbone,
            "variant": normalize_variant_name(variant),
            "seed": self.config.seed,
            "step": int(step),
            "epoch": int(epoch),
            "val_metric": None if val_metric is None else float(val_metric),
            "diversity": float(diversity),
            "paired_diversity": float(paired_diversity),
            "gain": float(gain),
            "relative_gain": float(relative_gain),
            "delta_nll": float(delta_nll),
            "diversity_metric": self.adapter.diversity_metric,
            "gain_metric": self.adapter.gain_metric_name(),
            "objective_sense": self.adapter.objective_sense,
        }

    def dump(self, stats: dict[str, Any]) -> Path:
        size = "unknown" if stats["size"] is None else str(stats["size"])
        run_dir = (
            Path(self.config.save_root)
            / f"{stats['task']}{size}_{stats['backbone']}"
            / f"seed{stats['seed']}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / f"{stats['variant']}.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stats.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(stats)
        return csv_path
