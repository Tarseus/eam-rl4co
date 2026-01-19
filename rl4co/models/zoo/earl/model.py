from typing import Any, Callable
from typing import IO, Any, Optional, Union, cast

import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.earl.evolution import evolution_worker, EA
from rl4co.utils.ops import gather_by_index, unbatchify, batchify
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.ops import get_distance_matrix

import concurrent.futures
import numpy as np
import time
import numba as nb

from tensordict import TensorDict
import torch

from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)

log = get_pylogger(__name__)

def evolve_prob_schedule(epoch, max_epoch, initial_prob, final_prob):
    return np.cos(np.pi * epoch / max_epoch) * (final_prob - initial_prob) + initial_prob

def sigmoid_schedule(epoch, max_epoch, initial_prob, final_prob):
    x = 10 * (epoch / max_epoch - 0.5)
    sigmoid = 1 / (1 + np.exp(-x))
    return initial_prob + (final_prob - initial_prob) * sigmoid

def step_schedule(epoch, ea_prob, ea_epoch):
    return ea_prob if (epoch <= ea_epoch or ea_epoch < 0)else 0.0

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


def _infer_num_nodes(env: RL4COEnvBase) -> Optional[int]:
    gen = getattr(env, "generator", None)
    if gen is None:
        return None
    for attr in ("num_loc", "num_items"):
        val = getattr(gen, attr, None)
        if val is not None:
            return int(val)
    return None


def _reshape_actions(actions: torch.Tensor, batch_size: int, n_traj: Optional[int]):
    if actions is None or actions.dim() != 2 or batch_size <= 0:
        return None
    total = actions.shape[0]
    if total % batch_size != 0:
        return None
    inferred = total // batch_size
    n_traj = inferred if n_traj is None else inferred
    return actions.view(batch_size, inferred, actions.shape[1])


def _edge_set(seq: np.ndarray) -> set[tuple[int, int]]:
    edges = set()
    for i in range(len(seq) - 1):
        a = int(seq[i])
        b = int(seq[i + 1])
        if a == b:
            continue
        if a < b:
            edges.add((a, b))
        else:
            edges.add((b, a))
    return edges


def _average_pairwise(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _edge_diversity(actions_np: np.ndarray) -> float:
    batch_size, n_traj, _ = actions_np.shape
    per_batch = []
    for b in range(batch_size):
        if n_traj < 2:
            continue
        edge_sets = [_edge_set(actions_np[b, i]) for i in range(n_traj)]
        total = 0.0
        count = 0
        for i in range(n_traj - 1):
            for j in range(i + 1, n_traj):
                union = edge_sets[i] | edge_sets[j]
                if not union:
                    dist = 0.0
                else:
                    inter = edge_sets[i] & edge_sets[j]
                    dist = 1.0 - (len(inter) / len(union))
                total += dist
                count += 1
        per_batch.append(total / count if count > 0 else 0.0)
    return _average_pairwise(per_batch)


def _rank_sequence(seq: np.ndarray, n_nodes: int, ignore_zero: bool) -> list[int]:
    default_rank = n_nodes + 1
    ranks = [default_rank] * n_nodes
    for pos, node in enumerate(seq):
        node = int(node)
        if ignore_zero and node == 0:
            continue
        if ignore_zero:
            if not (1 <= node <= n_nodes):
                continue
            idx = node - 1
        else:
            if not (0 <= node < n_nodes):
                continue
            idx = node
        if ranks[idx] == default_rank:
            ranks[idx] = pos
    return ranks


def _discordant_pairs(rank_a: list[int], rank_b: list[int]) -> tuple[int, int]:
    n = len(rank_a)
    discordant = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n - 1):
        ai = rank_a[i]
        bi = rank_b[i]
        for j in range(i + 1, n):
            da = ai - rank_a[j]
            db = bi - rank_b[j]
            if da == 0 or db == 0:
                continue
            if da * db < 0:
                discordant += 1
    return discordant, total_pairs


def _order_diversity(actions_np: np.ndarray, n_nodes: int, ignore_zero: bool) -> float:
    batch_size, n_traj, _ = actions_np.shape
    per_batch = []
    for b in range(batch_size):
        if n_traj < 2:
            continue
        ranks = [_rank_sequence(actions_np[b, i], n_nodes, ignore_zero) for i in range(n_traj)]
        total = 0.0
        count = 0
        for i in range(n_traj - 1):
            for j in range(i + 1, n_traj):
                discordant, total_pairs = _discordant_pairs(ranks[i], ranks[j])
                if total_pairs == 0:
                    dist = 0.0
                else:
                    dist = discordant / total_pairs
                total += dist
                count += 1
        per_batch.append(total / count if count > 0 else 0.0)
    return _average_pairwise(per_batch)


def _infer_num_traj(actions: Optional[torch.Tensor], batch_size: int) -> int:
    actions_b = _reshape_actions(actions, batch_size, None)
    if actions_b is None:
        return 1
    return int(actions_b.shape[1])


def _random_2opt(actions: torch.Tensor, num_iters: int, keep_first: bool = True) -> torch.Tensor:
    if actions is None or actions.dim() != 2 or num_iters <= 0:
        return actions
    batch_size, seq_len = actions.shape
    if seq_len < 4 or batch_size == 0:
        return actions
    start_low = 1 if keep_first else 0
    if start_low >= seq_len - 1:
        return actions
    actions_np = actions.detach().cpu().numpy()
    rng = np.random.default_rng()
    for _ in range(num_iters):
        i = rng.integers(start_low, seq_len - 1, size=batch_size)
        j = rng.integers(i + 1, seq_len, size=batch_size)
        for b in range(batch_size):
            a = int(i[b])
            b_idx = int(j[b])
            actions_np[b, a : b_idx + 1] = actions_np[b, a : b_idx + 1][::-1]
    return torch.from_numpy(actions_np).to(device=actions.device)


DEFAULT_GA_POP_SIZE = 50


def _infer_ga_pop_size(
    actions: Optional[torch.Tensor], batch_size: int, default_size: int = DEFAULT_GA_POP_SIZE
) -> int:
    actions_b = _reshape_actions(actions, batch_size, None)
    if actions_b is not None and actions_b.shape[1] > 1:
        return int(actions_b.shape[1])
    return int(default_size)

class EAM(REINFORCE):
    """
        Evolutionary Algorithm Model (EAM)
        Copy of POMO with the following changes:
        - use Evolutionary Algorithm at each end of rollout, then recalculate reward and log_likelihood
        - use the same policy as POMO
        - capable of using AM as baseline
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: str | Callable = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        ea_kwargs: dict = {},
        shared_buffer = None,
        **kwargs,
    ):

        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = AttentionModelPolicy(
                env_name=env.name, **policy_kwargs_with_defaults
            )

        self.baseline_str = baseline

        # Initialize with the shared baseline
        super(EAM, self).__init__(env, policy, baseline, **kwargs)

        self.num_starts = num_starts
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        if baseline == "shared":
            # Add `_multistart` to decode type for train, val and test in policy
            for phase in ["train", "val", "test"]:
                self.set_decode_type_multistart(phase)
        
        if shared_buffer is not None:
            self.shared_buffer = shared_buffer
            self.shared_buffer.set_env(env)
            self.shared_buffer.set_decode_type(self.policy.train_decode_type)
        else:
            self.ea = EA(env, ea_kwargs)
        
        self.ea_prob = ea_kwargs.get("ea_prob")
        self.ea_epoch = ea_kwargs.get("ea_epoch")
        self.improve_mode = ea_kwargs.get("improve_mode", "ga")
        self.random_2opt_iters = ea_kwargs.get("random_2opt_iters")
        self.local_search_max_iterations = ea_kwargs.get("local_search_max_iterations")
        self.local_search_num_threads = ea_kwargs.get("local_search_num_threads")
        self._ga_num_generations = ea_kwargs.get("num_generations", 1)
        self._ga_diag_counter = 0

    def on_train_epoch_start(self):
        self.improve_prob = step_schedule(self.current_epoch, self.ea_prob, self.ea_epoch)

    def _get_improve_iters(self, override: Optional[int]) -> int:
        if override is not None:
            return max(1, int(override))
        return max(1, int(self._ga_num_generations or 1))

    def _apply_random_2opt(self, actions: torch.Tensor, num_iters: int) -> Optional[torch.Tensor]:
        if actions is None:
            return None
        return _random_2opt(actions, num_iters, keep_first=True)

    def _apply_local_search(
        self, actions: torch.Tensor, td: TensorDict, max_iterations: int
    ) -> Optional[torch.Tensor]:
        if actions is None:
            return None
        td_cpu = td.cpu()
        actions_cpu = actions.detach().cpu()
        n_traj = _infer_num_traj(actions_cpu, td.batch_size[0])
        if n_traj > 1:
            td_cpu = batchify(td_cpu, n_traj)
        kwargs = {"max_iterations": max_iterations}
        if self.local_search_num_threads is not None:
            kwargs["num_threads"] = self.local_search_num_threads
        try:
            improved = self.env.local_search(td_cpu, actions_cpu, **kwargs)
        except TypeError:
            kwargs.pop("num_threads", None)
            improved = self.env.local_search(td_cpu, actions_cpu, **kwargs)
        return improved.to(device=actions.device)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        
        td = self.env.reset(batch)
        
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start
            
        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)
        
        # Evaluate policy
        if phase == "train":
            init_td = td.clone()
            original_out = None
            improved_out = None
            t_decode = 0.0
            t_ga = 0.0
            t_diag = 0.0
            
            def run_original_policy():
                nonlocal t_decode
                t0 = time.perf_counter()
                if self.baseline_str == "rollout":
                    result = self.policy(td, self.env, phase=phase, num_starts=1, return_entropy=True)
                else:
                    result = self.policy(td, self.env, phase=phase, num_starts=n_start, return_entropy=True)
                t_decode += time.perf_counter() - t0
                return result
            
            def run_improved_policy(original_actions, td):
                nonlocal t_decode, t_ga
                
                if np.random.random() > self.improve_prob:
                    return None
                
                device = next(self.policy.parameters()).device
                improved_actions = None
                
                if self.improve_mode == "ga":
                    if hasattr(self, "ea"):
                        t0 = time.perf_counter()
                        improved_actions, _ = evolution_worker(
                            original_actions,
                            td,
                            self.ea,
                            self.env,
                        )
                        t_ga += time.perf_counter() - t0
                elif self.improve_mode == "resample":
                    t0 = time.perf_counter()
                    if self.baseline_str == "rollout":
                        result = self.policy(
                            td, self.env, phase=phase, num_starts=1, return_entropy=True
                        )
                    else:
                        result = self.policy(
                            td,
                            self.env,
                            phase=phase,
                            num_starts=n_start,
                            return_entropy=True,
                        )
                    t_decode += time.perf_counter() - t0
                    return result
                elif self.improve_mode == "random_2opt":
                    t0 = time.perf_counter()
                    num_iters = self._get_improve_iters(self.random_2opt_iters)
                    improved_actions = self._apply_random_2opt(original_actions, num_iters)
                    t_ga += time.perf_counter() - t0
                elif self.improve_mode == "local_search":
                    t0 = time.perf_counter()
                    max_iters = self._get_improve_iters(self.local_search_max_iterations)
                    improved_actions = self._apply_local_search(original_actions, td, max_iters)
                    t_ga += time.perf_counter() - t0
                else:
                    raise ValueError(f"Unknown improve_mode: {self.improve_mode}")
                
                if improved_actions is not None:
                    t0 = time.perf_counter()
                    if self.baseline_str == "rollout":
                        result = self.policy(
                            td, 
                            self.env,
                            phase=phase,
                            num_starts=1,
                            actions=improved_actions.to(device=device),
                        )
                        
                        result.update({"actions": improved_actions.to(device=device)})
                    else:
                        result = self.policy(
                            td, 
                            self.env, 
                            phase=phase, 
                            num_starts=n_start, 
                            actions=improved_actions.to(device=device),
                        )
                        if result["actions"].shape[1] < original_actions.shape[1]:
                            padding_size = original_actions.shape[1] - result["actions"].shape[1]
                            result.update({"actions": torch.nn.functional.pad(result["actions"], (0, 0, 0, padding_size))})
                    t_decode += time.perf_counter() - t0

                    return result
                    
                return None
            
            original_out = run_original_policy()
            improved_out = run_improved_policy(original_out["actions"], init_td)

            ga_used = improved_out is not None
            compute_diag = False
            if ga_used:
                self._ga_diag_counter += 1
                compute_diag = self._ga_diag_counter % 10 == 0

            delta_nll = None
            ga_cost_gain = None
            ga_cost_gain_rel = None
            edge_div = None
            order_div = None
            actions = None
            if ga_used:
                actions = improved_out.get("actions", None)
                pop_size = _infer_ga_pop_size(actions, td.batch_size[0])
                mean_base = original_out["reward"].mean()
                mean_improved = improved_out["reward"].mean()
                ga_cost_gain = (mean_improved - mean_base) * pop_size
                denom = mean_base.abs() * pop_size + 1e-8
                ga_cost_gain_rel = ga_cost_gain / denom
            if ga_used and compute_diag:
                t0 = time.perf_counter()
                with torch.no_grad():
                    delta_nll = (
                        (-improved_out["log_likelihood"]).mean()
                        - (-original_out["log_likelihood"]).mean()
                    )
                    actions_b = _reshape_actions(actions, td.batch_size[0], None)
                    edge_div = 0.0
                    order_div = 0.0
                    if actions_b is not None:
                        actions_np = actions_b.detach().cpu().numpy()
                        edge_div = _edge_diversity(actions_np)
                        n_nodes = _infer_num_nodes(self.env)
                        if n_nodes is not None and n_nodes > 1:
                            order_div = _order_diversity(
                                actions_np,
                                n_nodes,
                                self.env.name in DEPOT_ENVS,
                            )
                t_diag += time.perf_counter() - t0

            if self.baseline_str == "rollout":
                # using am as baseline
                original_reward = unbatchify(original_out["reward"], (n_aug, 1))
                original_log_likelihood = unbatchify(original_out["log_likelihood"], (n_aug, 1))
            else:
                # using pomo as baseline 
                original_reward = unbatchify(original_out["reward"], (n_aug, n_start))
                original_log_likelihood = unbatchify(original_out["log_likelihood"], (n_aug, n_start))
            self.calculate_loss(td, batch, original_out, original_reward, original_log_likelihood)
            original_loss = original_out["loss"]
            
            if improved_out is not None:
                if self.baseline_str == "rollout":
                    # using am as baseline
                    improved_reward = unbatchify(improved_out["reward"], (n_aug, 1))
                    improved_log_likelihood = unbatchify(improved_out["log_likelihood"], (n_aug, 1))
                else:
                    improved_reward = unbatchify(improved_out["reward"], (n_aug, n_start))
                    improved_log_likelihood = unbatchify(improved_out["log_likelihood"], (n_aug, n_start))
                
                out = original_out
                combined_out = {
                    k: torch.cat([original_out[k], improved_out[k]], dim=0) 
                    for k in original_out.keys() if k in improved_out and isinstance(original_out[k], torch.Tensor)
                }
                combined_reward = torch.cat([original_reward, improved_reward], dim=0)
                combined_log_likelihood = torch.cat([original_log_likelihood, improved_log_likelihood], dim=0)
                
                batch_size = td.batch_size[0]
                combined_td = TensorDict({}, batch_size=[batch_size*2])
                for k, v in td.items():
                    if isinstance(v, torch.Tensor):
                        combined_td[k] = torch.cat([td[k], init_td[k]], dim=0)
                
                self.calculate_loss(combined_td, batch, combined_out, combined_reward, combined_log_likelihood)
        
                out.update({
                    "loss": combined_out["loss"],
                })
            else:
                out = original_out

            out.update(
                {
                    "t_decode": torch.tensor(t_decode, device=td.device),
                    "t_ga": torch.tensor(t_ga, device=td.device),
                    "t_diag": torch.tensor(t_diag, device=td.device),
                }
            )
            if ga_used and ga_cost_gain is not None:
                out.update(
                    {
                        "ga_cost_gain": ga_cost_gain.detach(),
                        "ga_cost_gain_rel": ga_cost_gain_rel.detach(),
                    }
                )
            if ga_used and compute_diag and delta_nll is not None:
                out.update(
                    {
                        "delta_nll": delta_nll.detach(),
                        "diversity_edge": torch.tensor(edge_div, device=td.device),
                        "diversity_node": torch.tensor(order_div, device=td.device),
                    }
                )
            
        else:
            if self.baseline_str == "rollout":
                # using am as baseline
                out = self.policy(td, self.env, phase=phase, num_starts=1)
            else:
                out = self.policy(td, self.env, phase=phase, num_starts=n_start)
            if phase == "val":
                improved_out = None
                improved_actions = None
                original_actions = out.get("actions", None)
                device = next(self.policy.parameters()).device
                if self.improve_mode == "ga":
                    if hasattr(self, "ea") and original_actions is not None:
                        improved_actions, _ = evolution_worker(
                            original_actions,
                            td,
                            self.ea,
                            self.env,
                        )
                elif self.improve_mode == "resample":
                    if self.baseline_str == "rollout":
                        improved_out = self.policy(
                            td, self.env, phase=phase, num_starts=1
                        )
                    else:
                        improved_out = self.policy(
                            td, self.env, phase=phase, num_starts=n_start
                        )
                elif self.improve_mode == "random_2opt":
                    num_iters = self._get_improve_iters(self.random_2opt_iters)
                    improved_actions = self._apply_random_2opt(original_actions, num_iters)
                elif self.improve_mode == "local_search":
                    max_iters = self._get_improve_iters(self.local_search_max_iterations)
                    improved_actions = self._apply_local_search(original_actions, td, max_iters)
                else:
                    raise ValueError(f"Unknown improve_mode: {self.improve_mode}")

                if improved_out is None and improved_actions is not None:
                    if self.baseline_str == "rollout":
                        improved_out = self.policy(
                            td,
                            self.env,
                            phase=phase,
                            num_starts=1,
                            actions=improved_actions.to(device=device),
                        )
                        improved_out.update(
                            {"actions": improved_actions.to(device=device)}
                        )
                    else:
                        improved_out = self.policy(
                            td,
                            self.env,
                            phase=phase,
                            num_starts=n_start,
                            actions=improved_actions.to(device=device),
                        )
                        if (
                            original_actions is not None
                            and improved_out["actions"].shape[1]
                            < original_actions.shape[1]
                        ):
                            padding_size = (
                                original_actions.shape[1] - improved_out["actions"].shape[1]
                            )
                            improved_out.update(
                                {
                                    "actions": torch.nn.functional.pad(
                                        improved_out["actions"], (0, 0, 0, padding_size)
                                    )
                                }
                            )

                if improved_out is not None:
                    better = improved_out["reward"] > out["reward"]
                    if torch.any(better):
                        out["reward"] = torch.where(
                            better, improved_out["reward"], out["reward"]
                        )
                        if "log_likelihood" in out and "log_likelihood" in improved_out:
                            out["log_likelihood"] = torch.where(
                                better,
                                improved_out["log_likelihood"],
                                out["log_likelihood"],
                            )
                        if (
                            out.get("actions", None) is not None
                            and improved_out.get("actions", None) is not None
                        ):
                            mask = better.view(
                                -1, *([1] * (out["actions"].dim() - 1))
                            )
                            out["actions"] = torch.where(
                                mask, improved_out["actions"], out["actions"]
                            )
            
        out.update({"reward": out["reward"]})
        if self.baseline_str == "shared":
            max_reward, max_idxs = out["reward"].max(dim=-1)
            out.update({"max_reward": max_reward})
        
        if phase != "train" and self.baseline_str == "shared":
            reward = unbatchify(out["reward"], (n_aug, n_start))
            out.update({"reward": reward})
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def on_train_batch_start(self, batch, batch_idx):
        self._t_total_start = time.perf_counter()
        self._t_update_start = None

    def on_before_backward(self, loss):
        if self._t_update_start is None:
            self._t_update_start = time.perf_counter()

    def on_after_optimizer_step(self, optimizer):
        if self._t_update_start is not None:
            t_update = time.perf_counter() - self._t_update_start
            self.log("train/t_update", t_update, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self._t_update_start = None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if getattr(self, "_t_total_start", None) is not None:
            t_total = time.perf_counter() - self._t_total_start
            self.log("train/t_total_raw", t_total, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
    
    def instantiate_metrics(self, metrics: dict):
        """Dictionary of metrics to be logged at each phase"""
        if not metrics:
            log.info("No metrics specified, using default")
        self.train_metrics = metrics.get("train", ["loss", 
                                                   "reward", 
                                                   "max_reward",
                                                   "alpha",
                                                   "rate_mean",
                                                   "rate_std",
                                                   "entropy",
                                                   "delta_nll",
                                                   "diversity_edge",
                                                   "diversity_node",
                                                   "ga_cost_gain",
                                                   "ga_cost_gain_rel",
                                                   "t_decode",
                                                   "t_ga",
                                                   "t_diag"])
        self.val_metrics = metrics.get("val", ["reward", "max_reward", "max_aug_reward"])
        self.test_metrics = metrics.get("test", ["reward", "max_reward", "max_aug_reward"])
        self.log_on_step = metrics.get("log_on_step", True)
        
    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        if bl_val.dim() == 1:
            if bl_val.shape[0] * 2 == reward.shape[0]:
                bl_val = torch.cat([bl_val, bl_val], dim=0)
            else:
                bl_val = bl_val.unsqueeze(1).expand_as(reward)

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        advantage = self.advantage_scaler(advantage)
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out
    
from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.symnco.losses import (
    invariance_loss,
    problem_symmetricity_loss,
    solution_symmetricity_loss,
)
from rl4co.models.zoo.symnco.policy import SymNCOPolicy
from rl4co.utils.ops import gather_by_index, get_num_starts, unbatchify
from rl4co.utils.pylogger import get_pylogger

class SymEAM(REINFORCE):
    """
        Evolutionary Algorithm Model (EAM) using SymNCO as baseline
        Copy of SymNCO with the following changes:
        - use Evolutionary Algorithm at each end of rollout, then recalculate reward and log_likelihood
        - use the same policy as SymNCO
        
    Args:
        env: TorchRL environment to use for the algorithm
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        num_augment: Number of augmentations
        augment_fn: Function to use for augmentation, defaulting to dihedral_8_augmentation
        feats: List of features to augment
        alpha: weight for invariance loss
        beta: weight for solution symmetricity loss
        num_starts: Number of starts for multi-start. If None, use the number of available actions
        **kwargs: Keyword arguments passed to the superclass
    """
    
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, SymNCOPolicy] = None,
        policy_kwargs: dict = {},
        baseline: str = "symnco",
        num_augment: int = 4,
        augment_fn: Union[str, callable] = "symmetric",
        feats: list = None,
        alpha: float = 0.2,
        beta: float = 1,
        num_starts: int = 0,
        ea_kwargs: dict = {},
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy = SymNCOPolicy(env_name=env.name, **policy_kwargs)

        assert baseline == "symnco", "SymNCO only supports custom-symnco baseline"
        baseline = "no"  # Pass no baseline to superclass since there are multiple custom baselines

        # Pass no baseline to superclass since there are multiple custom baselines
        super().__init__(env, policy, baseline, **kwargs)

        self.num_starts = num_starts
        self.num_augment = num_augment
        self.augment = StateAugmentation(
            num_augment=self.num_augment, augment_fn=augment_fn, feats=feats
        )
        self.alpha = alpha  # weight for invariance loss
        self.beta = beta  # weight for solution symmetricity loss

        # Add `_multistart` to decode type for train, val and test in policy if num_starts > 1
        if self.num_starts > 1:
            for phase in ["train", "val", "test"]:
                self.set_decode_type_multistart(phase)
                
        self.ea_prob = ea_kwargs.get("ea_prob")
        self.ea_epoch = ea_kwargs.get("ea_epoch")
        self.ea = EA(env, ea_kwargs)
        self._ga_diag_counter = 0
        
    def on_train_epoch_start(self):
        self.improve_prob = step_schedule(self.current_epoch, self.ea_prob, self.ea_epoch)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = get_num_starts(td, self.env.name) if n_start is None else n_start

        # Symmetric augmentation
        if n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        if phase == "train":
            init_td = td.clone()
            
            original_out = None
            improved_out = None
            t_decode = 0.0
            t_ga = 0.0
            t_diag = 0.0
            
            def run_original_policy():
                nonlocal t_decode
                t0 = time.perf_counter()
                result = self.policy(td,
                                     self.env,
                                     phase=phase,
                                     num_starts=n_start,
                                     return_entropy=True)
                t_decode += time.perf_counter() - t0
                return result
            
            def run_improved_policy(original_actions):
                nonlocal t_decode, t_ga
                td = init_td
                
                if np.random.random() > self.improve_prob:
                    return None
                
                device = next(self.policy.parameters()).device
                improved_actions = None
                
                t0 = time.perf_counter()
                improved_actions, _ = evolution_worker(original_actions, td,
                                                         self.ea, self.env,)
                t_ga += time.perf_counter() - t0
                
                if improved_actions is not None:
                    t0 = time.perf_counter()
                    result = self.policy(td, 
                                         self.env, 
                                         phase=phase, 
                                         num_starts=n_start, 
                                         actions=improved_actions.to(device=device),
                                        )
                    
                    if result["actions"].shape[1] < original_actions.shape[1]:
                        padding_size = original_actions.shape[1] - result["actions"].shape[1]
                        result.update({"actions": torch.nn.functional.pad(result["actions"], (0, 0, 0, padding_size))})
                    t_decode += time.perf_counter() - t0
                
                    return result
                
                return None
            
            original_out = run_original_policy()
            improved_out = run_improved_policy(original_out["actions"])
            ga_used = improved_out is not None
            compute_diag = False
            if ga_used:
                self._ga_diag_counter += 1
                compute_diag = self._ga_diag_counter % 10 == 0

            delta_nll = None
            ga_cost_gain = None
            ga_cost_gain_rel = None
            edge_div = None
            order_div = None
            if ga_used:
                actions = improved_out.get("actions", None)
                pop_size = _infer_ga_pop_size(actions, td.batch_size[0])
                mean_base = original_out["reward"].mean()
                mean_improved = improved_out["reward"].mean()
                ga_cost_gain = (mean_improved - mean_base) * pop_size
                denom = mean_base.abs() * pop_size + 1e-8
                ga_cost_gain_rel = ga_cost_gain / denom
            if ga_used and compute_diag:
                t0 = time.perf_counter()
                with torch.no_grad():
                    delta_nll = (
                        (-improved_out["log_likelihood"]).mean()
                        - (-original_out["log_likelihood"]).mean()
                    )
                    actions = improved_out.get("actions", None)
                    actions_b = _reshape_actions(actions, td.batch_size[0], None)
                    edge_div = 0.0
                    order_div = 0.0
                    if actions_b is not None:
                        actions_np = actions_b.detach().cpu().numpy()
                        edge_div = _edge_diversity(actions_np)
                        n_nodes = _infer_num_nodes(self.env)
                        if n_nodes is not None and n_nodes > 1:
                            order_div = _order_diversity(
                                actions_np,
                                n_nodes,
                                self.env.name in DEPOT_ENVS,
                            )
                t_diag += time.perf_counter() - t0
            
            out = original_out
            
            if improved_out is not None:
                original_reward = unbatchify(original_out["reward"], (n_aug, n_start))
                original_log_likelihood = unbatchify(original_out["log_likelihood"], (n_aug, n_start))
                
                improved_reward = unbatchify(improved_out["reward"], (n_aug, n_start))
                improved_log_likelihood = unbatchify(improved_out["log_likelihood"], (n_aug, n_start))

                keys_to_merge = ["reward", "log_likelihood", "actions", "proj_embeddings"]
                combined_out = {
                    k: torch.cat([original_out[k], improved_out[k]], dim=0)
                    for k in keys_to_merge if k in original_out and k in improved_out
                }
                combined_reward = torch.cat([original_reward, improved_reward], dim=0)
                combined_log_likelihood = torch.cat([original_log_likelihood, improved_log_likelihood], dim=0)

                batch_size = td.batch_size[0]
                combined_td = TensorDict({}, batch_size=[batch_size*2])
                for k in td.keys():
                    if isinstance(td[k], torch.Tensor):
                        combined_td[k] = torch.cat([td[k], init_td[k]], dim=0)

                loss_ps = problem_symmetricity_loss(combined_reward, combined_log_likelihood) if n_start > 1 else 0
                loss_ss = solution_symmetricity_loss(combined_reward, combined_log_likelihood) if n_aug > 1 else 0
                loss_inv = invariance_loss(combined_out["proj_embeddings"], n_aug) if n_aug > 1 else 0
                loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv

                out.update({
                    "loss": loss,
                    "loss_ss": loss_ss,
                    "loss_ps": loss_ps,
                    "loss_inv": loss_inv,
                })
            else:
                original_reward = unbatchify(original_out["reward"], (n_aug, n_start))
                original_log_likelihood = unbatchify(original_out["log_likelihood"], (n_aug, n_start))
                
                loss_ps = problem_symmetricity_loss(original_reward, original_log_likelihood) if n_start > 1 else 0
                loss_ss = solution_symmetricity_loss(original_reward, original_log_likelihood) if n_aug > 1 else 0
                loss_inv = invariance_loss(out["proj_embeddings"], n_aug) if n_aug > 1 else 0
                
                loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv
                
                out.update(
                    {
                        "loss": loss,
                        "loss_ss": loss_ss,
                        "loss_ps": loss_ps,
                        "loss_inv": loss_inv,
                    }
                )

            out.update(
                {
                    "t_decode": torch.tensor(t_decode, device=td.device),
                    "t_ga": torch.tensor(t_ga, device=td.device),
                    "t_diag": torch.tensor(t_diag, device=td.device),
                }
            )
            if ga_used and ga_cost_gain is not None:
                out.update(
                    {
                        "ga_cost_gain": ga_cost_gain.detach(),
                        "ga_cost_gain_rel": ga_cost_gain_rel.detach(),
                    }
                )
            if ga_used and compute_diag and delta_nll is not None:
                out.update(
                    {
                        "delta_nll": delta_nll.detach(),
                        "diversity_edge": torch.tensor(edge_div, device=td.device),
                        "diversity_node": torch.tensor(order_div, device=td.device),
                    }
                )
        else:
            out = self.policy(td, self.env, phase=phase, num_starts=n_start)
            
            reward = unbatchify(out["reward"], (n_start, n_aug))
            
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=1)
                out.update({"max_reward": max_reward})

                # Reshape batch to [batch, n_start, n_aug]
                if out.get("actions", None) is not None:
                    actions = unbatchify(out["actions"], (n_start, n_aug))
                    out.update(
                        {"best_multistart_actions": gather_by_index(actions, max_idxs)}
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})
                if out.get("best_multistart_actions", None) is not None:
                    out.update(
                        {
                            "best_aug_actions": gather_by_index(
                                out["best_multistart_actions"], max_idxs
                            )
                        }
                    )
            
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def on_train_batch_start(self, batch, batch_idx):
        self._t_total_start = time.perf_counter()
        self._t_update_start = None

    def on_before_backward(self, loss):
        if self._t_update_start is None:
            self._t_update_start = time.perf_counter()

    def on_after_optimizer_step(self, optimizer):
        if self._t_update_start is not None:
            t_update = time.perf_counter() - self._t_update_start
            self.log("train/t_update", t_update, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self._t_update_start = None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if getattr(self, "_t_total_start", None) is not None:
            t_total = time.perf_counter() - self._t_total_start
            self.log("train/t_total_raw", t_total, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

    def instantiate_metrics(self, metrics: dict):
        """Dictionary of metrics to be logged at each phase"""
        if not metrics:
            log.info("No metrics specified, using default")
        self.train_metrics = metrics.get(
            "train",
            [
                "loss",
                "reward",
                "max_reward",
                "alpha",
                "rate_mean",
                "rate_std",
                "entropy",
                "delta_nll",
                "diversity_edge",
                "diversity_node",
                "ga_cost_gain",
                "ga_cost_gain_rel",
                "t_decode",
                "t_ga",
                "t_diag",
            ],
        )
        self.val_metrics = metrics.get("val", ["reward", "max_reward", "max_aug_reward"])
        self.test_metrics = metrics.get("test", ["reward", "max_reward", "max_aug_reward"])
        self.log_on_step = metrics.get("log_on_step", True)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.matnet.policy import MatNetPolicy, MultiStageFFSPPolicy
from rl4co.utils.pylogger import get_pylogger

def select_matnet_policy(env, **policy_params):
    if env.name == "ffsp":
        if env.flatten_stages:
            return MatNetPolicy(env_name=env.name, **policy_params)
        else:
            return MultiStageFFSPPolicy(stage_cnt=env.num_stage, **policy_params)
    else:
        return MatNetPolicy(env_name=env.name, **policy_params)


class MatNetEAM(EAM):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module | MatNetPolicy = None,
        num_starts: int = None,
        policy_params: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = select_matnet_policy(env=env, **policy_params)

        # Check if using augmentation and the validation of augmentation function
        if kwargs.get("num_augment", 0) != 0:
            log.warning("MatNet is using augmentation.")
            if (
                kwargs.get("augment_fn") in ["symmetric", "dihedral8"]
                or kwargs.get("augment_fn") is None
            ):
                log.error(
                    "MatNet does not use symmetric or dihedral augmentation. Seeting no augmentation function."
                )
                kwargs["num_augment"] = 0
        else:
            kwargs["num_augment"] = 0

        super(MatNetEAM, self).__init__(
            env=env,
            policy=policy,
            num_starts=num_starts,
            baseline="shared",
            **kwargs,
        )
