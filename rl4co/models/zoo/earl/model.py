from typing import IO, Any, Callable, Optional, Union, cast

import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.earl.evolution import evolution_worker, EA
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

import numpy as np

from tensordict import TensorDict
import torch

log = get_pylogger(__name__)

def evolve_prob_schedule(epoch, max_epoch, initial_prob, final_prob):
    return np.cos(np.pi * epoch / max_epoch) * (final_prob - initial_prob) + initial_prob

def sigmoid_schedule(epoch, max_epoch, initial_prob, final_prob):
    x = 10 * (epoch / max_epoch - 0.5)
    sigmoid = 1 / (1 + np.exp(-x))
    return initial_prob + (final_prob - initial_prob) * sigmoid

def step_schedule(epoch, ea_prob, ea_epoch):
    return ea_prob if (epoch <= ea_epoch or ea_epoch < 0)else 0.0


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
        ea_kwargs: dict | None = None,
        shared_buffer = None,
        **kwargs,
    ):
        ea_kwargs = {} if ea_kwargs is None else dict(ea_kwargs)
        ea_kwargs.setdefault("num_generations", 3)
        ea_kwargs.setdefault("mutation_rate", 0.1)
        ea_kwargs.setdefault("crossover_rate", 0.6)
        ea_kwargs.setdefault("selection_rate", 0.2)
        ea_kwargs.setdefault("alpha", 0.5)
        ea_kwargs.setdefault("beta", 3.0)
        ea_kwargs.setdefault("ea_prob", 1.0)
        ea_kwargs.setdefault("ea_epoch", -1)

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
        
        self.ea_prob = float(ea_kwargs.get("ea_prob"))
        self.ea_epoch = int(ea_kwargs.get("ea_epoch"))
        self.val_improve = ea_kwargs.get("val_improve", True)
        self.val_improve_prob = ea_kwargs.get("val_improve_prob", 1.0)
        self.val_num_generations = ea_kwargs.get("val_num_generations")
        self._ga_num_generations = ea_kwargs.get("num_generations", 1)

    def on_train_epoch_start(self):
        self.improve_prob = step_schedule(self.current_epoch, self.ea_prob, self.ea_epoch)

    def _align_improved_actions(
        self, improved_actions: Optional[torch.Tensor], original_actions: Optional[torch.Tensor]
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

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        
        td = self.env.reset(batch)
        init_td = td.clone()
        
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start
        raw_val_reward = None
            
        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)
        
        # Evaluate policy
        if phase == "train":
            original_out = None
            improved_out = None
            
            def run_original_policy():
                if self.baseline_str == "rollout":
                    result = self.policy(td, self.env, phase=phase, num_starts=1, return_entropy=True)
                else:
                    result = self.policy(td, self.env, phase=phase, num_starts=n_start, return_entropy=True)
                return result
            
            def run_improved_policy(original_actions, td):
                if np.random.random() > self.improve_prob:
                    return None
                
                device = next(self.policy.parameters()).device
                improved_actions = None
                
                if hasattr(self, "ea"):
                    improved_actions, _ = evolution_worker(
                        original_actions,
                        td,
                        self.ea,
                        self.env,
                    )
                
                if improved_actions is not None:
                    improved_actions = improved_actions.to(device=device)
                    improved_actions = self._align_improved_actions(
                        improved_actions, original_actions
                    )
                    if self.baseline_str == "rollout":
                        result = self.policy(
                            td, 
                            self.env,
                            phase=phase,
                            num_starts=1,
                            actions=improved_actions,
                        )
                        
                        result.update({"actions": improved_actions})
                    else:
                        result = self.policy(
                            td, 
                            self.env, 
                            phase=phase, 
                            num_starts=n_start, 
                            actions=improved_actions,
                        )
                        if result["actions"].shape[1] < original_actions.shape[1]:
                            padding_size = original_actions.shape[1] - result["actions"].shape[1]
                            result.update({"actions": torch.nn.functional.pad(result["actions"], (0, padding_size))})

                    return result
                    
                return None
            
            original_out = run_original_policy()
            improved_out = run_improved_policy(original_out["actions"], init_td)

            if self.baseline_str == "rollout":
                # using am as baseline
                original_reward = unbatchify(original_out["reward"], (n_aug, 1))
                original_log_likelihood = unbatchify(original_out["log_likelihood"], (n_aug, 1))
            else:
                # using pomo as baseline 
                original_reward = unbatchify(original_out["reward"], (n_aug, n_start))
                original_log_likelihood = unbatchify(original_out["log_likelihood"], (n_aug, n_start))
            self.calculate_loss(td, batch, original_out, original_reward, original_log_likelihood)
            
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
            
        else:
            if self.baseline_str == "rollout":
                # using am as baseline
                out = self.policy(td, self.env, phase=phase, num_starts=1)
            else:
                out = self.policy(td, self.env, phase=phase, num_starts=n_start)
            if phase == "val":
                raw_val_reward = out.get("reward", None)
                improved_out = None
                if self.val_improve and (
                    self.val_improve_prob is None
                    or np.random.random() <= self.val_improve_prob
                ):
                    improved_actions = None
                    original_actions = out.get("actions", None)
                    device = next(self.policy.parameters()).device
                    if hasattr(self, "ea") and original_actions is not None:
                        prev_num_generations = None
                        if self.val_num_generations is not None:
                            prev_num_generations = self.ea.num_generations
                            self.ea.num_generations = self.val_num_generations
                        try:
                            improved_actions, _ = evolution_worker(
                                original_actions,
                                init_td,
                                self.ea,
                                self.env,
                            )
                        finally:
                            if prev_num_generations is not None:
                                self.ea.num_generations = prev_num_generations

                    if improved_out is None and improved_actions is not None:
                        improved_actions = improved_actions.to(device=device)
                        improved_actions = self._align_improved_actions(
                            improved_actions, original_actions
                        )
                        if self.baseline_str == "rollout":
                            improved_out = self.policy(
                                init_td,
                                self.env,
                                phase=phase,
                                num_starts=1,
                                actions=improved_actions,
                            )
                            improved_out.update({"actions": improved_actions})
                        else:
                            improved_out = self.policy(
                                init_td,
                                self.env,
                                phase=phase,
                                num_starts=n_start,
                                actions=improved_actions,
                            )
                            if (
                                original_actions is not None
                                and improved_out["actions"].shape[1]
                                < original_actions.shape[1]
                            ):
                                padding_size = (
                                    original_actions.shape[1]
                                    - improved_out["actions"].shape[1]
                                )
                                improved_out.update(
                                    {
                                        "actions": torch.nn.functional.pad(
                                            improved_out["actions"], (0, padding_size)
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

        if phase == "val" and raw_val_reward is not None:
            if self.baseline_str == "shared":
                raw_reward = unbatchify(raw_val_reward, (n_aug, n_start))
                out.update({"reward_no_ls": raw_reward})
                raw_reward_ = raw_reward
                if n_start > 1:
                    raw_max_reward, _ = raw_reward.max(dim=-1)
                    out.update({"max_reward_no_ls": raw_max_reward})
                    raw_reward_ = raw_max_reward
                if n_aug > 1:
                    raw_max_aug_reward, _ = raw_reward_.max(dim=1)
                    out.update({"max_aug_reward_no_ls": raw_max_aug_reward})
            else:
                out.update({"reward_no_ls": raw_val_reward})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def instantiate_metrics(self, metrics: dict):
        """Dictionary of metrics to be logged at each phase"""
        if not metrics:
            log.info("No metrics specified, using default")
        self.train_metrics = metrics.get(
            "train", ["loss", "reward", "max_reward", "alpha", "rate_mean", "rate_std", "entropy"]
        )
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
            
            def run_original_policy():
                result = self.policy(td,
                                     self.env,
                                     phase=phase,
                                     num_starts=n_start,
                                     return_entropy=True)
                return result
            
            def run_improved_policy(original_actions):
                td = init_td
                
                if np.random.random() > self.improve_prob:
                    return None
                
                device = next(self.policy.parameters()).device
                improved_actions = None
                
                improved_actions, _ = evolution_worker(
                    original_actions,
                    td,
                    self.ea,
                    self.env,
                )
                
                if improved_actions is not None:
                    result = self.policy(td, 
                                         self.env, 
                                         phase=phase, 
                                         num_starts=n_start, 
                                         actions=improved_actions.to(device=device),
                                        )
                    
                    if result["actions"].shape[1] < original_actions.shape[1]:
                        padding_size = original_actions.shape[1] - result["actions"].shape[1]
                        result.update({"actions": torch.nn.functional.pad(result["actions"], (0, padding_size))})
                
                    return result
                
                return None
            
            original_out = run_original_policy()
            improved_out = run_improved_policy(original_out["actions"])
            
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

    def instantiate_metrics(self, metrics: dict):
        """Dictionary of metrics to be logged at each phase"""
        if not metrics:
            log.info("No metrics specified, using default")
        self.train_metrics = metrics.get(
            "train",
            ["loss", "reward", "max_reward", "alpha", "rate_mean", "rate_std", "entropy"],
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
