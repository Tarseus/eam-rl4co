from typing import Any, Callable

import json
from pathlib import Path

import torch
import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.free_loss import compile_free_loss, ir_from_json
from rl4co.models.rl.reinforce.preference_losses import pl_loss, po_loss
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class POMO(REINFORCE):
    """POMO Model for neural combinatorial optimization based on REINFORCE
    Based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011.

    Note:
        If no policy kwargs is passed, we use the Attention Model policy with the following arguments:
        Differently to the base class:
        - `num_encoder_layers=6` (instead of 3)
        - `normalization="instance"` (instead of "batch")
        - `use_graph_context=False` (instead of True)
        The latter is due to the fact that the paper does not use the graph context in the policy, which seems to be
        helpful in overfitting to the training graph size.

    Args:
        env: TorchRL Environment
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        baseline: Baseline to use for the algorithm. Note that POMO only supports shared baseline,
            so we will throw an error if anything else is passed.
        num_augment: Number of augmentations (used only for validation and test)
        augment_fn: Function to use for augmentation, defaulting to dihedral8
        first_aug_identity: Whether to include the identity augmentation in the first position
        feats: List of features to augment
        num_starts: Number of starts for multi-start. If None, use the number of available actions
        loss_type: Loss type to use. One of {"rl_loss", "po_loss", "pl_loss", "free_loss"}.
        alpha: Scaling factor for log-likelihood in preference losses.
        loss_kwargs: Optional keyword args reserved for preference losses.
        pl_impl: Implementation choice for listwise loss, {"ptp", "stable"}.
        free_loss_ir_json_path: Path to JSON IR for free_loss (required if loss_type="free_loss").
        **kwargs: Keyword arguments passed to the superclass
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
        loss_type: str = "rl_loss",
        alpha: float = 1.0,
        loss_kwargs: dict | None = None,
        pl_impl: str = "stable",
        free_loss_ir_json_path: str | None = None,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

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

        assert baseline == "shared", "POMO only supports shared baseline"

        # Initialize with the shared baseline
        super(POMO, self).__init__(env, policy, baseline, **kwargs)

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

        # Add `_multistart` to decode type for train, val and test in policy
        for phase in ["train", "val", "test"]:
            self.set_decode_type_multistart(phase)

        self.loss_type = loss_type
        self.alpha = float(alpha)
        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.pl_impl = pl_impl
        self.free_loss_ir_json_path = free_loss_ir_json_path
        self.free_loss = None
        if self.loss_type == "free_loss":
            self._load_free_loss()
        if self.loss_kwargs:
            log.warning(
                "loss_kwargs is currently unused and will be ignored: %s",
                self.loss_kwargs,
            )

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
        out = self.policy(td, self.env, phase=phase, num_starts=n_start)

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
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

    def calculate_loss(
        self,
        td,
        batch,
        policy_out: dict,
        reward: torch.Tensor | None = None,
        log_likelihood: torch.Tensor | None = None,
    ):
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        if self.loss_type == "rl_loss":
            return super().calculate_loss(td, batch, policy_out, reward, log_likelihood)
        if self.loss_type == "po_loss":
            loss, pref_rate = po_loss(reward, log_likelihood, alpha=self.alpha)
            policy_out.update(
                {
                    "loss": loss,
                    "po_loss": loss.detach(),
                    "po_pref_rate": pref_rate.detach(),
                }
            )
            return policy_out
        if self.loss_type == "pl_loss":
            loss = pl_loss(
                reward,
                log_likelihood,
                alpha=self.alpha,
                impl=self.pl_impl,
            )
            policy_out.update({"loss": loss, "pl_loss": loss.detach()})
            return policy_out
        if self.loss_type == "free_loss":
            loss, pair_count = self._free_loss_loss_fn(reward, log_likelihood)
            policy_out.update(
                {
                    "loss": loss,
                    "free_loss": loss.detach(),
                    "free_loss_pair_count": pair_count,
                }
            )
            return policy_out

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _load_free_loss(self) -> None:
        if self.free_loss_ir_json_path is None:
            raise ValueError(
                "When loss_type is 'free_loss', free_loss_ir_json_path must be set."
            )
        path = Path(self.free_loss_ir_json_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"free_loss_ir_json_path does not exist: {path.as_posix()}"
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        ir_obj = payload.get("ir", payload)
        ir = ir_from_json(ir_obj)
        self.free_loss = compile_free_loss(ir)

    def _free_loss_loss_fn(
        self, reward: torch.Tensor, log_likelihood: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.free_loss is None:
            raise RuntimeError(
                "free_loss is not compiled; check free_loss_ir_json_path."
            )

        objective = -reward
        mask = objective[:, :, None] < objective[:, None, :]
        b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
        pair_count = torch.tensor(
            float(b_idx.numel()), device=reward.device, dtype=reward.dtype
        )

        if b_idx.numel() == 0:
            advantage = reward - reward.float().mean(dim=1, keepdim=True)
            loss = -(advantage * log_likelihood).mean()
            return loss, pair_count

        cost_a = objective[b_idx, winner_idx]
        cost_b = objective[b_idx, loser_idx]
        logp_w = log_likelihood[b_idx, winner_idx]
        logp_l = log_likelihood[b_idx, loser_idx]
        weight = torch.ones_like(cost_a)
        batch = {
            "cost_a": cost_a,
            "cost_b": cost_b,
            "log_prob_w": logp_w,
            "log_prob_l": logp_l,
            "weight": weight,
        }
        loss = self.free_loss.loss_fn(batch=batch, model_output={}, extra={"alpha": self.alpha})
        return loss, pair_count
