from typing import IO, Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.core.saving import _load_from_checkpoint
from tensordict import TensorDict
from typing_extensions import Self

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.common.utils import RewardScaler
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline, get_reinforce_baseline
from rl4co.utils.lightning import get_lightning_device
from rl4co.utils.ops import unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCE(RL4COLitModule):
    """REINFORCE algorithm, also known as policy gradients.
    See superclass `RL4COLitModule` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline
        baseline_kwargs: Keyword arguments for baseline. Ignored if baseline is not a string
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: REINFORCEBaseline | str = "rollout",
        baseline_kwargs: dict = {},
        reward_scale: str = None,
        loss_mode: str = "rl",
        **kwargs,
    ):
        super().__init__(env, policy, **kwargs)

        self.save_hyperparameters(logger=False)

        if baseline == "critic":
            log.warning(
                "Using critic as baseline. If you want more granular support, use the A2C module instead."
            )

        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline
        self.advantage_scaler = RewardScaler(reward_scale)
        self.loss_mode = loss_mode.lower()
        if self.loss_mode not in {"rl", "po", "bopo"}:
            raise ValueError(
                f"Unknown loss_mode='{loss_mode}'. Supported modes are: rl, po, bopo"
            )

    @staticmethod
    def _reshape_instance_rollouts(
        tensor: torch.Tensor, batch_size: int, num_starts: int
    ) -> torch.Tensor:
        """Reshape rollout tensor into [D, B, ...] without crossing instance boundaries."""
        if tensor.dim() >= 2 and tensor.size(0) == batch_size and tensor.size(1) == num_starts:
            return tensor
        if tensor.size(0) != batch_size * num_starts:
            raise ValueError(
                f"Expected rollout tensor first dim to be D*B={batch_size * num_starts}, got {tensor.size(0)}"
            )
        return unbatchify(tensor, num_starts)

    @staticmethod
    def _po_instance_loss(reward_db: torch.Tensor, ll_db: torch.Tensor) -> torch.Tensor:
        """PO full pair builder within each instance (no cross-instance pairs)."""
        d, b = reward_db.shape
        tri_mask = torch.triu(
            torch.ones(b, b, dtype=torch.bool, device=reward_db.device), diagonal=1
        )
        per_instance_losses = []
        for i in range(d):
            r = reward_db[i]
            ll = ll_db[i]
            r_diff = r.unsqueeze(1) - r.unsqueeze(0)
            ll_diff = ll.unsqueeze(1) - ll.unsqueeze(0)
            pref = torch.sign(r_diff[tri_mask])
            # Skip ties in reward as they do not induce a strict preference pair.
            valid = pref != 0
            if valid.any():
                pair_loss = -F.logsigmoid(pref[valid] * ll_diff[tri_mask][valid])
                per_instance_losses.append(pair_loss.mean())
            else:
                per_instance_losses.append(torch.zeros((), device=reward_db.device))
        return torch.stack(per_instance_losses, dim=0).mean()

    @staticmethod
    def _bopo_instance_loss(reward_db: torch.Tensor, ll_db: torch.Tensor) -> torch.Tensor:
        """BOPO anchored pairing within each instance (no cross-instance pairs)."""
        d, b = reward_db.shape
        per_instance_losses = []
        arange_b = torch.arange(b, device=reward_db.device)
        for i in range(d):
            r = reward_db[i]
            ll = ll_db[i]
            anchor_idx = torch.argmax(r)
            mask = arange_b != anchor_idx
            other_r = r[mask]
            other_ll = ll[mask]
            if other_r.numel() == 0:
                per_instance_losses.append(torch.zeros((), device=reward_db.device))
                continue

            # BOPO filtering remains strictly instance-local.
            pref = torch.sign(r[anchor_idx] - other_r)
            valid = pref != 0
            if valid.any():
                loss_i = -F.logsigmoid(pref[valid] * (ll[anchor_idx] - other_ll[valid]))
                per_instance_losses.append(loss_i.mean())
            else:
                per_instance_losses.append(torch.zeros((), device=reward_db.device))
        return torch.stack(per_instance_losses, dim=0).mean()

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        # Useful for bucketed JSSP training: log current batch shape metadata.
        if "start_op_per_job" in td.keys() and "proc_times" in td.keys():
            num_jobs = float(td["start_op_per_job"].shape[-1])
            num_machines = float(td["proc_times"].shape[-2])
            self.log(
                f"{phase}/batch_num_jobs",
                num_jobs,
                on_step=phase == "train",
                on_epoch=phase != "train",
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{phase}/batch_num_machines",
                num_machines,
                on_step=phase == "train",
                on_epoch=phase != "train",
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase, select_best=phase != "train")

        # Compute loss
        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

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

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        advantage = self.advantage_scaler(advantage)
        reinforce_term = advantage * log_likelihood
        num_starts = policy_out.get("num_starts", 0)
        base_batch_size = policy_out.get("batch_size", None)

        # For batched multi-rollout RL/PO/BOPO (D instances x B rollouts), keep
        # candidate pools instance-local: [D, B, ...].
        if isinstance(num_starts, int) and num_starts > 1 and isinstance(base_batch_size, int):
            reward_db = self._reshape_instance_rollouts(reward, base_batch_size, num_starts)
            ll_db = self._reshape_instance_rollouts(
                log_likelihood, base_batch_size, num_starts
            )
            term_db = self._reshape_instance_rollouts(
                reinforce_term, base_batch_size, num_starts
            )

            # TODO: PO/BOPO for >2 rollout dims (e.g., augmentation dimensions) is not
            # safely batchized yet. Keep current stage restricted to [D, B].
            if reward_db.dim() != 2 or ll_db.dim() != 2 or term_db.dim() != 2:
                raise NotImplementedError(
                    "PO/BOPO same-shape batching currently supports 2D [D, B] rollout tensors only"
                )

            if self.loss_mode == "rl":
                reinforce_loss = -term_db.mean(dim=1).mean()
            elif self.loss_mode == "po":
                reinforce_loss = self._po_instance_loss(reward_db, ll_db)
            else:  # bopo
                reinforce_loss = self._bopo_instance_loss(reward_db, ll_db)
        else:
            if self.loss_mode != "rl":
                raise NotImplementedError(
                    "PO/BOPO requires num_starts > 1 to form per-instance candidate pools"
                )
            reinforce_loss = -reinforce_term.mean()
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

    def post_setup_hook(self, stage="fit"):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.data_cfg["val_data_size"],
        )

    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.data_cfg["val_data_size"],
        )
        # Need to call super() for the dataset to be reset
        super().on_train_epoch_end()

    def wrap_dataset(self, dataset):
        """Wrap dataset from baseline evaluation. Used in greedy rollout baseline"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
        )

    def set_decode_type_multistart(self, phase: str):
        """Set decode type to `multistart` for train, val and test in policy.
        For example, if the decode type is `greedy`, it will be set to `multistart_greedy`.

        Args:
            phase: Phase to set decode type for. Must be one of `train`, `val` or `test`.
        """
        attribute = f"{phase}_decode_type"
        attr_get = getattr(self.policy, attribute)
        # If does not exist, log error
        if attr_get is None:
            log.error(f"Decode type for {phase} is None. Cannot prepend `multistart_`.")
            return
        elif "multistart" in attr_get:
            return
        else:
            setattr(self.policy, attribute, f"multistart_{attr_get}")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: _PATH | IO,
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = False,
        load_baseline: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Load model from checkpoint/

        Note:
            This is a modified version of `load_from_checkpoint` from `pytorch_lightning.core.saving`.
            It deals with matching keys for the baseline by first running setup
        """

        if strict:
            log.warning("Setting strict=False for loading model from checkpoint.")
            strict = False

        # Do not use strict
        loaded = _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )

        # Load baseline state dict
        if load_baseline:
            # setup baseline first
            loaded.setup()
            loaded.post_setup_hook()
            # load baseline state dict
            state_dict = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )["state_dict"]
            # get only baseline parameters
            state_dict = {k: v for k, v in state_dict.items() if "baseline" in k}
            state_dict = {k.replace("baseline.", "", 1): v for k, v in state_dict.items()}
            loaded.baseline.load_state_dict(state_dict)

        return cast(Self, loaded)
