from typing import Any, Callable, Sequence

import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.free_loss import compile_free_loss, ir_from_json
from rl4co.models.rl.reinforce.preference_losses import (
    bopo_loss,
    pl_loss,
    po_loss,
    slim_loss,
    sll_loss,
)
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.pomo.po4cops_cvrp_policy import PO4COPsCVRPPolicy
from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

_DEFAULT_FREE_LOSS_OBSERVABLES = ("seq_len", "log_prob_mean", "advantage")


def _normalize_free_loss_observables(observables: Sequence[str] | None) -> tuple[str, ...]:
    values = observables if observables else _DEFAULT_FREE_LOSS_OBSERVABLES
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        key = str(raw or "").strip()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return tuple(out) if out else _DEFAULT_FREE_LOSS_OBSERVABLES


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
        loss_type: Loss type to use. One of {"rl_loss", "po_loss", "pl_loss", "bopo_loss", "slim_loss", "sll_loss", "free_loss"}.
        alpha: Scaling factor for log-likelihood in preference losses.
        po_impl: Implementation choice for pairwise preference loss, {"bt", "exponential"}.
        loss_kwargs: Optional keyword args reserved for preference losses.
        pl_impl: Implementation choice for listwise loss, {"ptp", "stable"}.
        bopo_pair_mode: Pairing mode for BOPO loss, {"anchor_best", "all_pairs"}.
        bopo_select_strategy: Selection strategy for BOPO loss, {"paper", "top_k", "quantile"}.
        bopo_select_k: Number of top solutions to select for BOPO loss.
        bopo_select_quantile: Quantile threshold for BOPO loss.
        sll_impl: Implementation variant for SLL/SLIM loss, {"sll", "slim", "listnet"}.
        sll_temperature: Temperature for SLL/SLIM loss softmax.
        free_loss_ir_json_path: Path to JSON IR for free_loss (required if loss_type="free_loss").
        pref_builder_ir_json_path: Optional path to a preference-builder JSON artifact.
        pref_pair_json_path: Optional path to a co-evolution `best_pair.json`; when set, the model
            resolves sibling `best_builder.json` and `best_loss.json` artifacts automatically.
        pref_builder_kwargs: Optional extra kwargs forwarded to `generated_builder(..., extra)`.
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
        po_impl: str = "bt",
        loss_kwargs: dict | None = None,
        pl_impl: str = "stable",
        bopo_pair_mode: str = "anchor_best",
        bopo_select_strategy: str = "paper",
        bopo_select_k: int | None = None,
        bopo_select_quantile: float = 0.5,
        sll_impl: str = "sll",
        sll_temperature: float = 1.0,
        free_loss_ir_json_path: str | None = None,
        pref_builder_ir_json_path: str | None = None,
        pref_pair_json_path: str | None = None,
        pref_builder_kwargs: dict | None = None,
        free_loss_observables: Sequence[str] | None = None,
        preference_po_anchor_weight: float = 0.0,
        preference_po_anchor_alpha: float = 0.05,
        memory_efficient_preference: bool = False,
        memory_efficient_checkpoint_encoder: bool = True,
        memory_efficient_checkpoint_decoder: bool = True,
        memory_efficient_verify_replay: bool = False,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            use_po4cops_compat = bool(policy_kwargs.pop("po4cops_compat", False))
            if use_po4cops_compat:
                policy_kwargs_with_defaults = {
                    "embedding_dim": policy_kwargs.pop("embed_dim", 128),
                    "encoder_layer_num": policy_kwargs.pop("num_encoder_layers", 6),
                    "decoder_layer_num": policy_kwargs.pop("decoder_layer_num", 1),
                    "qkv_dim": policy_kwargs.pop("qkv_dim", 16),
                    "head_num": policy_kwargs.pop("num_heads", 8),
                    "ff_hidden_dim": policy_kwargs.pop("feedforward_hidden", 512),
                    "logit_clipping": policy_kwargs.pop("tanh_clipping", 50),
                    "eval_type": policy_kwargs.pop("eval_type", "argmax"),
                    "env_name": env.name,
                }
                policy_kwargs_with_defaults.update(policy_kwargs)
                if env.name == "tsp":
                    policy = PO4COPsTSPPolicy(**policy_kwargs_with_defaults)
                elif env.name == "cvrp":
                    policy = PO4COPsCVRPPolicy(**policy_kwargs_with_defaults)
                else:
                    raise ValueError(
                        f"po4cops_compat currently supports only tsp/cvrp, got: {env.name}"
                    )
            else:
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
        self.po_impl = po_impl
        self.loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
        self.pl_impl = pl_impl
        self.bopo_pair_mode = bopo_pair_mode
        self.bopo_select_strategy = bopo_select_strategy
        self.bopo_select_k = bopo_select_k
        self.bopo_select_quantile = float(bopo_select_quantile)
        self.sll_impl = sll_impl
        self.sll_temperature = float(sll_temperature)
        self.free_loss_ir_json_path = free_loss_ir_json_path
        self.pref_builder_ir_json_path = pref_builder_ir_json_path
        self.pref_pair_json_path = pref_pair_json_path
        self.pref_builder_kwargs = {} if pref_builder_kwargs is None else dict(pref_builder_kwargs)
        self.free_loss_observables = _normalize_free_loss_observables(free_loss_observables)
        self.preference_po_anchor_weight = float(preference_po_anchor_weight)
        self.preference_po_anchor_alpha = float(preference_po_anchor_alpha)
        if not 0.0 <= self.preference_po_anchor_weight <= 1.0:
            raise ValueError("preference_po_anchor_weight must be in [0, 1]")
        if self.preference_po_anchor_alpha <= 0.0:
            raise ValueError("preference_po_anchor_alpha must be > 0")
        self.memory_efficient_preference = bool(memory_efficient_preference)
        self.memory_efficient_checkpoint_encoder = bool(memory_efficient_checkpoint_encoder)
        self.memory_efficient_checkpoint_decoder = bool(memory_efficient_checkpoint_decoder)
        self.memory_efficient_verify_replay = bool(memory_efficient_verify_replay)
        self.free_loss = None
        self.pref_builder = None
        self._pref_extract_feature_cache = None
        self._pref_build_runtime_observables = None
        self._pref_batch_cls = None
        self._resolve_pref_pair_artifacts()
        if self.pref_pair_json_path and self.loss_type == "rl_loss" and self.free_loss_ir_json_path:
            self.loss_type = "free_loss"
            log.info("Resolved loss from pref_pair_json_path; switching loss_type to free_loss")
        if self.pref_builder_ir_json_path:
            self._load_pref_builder()
        if self.loss_type == "free_loss":
            self._load_free_loss()
        elif self.pref_builder is not None:
            log.warning(
                "pref_builder_ir_json_path is set but loss_type=%s; the builder will be ignored",
                self.loss_type,
            )
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

        if phase == "train" and self.memory_efficient_preference:
            policy_device = next(self.policy.parameters()).device
            td = td.to(policy_device)
            return self._memory_efficient_preference_step(
                td=td,
                batch=batch,
                n_start=n_start,
                dataloader_idx=dataloader_idx,
            )

        # Evaluate policy
        policy_kwargs: dict[str, Any] = {"phase": phase, "num_starts": n_start}
        if phase == "train" and self.loss_type == "free_loss":
            observables = set(self.free_loss_observables)
            want_seq_len = bool(observables & {"seq_len", "log_prob_mean", "entropy_mean"})
            want_entropy = bool(observables & {"entropy", "entropy_mean"})
            want_step_logp = "log_prob_step" in observables
            # When step-level log-probs are requested, sequence length can be
            # inferred without materializing the full action history.
            want_actions = bool(want_seq_len and not want_step_logp)
            policy_kwargs.update(
                {
                    "return_actions": want_actions,
                    "return_entropy": want_entropy,
                    "return_sum_log_likelihood": not want_step_logp,
                }
            )
        elif phase == "train" and self.loss_type == "bopo_loss":
            # BOPO uses mean log-prob per decoding step in the official code.
            # Request actions so we can recover the rollout length cheaply here.
            policy_kwargs.update({"return_actions": True})
        out = self.policy(td, self.env, **policy_kwargs)

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            raw_log_likelihood = out["log_likelihood"]
            if self.loss_type == "free_loss" and raw_log_likelihood.ndim > 1:
                log_likelihood_step = unbatchify(raw_log_likelihood, (n_aug, n_start))
                log_likelihood = log_likelihood_step.sum(dim=-1)
                out["log_likelihood_step"] = log_likelihood_step
                out["log_likelihood"] = log_likelihood
            else:
                log_likelihood = unbatchify(raw_log_likelihood, (n_aug, n_start))
                out["log_likelihood"] = log_likelihood
            if self.loss_type == "free_loss" and isinstance(out.get("entropy"), torch.Tensor):
                out["entropy"] = unbatchify(out["entropy"], (n_aug, n_start))
            if self.loss_type == "free_loss" and isinstance(out.get("actions"), torch.Tensor):
                out["actions"] = unbatchify(out["actions"], (n_aug, n_start))
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

    def _memory_efficient_preference_step(
        self,
        *,
        td,
        batch,
        n_start: int,
        dataloader_idx: int | None,
        log_metrics: bool = True,
    ):
        if not isinstance(self.policy, (PO4COPsTSPPolicy, PO4COPsCVRPPolicy)):
            raise TypeError(
                "memory_efficient_preference requires a PO4COPs TSP or CVRP policy"
            )
        if self.loss_type not in {"po_loss", "bopo_loss", "slim_loss", "free_loss"}:
            raise ValueError(
                "memory_efficient_preference supports po_loss, bopo_loss, slim_loss, and free_loss; "
                f"got {self.loss_type}"
            )
        if n_start is None or n_start <= 1:
            raise ValueError("memory_efficient_preference requires num_starts > 1")
        if set(self.free_loss_observables) & {
            "log_prob_step",
            "entropy",
            "entropy_mean",
        }:
            raise ValueError(
                "memory_efficient_preference does not support step-level log-probability "
                "or entropy observables"
            )

        with torch.no_grad():
            rollout_out = self.policy(
                td,
                self.env,
                phase="train",
                num_starts=n_start,
                return_actions=True,
                return_entropy=False,
                return_sum_log_likelihood=True,
            )

        reward = unbatchify(rollout_out["reward"], (0, n_start)).detach()
        sampled_log_likelihood = unbatchify(
            rollout_out["log_likelihood"],
            (0, n_start),
        ).detach()
        actions = unbatchify(rollout_out["actions"], (0, n_start)).detach()

        leaf_log_likelihood = sampled_log_likelihood.float().requires_grad_(True)
        objective_out = {
            "reward": reward,
            "log_likelihood": leaf_log_likelihood,
            "actions": actions,
        }
        instance_losses = []
        for instance_index in range(reward.shape[0]):
            instance_out = {
                "reward": reward[instance_index : instance_index + 1],
                "log_likelihood": leaf_log_likelihood[
                    instance_index : instance_index + 1
                ],
                "actions": actions[instance_index : instance_index + 1],
            }
            self.calculate_loss(
                td,
                batch,
                instance_out,
                reward=instance_out["reward"].float(),
                log_likelihood=instance_out["log_likelihood"],
            )
            instance_losses.append(instance_out["loss"])
        objective_loss = torch.stack(instance_losses).mean()
        objective_out["loss"] = objective_loss
        coefficients = torch.autograd.grad(
            objective_loss,
            leaf_log_likelihood,
            create_graph=False,
            retain_graph=False,
        )[0].detach()

        replay_out = self.policy(
            td,
            self.env,
            phase="train",
            num_starts=n_start,
            return_actions=False,
            return_entropy=False,
            return_sum_log_likelihood=True,
            forced_actions=rollout_out["actions"],
            checkpoint_encoder_layers=self.memory_efficient_checkpoint_encoder,
            checkpoint_selected_log_probs=self.memory_efficient_checkpoint_decoder,
        )
        replay_log_likelihood = unbatchify(
            replay_out["log_likelihood"],
            (0, n_start),
        )
        replay_error = (
            replay_log_likelihood.detach().float() - sampled_log_likelihood.float()
        ).abs().max()
        replay_error_value = float(replay_error.item())
        if self.memory_efficient_verify_replay and replay_error_value > 1e-5:
            raise RuntimeError(
                "Forced replay changed trajectory log-likelihoods: "
                f"max_abs_error={replay_error_value}"
            )

        surrogate = (coefficients * replay_log_likelihood.float()).sum()
        loss = objective_loss.detach() + surrogate - surrogate.detach()
        objective_out.update(
            {
                "loss": loss,
                "reward": rollout_out["reward"],
                "log_likelihood": replay_out["log_likelihood"],
                "memory_efficient_replay_error": replay_error.detach(),
                "memory_efficient_coefficient_abs_mean": coefficients.abs().mean(),
            }
        )
        max_reward = reward.max(dim=-1).values
        objective_out["max_reward"] = max_reward

        metrics = (
            self.log_metrics(
                objective_out,
                "train",
                dataloader_idx=dataloader_idx,
            )
            if log_metrics
            else {}
        )
        return {
            "loss": loss,
            "memory_efficient_replay_error": replay_error.detach(),
            "memory_efficient_coefficient_abs_mean": coefficients.abs().mean().detach(),
            **metrics,
        }

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
            loss, pref_rate = po_loss(
                reward,
                log_likelihood,
                alpha=self.alpha,
                impl=self.po_impl,
            )
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
        if self.loss_type == "bopo_loss":
            actions = policy_out.get("actions")
            sequence_length = None
            if isinstance(actions, torch.Tensor):
                sequence_length = torch.full_like(log_likelihood, float(actions.shape[-1]))
            loss, pair_count = bopo_loss(
                reward,
                log_likelihood,
                alpha=self.alpha,
                pair_mode=self.bopo_pair_mode,
                select_strategy=self.bopo_select_strategy,
                select_k=self.bopo_select_k,
                select_quantile=self.bopo_select_quantile,
                sequence_length=sequence_length,
            )
            policy_out.update(
                {
                    "loss": loss,
                    "bopo_loss": loss.detach(),
                    "bopo_pair_count": pair_count.detach() if isinstance(pair_count, torch.Tensor) else pair_count,
                }
            )
            return policy_out
        if self.loss_type == "slim_loss":
            actions = policy_out.get("actions")
            sequence_length = None
            if isinstance(actions, torch.Tensor):
                sequence_length = float(actions.shape[-1])
            loss = slim_loss(
                reward,
                log_likelihood,
                sequence_length=sequence_length,
            )
            policy_out.update({"loss": loss, "slim_loss": loss.detach()})
            return policy_out
        if self.loss_type == "sll_loss":
            loss = sll_loss(
                reward,
                log_likelihood,
                alpha=self.alpha,
                impl=self.sll_impl,
                temperature=self.sll_temperature,
            )
            policy_out.update({"loss": loss, "sll_loss": loss.detach()})
            return policy_out
        if self.loss_type == "free_loss":
            preference_loss, pair_count = self._free_loss_loss_fn(
                reward, log_likelihood, policy_out
            )
            anchor_weight = float(self.preference_po_anchor_weight)
            if anchor_weight > 0.0:
                anchor_loss, _ = po_loss(
                    reward,
                    log_likelihood,
                    alpha=float(self.preference_po_anchor_alpha),
                    impl="exponential",
                )
                loss = (
                    (1.0 - anchor_weight) * preference_loss
                    + anchor_weight * anchor_loss
                )
            else:
                anchor_loss = preference_loss.new_zeros(())
                loss = preference_loss
            policy_out.update(
                {
                    "loss": loss,
                    "free_loss": preference_loss.detach(),
                    "free_loss_pair_count": pair_count,
                    "preference_po_anchor_loss": anchor_loss.detach(),
                    "preference_po_anchor_weight": anchor_weight,
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

    def _load_free_loss_runtime_helpers(self) -> None:
        if (
            self._pref_extract_feature_cache is not None
            and self._pref_build_runtime_observables is not None
            and self._pref_batch_cls is not None
        ):
            return

        self._ensure_ptp_root_on_path()
        try:
            from fitness.free_loss_fidelity import (
                PrefBatch,
                build_runtime_observables,
                extract_feature_cache,
            )
        except ImportError as exc:
            raise ImportError(
                "Failed to import PTP free-loss runtime modules. "
                "Ensure the repository still contains the PTP/ directory."
            ) from exc

        self._pref_extract_feature_cache = extract_feature_cache
        self._pref_build_runtime_observables = build_runtime_observables
        self._pref_batch_cls = PrefBatch

    def _resolve_pref_pair_artifacts(self) -> None:
        if self.pref_pair_json_path is None:
            return

        pair_path = Path(self.pref_pair_json_path).expanduser()
        if not pair_path.is_file():
            raise FileNotFoundError(f"pref_pair_json_path does not exist: {pair_path.as_posix()}")

        run_dir = pair_path.parent
        builder_path = run_dir / "best_builder.json"
        loss_path = run_dir / "best_loss.json"
        if self.pref_builder_ir_json_path is None:
            if not builder_path.is_file():
                raise FileNotFoundError(
                    f"best_builder.json not found next to pref_pair_json_path: {builder_path.as_posix()}"
                )
            self.pref_builder_ir_json_path = builder_path.as_posix()
        if self.free_loss_ir_json_path is None:
            if not loss_path.is_file():
                raise FileNotFoundError(
                    f"best_loss.json not found next to pref_pair_json_path: {loss_path.as_posix()}"
                )
            self.free_loss_ir_json_path = loss_path.as_posix()

        try:
            with pair_path.open("r", encoding="utf-8") as f:
                pair_payload = json.load(f)
            pair_gid = str(pair_payload.get("g_id", "")).strip()
            pair_fid = str(pair_payload.get("f_id", "")).strip()
        except Exception:
            return

        for expected_id, artifact_path, key in (
            (pair_gid, self.pref_builder_ir_json_path, "id"),
            (pair_fid, self.free_loss_ir_json_path, "id"),
        ):
            if not expected_id or not artifact_path:
                continue
            try:
                with Path(artifact_path).expanduser().open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                actual_id = str(payload.get(key, "")).strip()
            except Exception:
                continue
            if actual_id and actual_id != expected_id:
                log.warning(
                    "Resolved artifact %s id=%s does not match pref_pair expected id=%s",
                    artifact_path,
                    actual_id,
                    expected_id,
                )

    @staticmethod
    def _ensure_ptp_root_on_path() -> None:
        repo_root = Path(__file__).resolve().parents[4]
        ptp_root = repo_root / "PTP"
        ptp_root_str = str(ptp_root.resolve())
        if ptp_root.is_dir() and ptp_root_str not in sys.path:
            sys.path.insert(0, ptp_root_str)

    def _load_pref_builder(self) -> None:
        if self.pref_builder_ir_json_path is None:
            raise ValueError("pref_builder_ir_json_path must be set before loading a preference builder.")

        self._ensure_ptp_root_on_path()
        try:
            from ptp_discovery.pref_builder_compiler import compile_preference_builder
            from ptp_discovery.pref_builder_ir import ir_from_json as pref_builder_ir_from_json
        except ImportError as exc:
            raise ImportError(
                "Failed to import PTP preference-builder modules. "
                "Ensure the repository still contains the PTP/ directory."
            ) from exc
        self._load_free_loss_runtime_helpers()

        path = Path(self.pref_builder_ir_json_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"pref_builder_ir_json_path does not exist: {path.as_posix()}"
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        ir_obj = payload.get("ir", payload)
        ir = pref_builder_ir_from_json(ir_obj)
        self.pref_builder = compile_preference_builder(ir)

    def _free_loss_loss_fn(
        self, reward: torch.Tensor, log_likelihood: torch.Tensor, policy_out: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.free_loss is None:
            raise RuntimeError(
                "free_loss is not compiled; check free_loss_ir_json_path."
            )
        self._load_free_loss_runtime_helpers()
        if self._pref_extract_feature_cache is None or self._pref_build_runtime_observables is None:
            raise RuntimeError("Free-loss runtime helpers are not initialized.")

        objective = -reward
        seq_len = None
        seq_len_fallback = None
        actions = policy_out.get("actions")
        if isinstance(actions, torch.Tensor):
            seq_len = torch.full_like(log_likelihood, float(actions.shape[-1]))
        elif isinstance(policy_out.get("log_likelihood_step"), torch.Tensor):
            seq_len = torch.full_like(
                log_likelihood,
                float(policy_out["log_likelihood_step"].shape[-1]),
            )
        else:
            size_value = None
            generator = getattr(self.env, "generator", None)
            for attr in ("num_loc", "num_jobs", "num_job"):
                value = getattr(generator, attr, None)
                if value is not None:
                    size_value = float(value)
                    break
            if size_value is not None:
                seq_len_fallback = size_value
                seq_len = torch.full_like(log_likelihood, size_value)

        extra = self._pref_build_runtime_observables(
            reward,
            log_likelihood,
            observables=self.free_loss_observables,
            seq_len=seq_len,
            log_prob_step=policy_out.get("log_likelihood_step"),
            entropy=policy_out.get("entropy"),
            seq_len_fallback=seq_len_fallback,
        )
        feature_cache = self._pref_extract_feature_cache(
            objective=objective,
            log_prob=log_likelihood,
            extra=extra,
        )

        loss_batch: dict[str, torch.Tensor]
        pair_count_value = 0

        if self.pref_builder is not None:
            pref_batch = self.pref_builder.build_fn(
                feature_cache,
                {
                    "alpha": self.alpha,
                    "hyperparams": dict(self.pref_builder_kwargs),
                    **self.pref_builder_kwargs,
                },
            )
            if (
                getattr(self, "detach_pref_weights", False)
                and isinstance(pref_batch.weight, torch.Tensor)
            ):
                pref_batch.weight = pref_batch.weight.detach()
            pair_count_value = int(pref_batch.num_examples())
            if pair_count_value > 0:
                loss_batch = pref_batch.to_pairwise_loss_batch(feature_cache)
            else:
                loss_batch = {}
        else:
            loss_batch = {}

        if not loss_batch:
            mask = objective[:, :, None] < objective[:, None, :]
            b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
            pair_count_value = int(b_idx.numel())
            if pair_count_value > 0:
                if self._pref_batch_cls is None:
                    raise RuntimeError("Preference batch class is not initialized.")
                pref_batch = self._pref_batch_cls(
                    mode="pairwise",
                    pair_idx=(b_idx, winner_idx, loser_idx),
                )
                loss_batch = pref_batch.to_pairwise_loss_batch(feature_cache)

        pair_count = torch.tensor(
            float(pair_count_value), device=reward.device, dtype=reward.dtype
        )

        if pair_count_value == 0:
            advantage = reward - reward.float().mean(dim=1, keepdim=True)
            loss = -(advantage * log_likelihood).mean()
            return loss, pair_count

        loss = self.free_loss.loss_fn(
            batch=loss_batch,
            model_output=feature_cache,
            extra={"alpha": self.alpha},
        )
        return loss, pair_count
