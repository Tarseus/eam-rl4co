import abc

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _preview_tensor_row(value: Tensor, limit: int = 12):
    value = value.detach().cpu()
    if value.numel() == 1:
        return value.item()
    flat = value.reshape(-1)
    preview = flat[:limit].tolist()
    if flat.numel() > limit:
        preview.append("...")
    return preview


def _format_fixed_action_diagnostic(
    td: TensorDict,
    mask: Tensor,
    provided_action: Tensor,
    actions: Optional[Tensor],
    step: int,
    decode_type: str,
    invalid_idx: int,
) -> str:
    sample_action = int(provided_action[invalid_idx].detach().cpu().item())
    action_dim = int(mask.size(-1))
    safe_action = min(max(sample_action, 0), action_dim - 1)
    feasible_actions = (
        torch.nonzero(mask[invalid_idx].detach().cpu(), as_tuple=False)
        .squeeze(-1)
        .tolist()
    )
    diag_parts = [
        f"decode_type={decode_type!r}",
        f"step={step}",
        f"sample_index={invalid_idx}",
        f"provided_action={sample_action}",
        f"safe_action={safe_action}",
        f"action_dim={action_dim}",
        f"done={bool(td['done'][invalid_idx].detach().cpu().item()) if 'done' in td.keys() else 'n/a'}",
        f"feasible_action_count={len(feasible_actions)}",
        f"feasible_actions_preview={feasible_actions[:12]}",
    ]

    for key in ("current_node", "tour_length", "i", "reward"):
        if key in td.keys():
            diag_parts.append(f"{key}={_preview_tensor_row(td[key][invalid_idx])}")

    if "max_length" in td.keys():
        max_length_row = td["max_length"][invalid_idx]
        diag_parts.append(f"max_length_preview={_preview_tensor_row(max_length_row)}")
        if max_length_row.numel() > 1 and 0 <= safe_action < max_length_row.shape[-1]:
            diag_parts.append(
                f"max_length_at_action={max_length_row.detach().cpu().reshape(-1)[safe_action].item()}"
            )

    if actions is not None:
        prefix_end = min(actions.size(-1), step + 4)
        diag_parts.append(
            f"action_prefix={actions[invalid_idx, :prefix_end].detach().cpu().tolist()}"
        )

    return ", ".join(diag_parts)


class ConstructiveEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for the encoder of constructive models"""

    @abc.abstractmethod
    def forward(self, td: TensorDict) -> Tuple[Any, Tensor]:
        """Forward pass for the encoder

        Args:
            td: TensorDict containing the input data

        Returns:
            Tuple containing:
              - latent representation (any type)
              - initial embeddings (from feature space to embedding space)
        """
        raise NotImplementedError("Implement me in subclass!")


class ConstructiveDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for constructive models. The decoder is responsible for generating the logits for the action"""

    @abc.abstractmethod
    def forward(
        self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Obtain logits for current action to the next ones

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder. Can be any type
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the logits and the action mask
        """
        raise NotImplementedError("Implement me in subclass!")

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, Any]:
        """By default, we don't need to do anything here.

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder
            env: Environment for decoding
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the updated Tensordict, environment, and hidden state
        """
        return td, env, hidden


class NoEncoder(ConstructiveEncoder):
    """Default encoder decoder-only models, i.e. autoregressive models that re-encode all the state at each decoding step."""

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Return Nones for the hidden state and initial embeddings"""
        return None, None


class ConstructivePolicy(nn.Module):
    """
    Base class for constructive policies. Constructive policies take as input and instance and output a solution (sequence of actions).
    "Constructive" means that a solution is created from scratch by the model.

    The structure follows roughly the following steps:
        1. Create a hidden state from the encoder
        2. Initialize decoding strategy (such as greedy, sampling, etc.)
        3. Decode the action given the hidden state and the environment state at the current step
        4. Update the environment state with the action. Repeat 3-4 until all sequences are done
        5. Obtain log likelihood, rewards etc.

    Note that an encoder is not strictly needed (see :class:`NoEncoder`).). A decoder however is always needed either in the form of a
    network or a function.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward may not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/ai4co/rl4co/issues/72.

    Args:
        encoder: Encoder to use
        decoder: Decoder to use
        env_name: Environment name to solve (used for automatically instantiating networks)
        temperature: Temperature for the softmax during decoding
        tanh_clipping: Clipping value for the tanh activation (see Bello et al. 2016) during decoding
        mask_logits: Whether to mask the logits or not during decoding
        train_decode_type: Decoding strategy for training
        val_decode_type: Decoding strategy for validation
        test_decode_type: Decoding strategy for testing
    """

    def __init__(
        self,
        encoder: ConstructiveEncoder | Callable,
        decoder: ConstructiveDecoder | Callable,
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(ConstructivePolicy, self).__init__()

        if len(unused_kw) > 0:
            log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.env_name = env_name

        # Encoder and decoder
        if encoder is None:
            log.warning("`None` was provided as encoder. Using `NoEncoder`.")
            encoder = NoEncoder()
        self.encoder = encoder
        self.decoder = decoder

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[str | RL4COEnvBase] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")
        requested_num_starts = decoding_kwargs.get("num_starts", None)
        if actions is not None:
            # Preserve multistart behavior when evaluating fixed action sequences. In some call
            # sites the phase decode type may not explicitly contain "multistart", but the caller
            # still passes `num_starts > 1` together with multistart-generated action sequences.
            # In that case we must still run the multistart pre-hook so the first action is
            # consumed before the main decoding loop.
            is_multistart_eval = "multistart" in str(decode_type) or (
                requested_num_starts is not None and int(requested_num_starts) > 1
            )
            decode_type = (
                "multistart_evaluate" if is_multistart_eval else "evaluate"
            )

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        pre_action = (
            actions[..., 0]
            if actions is not None and "multistart" in str(decode_type)
            else None
        )
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env, action=pre_action)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = len(getattr(decode_strategy, "actions", [])) if actions is not None else 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            provided_action = None
            if actions is not None:
                if step < actions.size(-1):
                    provided_action = actions[..., step]
                else:
                    raise ValueError(
                        "Provided `actions` sequence is shorter than required to finish decoding: "
                        f"decode_type={decode_type!r}, actions.shape={tuple(actions.shape)}, "
                        f"needed_step_index={step} (0-based). "
                        "This indicates the supplied solution is non-canonical or inconsistent with the "
                        "current environment state."
                    )

            if provided_action is not None and mask is not None:
                action_dim = mask.size(-1)
                in_range = (provided_action >= 0) & (provided_action < action_dim)
                safe_action = provided_action.clamp(min=0, max=action_dim - 1)
                feasible = in_range & mask.gather(1, safe_action.unsqueeze(-1)).squeeze(-1)
                if not feasible.all():
                    invalid_count = int((~feasible).sum().item())
                    first_bad = int(torch.nonzero(~feasible, as_tuple=False)[0].item())
                    diagnostic = _format_fixed_action_diagnostic(
                        td=td,
                        mask=mask,
                        provided_action=provided_action,
                        actions=actions,
                        step=step,
                        decode_type=str(decode_type),
                        invalid_idx=first_bad,
                    )
                    log.error(
                        "Fixed-action decoding infeasibility detected: %s",
                        diagnostic,
                    )
                    raise ValueError(
                        "Provided actions contain infeasible entries during fixed-action decoding: "
                        f"decode_type={decode_type!r}, step={step}, invalid_count={invalid_count}, "
                        f"actions.shape={tuple(actions.shape) if actions is not None else None}. "
                        "This indicates the candidate solution is illegal for the current state. "
                        f"First invalid sample diagnostics: {diagnostic}"
                    )
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=provided_action,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict
