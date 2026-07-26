import math

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from tensordict import TensorDict

from rl4co.utils.ops import batchify, select_start_nodes, unbatchify


_USE_SDPA = os.getenv("RL4CO_POMO_USE_SDPA", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _get_encoding(encoded_nodes: torch.Tensor, node_index_to_pick: torch.Tensor) -> torch.Tensor:
    original_dim = node_index_to_pick.dim()
    if original_dim == 1:
        node_index_to_pick = node_index_to_pick[:, None]

    batch_size = node_index_to_pick.size(0)
    index_shape = node_index_to_pick.shape
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[..., None].expand(*index_shape, embedding_dim)
    if len(index_shape) == 3:
        gathering_index = gathering_index.reshape(batch_size, -1, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    if original_dim == 1:
        return picked_nodes
    return picked_nodes


def _reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)


def _multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank3_ninf_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    # CUDA 11.8 / RTX 3090 has no cuDNN SDPA execution plan for the decoder's
    # dynamic additive mask. The encoder is unmasked and can still use the
    # fused, memory-efficient kernel.
    if _USE_SDPA and rank3_ninf_mask is None:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(float(key_dim)),
        )
    else:
        score = torch.matmul(q, k.transpose(2, 3))
        score_scaled = score / math.sqrt(float(key_dim))
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
                batch_s, head_num, n, input_s
            )

        weights = F.softmax(score_scaled, dim=3)
        out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


def _select_actions_from_probs(
    probs: torch.Tensor,
    use_sampling: bool,
    use_hybrid: bool,
) -> torch.Tensor:
    if use_sampling:
        selected = probs.reshape(-1, probs.size(-1)).multinomial(1).squeeze(-1)
        selected = selected.view(probs.size(0), probs.size(1))
        if use_hybrid and selected.size(1) > 0:
            # Keep one deterministic greedy trajectory in the sampled pool.
            selected[:, 0] = probs[:, 0].argmax(dim=-1)
        return selected
    return probs.argmax(dim=2)


class _AddAndNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        return normalized.transpose(1, 2)


class _FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(input1)))


class _EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: int, ff_hidden_dim: int):
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_and_norm_1 = _AddAndNorm(embedding_dim)
        self.feed_forward = _FeedForward(embedding_dim, ff_hidden_dim)
        self.add_and_norm_2 = _AddAndNorm(embedding_dim)

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        q = _reshape_by_heads(self.Wq(input1), head_num=self.head_num)
        k = _reshape_by_heads(self.Wk(input1), head_num=self.head_num)
        v = _reshape_by_heads(self.Wv(input1), head_num=self.head_num)
        out_concat = _multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_and_norm_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_and_norm_2(out1, out2)
        return out3


class _PO4COPsEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        encoder_layer_num: int,
        head_num: int,
        qkv_dim: int,
        ff_hidden_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
                for _ in range(encoder_layer_num)
            ]
        )

    def forward(
        self,
        data: torch.Tensor,
        checkpoint_layers: bool = False,
    ) -> torch.Tensor:
        out = self.embedding(data)
        for layer in self.layers:
            if checkpoint_layers and torch.is_grad_enabled():
                out = activation_checkpoint(layer, out, use_reentrant=False)
            else:
                out = layer(out)
        return out


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        mode: str,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        sqrt_embedding_dim: float,
        logit_clipping: float,
        ff_hidden_dim: int,
    ):
        super().__init__()
        self.mode = mode
        self.head_num = head_num
        self.sqrt_embedding_dim = sqrt_embedding_dim
        self.logit_clipping = logit_clipping
        self.first = False

        if mode == "feature":
            self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
            self.add_and_norm_1 = _AddAndNorm(embedding_dim)
            self.feed_forward = _FeedForward(embedding_dim, ff_hidden_dim)
            self.add_and_norm_2 = _AddAndNorm(embedding_dim)
        else:
            self.Wq_first = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.Wlogit_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None
        self.v = None
        self.logitk = None
        self.q_first = None

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        if self.mode == "feature":
            self.k = _reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
            self.v = _reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)
        else:
            self.logitk = self.Wlogit_k(encoded_nodes)

    def set_q1(self, encoded_q1: torch.Tensor) -> None:
        self.first = True
        if self.mode == "feature":
            self.q_first = _reshape_by_heads(self.Wq_first(encoded_q1), head_num=self.head_num)
        else:
            self.q_first = self.Wq_first(encoded_q1)

    def forward(self, input_tensor: torch.Tensor, ninf_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_with_cache(
            input_tensor,
            ninf_mask,
            k=self.k,
            v=self.v,
            logitk=self.logitk,
            q_first=self.q_first if self.first else None,
        )

    def forward_with_cache(
        self,
        input_tensor: torch.Tensor,
        ninf_mask: torch.Tensor | None,
        *,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        logitk: torch.Tensor | None,
        q_first: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == "feature":
            q = _reshape_by_heads(self.Wq_last(input_tensor), head_num=self.head_num)
            if q_first is not None:
                q = q + q_first
            out_concat = _multi_head_attention(q, k, v, rank3_ninf_mask=ninf_mask)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.add_and_norm_1(input_tensor, multi_head_out)
            out2 = self.feed_forward(out1)
            out3 = self.add_and_norm_2(out1, out2)
            return out3

        q = self.Wq_last(input_tensor)
        if q_first is not None:
            q = q + q_first

        score = torch.matmul(q, logitk.transpose(1, 2))
        score_scaled = score / self.sqrt_embedding_dim
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped if ninf_mask is None else score_clipped + ninf_mask
        return F.softmax(score_masked, dim=2)


class _PO4COPsDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        decoder_layer_num: int,
        head_num: int,
        qkv_dim: int,
        sqrt_embedding_dim: float,
        logit_clipping: float,
        ff_hidden_dim: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DecoderLayer(
                    mode="feature",
                    embedding_dim=embedding_dim,
                    head_num=head_num,
                    qkv_dim=qkv_dim,
                    sqrt_embedding_dim=sqrt_embedding_dim,
                    logit_clipping=logit_clipping,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(decoder_layer_num)
            ]
        )
        self.layers.append(
            _DecoderLayer(
                mode="logit",
                embedding_dim=embedding_dim,
                head_num=head_num,
                qkv_dim=qkv_dim,
                sqrt_embedding_dim=sqrt_embedding_dim,
                logit_clipping=logit_clipping,
                ff_hidden_dim=ff_hidden_dim,
            )
        )

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        for layer in self.layers:
            layer.set_kv(encoded_nodes)

    def set_q1(self, encoded_q1: torch.Tensor) -> None:
        self.layers[0].set_q1(encoded_q1)

    def forward(self, encoded_last_node: torch.Tensor, ninf_mask: torch.Tensor) -> torch.Tensor:
        out = encoded_last_node
        for layer in self.layers:
            out = layer(out, ninf_mask)
        return out

    def cache_tensors(self) -> tuple[torch.Tensor, ...]:
        reference = next(
            value
            for layer in self.layers
            for value in (layer.k, layer.v, layer.logitk, layer.q_first)
            if value is not None
        )
        empty = reference.new_empty((0,))
        cache: list[torch.Tensor] = []
        for layer in self.layers:
            if layer.mode == "feature":
                cache.extend(
                    (
                        layer.k,
                        layer.v,
                        layer.q_first if layer.first else empty,
                    )
                )
            else:
                cache.extend(
                    (
                        layer.logitk,
                        layer.q_first if layer.first else empty,
                    )
                )
        return tuple(cache)

    def forward_with_cache(
        self,
        encoded_last_node: torch.Tensor,
        ninf_mask: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        out = encoded_last_node
        offset = 0
        for layer in self.layers:
            if layer.mode == "feature":
                k, v, q_first = cache_tensors[offset : offset + 3]
                offset += 3
                out = layer.forward_with_cache(
                    out,
                    ninf_mask,
                    k=k,
                    v=v,
                    logitk=None,
                    q_first=q_first if q_first.numel() else None,
                )
            else:
                logitk, q_first = cache_tensors[offset : offset + 2]
                offset += 2
                out = layer.forward_with_cache(
                    out,
                    ninf_mask,
                    k=None,
                    v=None,
                    logitk=logitk,
                    q_first=q_first if q_first.numel() else None,
                )
        if offset != len(cache_tensors):
            raise ValueError(
                f"Expected {offset} decoder cache tensors, got {len(cache_tensors)}"
            )
        return out


class PO4COPsTSPPolicy(nn.Module):
    """PO4COPs-compatible TSP policy used to reproduce the original POMO-PO training dynamics."""

    def __init__(
        self,
        env_name: str = "tsp",
        embedding_dim: int = 128,
        encoder_layer_num: int = 6,
        decoder_layer_num: int = 1,
        qkv_dim: int = 16,
        head_num: int = 8,
        ff_hidden_dim: int = 512,
        logit_clipping: float = 50.0,
        start_node: str = "pomo",
        eval_type: str = "argmax",
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kwargs,
    ):
        super().__init__()
        if env_name != "tsp":
            raise ValueError("PO4COPsTSPPolicy currently supports only TSP.")

        self.env_name = env_name
        self.start_node = str(start_node).strip().lower()
        if self.start_node not in {"same", "random", "pomo"}:
            raise ValueError(
                f"Unsupported start_node={start_node!r}; use 'same', 'random', or 'pomo'."
            )
        self.eval_type = str(eval_type).strip().lower()
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        sqrt_embedding_dim = math.sqrt(float(embedding_dim))

        self.encoder = _PO4COPsEncoder(
            embedding_dim=embedding_dim,
            encoder_layer_num=encoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.decoder = _PO4COPsDecoder(
            embedding_dim=embedding_dim,
            decoder_layer_num=decoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            logit_clipping=logit_clipping,
            ff_hidden_dim=ff_hidden_dim,
        )

        self.encoded_nodes = None

    def _select_initial_actions(
        self,
        td_base: TensorDict,
        env,
        num_starts: int,
    ) -> torch.Tensor:
        if self.start_node == "pomo":
            return select_start_nodes(td_base, env, num_starts)

        batch_size = td_base.shape[0]
        num_loc = td_base["locs"].shape[1]
        if self.start_node == "same":
            return torch.zeros(batch_size * num_starts, dtype=torch.long, device=td_base.device)

        starts = torch.randint(
            low=0,
            high=num_loc,
            size=(batch_size, num_starts),
            device=td_base.device,
        )
        return starts.transpose(0, 1).reshape(-1)

    def pre_forward(self, reset_td: TensorDict) -> None:
        self.encoded_nodes = self.encoder(reset_td["locs"])
        self.decoder.set_kv(self.encoded_nodes)

    def forward(
        self,
        td: TensorDict,
        env,
        phase: str = "train",
        num_starts: int = 0,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_sum_log_likelihood: bool = True,
        forced_actions: torch.Tensor | None = None,
        forced_prefix_actions: torch.Tensor | None = None,
        checkpoint_encoder_layers: bool = False,
        checkpoint_selected_log_probs: bool = False,
        **unused_kwargs,
    ) -> dict:
        if num_starts is None or num_starts <= 0:
            num_starts = env.get_num_starts(td)

        td_base = td.clone()
        base_batch = td_base.shape[0]

        # Match PO4COPs: k/v are computed once from the base batch [B, N, E]
        encoded_nodes = self.encoder(
            td_base["locs"],
            checkpoint_layers=checkpoint_encoder_layers,
        )
        self.decoder.set_kv(encoded_nodes)

        if forced_actions is not None and forced_prefix_actions is not None:
            raise ValueError(
                "forced_actions and forced_prefix_actions are mutually exclusive"
            )
        forced_actions_3d = None
        forced_input = (
            forced_actions
            if forced_actions is not None
            else forced_prefix_actions
        )
        force_complete_trajectory = forced_actions is not None
        if forced_input is not None:
            if forced_input.ndim == 2:
                if forced_input.shape[0] != base_batch * num_starts:
                    raise ValueError(
                        "Flattened forced action input must have leading dimension "
                        f"{base_batch * num_starts}, got {forced_input.shape[0]}"
                    )
                forced_actions_3d = forced_input.reshape(
                    num_starts,
                    base_batch,
                    -1,
                ).permute(1, 0, 2)
            elif forced_input.ndim == 3:
                if forced_input.shape[:2] != (base_batch, num_starts):
                    raise ValueError(
                        "Rank-3 forced action input must have shape "
                        f"[{base_batch}, {num_starts}, T], got {tuple(forced_input.shape)}"
                    )
                forced_actions_3d = forced_input
            else:
                raise ValueError(
                    "forced action input must have rank 2 or 3, got rank "
                    f"{forced_input.ndim}"
                )
            forced_actions_3d = forced_actions_3d.to(
                device=td_base.device,
                dtype=torch.long,
            )

        # Environment rollout still uses flattened multistart batch [B*S, ...]
        td_flat = batchify(td_base, num_starts)
        if forced_actions_3d is None:
            first_action_flat = self._select_initial_actions(td_base, env, num_starts)
        else:
            first_action_flat = forced_actions_3d[:, :, 0].transpose(0, 1).reshape(-1)
        td_flat.set("action", first_action_flat)
        td_flat = env.step(td_flat)["next"]

        first_action = unbatchify(first_action_flat, num_starts)
        current_node = unbatchify(td_flat["current_node"], num_starts)
        encoded_first_node = _get_encoding(encoded_nodes, current_node)
        self.decoder.set_q1(encoded_first_node)
        decoder_cache = self.decoder.cache_tensors()

        actions = [first_action]
        log_probs = [torch.zeros_like(first_action, dtype=encoded_nodes.dtype)]
        entropies = [torch.zeros_like(first_action, dtype=encoded_nodes.dtype)]

        decode_type = getattr(self, f"{phase}_decode_type", "sampling")
        if decode_type.startswith("multistart_"):
            decode_type = decode_type[len("multistart_") :]
        decode_type = str(decode_type).strip().lower()
        use_hybrid = self.eval_type == "hybrid" or decode_type == "hybrid"
        use_sampling = (
            (phase == "train")
            or (decode_type == "sampling")
            or (decode_type == "softmax")
            or use_hybrid
        )

        done = td_flat["done"]
        step_index = 1
        while not done.all():
            current_node = unbatchify(td_flat["current_node"], num_starts)
            encoded_last_node = _get_encoding(encoded_nodes, current_node)

            action_mask = unbatchify(td_flat["action_mask"], num_starts)
            ninf_mask = torch.where(
                action_mask,
                torch.zeros_like(action_mask, dtype=encoded_nodes.dtype),
                float("-inf"),
            )
            forcing_step = (
                forced_actions_3d is not None
                and step_index < forced_actions_3d.shape[2]
            )
            if not forcing_step:
                probs = self.decoder(encoded_last_node, ninf_mask)
                selected = _select_actions_from_probs(probs, use_sampling, use_hybrid)
                prob = probs.gather(2, selected.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
                selected_log_prob = prob.log()
                if return_entropy:
                    probs_safe = probs.clamp_min(1e-12)
                    entropies.append(-(probs_safe * probs_safe.log()).sum(dim=2))
            else:
                selected = forced_actions_3d[:, :, step_index]

                def selected_log_prob_fn(
                    last_node: torch.Tensor,
                    mask: torch.Tensor,
                    action: torch.Tensor,
                    *cache: torch.Tensor,
                ) -> torch.Tensor:
                    action_probs = self.decoder.forward_with_cache(
                        last_node,
                        mask,
                        *cache,
                    )
                    selected_prob = action_probs.gather(
                        2,
                        action.unsqueeze(-1),
                    ).squeeze(-1)
                    return selected_prob.clamp_min(1e-12).log()

                if checkpoint_selected_log_probs and torch.is_grad_enabled():
                    selected_log_prob = activation_checkpoint(
                        selected_log_prob_fn,
                        encoded_last_node,
                        ninf_mask,
                        selected,
                        *decoder_cache,
                        use_reentrant=False,
                    )
                else:
                    selected_log_prob = selected_log_prob_fn(
                        encoded_last_node,
                        ninf_mask,
                        selected,
                        *decoder_cache,
                    )
                if return_entropy:
                    raise ValueError(
                        "return_entropy is unsupported with forced_actions because the "
                        "memory-efficient path does not retain full probability tensors"
                    )
            log_probs.append(selected_log_prob)
            actions.append(selected)

            selected_flat = selected.transpose(0, 1).reshape(-1)
            td_flat.set("action", selected_flat)
            td_flat = env.step(td_flat)["next"]
            done = td_flat["done"]
            step_index += 1

        if (
            force_complete_trajectory
            and forced_actions_3d is not None
            and step_index != forced_actions_3d.shape[2]
        ):
            raise ValueError(
                "forced_actions length does not match the completed environment rollout: "
                f"used {step_index}, provided {forced_actions_3d.shape[2]}"
            )

        actions_3d = torch.stack(actions, dim=2)  # [B, S, T]
        log_probs_3d = torch.stack(log_probs, dim=2)  # [B, S, T]
        log_likelihood_2d = log_probs_3d.sum(dim=2)  # [B, S]

        actions_flat = actions_3d.permute(1, 0, 2).reshape(base_batch * num_starts, -1)
        if return_sum_log_likelihood:
            log_likelihood = log_likelihood_2d.transpose(0, 1).reshape(-1)
        else:
            log_likelihood = log_probs_3d.permute(1, 0, 2).reshape(base_batch * num_starts, -1)
        reward = env.get_reward(td_flat, actions_flat)

        out = {"reward": reward, "log_likelihood": log_likelihood}
        if return_actions:
            out["actions"] = actions_flat
        if return_entropy:
            entropy_2d = torch.stack(entropies, dim=2).sum(dim=2)
            out["entropy"] = entropy_2d.transpose(0, 1).reshape(-1)
        return out
