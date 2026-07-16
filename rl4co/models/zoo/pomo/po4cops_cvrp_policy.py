import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from tensordict import TensorDict

from rl4co.models.zoo.pomo.po4cops_tsp_policy import (
    _AddAndNorm,
    _EncoderLayer,
    _FeedForward,
    _get_encoding,
    _multi_head_attention,
    _reshape_by_heads,
    _select_actions_from_probs,
)
from rl4co.utils.ops import batchify, select_start_nodes, unbatchify


class _PO4COPsCVRPEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        encoder_layer_num: int,
        head_num: int,
        qkv_dim: int,
        ff_hidden_dim: int,
    ):
        super().__init__()
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList(
            [
                _EncoderLayer(embedding_dim, head_num, qkv_dim, ff_hidden_dim)
                for _ in range(encoder_layer_num)
            ]
        )

    def forward(
        self,
        depot_xy: torch.Tensor,
        node_xy_demand: torch.Tensor,
        checkpoint_layers: bool = False,
    ) -> torch.Tensor:
        out = torch.cat(
            (self.embedding_depot(depot_xy), self.embedding_node(node_xy_demand)),
            dim=1,
        )
        for layer in self.layers:
            if checkpoint_layers and torch.is_grad_enabled():
                out = activation_checkpoint(layer, out, use_reentrant=False)
            else:
                out = layer(out)
        return out


class _CVRPFeatureDecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        ff_hidden_dim: int,
    ):
        super().__init__()
        self.head_num = head_num
        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.add_and_norm_1 = _AddAndNorm(embedding_dim)
        self.feed_forward = _FeedForward(embedding_dim, ff_hidden_dim)
        self.add_and_norm_2 = _AddAndNorm(embedding_dim)

        self.k = None
        self.v = None

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        self.k = _reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = _reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)

    def forward(
        self,
        input_tensor: torch.Tensor,
        load: torch.Tensor,
        ninf_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_cache(
            input_tensor,
            load,
            ninf_mask,
            self.k,
            self.v,
        )

    def forward_with_cache(
        self,
        input_tensor: torch.Tensor,
        load: torch.Tensor,
        ninf_mask: torch.Tensor | None,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        input_cat = torch.cat((input_tensor, load[..., None]), dim=2)
        q = _reshape_by_heads(self.Wq_last(input_cat), head_num=self.head_num)
        out_concat = _multi_head_attention(q, k, v, rank3_ninf_mask=ninf_mask)
        multi_head_out = self.multi_head_combine(out_concat)
        out1 = self.add_and_norm_1(input_tensor, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_and_norm_2(out1, out2)
        return out3


class _CVRPLogitDecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        sqrt_embedding_dim: float,
        logit_clipping: float,
    ):
        super().__init__()
        self.sqrt_embedding_dim = sqrt_embedding_dim
        self.logit_clipping = logit_clipping
        self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wlogit_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.logitk = None

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        self.logitk = self.Wlogit_k(encoded_nodes)

    def forward(self, input_tensor: torch.Tensor, ninf_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cache(input_tensor, ninf_mask, self.logitk)

    def forward_with_cache(
        self,
        input_tensor: torch.Tensor,
        ninf_mask: torch.Tensor,
        logitk: torch.Tensor,
    ) -> torch.Tensor:
        q = self.Wq_last(input_tensor)
        score = torch.matmul(q, logitk.transpose(1, 2))
        score_scaled = score / self.sqrt_embedding_dim
        score_clipped = self.logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        return F.softmax(score_masked, dim=2)


class _PO4COPsCVRPDecoder(nn.Module):
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
        self.feature_layers = nn.ModuleList(
            [
                _CVRPFeatureDecoderLayer(
                    embedding_dim=embedding_dim,
                    head_num=head_num,
                    qkv_dim=qkv_dim,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(decoder_layer_num)
            ]
        )
        self.logit_layer = _CVRPLogitDecoderLayer(
            embedding_dim=embedding_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            logit_clipping=logit_clipping,
        )

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        for layer in self.feature_layers:
            layer.set_kv(encoded_nodes)
        self.logit_layer.set_kv(encoded_nodes)

    def forward(
        self,
        encoded_last_node: torch.Tensor,
        load: torch.Tensor,
        ninf_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = encoded_last_node
        for layer in self.feature_layers:
            out = layer(out, load, ninf_mask)
        return self.logit_layer(out, ninf_mask)

    def cache_tensors(self) -> tuple[torch.Tensor, ...]:
        cache: list[torch.Tensor] = []
        for layer in self.feature_layers:
            cache.extend((layer.k, layer.v))
        cache.append(self.logit_layer.logitk)
        return tuple(cache)

    def forward_with_cache(
        self,
        encoded_last_node: torch.Tensor,
        load: torch.Tensor,
        ninf_mask: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        expected = 2 * len(self.feature_layers) + 1
        if len(cache_tensors) != expected:
            raise ValueError(
                f"Expected {expected} decoder cache tensors, got {len(cache_tensors)}"
            )
        out = encoded_last_node
        offset = 0
        for layer in self.feature_layers:
            k, v = cache_tensors[offset : offset + 2]
            offset += 2
            out = layer.forward_with_cache(out, load, ninf_mask, k, v)
        return self.logit_layer.forward_with_cache(
            out,
            ninf_mask,
            cache_tensors[offset],
        )


class PO4COPsCVRPPolicy(nn.Module):
    """PO4COPs-compatible CVRP policy aligned with the original POMO CVRP model."""

    def __init__(
        self,
        env_name: str = "cvrp",
        embedding_dim: int = 128,
        encoder_layer_num: int = 6,
        decoder_layer_num: int = 1,
        qkv_dim: int = 16,
        head_num: int = 8,
        ff_hidden_dim: int = 512,
        logit_clipping: float = 50.0,
        eval_type: str = "argmax",
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kwargs,
    ):
        super().__init__()
        if env_name != "cvrp":
            raise ValueError("PO4COPsCVRPPolicy currently supports only CVRP.")

        self.env_name = env_name
        self.eval_type = str(eval_type).strip().lower()
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        sqrt_embedding_dim = math.sqrt(float(embedding_dim))

        self.encoder = _PO4COPsCVRPEncoder(
            embedding_dim=embedding_dim,
            encoder_layer_num=encoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            ff_hidden_dim=ff_hidden_dim,
        )
        self.decoder = _PO4COPsCVRPDecoder(
            embedding_dim=embedding_dim,
            decoder_layer_num=decoder_layer_num,
            head_num=head_num,
            qkv_dim=qkv_dim,
            sqrt_embedding_dim=sqrt_embedding_dim,
            logit_clipping=logit_clipping,
            ff_hidden_dim=ff_hidden_dim,
        )

        self.encoded_nodes = None

    @staticmethod
    def _split_inputs(td: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        depot_xy = td["locs"][:, :1, :]
        node_xy_demand = torch.cat((td["locs"][:, 1:, :], td["demand"][..., None]), dim=2)
        return depot_xy, node_xy_demand

    def pre_forward(self, reset_td: TensorDict) -> None:
        depot_xy, node_xy_demand = self._split_inputs(reset_td)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
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
        checkpoint_encoder_layers: bool = False,
        checkpoint_selected_log_probs: bool = False,
        **unused_kwargs,
    ) -> dict:
        if num_starts is None or num_starts <= 0:
            num_starts = env.get_num_starts(td)

        td_base = td.clone()
        base_batch = td_base.shape[0]
        depot_xy, node_xy_demand = self._split_inputs(td_base)

        encoded_nodes = self.encoder(
            depot_xy,
            node_xy_demand,
            checkpoint_layers=checkpoint_encoder_layers,
        )
        self.decoder.set_kv(encoded_nodes)

        forced_actions_3d = None
        if forced_actions is not None:
            if forced_actions.ndim == 2:
                if forced_actions.shape[0] != base_batch * num_starts:
                    raise ValueError(
                        "Flattened forced_actions must have leading dimension "
                        f"{base_batch * num_starts}, got {forced_actions.shape[0]}"
                    )
                forced_actions_3d = forced_actions.reshape(
                    num_starts,
                    base_batch,
                    -1,
                ).permute(1, 0, 2)
            elif forced_actions.ndim == 3:
                if forced_actions.shape[:2] != (base_batch, num_starts):
                    raise ValueError(
                        "Rank-3 forced_actions must have shape "
                        f"[{base_batch}, {num_starts}, T], got {tuple(forced_actions.shape)}"
                    )
                forced_actions_3d = forced_actions
            else:
                raise ValueError(
                    f"forced_actions must have rank 2 or 3, got rank {forced_actions.ndim}"
                )
            forced_actions_3d = forced_actions_3d.to(
                device=td_base.device,
                dtype=torch.long,
            )
            if forced_actions_3d.shape[2] < 2:
                raise ValueError("CVRP forced_actions must contain depot and start actions")

        td_flat = batchify(td_base, num_starts)

        if forced_actions_3d is None:
            depot_action_flat = torch.zeros(
                base_batch * num_starts, dtype=torch.long, device=td.device
            )
        else:
            depot_action_flat = forced_actions_3d[:, :, 0].transpose(0, 1).reshape(-1)
            if (depot_action_flat != 0).any():
                raise ValueError("CVRP forced replay must begin at depot action 0")
        td_flat.set("action", depot_action_flat)
        td_flat = env.step(td_flat)["next"]

        if forced_actions_3d is None:
            start_action_flat = select_start_nodes(td_base, env, num_starts)
        else:
            start_action_flat = forced_actions_3d[:, :, 1].transpose(0, 1).reshape(-1)
        td_flat.set("action", start_action_flat)
        td_flat = env.step(td_flat)["next"]

        depot_action = unbatchify(depot_action_flat, num_starts)
        start_action = unbatchify(start_action_flat, num_starts)
        actions = [depot_action, start_action]
        log_probs = [
            torch.zeros_like(depot_action, dtype=encoded_nodes.dtype),
            torch.zeros_like(start_action, dtype=encoded_nodes.dtype),
        ]
        entropies = [
            torch.zeros_like(depot_action, dtype=encoded_nodes.dtype),
            torch.zeros_like(start_action, dtype=encoded_nodes.dtype),
        ]
        decoder_cache = self.decoder.cache_tensors()

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
        step_index = 2
        while not done.all():
            current_node = unbatchify(td_flat["current_node"], num_starts)
            encoded_last_node = _get_encoding(encoded_nodes, current_node)
            remaining_load = unbatchify(
                td_flat["vehicle_capacity"] - td_flat["used_capacity"], num_starts
            ).squeeze(-1)

            action_mask = unbatchify(td_flat["action_mask"], num_starts)
            ninf_mask = torch.where(
                action_mask,
                torch.zeros_like(action_mask, dtype=encoded_nodes.dtype),
                float("-inf"),
            )
            if forced_actions_3d is None:
                probs = self.decoder(encoded_last_node, remaining_load, ninf_mask)

                if use_sampling:
                    while True:
                        selected = _select_actions_from_probs(probs, use_sampling, use_hybrid)
                        prob = probs.gather(2, selected.unsqueeze(-1)).squeeze(-1)
                        if (prob > 0).all():
                            break
                else:
                    selected = probs.argmax(dim=2)
                    prob = probs.gather(2, selected.unsqueeze(-1)).squeeze(-1)

                prob = prob.clamp_min(1e-12)
                selected_log_prob = prob.log()
                if return_entropy:
                    probs_safe = probs.clamp_min(1e-12)
                    entropies.append(-(probs_safe * probs_safe.log()).sum(dim=2))
            else:
                if step_index >= forced_actions_3d.shape[2]:
                    raise ValueError(
                        "forced_actions ended before the environment rollout completed"
                    )
                selected = forced_actions_3d[:, :, step_index]

                def selected_log_prob_fn(
                    last_node: torch.Tensor,
                    load: torch.Tensor,
                    mask: torch.Tensor,
                    action: torch.Tensor,
                    *cache: torch.Tensor,
                ) -> torch.Tensor:
                    action_probs = self.decoder.forward_with_cache(
                        last_node,
                        load,
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
                        remaining_load,
                        ninf_mask,
                        selected,
                        *decoder_cache,
                        use_reentrant=False,
                    )
                else:
                    selected_log_prob = selected_log_prob_fn(
                        encoded_last_node,
                        remaining_load,
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

        if forced_actions_3d is not None and step_index != forced_actions_3d.shape[2]:
            raise ValueError(
                "forced_actions length does not match the completed environment rollout: "
                f"used {step_index}, provided {forced_actions_3d.shape[2]}"
            )

        actions_3d = torch.stack(actions, dim=2)
        log_probs_3d = torch.stack(log_probs, dim=2)
        log_likelihood_2d = log_probs_3d.sum(dim=2)

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
