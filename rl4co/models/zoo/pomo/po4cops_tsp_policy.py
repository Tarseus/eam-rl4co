import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from rl4co.utils.ops import batchify, select_start_nodes, unbatchify


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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = self.embedding(data)
        for layer in self.layers:
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
        if self.mode == "feature":
            q = _reshape_by_heads(self.Wq_last(input_tensor), head_num=self.head_num)
            if self.first:
                q = q + self.q_first
            out_concat = _multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.add_and_norm_1(input_tensor, multi_head_out)
            out2 = self.feed_forward(out1)
            out3 = self.add_and_norm_2(out1, out2)
            return out3

        q = self.Wq_last(input_tensor)
        if self.first:
            q = q + self.q_first

        score = torch.matmul(q, self.logitk.transpose(1, 2))
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
        self.eval_type = eval_type
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
        **unused_kwargs,
    ) -> dict:
        if num_starts is None or num_starts <= 0:
            num_starts = env.get_num_starts(td)

        td_base = td.clone()
        base_batch = td_base.shape[0]

        # Match PO4COPs: k/v are computed once from the base batch [B, N, E]
        encoded_nodes = self.encoder(td_base["locs"])
        self.decoder.set_kv(encoded_nodes)

        # Environment rollout still uses flattened multistart batch [B*S, ...]
        td_flat = batchify(td_base, num_starts)
        first_action_flat = select_start_nodes(td_base, env, num_starts)
        td_flat.set("action", first_action_flat)
        td_flat = env.step(td_flat)["next"]

        first_action = unbatchify(first_action_flat, num_starts)
        current_node = unbatchify(td_flat["current_node"], num_starts)
        encoded_first_node = _get_encoding(encoded_nodes, current_node)
        self.decoder.set_q1(encoded_first_node)

        actions = [first_action]
        log_probs = [torch.zeros_like(first_action, dtype=encoded_nodes.dtype)]

        decode_type = getattr(self, f"{phase}_decode_type", "sampling")
        if decode_type.startswith("multistart_"):
            decode_type = decode_type[len("multistart_") :]
        use_sampling = (phase == "train") or (decode_type == "sampling") or (
            decode_type == "softmax"
        )

        done = td_flat["done"]
        while not done.all():
            current_node = unbatchify(td_flat["current_node"], num_starts)
            encoded_last_node = _get_encoding(encoded_nodes, current_node)

            action_mask = unbatchify(td_flat["action_mask"], num_starts)
            ninf_mask = torch.where(
                action_mask,
                torch.zeros_like(action_mask, dtype=encoded_nodes.dtype),
                float("-inf"),
            )
            probs = self.decoder(encoded_last_node, ninf_mask)

            if use_sampling:
                selected = probs.reshape(-1, probs.size(-1)).multinomial(1).squeeze(-1)
                selected = selected.view(probs.size(0), probs.size(1))
            else:
                selected = probs.argmax(dim=2)

            prob = probs.gather(2, selected.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            log_probs.append(prob.log())
            actions.append(selected)

            selected_flat = selected.transpose(0, 1).reshape(-1)
            td_flat.set("action", selected_flat)
            td_flat = env.step(td_flat)["next"]
            done = td_flat["done"]

        actions_3d = torch.stack(actions, dim=2)  # [B, S, T]
        log_likelihood_2d = torch.stack(log_probs, dim=2).sum(dim=2)  # [B, S]

        actions_flat = actions_3d.permute(1, 0, 2).reshape(base_batch * num_starts, -1)
        log_likelihood = log_likelihood_2d.transpose(0, 1).reshape(-1)
        reward = env.get_reward(td_flat, actions_flat)

        out = {"reward": reward, "log_likelihood": log_likelihood}
        if return_actions:
            out["actions"] = actions_flat
        return out
