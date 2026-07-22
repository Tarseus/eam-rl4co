from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rl4co.models import AttentionModelPolicy, SymNCOPolicy


METHODS = ("am", "symnco")
PROBLEMS = ("tsp", "cvrp")


def _original_am_state(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Expected an official AM checkpoint with a 'model' mapping: {path}")
    return payload["model"]


def convert_original_am_state(
    source: dict[str, Tensor], target: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Convert Kool et al.'s official AM weights to the RL4CO AM layout.

    The two implementations use the same computation but store multi-head
    projections differently. The official checkpoint keeps one matrix per head;
    RL4CO stores fused QKV matrices and standard ``nn.Linear`` weights.
    """

    converted = copy.deepcopy(target)

    def put(target_key: str, value: Tensor) -> None:
        if target_key not in converted:
            raise KeyError(f"Target policy does not contain {target_key}")
        value = value.detach().clone().to(dtype=converted[target_key].dtype)
        if value.shape != converted[target_key].shape:
            raise ValueError(
                f"Shape mismatch for {target_key}: source={tuple(value.shape)} "
                f"target={tuple(converted[target_key].shape)}"
            )
        converted[target_key] = value

    put("encoder.init_embedding.init_embed.weight", source["init_embed.weight"])
    put("encoder.init_embedding.init_embed.bias", source["init_embed.bias"])
    if "init_embed_depot.weight" in source:
        put(
            "encoder.init_embedding.init_embed_depot.weight",
            source["init_embed_depot.weight"],
        )
        put(
            "encoder.init_embedding.init_embed_depot.bias",
            source["init_embed_depot.bias"],
        )

    layer_ids = sorted(
        {
            int(key.split(".")[2])
            for key in source
            if key.startswith("embedder.layers.")
        }
    )
    for layer in layer_ids:
        src = f"embedder.layers.{layer}"
        dst = f"encoder.net.layers.{layer}"
        qkv = torch.cat(
            [
                source[f"{src}.0.module.W_query"].permute(0, 2, 1).reshape(128, 128),
                source[f"{src}.0.module.W_key"].permute(0, 2, 1).reshape(128, 128),
                source[f"{src}.0.module.W_val"].permute(0, 2, 1).reshape(128, 128),
            ],
            dim=0,
        )
        put(f"{dst}.0.module.Wqkv.weight", qkv)
        put(
            f"{dst}.0.module.Wqkv.bias",
            torch.zeros_like(converted[f"{dst}.0.module.Wqkv.bias"]),
        )
        put(
            f"{dst}.0.module.out_proj.weight",
            source[f"{src}.0.module.W_out"].reshape(128, 128).t(),
        )
        put(
            f"{dst}.0.module.out_proj.bias",
            torch.zeros_like(converted[f"{dst}.0.module.out_proj.bias"]),
        )
        for norm_index in (1, 3):
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                put(
                    f"{dst}.{norm_index}.normalizer.{suffix}",
                    source[f"{src}.{norm_index}.normalizer.{suffix}"],
                )
        put(f"{dst}.2.module.lins.0.weight", source[f"{src}.2.module.0.weight"])
        put(f"{dst}.2.module.lins.0.bias", source[f"{src}.2.module.0.bias"])
        put(f"{dst}.2.module.lins.1.weight", source[f"{src}.2.module.2.weight"])
        put(f"{dst}.2.module.lins.1.bias", source[f"{src}.2.module.2.bias"])

    if "W_placeholder" in source:
        put("decoder.context_embedding.W_placeholder", source["W_placeholder"])
    put(
        "decoder.context_embedding.project_context.weight",
        source["project_step_context.weight"],
    )
    put("decoder.pointer.project_out.weight", source["project_out.weight"])
    put(
        "decoder.project_node_embeddings.weight",
        source["project_node_embeddings.weight"],
    )
    put(
        "decoder.project_fixed_context.weight",
        source["project_fixed_context.weight"],
    )
    return converted


def build_policy(
    *, method: str, problem: str, init_checkpoint: Path | None = None
) -> tuple[AttentionModelPolicy | SymNCOPolicy, dict[str, Any]]:
    method = method.lower()
    problem = problem.lower()
    if method not in METHODS:
        raise ValueError(f"Unsupported method={method!r}; choose from {METHODS}")
    if problem not in PROBLEMS:
        raise ValueError(f"Unsupported problem={problem!r}; choose from {PROBLEMS}")

    payload = None
    if init_checkpoint is not None:
        init_checkpoint = init_checkpoint.resolve()
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)

    num_encoder_layers = 3
    normalization = "batch"
    policy_state = None
    if method == "symnco" and isinstance(payload, dict):
        if isinstance(payload.get("policy_state_dict"), dict):
            policy_state = payload["policy_state_dict"]
        elif isinstance(payload.get("state_dict"), dict):
            policy_state = {
                key.removeprefix("policy."): value
                for key, value in payload["state_dict"].items()
                if key.startswith("policy.")
            }
    if policy_state is not None:
        layer_ids = {
            int(key.split(".")[3])
            for key in policy_state
            if key.startswith("encoder.net.layers.")
        }
        if layer_ids:
            num_encoder_layers = max(layer_ids) + 1
        normalization = (
            "batch"
            if any(key.endswith("normalizer.running_mean") for key in policy_state)
            else "instance"
        )

    policy_cls = AttentionModelPolicy if method == "am" else SymNCOPolicy
    policy = policy_cls(
        env_name=problem,
        embed_dim=128,
        num_encoder_layers=num_encoder_layers,
        num_heads=8,
        normalization=normalization,
        train_decode_type=("sampling" if method == "am" else "multistart_sampling"),
        val_decode_type="multistart_greedy",
        test_decode_type="multistart_greedy",
    )
    metadata: dict[str, Any] = {
        "initialization": "random",
        "init_checkpoint": None,
    }
    if init_checkpoint is None:
        return policy, metadata

    init_checkpoint = init_checkpoint.resolve()
    metadata["init_checkpoint"] = str(init_checkpoint)

    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        converted = convert_original_am_state(payload["model"], policy.state_dict())
        policy.load_state_dict(converted, strict=True)
        metadata["initialization"] = (
            "official_am" if method == "am" else "official_am_policy_warmstart"
        )
        metadata["randomly_initialized_projection_head"] = method == "symnco"
        return policy, metadata

    if isinstance(payload, dict) and isinstance(payload.get("policy_state_dict"), dict):
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        metadata["initialization"] = "scale1000_resume_policy"
        return policy, metadata

    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state = {
            key.removeprefix("policy."): value
            for key, value in payload["state_dict"].items()
            if key.startswith("policy.")
        }
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if unexpected or missing:
            raise RuntimeError(
                f"RL4CO checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        metadata["initialization"] = "rl4co_checkpoint"
        return policy, metadata

    raise ValueError(f"Unsupported checkpoint structure: {init_checkpoint}")
