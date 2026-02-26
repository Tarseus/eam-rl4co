from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from fitness.free_loss_fidelity import PrefBatch

from .free_loss_compiler import _OpsAccessor, _build_operator_table, _validate_user_code
from .pref_builder_ir import PreferenceBuilderIR


BuildFn = Callable[[Mapping[str, torch.Tensor], Mapping[str, Any] | None], PrefBatch]


class PreferenceBuilderCompileError(Exception):
    pass


_REAL_IMPORT = __import__


def _restricted_import(
    name: str,
    globals: Mapping[str, Any] | None = None,  # noqa: A002
    locals: Mapping[str, Any] | None = None,  # noqa: A002
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Restrict imports during exec() of user/LLM-provided builder code.

    Some PyTorch APIs (e.g., Tensor.nonzero(as_tuple=True) in certain versions)
    consult the active frame's builtins for `__import__`. We provide a minimal
    importer so those operations can function, while keeping arbitrary imports
    disabled for sandboxed builder code.
    """

    mod = str(name or "")
    if mod == "torch" or mod.startswith("torch."):
        return _REAL_IMPORT(mod, globals, locals, fromlist, level)
    raise ImportError(f"Imports are disabled in preference builder sandbox (attempted: {mod!r})")


@dataclass
class CompiledPreferenceBuilder:
    ir: PreferenceBuilderIR
    build_fn: BuildFn


def validate_pref_batch(
    pref_batch: PrefBatch,
    feature_cache: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(pref_batch, PrefBatch):
        raise ValueError(f"pref_batch must be PrefBatch; got type={type(pref_batch)}")

    objective = feature_cache.get("objective")
    log_prob = feature_cache.get("log_prob")
    if not isinstance(objective, torch.Tensor) or not isinstance(log_prob, torch.Tensor):
        raise ValueError("feature_cache must contain tensor keys: objective, log_prob")
    if objective.ndim != 2:
        raise ValueError(f"feature_cache['objective'] must be (B,K); got shape={tuple(objective.shape)}")
    if log_prob.shape != objective.shape:
        raise ValueError(
            f"feature_cache['log_prob'] must match objective shape; "
            f"log_prob={tuple(log_prob.shape)} objective={tuple(objective.shape)}"
        )

    mode = str(pref_batch.mode or "").strip().lower()
    if mode not in {"pairwise", "setwise", "listwise"}:
        raise ValueError(f"PrefBatch.mode invalid: {pref_batch.mode!r}")

    if mode == "pairwise":
        if pref_batch.pair_idx is None:
            raise ValueError("PrefBatch.pair_idx is required for pairwise mode")
        if not isinstance(pref_batch.pair_idx, tuple) or len(pref_batch.pair_idx) != 3:
            raise ValueError("PrefBatch.pair_idx must be a tuple of (b_idx, winner_idx, loser_idx)")

        b_idx, winner_idx, loser_idx = pref_batch.pair_idx
        for name, t in (("b_idx", b_idx), ("winner_idx", winner_idx), ("loser_idx", loser_idx)):
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"PrefBatch.pair_idx[{name}] must be a torch.Tensor")
            if t.dtype not in (torch.int64, torch.long):
                raise ValueError(f"PrefBatch.pair_idx[{name}] must be int64/long; got {t.dtype}")
            if t.ndim != 1:
                raise ValueError(f"PrefBatch.pair_idx[{name}] must be 1D; got shape={tuple(t.shape)}")

        if not (b_idx.numel() == winner_idx.numel() == loser_idx.numel()):
            raise ValueError("PrefBatch.pair_idx tensors must have the same length")

        B, K = int(objective.shape[0]), int(objective.shape[1])
        if b_idx.numel() > 0:
            if int(b_idx.min().item()) < 0 or int(b_idx.max().item()) >= B:
                raise ValueError("PrefBatch.pair_idx batch indices out of range")
            for name, t in (("winner_idx", winner_idx), ("loser_idx", loser_idx)):
                if int(t.min().item()) < 0 or int(t.max().item()) >= K:
                    raise ValueError(f"PrefBatch.pair_idx {name} out of range")

        weight = pref_batch.weight
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                raise ValueError("PrefBatch.weight must be a torch.Tensor when provided")
            if weight.ndim != 1:
                raise ValueError(f"PrefBatch.weight must be 1D; got shape={tuple(weight.shape)}")
            if weight.numel() != b_idx.numel():
                raise ValueError(
                    f"PrefBatch.weight length mismatch: weight={int(weight.numel())} pairs={int(b_idx.numel())}"
                )
            if not torch.isfinite(weight).all().item():
                raise ValueError("PrefBatch.weight must be finite")


def compile_preference_builder(
    ir: PreferenceBuilderIR,
    *,
    operator_whitelist: Sequence[str] | None = None,
) -> CompiledPreferenceBuilder:
    code_str = (ir.code or "").strip()
    if not code_str:
        raise PreferenceBuilderCompileError("PreferenceBuilderIR.code is empty.")

    try:
        _validate_user_code(code_str)
    except Exception as exc:  # noqa: BLE001
        raise PreferenceBuilderCompileError(f"Builder static code validation failed: {exc}") from exc

    ops_table = _build_operator_table()
    if operator_whitelist:
        ops_table = {k: v for k, v in ops_table.items() if k in operator_whitelist}
    ops_accessor = _OpsAccessor(ops_table)

    # Execute in a tightly restricted namespace. We deliberately strip most
    # builtins to avoid access to filesystem, subprocesses, etc.
    #
    # NOTE: We keep a tiny allowlist of safe builtins to make LLM/codegen
    # builders ergonomic (e.g., `float(...)`, `int(...)`, simple loops). Dangerous
    # builtins like `open`, `eval`, `exec`, `__import__`, etc. remain absent,
    # and additional safety is enforced by the AST validator above.
    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "__import__": _restricted_import,
            "float": float,
            "int": int,
            "bool": bool,
            "dict": dict,
            "list": list,
            "tuple": tuple,
            "set": set,
            "min": min,
            "max": max,
            "abs": abs,
            "len": len,
            "sum": sum,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "isinstance": isinstance,
        },
        "torch": torch,
        "F": F,
        "ops": ops_accessor,
        "PrefBatch": PrefBatch,
    }
    local_ns: Dict[str, Any] = {}
    try:
        exec(code_str, safe_globals, local_ns)
    except Exception as exc:  # noqa: BLE001
        raise PreferenceBuilderCompileError(f"Failed to exec builder code from IR: {exc}") from exc

    fn = local_ns.get("generated_builder")
    if not callable(fn):
        raise PreferenceBuilderCompileError(
            "Builder code did not define a callable 'generated_builder(feature_cache, extra)'."
        )

    def build_fn(
        feature_cache: Mapping[str, torch.Tensor],
        extra: Mapping[str, Any] | None,
    ) -> PrefBatch:
        merged_extra: Dict[str, Any] = {
            "ops": ops_accessor,
            "operators": ops_accessor,
            "torch": torch,
            "F": F,
            "torch.nn.functional": F,
        }
        if extra:
            merged_extra.update(dict(extra))
        out = fn(feature_cache, merged_extra)
        validate_pref_batch(out, feature_cache)
        return out

    return CompiledPreferenceBuilder(ir=ir, build_fn=build_fn)
