from __future__ import annotations

import csv
from contextlib import nullcontext
from dataclasses import dataclass, asdict, field
import math
import os
import re
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple

import logging
import torch
from torch.optim import Adam

from .co_features import build_model_output, gather_pairwise_deltas
from .ptp_high_fidelity import (
    DEFAULT_LOSS_OBSERVABLES,
    HighFidelityConfig,
    _set_seed,
    resolve_pomo_size,
    aggregate_objectives_by_size,
    get_hf_epoch_plan,
    get_total_hf_train_steps,
)
from ptp_discovery.free_loss_compiler import CompiledFreeLoss


logger = logging.getLogger(__name__)


_OFFLINE_TENSORDICT_CACHE: dict[str, Any] = {}
_PAIRWISE_OPTIONAL_KEY_FAMILIES: dict[str, tuple[str, str, str]] = {
    "seq_len": ("seq_len_w", "seq_len_l", "seq_len_gap"),
    "log_prob_mean": ("log_prob_w_mean", "log_prob_l_mean", "log_prob_mean_gap"),
    "advantage": ("advantage_w", "advantage_l", "advantage_gap"),
    "entropy": ("entropy_w", "entropy_l", "entropy_gap"),
    "entropy_mean": ("entropy_w_mean", "entropy_l_mean", "entropy_mean_gap"),
    "log_prob_ref": ("log_prob_ref_w", "log_prob_ref_l", "log_prob_ref_gap"),
    "log_prob_ratio": ("log_prob_ratio_w", "log_prob_ratio_l", "log_prob_ratio_gap"),
}


def _normalize_precision_mode(value: str | None) -> str:
    mode = str(value or "32-true").strip().lower()
    if mode in {"16", "16-mixed", "fp16", "fp16-mixed"}:
        return "16-mixed"
    if mode in {"bf16", "bf16-mixed"}:
        return "bf16-mixed"
    return "32-true"


def _normalize_po_impl(value: str | None) -> str:
    impl = str(value or "bt").strip().lower()
    if impl not in {"bt", "exponential"}:
        return "bt"
    return impl


def _autocast_context(device: torch.device, precision: str):
    mode = _normalize_precision_mode(precision)
    if device.type != "cuda":
        return nullcontext()
    if mode == "16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mode == "bf16-mixed":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and _normalize_precision_mode(precision) == "16-mixed"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # noqa: BLE001
        try:
            return torch.cuda.amp.GradScaler(enabled=enabled)
        except Exception:  # noqa: BLE001
            return None


def _abs_path(path: str) -> str:
    if not path:
        return path
    return os.path.abspath(os.path.expanduser(str(path)))


def _load_offline_tensordict(path: str):
    """Load an offline instance TensorDict/dict to CPU with a global in-process cache."""

    from tensordict import TensorDict, TensorDictBase

    p = _abs_path(str(path))
    cached = _OFFLINE_TENSORDICT_CACHE.get(p)
    if cached is not None:
        return cached

    if not os.path.exists(p):
        raise FileNotFoundError(f"Offline instances file not found: {path} (abs={p})")

    # Offline instance files are full TensorDict payloads, not weight-only checkpoints.
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, TensorDictBase):
        td = obj.to("cpu")
    elif isinstance(obj, dict):
        # Best-effort: wrap dict[str, Tensor] into a TensorDict.
        if not obj:
            raise ValueError(f"Offline instances dict is empty: {path}")
        first = next(iter(obj.values()))
        if not isinstance(first, torch.Tensor):
            raise TypeError(f"Offline instances dict must contain tensors: {path}")
        n = int(first.shape[0])
        td = TensorDict({str(k): v.to("cpu") for k, v in obj.items()}, batch_size=[n])
    else:
        raise TypeError(
            f"Offline instances must be a TensorDict or dict[str, Tensor], got {type(obj)}: {path}"
        )

    _OFFLINE_TENSORDICT_CACHE[p] = td
    return td


class OfflineSplitGenerator:
    """Offline generator wrapper with deterministic, no-shuffle sequential batches.

    This is used to fully offline-ize RL4CO training/validation in stage3 mini-train.

    - Each new generator instance starts cursors at 0, ensuring identical batch sequences
      across different candidates.
    - Data is always loaded to CPU; callers can move the returned TensorDict to GPU.
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        device: str = "cpu",
        extra_attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.train_path = str(train_path)
        self.val_path = str(val_path)
        self.device = str(device or "cpu")

        self._train_td = _load_offline_tensordict(self.train_path)
        self._val_td = _load_offline_tensordict(self.val_path)

        self.train_cursor = 0
        self.val_cursor = 0
        self._split = "train"

        if isinstance(extra_attrs, Mapping):
            for key, value in dict(extra_attrs).items():
                if not key or str(key).startswith("_"):
                    continue
                setattr(self, str(key), value)

    def set_split(self, phase: str) -> None:
        p = str(phase or "train").strip().lower()
        self._split = "train" if p == "train" else "val"

    def _slice_wrap(self, td, cursor: int, batch_size: int):
        n = int(getattr(td, "batch_size", [0])[0] if hasattr(td, "batch_size") else 0)
        if n <= 0:
            raise ValueError("OfflineSplitGenerator got empty dataset")
        b = max(int(batch_size), 1)
        cur = int(cursor) % n
        end = cur + b
        if end <= n:
            out = td[cur:end]
            new_cursor = end % n
        else:
            first = td[cur:n]
            second = td[0 : (end - n)]
            out = torch.cat([first, second], dim=0)
            new_cursor = (end - n) % n
        return out.to(self.device) if self.device else out, int(new_cursor)

    def __call__(self, batch_size: int):
        if self._split == "train":
            out, self.train_cursor = self._slice_wrap(self._train_td, self.train_cursor, batch_size)
            return out
        out, self.val_cursor = self._slice_wrap(self._val_td, self.val_cursor, batch_size)
        return out


def _infer_pairwise_reference_tensor(batch: Mapping[str, Any]) -> torch.Tensor | None:
    for key in (
        "log_prob_w",
        "log_prob_l",
        "weight",
        "cost_a",
        "cost_b",
        "cost_gap",
        "delta_z",
        "delta_rank",
        "delta_regret",
    ):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _ensure_pairwise_gap_family(
    batch: Dict[str, torch.Tensor],
    *,
    winner_key: str,
    loser_key: str,
    gap_key: str,
) -> None:
    winner = batch.get(winner_key)
    loser = batch.get(loser_key)
    gap = batch.get(gap_key)

    if not isinstance(gap, torch.Tensor) and isinstance(winner, torch.Tensor) and isinstance(loser, torch.Tensor):
        gap = loser - winner
        batch[gap_key] = gap
    if not isinstance(winner, torch.Tensor) and isinstance(loser, torch.Tensor) and isinstance(gap, torch.Tensor):
        winner = loser - gap
        batch[winner_key] = winner
    if not isinstance(loser, torch.Tensor) and isinstance(winner, torch.Tensor) and isinstance(gap, torch.Tensor):
        loser = winner + gap
        batch[loser_key] = loser


def prepare_pairwise_loss_batch(
    full_batch: Mapping[str, Any],
    expects: Sequence[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """Filter a full pairwise batch to the requested keys and fill safe derived signals.

    This keeps gates and runtime mini-train on the same schema/fallback rules so that
    candidate validation cannot silently diverge from actual execution.
    """

    batch_full: Dict[str, torch.Tensor] = {
        str(key): value for key, value in full_batch.items() if isinstance(value, torch.Tensor)
    }
    ref = _infer_pairwise_reference_tensor(batch_full)

    if (
        "cost_gap" not in batch_full
        and isinstance(batch_full.get("cost_a"), torch.Tensor)
        and isinstance(batch_full.get("cost_b"), torch.Tensor)
    ):
        batch_full["cost_gap"] = batch_full["cost_b"] - batch_full["cost_a"]

    if (
        "advantage_w" not in batch_full
        and isinstance(batch_full.get("cost_a"), torch.Tensor)
        and "advantage_gap" not in batch_full
    ):
        batch_full["advantage_w"] = -batch_full["cost_a"]
    if (
        "advantage_l" not in batch_full
        and isinstance(batch_full.get("cost_b"), torch.Tensor)
        and "advantage_gap" not in batch_full
    ):
        batch_full["advantage_l"] = -batch_full["cost_b"]
    if "advantage_gap" not in batch_full:
        if isinstance(batch_full.get("advantage_w"), torch.Tensor) and isinstance(batch_full.get("advantage_l"), torch.Tensor):
            batch_full["advantage_gap"] = batch_full["advantage_l"] - batch_full["advantage_w"]
        elif isinstance(batch_full.get("cost_gap"), torch.Tensor):
            batch_full["advantage_gap"] = -batch_full["cost_gap"]
        elif (
            isinstance(batch_full.get("cost_a"), torch.Tensor)
            and isinstance(batch_full.get("cost_b"), torch.Tensor)
        ):
            batch_full["advantage_gap"] = batch_full["cost_a"] - batch_full["cost_b"]
        elif isinstance(ref, torch.Tensor):
            batch_full["advantage_gap"] = torch.zeros_like(ref)

    for winner_key, loser_key, gap_key in _PAIRWISE_OPTIONAL_KEY_FAMILIES.values():
        _ensure_pairwise_gap_family(
            batch_full,
            winner_key=winner_key,
            loser_key=loser_key,
            gap_key=gap_key,
        )

    if "weight" not in batch_full and isinstance(ref, torch.Tensor):
        batch_full["weight"] = torch.ones_like(ref)

    requested = [str(key).strip() for key in (expects or []) if str(key).strip()]
    if not requested:
        return dict(batch_full)

    out: Dict[str, torch.Tensor] = {key: batch_full[key] for key in requested if key in batch_full}
    if isinstance(ref, torch.Tensor):
        for key in requested:
            if key in out:
                continue
            if key in {"advantage_w", "advantage_l", "advantage_gap"}:
                out[key] = torch.zeros_like(ref)
            elif key == "weight":
                out[key] = torch.ones_like(ref)
    return out


@dataclass
class PrefBatch:
    """Intermediate preference batch built from a fixed feature_cache.

    This keeps the training loop modular:
        rollout -> feature_cache -> pref_builder -> PrefBatch -> compiled loss
    """

    mode: str  # "pairwise" | "setwise" | "listwise"
    # Pairwise indices: (batch_idx, winner_idx, loser_idx), each (P,)
    pair_idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    # Optional listwise indices (e.g., sorted or selected solution ids), shape is builder-defined.
    list_idx: torch.Tensor | None = None
    # Optional per-example weights, shape depends on mode (pairwise: (P,))
    weight: torch.Tensor | None = None
    # Arbitrary metadata (non-tensor), e.g., builder name / sampling params.
    meta: Dict[str, Any] = field(default_factory=dict)

    def num_examples(self) -> int:
        if self.mode == "pairwise" and self.pair_idx is not None:
            return int(self.pair_idx[0].numel())
        if self.list_idx is not None:
            return int(self.list_idx.numel())
        return 0

    def to_pairwise_loss_batch(
        self, feature_cache: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.mode != "pairwise":
            raise ValueError(f"PrefBatch.to_pairwise_loss_batch called for mode={self.mode}")
        if self.pair_idx is None:
            raise ValueError("pair_idx is required for pairwise mode")

        b_idx, winner_idx, loser_idx = self.pair_idx
        objective = feature_cache["objective"]
        log_prob = feature_cache["log_prob"]

        cost_a_tensor = objective[b_idx, winner_idx]
        cost_b_tensor = objective[b_idx, loser_idx]
        logp_w_tensor = log_prob[b_idx, winner_idx]
        logp_l_tensor = log_prob[b_idx, loser_idx]

        features: Dict[str, torch.Tensor] = {}
        for key in ("obj_z", "rank", "regret"):
            value = feature_cache.get(key)
            if isinstance(value, torch.Tensor):
                features[key] = value
        pairwise_deltas = gather_pairwise_deltas(
            features, b_idx=b_idx, winner_idx=winner_idx, loser_idx=loser_idx
        )

        weight = self.weight
        if weight is None:
            weight = torch.ones_like(logp_w_tensor)

        batch = {
            "cost_a": cost_a_tensor,
            "cost_b": cost_b_tensor,
            "cost_gap": cost_b_tensor - cost_a_tensor,
            "log_prob_w": logp_w_tensor,
            "log_prob_l": logp_l_tensor,
            **pairwise_deltas,
            "weight": weight,
        }
        for key, (winner_key, loser_key, gap_key) in _PAIRWISE_OPTIONAL_KEY_FAMILIES.items():
            value = feature_cache.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            winner_value = value[b_idx, winner_idx]
            loser_value = value[b_idx, loser_idx]
            batch[winner_key] = winner_value
            batch[loser_key] = loser_value
            batch[gap_key] = loser_value - winner_value

        step_log_prob = feature_cache.get("log_prob_step")
        if isinstance(step_log_prob, torch.Tensor):
            batch["log_prob_step_w"] = step_log_prob[b_idx, winner_idx]
            batch["log_prob_step_l"] = step_log_prob[b_idx, loser_idx]

        return batch


def normalize_loss_observables(observables: Sequence[str] | None) -> Tuple[str, ...]:
    values = observables if observables else DEFAULT_LOSS_OBSERVABLES
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        key = str(raw or "").strip()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return tuple(out) if out else DEFAULT_LOSS_OBSERVABLES


def build_runtime_observables(
    reward: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    observables: Sequence[str] | None,
    seq_len: torch.Tensor | None = None,
    log_prob_step: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    seq_len_fallback: int | float | None = None,
) -> Dict[str, torch.Tensor]:
    obs_set = set(normalize_loss_observables(observables))
    extra: Dict[str, torch.Tensor] = {}

    need_seq_len = bool(obs_set & {"seq_len", "log_prob_mean", "entropy_mean"})
    seq_len_tensor: torch.Tensor | None = None
    if need_seq_len:
        if isinstance(seq_len, torch.Tensor):
            seq_len_tensor = seq_len.to(device=log_prob.device, dtype=log_prob.dtype)
        else:
            fallback = float(seq_len_fallback if seq_len_fallback is not None else 1.0)
            seq_len_tensor = torch.full_like(log_prob, fallback)
        if "seq_len" in obs_set:
            extra["seq_len"] = seq_len_tensor

    if "log_prob_mean" in obs_set:
        seq_len_safe = seq_len_tensor.clamp_min(1.0) if seq_len_tensor is not None else torch.ones_like(log_prob)
        extra["log_prob_mean"] = log_prob / seq_len_safe
    if "advantage" in obs_set:
        extra["advantage"] = reward - reward.mean(dim=1, keepdim=True)
    if "log_prob_step" in obs_set and isinstance(log_prob_step, torch.Tensor):
        extra["log_prob_step"] = log_prob_step
    if isinstance(entropy, torch.Tensor):
        if "entropy" in obs_set:
            extra["entropy"] = entropy
        if "entropy" in obs_set or "entropy_mean" in obs_set:
            if seq_len_tensor is None:
                fallback = float(seq_len_fallback if seq_len_fallback is not None else 1.0)
                seq_len_tensor = torch.full_like(log_prob, fallback)
                if "seq_len" in obs_set and "seq_len" not in extra:
                    extra["seq_len"] = seq_len_tensor
            extra["entropy_mean"] = entropy / seq_len_tensor.clamp_min(1.0)
    return extra


class PrefBuilder(Protocol):
    def build(
        self,
        feature_cache: Mapping[str, torch.Tensor],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> PrefBatch: ...


def extract_feature_cache(
    objective: torch.Tensor,
    log_prob: torch.Tensor,
    *,
    extra: Mapping[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    """Build the fixed feature cache used by preference builders and losses.

    This function intentionally reuses `co_features.build_model_output` to keep
    feature definitions in one place.
    """

    model_output, _ = build_model_output(objective=objective, log_prob=log_prob)
    cache: Dict[str, torch.Tensor] = dict(model_output)
    if extra:
        for k, v in extra.items():
            if isinstance(v, torch.Tensor):
                cache[str(k)] = v
    return cache


class _AllPairsPrefBuilder:
    """Default preference builder: equivalent to the previous all-pairs logic."""

    def build(
        self,
        feature_cache: Mapping[str, torch.Tensor],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> PrefBatch:
        objective = feature_cache["objective"]
        (b_idx, winner_idx, loser_idx), _ = _build_preference_pairs(objective)
        out_meta: Dict[str, Any] = {"builder": "all_pairs"}
        if meta:
            out_meta.update(dict(meta))
        return PrefBatch(
            mode="pairwise",
            pair_idx=(b_idx, winner_idx, loser_idx),
            weight=None,
            meta=out_meta,
        )


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        value = float(val)
        self.val = value
        self.sum += value * n
        self.count += int(n)
        if self.count > 0:
            self.avg = self.sum / self.count


@dataclass
class FreeLossFidelityConfig:
    hf: HighFidelityConfig
    f1_steps: int = 32
    f2_steps: int = 0
    f3_enabled: bool = False
    # Optional warm start for HF training. Intended for continuing from an existing
    # RL4CO Lightning checkpoint (e.g., `baseline/epoch_409.ckpt`).
    init_checkpoint_path: str | None = None
    init_checkpoint_epoch: int | None = None
    # Optional split evaluation:
    # - scratch_hf_epochs: train/eval from random init for N epochs (no checkpoint load)
    # - warmstart_hf_epochs: train/eval from init_checkpoint_path for N epochs
    # When either is > 0, baseline comparisons and hf_like_score use the warm-start phase
    # when present; scratch results are reported separately for diagnostics.
    scratch_hf_epochs: int = 0
    warmstart_hf_epochs: int = 0
    # Optional offset into baseline_epoch_objectives when comparing against baseline.
    # Useful when you want warmstart_hf_epochs to align to a later slice of a longer baseline.
    baseline_epoch_compare_offset: int = 0
    baseline_epoch_violation_weight: float = 1.0
    # Fraction of epochs (from the end) used for "better-than-baseline" checks.
    # Example: 0.9 means the last 90% epochs must beat the baseline, ignoring the first 10%.
    # Default 1.0 preserves the historical "all epochs must beat baseline" behavior.
    baseline_epoch_tail_frac: float = 1.0
    # Windowed comparison against the baseline's epoch validation objectives.
    # When baseline epoch objectives are available, we compare the mean objective
    # in the first `k` epochs and the last `k` epochs (smaller is better).
    baseline_epoch_window_k: int = 10
    baseline_epoch_window_violation_weight: float = 1.0


def _mean(xs: Sequence[float]) -> float:
    values = [float(v) for v in xs]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _epoch_window_means(
    objectives: Sequence[float],
    *,
    k: int,
) -> Tuple[float | None, float | None]:
    """Return (early_mean, late_mean) for the first/last k epochs.

    If objectives is empty, both means are None. When k exceeds available epochs,
    it is clamped to the sequence length.
    """

    values = [float(v) for v in objectives]
    if not values:
        return None, None
    kk = max(int(k), 0)
    if kk <= 0:
        return None, None
    early_k = min(kk, len(values))
    late_k = min(kk, len(values))
    early_mean = _mean(values[:early_k])
    late_mean = _mean(values[-late_k:])
    return float(early_mean), float(late_mean)


def _infer_epoch_from_checkpoint_path(path: str) -> int | None:
    name = os.path.basename(str(path))
    m = re.search(r"(?:^|[._-])epoch_(\\d+)(?:\\D|$)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_epoch_values_from_metrics_csv(
    metrics_csv_path: str,
    *,
    epoch_col: str = "epoch",
    value_col: str = "val/reward",
) -> Dict[int, float]:
    """Parse RL4CO CSV logger output into {epoch: value}.

    RL4CO's `metrics.csv` may contain rows with blank `epoch` (step-level logs).
    This loader keeps only rows with a valid `epoch` and a numeric `value_col`,
    and uses the last observed value for each epoch.
    """

    out: Dict[int, float] = {}
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return out
        if epoch_col not in reader.fieldnames:
            raise KeyError(f"metrics.csv missing epoch column '{epoch_col}': {metrics_csv_path}")
        if value_col not in reader.fieldnames:
            raise KeyError(f"metrics.csv missing value column '{value_col}': {metrics_csv_path}")
        for row in reader:
            epoch_raw = row.get(epoch_col, "")
            if epoch_raw is None or str(epoch_raw).strip() == "":
                continue
            try:
                epoch = int(float(epoch_raw))
            except (TypeError, ValueError):
                continue
            val_raw = row.get(value_col, "")
            if val_raw is None or str(val_raw).strip() == "":
                continue
            try:
                val = float(val_raw)
            except (TypeError, ValueError):
                continue
            out[int(epoch)] = float(val)
    return out


def baseline_epoch_objectives_from_metrics_csv(
    metrics_csv_path: str,
    *,
    value_col: str,
    start_epoch: int,
    num_epochs: int,
    objective_sign: str = "neg_reward",
) -> List[float]:
    """Return a length-`num_epochs` list of baseline objectives from RL4CO `metrics.csv`.

    The metrics file typically logs `val/reward` (higher is better, often negative tour length).
    This function converts to the objective used by this repo:
      - objective_sign == "reward": objective = reward
      - objective_sign == "neg_reward": objective = -reward
    """

    epoch_to_val = load_epoch_values_from_metrics_csv(metrics_csv_path, value_col=value_col)
    objectives: List[float] = []
    sign = str(objective_sign or "neg_reward").strip().lower()
    for epoch in range(int(start_epoch), int(start_epoch) + int(num_epochs)):
        if epoch not in epoch_to_val:
            raise KeyError(
                f"metrics.csv missing epoch={epoch} for baseline slice "
                f"(start_epoch={start_epoch}, num_epochs={num_epochs}) at {metrics_csv_path}"
            )
        reward = float(epoch_to_val[int(epoch)])
        obj = reward if sign == "reward" else -reward
        objectives.append(float(obj))
    return objectives


def _extract_state_dict_from_checkpoint(payload: object) -> Mapping[str, torch.Tensor] | None:
    if isinstance(payload, dict):
        sd = payload.get("state_dict")
        if isinstance(sd, dict):
            return sd  # Lightning-style
        sd = payload.get("model_state_dict")
        if isinstance(sd, dict):
            return sd
    return None


def _load_policy_weights_from_checkpoint(policy, ckpt_path: str) -> None:
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"init_checkpoint_path does not exist: {ckpt_path}")

    # Prefer safe tensor-only loading where available. RL4CO Lightning checkpoints
    # can include pickled non-tensor objects (e.g., env instances), which makes
    # `weights_only=True` fail. In that case, fall back to a full load for a
    # user-provided (trusted) checkpoint path.
    ckpt: object
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "weights_only checkpoint load failed (%s); retrying weights_only=False for %s",
            type(exc).__name__,
            os.path.abspath(ckpt_path),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = _extract_state_dict_from_checkpoint(ckpt)
    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint format (missing state_dict): {ckpt_path}")

    target_sd = policy.state_dict()

    # Try common Lightning key prefixes to recover the underlying policy weights.
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    prefixes = [
        "policy.",
        "model.policy.",
        "model.",
        "net.",
        "module.",
        "",
    ]
    for prefix in prefixes:
        if prefix:
            sliced = {k[len(prefix) :]: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith(prefix)}
        else:
            sliced = {k: v for k, v in state_dict.items() if isinstance(k, str)}
        if not sliced:
            continue
        candidates.append((prefix, sliced))

    def _score(sd: Mapping[str, torch.Tensor]) -> int:
        score = 0
        for k, v in sd.items():
            if k not in target_sd:
                continue
            tv = target_sd[k]
            if isinstance(v, torch.Tensor) and isinstance(tv, torch.Tensor) and tuple(v.shape) == tuple(tv.shape):
                score += 1
        return score

    best_prefix = None
    best_sd: Dict[str, torch.Tensor] | None = None
    best_score = -1
    for prefix, cand_sd in candidates:
        s = _score(cand_sd)
        if s > best_score:
            best_score = s
            best_prefix = prefix
            best_sd = cand_sd

    if best_sd is None or best_score <= 0:
        hint = ""
        try:
            # Common mismatch: RL4CO POMO checkpoints trained with the PO4COPs-compatible policy
            # (keys like `policy.encoder.layers.0.Wq.weight`) loaded into an AttentionModelPolicy.
            if any(
                isinstance(k, str) and k.startswith("policy.encoder.layers.0.Wq")
                for k in state_dict.keys()
            ):
                hint = " Hint: set policy_kwargs.po4cops_compat=true to build a checkpoint-compatible POMO policy."
        except Exception:  # noqa: BLE001
            hint = ""
        raise ValueError(
            f"Could not match checkpoint weights to policy state_dict (ckpt={ckpt_path}). "
            f"state_dict_keys={len(state_dict)} policy_keys={len(target_sd)}.{hint}"
        )

    missing, unexpected = policy.load_state_dict(best_sd, strict=False)
    logger.info(
        "Loaded init checkpoint into policy: path=%s prefix=%s matched=%d missing=%d unexpected=%d",
        os.path.abspath(ckpt_path),
        str(best_prefix),
        int(best_score),
        int(len(missing)),
        int(len(unexpected)),
    )


def _build_preference_pairs(
    objective: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Vectorized construction of winner/loser indices.

    For each instance in the batch, we consider all pairs (i, j) such that
    objective[i] < objective[j] (i is better than j). This yields three
    index tensors (batch_idx, winner_idx, loser_idx) plus the total pair
    count. We intentionally do not compute structural features here to keep
    the free-loss evaluation lightweight.
    """

    # objective: (batch, pomo)
    # mask[b, i, j] = True if i is better (lower cost) than j for instance b.
    mask = objective[:, :, None] < objective[:, None, :]
    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)
    pair_count = int(b_idx.numel())
    return (b_idx, winner_idx, loser_idx), pair_count


def _rl4co_env_name(cfg: HighFidelityConfig) -> str:
    env_name = getattr(cfg, "env_name", "") or getattr(cfg, "problem", "tsp")
    return str(env_name).strip().lower()


def _rl4co_size_key(env_name: str) -> str | None:
    return {
        "tsp": "num_loc",
        "cvrp": "num_loc",
        "jssp": "num_jobs",
        "fjsp": "num_jobs",
        "ffsp": "num_job",
    }.get(env_name)


def _rl4co_policy_name(cfg: HighFidelityConfig, env_name: str) -> str:
    policy_name = str(getattr(cfg, "policy_name", "") or "").strip().lower()
    if policy_name:
        return policy_name
    return {
        "tsp": "pomo",
        "cvrp": "pomo",
        "jssp": "l2d",
        "fjsp": "l2d",
        "ffsp": "matnet",
    }.get(env_name, "pomo")


def _rl4co_rollout_strategy(cfg: HighFidelityConfig, policy_name: str) -> str:
    strategy = str(getattr(cfg, "rollout_strategy", "auto") or "auto").strip().lower()
    if strategy and strategy != "auto":
        return strategy
    if policy_name in {"pomo", "matnet"}:
        return "policy_multistart"
    return "batchify_sampling"


def _rl4co_set_multistart_decode(policy) -> None:
    for phase in ("train", "val", "test"):
        attr = f"{phase}_decode_type"
        val = getattr(policy, attr, None)
        if val is None:
            continue
        if "multistart" in str(val):
            continue
        setattr(policy, attr, f"multistart_{val}")


def _rl4co_build_env(
    cfg: HighFidelityConfig,
    problem_size: int,
):
    from rl4co.envs import CVRPEnv, FJSPEnv, JSSPEnv, TSPEnv
    from rl4co.envs.scheduling.ffsp.env import FFSPEnv

    env_name = _rl4co_env_name(cfg)
    env_kwargs = dict(getattr(cfg, "env_kwargs", {}) or {})
    generator_params = dict(getattr(cfg, "generator_params", {}) or {})

    offline_train_path = generator_params.pop("offline_train_path", None)
    offline_val_path = generator_params.pop("offline_val_path", None)
    offline_val_paths = generator_params.pop("offline_val_paths", None)
    if offline_val_paths and isinstance(offline_val_paths, Mapping):
        try:
            offline_val_path = offline_val_paths.get(str(problem_size), offline_val_path)
        except Exception:  # noqa: BLE001
            pass

    size_key = _rl4co_size_key(env_name)
    if size_key is not None:
        generator_params[size_key] = int(problem_size)

    env_map = {
        "tsp": TSPEnv,
        "cvrp": CVRPEnv,
        "jssp": JSSPEnv,
        "fjsp": FJSPEnv,
        "ffsp": FFSPEnv,
    }
    if env_name not in env_map:
        raise ValueError(f"Unsupported env_name for RL4CO backend: {env_name}")

    env = env_map[env_name](generator_params=generator_params, **env_kwargs)

    if offline_train_path or offline_val_path:
        if not offline_train_path or not offline_val_path:
            raise ValueError(
                "Offline generator requires both offline_train_path and offline_val_path "
                f"(got train={offline_train_path!r} val={offline_val_path!r})"
            )
        original_generator = getattr(env, "generator", None)
        preserved_generator_attrs: Dict[str, Any] = {}
        if original_generator is not None:
            for attr_name in ("vehicle_capacity",):
                if hasattr(original_generator, attr_name):
                    preserved_generator_attrs[str(attr_name)] = getattr(original_generator, attr_name)
        env.generator = OfflineSplitGenerator(
            train_path=str(offline_train_path),
            val_path=str(offline_val_path),
            device="cpu",
            extra_attrs=preserved_generator_attrs,
        )

    return env


def _rl4co_build_policy(cfg: HighFidelityConfig, env):
    from rl4co.models.zoo.am import AttentionModelPolicy
    from rl4co.models.zoo.l2d.policy import L2DPolicy
    from rl4co.models.zoo.matnet.model import select_matnet_policy
    from rl4co.models.zoo.pomo.po4cops_cvrp_policy import PO4COPsCVRPPolicy
    from rl4co.models.zoo.pomo.po4cops_tsp_policy import PO4COPsTSPPolicy

    env_name = _rl4co_env_name(cfg)
    policy_name = _rl4co_policy_name(cfg, env_name)
    policy_kwargs = dict(getattr(cfg, "policy_kwargs", {}) or {})

    if policy_name == "pomo":
        use_po4cops_compat = bool(policy_kwargs.pop("po4cops_compat", False))
        if use_po4cops_compat:
            # Mirror `rl4co.models.zoo.pomo.model.POMO` policy construction.
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
            policy_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_defaults.update(policy_kwargs)
            policy = AttentionModelPolicy(env_name=env.name, **policy_defaults)
    elif policy_name == "am":
        policy_defaults = {
            "num_encoder_layers": 6,
            "normalization": "instance",
            "use_graph_context": False,
        }
        policy_defaults.update(policy_kwargs)
        policy = AttentionModelPolicy(env_name=env.name, **policy_defaults)
    elif policy_name == "l2d":
        policy_kwargs.setdefault("test_decode_type", "greedy")
        policy = L2DPolicy(env_name=env.name, **policy_kwargs)
    elif policy_name == "matnet":
        policy = select_matnet_policy(env=env, **policy_kwargs)
    else:
        raise ValueError(f"Unsupported policy_name for RL4CO backend: {policy_name}")

    rollout_strategy = _rl4co_rollout_strategy(cfg, policy_name)
    if rollout_strategy == "policy_multistart":
        _rl4co_set_multistart_decode(policy)
    return policy, rollout_strategy


def _rl4co_objective_from_reward(reward: torch.Tensor, cfg: HighFidelityConfig) -> torch.Tensor:
    sign = str(getattr(cfg, "objective_sign", "neg_reward") or "neg_reward").strip().lower()
    if sign == "reward":
        return reward
    return -reward


def _rl4co_rollout_full(
    env,
    policy,
    batch_size: int,
    num_rollouts: int,
    *,
    phase: str,
    rollout_strategy: str,
    device: torch.device,
    precision: str = "32-true",
    return_actions: bool = False,
    return_entropy: bool = False,
    return_step_logp: bool = False,
) -> Dict[str, torch.Tensor | None]:
    from rl4co.utils.ops import batchify, unbatchify

    gen = getattr(env, "generator", None)
    if gen is None:
        raise RuntimeError("RL4CO env has no generator")
    if hasattr(gen, "set_split") and callable(getattr(gen, "set_split")):
        try:
            gen.set_split(str(phase))
        except Exception:  # noqa: BLE001
            pass

    batch = gen(batch_size)
    batch = batch.to(device)
    td = env.reset(batch)

    with _autocast_context(device, precision):
        if rollout_strategy == "policy_multistart":
            out = policy(
                td,
                env,
                phase=phase,
                num_starts=num_rollouts,
                return_actions=return_actions,
                return_entropy=return_entropy,
                return_sum_log_likelihood=not return_step_logp,
            )
            reward = unbatchify(out["reward"], num_rollouts)
        else:
            td_rep = batchify(td, num_rollouts) if num_rollouts > 1 else td
            out = policy(
                td_rep,
                env,
                phase=phase,
                return_actions=return_actions,
                return_entropy=return_entropy,
                return_sum_log_likelihood=not return_step_logp,
            )
            reward = unbatchify(out["reward"], num_rollouts)

    raw_log_likelihood = out["log_likelihood"]
    log_likelihood_step = None
    if return_step_logp:
        log_likelihood_step = unbatchify(raw_log_likelihood, num_rollouts)
        log_likelihood = log_likelihood_step.sum(dim=-1)
    else:
        log_likelihood = unbatchify(raw_log_likelihood, num_rollouts)

    actions = None
    if return_actions and isinstance(out.get("actions"), torch.Tensor):
        actions = unbatchify(out["actions"], num_rollouts)

    entropy = None
    if return_entropy and isinstance(out.get("entropy"), torch.Tensor):
        entropy = unbatchify(out["entropy"], num_rollouts)

    seq_len = None
    if isinstance(actions, torch.Tensor):
        seq_len = torch.full_like(log_likelihood, float(actions.shape[-1]))
    elif isinstance(log_likelihood_step, torch.Tensor):
        seq_len = torch.full_like(log_likelihood, float(log_likelihood_step.shape[-1]))

    return {
        "reward": reward,
        "log_likelihood": log_likelihood,
        "log_likelihood_step": log_likelihood_step,
        "entropy": entropy,
        "actions": actions,
        "seq_len": seq_len,
    }


def _rl4co_rollout(
    env,
    policy,
    batch_size: int,
    num_rollouts: int,
    *,
    phase: str,
    rollout_strategy: str,
    device: torch.device,
    precision: str = "32-true",
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = _rl4co_rollout_full(
        env,
        policy,
        batch_size,
        num_rollouts,
        phase=phase,
        rollout_strategy=rollout_strategy,
        device=device,
        precision=precision,
        return_actions=False,
        return_entropy=False,
        return_step_logp=False,
    )
    return out["reward"], out["log_likelihood"]


def _train_one_batch_with_free_loss_rl4co(
    env,
    policy,
    optimizer: Adam,
    compiled_loss: CompiledFreeLoss,
    hf_cfg: HighFidelityConfig,
    rollout_strategy: str,
    device: torch.device,
    scaler=None,
    *,
    pref_builder: PrefBuilder | None = None,
) -> Tuple[float, float, int]:
    batch_size = hf_cfg.train_batch_size
    num_rollouts = resolve_pomo_size(hf_cfg.pomo_size, hf_cfg.train_problem_size)
    observables = set(normalize_loss_observables(getattr(hf_cfg, "loss_observables", None)))
    want_seq_len = bool(observables & {"seq_len", "log_prob_mean", "entropy_mean"})
    want_entropy = bool(observables & {"entropy", "entropy_mean"})
    want_step_logp = "log_prob_step" in observables
    # If per-step log-probs are already returned, seq_len can be inferred from
    # that tensor. Avoid also materializing actions for long FFSP rollouts.
    want_actions = bool(want_seq_len and not want_step_logp)

    policy.train()
    rollout = _rl4co_rollout_full(
        env,
        policy,
        batch_size,
        num_rollouts,
        phase="train",
        rollout_strategy=rollout_strategy,
        device=device,
        precision=str(getattr(hf_cfg, "precision", "32-true") or "32-true"),
        return_actions=want_actions,
        return_entropy=want_entropy,
        return_step_logp=want_step_logp,
    )
    reward = rollout["reward"].float()
    log_likelihood = rollout["log_likelihood"].float()

    objective = _rl4co_objective_from_reward(reward, hf_cfg)
    log_prob = log_likelihood

    extra = build_runtime_observables(
        reward,
        log_prob,
        observables=tuple(observables),
        seq_len=rollout["seq_len"],
        log_prob_step=rollout["log_likelihood_step"],
        entropy=rollout["entropy"],
        seq_len_fallback=hf_cfg.train_problem_size,
    )
    feature_cache = extract_feature_cache(objective, log_prob, extra=extra)
    pair_count = 0

    mode = getattr(compiled_loss.ir.implementation_hint, "mode", "pairwise")
    mode = str(mode or "pairwise").strip().lower()

    if mode == "setwise":
        loss = compiled_loss.loss_fn(
            batch={},
            model_output=feature_cache,
            extra={"alpha": hf_cfg.alpha},
        )
    else:
        builder = pref_builder or _AllPairsPrefBuilder()
        pref = builder.build(
            feature_cache,
            meta={
                "stage": "train",
                "rollout_strategy": str(rollout_strategy),
                "problem": str(getattr(hf_cfg, "problem", "")),
                "problem_size": int(hf_cfg.train_problem_size),
            },
        )
        pair_count = pref.num_examples()
        if pair_count == 0:
            advantage = reward - reward.mean(dim=1, keepdim=True)
            loss = -(advantage * log_prob).mean()
        else:
            batch = prepare_pairwise_loss_batch(
                pref.to_pairwise_loss_batch(feature_cache),
                compiled_loss.ir.implementation_hint.expects or [],
            )
            loss = compiled_loss.loss_fn(
                batch=batch,
                model_output=feature_cache,
                extra={"alpha": hf_cfg.alpha},
            )

    max_reward, _ = reward.max(dim=1)
    score_mean = _rl4co_objective_from_reward(max_reward, hf_cfg).float().mean()

    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss).all():
        raise RuntimeError("Non-finite loss encountered during mini-train")
    if scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return score_mean.item(), float(loss.item()), pair_count


@torch.no_grad()
def _evaluate_rl4co_model(
    *,
    policy,
    cfg: HighFidelityConfig,
    problem_size: int,
    device: torch.device,
    num_episodes: int,
    batch_size: int,
    rollout_strategy: str,
) -> float:
    env = _rl4co_build_env(cfg, problem_size)
    env = env.to(device)
    policy.eval()

    num_rollouts = resolve_pomo_size(cfg.pomo_size, problem_size)
    score_meter = AverageMeter()
    episodes_done = 0

    while episodes_done < num_episodes:
        remaining = num_episodes - episodes_done
        current_batch = min(batch_size, remaining)

        reward, _ = _rl4co_rollout(
            env,
            policy,
            current_batch,
            num_rollouts,
            phase="test",
            rollout_strategy=rollout_strategy,
            device=device,
            precision=str(getattr(cfg, "precision", "32-true") or "32-true"),
        )
        max_reward, _ = reward.max(dim=1)
        score = _rl4co_objective_from_reward(max_reward, cfg).float().mean().item()
        score_meter.update(score, n=current_batch)
        episodes_done += current_batch

    return float(score_meter.avg)


def _evaluate_free_loss_candidate_rl4co(
    compiled_loss: CompiledFreeLoss,
    cfg: FreeLossFidelityConfig,
    *,
    baseline_early_valid: float | None = None,
    early_eval_steps: int = 0,
    baseline_epoch_objectives: Sequence[float] | None = None,
    pref_builder: PrefBuilder | None = None,
) -> Dict[str, Any]:
    _set_seed(cfg.hf.seed)

    device_str = cfg.hf.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    steps_per_epoch, epochs_total_cfg = get_hf_epoch_plan(cfg.hf)
    scratch_epochs = max(int(getattr(cfg, "scratch_hf_epochs", 0) or 0), 0)
    warm_epochs = max(int(getattr(cfg, "warmstart_hf_epochs", 0) or 0), 0)
    split_enabled = bool(scratch_epochs or warm_epochs)
    if steps_per_epoch <= 0 or epochs_total_cfg <= 0:
        if split_enabled:
            logger.warning(
                "Ignoring scratch_hf_epochs/warmstart_hf_epochs because hf_epochs and hf_instances_per_epoch are not set (>0)."
            )
        split_enabled = False
    if split_enabled and cfg.hf.hf_epochs > 0 and (scratch_epochs + warm_epochs) != int(cfg.hf.hf_epochs):
        logger.warning(
            "scratch_hf_epochs + warmstart_hf_epochs != hf_epochs (%d + %d != %d); continuing anyway.",
            scratch_epochs,
            warm_epochs,
            int(cfg.hf.hf_epochs),
        )

    steps_f2_cfg = max(int(cfg.f2_steps), 0)

    if steps_per_epoch <= 0 or epochs_total_cfg <= 0:
        # Legacy fallback: step-based HF evaluation (hf_steps).
        env = _rl4co_build_env(cfg.hf, cfg.hf.train_problem_size)
        env = env.to(device)
        policy, rollout_strategy = _rl4co_build_policy(cfg.hf, env)
        if cfg.init_checkpoint_path:
            _load_policy_weights_from_checkpoint(policy, str(cfg.init_checkpoint_path))
        policy = policy.to(device)
        scaler = _make_grad_scaler(device, str(getattr(cfg.hf, "precision", "32-true") or "32-true"))
        optimizer = Adam(
            policy.parameters(),
            lr=float(cfg.hf.learning_rate),
            weight_decay=float(cfg.hf.weight_decay),
        )

        score_meter = AverageMeter()
        loss_meter = AverageMeter()
        total_pairs = 0

        steps_f1 = get_total_hf_train_steps(cfg.hf)
        steps_f2 = steps_f2_cfg
        steps = steps_f1 + steps_f2

        score_meter_f1 = AverageMeter()
        loss_meter_f1 = AverageMeter()
        total_pairs_f1 = 0
        score_meter_f2 = AverageMeter()
        loss_meter_f2 = AverageMeter()
        total_pairs_f2 = 0

        logger.info(
            "RL4CO free-loss (step-mode): f1_steps=%d, f2_steps=%d, total_steps=%d, train_problem_size=%d, "
            "rollouts=%d, batch_size=%d, device=%s, env=%s",
            steps_f1,
            steps_f2,
            steps,
            cfg.hf.train_problem_size,
            resolve_pomo_size(cfg.hf.pomo_size, cfg.hf.train_problem_size),
            cfg.hf.train_batch_size,
            str(device),
            _rl4co_env_name(cfg.hf),
        )

        log_interval = max(steps // 10, 1)
        early_eval_steps = max(int(early_eval_steps or 0), 0)
        early_eval_effective = min(early_eval_steps, steps) if early_eval_steps > 0 else 0
        early_validation_objective: float | None = None
        early_stopped = False

        for step in range(steps):
            score, loss, pair_count = _train_one_batch_with_free_loss_rl4co(
                env=env,
                policy=policy,
                optimizer=optimizer,
                compiled_loss=compiled_loss,
                hf_cfg=cfg.hf,
                rollout_strategy=rollout_strategy,
                device=device,
                scaler=scaler,
                pref_builder=pref_builder,
            )
            score_meter.update(score)
            loss_meter.update(loss)
            total_pairs += int(pair_count)
            if step < steps_f1:
                score_meter_f1.update(score)
                loss_meter_f1.update(loss)
                total_pairs_f1 += int(pair_count)
            else:
                score_meter_f2.update(score)
                loss_meter_f2.update(loss)
                total_pairs_f2 += int(pair_count)

            if (step + 1) % log_interval == 0 or step == 0:
                logger.info(
                    "RL4CO free-loss step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f), pairs_step=%d, pairs_total=%d",
                    step + 1,
                    steps,
                    score,
                    float(score_meter.avg),
                    loss,
                    float(loss_meter.avg),
                    int(pair_count),
                    total_pairs,
                )

            if early_eval_effective > 0 and (step + 1) == early_eval_effective:
                early_validation_objective = _evaluate_rl4co_model(
                    policy=policy,
                    cfg=cfg.hf,
                    problem_size=cfg.hf.train_problem_size,
                    device=device,
                    num_episodes=cfg.hf.num_validation_episodes,
                    batch_size=cfg.hf.validation_batch_size,
                    rollout_strategy=rollout_strategy,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if baseline_early_valid is not None and early_validation_objective > baseline_early_valid:
                    early_stopped = True
                    logger.info(
                        "RL4CO early stop at step %d: candidate early_valid=%.6f baseline_early=%.6f",
                        step + 1,
                        early_validation_objective,
                        baseline_early_valid,
                    )
                    break

        if early_stopped and early_validation_objective is not None:
            main_valid_obj = float(early_validation_objective)
        else:
            main_valid_obj = float(
                _evaluate_rl4co_model(
                    policy=policy,
                    cfg=cfg.hf,
                    problem_size=cfg.hf.train_problem_size,
                    device=device,
                    num_episodes=cfg.hf.num_validation_episodes,
                    batch_size=cfg.hf.validation_batch_size,
                    rollout_strategy=rollout_strategy,
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        primary_phase = "single_steps"
        primary_epochs_total = 0
        epoch_validation_objectives = []
        early_eval = {
            "enabled": bool(early_eval_effective),
            "steps": int(early_eval_effective),
            "baseline_validation_objective": baseline_early_valid,
            "candidate_validation_objective": early_validation_objective,
            "early_stopped": early_stopped,
        }
        scratch_epoch_eval = None
        warmstart_epoch_eval = None
        train_score_mean = float(score_meter.avg)
        train_loss_mean = float(loss_meter.avg)
        pair_count = int(total_pairs)
        f1_train_score_mean = float(score_meter_f1.avg) if steps_f1 > 0 else None
        f1_train_loss_mean = float(loss_meter_f1.avg) if steps_f1 > 0 else None
        f2_train_score_mean = float(score_meter_f2.avg) if steps_f2 > 0 else None
        f2_train_loss_mean = float(loss_meter_f2.avg) if steps_f2 > 0 else None
        f1_pair_count = int(total_pairs_f1)
        f2_pair_count = int(total_pairs_f2)
    def _run_phase(
        *,
        phase: str,
        phase_epochs: int,
        init_ckpt: str | None,
        use_early_stop: bool,
        extra_steps_f2: int,
        early_eval_steps_phase: int,
        baseline_early_valid_phase: float | None,
    ) -> Dict[str, Any]:
        _set_seed(cfg.hf.seed)

        env = _rl4co_build_env(cfg.hf, cfg.hf.train_problem_size)
        env = env.to(device)
        policy, rollout_strategy = _rl4co_build_policy(cfg.hf, env)
        if init_ckpt:
            _load_policy_weights_from_checkpoint(policy, str(init_ckpt))
        policy = policy.to(device)
        scaler = _make_grad_scaler(device, str(getattr(cfg.hf, "precision", "32-true") or "32-true"))
        optimizer = Adam(
            policy.parameters(),
            lr=float(cfg.hf.learning_rate),
            weight_decay=float(cfg.hf.weight_decay),
        )

        phase_epochs = max(int(phase_epochs), 0)
        steps_f1_phase = int(phase_epochs) * int(steps_per_epoch)
        steps_f2_phase = max(int(extra_steps_f2), 0)
        steps_phase = steps_f1_phase + steps_f2_phase

        score_meter = AverageMeter()
        loss_meter = AverageMeter()
        total_pairs = 0
        score_meter_f1 = AverageMeter()
        loss_meter_f1 = AverageMeter()
        total_pairs_f1 = 0
        score_meter_f2 = AverageMeter()
        loss_meter_f2 = AverageMeter()
        total_pairs_f2 = 0
        epoch_objectives: List[float] = []

        logger.info(
            "RL4CO free-loss phase=%s: epochs=%d steps_f1=%d steps_f2=%d total_steps=%d "
            "(ckpt=%s, device=%s, env=%s)",
            phase,
            phase_epochs,
            steps_f1_phase,
            steps_f2_phase,
            steps_phase,
            str(init_ckpt) if init_ckpt else "none",
            str(device),
            _rl4co_env_name(cfg.hf),
        )

        if steps_phase <= 0:
            return {
                "phase": phase,
                "policy": policy,
                "rollout_strategy": rollout_strategy,
                "steps_f1": steps_f1_phase,
                "steps_f2": steps_f2_phase,
                "steps": steps_phase,
                "epochs_total": phase_epochs,
                "epoch_objectives": [],
                "final_validation_objective": None,
                "early_eval": {"enabled": False, "steps": 0, "early_stopped": False},
                "train_score_mean": None,
                "train_loss_mean": None,
                "pair_count": 0,
                "f1": {"train_score_mean": None, "train_loss_mean": None, "pair_count": 0},
                "f2": {"train_score_mean": None, "train_loss_mean": None, "pair_count": 0},
            }

        log_interval = max(steps_phase // 10, 1)
        early_eval_steps_phase = max(int(early_eval_steps_phase or 0), 0)
        early_eval_effective = (
            min(early_eval_steps_phase, steps_f1_phase) if early_eval_steps_phase > 0 else 0
        )
        early_validation_objective: float | None = None
        early_stopped = False

        for step in range(steps_phase):
            score, loss, pair_count = _train_one_batch_with_free_loss_rl4co(
                env=env,
                policy=policy,
                optimizer=optimizer,
                compiled_loss=compiled_loss,
                hf_cfg=cfg.hf,
                rollout_strategy=rollout_strategy,
                device=device,
                scaler=scaler,
                pref_builder=pref_builder,
            )
            score_meter.update(score)
            loss_meter.update(loss)
            total_pairs += int(pair_count)

            if step < steps_f1_phase:
                score_meter_f1.update(score)
                loss_meter_f1.update(loss)
                total_pairs_f1 += int(pair_count)
            else:
                score_meter_f2.update(score)
                loss_meter_f2.update(loss)
                total_pairs_f2 += int(pair_count)

            if (step + 1) % log_interval == 0 or step == 0:
                logger.info(
                    "RL4CO free-loss[%s] step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f), pairs_step=%d, pairs_total=%d",
                    phase,
                    step + 1,
                    steps_phase,
                    score,
                    float(score_meter.avg),
                    loss,
                    float(loss_meter.avg),
                    int(pair_count),
                    total_pairs,
                )

            if steps_f1_phase > 0 and (step + 1) % steps_per_epoch == 0:
                epoch_idx = (step + 1) // steps_per_epoch
                if epoch_idx <= phase_epochs:
                    epoch_valid_obj = _evaluate_rl4co_model(
                        policy=policy,
                        cfg=cfg.hf,
                        problem_size=cfg.hf.train_problem_size,
                        device=device,
                        num_episodes=cfg.hf.num_validation_episodes,
                        batch_size=cfg.hf.validation_batch_size,
                        rollout_strategy=rollout_strategy,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    epoch_objectives.append(epoch_valid_obj)
                    logger.info(
                        "RL4CO free-loss[%s] epoch %d/%d: validation_objective=%.6f",
                        phase,
                        epoch_idx,
                        phase_epochs,
                        epoch_valid_obj,
                    )

            if use_early_stop and early_eval_effective > 0 and (step + 1) == early_eval_effective:
                early_validation_objective = _evaluate_rl4co_model(
                    policy=policy,
                    cfg=cfg.hf,
                    problem_size=cfg.hf.train_problem_size,
                    device=device,
                    num_episodes=cfg.hf.num_validation_episodes,
                    batch_size=cfg.hf.validation_batch_size,
                    rollout_strategy=rollout_strategy,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (
                    baseline_early_valid_phase is not None
                    and early_validation_objective > baseline_early_valid_phase
                ):
                    early_stopped = True
                    logger.info(
                        "RL4CO early stop[%s] at step %d: candidate early_valid=%.6f baseline_early=%.6f",
                        phase,
                        step + 1,
                        early_validation_objective,
                        baseline_early_valid_phase,
                    )
                    break

        if early_stopped and early_validation_objective is not None:
            final_valid_obj = float(early_validation_objective)
        else:
            final_valid_obj = float(
                _evaluate_rl4co_model(
                    policy=policy,
                    cfg=cfg.hf,
                    problem_size=cfg.hf.train_problem_size,
                    device=device,
                    num_episodes=cfg.hf.num_validation_episodes,
                    batch_size=cfg.hf.validation_batch_size,
                    rollout_strategy=rollout_strategy,
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            env = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        return {
            "phase": phase,
            "policy": policy,
            "rollout_strategy": rollout_strategy,
            "steps_f1": int(steps_f1_phase),
            "steps_f2": int(steps_f2_phase),
            "steps": int(steps_phase),
            "epochs_total": int(phase_epochs),
            "epoch_objectives": list(epoch_objectives),
            "final_validation_objective": float(final_valid_obj),
            "early_eval": {
                "enabled": bool(early_eval_effective),
                "steps": int(early_eval_effective),
                "baseline_validation_objective": baseline_early_valid_phase,
                "candidate_validation_objective": early_validation_objective,
                "early_stopped": early_stopped,
            },
            "train_score_mean": float(score_meter.avg),
            "train_loss_mean": float(loss_meter.avg),
            "pair_count": int(total_pairs),
            "f1": {
                "train_score_mean": float(score_meter_f1.avg) if steps_f1_phase > 0 else None,
                "train_loss_mean": float(loss_meter_f1.avg) if steps_f1_phase > 0 else None,
                "pair_count": int(total_pairs_f1),
            },
            "f2": {
                "train_score_mean": float(score_meter_f2.avg) if steps_f2_phase > 0 else None,
                "train_loss_mean": float(loss_meter_f2.avg) if steps_f2_phase > 0 else None,
                "pair_count": int(total_pairs_f2),
            },
        }

    if steps_per_epoch > 0 and epochs_total_cfg > 0:
        if split_enabled:
            scratch_res: Dict[str, Any] | None = None
            warm_res: Dict[str, Any] | None = None
            has_init_ckpt = bool(cfg.init_checkpoint_path)

            if has_init_ckpt:
                if warm_epochs > 0:
                    warm_res = _run_phase(
                        phase="warmstart",
                        phase_epochs=warm_epochs,
                        init_ckpt=cfg.init_checkpoint_path,
                        use_early_stop=True,
                        extra_steps_f2=steps_f2_cfg,
                        early_eval_steps_phase=int(early_eval_steps or 0),
                        baseline_early_valid_phase=baseline_early_valid,
                    )
                elif scratch_epochs > 0:
                    logger.info(
                        "warmstart_hf_epochs=0 for checkpoint-backed init; falling back to scratch phase."
                    )
                    scratch_res = _run_phase(
                        phase="scratch",
                        phase_epochs=scratch_epochs,
                        init_ckpt=None,
                        use_early_stop=False,
                        extra_steps_f2=0,
                        early_eval_steps_phase=0,
                        baseline_early_valid_phase=None,
                    )
            else:
                if scratch_epochs > 0:
                    scratch_res = _run_phase(
                        phase="scratch",
                        phase_epochs=scratch_epochs,
                        init_ckpt=None,
                        use_early_stop=False,
                        extra_steps_f2=0,
                        early_eval_steps_phase=0,
                        baseline_early_valid_phase=None,
                    )
                elif warm_epochs > 0:
                    logger.warning(
                        "warmstart_hf_epochs=%d but init_checkpoint_path is not set; skipping warm-start phase.",
                        warm_epochs,
                    )

            primary_res = warm_res if warm_res is not None else scratch_res
            if primary_res is None:
                raise RuntimeError("Split HF evaluation enabled but no phase was executed.")

            if scratch_res is not None and warm_res is not None:
                primary_phase = "scratch+warmstart"
                primary_epochs_total = int(scratch_epochs + warm_epochs)
                epoch_validation_objectives = list(scratch_res["epoch_objectives"]) + list(
                    warm_res["epoch_objectives"]
                )
                # Early-eval is meaningful only for the warm-start phase (used for early stopping).
                early_eval = dict(warm_res.get("early_eval") or {})
            else:
                primary_phase = str(primary_res["phase"])
                primary_epochs_total = int(primary_res["epochs_total"])
                epoch_validation_objectives = list(primary_res["epoch_objectives"])
                early_eval = dict(primary_res.get("early_eval") or {})

            phase_list = [
                r
                for r in (scratch_res, warm_res)
                if r is not None and int(r.get("steps", 0) or 0) > 0
            ]
            steps_f1 = sum(int(r.get("steps_f1", 0) or 0) for r in phase_list)
            steps_f2 = sum(int(r.get("steps_f2", 0) or 0) for r in phase_list)
            steps = steps_f1 + steps_f2
            pair_count = sum(int(r.get("pair_count", 0) or 0) for r in phase_list)

            train_score_mean = None
            train_loss_mean = None
            if steps > 0:
                train_score_mean = sum(float(r["train_score_mean"]) * float(r["steps"]) for r in phase_list) / float(steps)
                train_loss_mean = sum(float(r["train_loss_mean"]) * float(r["steps"]) for r in phase_list) / float(steps)

            f1_steps_total = steps_f1
            f2_steps_total = steps_f2
            f1_pair_count = sum(int((r.get("f1") or {}).get("pair_count", 0) or 0) for r in phase_list)
            f2_pair_count = sum(int((r.get("f2") or {}).get("pair_count", 0) or 0) for r in phase_list)

            f1_train_score_mean = None
            f1_train_loss_mean = None
            if f1_steps_total > 0:
                f1_train_score_mean = sum(
                    float((r.get("f1") or {}).get("train_score_mean") or 0.0) * float(r.get("steps_f1") or 0)
                    for r in phase_list
                ) / float(f1_steps_total)
                f1_train_loss_mean = sum(
                    float((r.get("f1") or {}).get("train_loss_mean") or 0.0) * float(r.get("steps_f1") or 0)
                    for r in phase_list
                ) / float(f1_steps_total)

            f2_train_score_mean = None
            f2_train_loss_mean = None
            if f2_steps_total > 0:
                f2_train_score_mean = sum(
                    float((r.get("f2") or {}).get("train_score_mean") or 0.0) * float(r.get("steps_f2") or 0)
                    for r in phase_list
                ) / float(f2_steps_total)
                f2_train_loss_mean = sum(
                    float((r.get("f2") or {}).get("train_loss_mean") or 0.0) * float(r.get("steps_f2") or 0)
                    for r in phase_list
                ) / float(f2_steps_total)

            policy = primary_res["policy"]
            rollout_strategy = primary_res["rollout_strategy"]
            main_valid_obj = float(primary_res["final_validation_objective"])

            scratch_epoch_eval = (
                {
                    "phase": "scratch",
                    "epochs_total": int(scratch_res["epochs_total"]),
                    "objectives": list(scratch_res["epoch_objectives"]),
                    "final_validation_objective": scratch_res["final_validation_objective"],
                }
                if scratch_res is not None and scratch_epochs > 0
                else None
            )
            warmstart_epoch_eval = (
                {
                    "phase": "warmstart",
                    "epochs_total": int(warm_res["epochs_total"]),
                    "objectives": list(warm_res["epoch_objectives"]),
                    "final_validation_objective": warm_res["final_validation_objective"],
                    "init_checkpoint_path": cfg.init_checkpoint_path,
                    "init_checkpoint_epoch": cfg.init_checkpoint_epoch,
                }
                if warm_res is not None and warm_epochs > 0
                else None
            )
        else:
            primary_phase = "single"
            primary_epochs_total = int(epochs_total_cfg)
            warm_res = _run_phase(
                phase="single",
                phase_epochs=primary_epochs_total,
                init_ckpt=cfg.init_checkpoint_path,
                use_early_stop=True,
                extra_steps_f2=steps_f2_cfg,
                early_eval_steps_phase=int(early_eval_steps or 0),
                baseline_early_valid_phase=baseline_early_valid,
            )
            epoch_validation_objectives = list(warm_res["epoch_objectives"])
            early_eval = dict(warm_res.get("early_eval") or {})
            policy = warm_res["policy"]
            rollout_strategy = warm_res["rollout_strategy"]
            main_valid_obj = float(warm_res["final_validation_objective"])

            steps_f1 = int(warm_res["steps_f1"])
            steps_f2 = int(warm_res["steps_f2"])
            steps = int(warm_res["steps"])
            train_score_mean = float(warm_res["train_score_mean"])
            train_loss_mean = float(warm_res["train_loss_mean"])
            pair_count = int(warm_res["pair_count"])
            f1_train_score_mean = warm_res["f1"]["train_score_mean"]
            f1_train_loss_mean = warm_res["f1"]["train_loss_mean"]
            f2_train_score_mean = warm_res["f2"]["train_score_mean"]
            f2_train_loss_mean = warm_res["f2"]["train_loss_mean"]
            f1_pair_count = int(warm_res["f1"]["pair_count"])
            f2_pair_count = int(warm_res["f2"]["pair_count"])
            scratch_epoch_eval = None
            warmstart_epoch_eval = None

    size_objectives: Dict[int, float] = {int(cfg.hf.train_problem_size): float(main_valid_obj)}
    for size in cfg.hf.valid_problem_sizes:
        size_int = int(size)
        if size_int in size_objectives:
            continue
        size_objectives[size_int] = _evaluate_rl4co_model(
            policy=policy,
            cfg=cfg.hf,
            problem_size=size_int,
            device=device,
            num_episodes=cfg.hf.num_validation_episodes,
            batch_size=cfg.hf.validation_batch_size,
            rollout_strategy=rollout_strategy,
        )
    gen_objectives = {
        k: v for k, v in size_objectives.items() if k != int(cfg.hf.train_problem_size)
    }

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, max_gen_obj - main_valid_obj)

    epoch_objective_mean: float | None = None
    if epoch_validation_objectives:
        epoch_objective_mean = float(
            sum(epoch_validation_objectives) / len(epoch_validation_objectives)
        )

    epoch_baseline_violations: int | None = None
    epoch_better_than_baseline: bool | None = None
    epoch_tail_baseline_violations: int | None = None
    epoch_tail_better_than_baseline: bool | None = None
    epoch_baseline_margins: List[float] | None = None
    if baseline_epoch_objectives:
        offset = max(int(getattr(cfg, "baseline_epoch_compare_offset", 0) or 0), 0)
        baseline_list_full = [float(v) for v in baseline_epoch_objectives]
        baseline_list = baseline_list_full[offset : offset + int(primary_epochs_total)]
        compare_len = min(len(epoch_validation_objectives), len(baseline_list))
        epoch_baseline_margins = []
        for i in range(compare_len):
            margin = float(epoch_validation_objectives[i]) - baseline_list[i]
            epoch_baseline_margins.append(margin)
        violations = sum(1 for m in epoch_baseline_margins if m > 0.0)
        epoch_baseline_violations = int(violations)
        # Only mark "better" when we have a full epoch-by-epoch comparison for the
        # configured evaluation horizon (i.e., no missing epochs).
        epoch_better_than_baseline = (
            compare_len == int(primary_epochs_total)
            and compare_len == len(epoch_validation_objectives)
            and epoch_baseline_violations == 0
        )

        # Tail-only comparison for `better_than_baseline` (default: all epochs).
        try:
            tail_frac = float(getattr(cfg, "baseline_epoch_tail_frac", 1.0) or 1.0)
        except (TypeError, ValueError):
            tail_frac = 1.0
        tail_frac = min(max(tail_frac, 0.0), 1.0)
        if (
            compare_len == int(primary_epochs_total)
            and compare_len == len(epoch_validation_objectives)
            and compare_len > 0
            and tail_frac > 0.0
        ):
            tail_count = int(math.ceil(tail_frac * float(compare_len)))
            tail_count = max(1, min(tail_count, compare_len))
            tail_start = compare_len - tail_count
            tail_margins = epoch_baseline_margins[tail_start:]
            tail_violations = sum(1 for m in tail_margins if m > 0.0)
            epoch_tail_baseline_violations = int(tail_violations)
            epoch_tail_better_than_baseline = epoch_tail_baseline_violations == 0
        else:
            epoch_tail_baseline_violations = None
            epoch_tail_better_than_baseline = None

    # Epoch-window comparison (early k + late k) against baseline epoch objectives.
    window_k = int(getattr(cfg, "baseline_epoch_window_k", 10) or 10)
    cand_early_mean, cand_late_mean = _epoch_window_means(
        epoch_validation_objectives, k=window_k
    )
    epoch_window_eval: Dict[str, Any] = {
        "k": int(window_k),
        "early_mean": cand_early_mean,
        "late_mean": cand_late_mean,
        "objectives": list(epoch_validation_objectives),
    }

    base_early_mean: float | None = None
    base_late_mean: float | None = None
    if baseline_epoch_objectives:
        offset = max(int(getattr(cfg, "baseline_epoch_compare_offset", 0) or 0), 0)
        baseline_slice = list(baseline_epoch_objectives)[
            offset : offset + int(primary_epochs_total)
        ]
        base_early_mean, base_late_mean = _epoch_window_means(
            baseline_slice, k=window_k
        )
    baseline_epoch_window_eval: Dict[str, Any] = {
        "early_mean": base_early_mean,
        "late_mean": base_late_mean,
    }

    epoch_window_margins: Dict[str, float] | None = None
    epoch_window_violations: int | None = None
    epoch_window_better_than_baseline: bool | None = None
    if (
        cand_early_mean is not None
        and cand_late_mean is not None
        and base_early_mean is not None
        and base_late_mean is not None
    ):
        early_margin = float(cand_early_mean) - float(base_early_mean)
        late_margin = float(cand_late_mean) - float(base_late_mean)
        epoch_window_margins = {"early": early_margin, "late": late_margin}
        epoch_window_violations = int(sum(1 for m in (early_margin, late_margin) if m > 0.0))
        epoch_window_better_than_baseline = epoch_window_violations == 0

    agg_method = str(cfg.hf.size_aggregation or "legacy").strip().lower()
    base_objective = (
        epoch_objective_mean if epoch_objective_mean is not None else float(main_valid_obj)
    )
    if agg_method == "legacy":
        hf_like_score = base_objective + cfg.hf.generalization_penalty_weight * generalization_penalty
    else:
        hf_like_score = aggregate_objectives_by_size(
            size_objectives,
            method=agg_method,
            cvar_alpha=float(cfg.hf.size_cvar_alpha),
        )
    if epoch_baseline_violations is not None:
        hf_like_score += cfg.baseline_epoch_violation_weight * float(epoch_baseline_violations)
    if epoch_window_violations is not None:
        hf_like_score += cfg.baseline_epoch_window_violation_weight * float(epoch_window_violations)

    return {
        "hf_like_score": hf_like_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "size_objectives": size_objectives,
        "size_aggregation": agg_method,
        "size_cvar_alpha": float(cfg.hf.size_cvar_alpha),
        "epoch_objective_mean": epoch_objective_mean,
        "epoch_baseline_violations": epoch_baseline_violations,
        "epoch_better_than_baseline": epoch_better_than_baseline,
        "epoch_tail_baseline_violations": epoch_tail_baseline_violations,
        "epoch_tail_better_than_baseline": epoch_tail_better_than_baseline,
        "epoch_window_eval": epoch_window_eval,
        "baseline_epoch_window_eval": baseline_epoch_window_eval,
        "epoch_window_margins": epoch_window_margins,
        "epoch_window_violations": epoch_window_violations,
        "epoch_window_better_than_baseline": epoch_window_better_than_baseline,
        "epoch_eval": {
            "enabled": bool(steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch > 0 else None,
            "epochs_total": int(primary_epochs_total),
            "objectives": epoch_validation_objectives,
            "objective_mean": epoch_objective_mean,
            "baseline_margins": epoch_baseline_margins,
            "baseline_violations": epoch_baseline_violations,
            "better_than_baseline": epoch_better_than_baseline,
            "phase": primary_phase,
            "baseline_compare_offset": int(getattr(cfg, "baseline_epoch_compare_offset", 0) or 0),
            "tail_frac": float(getattr(cfg, "baseline_epoch_tail_frac", 1.0) or 1.0),
            "tail_baseline_violations": epoch_tail_baseline_violations,
            "tail_better_than_baseline": epoch_tail_better_than_baseline,
        },
        "scratch_epoch_eval": scratch_epoch_eval,
        "warmstart_epoch_eval": warmstart_epoch_eval,
        "train_score_mean": float(train_score_mean) if train_score_mean is not None else None,
        "train_loss_mean": float(train_loss_mean) if train_loss_mean is not None else None,
        "pair_count": int(pair_count),
        "early_eval": {
            "enabled": bool(early_eval.get("enabled")),
            "steps": int(early_eval.get("steps") or 0),
            "baseline_validation_objective": early_eval.get("baseline_validation_objective"),
            "candidate_validation_objective": early_eval.get("candidate_validation_objective"),
            "early_stopped": bool(early_eval.get("early_stopped")),
        },
        "phases": {
            "f1": {
                "steps": int(steps_f1),
                "train_score_mean": float(f1_train_score_mean) if steps_f1 > 0 else None,
                "train_loss_mean": float(f1_train_loss_mean) if steps_f1 > 0 else None,
                "pair_count": int(f1_pair_count),
            },
            "f2": {
                "steps": int(steps_f2),
                "train_score_mean": float(f2_train_score_mean) if steps_f2 > 0 else None,
                "train_loss_mean": float(f2_train_loss_mean) if steps_f2 > 0 else None,
                "pair_count": int(f2_pair_count),
            },
        },
        "config": {
            "hf": asdict(cfg.hf),
            "free_loss": {
                "f1_steps": cfg.f1_steps,
                "total_train_steps": steps,
                "init_checkpoint_path": cfg.init_checkpoint_path,
                "init_checkpoint_epoch": (
                    int(cfg.init_checkpoint_epoch)
                    if cfg.init_checkpoint_epoch is not None
                    else _infer_epoch_from_checkpoint_path(str(cfg.init_checkpoint_path))
                    if cfg.init_checkpoint_path
                    else None
                ),
                "scratch_hf_epochs": int(getattr(cfg, "scratch_hf_epochs", 0) or 0),
                "warmstart_hf_epochs": int(getattr(cfg, "warmstart_hf_epochs", 0) or 0),
                "baseline_epoch_compare_offset": int(getattr(cfg, "baseline_epoch_compare_offset", 0) or 0),
                "f2_steps": cfg.f2_steps,
                "f3_enabled": cfg.f3_enabled,
                "baseline_epoch_violation_weight": cfg.baseline_epoch_violation_weight,
                "baseline_epoch_tail_frac": float(getattr(cfg, "baseline_epoch_tail_frac", 1.0) or 1.0),
                "baseline_epoch_window_k": cfg.baseline_epoch_window_k,
                "baseline_epoch_window_violation_weight": cfg.baseline_epoch_window_violation_weight,
            },
        },
        "loss_ir": {
            "name": compiled_loss.ir.name,
            "intuition": compiled_loss.ir.intuition,
            "hyperparams": compiled_loss.ir.hyperparams,
            "operators_used": compiled_loss.ir.operators_used,
        },
    }


def evaluate_po_baseline_rl4co(
    cfg: HighFidelityConfig,
    *,
    early_eval_steps: int | None = None,
    baseline_early_valid: float | None = None,
    baseline_epoch_objectives: Sequence[float] | None = None,
    init_checkpoint_path: str | None = None,
    init_checkpoint_epoch: int | None = None,
    scratch_hf_epochs: int = 0,
    warmstart_hf_epochs: int = 0,
    baseline_epoch_compare_offset: int = 0,
    baseline_epoch_violation_weight: float = 1.0,
    baseline_epoch_tail_frac: float = 1.0,
    baseline_epoch_window_k: int = 10,
    baseline_epoch_window_violation_weight: float = 1.0,
) -> Dict[str, Any]:
    from rl4co.models.rl.reinforce.preference_losses import po_loss

    _set_seed(cfg.seed)

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    po_impl = _normalize_po_impl(getattr(cfg, "po_impl", "bt"))
    steps_per_epoch, epochs_total_cfg = get_hf_epoch_plan(cfg)
    scratch_epochs = max(int(scratch_hf_epochs or 0), 0)
    warm_epochs = max(int(warmstart_hf_epochs or 0), 0)
    split_enabled = bool(scratch_epochs or warm_epochs)
    if steps_per_epoch <= 0 or epochs_total_cfg <= 0:
        split_enabled = False
    elif cfg.hf_epochs > 0 and (scratch_epochs + warm_epochs) != int(cfg.hf_epochs):
        logger.warning(
            "PO baseline split epochs do not sum to hf_epochs (%d + %d != %d); continuing anyway.",
            scratch_epochs,
            warm_epochs,
            int(cfg.hf_epochs),
        )

    def _run_phase(
        *,
        phase: str,
        total_steps: int,
        phase_epochs: int,
        init_ckpt: str | None,
        use_early_stop: bool,
        early_eval_steps_phase: int,
        baseline_early_valid_phase: float | None,
    ) -> Dict[str, Any]:
        _set_seed(cfg.seed)
        env = _rl4co_build_env(cfg, cfg.train_problem_size)
        env = env.to(device)
        policy, rollout_strategy = _rl4co_build_policy(cfg, env)
        if init_ckpt:
            _load_policy_weights_from_checkpoint(policy, str(init_ckpt))
        policy = policy.to(device)
        scaler = _make_grad_scaler(device, str(getattr(cfg, "precision", "32-true") or "32-true"))
        optimizer = Adam(
            policy.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )

        score_meter = AverageMeter()
        loss_meter = AverageMeter()
        epoch_objectives: List[float] = []
        early_validation_objective: float | None = None
        early_stopped = False

        if early_eval_steps_phase is None:
            early_eval_steps_phase = min(100, total_steps)
        else:
            early_eval_steps_phase = min(max(int(early_eval_steps_phase), 0), total_steps)
        log_interval = max(total_steps // 20, 1) if total_steps > 0 else 1

        logger.info(
            "RL4CO baseline PO phase=%s: epochs=%d total_steps=%d (ckpt=%s, device=%s, env=%s)",
            phase,
            int(phase_epochs),
            int(total_steps),
            str(init_ckpt) if init_ckpt else "none",
            str(device),
            _rl4co_env_name(cfg),
        )

        num_rollouts = resolve_pomo_size(cfg.pomo_size, cfg.train_problem_size)
        for step in range(int(total_steps)):
            reward, log_likelihood = _rl4co_rollout(
                env,
                policy,
                cfg.train_batch_size,
                num_rollouts,
                phase="train",
                rollout_strategy=rollout_strategy,
                device=device,
                precision=str(getattr(cfg, "precision", "32-true") or "32-true"),
            )
            reward = reward.float()
            log_likelihood = log_likelihood.float()
            loss, _ = po_loss(
                reward,
                log_likelihood,
                alpha=float(cfg.alpha),
                impl=str(po_impl),
            )

            max_reward, _ = reward.max(dim=1)
            score = _rl4co_objective_from_reward(max_reward, cfg).float().mean()

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            score_meter.update(score.item())
            loss_meter.update(float(loss.item()))

            if (step + 1) % log_interval == 0 or step == 0:
                logger.info(
                    "RL4CO baseline PO[%s] step %d/%d: score=%.6f (avg=%.6f), loss=%.6f (avg=%.6f)",
                    phase,
                    step + 1,
                    total_steps,
                    score.item(),
                    float(score_meter.avg),
                    float(loss.item()),
                    float(loss_meter.avg),
                )

            if steps_per_epoch > 0 and phase_epochs > 0 and (step + 1) % steps_per_epoch == 0:
                epoch_idx = (step + 1) // steps_per_epoch
                if epoch_idx <= int(phase_epochs):
                    epoch_valid_obj = _evaluate_rl4co_model(
                        policy=policy,
                        cfg=cfg,
                        problem_size=cfg.train_problem_size,
                        device=device,
                        num_episodes=cfg.num_validation_episodes,
                        batch_size=cfg.validation_batch_size,
                        rollout_strategy=rollout_strategy,
                    )
                    epoch_objectives.append(epoch_valid_obj)
                    logger.info(
                        "RL4CO baseline PO[%s] epoch %d/%d: validation_objective=%.6f",
                        phase,
                        epoch_idx,
                        int(phase_epochs),
                        epoch_valid_obj,
                    )

            if use_early_stop and early_eval_steps_phase > 0 and (step + 1) == early_eval_steps_phase:
                early_validation_objective = _evaluate_rl4co_model(
                    policy=policy,
                    cfg=cfg,
                    problem_size=cfg.train_problem_size,
                    device=device,
                    num_episodes=cfg.num_validation_episodes,
                    batch_size=cfg.validation_batch_size,
                    rollout_strategy=rollout_strategy,
                )
                if (
                    baseline_early_valid_phase is not None
                    and early_validation_objective > baseline_early_valid_phase
                ):
                    early_stopped = True
                    logger.info(
                        "RL4CO baseline PO early stop[%s] at step %d: candidate early_valid=%.6f baseline_early=%.6f",
                        phase,
                        step + 1,
                        early_validation_objective,
                        baseline_early_valid_phase,
                    )
                    break

        if early_stopped and early_validation_objective is not None:
            final_valid_obj = float(early_validation_objective)
        else:
            final_valid_obj = float(
                _evaluate_rl4co_model(
                    policy=policy,
                    cfg=cfg,
                    problem_size=cfg.train_problem_size,
                    device=device,
                    num_episodes=cfg.num_validation_episodes,
                    batch_size=cfg.validation_batch_size,
                    rollout_strategy=rollout_strategy,
                )
            )

        try:
            env = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        return {
            "phase": phase,
            "policy": policy,
            "rollout_strategy": rollout_strategy,
            "steps": int(total_steps),
            "epochs_total": int(phase_epochs),
            "epoch_objectives": list(epoch_objectives),
            "final_validation_objective": float(final_valid_obj),
            "early_eval": {
                "enabled": bool(early_eval_steps_phase),
                "steps": int(early_eval_steps_phase),
                "baseline_validation_objective": baseline_early_valid_phase,
                "candidate_validation_objective": early_validation_objective,
                "early_stopped": early_stopped,
            },
            "train_score_mean": float(score_meter.avg) if total_steps > 0 else None,
            "train_loss_mean": float(loss_meter.avg) if total_steps > 0 else None,
        }

    if split_enabled:
        scratch_res: Dict[str, Any] | None = None
        warm_res: Dict[str, Any] | None = None
        has_init_ckpt = bool(init_checkpoint_path)

        if has_init_ckpt:
            if warm_epochs > 0:
                warm_res = _run_phase(
                    phase="warmstart",
                    total_steps=int(warm_epochs) * int(steps_per_epoch),
                    phase_epochs=int(warm_epochs),
                    init_ckpt=init_checkpoint_path,
                    use_early_stop=True,
                    early_eval_steps_phase=int(early_eval_steps or 0),
                    baseline_early_valid_phase=baseline_early_valid,
                )
            elif scratch_epochs > 0:
                logger.info(
                    "warmstart_hf_epochs=0 for checkpoint-backed PO baseline init; falling back to scratch phase."
                )
                scratch_res = _run_phase(
                    phase="scratch",
                    total_steps=int(scratch_epochs) * int(steps_per_epoch),
                    phase_epochs=int(scratch_epochs),
                    init_ckpt=None,
                    use_early_stop=False,
                    early_eval_steps_phase=0,
                    baseline_early_valid_phase=None,
                )
        else:
            if scratch_epochs > 0:
                scratch_res = _run_phase(
                    phase="scratch",
                    total_steps=int(scratch_epochs) * int(steps_per_epoch),
                    phase_epochs=int(scratch_epochs),
                    init_ckpt=None,
                    use_early_stop=False,
                    early_eval_steps_phase=0,
                    baseline_early_valid_phase=None,
                )
            elif warm_epochs > 0:
                logger.warning(
                    "warmstart_hf_epochs=%d but init_checkpoint_path is not set; skipping warm-start PO phase.",
                    warm_epochs,
                )

        primary_res = warm_res if warm_res is not None else scratch_res
        if primary_res is None:
            raise RuntimeError("PO baseline split evaluation enabled but no phase was executed.")

        if scratch_res is not None and warm_res is not None:
            primary_phase = "scratch+warmstart"
            primary_epochs_total = int(scratch_epochs + warm_epochs)
            epoch_validation_objectives = list(scratch_res["epoch_objectives"]) + list(
                warm_res["epoch_objectives"]
            )
            early_eval = dict(warm_res.get("early_eval") or {})
        else:
            primary_phase = str(primary_res.get("phase") or "single")
            primary_epochs_total = int(primary_res.get("epochs_total") or 0)
            epoch_validation_objectives = list(primary_res.get("epoch_objectives") or [])
            early_eval = dict(primary_res.get("early_eval") or {})

        policy = primary_res["policy"]
        rollout_strategy = primary_res["rollout_strategy"]
        main_valid_obj = float(primary_res["final_validation_objective"])
        train_score_mean = primary_res.get("train_score_mean")
        train_loss_mean = primary_res.get("train_loss_mean")
        scratch_epoch_eval = (
            {
                "phase": "scratch",
                "epochs_total": int(scratch_res["epochs_total"]),
                "objectives": list(scratch_res["epoch_objectives"]),
                "final_validation_objective": scratch_res["final_validation_objective"],
            }
            if scratch_res is not None and scratch_epochs > 0
            else None
        )
        warmstart_epoch_eval = (
            {
                "phase": "warmstart",
                "epochs_total": int(warm_res["epochs_total"]),
                "objectives": list(warm_res["epoch_objectives"]),
                "final_validation_objective": warm_res["final_validation_objective"],
                "init_checkpoint_path": init_checkpoint_path,
                "init_checkpoint_epoch": init_checkpoint_epoch,
            }
            if warm_res is not None and warm_epochs > 0
            else None
        )
        effective_early_steps = int(early_eval.get("steps") or 0)
        early_validation_objective = early_eval.get("candidate_validation_objective")
    else:
        total_steps = get_total_hf_train_steps(cfg)
        warm_res = _run_phase(
            phase="single",
            total_steps=int(total_steps),
            phase_epochs=int(epochs_total_cfg),
            init_ckpt=init_checkpoint_path,
            use_early_stop=True,
            early_eval_steps_phase=int(early_eval_steps or 0) if early_eval_steps is not None else None,
            baseline_early_valid_phase=baseline_early_valid,
        )
        primary_phase = "single"
        primary_epochs_total = int(epochs_total_cfg)
        epoch_validation_objectives = list(warm_res["epoch_objectives"])
        early_eval = dict(warm_res.get("early_eval") or {})
        policy = warm_res["policy"]
        rollout_strategy = warm_res["rollout_strategy"]
        main_valid_obj = float(warm_res["final_validation_objective"])
        train_score_mean = warm_res.get("train_score_mean")
        train_loss_mean = warm_res.get("train_loss_mean")
        scratch_epoch_eval = None
        warmstart_epoch_eval = None
        effective_early_steps = int(early_eval.get("steps") or 0)
        early_validation_objective = early_eval.get("candidate_validation_objective")

    gen_objectives: Dict[int, float] = {}
    for size in cfg.valid_problem_sizes:
        size_int = int(size)
        gen_objectives[size_int] = _evaluate_rl4co_model(
            policy=policy,
            cfg=cfg,
            problem_size=size_int,
            device=device,
            num_episodes=cfg.num_validation_episodes,
            batch_size=cfg.validation_batch_size,
            rollout_strategy=rollout_strategy,
        )

    size_objectives: Dict[int, float] = {int(cfg.train_problem_size): float(main_valid_obj)}
    for size_int, obj in gen_objectives.items():
        size_objectives[int(size_int)] = float(obj)

    max_gen_obj = max(gen_objectives.values()) if gen_objectives else main_valid_obj
    generalization_penalty = max(0.0, float(max_gen_obj) - float(main_valid_obj))

    epoch_objective_mean: float | None = None
    if epoch_validation_objectives:
        epoch_objective_mean = float(
            sum(epoch_validation_objectives) / len(epoch_validation_objectives)
        )

    epoch_baseline_violations: int | None = None
    epoch_better_than_baseline: bool | None = None
    epoch_tail_baseline_violations: int | None = None
    epoch_tail_better_than_baseline: bool | None = None
    epoch_baseline_margins: List[float] | None = None
    if baseline_epoch_objectives:
        offset = max(int(baseline_epoch_compare_offset or 0), 0)
        baseline_list_full = [float(v) for v in baseline_epoch_objectives]
        baseline_list = baseline_list_full[offset : offset + int(primary_epochs_total)]
        compare_len = min(len(epoch_validation_objectives), len(baseline_list))
        epoch_baseline_margins = []
        for i in range(compare_len):
            margin = float(epoch_validation_objectives[i]) - baseline_list[i]
            epoch_baseline_margins.append(margin)
        violations = sum(1 for m in epoch_baseline_margins if m > 0.0)
        epoch_baseline_violations = int(violations)
        epoch_better_than_baseline = (
            compare_len == int(primary_epochs_total)
            and compare_len == len(epoch_validation_objectives)
            and epoch_baseline_violations == 0
        )

        try:
            tail_frac = float(baseline_epoch_tail_frac or 1.0)
        except (TypeError, ValueError):
            tail_frac = 1.0
        tail_frac = min(max(tail_frac, 0.0), 1.0)
        if (
            compare_len == int(primary_epochs_total)
            and compare_len == len(epoch_validation_objectives)
            and compare_len > 0
            and tail_frac > 0.0
        ):
            tail_count = int(math.ceil(tail_frac * float(compare_len)))
            tail_count = max(1, min(tail_count, compare_len))
            tail_start = compare_len - tail_count
            tail_margins = epoch_baseline_margins[tail_start:]
            tail_violations = sum(1 for m in tail_margins if m > 0.0)
            epoch_tail_baseline_violations = int(tail_violations)
            epoch_tail_better_than_baseline = epoch_tail_baseline_violations == 0

    agg_method = str(getattr(cfg, "size_aggregation", "legacy") or "legacy").strip().lower()
    base_objective = epoch_objective_mean if epoch_objective_mean is not None else float(main_valid_obj)
    if agg_method == "legacy":
        hf_score = float(main_valid_obj) + cfg.generalization_penalty_weight * generalization_penalty
        fitness_score = base_objective + cfg.generalization_penalty_weight * generalization_penalty
    else:
        hf_score = aggregate_objectives_by_size(
            size_objectives,
            method=agg_method,
            cvar_alpha=float(getattr(cfg, "size_cvar_alpha", 0.2)),
        )
        fitness_score = float(hf_score)
    if epoch_baseline_violations is not None:
        fitness_score += float(baseline_epoch_violation_weight) * float(epoch_baseline_violations)

    win_k = int(baseline_epoch_window_k or 10)
    early_mean, late_mean = _epoch_window_means(epoch_validation_objectives, k=win_k)
    epoch_window_eval: Dict[str, Any] = {
        "k": int(win_k),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "objectives": list(epoch_validation_objectives),
    }
    base_early_mean: float | None = None
    base_late_mean: float | None = None
    if baseline_epoch_objectives:
        offset = max(int(baseline_epoch_compare_offset or 0), 0)
        baseline_slice = list(baseline_epoch_objectives)[offset : offset + int(primary_epochs_total)]
        base_early_mean, base_late_mean = _epoch_window_means(baseline_slice, k=win_k)
    baseline_epoch_window_eval: Dict[str, Any] = {
        "early_mean": base_early_mean,
        "late_mean": base_late_mean,
    }
    epoch_window_margins: Dict[str, float] | None = None
    epoch_window_violations: int | None = None
    epoch_window_better_than_baseline: bool | None = None
    if (
        early_mean is not None
        and late_mean is not None
        and base_early_mean is not None
        and base_late_mean is not None
    ):
        early_margin = float(early_mean) - float(base_early_mean)
        late_margin = float(late_mean) - float(base_late_mean)
        epoch_window_margins = {"early": early_margin, "late": late_margin}
        epoch_window_violations = int(sum(1 for m in (early_margin, late_margin) if m > 0.0))
        epoch_window_better_than_baseline = epoch_window_violations == 0
    if epoch_window_violations is not None:
        fitness_score += float(baseline_epoch_window_violation_weight) * float(epoch_window_violations)

    return {
        "hf_score": hf_score,
        "fitness_score": fitness_score,
        "validation_objective": main_valid_obj,
        "generalization_penalty": generalization_penalty,
        "generalization_objectives": gen_objectives,
        "size_objectives": size_objectives,
        "size_aggregation": agg_method,
        "size_cvar_alpha": float(getattr(cfg, "size_cvar_alpha", 0.2)),
        "epoch_objective_mean": epoch_objective_mean,
        "epoch_baseline_violations": epoch_baseline_violations,
        "epoch_better_than_baseline": epoch_better_than_baseline,
        "epoch_tail_baseline_violations": epoch_tail_baseline_violations,
        "epoch_tail_better_than_baseline": epoch_tail_better_than_baseline,
        "baseline_epoch_window_eval": baseline_epoch_window_eval,
        "epoch_window_margins": epoch_window_margins,
        "epoch_window_violations": epoch_window_violations,
        "epoch_window_better_than_baseline": epoch_window_better_than_baseline,
        "scratch_epoch_eval": scratch_epoch_eval,
        "warmstart_epoch_eval": warmstart_epoch_eval,
        "train_score_mean": float(train_score_mean) if train_score_mean is not None else None,
        "train_loss_mean": float(train_loss_mean) if train_loss_mean is not None else None,
        "early_validation_objective": early_validation_objective,
        "early_eval_steps": effective_early_steps,
        "epoch_window_eval": epoch_window_eval,
        "epoch_eval": {
            "enabled": bool(steps_per_epoch),
            "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch > 0 else None,
            "epochs_total": int(primary_epochs_total),
            "objectives": epoch_validation_objectives,
            "objective_mean": epoch_objective_mean,
            "baseline_margins": epoch_baseline_margins,
            "baseline_violations": epoch_baseline_violations,
            "better_than_baseline": epoch_better_than_baseline,
            "phase": primary_phase,
            "baseline_compare_offset": int(baseline_epoch_compare_offset or 0),
            "tail_frac": float(baseline_epoch_tail_frac or 1.0),
            "tail_baseline_violations": epoch_tail_baseline_violations,
            "tail_better_than_baseline": epoch_tail_better_than_baseline,
        },
        "early_eval": {
            "enabled": bool(early_eval.get("enabled")),
            "steps": int(early_eval.get("steps") or 0),
            "baseline_validation_objective": early_eval.get("baseline_validation_objective"),
            "candidate_validation_objective": early_eval.get("candidate_validation_objective"),
            "early_stopped": bool(early_eval.get("early_stopped")),
        },
        "config": {
            "hf": asdict(cfg),
            "baseline_type": "po_loss",
            "init_checkpoint_path": init_checkpoint_path,
            "init_checkpoint_epoch": (
                int(init_checkpoint_epoch)
                if init_checkpoint_epoch is not None
                else _infer_epoch_from_checkpoint_path(str(init_checkpoint_path))
                if init_checkpoint_path
                else None
            ),
            "scratch_hf_epochs": int(scratch_hf_epochs or 0),
            "warmstart_hf_epochs": int(warmstart_hf_epochs or 0),
            "baseline_epoch_compare_offset": int(baseline_epoch_compare_offset or 0),
            "baseline_epoch_violation_weight": float(baseline_epoch_violation_weight),
            "baseline_epoch_tail_frac": float(baseline_epoch_tail_frac or 1.0),
            "baseline_epoch_window_k": int(baseline_epoch_window_k or 10),
            "baseline_epoch_window_violation_weight": float(
                baseline_epoch_window_violation_weight
            ),
        },
    }


def evaluate_free_loss_candidate(
    compiled_loss: CompiledFreeLoss,
    cfg: FreeLossFidelityConfig,
    *,
    baseline_early_valid: float | None = None,
    early_eval_steps: int = 0,
    baseline_epoch_objectives: Sequence[float] | None = None,
    pref_builder: PrefBuilder | None = None,
) -> Dict[str, Any]:
    backend = str(getattr(cfg.hf, "backend", "rl4co") or "rl4co").strip().lower()
    if backend != "rl4co":
        raise NotImplementedError(
            "PTP POMO training backend has been removed; only backend='rl4co' is supported."
        )
    return _evaluate_free_loss_candidate_rl4co(
        compiled_loss,
        cfg,
        baseline_early_valid=baseline_early_valid,
        early_eval_steps=early_eval_steps,
        baseline_epoch_objectives=baseline_epoch_objectives,
        pref_builder=pref_builder,
    )
