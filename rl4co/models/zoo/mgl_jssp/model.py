import json
from pathlib import Path
import sys
from typing import Any, Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader

from rl4co.models.rl.reinforce.free_loss import compile_free_loss, ir_from_json
from rl4co.models.rl.reinforce.preference_losses import slim_loss, sll_loss
from rl4co.models.zoo.mgl_jssp.data import (
    JSSPInstanceDataset,
    JSSPShapeBucketSampler,
    load_dataset,
)
from rl4co.models.zoo.mgl_jssp.net import CAMEncoder3, LSTMDecoder2
from rl4co.models.zoo.mgl_jssp.sampling import (
    Solutions,
    po_loss,
    rl_loss,
    sample_training_pair,
    sampling,
    solve_jsp,
    sro_loss,
    solution_ratio,
    trajectory_log_probs,
)
from rl4co.utils.optim_helpers import create_optimizer
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


def _normalize_shape_list(
    shapes: Sequence[Sequence[int]] | None,
) -> list[tuple[int, int]] | None:
    if shapes is None:
        return None

    normalized: list[tuple[int, int]] = []
    for raw in shapes:
        if len(raw) != 2:
            raise ValueError(f"Each shape must have exactly 2 entries, got {raw!r}.")
        normalized.append((int(raw[0]), int(raw[1])))
    return normalized


def _format_shape_list(shapes: Sequence[tuple[int, int]] | None) -> str:
    if not shapes:
        return "[]"
    return "[" + ", ".join(f"{j}x{m}" for j, m in shapes) + "]"


class MGLJSSPModel(L.LightningModule):
    def __init__(
        self,
        env,
        baseline: str = "bopo",
        train_data_dir: str = "data/jssp_bopo/train",
        val_data_dir: str = "data/jssp_bopo/validation",
        test_data_dir: str | None = None,
        use_cached: bool = True,
        batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        dataloader_num_workers: int = 0,
        optimizer: str = "Adam",
        optimizer_kwargs: dict | None = None,
        enc_hidden: int = 64,
        enc_out: int = 128,
        mem_hidden: int = 64,
        mem_out: int = 128,
        clf_hidden: int = 128,
        B: int = 128,
        K: int = 16,
        D: int = 1,
        pair_mode: str = "anchor_best",
        po_impl: str = "bt",
        po_alpha: float = 1.0,
        sll_impl: str = "sll",
        sll_temperature: float = 1.0,
        val_B: int = 128,
        test_B: int = 128,
        greedy: int = 0,
        init_external_checkpoint_path: str | None = None,
        metrics: dict | None = None,
        log_on_step: bool = False,
        alpha: float = 1.0,
        free_loss_ir_json_path: str | None = None,
        pref_builder_ir_json_path: str | None = None,
        pref_pair_json_path: str | None = None,
        pref_builder_kwargs: dict | None = None,
        free_loss_observables: Sequence[str] | None = None,
        # Phase 4: Bucket-by-shape options
        use_shape_buckets: bool = True,
        bucket_drop_last: bool = False,
        allowed_shapes: list[list[int]] | None = None,
        required_allowed_shapes: list[list[int]] | None = None,
        expected_train_dataset_size: int | None = None,
        expected_val_dataset_size: int | None = None,
        expected_test_dataset_size: int | None = None,
        **unused_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["env"])
        self.env = env
        self.baseline = str(baseline).strip().lower()
        self.optimizer_name = optimizer
        self.optimizer_kwargs = {"lr": 2e-4} if optimizer_kwargs is None else dict(optimizer_kwargs)
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.use_cached = bool(use_cached)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.test_batch_size = int(test_batch_size)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.B = int(B)
        self.K = int(K)
        self.D = int(D)
        self.pair_mode = str(pair_mode or "anchor_best").strip().lower()
        self.po_impl = str(po_impl or "bt").strip().lower()
        self.po_alpha = float(po_alpha)
        self.sll_impl = str(sll_impl or "sll").strip().lower()
        self.sll_temperature = float(sll_temperature)
        self.alpha = float(alpha)
        self.val_B = int(val_B)
        self.test_B = int(test_B)
        self.use_greedy = bool(greedy)
        self.clf_hidden = int(clf_hidden)
        self.log_on_step = bool(log_on_step)
        self.train_metrics = (metrics or {}).get("train", ["loss", "reward"])
        self.val_metrics = (metrics or {}).get("val", ["reward", "gap", "makespan"])
        self.test_metrics = (metrics or {}).get("test", self.val_metrics)
        # Phase 4: Bucket-by-shape options
        self.use_shape_buckets = bool(use_shape_buckets)
        self.bucket_drop_last = bool(bucket_drop_last)
        self.free_loss_ir_json_path = free_loss_ir_json_path
        self.pref_builder_ir_json_path = pref_builder_ir_json_path
        self.pref_pair_json_path = pref_pair_json_path
        self.pref_builder_kwargs = {} if pref_builder_kwargs is None else dict(pref_builder_kwargs)
        self.free_loss_observables = _normalize_free_loss_observables(free_loss_observables)
        self.free_loss = None
        self.pref_builder = None
        self._pref_extract_feature_cache = None
        self._pref_build_runtime_observables = None
        self._pref_batch_cls = None
        # Parse allowed_shapes: [[10,10], [15,15], [20,20]] -> [(10,10), (15,15), (20,20)]
        self.allowed_shapes = _normalize_shape_list(allowed_shapes)
        self.required_allowed_shapes = _normalize_shape_list(required_allowed_shapes)
        self.expected_train_dataset_size = (
            None if expected_train_dataset_size is None else int(expected_train_dataset_size)
        )
        self.expected_val_dataset_size = (
            None if expected_val_dataset_size is None else int(expected_val_dataset_size)
        )
        self.expected_test_dataset_size = (
            None if expected_test_dataset_size is None else int(expected_test_dataset_size)
        )

        if unused_kwargs:
            log.warning("Ignoring unused MGLJSSPModel kwargs: %s", sorted(unused_kwargs.keys()))

        if self.baseline not in {"bopo", "rl", "po", "slim", "sll"}:
            raise ValueError(f"Unsupported MGL JSSP baseline: {self.baseline!r}")
        self._resolve_pref_pair_artifacts()
        self._free_loss_enabled = bool(self.free_loss_ir_json_path)
        if self.pref_builder_ir_json_path:
            self._load_pref_builder()
        if self._free_loss_enabled:
            self._load_free_loss()
        elif self.pref_builder is not None:
            log.warning(
                "pref_builder_ir_json_path is set but free_loss_ir_json_path is missing; "
                "the builder will be ignored"
            )
        # Phase 3: All RL/PO/BOPO support batch_size > 1 for 10x10
        if self.B <= 0 or self.val_B <= 0 or self.test_B <= 0:
            raise ValueError("B / val_B / test_B must be positive.")
        if self.D <= 0:
            raise ValueError("D must be positive.")
        if self.baseline == "bopo" and self.B % self.K != 0:
            raise ValueError(f"MGL JSSP BOPO requires B % K == 0, got B={self.B}, K={self.K}.")
        if self.po_alpha <= 0:
            raise ValueError(f"MGL JSSP po_alpha must be positive, got {self.po_alpha}.")
        if self.sll_impl not in {"sll", "slim", "listnet"}:
            raise ValueError(f"Unsupported MGL JSSP sll_impl: {self.sll_impl!r}")
        if self.sll_temperature <= 0:
            raise ValueError(
                f"MGL JSSP sll_temperature must be positive, got {self.sll_temperature}."
            )
        if self.required_allowed_shapes is not None:
            if self.allowed_shapes is None:
                raise ValueError(
                    "required_allowed_shapes was provided but allowed_shapes is unset. "
                    f"Expected {_format_shape_list(self.required_allowed_shapes)}."
                )
            if self.allowed_shapes != self.required_allowed_shapes:
                raise ValueError(
                    "allowed_shapes does not match required_allowed_shapes: "
                    f"got {_format_shape_list(self.allowed_shapes)}, "
                    f"expected {_format_shape_list(self.required_allowed_shapes)}."
                )

        self.encoder = CAMEncoder3(15, hidden_size=enc_hidden, embed_size=enc_out)
        self.decoder = LSTMDecoder2(
            encoder_size=self.encoder.out_size,
            context_size=11,
            hidden_size=mem_hidden,
            att_size=mem_out,
        )

        if init_external_checkpoint_path:
            self._load_external_checkpoint(init_external_checkpoint_path)

    def _load_external_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"init_external_checkpoint_path not found: {path.as_posix()}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, tuple) and len(payload) == 2:
            enc_state, decoder = payload
            self.encoder.load_state_dict(enc_state)
            self.decoder.load_state_dict(decoder.state_dict())
        else:
            raise ValueError(
                f"Unsupported external checkpoint format for MGL JSSP init: {path.as_posix()}"
            )
        log.info("Loaded external MGL checkpoint from %s", path.as_posix())

    @staticmethod
    def _ensure_ptp_root_on_path() -> None:
        repo_root = Path(__file__).resolve().parents[4]
        ptp_root = repo_root / "PTP"
        ptp_root_str = str(ptp_root.resolve())
        if ptp_root.is_dir() and ptp_root_str not in sys.path:
            sys.path.insert(0, ptp_root_str)

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
            raise FileNotFoundError(
                f"pref_pair_json_path does not exist: {pair_path.as_posix()}"
            )

        run_dir = pair_path.parent
        builder_path = run_dir / "best_builder.json"
        loss_path = run_dir / "best_loss.json"
        if self.pref_builder_ir_json_path is None:
            if not builder_path.is_file():
                raise FileNotFoundError(
                    "best_builder.json not found next to pref_pair_json_path: "
                    f"{builder_path.as_posix()}"
                )
            self.pref_builder_ir_json_path = builder_path.as_posix()
        if self.free_loss_ir_json_path is None:
            if not loss_path.is_file():
                raise FileNotFoundError(
                    "best_loss.json not found next to pref_pair_json_path: "
                    f"{loss_path.as_posix()}"
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

    def _load_pref_builder(self) -> None:
        if self.pref_builder_ir_json_path is None:
            raise ValueError(
                "pref_builder_ir_json_path must be set before loading a preference builder."
            )

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

    def _load_free_loss(self) -> None:
        if self.free_loss_ir_json_path is None:
            raise ValueError(
                "When pref-pair/free-loss training is enabled, free_loss_ir_json_path must be set."
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

    def _free_loss_rollout(
        self, instances: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.free_loss is None:
            raise RuntimeError("free_loss is not compiled; check pref_pair/free-loss paths.")
        self._load_free_loss_runtime_helpers()
        if self._pref_extract_feature_cache is None or self._pref_build_runtime_observables is None:
            raise RuntimeError("Free-loss runtime helpers are not initialized.")

        trajs, logits, makespans, entropies = solve_jsp(
            instances,
            batch_size_per_instance=self.B,
            device=str(self.device),
            encoder=self.encoder,
            decoder=self.decoder,
            use_greedy=self.use_greedy,
        )

        num_instances = len(instances)
        num_steps = int(trajs.size(1))
        num_jobs = int(instances[0]["j"])
        logits_reshaped = logits.view(num_instances, self.B, num_steps, num_jobs)
        trajs_reshaped = trajs.view(num_instances, self.B, num_steps)
        objective = makespans.view(num_instances, self.B).float()
        reward_matrix = -objective

        step_log_prob = torch.log_softmax(logits_reshaped, dim=-1).gather(
            -1, trajs_reshaped.unsqueeze(-1)
        ).squeeze(-1)
        log_likelihood = step_log_prob.sum(dim=-1)
        entropy_total = entropies.view(num_instances, self.B, num_steps).sum(dim=-1)
        seq_len = torch.full_like(log_likelihood, float(num_steps))

        extra = self._pref_build_runtime_observables(
            reward_matrix,
            log_likelihood,
            observables=self.free_loss_observables,
            seq_len=seq_len,
            log_prob_step=step_log_prob if "log_prob_step" in set(self.free_loss_observables) else None,
            entropy=entropy_total,
            seq_len_fallback=num_steps,
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
            pair_count_value = int(pref_batch.num_examples())
            loss_batch = (
                pref_batch.to_pairwise_loss_batch(feature_cache) if pair_count_value > 0 else {}
            )
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

        if pair_count_value == 0:
            advantage = reward_matrix - reward_matrix.mean(dim=1, keepdim=True)
            loss = -(advantage.detach() * log_likelihood).mean()
        else:
            loss = self.free_loss.loss_fn(
                batch=loss_batch,
                model_output=feature_cache,
                extra={"alpha": self.alpha},
            )

        best_makespan = objective.min(dim=1)[0].min()
        reward = -best_makespan.to(self.device)
        best_per_instance = objective.min(dim=1)[0].clamp_min(1e-8)
        worst_per_instance = objective.max(dim=1)[0].clamp_min(1e-8)
        quality = (worst_per_instance / best_per_instance).mean()
        aux_metric = quality.to(device=self.device, dtype=torch.float32)
        pair_count = torch.tensor(float(pair_count_value), device=self.device)
        return loss, reward, aux_metric, pair_count

    def _filter_instances_by_allowed_shapes(
        self, instances: list[dict[str, Any]], split_name: str, data_dir: str
    ) -> list[dict[str, Any]]:
        if self.allowed_shapes is None:
            return instances

        filtered = [
            ins for ins in instances if (int(ins["j"]), int(ins["m"])) in self.allowed_shapes
        ]
        if not filtered:
            allowed = ", ".join(f"{j}x{m}" for j, m in self.allowed_shapes)
            raise ValueError(
                f"No {split_name} instances match allowed_shapes=[{allowed}] in {data_dir}."
            )

        if len(filtered) != len(instances):
            log.info(
                "Filtered %s split by allowed_shapes: %d -> %d instances",
                split_name,
                len(instances),
                len(filtered),
            )
        return filtered

    def _validate_dataset_shapes_and_size(
        self,
        instances: list[dict[str, Any]],
        split_name: str,
        expected_size: int | None,
    ) -> None:
        shape_counts: dict[tuple[int, int], int] = {}
        for ins in instances:
            shape = (int(ins["j"]), int(ins["m"]))
            shape_counts[shape] = shape_counts.get(shape, 0) + 1

        log.info(
            "JSSP %s split loaded %d instances with shapes: %s",
            split_name,
            len(instances),
            ", ".join(
                f"{shape[0]}x{shape[1]}={count}"
                for shape, count in sorted(shape_counts.items())
            ) or "none",
        )

        if self.required_allowed_shapes is not None:
            disallowed = sorted(set(shape_counts) - set(self.required_allowed_shapes))
            if disallowed:
                raise ValueError(
                    f"{split_name} split contains shapes outside required_allowed_shapes: "
                    f"got {_format_shape_list(disallowed)}, "
                    f"required {_format_shape_list(self.required_allowed_shapes)}."
                )

        if expected_size is not None and len(instances) != expected_size:
            raise ValueError(
                f"{split_name} split size mismatch: got {len(instances)}, expected {expected_size}. "
                f"allowed_shapes={_format_shape_list(self.allowed_shapes)}."
            )

    def setup(self, stage: str | None = None) -> None:
        train_instances = load_dataset(self.train_data_dir, use_cached=self.use_cached, device="cpu")
        val_instances = load_dataset(self.val_data_dir, use_cached=self.use_cached, device="cpu")
        train_instances = self._filter_instances_by_allowed_shapes(
            train_instances, "train", self.train_data_dir
        )
        val_instances = self._filter_instances_by_allowed_shapes(
            val_instances, "val", self.val_data_dir
        )
        self._validate_dataset_shapes_and_size(
            train_instances, "train", self.expected_train_dataset_size
        )
        self._validate_dataset_shapes_and_size(
            val_instances, "val", self.expected_val_dataset_size
        )
        self.train_dataset = JSSPInstanceDataset(train_instances)
        self.val_dataset = JSSPInstanceDataset(val_instances)
        if self.test_data_dir:
            test_instances = load_dataset(self.test_data_dir, use_cached=self.use_cached, device="cpu")
            test_instances = self._filter_instances_by_allowed_shapes(
                test_instances, "test", self.test_data_dir
            )
            self._validate_dataset_shapes_and_size(
                test_instances, "test", self.expected_test_dataset_size
            )
            self.test_dataset = JSSPInstanceDataset(test_instances)
        else:
            self.test_dataset = self.val_dataset

        # Phase 4: Log bucket stats if using shape buckets
        if self.use_shape_buckets and stage == "fit":
            sampler = JSSPShapeBucketSampler(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=self.bucket_drop_last,
                allowed_shapes=self.allowed_shapes,
            )
            bucket_stats = sampler.get_bucket_stats()
            log.info("=== JSSP Shape Bucket Stats ===")
            total_instances = 0
            total_batches = 0
            for shape, (n_inst, n_batches) in sorted(bucket_stats.items()):
                log.info(f"  Shape {shape[0]}x{shape[1]}: {n_inst} instances, {n_batches} batches")
                total_instances += n_inst
                total_batches += n_batches
            log.info(f"  TOTAL: {total_instances} instances, {total_batches} batches")
            log.info("===============================")

    def configure_optimizers(self):
        return create_optimizer(self.parameters(), self.optimizer_name, **self.optimizer_kwargs)

    def training_step(self, batch: list[dict[str, Any]] | dict[str, Any], batch_idx: int):
        loss_total = None
        reward_total = None
        aux_total = None
        pair_count_total = None

        # Get actual batch size (number of instances)
        actual_batch_size = len(batch) if isinstance(batch, list) else 1

        # Phase 4: Log current batch shape
        if self.log_on_step:
            first_instance = batch[0] if isinstance(batch, list) else batch
            current_shape = (first_instance["j"], first_instance["m"])
            self.log(
                "train/shape_j",
                float(current_shape[0]),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
                batch_size=actual_batch_size,
            )
            self.log(
                "train/shape_m",
                float(current_shape[1]),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
                batch_size=actual_batch_size,
            )

        for _ in range(self.D):
            loss, reward, aux_metric, pair_count = self._training_rollout(batch)
            loss_total = loss if loss_total is None else (loss_total + loss)
            reward_total = reward if reward_total is None else (reward_total + reward)
            aux_total = aux_metric if aux_total is None else (aux_total + aux_metric)
            if pair_count is not None:
                pair_count_total = pair_count if pair_count_total is None else (pair_count_total + pair_count)

        loss_total = loss_total / self.D
        reward_total = reward_total / self.D
        aux_total = aux_total / self.D
        if pair_count_total is not None:
            pair_count_total = pair_count_total / self.D

        if "loss" in self.train_metrics:
            self.log(
                "train/loss",
                loss_total,
                on_step=self.log_on_step,
                on_epoch=not self.log_on_step,
                prog_bar=True,
                sync_dist=True,
                batch_size=actual_batch_size,
            )
        if "reward" in self.train_metrics:
            self.log(
                "train/reward",
                reward_total,
                on_step=self.log_on_step,
                on_epoch=not self.log_on_step,
                prog_bar=True,
                sync_dist=True,
                batch_size=actual_batch_size,
            )
        if "quality" in self.train_metrics:
            self.log(
                "train/quality",
                aux_total,
                on_step=self.log_on_step,
                on_epoch=not self.log_on_step,
                prog_bar=False,
                sync_dist=True,
                batch_size=actual_batch_size,
            )
        if pair_count_total is not None and "pair_count" in self.train_metrics:
            self.log(
                "train/pair_count",
                pair_count_total,
                on_step=self.log_on_step,
                on_epoch=not self.log_on_step,
                prog_bar=False,
                sync_dist=True,
                batch_size=actual_batch_size,
            )
        return loss_total

    def _training_rollout(
        self, batch: list[dict[str, Any]] | dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        device = str(self.device)

        # Phase 4: All RL/PO/BOPO support batch as list[dict]
        instances = batch if isinstance(batch, list) else [batch]
        num_instances = len(instances)

        # Verify all instances have the same shape
        first_shape = (instances[0]["j"], instances[0]["m"])
        for i, ins in enumerate(instances):
            shape = (ins["j"], ins["m"])
            if shape != first_shape:
                raise ValueError(f"All instances must have same shape: instance 0 is {first_shape[0]}x{first_shape[1]}, instance {i} is {shape[0]}x{shape[1]}")

        if self._free_loss_enabled:
            return self._free_loss_rollout(instances)

        if self.baseline == "bopo":
            better, worse, best_makespan, total_num_pairs = sample_training_pair(
                instances,
                self.encoder,
                self.decoder,
                B=self.B,
                K=self.K,
                use_greedy=self.use_greedy,
                pair_mode=self.pair_mode,
                device=device,
            )
            loss, quality = sro_loss(better, worse)
            pair_count = torch.tensor(float(total_num_pairs), device=self.device)
            reward = -best_makespan.to(self.device)
            aux_metric = torch.tensor(float(quality), dtype=torch.float32, device=self.device)
            return loss, reward, aux_metric, pair_count

        if self.baseline == "po":
            # Solve for all instances
            trajs, logits, makespans, _ = solve_jsp(
                instances,
                batch_size_per_instance=self.B,
                device=device,
                encoder=self.encoder,
                decoder=self.decoder,
                use_greedy=self.use_greedy,
            )

            # Reshape to (num_instances, B, ...) for per-instance loss aggregation
            num_steps = trajs.size(1)
            num_jobs = instances[0]["j"]
            trajs_reshaped = trajs.view(num_instances, self.B, num_steps)
            logits_reshaped = logits.view(num_instances, self.B, num_steps, num_jobs)
            makespans_reshaped = makespans.view(num_instances, self.B)

            # Compute PO loss per-instance (strictly within-instance pairs), then average
            total_loss = 0.0
            total_quality = 0.0
            best_makespan_list = []

            for i in range(num_instances):
                samples_i = Solutions(
                    trajs=trajs_reshaped[i],
                    logits=logits_reshaped[i],
                    mss=makespans_reshaped[i]
                )
                loss_i, quality_i = po_loss(samples_i, impl=self.po_impl, alpha=self.po_alpha)
                total_loss = total_loss + loss_i
                total_quality = total_quality + quality_i
                best_makespan_list.append(makespans_reshaped[i].min())

            loss = total_loss / num_instances
            quality = total_quality / num_instances
            best_makespan = torch.stack(best_makespan_list).min()

            reward = -best_makespan.to(self.device)
            aux_metric = torch.tensor(float(quality), dtype=torch.float32, device=self.device)
            return loss, reward, aux_metric, None

        if self.baseline in {"slim", "sll"}:
            trajs, logits, makespans, _ = solve_jsp(
                instances,
                batch_size_per_instance=self.B,
                device=device,
                encoder=self.encoder,
                decoder=self.decoder,
                use_greedy=self.use_greedy,
            )

            num_steps = trajs.size(1)
            num_jobs = instances[0]["j"]
            trajs_reshaped = trajs.view(num_instances, self.B, num_steps)
            logits_reshaped = logits.view(num_instances, self.B, num_steps, num_jobs)
            makespans_reshaped = makespans.view(num_instances, self.B)

            total_loss = 0.0
            total_quality = 0.0
            best_makespan_list = []

            for i in range(num_instances):
                log_probs_i = trajectory_log_probs(logits_reshaped[i], trajs_reshaped[i]).unsqueeze(0)
                reward_i = (-makespans_reshaped[i]).unsqueeze(0)
                if self.baseline == "slim":
                    loss_i = slim_loss(
                        reward_i,
                        log_probs_i,
                        sequence_length=float(num_steps),
                    )
                else:
                    loss_i = sll_loss(
                        reward_i,
                        log_probs_i,
                        alpha=self.alpha,
                        impl=self.sll_impl,
                        temperature=self.sll_temperature,
                    )
                total_loss = total_loss + loss_i
                total_quality = total_quality + solution_ratio(makespans_reshaped[i])
                best_makespan_list.append(makespans_reshaped[i].min())

            loss = total_loss / num_instances
            quality = total_quality / num_instances
            best_makespan = torch.stack(best_makespan_list).min()

            reward = -best_makespan.to(self.device)
            aux_metric = torch.tensor(float(quality), dtype=torch.float32, device=self.device)
            return loss, reward, aux_metric, None

        # RL baseline: supports batch_size > 1 (list of instances)
        trajs, logits, makespans, _ = solve_jsp(
            instances,
            batch_size_per_instance=self.B,
            device=device,
            encoder=self.encoder,
            decoder=self.decoder,
            use_greedy=self.use_greedy,
        )

        # Reshape to (num_instances, B, ...) for per-instance loss aggregation
        num_steps = trajs.size(1)
        num_jobs = instances[0]["j"]
        trajs_reshaped = trajs.view(num_instances, self.B, num_steps)
        logits_reshaped = logits.view(num_instances, self.B, num_steps, num_jobs)
        makespans_reshaped = makespans.view(num_instances, self.B)

        # Compute loss per-instance, then average
        total_loss = 0.0
        total_quality = 0.0
        best_makespan_list = []

        for i in range(num_instances):
            samples_i = Solutions(
                trajs=trajs_reshaped[i],
                logits=logits_reshaped[i],
                mss=makespans_reshaped[i]
            )
            loss_i, quality_i = rl_loss(samples_i)
            total_loss = total_loss + loss_i
            total_quality = total_quality + quality_i
            best_makespan_list.append(makespans_reshaped[i].min())

        loss = total_loss / num_instances
        quality = total_quality / num_instances
        best_makespan = torch.stack(best_makespan_list).min()

        reward = -best_makespan.to(self.device)
        aux_metric = torch.tensor(float(quality), dtype=torch.float32, device=self.device)
        return loss, reward, aux_metric, None

    def validation_step(self, batch: list[dict[str, Any]] | dict[str, Any], batch_idx: int):
        return self._eval_step(batch, phase="val", sample_size=self.val_B)

    def test_step(self, batch: list[dict[str, Any]] | dict[str, Any], batch_idx: int):
        return self._eval_step(batch, phase="test", sample_size=self.test_B)

    def _eval_step(self, batch: list[dict[str, Any]] | dict[str, Any], phase: str, sample_size: int):
        # Phase 7: eval supports multi-batch (same-shape only)
        instances = batch if isinstance(batch, list) else [batch]
        num_instances = len(instances)
        actual_batch_size = num_instances

        # Sample for all instances (each instance independently)
        # sampling() accepts list, returns (N*B,)
        makespans, entropies, _ = sampling(
            instances,
            self.encoder,
            self.decoder,
            bs=sample_size,
            use_greedy=self.use_greedy,
            device=str(self.device),
        )

        # Reshape to (num_instances, sample_size)
        makespans_reshaped = makespans.view(num_instances, sample_size)
        # entropies is (N*B, num_steps), take mean across all dims
        mean_entropy = entropies.mean()

        # Compute per-instance best makespan
        best_makespans = makespans_reshaped.min(dim=1)[0]

        # Aggregate: average reward, min makespan across batch
        mean_reward = (-best_makespans).mean()
        min_makespan = best_makespans.min()

        # Compute gaps if reference makespans are available
        gaps = []
        for i, instance in enumerate(instances):
            ref_makespan = instance.get("makespan", None)
            if ref_makespan is not None:
                ref = float(ref_makespan.item()) if isinstance(ref_makespan, torch.Tensor) else float(ref_makespan)
                if ref > 0:
                    gap_val = (best_makespans[i] / ref - 1.0) * 100.0
                    gaps.append(gap_val)

        metrics = {
            "reward": mean_reward,
            "makespan": min_makespan,
            "entropy": mean_entropy,
        }
        if gaps:
            metrics["gap"] = torch.tensor(gaps, device=self.device).mean()

        metric_names = self.val_metrics if phase == "val" else self.test_metrics
        for name, value in metrics.items():
            if name in metric_names:
                self.log(
                    f"{phase}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=name in {"reward", "gap", "makespan"},
                    sync_dist=True,
                    batch_size=actual_batch_size,
                )
        return metrics

    def train_dataloader(self):
        if self.use_shape_buckets:
            # Phase 4: Use shape bucket sampler
            sampler = JSSPShapeBucketSampler(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=self.bucket_drop_last,
                allowed_shapes=self.allowed_shapes,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.dataloader_num_workers,
                collate_fn=JSSPInstanceDataset.collate_fn,
            )
        else:
            # Original: single-shape only
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.dataloader_num_workers,
                collate_fn=JSSPInstanceDataset.collate_fn,
            )

    def val_dataloader(self):
        # Val/test: always use simple DataLoader (no need for bucketing during evaluation)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=JSSPInstanceDataset.collate_fn,
        )

    def test_dataloader(self):
        # Val/test: always use simple DataLoader (no need for bucketing during evaluation)
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=JSSPInstanceDataset.collate_fn,
        )
