from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

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
)
from rl4co.utils.optim_helpers import create_optimizer
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


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
        val_B: int = 128,
        test_B: int = 128,
        greedy: int = 0,
        init_external_checkpoint_path: str | None = None,
        metrics: dict | None = None,
        log_on_step: bool = False,
        # Phase 4: Bucket-by-shape options
        use_shape_buckets: bool = True,
        bucket_drop_last: bool = False,
        allowed_shapes: list[list[int]] | None = None,
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
        # Parse allowed_shapes: [[10,10], [15,15], [20,20]] -> [(10,10), (15,15), (20,20)]
        if allowed_shapes is None:
            self.allowed_shapes = None
        else:
            self.allowed_shapes = [tuple(s) for s in allowed_shapes]

        if unused_kwargs:
            log.warning("Ignoring unused MGLJSSPModel kwargs: %s", sorted(unused_kwargs.keys()))

        if self.baseline not in {"bopo", "rl", "po"}:
            raise ValueError(f"Unsupported MGL JSSP baseline: {self.baseline!r}")
        # Phase 3: All RL/PO/BOPO support batch_size > 1 for 10x10
        if self.B <= 0 or self.val_B <= 0 or self.test_B <= 0:
            raise ValueError("B / val_B / test_B must be positive.")
        if self.D <= 0:
            raise ValueError("D must be positive.")
        if self.baseline == "bopo" and self.B % self.K != 0:
            raise ValueError(f"MGL JSSP BOPO requires B % K == 0, got B={self.B}, K={self.K}.")

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

    def setup(self, stage: str | None = None) -> None:
        train_instances = load_dataset(self.train_data_dir, use_cached=self.use_cached, device="cpu")
        val_instances = load_dataset(self.val_data_dir, use_cached=self.use_cached, device="cpu")
        self.train_dataset = JSSPInstanceDataset(train_instances)
        self.val_dataset = JSSPInstanceDataset(val_instances)
        if self.test_data_dir:
            test_instances = load_dataset(self.test_data_dir, use_cached=self.use_cached, device="cpu")
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
        if self.baseline == "po" and "pref_rate" in self.train_metrics:
            self.log(
                "train/pref_rate",
                aux_total,
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
            total_pref_rate = 0.0
            best_makespan_list = []

            for i in range(num_instances):
                samples_i = Solutions(
                    trajs=trajs_reshaped[i],
                    logits=logits_reshaped[i],
                    mss=makespans_reshaped[i]
                )
                loss_i, pref_rate_i = po_loss(samples_i, impl=self.po_impl)
                total_loss = total_loss + loss_i
                total_pref_rate = total_pref_rate + pref_rate_i
                best_makespan_list.append(makespans_reshaped[i].min())

            loss = total_loss / num_instances
            quality = total_pref_rate / num_instances
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
        if self.use_shape_buckets:
            # Phase 7: Use shape bucket sampler for val too
            sampler = JSSPShapeBucketSampler(
                self.val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                drop_last=False,
                allowed_shapes=self.allowed_shapes,
            )
            return DataLoader(
                self.val_dataset,
                batch_sampler=sampler,
                num_workers=self.dataloader_num_workers,
                collate_fn=JSSPInstanceDataset.collate_fn,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
                collate_fn=JSSPInstanceDataset.collate_fn,
            )

    def test_dataloader(self):
        if self.use_shape_buckets:
            # Phase 7: Use shape bucket sampler for test too
            sampler = JSSPShapeBucketSampler(
                self.test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                drop_last=False,
                allowed_shapes=self.allowed_shapes,
            )
            return DataLoader(
                self.test_dataset,
                batch_sampler=sampler,
                num_workers=self.dataloader_num_workers,
                collate_fn=JSSPInstanceDataset.collate_fn,
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
                collate_fn=JSSPInstanceDataset.collate_fn,
            )
