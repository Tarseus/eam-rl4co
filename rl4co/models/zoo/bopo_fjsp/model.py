from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from rl4co.models.zoo.bopo_fjsp.data import FJSPInstanceDataset, load_dataset
from rl4co.models.zoo.bopo_fjsp.net import CAMEncoder, LSTMDecoder
from rl4co.models.zoo.bopo_fjsp.sampling import FlexibleJobShopStates, sample_training_pair, sampling, sro_loss
from rl4co.utils.optim_helpers import create_optimizer
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BOPOFJSPModel(L.LightningModule):
    def __init__(
        self,
        env,
        train_data_dir: str = "BOPO/FJSP/dataset",
        val_data_dir: str = "BOPO/FJSP/benchmarks/validation",
        test_data_dir: str | None = "BOPO/FJSP/benchmarks/LA-e",
        use_cached: bool = True,
        batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        optimizer: str = "Adam",
        optimizer_kwargs: dict | None = None,
        enc_hidden: int = 64,
        enc_out: int = 128,
        mem_hidden: int = 64,
        mem_out: int = 128,
        clf_hidden: int = 128,
        B: int = 256,
        K: int = 16,
        val_B: int = 256,
        test_B: int = 32,
        greedy: int = 1,
        init_external_checkpoint_path: str | None = None,
        metrics: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["env"])
        self.env = env
        self.optimizer_name = optimizer
        self.optimizer_kwargs = {"lr": 2e-5} if optimizer_kwargs is None else dict(optimizer_kwargs)
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.use_cached = bool(use_cached)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.test_batch_size = int(test_batch_size)
        self.B = int(B)
        self.K = int(K)
        self.val_B = int(val_B)
        self.test_B = int(test_B)
        self.use_greedy = bool(greedy)
        self.clf_hidden = int(clf_hidden)
        self.train_metrics = (metrics or {}).get("train", ["loss", "reward"])
        self.val_metrics = (metrics or {}).get("val", ["reward", "gap", "makespan"])
        self.test_metrics = (metrics or {}).get("test", self.val_metrics)

        context_size = FlexibleJobShopStates.size
        self.encoder = CAMEncoder(15, hidden_size=enc_hidden, embed_size=enc_out)
        self.decoder = LSTMDecoder(
            encoder_size=self.encoder.out_size,
            context_size=context_size,
            hidden_size=mem_hidden,
            att_size=mem_out,
        )

        if self.batch_size != 1 or self.val_batch_size != 1 or self.test_batch_size != 1:
            raise ValueError("BOPOFJSPModel uses one instance per optimizer step; keep batch_size/val_batch_size/test_batch_size at 1.")
        if self.B % self.K != 0:
            raise ValueError(f"BOPO FJSP requires B % K == 0, got B={self.B}, K={self.K}.")

        if init_external_checkpoint_path:
            self._load_external_checkpoint(init_external_checkpoint_path)

    def _load_external_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"init_external_checkpoint_path not found: {path.as_posix()}")
        enc_state, decoder = torch.load(path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(enc_state)
        self.decoder.load_state_dict(decoder.state_dict())
        log.info("Loaded external BOPO checkpoint from %s", path.as_posix())

    def setup(self, stage: str | None = None) -> None:
        train_instances = load_dataset(self.train_data_dir, use_cached=self.use_cached, device="cpu")
        val_instances = load_dataset(self.val_data_dir, use_cached=self.use_cached, device="cpu")
        self.train_dataset = FJSPInstanceDataset(train_instances)
        self.val_dataset = FJSPInstanceDataset(val_instances)
        if self.test_data_dir:
            test_instances = load_dataset(self.test_data_dir, use_cached=self.use_cached, device="cpu")
            self.test_dataset = FJSPInstanceDataset(test_instances)
        else:
            self.test_dataset = self.val_dataset

    def configure_optimizers(self):
        return create_optimizer(self.parameters(), self.optimizer_name, **self.optimizer_kwargs)

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        better, worse, makespan = sample_training_pair(
            batch,
            self.encoder,
            self.decoder,
            B=self.B,
            K=self.K,
            use_greedy=self.use_greedy,
            device=str(self.device),
        )
        loss = sro_loss(better, worse)
        reward = -torch.tensor(float(min(makespan)), device=self.device)
        if "loss" in self.train_metrics:
            self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=1)
        if "reward" in self.train_metrics:
            self.log("train/reward", reward, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=1)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        return self._eval_step(batch, phase="val", sample_size=self.val_B)

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        return self._eval_step(batch, phase="test", sample_size=self.test_B)

    def _eval_step(self, batch: dict[str, Any], phase: str, sample_size: int):
        makespans = torch.tensor(
            sampling(
                batch,
                self.encoder,
                self.decoder,
                bs=sample_size,
                use_greedy=self.use_greedy,
                device=str(self.device),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        best_makespan = makespans.min()
        reward = -best_makespan
        metrics = {"reward": reward, "makespan": best_makespan}
        if batch.get("makespan") is not None:
            gap = (best_makespan / float(batch["makespan"]) - 1.0) * 100.0
            metrics["gap"] = gap
        metric_names = self.val_metrics if phase == "val" else self.test_metrics
        for name, value in metrics.items():
            if name in metric_names:
                self.log(
                    f"{phase}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=1,
                )
        return metrics

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=FJSPInstanceDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=FJSPInstanceDataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=FJSPInstanceDataset.collate_fn,
        )
