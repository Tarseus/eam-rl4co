from __future__ import annotations

import torch

from tensordict import TensorDict
from torch.utils.data import Dataset

from rl4co.data.samplers import ShapeBucketBatchSampler


class JSSPBucketedDataset(Dataset):
    """Heterogeneous-by-shape dataset for JSSP bucketed batching.

    Samples may come from different shapes globally, but a batch sampler must ensure
    each batch only contains one shape bucket.
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]], bucket_ids: list[str]):
        if len(samples) != len(bucket_ids):
            raise ValueError(
                f"samples and bucket_ids must have same length, got {len(samples)} vs {len(bucket_ids)}"
            )
        self.samples = samples
        self.bucket_ids = bucket_ids
        self._extra: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = dict(self.samples[idx])
        for key, value in self._extra.items():
            item[key] = value[idx]
        return item

    def add_key(self, key: str, value: torch.Tensor):
        if len(value) != len(self.samples):
            raise ValueError(
                f"extra value length mismatch for key '{key}': {len(value)} vs dataset {len(self.samples)}"
            )
        self._extra[key] = value
        return self

    def get_batch_sampler(self, batch_size: int, shuffle: bool = False):
        return ShapeBucketBatchSampler(
            bucket_ids=self.bucket_ids,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> TensorDict:
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        ref_shapes = {k: v.shape for k, v in batch[0].items()}
        for item in batch[1:]:
            item_shapes = {k: v.shape for k, v in item.items()}
            if item_shapes != ref_shapes:
                raise ValueError(
                    "Mixed-shape instances are not allowed in the same batch. "
                    f"Reference shapes: {ref_shapes}, got: {item_shapes}"
                )

        stacked = {
            key: torch.stack([entry[key] for entry in batch], dim=0) for key in batch[0]
        }
        return TensorDict(stacked, batch_size=torch.Size([len(batch)]))
