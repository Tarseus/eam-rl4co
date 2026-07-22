from __future__ import annotations

import math
import random

from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ShapeBucketBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups dataset indices by shape bucket.

    Every yielded mini-batch contains indices from a single bucket only.
    """

    def __init__(
        self,
        bucket_ids: Sequence[str],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.bucket_ids = list(bucket_ids)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

        bucket_to_indices = defaultdict(list)
        for idx, bucket_id in enumerate(self.bucket_ids):
            bucket_to_indices[bucket_id].append(idx)
        self._bucket_to_indices = dict(bucket_to_indices)

    def __iter__(self) -> Iterator[list[int]]:
        all_batches: list[list[int]] = []
        for indices in self._bucket_to_indices.values():
            local = list(indices)
            if self.shuffle:
                random.shuffle(local)

            for start in range(0, len(local), self.batch_size):
                batch = local[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self._bucket_to_indices.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self._bucket_to_indices.values()
        )
