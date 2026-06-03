from __future__ import annotations

import math
import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


class SlideLabelBalancedBatchSampler(Sampler[List[int]]):
    """
    For validation:
    - each batch tries to sample half positive / half negative
    - based on dataset.df['slide_label']
    - oversamples smaller class if needed
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_batches: int | None = None,
        drop_last: bool = True,
        seed: int = 42,
        verbose: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if "slide_label" not in dataset.df.columns:
            raise ValueError("dataset.df must contain 'slide_label'")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.verbose = verbose

        df = dataset.df.reset_index(drop=True)

        self.pos_indices = df.index[df["slide_label"].fillna(-1).astype(int) == 1].tolist()
        self.neg_indices = df.index[df["slide_label"].fillna(-1).astype(int) == 0].tolist()

        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise RuntimeError(
                f"Need both positive and negative samples in val set, "
                f"got pos={len(self.pos_indices)}, neg={len(self.neg_indices)}"
            )

        if num_batches is None:
            if self.drop_last:
                self.num_batches = max(1, len(dataset) // self.batch_size)
            else:
                self.num_batches = max(1, math.ceil(len(dataset) / self.batch_size))
        else:
            self.num_batches = int(num_batches)

        if self.verbose:
            print("[SlideLabelBalancedBatchSampler]")
            print(f"  pos = {len(self.pos_indices)}")
            print(f"  neg = {len(self.neg_indices)}")
            print(f"  batch_size = {self.batch_size}")
            print(f"  num_batches = {self.num_batches}")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)

        pos_pool = list(self.pos_indices)
        neg_pool = list(self.neg_indices)
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)

        pos_ptr = 0
        neg_ptr = 0

        pos_take = self.batch_size // 2
        neg_take = self.batch_size - pos_take

        for _ in range(self.num_batches):
            batch = []

            def take_from_pool(pool, ptr, need):
                chosen = []
                while len(chosen) < need:
                    remain = len(pool) - ptr
                    if remain <= 0:
                        rng.shuffle(pool)
                        ptr = 0
                        remain = len(pool)
                    take_now = min(need - len(chosen), remain)
                    chosen.extend(pool[ptr: ptr + take_now])
                    ptr += take_now
                return chosen, ptr

            pos_chosen, pos_ptr = take_from_pool(pos_pool, pos_ptr, pos_take)
            neg_chosen, neg_ptr = take_from_pool(neg_pool, neg_ptr, neg_take)

            batch.extend(pos_chosen)
            batch.extend(neg_chosen)

            if len(batch) == self.batch_size:
                rng.shuffle(batch)
                yield batch
            elif len(batch) > 0 and not self.drop_last:
                rng.shuffle(batch)
                yield batch