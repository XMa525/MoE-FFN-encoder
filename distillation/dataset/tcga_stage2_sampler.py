from __future__ import annotations

import math
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class ProjectBalancedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that tries to balance projects within each batch.

    Assumptions
    -----------
    The dataset should expose one of:
      - dataset.df["project"]
      - dataset.project_to_id / dataset.id_to_project with __getitem__
      - items containing "project"

    Recommended use
    ---------------
    Use with TCGARolePatchDataset, where dataset.df already contains "project".

    Behavior
    --------
    - Each batch draws roughly equal number of samples from each project.
    - When a project runs out, its pool is reshuffled and reused, so epoch length
      is determined by num_batches_per_epoch.
    - This is intentional: for highly imbalanced datasets, balanced sampling
      usually requires oversampling minority projects.

    Notes
    -----
    - This sampler returns batches of indices, so pass it as `batch_sampler=...`
      to DataLoader and do not also pass `batch_size` or `shuffle`.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_batches_per_epoch: int | None = None,
        drop_last: bool = True,
        seed: int = 42,
        cache_path: str | None = None,
        verbose: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.cache_path = cache_path
        self.verbose = verbose

        self.project_to_indices = self._build_or_load_project_indices()
        self.projects = sorted(self.project_to_indices.keys())

        if len(self.projects) == 0:
            raise RuntimeError("No projects found for ProjectBalancedBatchSampler")

        if num_batches_per_epoch is None:
            # Default: roughly one pass over the dataset size.
            if self.drop_last:
                self.num_batches_per_epoch = max(1, len(dataset) // self.batch_size)
            else:
                self.num_batches_per_epoch = max(1, math.ceil(len(dataset) / self.batch_size))
        else:
            self.num_batches_per_epoch = int(num_batches_per_epoch)

        if self.verbose:
            print("[ProjectBalancedBatchSampler] project counts:")
            for p in self.projects:
                print(f"  {p}: {len(self.project_to_indices[p])}")
            print(f"[ProjectBalancedBatchSampler] batch_size = {self.batch_size}")
            print(f"[ProjectBalancedBatchSampler] num_batches_per_epoch = {self.num_batches_per_epoch}")

    def _build_or_load_project_indices(self) -> Dict[str, List[int]]:
        if self.cache_path is not None and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                obj = pickle.load(f)
            if self.verbose:
                print(f"[ProjectBalancedBatchSampler] Loaded cache: {self.cache_path}")
            return obj

        project_to_indices: Dict[str, List[int]] = defaultdict(list)

        if hasattr(self.dataset, "df") and "project" in self.dataset.df.columns:
            for idx, project in enumerate(self.dataset.df["project"].astype(str).tolist()):
                project_to_indices[str(project)].append(idx)
        else:
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                if isinstance(item, dict) and "project" in item:
                    project = str(item["project"])
                else:
                    raise ValueError(
                        "Dataset must expose project information either via dataset.df['project'] "
                        "or via __getitem__ returning a dict with key 'project'."
                    )
                project_to_indices[project].append(idx)

        if self.cache_path is not None:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(dict(project_to_indices), f)
            if self.verbose:
                print(f"[ProjectBalancedBatchSampler] Saved cache: {self.cache_path}")

        return dict(project_to_indices)

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)

        project_pools: Dict[str, List[int]] = {}
        project_ptrs: Dict[str, int] = {}

        for p, indices in self.project_to_indices.items():
            pool = list(indices)
            rng.shuffle(pool)
            project_pools[p] = pool
            project_ptrs[p] = 0

        num_projects = len(self.projects)
        base_take = self.batch_size // num_projects
        remainder = self.batch_size % num_projects

        for _ in range(self.num_batches_per_epoch):
            batch: List[int] = []

            # Distribute remainder across the first few projects, rotating each batch
            rotated_projects = list(self.projects)
            rng.shuffle(rotated_projects)

            take_plan = {p: base_take for p in rotated_projects}
            for p in rotated_projects[:remainder]:
                take_plan[p] += 1

            for p in rotated_projects:
                need = take_plan[p]
                if need <= 0:
                    continue

                pool = project_pools[p]
                ptr = project_ptrs[p]

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

                project_ptrs[p] = ptr
                batch.extend(chosen)

            if len(batch) < self.batch_size and not self.drop_last:
                # pad from any projects
                while len(batch) < self.batch_size:
                    p = rng.choice(rotated_projects)
                    pool = project_pools[p]
                    ptr = project_ptrs[p]
                    if ptr >= len(pool):
                        rng.shuffle(pool)
                        ptr = 0
                    batch.append(pool[ptr])
                    project_ptrs[p] = ptr + 1

            if len(batch) == self.batch_size:
                rng.shuffle(batch)
                yield batch
            elif len(batch) > 0 and not self.drop_last:
                rng.shuffle(batch)
                yield batch