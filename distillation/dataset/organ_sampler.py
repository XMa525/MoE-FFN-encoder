import random
import torch
from torch.utils.data import Sampler
from collections import defaultdict
from tqdm import tqdm
import os
import pickle


class OrganBalancedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size, cache_path="organ_indices_t015_aligned.pkl"):

        self.batch_size = batch_size
        epoch_batches=1000
        # -----------------------------
        # Handle Subset
        # -----------------------------
        if isinstance(dataset, torch.utils.data.Subset):
            self.dataset = dataset.dataset
            self.subset_indices = dataset.indices
        else:
            self.dataset = dataset
            self.subset_indices = list(range(len(dataset)))

        subset_set = set(self.subset_indices)

        # -----------------------------
        # Load or build organ index
        # -----------------------------
        if os.path.exists(cache_path):

            print(f"\n⚡ Loading organ cache: {cache_path}")

            with open(cache_path, "rb") as f:
                organ_indices_global = pickle.load(f)

        else:

            print("\n⚡ Building organ index ...")

            organ_indices_global = defaultdict(list)

            # if hasattr(self.dataset, "samples"):
            #     path_list = [s[0] for s in self.dataset.samples]

            # elif hasattr(self.dataset, "image_paths"):
            #     path_list = self.dataset.image_paths

            # elif hasattr(self.dataset, "paths"):
            #     path_list = self.dataset.paths

            # else:
            #     raise RuntimeError(
            #         "Dataset must expose samples / image_paths / paths."
            #     )

            # for idx, path in tqdm(
            #     enumerate(path_list),
            #     total=len(path_list),
            #     desc="Parsing organ from path",
            # ):

            #     try:
            #         organ = os.path.normpath(path).split(os.sep)[-2]
            #         organ_indices_global[organ].append(idx)
            #     except Exception:
            #         continue
            if hasattr(self.dataset, "samples"):
                # samples: [(path, organ_id), ...]
                for idx, sample in tqdm(
                    enumerate(self.dataset.samples),
                    total=len(self.dataset.samples),
                    desc="Building organ index",
                ):
                    try:
                        _, organ_id = sample
                        organ_indices_global[int(organ_id)].append(idx)
                    except Exception:
                        continue
            else:
                raise RuntimeError("Dataset must expose samples as (path, organ_id).")

            print("\n⚡ Saving organ cache...")

            with open(cache_path, "wb") as f:
                pickle.dump(dict(organ_indices_global), f)

            print("✅ Organ cache saved.")

        # -----------------------------
        # Global index → subset index
        # -----------------------------
        print("\n⚡ Mapping global indices → subset indices...")

        subset_map = {g: i for i, g in enumerate(self.subset_indices)}

        self.organ_indices = {}

        total_items = sum(len(v) for v in organ_indices_global.values())

        pbar = tqdm(total=total_items, desc="Filtering subset")

        for organ, idxs in organ_indices_global.items():

            filtered = []

            for gidx in idxs:

                if gidx in subset_map:
                    filtered.append(subset_map[gidx])  # ⭐关键修复

                pbar.update(1)

            if len(filtered) > 0:
                self.organ_indices[organ] = filtered

        pbar.close()

        # -----------------------------
        # Organ statistics
        # -----------------------------
        self.organs = list(self.organ_indices.keys())
        self.num_organs = len(self.organs)

        if self.num_organs == 0:
            raise RuntimeError("No organs found in dataset.")

        if batch_size % self.num_organs != 0:
            raise ValueError(
                f"batch_size {batch_size} must be divisible by organs {self.num_organs}"
            )

        self.per_organ = batch_size // self.num_organs

        # epoch size
        self.epoch_batches =min(len(self.subset_indices) // batch_size, epoch_batches) 

        self.pools = list(self.organ_indices.values())


    # -----------------------------
    # Sampling
    # -----------------------------
    def __iter__(self):

        for _ in range(self.epoch_batches):

            batch = []

            for pool in self.pools:

                # pool太小允许重复
                if len(pool) < self.per_organ:
                    samples = random.choices(pool, k=self.per_organ)
                else:
                    samples = random.sample(pool, self.per_organ)

                batch.extend(samples)

            random.shuffle(batch)

            yield batch

    # -----------------------------
    # Epoch length
    # -----------------------------
    def __len__(self):

        return self.epoch_batches