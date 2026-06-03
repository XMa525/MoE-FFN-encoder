import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile, ImageOps
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SpiderPatchDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        cluster_cache_path=None,
        num_patch_tokens=256,
        missing_cluster_mode="error",   # "error" | "zeros"
        enable_tissue_filter=False,
        white_threshold=0.85,
        tissue_threshold=0.15,
        samples_cache_path=None,
        rebuild_samples_cache=False,
    ):
        self.samples = []
        self.transform = transform
        self.num_patch_tokens = num_patch_tokens
        self.missing_cluster_mode = missing_cluster_mode
        self.enable_tissue_filter = enable_tissue_filter
        self.white_threshold = white_threshold
        self.tissue_threshold = tissue_threshold
        self.samples_cache_path = samples_cache_path
        self.rebuild_samples_cache = rebuild_samples_cache

        organs = sorted(os.listdir(root))
        self.organ_to_id = {o: i for i, o in enumerate(organs)}
        loaded_from_cache = False
        kept_cnt = 0
        filtered_cnt = 0
        if (
            self.samples_cache_path is not None
            and os.path.exists(self.samples_cache_path)
            and not self.rebuild_samples_cache
        ):
            print(f"Loading filtered samples cache: {self.samples_cache_path}")
            with open(self.samples_cache_path, "rb") as f:
                cache_data = pickle.load(f)

            self.samples = cache_data["samples"]
            loaded_from_cache = True
            print(f"Loaded {len(self.samples)} cached samples")

        else:
            

            for organ in organs:
                organ_root = os.path.join(root, organ)
                image_dir = os.path.join(organ_root, organ, "images")

                if not os.path.exists(image_dir):
                    continue

                file_list = sorted(os.listdir(image_dir))

                for fname in tqdm(
                    file_list,
                    desc=f"Scanning {organ}",
                    leave=True,
                ):
                    if not fname.endswith((".png", ".jpg", ".jpeg")):
                        continue

                    path = os.path.join(image_dir, fname)
                    if self.enable_tissue_filter:
                        try:
                            img = Image.open(path).convert("RGB")
                            tissue_ratio = self._simple_tissue_ratio(img)
                        except Exception as e:
                            print(f"⚠️ Warning: Failed to read image for tissue filter {path}: {e}")
                            filtered_cnt += 1
                            continue

                        if tissue_ratio < self.tissue_threshold:
                            filtered_cnt += 1
                            continue
                    self.samples.append((path, self.organ_to_id[organ]))
                    kept_cnt += 1

        print(f"Loaded {len(self.samples)} patches")
        if self.enable_tissue_filter:
            print(
                f"Tissue filter enabled: white_threshold={self.white_threshold}, "
                f"tissue_threshold={self.tissue_threshold}"
            )
            if not loaded_from_cache:
                print(f"Kept patches: {kept_cnt}, Filtered patches: {filtered_cnt}")
        if self.samples_cache_path is not None and not loaded_from_cache:
            os.makedirs(os.path.dirname(self.samples_cache_path), exist_ok=True)
            with open(self.samples_cache_path, "wb") as f:
                pickle.dump(
                    {
                        "samples": self.samples,
                        "white_threshold": self.white_threshold,
                        "tissue_threshold": self.tissue_threshold,
                        "enable_tissue_filter": self.enable_tissue_filter,
                    },
                    f,
                )
            print(f"Saved filtered samples cache to: {self.samples_cache_path}")
    
        # ---------------- load cluster cache ----------------
        self.path_to_cluster_ids = None
        if cluster_cache_path is not None:
            self.path_to_cluster_ids = self._load_cluster_cache(cluster_cache_path)
            print(f"Loaded cluster cache: {len(self.path_to_cluster_ids)} entries")
            before_align = len(self.samples)
            self.samples = [
                (path, organ)
                for path, organ in self.samples
                if self._canonicalize_path(path) in self.path_to_cluster_ids
            ]
            after_align = len(self.samples)

            print(f"Aligned samples to cluster cache: {before_align} -> {after_align}")
            print(f"Dropped {before_align - after_align} samples not found in cluster cache")
            
    def _simple_tissue_ratio(self, img):
        gray = ImageOps.grayscale(img)
        arr = np.asarray(gray).astype(np.float32) / 255.0
        tissue_mask = arr < self.white_threshold
        return tissue_mask.mean()

    def _canonicalize_path(self, path):
        path = os.path.normpath(path).replace("\\", "/")
        marker = "data/raw/"
        if marker in path:
            return path.split(marker, 1)[1]
        return path
    def _load_cluster_cache(self, cluster_cache_path):
        if cluster_cache_path.endswith(".pkl"):
            with open(cluster_cache_path, "rb") as f:
                cache = pickle.load(f)
        elif cluster_cache_path.endswith(".pt"):
            cache = torch.load(cluster_cache_path, map_location="cpu")
        elif cluster_cache_path.endswith(".npy"):
            cache = np.load(cluster_cache_path, allow_pickle=True).item()
        else:
            raise ValueError(f"Unsupported cluster cache format: {cluster_cache_path}")

        if not isinstance(cache, dict):
            raise ValueError(f"Cluster cache must be a dict[path] -> cluster_ids, got {type(cache)}")

        normalized_cache = {}
        for k, v in cache.items():
            nk = self._canonicalize_path(k)
            normalized_cache[nk] = v

        return normalized_cache

    def __len__(self):
        return len(self.samples)
    
    def _get_cluster_ids(self, path):
        path = self._canonicalize_path(path)
        # 没提供 cache：返回占位
        if self.path_to_cluster_ids is None:
            return torch.zeros(self.num_patch_tokens, dtype=torch.long)

        if path not in self.path_to_cluster_ids:
            if self.missing_cluster_mode == "zeros":
                return torch.zeros(self.num_patch_tokens, dtype=torch.long)
            raise KeyError(f"Path not found in cluster cache: {path}")

        cluster_ids = self.path_to_cluster_ids[path]

        if isinstance(cluster_ids, torch.Tensor):
            cluster_ids = cluster_ids.cpu().numpy()

        cluster_ids = np.asarray(cluster_ids)

        if cluster_ids.ndim != 1 or len(cluster_ids) != self.num_patch_tokens:
            raise ValueError(
                f"cluster_ids for path {path} must have shape [{self.num_patch_tokens}], "
                f"got {cluster_ids.shape}"
            )
        
        return torch.as_tensor(cluster_ids, dtype=torch.long)

    def __getitem__(self, idx):
        path, organ = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load image {path}: {e}")
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        offline_cluster_ids = self._get_cluster_ids(path)

        return img, organ, offline_cluster_ids