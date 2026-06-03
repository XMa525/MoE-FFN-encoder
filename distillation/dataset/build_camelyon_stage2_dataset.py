import os
import random
from typing import List, Tuple, Optional, Dict

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import openslide


class CamelyonWSIBagDataset(Dataset):
    """
    用于 stage2 bag-level supervision 的 CAMELYON WSI dataset

    输入:
        csv_path:
            包含列:
                - slide_id
                - patient   (原始 WSI 文件名, 如 patient_156_node_1.tif)
                - label     (0/1)
        raw_dir:
            原始 WSI 所在目录
        h5_dir:
            CLAM patch h5 所在目录，文件名默认是 {slide_id}.h5
        patch_size:
            从 level-0 坐标读取的 patch size
        read_level:
            openslide 读取层级，通常 0
        resize_to:
            读完 patch 后 resize 到 (resize_to, resize_to)
        max_patches:
            每张 slide 最多抽多少 patch；None 表示全用
        sample_mode:
            "random" 或 "first"
        seed:
            随机种子
        return_pil:
            True -> 返回 List[PIL.Image]
            False -> 返回 List[Tensor] 之前你可自己加 transform，这里默认还是 PIL 更通用
        transform:
            可选，对每个 patch 做 transform
    """

    def __init__(
        self,
        csv_path: str,
        raw_dir: str,
        h5_dir: str,
        patch_size: int = 256,
        read_level: int = 0,
        resize_to: int = 224,
        max_patches: Optional[int] = 512,
        sample_mode: str = "random",
        seed: int = 42,
        return_pil: bool = True,
        transform=None,
        check_files: bool = True,
    ):
        super().__init__()

        self.csv_path = csv_path
        self.raw_dir = raw_dir
        self.h5_dir = h5_dir

        self.patch_size = patch_size
        self.read_level = read_level
        self.resize_to = resize_to
        self.max_patches = max_patches
        self.sample_mode = sample_mode
        self.seed = seed
        self.return_pil = return_pil
        self.transform = transform

        assert self.sample_mode in ["random", "first"]

        df = pd.read_csv(csv_path)
        required_cols = {"slide_id", "patient", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"csv missing required columns: {missing}")

        self.samples = []
        for _, row in df.iterrows():
            slide_id = str(row["slide_id"])
            patient = str(row["patient"])
            label = int(row["label"])

            wsi_path = os.path.join(raw_dir, patient)
            h5_path = os.path.join(h5_dir, f"{slide_id}.h5")

            if check_files:
                if not os.path.exists(wsi_path):
                    print(f"[WARN] missing WSI, skip: {wsi_path}")
                    continue
                if not os.path.exists(h5_path):
                    print(f"[WARN] missing h5, skip: {h5_path}")
                    continue

            self.samples.append(
                {
                    "slide_id": slide_id,
                    "patient": patient,
                    "label": label,
                    "wsi_path": wsi_path,
                    "h5_path": h5_path,
                }
            )

        if len(self.samples) == 0:
            raise ValueError("No valid samples found.")

        print(f"[CamelyonWSIBagDataset] num_samples = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def read_coords_from_h5(h5_path: str) -> torch.Tensor:
        with h5py.File(h5_path, "r") as f:
            if "coords" not in f:
                raise KeyError(f"'coords' not found in {h5_path}")
            coords = f["coords"][:]
        return torch.from_numpy(coords).long()

    @staticmethod
    def read_patch_from_wsi(
        slide: openslide.OpenSlide,
        coord_xy: Tuple[int, int],
        patch_size: int = 256,
        read_level: int = 0,
    ) -> Image.Image:
        x, y = int(coord_xy[0]), int(coord_xy[1])
        patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
        return patch

    def _sample_coords(self, coords: torch.Tensor, idx: int) -> torch.Tensor:
        if self.max_patches is None or coords.shape[0] <= self.max_patches:
            return coords

        if self.sample_mode == "first":
            return coords[: self.max_patches]

        # random
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)
        perm = torch.randperm(coords.shape[0], generator=rng)[: self.max_patches]
        return coords[perm]

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        slide_id = sample["slide_id"]
        label = sample["label"]
        wsi_path = sample["wsi_path"]
        h5_path = sample["h5_path"]

        coords = self.read_coords_from_h5(h5_path)
        coords = self._sample_coords(coords, idx)

        slide = openslide.OpenSlide(wsi_path)

        images: List = []
        for xy in coords.tolist():
            img = self.read_patch_from_wsi(
                slide=slide,
                coord_xy=xy,
                patch_size=self.patch_size,
                read_level=self.read_level,
            )

            if self.resize_to is not None:
                img = img.resize((self.resize_to, self.resize_to), resample=Image.BICUBIC)

            if self.transform is not None:
                img = self.transform(img)

            images.append(img)

        slide.close()

        return {
            "slide_id": slide_id,
            "label": label,
            "images": images,                # List[PIL] or transformed list
            "coords": coords,                # [N, 2]
            "num_patches": int(coords.shape[0]),
            "wsi_path": wsi_path,
            "h5_path": h5_path,
        }


def split_stage2_wsi_csv(
    input_csv: str,
    train_csv: str,
    val_csv: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    把 stage2_wsi_sample_100.csv 切成 train / val，按 label 分层
    """
    df = pd.read_csv(input_csv)
    if "label" not in df.columns:
        raise ValueError("input csv must contain label column")

    train_parts = []
    val_parts = []

    for label, g in df.groupby("label"):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(round(len(g) * val_ratio)))
        val_parts.append(g.iloc[:n_val])
        train_parts.append(g.iloc[n_val:])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"[Saved] train -> {train_csv}, n={len(train_df)}")
    print(f"[Saved] val   -> {val_csv}, n={len(val_df)}")
    print("[Train label counts]")
    print(train_df["label"].value_counts().sort_index())
    print("[Val label counts]")
    print(val_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    # ===== 1) 先切 train / val =====
    split_stage2_wsi_csv(
        input_csv="../data/CAMELYON17/stage2_wsi_sample_100.csv",
        train_csv="../data/CAMELYON17/stage2_wsi_train.csv",
        val_csv="../data/CAMELYON17/stage2_wsi_val.csv",
        val_ratio=0.2,
        seed=42,
    )

    # ===== 2) 再做 dataset smoke test =====
    dataset = CamelyonWSIBagDataset(
        csv_path="../data/CAMELYON17/stage2_wsi_train.csv",
        raw_dir="../data/CAMELYON17/images",
        h5_dir="../data/CAMELYON17/patches/patches",
        patch_size=256,
        read_level=0,
        resize_to=224,
        max_patches=128,
        sample_mode="random",
        seed=42,
        return_pil=True,
        transform=None,
        check_files=True,
    )

    item = dataset[0]
    print("\n===== Smoke Test =====")
    print("slide_id    :", item["slide_id"])
    print("label       :", item["label"])
    print("num_patches :", item["num_patches"])
    print("first image :", type(item["images"][0]), item["images"][0].size if hasattr(item["images"][0], "size") else None)
    print("coords shape:", tuple(item["coords"].shape))