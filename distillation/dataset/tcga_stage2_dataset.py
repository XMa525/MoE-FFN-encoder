from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import openslide
import pandas as pd
import torch
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


class TCGARolePatchDataset(Dataset):
    """
    Stage2 TCGA patch dataset based on a single CSV pool.

    最小改动修正版：
    - 主样本仍然是单个 patch
    - WSI bag 不再在 __getitem__ 里实际读取图像
    - 只返回 bag 所需的 slide_id / label 元信息
    - collate_fn 中仅为 batch[0] 采样一次 WSI bag，避免整 batch 重复采样
    """

    REQUIRED_COLS = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]

    def __init__(
        self,
        csv_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        transform=None,
        indices: Optional[Sequence[int]] = None,
        project_to_id: Optional[dict] = None,
        filter_prefilter_white: bool = False,
        verbose: bool = True,

        # ============== WSI bag sampling 配置 ==============
        use_wsi_bag_sampling: bool = False,
        wsi_bag_size: int = 64,
        wsi_min_bag_size: int = 8,
        slide_label_col: str = "slide_label",
        random_seed: int = 42,

        # ============== spatial neighbor sampling 配置 ==============
        use_spatial_neighbor_sampling: bool = False,
        spatial_neighbor_csv: Optional[str] = None,
        spatial_neighbor_max_k: int = 8,
    ):
        super().__init__()

        # =====================================================
        # 兼容两种输入方式：
        # 1) 旧方式：csv_path
        # 2) 新方式：dataframe
        # =====================================================
        if dataframe is not None:
            df = dataframe
            data_source_desc = "<dataframe>"
            from_dataframe = True
        else:
            if csv_path is None:
                raise ValueError("Either csv_path or dataframe must be provided")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            df = pd.read_csv(csv_path)
            data_source_desc = csv_path
            from_dataframe = False

        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset source: {missing}")

        # 只在从 CSV 读入时做 canonicalize；从 dataframe 传入默认外部已处理好
        if (not from_dataframe) and ("svs_path" in df.columns):
            df = df.copy()
            df["svs_path"] = df["svs_path"].map(canonicalize_path)


        if filter_prefilter_white and "prefilter_white" in df.columns:
            df = df[df["prefilter_white"].fillna(0).astype(int) == 0].copy()

        # 先按 indices 切，再 reset_index
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            df = df.iloc[indices]

        df = df.reset_index(drop=True)

        self.df = df
        self.transform = transform
        self.rng = np.random.default_rng(random_seed)

        if "project" not in self.df.columns:
            self.df["project"] = "TCGA"

        if "slide_id" not in self.df.columns:
            self.df["slide_id"] = self.df["svs_path"].astype(str)

        if project_to_id is None:
            uniq_projects = sorted(self.df["project"].astype(str).unique().tolist())
            self.project_to_id = {p: i for i, p in enumerate(uniq_projects)}
        else:
            self.project_to_id = dict(project_to_id)

        self.id_to_project = {v: k for k, v in self.project_to_id.items()}

        # ==================== WSI bag setup ====================
        self.use_wsi_bag_sampling = bool(use_wsi_bag_sampling)
        self.wsi_bag_size = int(wsi_bag_size)
        self.wsi_min_bag_size = int(wsi_min_bag_size)
        self.slide_label_col = slide_label_col

        self.slide_to_indices = None

        if self.use_wsi_bag_sampling:
            if self.slide_label_col not in self.df.columns:
                fallback_cols = ["label", "bag_label", "slide_target"]
                found = None
                for c in fallback_cols:
                    if c in self.df.columns:
                        found = c
                        break
                if found is None:
                    raise ValueError(
                        f"use_wsi_bag_sampling=True, but no slide label column found. "
                        f"Expected '{self.slide_label_col}' or one of {fallback_cols}"
                    )
                self.slide_label_col = found

            nunique_per_slide = self.df.groupby("slide_id")[self.slide_label_col].nunique(dropna=False)
            bad = nunique_per_slide[nunique_per_slide > 1]
            if len(bad) > 0:
                raise ValueError(
                    f"Inconsistent slide labels detected in column '{self.slide_label_col}'. "
                    f"Example bad slides: {bad.index[:10].tolist()}"
                )

            self.slide_to_indices = {}
            grouped = self.df.groupby("slide_id").indices
            for sid, idxs in grouped.items():
                idxs = np.asarray(idxs, dtype=np.int64)
                if len(idxs) >= self.wsi_min_bag_size:
                    self.slide_to_indices[str(sid)] = idxs

            valid_slide_ids = set(self.slide_to_indices.keys())
            self.df = self.df[self.df["slide_id"].astype(str).isin(valid_slide_ids)].reset_index(drop=True)

            self.slide_to_indices = {}
            grouped = self.df.groupby("slide_id").indices
            for sid, idxs in grouped.items():
                idxs = np.asarray(idxs, dtype=np.int64)
                if len(idxs) >= self.wsi_min_bag_size:
                    self.slide_to_indices[str(sid)] = idxs

        # ==================== spatial neighbor setup ====================
        self.use_spatial_neighbor_sampling = bool(use_spatial_neighbor_sampling)
        self.spatial_neighbor_csv = spatial_neighbor_csv
        self.spatial_neighbor_max_k = int(spatial_neighbor_max_k)

        self.slide_to_df_indices = {
            str(sid): np.asarray(idxs, dtype=np.int64)
            for sid, idxs in self.df.groupby("slide_id").indices.items()
        }

        self.spatial_neighbor_bank = {}
        if self.use_spatial_neighbor_sampling:
            if self.spatial_neighbor_csv is None or not os.path.exists(self.spatial_neighbor_csv):
                raise FileNotFoundError(
                    f"spatial_neighbor_csv not found: {self.spatial_neighbor_csv}"
                )
            self._load_spatial_neighbor_bank()

        if verbose:
            print(f"[TCGARolePatchDataset] Loaded {len(self.df)} rows from {data_source_desc}")
            print("[TCGARolePatchDataset] Project counts:")
            print(self.df["project"].value_counts())

            if self.use_wsi_bag_sampling:
                print("[TCGARolePatchDataset] WSI bag sampling enabled")
                print(f"[TCGARolePatchDataset] slide_label_col = {self.slide_label_col}")
                print(f"[TCGARolePatchDataset] eligible slides = {len(self.slide_to_indices)}")
                print(f"[TCGARolePatchDataset] wsi_bag_size = {self.wsi_bag_size}")
                print(f"[TCGARolePatchDataset] wsi_min_bag_size = {self.wsi_min_bag_size}")
        if verbose and self.use_spatial_neighbor_sampling:
            self.summarize_neighbor_coverage(max_check=5000)
    def __len__(self) -> int:
        return len(self.df)

    # ==================== 统一读 patch ====================
    def _read_patch(self, svs_path: str, x: int, y: int, patch_level: int, patch_size: int):
        slide = openslide.OpenSlide(svs_path)
        try:
            image = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()

        if self.transform is not None:
            image = self.transform(image)

        return image

    # ==================== 为 collate_fn 提供：采样一个 WSI bag ====================
    def sample_wsi_bag_by_slide_id(self, slide_id: str):
        """
        供 collate_fn 调用。
        只为一个 slide 实际采样和读取 bag 图像。
        """
        if self.slide_to_indices is None:
            return None, None

        slide_id = str(slide_id)
        if slide_id not in self.slide_to_indices:
            return None, None

        idx_pool = self.slide_to_indices[slide_id]
        num_available = len(idx_pool)
        k = min(self.wsi_bag_size, num_available)

        if num_available <= k:
            sampled_idx = idx_pool
        else:
            sampled_idx = self.rng.choice(idx_pool, size=k, replace=False)

        bag_rows = self.df.iloc[sampled_idx]

        wsi_images = []
        for _, row in bag_rows.iterrows():
            img = self._read_patch(
                svs_path=row["svs_path"],
                x=int(row["coord_x"]),
                y=int(row["coord_y"]),
                patch_level=int(row["patch_level"]),
                patch_size=int(row["patch_size"]),
            )
            wsi_images.append(img)

        if len(wsi_images) == 0:
            return None, None

        wsi_images = torch.stack(wsi_images, dim=0)
        slide_label = int(bag_rows.iloc[0][self.slide_label_col])
        return wsi_images, slide_label

    def _load_spatial_neighbor_bank(self):
        df_nb = pd.read_csv(self.spatial_neighbor_csv)

        required = ["slide_id", "coord_idx", "subclass_id", "neighbor_coord_indices"]
        missing = [c for c in required if c not in df_nb.columns]
        if missing:
            raise ValueError(
                f"spatial_neighbor_csv missing required columns: {missing}"
            )

        bank = {}
        kept = 0

        for _, row in df_nb.iterrows():
            slide_id = str(row["slide_id"])
            coord_idx = int(row["coord_idx"])
            subclass_id = int(row["subclass_id"])

            raw_neighbors = str(row["neighbor_coord_indices"]).strip()
            if raw_neighbors == "" or raw_neighbors.lower() == "nan":
                neighbor_coord_indices = []
            else:
                neighbor_coord_indices = [
                    int(x) for x in raw_neighbors.split(";") if str(x).strip() != ""
                ]

            if len(neighbor_coord_indices) == 0:
                continue

            bank[(slide_id, coord_idx)] = {
                "subclass_id": subclass_id,
                "neighbor_coord_indices": neighbor_coord_indices,
            }
            kept += 1

        self.spatial_neighbor_bank = bank
        print(
            f"[TCGARolePatchDataset] spatial neighbor bank loaded: {kept} centers "
            f"from {self.spatial_neighbor_csv}"
        )

    def _sample_spatial_neighbors(self, slide_id: str, center_coord_idx: int):
        if not self.use_spatial_neighbor_sampling:
            return None

        key = (str(slide_id), int(center_coord_idx))
        if key not in self.spatial_neighbor_bank:
            return None

        bank_item = self.spatial_neighbor_bank[key]
        nb_coord_indices = list(bank_item["neighbor_coord_indices"])

        slide_idx = self.slide_to_df_indices.get(str(slide_id), None)
        if slide_idx is None:
            return None

        sdf = self.df.iloc[slide_idx]
        sdf = sdf[sdf["coord_idx"].isin(nb_coord_indices)].copy()
        if len(sdf) == 0:
            return None

        if len(sdf) > self.spatial_neighbor_max_k:
            sdf = sdf.sample(
                n=self.spatial_neighbor_max_k,
                random_state=int(center_coord_idx) % 100000 + 17,
            ).reset_index(drop=True)
        else:
            sdf = sdf.reset_index(drop=True)

        nb_images = []
        for _, row in sdf.iterrows():
            img = self._read_patch(
                svs_path=row["svs_path"],
                x=int(row["coord_x"]),
                y=int(row["coord_y"]),
                patch_level=int(row["patch_level"]),
                patch_size=int(row["patch_size"]),
            )
            nb_images.append(img)

        if len(nb_images) == 0:
            return None

        return torch.stack(nb_images, dim=0)

    def summarize_neighbor_coverage(self, max_check: Optional[int] = None):
        if not self.use_spatial_neighbor_sampling:
            print("[TCGARolePatchDataset] spatial neighbor sampling disabled")
            return

        total = len(self.df) if max_check is None else min(len(self.df), int(max_check))
        found = 0
        num_neighbors_list = []

        for i in range(total):
            row = self.df.iloc[i]
            slide_id = str(row["slide_id"])
            coord_idx = int(row["coord_idx"]) if "coord_idx" in row and pd.notna(row["coord_idx"]) else -1

            if coord_idx < 0:
                continue

            key = (slide_id, coord_idx)
            if key not in self.spatial_neighbor_bank:
                continue

            nb_coord_indices = self.spatial_neighbor_bank[key]["neighbor_coord_indices"]

            slide_idx = self.slide_to_df_indices.get(slide_id, None)
            if slide_idx is None:
                continue

            sdf = self.df.iloc[slide_idx]
            sdf = sdf[sdf["coord_idx"].isin(nb_coord_indices)]

            k = len(sdf)
            if k > 0:
                found += 1
                num_neighbors_list.append(k)

        checked = total
        found_ratio = found / max(checked, 1)
        mean_neighbors = float(np.mean(num_neighbors_list)) if len(num_neighbors_list) > 0 else 0.0

        print(
            f"[TCGARolePatchDataset] neighbor coverage: "
            f"checked={checked}, found={found}, found_ratio={found_ratio:.4f}, "
            f"mean_neighbors={mean_neighbors:.2f}"
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        svs_path = row["svs_path"]
        x = int(row["coord_x"])
        y = int(row["coord_y"])
        patch_level = int(row["patch_level"])
        patch_size = int(row["patch_size"])

        image = self._read_patch(
            svs_path=svs_path,
            x=x,
            y=y,
            patch_level=patch_level,
            patch_size=patch_size,
        )

        project = str(row["project"])
        project_id = int(self.project_to_id[project])
        slide_id = str(row["slide_id"]) if "slide_id" in row else str(svs_path)

        item = {
            "image": image,
            "project": project,
            "project_id": project_id,
            "slide_id": slide_id,
            "svs_path": svs_path,
            "h5_path": str(row["h5_path"]) if "h5_path" in row else "",
            "coord_x": x,
            "coord_y": y,
            "coord_idx": int(row["coord_idx"]) if "coord_idx" in row and pd.notna(row["coord_idx"]) else -1,
            "patch_level": patch_level,
            "patch_size": patch_size,
            "pred_label": str(row["pred_label"]) if "pred_label" in row else "",
            "pred_confidence": float(row["pred_confidence"]) if "pred_confidence" in row and pd.notna(row["pred_confidence"]) else 0.0,
            "entropy": float(row["entropy"]) if "entropy" in row and pd.notna(row["entropy"]) else 0.0,
            "margin_top1_top2": float(row["margin_top1_top2"]) if "margin_top1_top2" in row and pd.notna(row["margin_top1_top2"]) else 0.0,
            "prefilter_white": int(row["prefilter_white"]) if "prefilter_white" in row and pd.notna(row["prefilter_white"]) else 0,
        }

        for col in self.df.columns:
            if col.startswith("score_"):
                val = row[col]
                item[col] = float(val) if pd.notna(val) else 0.0

        # ==================== slide label ====================
        if self.slide_label_col in row.index and pd.notna(row[self.slide_label_col]):
            item["slide_label"] = int(row[self.slide_label_col])
        else:
            item["slide_label"] = None

        # ==================== 关键修复：不在 __getitem__ 里实际读取 WSI bag ====================
        if self.use_wsi_bag_sampling:
            item["wsi_bag_slide_id"] = slide_id
            item["wsi_bag_enabled"] = True
        else:
            item["wsi_bag_slide_id"] = None
            item["wsi_bag_enabled"] = False

        # 保持字段兼容，避免下游代码因 key 缺失报错
        item["wsi_images"] = None

        # ==================== spatial neighbors ====================
        if self.use_spatial_neighbor_sampling:
            neighbor_images = self._sample_spatial_neighbors(
                slide_id=slide_id,
                center_coord_idx=item["coord_idx"],
            )
            item["neighbor_images"] = neighbor_images
        else:
            item["neighbor_images"] = None

        return item


def tcga_stage2_collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)

    out = {
        "image": images,
        "project": [x["project"] for x in batch],
        "project_id": torch.tensor([x["project_id"] for x in batch], dtype=torch.long),
        "slide_id": [x["slide_id"] for x in batch],
        "svs_path": [x["svs_path"] for x in batch],
        "h5_path": [x["h5_path"] for x in batch],
        "coord_x": torch.tensor([x["coord_x"] for x in batch], dtype=torch.long),
        "coord_y": torch.tensor([x["coord_y"] for x in batch], dtype=torch.long),
        "coord_idx": torch.tensor([x["coord_idx"] for x in batch], dtype=torch.long),
        "patch_level": torch.tensor([x["patch_level"] for x in batch], dtype=torch.long),
        "patch_size": torch.tensor([x["patch_size"] for x in batch], dtype=torch.long),
        "pred_label": [x["pred_label"] for x in batch],
        "pred_confidence": torch.tensor([x["pred_confidence"] for x in batch], dtype=torch.float32),
        "entropy": torch.tensor([x["entropy"] for x in batch], dtype=torch.float32),
        "margin_top1_top2": torch.tensor([x["margin_top1_top2"] for x in batch], dtype=torch.float32),
        "prefilter_white": torch.tensor([x["prefilter_white"] for x in batch], dtype=torch.long),
    }

    score_keys = [k for k in batch[0].keys() if k.startswith("score_")]
    for k in score_keys:
        out[k] = torch.tensor([x[k] for x in batch], dtype=torch.float32)

    # ==================== WSI bag fields ====================
    # 最小改动修复思路：
    # 只对 batch[0] 对应的 slide 实际采一次 bag
    dataset = getattr(batch[0]["image"], "_dataset_ref", None)

    # 上面这种方式拿不到 dataset，所以改成从 batch 元信息推断是否启用
    if batch[0].get("wsi_bag_enabled", False):
        # 注意：这里需要通过 batch 中任一元素反查 dataset 不方便，
        # 因此建议 DataLoader 外部使用一个闭包 collate_fn。
        # 为了维持“最小改动可替换”，这里直接报清晰错误提醒你用下方包装器。
        raise RuntimeError(
            "tcga_stage2_collate_fn now requires binding the dataset instance. "
            "Please use `build_tcga_stage2_collate_fn(dataset)` instead of passing "
            "`tcga_stage2_collate_fn` directly."
        )
    else:
        out["wsi_images"] = None
        out["wsi_slide_label"] = None
        out["wsi_slide_id"] = None
        out["wsi_svs_path"] = None

    # ==================== per-sample slide labels ====================
    out["slide_label_batch"] = torch.tensor(
        [
            -1 if x.get("slide_label", None) is None else int(x["slide_label"])
            for x in batch
        ],
        dtype=torch.long,
    )

    # ==================== spatial neighbor fields ====================
    out["neighbor_images_list"] = [
        x.get("neighbor_images", None) for x in batch
    ]

    return out


def build_tcga_stage2_collate_fn(dataset: TCGARolePatchDataset):
    """
    新的推荐 collate_fn 构造器：
    绑定 dataset 实例，这样 collate 时才能只采一次 WSI bag。
    """
    def _collate_fn(batch):
        images = torch.stack([x["image"] for x in batch], dim=0)

        out = {
            "image": images,
            "project": [x["project"] for x in batch],
            "project_id": torch.tensor([x["project_id"] for x in batch], dtype=torch.long),
            "slide_id": [x["slide_id"] for x in batch],
            "svs_path": [x["svs_path"] for x in batch],
            "h5_path": [x["h5_path"] for x in batch],
            "coord_x": torch.tensor([x["coord_x"] for x in batch], dtype=torch.long),
            "coord_y": torch.tensor([x["coord_y"] for x in batch], dtype=torch.long),
            "coord_idx": torch.tensor([x["coord_idx"] for x in batch], dtype=torch.long),
            "patch_level": torch.tensor([x["patch_level"] for x in batch], dtype=torch.long),
            "patch_size": torch.tensor([x["patch_size"] for x in batch], dtype=torch.long),
            "pred_label": [x["pred_label"] for x in batch],
            "pred_confidence": torch.tensor([x["pred_confidence"] for x in batch], dtype=torch.float32),
            "entropy": torch.tensor([x["entropy"] for x in batch], dtype=torch.float32),
            "margin_top1_top2": torch.tensor([x["margin_top1_top2"] for x in batch], dtype=torch.float32),
            "prefilter_white": torch.tensor([x["prefilter_white"] for x in batch], dtype=torch.long),
        }

        score_keys = [k for k in batch[0].keys() if k.startswith("score_")]
        for k in score_keys:
            out[k] = torch.tensor([x[k] for x in batch], dtype=torch.float32)

        # ==================== 关键修复：只采一次 WSI bag ====================
        if batch[0].get("wsi_bag_enabled", False):
            ref_slide_id = batch[0]["wsi_bag_slide_id"]
            wsi_images, bag_slide_label = dataset.sample_wsi_bag_by_slide_id(ref_slide_id)

            out["wsi_images"] = wsi_images
            out["wsi_slide_label"] = (
                torch.tensor(bag_slide_label, dtype=torch.float32)
                if bag_slide_label is not None else None
            )
            out["wsi_slide_id"] = ref_slide_id
            out["wsi_svs_path"] = batch[0]["svs_path"]
        else:
            out["wsi_images"] = None
            out["wsi_slide_label"] = None
            out["wsi_slide_id"] = None
            out["wsi_svs_path"] = None

        # ==================== per-sample slide labels ====================
        out["slide_label_batch"] = torch.tensor(
            [
                -1 if x.get("slide_label", None) is None else int(x["slide_label"])
                for x in batch
            ],
            dtype=torch.long,
        )

        # ==================== spatial neighbor fields ====================
        out["neighbor_images_list"] = [
            x.get("neighbor_images", None) for x in batch
        ]

        return out

    return _collate_fn