#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import openslide
from PIL import ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from tqdm import tqdm

from models.plugins.load_stage2_for_plugin import load_stage2_for_plugin

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
TCGA_BARCODE_RE = re.compile(
    r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]?(?:-[A-Z0-9]+)*)",
    re.IGNORECASE,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def extract_tcga_barcode(text: str) -> Optional[str]:
    if text is None:
        return None
    m = TCGA_BARCODE_RE.search(str(text))
    if m is None:
        return None
    return m.group(1)


def canonical_slide_id(slide_id: str) -> str:
    sid = extract_tcga_barcode(slide_id)
    return sid if sid is not None else str(slide_id)


def resolve_wsi_path(
    slide_id: str,
    project: str,
    tcga_root: Optional[str],
    source_path: Optional[str] = None,
) -> str:
    slide_id = canonical_slide_id(slide_id)

    if source_path is not None and str(source_path) != "" and str(source_path).lower() != "nan":
        source_path = normalize_path(source_path)

        if os.path.isfile(source_path):
            ext = os.path.splitext(source_path)[1].lower()
            if ext in VALID_WSI_EXTS:
                return source_path

        if os.path.isdir(source_path):
            for root, _, files in os.walk(source_path):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in VALID_WSI_EXTS:
                        continue
                    full_path = normalize_path(os.path.join(root, fn))
                    if slide_id in full_path:
                        return full_path

    if tcga_root is None:
        raise FileNotFoundError(
            f"Cannot resolve WSI for slide_id={slide_id}: source_path invalid and tcga_root is None"
        )

    project_dir = os.path.join(tcga_root, str(project))
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Project dir not found: {project_dir}")

    matched = []
    for root, _, files in os.walk(project_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in VALID_WSI_EXTS:
                continue
            full_path = normalize_path(os.path.join(root, fn))
            if slide_id in full_path:
                matched.append(full_path)

    if len(matched) == 0:
        raise FileNotFoundError(f"No WSI found for slide_id={slide_id} under {project_dir}")

    matched = sorted(matched, key=lambda x: (len(x), x))
    return matched[0]


class TCGAPluginPatchDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        split: str,
        tcga_root: str,
        image_size: int = 224,
        patch_level: int = 0,
        patch_size: int = 224,
        samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"split_csv not found: {split_csv}")

        df = pd.read_csv(split_csv)
        required_cols = ["slide_id", "label", "split", "project"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"split csv missing columns: {missing}")

        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split={split}")

        self.df = df
        self.split = split
        self.tcga_root = tcga_root
        self.patch_level = int(patch_level)
        self.patch_size = int(patch_size)
        self.samples_per_epoch = samples_per_epoch if samples_per_epoch is not None else len(df)
        self.base_seed = int(seed)
        self.transform = build_transform(image_size)

        self.slide_records = []
        for _, row in self.df.iterrows():
            slide_id = canonical_slide_id(row["slide_id"])
            label = int(row["label"])
            project = str(row["project"])
            source_path = row["source_path"] if "source_path" in row and pd.notna(row["source_path"]) else None

            wsi_path = resolve_wsi_path(
                slide_id=slide_id,
                project=project,
                tcga_root=self.tcga_root,
                source_path=source_path,
            )

            self.slide_records.append({
                "slide_id": slide_id,
                "label": label,
                "project": project,
                "wsi_path": normalize_path(wsi_path),
            })

        self.pos_indices = [i for i, x in enumerate(self.slide_records) if x["label"] == 1]
        self.neg_indices = [i for i, x in enumerate(self.slide_records) if x["label"] == 0]

        print(f"[{split}] num slides = {len(self.slide_records)}")
        print(f"[{split}] pos = {len(self.pos_indices)}, neg = {len(self.neg_indices)}")

    def __len__(self):
        return int(self.samples_per_epoch)

    def _sample_slide_record(self, idx: int):
        rng = random.Random(self.base_seed + idx)

        use_pos = rng.random() < 0.5
        if use_pos and len(self.pos_indices) > 0:
            slide_idx = rng.choice(self.pos_indices)
        elif (not use_pos) and len(self.neg_indices) > 0:
            slide_idx = rng.choice(self.neg_indices)
        else:
            slide_idx = rng.randrange(len(self.slide_records))

        return self.slide_records[slide_idx], rng

    def _random_patch_from_slide(self, slide: openslide.OpenSlide, rng: random.Random):
        level = self.patch_level
        patch_size = self.patch_size

        level_w, level_h = slide.level_dimensions[level]
        if level_w <= patch_size or level_h <= patch_size:
            x_level, y_level = 0, 0
        else:
            x_level = rng.randint(0, level_w - patch_size)
            y_level = rng.randint(0, level_h - patch_size)

        downsample = slide.level_downsamples[level]
        x0 = int(round(x_level * downsample))
        y0 = int(round(y_level * downsample))

        patch = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB")
        return patch, x0, y0

    def __getitem__(self, idx: int):
        rec, rng = self._sample_slide_record(idx)

        slide = openslide.OpenSlide(rec["wsi_path"])
        try:
            patch, x0, y0 = self._random_patch_from_slide(slide, rng)
        finally:
            slide.close()

        image = self.transform(patch)

        return {
            "image": image,
            "label": int(rec["label"]),
            "slide_id": rec["slide_id"],
            "project": rec["project"],
            "wsi_path": rec["wsi_path"],
            "coord_x": int(x0),
            "coord_y": int(y0),
        }


def plugin_patch_collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return {
        "images": images,
        "labels": labels,
        "slide_id": [x["slide_id"] for x in batch],
        "project": [x["project"] for x in batch],
        "wsi_path": [x["wsi_path"] for x in batch],
        "coord_x": [x["coord_x"] for x in batch],
        "coord_y": [x["coord_y"] for x in batch],
    }


def main():
    parser = argparse.ArgumentParser("Extract cached patch features for plugin training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--full_ckpt", type=str, required=True)
    parser.add_argument("--role_proto_dir", type=str, required=True)

    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--tcga_root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])

    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--samples_per_epoch", type=int, default=4000)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    encoder, role_proj_head, _, _ = load_stage2_for_plugin(
        config_path=args.config,
        full_ckpt_path=args.full_ckpt,
        role_proto_dir=args.role_proto_dir,
        device=device,
        shared_proto_learnable=False,
    )
    encoder.eval()
    role_proj_head.eval()

    dataset = TCGAPluginPatchDataset(
        split_csv=args.split_csv,
        split=args.split,
        tcga_root=args.tcga_root,
        image_size=args.image_size,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=plugin_patch_collate_fn,
    )

    meta_rows: List[Dict] = []

    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {args.split}"):
            images = batch["images"].to(device, non_blocking=True)

            student_out, gate_info_list, feature_dict, moe_feature_list = encoder(
                images,
                return_gates=True,
                mask=None,
                is_eval=True,
                return_features=True,
                offline_cluster_ids=None,
            )

            if len(moe_feature_list) > 0:
                feat = moe_feature_list[-1]
            else:
                feat = feature_dict["layer_12"]

            patch_feat_raw = feat[:, 1:, :]                          # [B, N, D]
            patch_feat_raw = patch_feat_raw.mean(dim=1)              # [B, D] patch-image级

            patch_feat_teacher = role_proj_head(patch_feat_raw)      # [B, D_role]
            patch_feat_teacher = F.normalize(patch_feat_teacher, dim=-1)

            B = patch_feat_raw.shape[0]
            for i in range(B):
                obj = {
                    "patch_feat_raw": patch_feat_raw[i].detach().cpu(),
                    "patch_feat_teacher_space": patch_feat_teacher[i].detach().cpu(),
                    "label": int(batch["labels"][i].item()),
                    "slide_id": batch["slide_id"][i],
                    "project": batch["project"][i],
                    "wsi_path": batch["wsi_path"][i],
                    "coord_x": int(batch["coord_x"][i]),
                    "coord_y": int(batch["coord_y"][i]),
                }

                out_path = os.path.join(args.out_dir, f"{global_idx:07d}.pt")
                torch.save(obj, out_path)

                meta_rows.append({
                    "cache_path": out_path,
                    "label": obj["label"],
                    "slide_id": obj["slide_id"],
                    "project": obj["project"],
                    "wsi_path": obj["wsi_path"],
                    "coord_x": obj["coord_x"],
                    "coord_y": obj["coord_y"],
                    "split": args.split,
                })
                global_idx += 1

    pd.DataFrame(meta_rows).to_csv(
        os.path.join(args.out_dir, f"{args.split}_cache_index.csv"),
        index=False,
    )
    print(f"[Done] saved {global_idx} cached samples to: {args.out_dir}")


if __name__ == "__main__":
    main()