#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import openslide
from PIL import ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from tqdm import tqdm

from models.plugin.load_stage2_for_plugin import load_stage2_for_plugin
from models.plugin.role_aware_tail_plugin import RoleAwareTailWithSharedSummary
from models.plugin.encoder_with_plugin import (
    EncoderWithRoleAwarePlugin,
    set_plugin_train_mode,
)
from models.plugin.plugin_losses import (
    compute_role_proto_anchor_loss,
    summarize_plugin_outputs,
)

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


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


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
    if sid is not None:
        return sid
    return str(slide_id)


def resolve_wsi_path(
    slide_id: str,
    project: str,
    tcga_root: str,
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
                    if os.path.splitext(fn)[1].lower() not in VALID_WSI_EXTS:
                        continue
                    full_path = normalize_path(os.path.join(root, fn))
                    if slide_id in full_path:
                        return full_path

    project_dir = os.path.join(tcga_root, str(project))
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Project dir not found: {project_dir}")

    matched = []
    for root, _, files in os.walk(project_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in VALID_WSI_EXTS:
                continue
            full_path = normalize_path(os.path.join(root, fn))
            if slide_id in full_path:
                matched.append(full_path)

    if len(matched) == 0:
        raise FileNotFoundError(f"No WSI found for slide_id={slide_id} under {project_dir}")

    matched = sorted(matched, key=lambda x: (len(x), x))
    return matched[0]


class TCGAPluginPatchDataset(Dataset):
    """
    每个样本 = 从 split 内随机抽一张 slide，再随机抽 1 个 patch。
    这是 patch-level plugin training。
    """

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
        super().__init__()

        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"split_csv not found: {split_csv}")

        df = pd.read_csv(split_csv)
        required_cols = ["slide_id", "label", "split", "project"]
        missing = [c for c in required_cols if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"split csv missing columns: {missing}")

        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split={split} in {split_csv}")

        self.df = df
        self.split = split
        self.tcga_root = tcga_root
        self.patch_level = int(patch_level)
        self.patch_size = int(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.base_seed = int(seed)
        self.transform = build_transform(image_size=image_size)

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

        if self.samples_per_epoch is None:
            self.samples_per_epoch = len(self.slide_records)

        self.pos_indices = [i for i, x in enumerate(self.slide_records) if x["label"] == 1]
        self.neg_indices = [i for i, x in enumerate(self.slide_records) if x["label"] == 0]

        print(f"[{split}] slides={len(self.slide_records)} pos={len(self.pos_indices)} neg={len(self.neg_indices)}")

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

    meta = {
        "slide_id": [x["slide_id"] for x in batch],
        "project": [x["project"] for x in batch],
        "wsi_path": [x["wsi_path"] for x in batch],
        "coord_x": [x["coord_x"] for x in batch],
        "coord_y": [x["coord_y"] for x in batch],
    }

    return {
        "images": images,
        "labels": labels,
        "meta": meta,
    }


def compute_tumor_gap(role_logits: torch.Tensor, tumor_role_id: int) -> torch.Tensor:
    sim_tumor = role_logits[..., tumor_role_id]
    other_ids = [i for i in range(role_logits.shape[-1]) if i != tumor_role_id]
    sim_other = role_logits[..., other_ids].max(dim=-1).values
    return sim_tumor - sim_other


def compute_patch_plugin_loss(
    patch_feat_raw: torch.Tensor,          # [B, N, D]
    patch_feat_plugin: torch.Tensor,       # [B, N, D]
    patch_role_logits_plugin: torch.Tensor,# [B, N, R]
    labels: torch.Tensor,                  # [B]
    tumor_role_id: int,
    pos_margin: float = 0.08,
    neg_margin: float = 0.00,
    feat_residual_weight: float = 0.01,
):
    """
    patch-level plugin training:
    - positive patch: tumor gap >= pos_margin
    - negative patch: tumor gap <= neg_margin
    - plus light residual consistency
    """
    gap = compute_tumor_gap(patch_role_logits_plugin, tumor_role_id=tumor_role_id)  # [B, N]
    gap_patch = gap.mean(dim=1)  # [B]

    pos_mask = labels == 1
    neg_mask = labels == 0

    loss_pos = gap_patch.new_tensor(0.0)
    loss_neg = gap_patch.new_tensor(0.0)

    if pos_mask.any():
        loss_pos = F.relu(pos_margin - gap_patch[pos_mask]).mean()

    if neg_mask.any():
        loss_neg = F.relu(gap_patch[neg_mask] - neg_margin).mean()

    loss_main = loss_pos + loss_neg
    loss_res = ((patch_feat_plugin - patch_feat_raw) ** 2).mean()
    total = loss_main + feat_residual_weight * loss_res

    stats = {
        "plugin_main_raw": safe_float(loss_main),
        "plugin_pos_raw": safe_float(loss_pos),
        "plugin_neg_raw": safe_float(loss_neg),
        "plugin_res_raw": safe_float(loss_res),
        "plugin_gap_mean": safe_float(gap_patch.mean()),
        "plugin_gap_pos_mean": safe_float(gap_patch[pos_mask].mean()) if pos_mask.any() else 0.0,
        "plugin_gap_neg_mean": safe_float(gap_patch[neg_mask].mean()) if neg_mask.any() else 0.0,
    }
    return total, stats


@torch.no_grad()
def evaluate(
    encoder_with_plugin: EncoderWithRoleAwarePlugin,
    loader: DataLoader,
    device: str,
    tumor_role_id: int,
    pos_margin: float,
    neg_margin: float,
    feat_residual_weight: float,
    proto_anchor_weight: float,
):
    encoder_with_plugin.eval()

    loss_all = []
    gap_pos_all = []
    gap_neg_all = []

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        out = encoder_with_plugin(
            images=images,
            is_eval=True,
            return_aux=True,
        )

        patch_feat_plugin_teacher = encoder_with_plugin.role_proj_head(out["patch_feat_plugin"])
        patch_feat_plugin_teacher = F.normalize(patch_feat_plugin_teacher, dim=-1)

        role_dict_plugin = encoder_with_plugin.summary_builder(patch_feat_plugin_teacher)

        loss, stats = compute_patch_plugin_loss(
            patch_feat_raw=out["patch_feat_raw"],
            patch_feat_plugin=out["patch_feat_plugin"],
            patch_role_logits_plugin=role_dict_plugin["patch_role_logits"],
            labels=labels,
            tumor_role_id=tumor_role_id,
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            feat_residual_weight=feat_residual_weight,
        )

        if encoder_with_plugin.shared_role_proto.prototypes.requires_grad and proto_anchor_weight > 0:
            loss = loss + proto_anchor_weight * compute_role_proto_anchor_loss(
                current_proto=encoder_with_plugin.shared_role_proto.get_prototypes(),
                init_proto=encoder_with_plugin.shared_role_proto.get_init_prototypes(),
                normalize=False,
                mode="cosine",
            )

        loss_all.append(safe_float(loss))

        gap = compute_tumor_gap(role_dict_plugin["patch_role_logits"], tumor_role_id=tumor_role_id).mean(dim=1)
        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.any():
            gap_pos_all.extend(gap[pos_mask].detach().cpu().tolist())
        if neg_mask.any():
            gap_neg_all.extend(gap[neg_mask].detach().cpu().tolist())

    val_loss = float(np.mean(loss_all)) if len(loss_all) > 0 else 0.0
    val_pos = float(np.mean(gap_pos_all)) if len(gap_pos_all) > 0 else 0.0
    val_neg = float(np.mean(gap_neg_all)) if len(gap_neg_all) > 0 else 0.0

    return {
        "val_loss": val_loss,
        "val_gap_pos_mean": val_pos,
        "val_gap_neg_mean": val_neg,
        "val_gap_margin": val_pos - val_neg,
    }


def main():
    parser = argparse.ArgumentParser("Train patch-level role-aware plugin")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--full_ckpt", type=str, required=True)
    parser.add_argument("--role_proto_dir", type=str, required=True)

    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--tcga_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--train_samples_per_epoch", type=int, default=4000)
    parser.add_argument("--val_samples_per_epoch", type=int, default=1000)

    parser.add_argument("--plugin_hidden_dim", type=int, default=128)
    parser.add_argument("--plugin_dropout", type=float, default=0.0)
    parser.add_argument("--plugin_init_scale", type=float, default=0.1)
    parser.add_argument("--use_role_logits", action="store_true")
    parser.add_argument("--use_top1_gap", action="store_true")
    parser.add_argument("--use_beta", action="store_true")

    parser.add_argument("--shared_proto_learnable", action="store_true")
    parser.add_argument("--proto_anchor_weight", type=float, default=1.0)

    parser.add_argument("--pos_margin", type=float, default=0.08)
    parser.add_argument("--neg_margin", type=float, default=0.00)
    parser.add_argument("--feat_residual_weight", type=float, default=0.01)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    train_set = TCGAPluginPatchDataset(
        split_csv=args.split_csv,
        split="train",
        tcga_root=args.tcga_root,
        image_size=args.image_size,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        samples_per_epoch=args.train_samples_per_epoch,
        seed=args.seed,
    )
    val_set = TCGAPluginPatchDataset(
        split_csv=args.split_csv,
        split="val",
        tcga_root=args.tcga_root,
        image_size=args.image_size,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        samples_per_epoch=args.val_samples_per_epoch,
        seed=args.seed + 1000,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=plugin_patch_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=plugin_patch_collate_fn,
    )

    encoder, role_proj_head, shared_role_proto, cfg = load_stage2_for_plugin(
        config_path=args.config,
        full_ckpt_path=args.full_ckpt,
        role_proto_dir=args.role_proto_dir,
        device=device,
        shared_proto_learnable=args.shared_proto_learnable,
    )

    plugin = RoleAwareTailWithSharedSummary(
        feat_dim=role_proj_head.in_features,
        num_roles=shared_role_proto.num_roles,
        hidden_dim=args.plugin_hidden_dim,
        dropout=args.plugin_dropout,
        use_role_logits=args.use_role_logits,
        use_top1_gap=args.use_top1_gap,
        use_beta=args.use_beta,
        init_scale=args.plugin_init_scale,
    ).to(device)

    encoder_with_plugin = EncoderWithRoleAwarePlugin(
        encoder=encoder,
        role_proj_head=role_proj_head,
        shared_role_proto=shared_role_proto,
        plugin=plugin,
        use_last_moe_output=True,
        freeze_encoder=True,
        freeze_role_proj=True,
    ).to(device)

    set_plugin_train_mode(
        encoder=encoder_with_plugin.encoder,
        role_proj_head=encoder_with_plugin.role_proj_head,
        shared_role_proto=encoder_with_plugin.shared_role_proto,
        plugin=encoder_with_plugin.plugin,
        aggregator=None,
        train_encoder=False,
        train_role_proj=False,
        train_shared_proto=args.shared_proto_learnable,
        train_plugin=True,
        train_aggregator=False,
    )

    role_name_to_id = shared_role_proto.role_name_to_id()
    if "tumor" not in role_name_to_id:
        raise ValueError(f"'tumor' not found in role names: {shared_role_proto.role_names}")
    tumor_role_id = role_name_to_id["tumor"]

    params = [p for p in encoder_with_plugin.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_metric = -1e18
    best_ckpt = os.path.join(args.out_dir, "best_plugin.pt")

    for epoch in range(1, args.epochs + 1):
        encoder_with_plugin.train()

        loss_all = []
        gap_pos_all = []
        gap_neg_all = []
        aux_stats_all = []

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            out = encoder_with_plugin(
                images=images,
                is_eval=False,
                return_aux=True,
            )

            patch_feat_plugin_teacher = encoder_with_plugin.role_proj_head(out["patch_feat_plugin"])
            patch_feat_plugin_teacher = F.normalize(patch_feat_plugin_teacher, dim=-1)

            role_dict_plugin = encoder_with_plugin.summary_builder(patch_feat_plugin_teacher)

            loss, stats = compute_patch_plugin_loss(
                patch_feat_raw=out["patch_feat_raw"],
                patch_feat_plugin=out["patch_feat_plugin"],
                patch_role_logits_plugin=role_dict_plugin["patch_role_logits"],
                labels=labels,
                tumor_role_id=tumor_role_id,
                pos_margin=args.pos_margin,
                neg_margin=args.neg_margin,
                feat_residual_weight=args.feat_residual_weight,
            )

            if args.shared_proto_learnable and args.proto_anchor_weight > 0:
                loss = loss + args.proto_anchor_weight * compute_role_proto_anchor_loss(
                    current_proto=encoder_with_plugin.shared_role_proto.get_prototypes(),
                    init_proto=encoder_with_plugin.shared_role_proto.get_init_prototypes(),
                    normalize=False,
                    mode="cosine",
                )

            loss.backward()
            optimizer.step()

            loss_all.append(safe_float(loss))
            aux_stats_all.append(summarize_plugin_outputs(out))

            gap = compute_tumor_gap(role_dict_plugin["patch_role_logits"], tumor_role_id=tumor_role_id).mean(dim=1)
            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.any():
                gap_pos_all.extend(gap[pos_mask].detach().cpu().tolist())
            if neg_mask.any():
                gap_neg_all.extend(gap[neg_mask].detach().cpu().tolist())

            pbar.set_postfix(loss=f"{np.mean(loss_all):.4f}")

        train_loss = float(np.mean(loss_all)) if len(loss_all) > 0 else 0.0
        train_pos = float(np.mean(gap_pos_all)) if len(gap_pos_all) > 0 else 0.0
        train_neg = float(np.mean(gap_neg_all)) if len(gap_neg_all) > 0 else 0.0
        train_margin = train_pos - train_neg

        val_stats = evaluate(
            encoder_with_plugin=encoder_with_plugin,
            loader=val_loader,
            device=device,
            tumor_role_id=tumor_role_id,
            pos_margin=args.pos_margin,
            neg_margin=args.neg_margin,
            feat_residual_weight=args.feat_residual_weight,
            proto_anchor_weight=args.proto_anchor_weight,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_gap_pos_mean": train_pos,
            "train_gap_neg_mean": train_neg,
            "train_gap_margin": train_margin,
            **val_stats,
        }

        if len(aux_stats_all) > 0:
            for k in aux_stats_all[0].keys():
                row[f"train_{k}"] = float(np.mean([x[k] for x in aux_stats_all]))

        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "plugin_train_history.csv"), index=False)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"train_margin={train_margin:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} "
            f"val_margin={val_stats['val_gap_margin']:.4f} "
            f"val_pos={val_stats['val_gap_pos_mean']:.4f} "
            f"val_neg={val_stats['val_gap_neg_mean']:.4f}"
        )

        cur_metric = val_stats["val_gap_margin"]
        if cur_metric > best_metric:
            best_metric = cur_metric
            torch.save(
                {
                    "epoch": epoch,
                    "plugin_state_dict": encoder_with_plugin.plugin.state_dict(),
                    "shared_role_proto_state_dict": encoder_with_plugin.shared_role_proto.state_dict(),
                    "role_names": encoder_with_plugin.shared_role_proto.role_names,
                    "feat_dim": role_proj_head.in_features,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"[Best] epoch={epoch}, val_gap_margin={best_metric:.4f}")

    print(f"\n[Done] best ckpt saved to: {best_ckpt}")


if __name__ == "__main__":
    main()