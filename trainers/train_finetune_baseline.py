#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import math
import copy
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import openslide
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Encoder imports
# =========================================================
from models.encoders.timm_pathology_moe_encoder import (
    UNIEncoder,
    UNI2HEncoder,
    HOptimus0Encoder,
)

from models.encoders.virchow2_moe_encoder import Virchow2Encoder
from models.encoders.dinov2_encoder import DINOv2Encoder
from models.encoders.openclip_moe_encoder import OpenCLIPEncoder


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir = Path(raw_dir)
    exts = [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"]

    exact_matches = []
    for ext in exts:
        exact_matches.extend(raw_dir.rglob(f"{slide_id}{ext}"))

    if len(exact_matches) == 1:
        return str(exact_matches[0])

    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Found multiple exact WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact_matches[:10])
        )

    fuzzy_matches = []
    for ext in exts:
        fuzzy_matches.extend(raw_dir.rglob(f"{slide_id}*{ext}"))

    if len(fuzzy_matches) == 1:
        return str(fuzzy_matches[0])

    if len(fuzzy_matches) > 1:
        exact_name = [p for p in fuzzy_matches if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])

        raise RuntimeError(
            f"Found multiple WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy_matches[:10])
        )

    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def resolve_wsi_path(row: pd.Series, raw_dir: Optional[str] = None) -> str:
    if "source_path" in row and pd.notna(row["source_path"]):
        source_path = str(row["source_path"])
        if os.path.exists(source_path):
            return source_path

    if raw_dir is not None:
        return find_wsi_path(raw_dir, str(row["slide_id"]))

    raise FileNotFoundError(
        f"Cannot resolve WSI path for slide_id={row['slide_id']}."
    )


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir = Path(h5_dir)

    exact = list(h5_dir.rglob(f"{slide_id}.h5"))
    if len(exact) == 1:
        return str(exact[0])

    if len(exact) > 1:
        raise RuntimeError(
            f"Found multiple exact h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact[:10])
        )

    fuzzy = list(h5_dir.rglob(f"{slide_id}*.h5"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])

    if len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])

        raise RuntimeError(
            f"Found multiple h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy[:10])
        )

    raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")


def read_coords_from_h5(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return coords.astype(np.int64)


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def load_slides_csv(slides_csv: str) -> pd.DataFrame:
    df = pd.read_csv(slides_csv)

    if "slide_id" not in df.columns and "image_id" in df.columns:
        df["slide_id"] = df["image_id"]

    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("slides_csv must contain 'label' or 'slide_binary_label'")

    required = {"slide_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"slides_csv missing columns: {missing}")

    return df.reset_index(drop=True)


def subsample_slides(
    df: pd.DataFrame,
    max_slides: Optional[int] = None,
    balance_by_label: bool = False,
    seed: int = 42,
    split_name: str = "",
) -> pd.DataFrame:
    """
    Optionally subsample slides for budget-matched baselines.

    Important: when max_slides is None / <=0 / >= len(df), this function returns
    the original split unchanged. Therefore, not passing the new CLI arguments
    preserves the old training behavior exactly.
    """
    df = df.reset_index(drop=True)

    if max_slides is None or max_slides <= 0 or len(df) <= max_slides:
        print(
            f"[Subsample:{split_name}] keep all slides: "
            f"n={len(df)}, max_slides={max_slides}, balance={balance_by_label}"
        )
        return df

    rng = np.random.default_rng(seed)

    if not balance_by_label:
        sel = rng.choice(len(df), size=max_slides, replace=False)
        out = df.iloc[sel].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(
            f"[Subsample:{split_name}] random: "
            f"{len(df)} -> {len(out)}, balance={balance_by_label}"
        )
        print(f"[Subsample:{split_name}] label counts: {out['label'].value_counts().to_dict()}")
        return out

    labels = sorted(df["label"].unique().tolist())
    num_classes = len(labels)
    if num_classes == 0:
        raise ValueError(f"[Subsample:{split_name}] no labels found.")

    base_n = max_slides // num_classes
    remainder = max_slides % num_classes

    selected_parts = []
    for i, lab in enumerate(labels):
        sub = df[df["label"] == lab].copy()
        target_n = base_n + (1 if i < remainder else 0)
        target_n = min(target_n, len(sub))

        if target_n <= 0:
            continue

        selected_parts.append(
            sub.sample(n=target_n, random_state=seed + int(i) * 997)
        )

    if len(selected_parts) == 0:
        raise RuntimeError(f"[Subsample:{split_name}] selected no slides.")

    out = pd.concat(selected_parts, axis=0)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(
        f"[Subsample:{split_name}] balanced: "
        f"{len(df)} -> {len(out)}, requested={max_slides}"
    )
    print(f"[Subsample:{split_name}] label counts: {out['label'].value_counts().to_dict()}")

    return out


# =========================================================
# Dataset
# =========================================================
class WSIBagDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        raw_dir: str,
        h5_dir: str,
        patch_size: int = 256,
        max_instances: Optional[int] = 256,
        train: bool = True,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.raw_dir = raw_dir
        self.h5_dir = h5_dir
        self.patch_size = patch_size
        self.max_instances = max_instances
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        slide_path = resolve_wsi_path(row, raw_dir=self.raw_dir)
        h5_path = find_h5_path(self.h5_dir, slide_id)

        coords = read_coords_from_h5(h5_path)

        if self.max_instances is not None and len(coords) > self.max_instances:
            if self.train:
                rng = np.random.default_rng(self.seed + idx)
            else:
                rng = np.random.default_rng(123456 + idx)

            sel = rng.choice(len(coords), size=self.max_instances, replace=False)
            coords = coords[sel]

        return {
            "slide_id": slide_id,
            "label": label,
            "slide_path": slide_path,
            "coords": coords,
        }


def mil_collate_fn(batch):
    assert len(batch) == 1, "Use batch_size=1 for online MIL baseline."
    return batch[0]


# =========================================================
# LoRA
# =========================================================
class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def maybe_wrap_linear_with_lora(
    module: nn.Module,
    name: str,
    rank: int,
    alpha: float,
    dropout: float,
):
    child = getattr(module, name, None)
    if isinstance(child, nn.Linear):
        setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        print(f"[LoRA] wrapped {module.__class__.__name__}.{name}")


def inject_lora_into_blocks(
    blocks,
    block_indices: List[int],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
):
    """
    Supports:
    - timm ViT blocks: UNI / UNI2-H / Virchow2 / H-optimus-0
    - HuggingFace DINOv2 blocks
    - OpenCLIP visual transformer resblocks:
        blk.attn.out_proj
        blk.mlp.c_fc
        blk.mlp.c_proj
    """
    n_wrapped = 0

    for i in block_indices:
        blk = blocks[i]
        before = sum(1 for _ in blk.modules() if isinstance(_, LoRALinear))

        # 1) timm ViT style attention
        if hasattr(blk, "attn"):
            if hasattr(blk.attn, "qkv"):
                maybe_wrap_linear_with_lora(blk.attn, "qkv", rank, alpha, dropout)
            if hasattr(blk.attn, "proj"):
                maybe_wrap_linear_with_lora(blk.attn, "proj", rank, alpha, dropout)

            # Do NOT wrap nn.MultiheadAttention.out_proj for OpenCLIP.
            # PyTorch MultiheadAttention.forward directly accesses out_proj.weight/bias
            # instead of calling out_proj(x), so replacing it with LoRALinear breaks forward.
            if hasattr(blk.attn, "out_proj"):
                print(
                    f"[LoRA][Skip] {blk.attn.__class__.__name__}.out_proj "
                    "because nn.MultiheadAttention expects .weight/.bias directly."
                )

        # 2) timm / OpenCLIP MLP
        if hasattr(blk, "mlp"):
            # timm MLP
            if hasattr(blk.mlp, "fc1"):
                maybe_wrap_linear_with_lora(blk.mlp, "fc1", rank, alpha, dropout)
            if hasattr(blk.mlp, "fc2"):
                maybe_wrap_linear_with_lora(blk.mlp, "fc2", rank, alpha, dropout)

            # OpenCLIP MLP
            if hasattr(blk.mlp, "c_fc"):
                maybe_wrap_linear_with_lora(blk.mlp, "c_fc", rank, alpha, dropout)
            if hasattr(blk.mlp, "c_proj"):
                maybe_wrap_linear_with_lora(blk.mlp, "c_proj", rank, alpha, dropout)

        # 3) HuggingFace DINOv2 style
        if hasattr(blk, "attention"):
            attn = blk.attention

            if hasattr(attn, "attention"):
                inner_attn = attn.attention
                for name in ["query", "key", "value"]:
                    if hasattr(inner_attn, name):
                        maybe_wrap_linear_with_lora(inner_attn, name, rank, alpha, dropout)

            if hasattr(attn, "output") and hasattr(attn.output, "dense"):
                maybe_wrap_linear_with_lora(attn.output, "dense", rank, alpha, dropout)

        after = sum(1 for _ in blk.modules() if isinstance(_, LoRALinear))
        n_wrapped += max(0, after - before)

    print(f"[LoRA] total wrapped Linear layers: {n_wrapped}")


# =========================================================
# Encoder builder
# =========================================================
def build_base_encoder(
    encoder_name: str,
    args,
):
    encoder_name = encoder_name.lower()

    if encoder_name == "uni":
        if args.uni_weight.strip() == "":
            raise ValueError("--uni_weight is required for encoder_name=uni")

        enc = UNIEncoder(
            weight_path=args.uni_weight,
            device=args.device,
        )

    elif encoder_name in ["uni2_h", "uni2h"]:
        if args.uni2_weight.strip() == "":
            raise ValueError("--uni2_weight is required for encoder_name=uni2_h")

        enc = UNI2HEncoder(
            weight_path=args.uni2_weight,
            device=args.device,
        )

    elif encoder_name in ["dinov2_small", "dinov2-s", "dinov2"]:
        enc = DINOv2Encoder(
            model_name=args.dinov2_model_name,
            device=args.device,
            cache_dir=args.dinov2_cache_dir,
        )

    elif encoder_name in ["virchow2", "virchow"]:
        if args.virchow2_weight.strip() == "":
            raise ValueError("--virchow2_weight is required for encoder_name=virchow2")

        enc = Virchow2Encoder(
            weight_path=args.virchow2_weight,
            device=args.device,
        )

    elif encoder_name in ["openclip", "open_clip"]:
        if args.openclip_weight.strip() == "":
            raise ValueError("--openclip_weight is required for encoder_name=openclip")

        enc = OpenCLIPEncoder(
            model_name=args.openclip_model_name,
            weight_path=args.openclip_weight,
            device=args.device,
            precision=args.openclip_precision,
            normalize=args.openclip_normalize,
        )

    elif encoder_name == "hoptimus0":
        enc = HOptimus0Encoder(
            device=args.device,
            local_hf_hub_id=(
                args.hopt_local_hf_hub_id
                if args.hopt_local_hf_hub_id.strip() != ""
                else None
            ),
            manual_arch_name=(
                args.hopt_manual_arch_name
                if args.hopt_manual_arch_name.strip() != ""
                else None
            ),
            manual_create_kwargs=None,
            weight_path=(
                args.hopt_weight
                if args.hopt_weight.strip() != ""
                else None
            ),
        )

    else:
        raise ValueError(f"Unsupported encoder_name: {encoder_name}")

    return enc


# =========================================================
# Trainable encoder wrapper
# =========================================================
class TrainablePathologyEncoder(nn.Module):
    """
    Unified wrapper for online MIL baseline.

    Token encoders:
        UNI / UNI2-H / Virchow2 / DINOv2-small / H-optimus-0
        output = concat(cls_token, mean_patch_token)

    Feature encoders:
        OpenCLIP
        output = global image feature from extract_features()
    """

    def __init__(
        self,
        base_encoder,
        train_mode: str = "frozen",
        train_last_n_blocks: int = 2,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.base_encoder = base_encoder
        self.model = self.base_encoder.model
        self.device = self.base_encoder.device

        self.is_feature_encoder = hasattr(self.base_encoder, "extract_features")
        self.is_openclip = self.is_feature_encoder and hasattr(self.model, "visual")

        if self.is_feature_encoder:
            self.embed_dim = int(self.base_encoder.out_dim)
            self.out_dim = self.embed_dim
            self.reg_tokens = 0

            self.blocks = None
            self.norm = None

            if hasattr(self.model, "visual"):
                visual = self.model.visual

                if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
                    self.blocks = visual.transformer.resblocks

                if hasattr(visual, "ln_post"):
                    self.norm = visual.ln_post

        else:
            self.embed_dim = int(self.base_encoder.embed_dim)
            self.reg_tokens = int(getattr(self.base_encoder, "reg_tokens", 0))
            self.out_dim = self.embed_dim * 2
            self.blocks = self.base_encoder.blocks
            self.norm = getattr(self.base_encoder, "norm", None)

        self.train_mode_name = train_mode

        # freeze all first
        for p in self.model.parameters():
            p.requires_grad = False

        if self.blocks is not None:
            depth = len(self.blocks)
            train_last_n_blocks = max(0, min(train_last_n_blocks, depth))
            self.train_block_indices = list(range(depth - train_last_n_blocks, depth))
        else:
            depth = 0
            self.train_block_indices = []

        print(
            f"[TrainablePathologyEncoder] "
            f"base={type(self.base_encoder).__name__}, "
            f"feature_encoder={self.is_feature_encoder}, "
            f"openclip={self.is_openclip}, "
            f"train_mode={train_mode}, depth={depth}, "
            f"train_blocks={self.train_block_indices}, "
            f"out_dim={self.out_dim}, reg_tokens={self.reg_tokens}"
        )

        if train_mode == "partial_ft":
            if self.blocks is None:
                raise RuntimeError(
                    f"Cannot use partial_ft because no transformer blocks were found "
                    f"for {type(self.base_encoder).__name__}."
                )

            for idx in self.train_block_indices:
                for p in self.blocks[idx].parameters():
                    p.requires_grad = True

            if self.norm is not None:
                for p in self.norm.parameters():
                    p.requires_grad = True

            # OpenCLIP visual projection is important for encode_image output
            if self.is_openclip:
                visual = self.model.visual

                if hasattr(visual, "proj"):
                    if isinstance(visual.proj, nn.Parameter):
                        visual.proj.requires_grad = True
                    elif isinstance(visual.proj, nn.Module):
                        for p in visual.proj.parameters():
                            p.requires_grad = True

        elif train_mode == "lora":
            if self.blocks is None:
                raise RuntimeError(
                    f"Cannot use lora because no transformer blocks were found "
                    f"for {type(self.base_encoder).__name__}."
                )

            inject_lora_into_blocks(
                blocks=self.blocks,
                block_indices=self.train_block_indices,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

            for n, p in self.model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

            # For OpenCLIP LoRA, also train final ln_post and visual.proj.
            if self.is_openclip:
                visual = self.model.visual

                if hasattr(visual, "ln_post"):
                    for p in visual.ln_post.parameters():
                        p.requires_grad = True

                if hasattr(visual, "proj"):
                    if isinstance(visual.proj, nn.Parameter):
                        visual.proj.requires_grad = True
                    elif isinstance(visual.proj, nn.Module):
                        for p in visual.proj.parameters():
                            p.requires_grad = True

        elif train_mode == "frozen":
            pass

        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def forward_dinov2_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Safe DINOv2 token forward used only inside train_finetune_baseline.py.
        This avoids modifying models/encoders/dinov2_encoder.py.
        Expected output: [B, N, D]
        """
        if isinstance(images, Image.Image):
            images = [images]

        if not hasattr(self.base_encoder, "processor"):
            raise AttributeError("DINOv2Encoder should have .processor")
        if not hasattr(self.base_encoder, "embeddings"):
            raise AttributeError("DINOv2Encoder should have .embeddings")
        if not hasattr(self.base_encoder, "blocks"):
            raise AttributeError("DINOv2Encoder should have .blocks")
        if not hasattr(self.base_encoder, "norm"):
            raise AttributeError("DINOv2Encoder should have .norm")

        inputs = self.base_encoder.processor(
            images=[img.convert("RGB") for img in images],
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        x = self.base_encoder.embeddings(pixel_values)

        if x.dim() != 3:
            raise RuntimeError(
                f"[DINO safe forward] embeddings should be [B, N, D], got {tuple(x.shape)}"
            )

        for blk in self.base_encoder.blocks:
            out = blk(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out

        x = self.base_encoder.norm(x)

        if x.dim() != 3:
            raise RuntimeError(
                f"[DINO safe forward] final tokens should be [B, N, D], got {tuple(x.shape)}"
            )

        return x

    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        if type(self.base_encoder).__name__ == "DINOv2Encoder":
            return self.forward_dinov2_tokens(images)

        tokens = self.base_encoder(images, return_tokens=True)

        if isinstance(tokens, tuple):
            tokens = tokens[0]

        if not isinstance(tokens, torch.Tensor):
            raise TypeError(
                f"Expected tensor tokens, got {type(tokens)} from {type(self.base_encoder).__name__}"
            )

        if tokens.dim() == 2:
            raise RuntimeError(
                f"Expected token output [B, N, D], got {tuple(tokens.shape)} "
                f"from {type(self.base_encoder).__name__}. "
                f"This means the encoder lost the patch batch dimension."
            )

        if tokens.dim() != 3:
            raise RuntimeError(
                f"Expected token output [B, N, D], got {tuple(tokens.shape)} "
                f"from {type(self.base_encoder).__name__}"
            )

        return tokens

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        if self.is_feature_encoder:
            return self.base_encoder.extract_features(images)

        tokens = self.forward_tokens(images)

        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens

        if patch_start >= tokens.shape[1]:
            patch_start = 1

        patch_mean = tokens[:, patch_start:, :].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=-1)

        return feat

    def get_export_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Export full backbone state_dict.
        If LoRA is used, merge LoRA weights into the base Linear layers first.
        """
        model_for_export = copy.deepcopy(self.model).cpu()
        model_for_export.eval()

        def _merge_lora_linear(parent_module: nn.Module, child_name: str):
            child = getattr(parent_module, child_name)

            if not isinstance(child, LoRALinear):
                return

            base = child.base
            merged = nn.Linear(
                in_features=base.in_features,
                out_features=base.out_features,
                bias=(base.bias is not None),
            )

            merged.weight.data.copy_(base.weight.data)

            if base.bias is not None:
                merged.bias.data.copy_(base.bias.data)

            delta_w = child.scaling * (child.lora_B @ child.lora_A)
            merged.weight.data += delta_w.to(merged.weight.dtype)

            setattr(parent_module, child_name, merged)

        for module in model_for_export.modules():
            for child_name, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    _merge_lora_linear(module, child_name)

        return model_for_export.state_dict()


# =========================================================
# ABMIL
# =========================================================
class ABMILHead(nn.Module):
    def __init__(self, in_dim: int, att_dim: int = 256, dropout: float = 0.25):
        super().__init__()

        self.att_mlp = nn.Sequential(
            nn.Linear(in_dim, att_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(att_dim, 1),
        )

        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, feats: torch.Tensor):
        # feats: [N, D]
        att_logits = self.att_mlp(feats).squeeze(-1)  # [N]
        att = torch.softmax(att_logits, dim=0)        # [N]

        bag_feat = torch.sum(
            feats * att.unsqueeze(-1),
            dim=0,
            keepdim=True,
        )  # [1, D]

        logit = self.cls(bag_feat).squeeze(0).squeeze(-1)

        return {
            "logit": logit,
            "att": att,
            "bag_feat": bag_feat,
        }


class OnlineMILModel(nn.Module):
    def __init__(self, encoder: TrainablePathologyEncoder, mil_head: ABMILHead):
        super().__init__()
        self.encoder = encoder
        self.mil_head = mil_head

    def forward(self, patch_images: List[Image.Image]):
        feats = self.encoder(patch_images)
        out = self.mil_head(feats)
        return out


# =========================================================
# Metrics
# =========================================================
def eval_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    auc = float("nan")

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass

    y_pred = (y_prob >= 0.5).astype(np.int64)

    acc = float((y_pred == y_true).mean())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1 = 2.0 * prec * rec / max(1e-8, prec + rec)

    return {
        "auc": auc,
        "acc": acc,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "sensitivity": float(rec),
        "specificity": float(spec),
    }


# =========================================================
# Train / Eval
# =========================================================
def build_optimizer(
    model: nn.Module,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
):
    encoder_params = []
    head_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if n.startswith("encoder."):
            encoder_params.append(p)
        else:
            head_params.append(p)

    param_groups = []

    if len(encoder_params) > 0:
        param_groups.append({
            "params": encoder_params,
            "lr": lr_encoder,
            "weight_decay": weight_decay,
        })

    if len(head_params) > 0:
        param_groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, epochs: int, min_lr_ratio: float = 0.1):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=min_lr_ratio,
    )


def load_patch_images_for_bag(
    slide_path: str,
    coords: np.ndarray,
    patch_size: int,
) -> List[Image.Image]:
    slide = openslide.OpenSlide(slide_path)

    try:
        patch_images = []
        for xy in coords.tolist():
            img = read_patch_from_wsi(
                slide=slide,
                coord_xy=(int(xy[0]), int(xy[1])),
                patch_size=patch_size,
                read_level=0,
            )

            # All current encoders use 224 input.
            # DINO processor and OpenCLIP transform can accept PIL directly,
            # but resizing here keeps behavior consistent with your original script.
            img = img.resize((224, 224), resample=Image.BICUBIC)
            patch_images.append(img)

    finally:
        slide.close()

    return patch_images


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device,
    patch_size: int,
    grad_clip: float = 1.0,
):
    model.train()

    # For frozen OpenCLIP, keep visual encoder in eval mode.
    # For partial_ft / lora, allow train mode.
    if (
        getattr(model.encoder, "is_feature_encoder", False)
        and getattr(model.encoder, "train_mode_name", "frozen") == "frozen"
    ):
        model.encoder.base_encoder.model.eval()

    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for batch in tqdm(loader, desc="Train", leave=False):
        label = torch.tensor(float(batch["label"]), device=device)

        patch_images = load_patch_images_for_bag(
            slide_path=batch["slide_path"],
            coords=batch["coords"],
            patch_size=patch_size,
        )

        out = model(patch_images)
        logit = out["logit"]

        loss = criterion(logit.view(1), label.view(1))

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        losses.append(float(loss.item()))

    return {
        "loss": float(np.mean(losses)) if len(losses) > 0 else float("nan")
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device,
    patch_size: int,
):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    losses = []
    y_true = []
    y_prob = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        label = torch.tensor(float(batch["label"]), device=device)

        patch_images = load_patch_images_for_bag(
            slide_path=batch["slide_path"],
            coords=batch["coords"],
            patch_size=patch_size,
        )

        out = model(patch_images)
        logit = out["logit"]
        loss = criterion(logit.view(1), label.view(1))

        prob = torch.sigmoid(logit).item()

        losses.append(float(loss.item()))
        y_true.append(int(batch["label"]))
        y_prob.append(float(prob))

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    cls_metrics = eval_binary_metrics(y_true, y_prob)

    return {
        "loss": float(np.mean(losses)) if len(losses) > 0 else float("nan"),
        **cls_metrics,
    }


# =========================================================
# Save export encoder
# =========================================================
def save_exportable_encoder_weight(
    encoder: TrainablePathologyEncoder,
    out_path: str,
):
    export_sd = encoder.get_export_state_dict()
    torch.save({"encoder": export_sd}, out_path)
    print(f"[Saved encoder for extraction] {out_path}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        "Online MIL fine-tuning baseline for multiple pathology backbones "
        "(frozen / partial_ft / lora)"
    )

    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument(
        "--encoder_name",
        type=str,
        required=True,
        choices=[
            "uni",
            "uni2_h",
            "uni2h",
            "dinov2_small",
            "dinov2-s",
            "dinov2",
            "virchow2",
            "virchow",
            "openclip",
            "open_clip",
            "hoptimus0",
        ],
    )

    # =====================================================
    # Backbone weights / configs
    # =====================================================
    parser.add_argument("--uni_weight", type=str, default="")
    parser.add_argument("--uni2_weight", type=str, default="")

    parser.add_argument(
        "--dinov2_model_name",
        type=str,
        default="facebook/dinov2-small",
    )
    parser.add_argument(
        "--dinov2_cache_dir",
        type=str,
        default="./pretrained_models",
    )

    parser.add_argument(
        "--virchow2_weight",
        type=str,
        default="models/distill_teacher/Virchow2/pytorch_model.bin",
    )

    parser.add_argument("--openclip_model_name", type=str, default="ViT-B-16")
    parser.add_argument("--openclip_weight", type=str, default="")
    parser.add_argument(
        "--openclip_precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--openclip_normalize",
        action="store_true",
        help="L2-normalize OpenCLIP image features.",
    )
    parser.add_argument(
        "--no_openclip_normalize",
        dest="openclip_normalize",
        action="store_false",
    )
    parser.set_defaults(openclip_normalize=True)

    parser.add_argument("--hopt_local_hf_hub_id", type=str, default="")
    parser.add_argument("--hopt_manual_arch_name", type=str, default="")
    parser.add_argument("--hopt_weight", type=str, default="")

    # =====================================================
    # Train mode
    # =====================================================
    parser.add_argument(
        "--train_mode",
        type=str,
        default="frozen",
        choices=["frozen", "partial_ft", "lora"],
    )
    parser.add_argument("--train_last_n_blocks", type=int, default=2)

    # =====================================================
    # LoRA
    # =====================================================
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # =====================================================
    # MIL
    # =====================================================
    parser.add_argument("--att_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)

    # =====================================================
    # Optimization
    # =====================================================
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr_encoder", type=float, default=5e-6)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # =====================================================
    # Data
    # =====================================================
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_instances", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # Optional slide-budget controls.
    # Defaults preserve original behavior exactly: no train/val subsampling.
    parser.add_argument(
        "--max_train_slides",
        type=int,
        default=0,
        help="If >0, subsample at most this many train slides. Default 0 keeps all train slides.",
    )
    parser.add_argument(
        "--max_val_slides",
        type=int,
        default=0,
        help="If >0, subsample at most this many val slides. Default 0 keeps all val slides.",
    )
    parser.add_argument(
        "--balance_train_slides_by_label",
        action="store_true",
        help="Subsample train slides with approximately balanced labels.",
    )
    parser.add_argument(
        "--balance_val_slides_by_label",
        action="store_true",
        help="Subsample val slides with approximately balanced labels.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)

    print(f"[Device] {device}")
    print(f"[Args] {json.dumps(vars(args), indent=2, ensure_ascii=False)}")

    df = load_slides_csv(args.slides_csv)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(
        f"[Data before subsample] "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(f"[Data before subsample] train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"[Data before subsample] val labels: {val_df['label'].value_counts().to_dict()}")
    print(f"[Data before subsample] test labels: {test_df['label'].value_counts().to_dict()}")

    train_df = subsample_slides(
        train_df,
        max_slides=args.max_train_slides,
        balance_by_label=args.balance_train_slides_by_label,
        seed=args.seed,
        split_name="train",
    )

    val_df = subsample_slides(
        val_df,
        max_slides=args.max_val_slides,
        balance_by_label=args.balance_val_slides_by_label,
        seed=args.seed + 1000,
        split_name="val",
    )

    # Test split is never subsampled for reporting.
    print(
        f"[Data after subsample] "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    print(f"[Data after subsample] train labels: {train_df['label'].value_counts().to_dict()}")
    print(f"[Data after subsample] val labels: {val_df['label'].value_counts().to_dict()}")
    print(f"[Data after subsample] test labels: {test_df['label'].value_counts().to_dict()}")

    if len(train_df) == 0:
        raise ValueError("No train samples found. Check split column / subsampling settings.")
    if len(val_df) == 0:
        raise ValueError("No val samples found. Check split column / subsampling settings.")
    if len(test_df) == 0:
        raise ValueError("No test samples found. Check split column.")

    train_set = WSIBagDataset(
        train_df,
        raw_dir=args.raw_dir,
        h5_dir=args.h5_dir,
        patch_size=args.patch_size,
        max_instances=args.max_instances,
        train=True,
        seed=args.seed,
    )

    val_set = WSIBagDataset(
        val_df,
        raw_dir=args.raw_dir,
        h5_dir=args.h5_dir,
        patch_size=args.patch_size,
        max_instances=args.max_instances,
        train=False,
        seed=args.seed,
    )

    test_set = WSIBagDataset(
        test_df,
        raw_dir=args.raw_dir,
        h5_dir=args.h5_dir,
        patch_size=args.patch_size,
        max_instances=args.max_instances,
        train=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=mil_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=mil_collate_fn,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=mil_collate_fn,
    )

    print("[Build] base encoder ...")
    base_encoder = build_base_encoder(args.encoder_name, args)

    encoder = TrainablePathologyEncoder(
        base_encoder=base_encoder,
        train_mode=args.train_mode,
        train_last_n_blocks=args.train_last_n_blocks,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    mil_head = ABMILHead(
        in_dim=encoder.out_dim,
        att_dim=args.att_dim,
        dropout=args.dropout,
    )

    model = OnlineMILModel(
        encoder=encoder,
        mil_head=mil_head,
    ).to(device)

    optimizer = build_optimizer(
        model=model,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        epochs=args.epochs,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"[Params] trainable={trainable_params:,} / total={total_params:,}")

    history = []
    best_score = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            patch_size=args.patch_size,
            grad_clip=args.grad_clip,
        )

        val_res = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            patch_size=args.patch_size,
        )

        scheduler.step()

        rec = {
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "val_loss": val_res["loss"],
            "val_auc": val_res["auc"],
            "val_acc": val_res["acc"],
            "val_f1": val_res["f1"],
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "val_sensitivity": val_res["sensitivity"],
            "val_specificity": val_res["specificity"],
        }
        history.append(rec)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_res['loss']:.4f} | "
            f"val_auc={val_res['auc']:.4f} | "
            f"val_acc={val_res['acc']:.4f} | "
            f"val_f1={val_res['f1']:.4f} | "
            f"val_rec={val_res['recall']:.4f} | "
            f"val_spec={val_res['specificity']:.4f}"
        )

        score = val_res["auc"]
        if np.isnan(score):
            score = -1.0

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"[Early Stop] no val AUC improvement for {args.patience} epochs.")
            break

    if best_state is None:
        raise RuntimeError("No valid best checkpoint found.")

    model.load_state_dict(best_state)

    val_best = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        patch_size=args.patch_size,
    )

    test_best = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        patch_size=args.patch_size,
    )

    full_ckpt = {
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_metrics": val_best,
        "test_metrics": test_best,
        "args": vars(args),
    }

    full_ckpt_path = Path(args.out_dir) / "best_model.pth"
    torch.save(full_ckpt, full_ckpt_path)
    print(f"[Saved full checkpoint] {full_ckpt_path}")

    # OpenCLIP frozen baseline uses original local weight directly.
    # Saving full OpenCLIP encoder again is usually unnecessary and very large.
    if args.encoder_name.lower() in ["openclip", "open_clip"] and args.train_mode == "frozen":
        print("[Skip export encoder] OpenCLIP frozen baseline uses local original weight directly.")
    else:
        save_exportable_encoder_weight(
            encoder=model.encoder,
            out_path=str(Path(args.out_dir) / "best_encoder_for_extract.pth"),
        )

    history_path = Path(args.out_dir) / "train_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"[Saved history] {history_path}")

    summary = {
        "encoder_name": args.encoder_name,
        "train_mode": args.train_mode,
        "feature_type": (
            "global_image_embedding"
            if model.encoder.is_feature_encoder
            else "cls_plus_mean_patch_token"
        ),
        "encoder_out_dim": int(model.encoder.out_dim),
        "best_epoch": int(best_epoch),

        "max_train_slides": int(args.max_train_slides),
        "max_val_slides": int(args.max_val_slides),
        "balance_train_slides_by_label": bool(args.balance_train_slides_by_label),
        "balance_val_slides_by_label": bool(args.balance_val_slides_by_label),
        "num_train_slides_used": int(len(train_df)),
        "num_val_slides_used": int(len(val_df)),
        "num_test_slides_used": int(len(test_df)),
        "max_instances": int(args.max_instances) if args.max_instances is not None else None,

        "best_val_auc": val_best["auc"],
        "best_val_acc": val_best["acc"],
        "best_val_f1": val_best["f1"],
        "best_val_precision": val_best["precision"],
        "best_val_recall": val_best["recall"],
        "best_val_sensitivity": val_best["sensitivity"],
        "best_val_specificity": val_best["specificity"],

        "test_auc": test_best["auc"],
        "test_acc": test_best["acc"],
        "test_f1": test_best["f1"],
        "test_precision": test_best["precision"],
        "test_recall": test_best["recall"],
        "test_sensitivity": test_best["sensitivity"],
        "test_specificity": test_best["specificity"],

        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
    }

    summary_path = Path(args.out_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Saved summary] {summary_path}")

    print("[Best Val]")
    print(val_best)

    print("[Test]")
    print(test_best)

    print(f"[Done] saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
