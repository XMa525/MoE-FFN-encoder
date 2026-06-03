#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virchow2 + frozen ABMIL supervision training.

Key change from the previous script:
- remove the simple slide classifier head
- load a pretrained ABMIL checkpoint
- freeze ABMIL parameters
- use Virchow2 patch features -> frozen ABMIL -> slide logits
- only update encoder side (LoRA or full encoder, depending on mode)

IMPORTANT:
1) You must replace the ABMIL import / construction in `load_frozen_abmil(...)`
   with your actual ABMIL model definition.
2) This script assumes your ABMIL forward can consume patch features of shape [B, N, D]
   and return slide logits of shape [B] or [B,1], or a tuple/list whose first item is logits.
3) If your ABMIL expects a different input dict or returns a different structure,
   only `load_frozen_abmil(...)` and `EncoderWithFrozenABMIL.forward(...)` need adjustment.

Outputs:
- best.pt
- best_encoder_state_dict.pt
- history.csv
- curves / prediction csv / metrics json
"""

from __future__ import annotations

import os
import math
import json
import time
import copy
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

import h5py
import numpy as np
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import openslide
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import timm
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


# =========================================================
# utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y"}


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_first_existing(d: pd.Series, keys: List[str], default=None):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return default


def infer_h5_path(wsi_path: str, h5_root: Optional[str] = None) -> Optional[str]:
    if wsi_path is None:
        return None
    stem = Path(wsi_path).stem
    if h5_root is None:
        candidate = Path(wsi_path).with_suffix(".h5")
        return str(candidate) if candidate.exists() else None
    candidate = Path(h5_root) / f"{stem}.h5"
    return str(candidate) if candidate.exists() else None


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


# =========================================================
# metrics / threshold
# =========================================================
def binary_metrics_from_probs(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)

    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auc": 0.0,
            "pos_rate": 0.0,
            "threshold": float(threshold),
        }

    y_pred = (y_prob >= threshold).astype(np.int64)

    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pos_rate": float(y_pred.mean()) if len(y_pred) > 0 else 0.0,
        "threshold": float(threshold),
    }
    return metrics


def find_best_threshold(
    y_true,
    y_prob,
    optimize_metric: str = "f1",
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    threshold_step: float = 0.01,
):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)

    if len(y_true) == 0:
        return 0.5, binary_metrics_from_probs(y_true, y_prob, threshold=0.5), []

    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    best_threshold = 0.5
    best_metrics = None
    best_score = -1e18
    curve_rows = []

    for thr in thresholds:
        m = binary_metrics_from_probs(y_true, y_prob, threshold=float(thr))
        score = m.get(optimize_metric, None)
        if score is None:
            raise ValueError(f"Unknown optimize metric: {optimize_metric}")

        curve_rows.append({
            "threshold": float(thr),
            "acc": m["acc"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "auc": m["auc"],
            "pos_rate": m["pos_rate"],
            "opt_score": float(score),
        })

        if (score > best_score) or (abs(score - best_score) < 1e-12 and abs(thr - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(thr)
            best_metrics = m

    return best_threshold, best_metrics, curve_rows


def save_prediction_csv(path, slide_ids, y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    df = pd.DataFrame({
        "slide_id": slide_ids,
        "label": y_true,
        "prob": y_prob,
        "pred": y_pred,
        "threshold": threshold,
    })
    df.to_csv(path, index=False)


# =========================================================
# LoRA
# =========================================================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + delta


def get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_keywords: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    block_ids: Optional[List[int]] = None,
):
    replace_names = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(k in name for k in target_keywords):
            continue

        if block_ids is not None:
            keep = False
            for bid in block_ids:
                if f"blocks.{bid}." in name:
                    keep = True
                    break
            if not keep:
                continue

        replace_names.append(name)

    for name in replace_names:
        parent, child_name = get_parent_module(model, name)
        old = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(old, rank=rank, alpha=alpha, dropout=dropout))

    return replace_names


# =========================================================
# Virchow2 encoder
# =========================================================
class Virchow2Backbone(nn.Module):
    """
    Output patch representation:
        concat([cls_token, mean(patch_tokens)]) => dim = 2560
    """
    def __init__(self, pretrained_ckpt: str):
        super().__init__()

        self.model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=False,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            init_values=1e-5,
        )

        state_dict = torch.load(pretrained_ckpt, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"⚠️ Strict loading failed, trying relaxed loading... {e}")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"[Virchow2] missing={len(missing)} unexpected={len(unexpected)}")

        if not hasattr(self.model, "pos_embed") or self.model.pos_embed is None:
            num_patches = self.model.patch_embed.num_patches + 1
            embed_dim = self.model.embed_dim
            self.model.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.out_dim = self.model.embed_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim != 3:
            raise RuntimeError(f"Expected token output [B, T, C], got shape={tuple(out.shape)}")

        cls_token = out[:, 0]
        reg_tokens = getattr(self.model, "reg_tokens", 0)
        patch_start = 1 + int(reg_tokens)
        patch_tokens = out[:, patch_start:]
        pooled = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        return pooled


# =========================================================
# ABMIL loader / wrapper
# =========================================================
def load_frozen_abmil(abmil_ckpt: str, feat_dim: int, device: torch.device):
    """
    Load frozen ABMIL from a checkpoint trained with the user's reference script.

    Compatible checkpoint formats:
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - raw state_dict

    Assumes the ABMIL was trained using:
        self.model = ABMIL(
            in_shape=(feat_dim,),
            att_dim=256,
            att_act="tanh",
            gated=False
        )

    Returns:
        a frozen nn.Module whose forward accepts patch feats [B, N, D]
        and outputs slide logits.
    """
    from torchmil.models import ABMIL

    class FrozenABMILWrapper(nn.Module):
        def __init__(self, in_dim: int, device: torch.device):
            super().__init__()
            self.model = ABMIL(
                in_shape=(in_dim,),
                att_dim=256,
                att_act="tanh",
                gated=False,
            )
            self.device = device
            self.to(device)

        def forward(self, bag_feats: torch.Tensor):
            """
            bag_feats: [B, N, D] or [N, D]
            return logits tensor
            """
            out = self.model(bag_feats)

            # same compatibility logic as your original ABMILWrapper
            if isinstance(out, torch.Tensor):
                return out

            if isinstance(out, dict):
                for key in ["logits", "pred", "scores", "output"]:
                    if key in out:
                        return out[key]

            if isinstance(out, (tuple, list)):
                for item in out:
                    if isinstance(item, torch.Tensor):
                        return item

            raise TypeError(f"Unsupported ABMIL output type: {type(out)}")

    abmil = FrozenABMILWrapper(in_dim=feat_dim, device=device)

    ckpt = torch.load(abmil_ckpt, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # maybe raw state_dict packed as dict
            state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    # remove possible DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state_dict[nk] = v
    state_dict = new_state_dict

    missing, unexpected = abmil.load_state_dict(state_dict, strict=False)
    print(f"[ABMIL] loaded from: {abmil_ckpt}")
    print(f"[ABMIL] missing keys: {missing}")
    print(f"[ABMIL] unexpected keys: {unexpected}")

    abmil = abmil.to(device)
    abmil.eval()

    for p in abmil.parameters():
        p.requires_grad = False

    return abmil

class EncoderWithFrozenABMIL(nn.Module):
    def __init__(self, encoder: nn.Module, abmil: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.abmil = abmil

    def encode_patches(self, patches: torch.Tensor, amp: bool = True):
        B, N, C, H, W = patches.shape
        x = patches.view(B * N, C, H, W)

        if amp and x.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)

        feats = feats.view(B, N, -1)
        return feats

    def forward(self, patches: torch.Tensor, amp: bool = True):
        patch_feats = self.encode_patches(patches, amp=amp)   # [B, N, D]

        # Assumption: ABMIL can consume [B, N, D]
        out = self.abmil(patch_feats)

        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        return logits, patch_feats


# =========================================================
# Dataset
# =========================================================
@dataclass
class SlideRecord:
    slide_id: str
    label: int
    wsi_path: str
    h5_path: str


class WSIPatchBagDataset(Dataset):
    def __init__(
        self,
        records: List[SlideRecord],
        transform,
        bag_size: int = 256,
        patch_size: int = 224,
        patch_level: int = 0,
        random_sample: bool = True,
        cache_open_slides: bool = True,
    ):
        self.records = records
        self.transform = transform
        self.bag_size = bag_size
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.random_sample = random_sample
        self.cache_open_slides = cache_open_slides
        self._slide_cache: Dict[str, openslide.OpenSlide] = {}

    def __len__(self):
        return len(self.records)

    def _get_slide(self, path: str):
        if self.cache_open_slides:
            if path not in self._slide_cache:
                self._slide_cache[path] = openslide.OpenSlide(path)
            return self._slide_cache[path]
        return openslide.OpenSlide(path)

    def _read_coords(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            coords = np.array(f["coords"])
            level = int(f["coords"].attrs.get("patch_level", f["coords"].attrs.get("level", self.patch_level)))
            size = int(f["coords"].attrs.get("patch_size", self.patch_size))
        return coords, level, size

    def _sample_coords(self, coords: np.ndarray):
        n = len(coords)
        if n == 0:
            raise RuntimeError("No coords found in h5.")

        if n >= self.bag_size:
            idx = np.random.choice(n, self.bag_size, replace=False) if self.random_sample else np.arange(self.bag_size)
            return coords[idx]

        rep = np.random.choice(n, self.bag_size - n, replace=True)
        idx = np.concatenate([np.arange(n), rep], axis=0)
        if self.random_sample:
            np.random.shuffle(idx)
        return coords[idx]

    def __getitem__(self, idx):
        rec = self.records[idx]
        coords, level, size = self._read_coords(rec.h5_path)
        coords = self._sample_coords(coords)

        slide = self._get_slide(rec.wsi_path)
        patches = []
        for xy in coords:
            x, y = int(xy[0]), int(xy[1])
            patch = slide.read_region((x, y), level, (size, size)).convert("RGB")
            patch = self.transform(patch)
            patches.append(patch)

        patches = torch.stack(patches, dim=0)
        label = torch.tensor(rec.label, dtype=torch.float32)

        return {
            "patches": patches,
            "label": label,
            "slide_id": rec.slide_id,
        }


def collate_bag(batch):
    return {
        "patches": torch.stack([b["patches"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "slide_id": [b["slide_id"] for b in batch],
    }


# =========================================================
# data prep
# =========================================================
def load_records(csv_path: str, h5_root: Optional[str] = None):
    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        slide_id = str(find_first_existing(row, ["slide_id", "id", "case_id"], default="unknown"))
        label = int(find_first_existing(row, ["label", "y", "target"]))
        wsi_path = find_first_existing(row, ["svs_path", "wsi_path", "source_path", "slide_path", "path"])
        h5_path = find_first_existing(row, ["h5_path"])
        split = find_first_existing(row, ["split"], default=None)

        if pd.isna(wsi_path):
            raise ValueError(f"Missing WSI path for slide {slide_id}")

        if pd.isna(h5_path) or h5_path is None:
            h5_path = infer_h5_path(str(wsi_path), h5_root=h5_root)

        if h5_path is None or not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 not found for slide {slide_id}: {h5_path}")
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI not found for slide {slide_id}: {wsi_path}")

        rows.append({
            "slide_id": slide_id,
            "label": label,
            "wsi_path": str(wsi_path),
            "h5_path": str(h5_path),
            "split": None if split is None or pd.isna(split) else str(split).lower(),
        })

    return pd.DataFrame(rows)


def split_records(df: pd.DataFrame, seed: int = 42, val_ratio: float = 0.2, test_ratio: float = 0.0):
    if "split" in df.columns and df["split"].notna().any():
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()
        return train_df, val_df, test_df

    train_df, temp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=df["label"],
    )

    if test_ratio > 0:
        rel_test = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=rel_test,
            random_state=seed,
            stratify=temp_df["label"],
        )
    else:
        val_df, test_df = temp_df, df.iloc[0:0].copy()

    return train_df, val_df, test_df


def df_to_records(df: pd.DataFrame):
    return [
        SlideRecord(
            slide_id=row["slide_id"],
            label=int(row["label"]),
            wsi_path=row["wsi_path"],
            h5_path=row["h5_path"],
        )
        for _, row in df.iterrows()
    ]


# =========================================================
# epoch subset sampling
# =========================================================
def build_epoch_subset_indices(
    dataset: WSIPatchBagDataset,
    subset_size: Optional[int],
    stratified: bool,
    seed: int,
    epoch: int,
):
    n = len(dataset)
    all_indices = np.arange(n)

    if subset_size is None or subset_size <= 0 or subset_size >= n:
        return all_indices.tolist()

    rng = np.random.default_rng(seed + epoch)

    if not stratified:
        chosen = rng.choice(all_indices, size=subset_size, replace=False)
        return chosen.tolist()

    labels = np.array([int(rec.label) for rec in dataset.records], dtype=np.int64)
    unique_labels = sorted(np.unique(labels).tolist())
    label_to_indices = {lab: np.where(labels == lab)[0] for lab in unique_labels}

    alloc = {}
    remaining = subset_size
    for i, lab in enumerate(unique_labels):
        class_idx = label_to_indices[lab]
        if i == len(unique_labels) - 1:
            k = remaining
        else:
            k = int(round(subset_size * len(class_idx) / n))
            k = min(k, len(class_idx))
            remaining -= k
        alloc[lab] = k

    if subset_size >= len(unique_labels):
        for lab in unique_labels:
            if alloc[lab] == 0 and len(label_to_indices[lab]) > 0:
                alloc[lab] = 1

        total_alloc = sum(alloc.values())
        if total_alloc > subset_size:
            overflow = total_alloc - subset_size
            labs_by_size = sorted(unique_labels, key=lambda x: len(label_to_indices[x]), reverse=True)
            for lab in labs_by_size:
                removable = max(0, alloc[lab] - 1)
                take = min(removable, overflow)
                alloc[lab] -= take
                overflow -= take
                if overflow == 0:
                    break
        elif total_alloc < subset_size:
            deficit = subset_size - total_alloc
            labs_by_size = sorted(unique_labels, key=lambda x: len(label_to_indices[x]), reverse=True)
            while deficit > 0:
                progressed = False
                for lab in labs_by_size:
                    max_add = len(label_to_indices[lab]) - alloc[lab]
                    if max_add > 0:
                        alloc[lab] += 1
                        deficit -= 1
                        progressed = True
                        if deficit == 0:
                            break
                if not progressed:
                    break

    chosen = []
    for lab in unique_labels:
        idxs = label_to_indices[lab]
        k = min(alloc[lab], len(idxs))
        if k > 0:
            picked = rng.choice(idxs, size=k, replace=False)
            chosen.extend(picked.tolist())

    chosen = list(dict.fromkeys(chosen))
    remaining_pool = [i for i in all_indices.tolist() if i not in set(chosen)]
    if len(chosen) < subset_size:
        fill_k = min(subset_size - len(chosen), len(remaining_pool))
        if fill_k > 0:
            fill = rng.choice(np.array(remaining_pool), size=fill_k, replace=False).tolist()
            chosen.extend(fill)
    elif len(chosen) > subset_size:
        chosen = rng.choice(np.array(chosen), size=subset_size, replace=False).tolist()

    rng.shuffle(chosen)
    return chosen


def make_epoch_train_loader(train_set: WSIPatchBagDataset, args, epoch: int):
    subset_indices = build_epoch_subset_indices(
        dataset=train_set,
        subset_size=args.train_subset_size_per_epoch,
        stratified=args.train_subset_stratified,
        seed=args.seed,
        epoch=epoch,
    )

    epoch_subset = Subset(train_set, subset_indices)
    loader = DataLoader(
        epoch_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_bag,
        persistent_workers=(args.num_workers > 0),
    )
    return loader, subset_indices


# =========================================================
# train / eval
# =========================================================
@torch.no_grad()
def evaluate_collect(
    model,
    loader,
    device,
    amp: bool = True,
    threshold: float = 0.5,
    max_steps: Optional[int] = None,
):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    y_true, y_prob, slide_ids = [], [], []
    total_loss = 0.0
    n_samples = 0
    n_steps = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and max_steps > 0 and step >= max_steps:
            break

        patches = batch["patches"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_ctx(device, amp):
            logits, _ = model(patches, amp=amp)
            loss = bce(logits, labels)

        prob = torch.sigmoid(logits)

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        n_steps += 1

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())
        slide_ids.extend(batch["slide_id"])

    metrics = binary_metrics_from_probs(y_true, y_prob, threshold=threshold)
    metrics.update({
        "loss": total_loss / max(n_samples, 1),
        "num_eval_steps": int(n_steps),
        "num_eval_samples": int(n_samples),
    })
    return metrics, np.asarray(y_true), np.asarray(y_prob), slide_ids


def train_one_epoch(model, loader, optimizer, scaler, device, amp: bool = True, grad_accum: int = 1):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    n_steps = 0
    y_true, y_prob = [], []

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        patches = batch["patches"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_ctx(device, amp):
            logits, _ = model(patches, amp=amp)
            loss = bce(logits, labels) / grad_accum

        if optimizer is not None:
            if amp and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

        prob = torch.sigmoid(logits.detach())
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())

        if optimizer is not None and (step + 1) % grad_accum == 0:
            if amp and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum * labels.size(0)
        n += labels.size(0)
        n_steps += 1

    if optimizer is not None and n_steps > 0 and (n_steps % grad_accum != 0):
        if amp and device.type == "cuda":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    train_metrics_05 = binary_metrics_from_probs(y_true, y_prob, threshold=0.5)

    return {
        "loss": total_loss / max(n, 1),
        "num_train_steps": int(n_steps),
        "num_train_samples": int(n),
        "train_auc_05": train_metrics_05["auc"],
        "train_acc_05": train_metrics_05["acc"],
        "train_f1_05": train_metrics_05["f1"],
        "train_precision_05": train_metrics_05["precision"],
        "train_recall_05": train_metrics_05["recall"],
        "train_pos_rate_05": train_metrics_05["pos_rate"],
    }


def count_trainable_params(model: nn.Module):
    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def configure_train_mode(model: EncoderWithFrozenABMIL, mode: str):
    for p in model.parameters():
        p.requires_grad = False

    # ABMIL stays frozen
    for p in model.abmil.parameters():
        p.requires_grad = False

    if mode == "freeze":
        return
    elif mode == "lora":
        for n, p in model.encoder.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True
    elif mode == "full":
        for p in model.encoder.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# visualization
# =========================================================
def plot_curve(x, ys: Dict[str, List[float]], xlabel: str, ylabel: str, title: str, save_path: str):
    plt.figure(figsize=(8, 5))
    for name, vals in ys.items():
        if len(vals) == 0:
            continue
        plt.plot(x, vals, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_prob_hist(y_true, y_prob, save_path: str, title: str):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    plt.figure(figsize=(7, 5))
    neg = y_prob[y_true == 0]
    pos = y_prob[y_true == 1]

    bins = np.linspace(0.0, 1.0, 21)
    if len(neg) > 0:
        plt.hist(neg, bins=bins, alpha=0.6, label=f"neg (n={len(neg)})", density=False)
    if len(pos) > 0:
        plt.hist(pos, bins=bins, alpha=0.6, label=f"pos (n={len(pos)})", density=False)

    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_threshold_curve(curve_rows, save_csv_path, save_png_path, title):
    df = pd.DataFrame(curve_rows)
    df.to_csv(save_csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["f1"], label="F1", marker="o", markersize=2)
    plt.plot(df["threshold"], df["precision"], label="Precision", alpha=0.8)
    plt.plot(df["threshold"], df["recall"], label="Recall", alpha=0.8)
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_png_path, dpi=200)
    plt.close()


def create_training_visualizations(history_df: pd.DataFrame, output_dir: str):
    epochs = history_df["epoch"].tolist()

    plot_curve(
        epochs,
        {
            "train_loss": history_df["train_loss"].tolist(),
            "val_loss": history_df["val_loss_05"].tolist(),
        },
        xlabel="Epoch",
        ylabel="Loss",
        title="Train / Val Loss",
        save_path=os.path.join(output_dir, "curve_loss.png"),
    )

    plot_curve(
        epochs,
        {
            "train_auc@0.5": history_df["train_auc_05"].tolist(),
            "val_auc": history_df["val_auc_05"].tolist(),
        },
        xlabel="Epoch",
        ylabel="AUC",
        title="Train / Val AUC",
        save_path=os.path.join(output_dir, "curve_auc.png"),
    )

    plot_curve(
        epochs,
        {
            "train_f1@0.5": history_df["train_f1_05"].tolist(),
            "val_f1@0.5": history_df["val_f1_05"].tolist(),
            "val_f1@best_thr": history_df["val_f1_bestthr"].tolist(),
        },
        xlabel="Epoch",
        ylabel="F1",
        title="Train / Val F1",
        save_path=os.path.join(output_dir, "curve_f1.png"),
    )

    plot_curve(
        epochs,
        {
            "best_val_threshold": history_df["val_best_threshold"].tolist(),
        },
        xlabel="Epoch",
        ylabel="Threshold",
        title="Best Validation Threshold by Epoch",
        save_path=os.path.join(output_dir, "curve_best_threshold.png"),
    )

    gap_auc = (history_df["train_auc_05"] - history_df["val_auc_05"]).tolist()
    gap_loss = (history_df["val_loss_05"] - history_df["train_loss"]).tolist()
    plot_curve(
        epochs,
        {
            "auc_gap(train-val)": gap_auc,
            "loss_gap(val-train)": gap_loss,
        },
        xlabel="Epoch",
        ylabel="Gap",
        title="Generalization Gap",
        save_path=os.path.join(output_dir, "curve_generalization_gap.png"),
    )


# =========================================================
# argparser
# =========================================================
def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--h5_root", type=str, default=None)
    parser.add_argument("--virchow2_ckpt", type=str, required=True)
    parser.add_argument("--abmil_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--mode", type=str, choices=["freeze", "lora", "full"], default="lora")

    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", nargs="+", default=["qkv", "proj", "fc1", "fc2"])
    parser.add_argument("--lora_last_k_blocks", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr_encoder", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", type=to_bool, default=True)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--monitor",
        type=str,
        default="auc",
        choices=["auc", "f1", "acc", "loss", "f1_bestthr", "recall_bestthr", "precision_bestthr"],
    )

    parser.add_argument("--bag_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--random_sample", type=to_bool, default=True)

    parser.add_argument("--train_subset_size_per_epoch", type=int, default=0)
    parser.add_argument("--train_subset_stratified", type=to_bool, default=True)
    parser.add_argument("--max_val_steps", type=int, default=0)

    parser.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "recall", "precision", "acc"])
    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_step", type=float, default=0.01)

    return parser


# =========================================================
# main
# =========================================================
def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_records(args.csv_path, h5_root=args.h5_root)
    train_df, val_df, test_df = split_records(
        df,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"[Data] total={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("[Data] train label counts:", train_df["label"].value_counts().to_dict())
    print("[Data] val label counts:", val_df["label"].value_counts().to_dict())
    if len(test_df) > 0:
        print("[Data] test label counts:", test_df["label"].value_counts().to_dict())

    transform = create_transform(
        input_size=(3, 224, 224),
        interpolation="bicubic",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        crop_pct=1.0,
        is_training=True,
    )
    eval_transform = create_transform(
        input_size=(3, 224, 224),
        interpolation="bicubic",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        crop_pct=1.0,
        is_training=False,
    )

    train_set = WSIPatchBagDataset(
        df_to_records(train_df),
        transform=transform,
        bag_size=args.bag_size,
        patch_size=args.patch_size,
        patch_level=args.patch_level,
        random_sample=args.random_sample,
    )
    val_set = WSIPatchBagDataset(
        df_to_records(val_df),
        transform=eval_transform,
        bag_size=args.bag_size,
        patch_size=args.patch_size,
        patch_level=args.patch_level,
        random_sample=False,
    )

    test_set = None
    if len(test_df) > 0:
        test_set = WSIPatchBagDataset(
            df_to_records(test_df),
            transform=eval_transform,
            bag_size=args.bag_size,
            patch_size=args.patch_size,
            patch_level=args.patch_level,
            random_sample=False,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_bag,
        persistent_workers=(max(1, args.num_workers // 2) > 0),
    )

    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            collate_fn=collate_bag,
            persistent_workers=(max(1, args.num_workers // 2) > 0),
        )

    encoder = Virchow2Backbone(pretrained_ckpt=args.virchow2_ckpt)

    lora_layers = []
    if args.mode == "lora":
        num_blocks = len(encoder.model.blocks)
        block_ids = list(range(max(0, num_blocks - args.lora_last_k_blocks), num_blocks))
        lora_layers = inject_lora(
            encoder.model,
            target_keywords=tuple(args.lora_targets),
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            block_ids=block_ids,
        )
        print(f"[LoRA] injected {len(lora_layers)} layers")
        for n in lora_layers[:30]:
            print("   ", n)

    abmil = load_frozen_abmil(args.abmil_ckpt, feat_dim=encoder.out_dim, device=device)
    abmil = abmil.to(device)
    abmil.eval()
    for p in abmil.parameters():
        p.requires_grad = False

    model = EncoderWithFrozenABMIL(
        encoder=encoder,
        abmil=abmil,
    )
    configure_train_mode(model, args.mode)
    model = model.to(device)

    total_params, trainable_params = count_trainable_params(model)
    print(f"[Params] total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M")

    enc_params = [p for p in model.parameters() if p.requires_grad]
    if len(enc_params) > 0:
        optimizer = torch.optim.AdamW(
            [{"params": enc_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay}]
        )
    else:
        optimizer = None
        print("[Warning] No trainable parameters. In freeze mode this becomes evaluation-only.")

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    cfg = vars(args).copy()
    cfg["lora_injected_layers"] = lora_layers
    save_json(cfg, os.path.join(args.output_dir, "config.json"))

    history = []
    best_state = None
    best_score = -1e18 if args.monitor != "loss" else 1e18
    bad_epochs = 0

    print(
        "[Train subset] "
        f"subset_size_per_epoch={args.train_subset_size_per_epoch} | "
        f"stratified={args.train_subset_stratified}"
    )

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        train_loader, epoch_subset_indices = make_epoch_train_loader(
            train_set=train_set,
            args=args,
            epoch=epoch,
        )

        epoch_labels = [train_set.records[i].label for i in epoch_subset_indices]
        epoch_label_counts = pd.Series(epoch_labels).value_counts().sort_index().to_dict() if len(epoch_labels) > 0 else {}

        print(
            f"[Epoch {epoch:03d}] sampled_train_slides={len(epoch_subset_indices)} "
            f"label_counts={epoch_label_counts}"
        )

        train_out = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            amp=args.amp and device.type == "cuda",
            grad_accum=args.grad_accum,
        )

        val_metrics_05, val_y, val_prob, val_slide_ids = evaluate_collect(
            model,
            val_loader,
            device,
            amp=args.amp and device.type == "cuda",
            threshold=0.5,
            max_steps=(None if args.max_val_steps <= 0 else args.max_val_steps),
        )

        val_best_threshold, val_best_metrics, val_thr_curve = find_best_threshold(
            val_y,
            val_prob,
            optimize_metric=args.threshold_metric,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "train_num_steps": train_out["num_train_steps"],
            "train_num_samples": train_out["num_train_samples"],
            "train_auc_05": train_out["train_auc_05"],
            "train_acc_05": train_out["train_acc_05"],
            "train_f1_05": train_out["train_f1_05"],
            "train_precision_05": train_out["train_precision_05"],
            "train_recall_05": train_out["train_recall_05"],
            "train_pos_rate_05": train_out["train_pos_rate_05"],
            "train_subset_num_slides": len(epoch_subset_indices),
            **{f"train_subset_label_{k}": v for k, v in epoch_label_counts.items()},
            "val_loss_05": val_metrics_05["loss"],
            "val_auc_05": val_metrics_05["auc"],
            "val_acc_05": val_metrics_05["acc"],
            "val_f1_05": val_metrics_05["f1"],
            "val_precision_05": val_metrics_05["precision"],
            "val_recall_05": val_metrics_05["recall"],
            "val_pos_rate_05": val_metrics_05["pos_rate"],
            "val_num_steps": val_metrics_05["num_eval_steps"],
            "val_num_samples": val_metrics_05["num_eval_samples"],
            "val_best_threshold": val_best_threshold,
            "val_acc_bestthr": val_best_metrics["acc"],
            "val_f1_bestthr": val_best_metrics["f1"],
            "val_precision_bestthr": val_best_metrics["precision"],
            "val_recall_bestthr": val_best_metrics["recall"],
            "val_auc_bestthr": val_best_metrics["auc"],
            "val_pos_rate_bestthr": val_best_metrics["pos_rate"],
            "time_sec": time.time() - t0,
        }

        history.append(record)
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(args.output_dir, "history.csv"), index=False)
        create_training_visualizations(history_df, args.output_dir)

        thr_curve_path = os.path.join(args.output_dir, f"val_threshold_curve_epoch{epoch:03d}.csv")
        thr_curve_png = os.path.join(args.output_dir, f"val_threshold_curve_epoch{epoch:03d}.png")
        save_threshold_curve(
            val_thr_curve,
            save_csv_path=thr_curve_path,
            save_png_path=thr_curve_png,
            title=f"Validation Threshold Curve - Epoch {epoch:03d}",
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_out['loss']:.4f} "
            f"train_auc@0.5={train_out['train_auc_05']:.4f} "
            f"train_f1@0.5={train_out['train_f1_05']:.4f} "
            f"val_loss@0.5={val_metrics_05['loss']:.4f} "
            f"val_auc={val_metrics_05['auc']:.4f} "
            f"val_f1@0.5={val_metrics_05['f1']:.4f} "
            f"val_best_thr={val_best_threshold:.2f} "
            f"val_{args.threshold_metric}@bestthr={val_best_metrics[args.threshold_metric]:.4f}"
        )

        if args.monitor == "loss":
            score = val_metrics_05["loss"]
            improved = score < best_score
        elif args.monitor == "auc":
            score = val_metrics_05["auc"]
            improved = score > best_score
        elif args.monitor == "f1":
            score = val_metrics_05["f1"]
            improved = score > best_score
        elif args.monitor == "acc":
            score = val_metrics_05["acc"]
            improved = score > best_score
        elif args.monitor == "f1_bestthr":
            score = val_best_metrics["f1"]
            improved = score > best_score
        elif args.monitor == "recall_bestthr":
            score = val_best_metrics["recall"]
            improved = score > best_score
        elif args.monitor == "precision_bestthr":
            score = val_best_metrics["precision"]
            improved = score > best_score
        else:
            raise ValueError(f"Unsupported monitor: {args.monitor}")

        if improved:
            best_score = score
            bad_epochs = 0
            best_state = {
                "epoch": epoch,
                "model": copy.deepcopy(model.state_dict()),
                "encoder": copy.deepcopy(model.encoder.state_dict()),
                "optimizer": None if optimizer is None else optimizer.state_dict(),
                "monitor": args.monitor,
                "best_score": best_score,
                "best_threshold": val_best_threshold,
                "val_metrics_05": val_metrics_05,
                "val_metrics_bestthr": val_best_metrics,
            }
            torch.save(best_state, os.path.join(args.output_dir, "best.pt"))
            torch.save(best_state["encoder"], os.path.join(args.output_dir, "best_encoder_state_dict.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[Early Stop] no improvement for {args.patience} epochs")
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint saved.")

    model.load_state_dict(best_state["model"])
    best_threshold = float(best_state["best_threshold"])
    print(f"[Best] epoch={best_state['epoch']} monitor={best_state['monitor']} best_score={best_state['best_score']:.4f}")
    print(f"[Best] selected threshold from val = {best_threshold:.2f}")
    print(f"[Best] encoder weights saved to: {os.path.join(args.output_dir, 'best_encoder_state_dict.pt')}")

    val_metrics_05, val_y, val_prob, val_slide_ids = evaluate_collect(
        model,
        val_loader,
        device,
        amp=args.amp and device.type == "cuda",
        threshold=0.5,
        max_steps=(None if args.max_val_steps <= 0 else args.max_val_steps),
    )
    val_metrics_best = binary_metrics_from_probs(val_y, val_prob, threshold=best_threshold)
    val_metrics_best["loss"] = val_metrics_05["loss"]
    val_metrics_best["num_eval_steps"] = val_metrics_05["num_eval_steps"]
    val_metrics_best["num_eval_samples"] = val_metrics_05["num_eval_samples"]

    save_json(val_metrics_05, os.path.join(args.output_dir, "best_val_metrics_threshold_0p5.json"))
    save_json(val_metrics_best, os.path.join(args.output_dir, "best_val_metrics_best_threshold.json"))
    save_prediction_csv(
        os.path.join(args.output_dir, "best_val_predictions.csv"),
        val_slide_ids, val_y, val_prob, best_threshold
    )
    plot_prob_hist(
        val_y, val_prob,
        save_path=os.path.join(args.output_dir, "best_val_prob_hist.png"),
        title=f"Validation Probability Histogram (best_thr={best_threshold:.2f})",
    )

    print("[Best Val @0.5]", val_metrics_05)
    print("[Best Val @best_threshold]", val_metrics_best)

    if test_loader is not None:
        test_metrics_05, test_y, test_prob, test_slide_ids = evaluate_collect(
            model,
            test_loader,
            device,
            amp=args.amp and device.type == "cuda",
            threshold=0.5,
            max_steps=None,
        )
        test_metrics_best = binary_metrics_from_probs(test_y, test_prob, threshold=best_threshold)
        test_metrics_best["loss"] = test_metrics_05["loss"]
        test_metrics_best["num_eval_steps"] = test_metrics_05["num_eval_steps"]
        test_metrics_best["num_eval_samples"] = test_metrics_05["num_eval_samples"]

        save_json(test_metrics_05, os.path.join(args.output_dir, "test_metrics_threshold_0p5.json"))
        save_json(test_metrics_best, os.path.join(args.output_dir, "test_metrics_best_threshold.json"))
        save_prediction_csv(
            os.path.join(args.output_dir, "test_predictions.csv"),
            test_slide_ids, test_y, test_prob, best_threshold
        )
        plot_prob_hist(
            test_y, test_prob,
            save_path=os.path.join(args.output_dir, "test_prob_hist.png"),
            title=f"Test Probability Histogram (best_thr={best_threshold:.2f})",
        )

        print("[Test @0.5]", test_metrics_05)
        print("[Test @best_threshold_from_val]", test_metrics_best)

    print(f"[Done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()