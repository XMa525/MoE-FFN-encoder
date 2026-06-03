#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
from torchmil.models import ABMIL, DSMIL

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.plugins.shared_role_prototype import SharedRolePrototype, PatchRoleSummaryFromSharedProto
from models.plugins.plugin_losses import compute_role_proto_anchor_loss

try:
    from models.plugins.role_aware_tail_plugin import RoleAwareTailWithSharedSummary as PluginClass
except Exception:
    try:
        from models.plugins.role_aware_tail_plugin import RoleAwareTailPlugin as PluginClass
    except Exception as e:
        raise ImportError(
            "Cannot import plugin class from models.plugins.role_aware_tail_plugin. "
            "Expected RoleAwareTailWithSharedSummary or RoleAwareTailPlugin."
        ) from e


# =========================================================
# utils
# =========================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


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


def try_compute_auc(labels: List[int], probs: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, probs))
    except Exception:
        return float("nan")


# =========================================================
# dataset
# =========================================================
class CachedSlideBagDataset(Dataset):
    """
    cache_index_csv minimally needs:
        - cache_path
        - label
        - slide_id
    optional:
        - project

    each cache file minimally needs:
        - patch_feat_raw   [D]
    optional:
        - patch_role_probs
        - patch_role_gaps
        - patch_top1_gap
        - patch_role_logits
        - patch_feat_teacher_space
    """

    def __init__(
        self,
        cache_index_csv: str,
        max_patches_per_slide: int = 256,
        random_sample_patches: bool = True,
        seed: int = 42,
    ):
        if not os.path.exists(cache_index_csv):
            raise FileNotFoundError(f"cache_index_csv not found: {cache_index_csv}")

        df = pd.read_csv(cache_index_csv)
        need = ["cache_path", "label", "slide_id"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"cache_index_csv must contain columns: {miss}")
        if len(df) == 0:
            raise ValueError(f"empty cache index: {cache_index_csv}")

        self.df = df.copy()
        self.max_patches_per_slide = int(max_patches_per_slide)
        self.random_sample_patches = bool(random_sample_patches)
        self.seed = int(seed)

        self.slide_to_indices: Dict[str, List[int]] = {}
        self.slide_labels: Dict[str, int] = {}
        self.slide_projects: Dict[str, str] = {}

        for i, row in self.df.iterrows():
            sid = str(row["slide_id"])
            self.slide_to_indices.setdefault(sid, []).append(int(i))
            self.slide_labels[sid] = int(row["label"])
            self.slide_projects[sid] = str(row["project"]) if "project" in row and pd.notna(row["project"]) else ""

        self.slide_ids = sorted(self.slide_to_indices.keys())
        print(
            f"[CachedSlideBagDataset] slides={len(self.slide_ids)} "
            f"patches={len(self.df)} "
            f"max_patches_per_slide={self.max_patches_per_slide}"
        )

    def __len__(self):
        return len(self.slide_ids)

    def _pick_patch_indices(self, slide_id: str, idx: int) -> List[int]:
        patch_indices = self.slide_to_indices[slide_id]
        if len(patch_indices) <= self.max_patches_per_slide:
            return list(patch_indices)

        if not self.random_sample_patches:
            return list(patch_indices[: self.max_patches_per_slide])

        rng = random.Random(self.seed + idx)
        return rng.sample(patch_indices, self.max_patches_per_slide)

    def __getitem__(self, idx: int):
        slide_id = self.slide_ids[idx]
        patch_indices = self._pick_patch_indices(slide_id, idx)

        patch_feat_raw = []
        patch_role_probs = []
        patch_role_gaps = []
        patch_top1_gap = []
        patch_role_logits = []
        patch_feat_teacher_space = []

        has_probs = True
        has_gaps = True
        has_top1 = True
        has_logits = True
        has_teacher = True

        cache_paths = []

        for row_idx in patch_indices:
            row = self.df.iloc[row_idx]
            obj = torch.load(row["cache_path"], map_location="cpu")
            cache_paths.append(str(row["cache_path"]))

            if "patch_feat_raw" not in obj:
                raise KeyError(f"{row['cache_path']} missing patch_feat_raw")
            patch_feat_raw.append(obj["patch_feat_raw"].float())

            if "patch_role_probs" in obj:
                patch_role_probs.append(obj["patch_role_probs"].float())
            else:
                has_probs = False

            if "patch_role_gaps" in obj:
                patch_role_gaps.append(obj["patch_role_gaps"].float())
            else:
                has_gaps = False

            if "patch_top1_gap" in obj:
                v = obj["patch_top1_gap"].float()
                if v.ndim == 0:
                    v = v.view(1)
                patch_top1_gap.append(v)
            else:
                has_top1 = False

            if "patch_role_logits" in obj:
                patch_role_logits.append(obj["patch_role_logits"].float())
            else:
                has_logits = False

            if "patch_feat_teacher_space" in obj:
                patch_feat_teacher_space.append(obj["patch_feat_teacher_space"].float())
            else:
                has_teacher = False

        out = {
            "slide_id": slide_id,
            "label": self.slide_labels[slide_id],
            "project": self.slide_projects[slide_id],
            "cache_paths": cache_paths,
            "patch_feat_raw": torch.stack(patch_feat_raw, dim=0),   # [N, D]
        }

        if has_probs:
            out["patch_role_probs"] = torch.stack(patch_role_probs, dim=0)
        if has_gaps:
            out["patch_role_gaps"] = torch.stack(patch_role_gaps, dim=0)
        if has_top1:
            out["patch_top1_gap"] = torch.stack(patch_top1_gap, dim=0)
        if has_logits:
            out["patch_role_logits"] = torch.stack(patch_role_logits, dim=0)
        if has_teacher:
            out["patch_feat_teacher_space"] = torch.stack(patch_feat_teacher_space, dim=0)

        return out


def pad_and_stack_2d_tensors(
    xs: List[torch.Tensor],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xs: list of [N_i, D]
    returns:
        out: [B, Nmax, D]
        mask: [B, Nmax] bool
    """
    bsz = len(xs)
    max_len = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]

    out = xs[0].new_full((bsz, max_len, feat_dim), pad_value)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)

    for i, x in enumerate(xs):
        n = x.shape[0]
        out[i, :n] = x
        mask[i, :n] = True

    return out, mask


def cached_bag_collate_fn(batch: List[Dict]):
    patch_feat_raw, patch_mask = pad_and_stack_2d_tensors(
        [x["patch_feat_raw"] for x in batch], pad_value=0.0
    )

    out = {
        "patch_feat_raw": patch_feat_raw,          # [B, Nmax, D]
        "patch_mask": patch_mask,                  # [B, Nmax]
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "slide_id": [x["slide_id"] for x in batch],
        "project": [x["project"] for x in batch],
        "cache_paths": [x["cache_paths"] for x in batch],
    }

    optional_keys = [
        "patch_role_probs",
        "patch_role_gaps",
        "patch_top1_gap",
        "patch_role_logits",
        "patch_feat_teacher_space",
    ]
    for key in optional_keys:
        if all(key in x for x in batch):
            out[key], _ = pad_and_stack_2d_tensors([x[key] for x in batch], pad_value=0.0)

    return out


# =========================================================
# models
# =========================================================
class ABMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str):
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
        out = self.model(bag_feats)

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


class DSMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str):
        super().__init__()
        self.model = DSMIL(
            in_shape=(in_dim,),
            att_dim=128,
            nonlinear_q=False,
            nonlinear_v=False,
            dropout=0.0,
        )
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        if bag_feats.ndim == 2:
            bag_feats = bag_feats.unsqueeze(0)

        out = self.model(bag_feats)

        if isinstance(out, torch.Tensor):
            return out

        if isinstance(out, dict):
            for key in ["logits", "pred", "scores", "output", "Y_pred"]:
                if key in out:
                    return out[key]

        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported DSMIL output type: {type(out)}")


# =========================================================
# helpers
# =========================================================
def build_role_summary_from_batch(
    batch: Dict[str, torch.Tensor],
    summary_builder: PatchRoleSummaryFromSharedProto,
    device: str,
    use_role_logits: bool,
):
    role_probs = batch.get("patch_role_probs", None)
    role_gaps = batch.get("patch_role_gaps", None)
    top1_gap = batch.get("patch_top1_gap", None)
    role_logits = batch.get("patch_role_logits", None)

    if role_probs is not None and role_gaps is not None and top1_gap is not None:
        out = {
            "patch_role_probs": role_probs.to(device),
            "patch_role_gaps": role_gaps.to(device),
            "patch_top1_gap": top1_gap.to(device),
        }
        if use_role_logits and role_logits is not None:
            out["patch_role_logits"] = role_logits.to(device)
        elif use_role_logits:
            out["patch_role_logits"] = None
        return out

    teacher_space = batch.get("patch_feat_teacher_space", None)
    if teacher_space is None:
        raise ValueError(
            "Batch has neither cached role summary nor patch_feat_teacher_space fallback. "
            "Please cache patch_role_probs/patch_role_gaps/patch_top1_gap, "
            "or cache patch_feat_teacher_space."
        )

    teacher_space = teacher_space.to(device)
    role_dict = summary_builder(teacher_space)

    return {
        "patch_role_probs": role_dict["patch_role_probs"],
        "patch_role_gaps": role_dict["patch_role_gaps"],
        "patch_top1_gap": role_dict["patch_top1_gap"],
        "patch_role_logits": role_dict["patch_role_logits"] if use_role_logits else None,
    }


def feature_preserve_loss(
    feat_raw: torch.Tensor,
    feat_plugin: torch.Tensor,
    patch_mask: torch.Tensor,
    mode: str = "l2",
):
    mask = patch_mask.unsqueeze(-1).float()
    if mode == "l2":
        diff = ((feat_plugin - feat_raw) ** 2) * mask
        denom = mask.sum().clamp_min(1.0) * feat_raw.shape[-1]
        return diff.sum() / denom
    elif mode == "cosine":
        raw = F.normalize(feat_raw, dim=-1)
        plugin = F.normalize(feat_plugin, dim=-1)
        cos = (raw * plugin).sum(dim=-1)
        loss = (1.0 - cos) * patch_mask.float()
        return loss.sum() / patch_mask.float().sum().clamp_min(1.0)
    else:
        raise ValueError(f"Unsupported preserve mode: {mode}")


def normalize_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits

    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]

    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:, 1]

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits = normalize_binary_logits(logits)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    acc = (preds == labels).float().mean()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    spec = tn / max(tn + fp, 1)

    auc = try_compute_auc(labels.detach().cpu().tolist(), probs.detach().cpu().tolist())
    return {
        "acc": safe_float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "sens": float(recall),
        "spec": float(spec),
    }


def build_aggregator(mil_model: str, in_dim: int, device: str):
    if mil_model == "abmil":
        return ABMILWrapper(in_dim=in_dim, device=device)
    elif mil_model == "dsmil":
        return DSMILWrapper(in_dim=in_dim, device=device)
    else:
        raise ValueError(f"Unsupported mil_model: {mil_model}")


def run_mil_per_bag(
    aggregator: nn.Module,
    patch_feat: torch.Tensor,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    bag_logits_list = []
    for i in range(patch_feat.shape[0]):
        feats_i = patch_feat[i][patch_mask[i]]   # [Ni, D]
        logits_i = aggregator(feats_i.unsqueeze(0))
        logits_i = normalize_binary_logits(logits_i)
        bag_logits_list.append(logits_i.view(1))
    return torch.cat(bag_logits_list, dim=0)   # [B]


@torch.no_grad()
def evaluate_plugin(
    plugin: nn.Module,
    aggregator: nn.Module,
    shared_role_proto: SharedRolePrototype,
    summary_builder: PatchRoleSummaryFromSharedProto,
    loader: DataLoader,
    device: str,
    use_role_logits: bool,
    feat_preserve_weight: float,
    proto_anchor_weight: float,
    preserve_mode: str,
):
    plugin.eval()
    aggregator.eval()
    if shared_role_proto.prototypes.requires_grad:
        shared_role_proto.eval()

    losses = []
    logits_all = []
    labels_all = []

    for batch in tqdm(loader, desc="Eval(plugin)", leave=False):
        patch_feat_raw = batch["patch_feat_raw"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels = batch["labels"].to(device).float()

        role_summary = build_role_summary_from_batch(
            batch=batch,
            summary_builder=summary_builder,
            device=device,
            use_role_logits=use_role_logits,
        )

        plugin_out = plugin(
            patch_feat=patch_feat_raw,
            patch_role_probs=role_summary["patch_role_probs"],
            patch_role_gaps=role_summary["patch_role_gaps"],
            patch_role_logits=role_summary["patch_role_logits"],
            patch_top1_gap=role_summary["patch_top1_gap"],
            return_aux=True,
        )
        if isinstance(plugin_out, tuple):
            patch_feat_plugin, _ = plugin_out
        else:
            patch_feat_plugin = plugin_out

        bag_logits = run_mil_per_bag(
            aggregator=aggregator,
            patch_feat=patch_feat_plugin,
            patch_mask=patch_mask,
        )

        loss_cls = F.binary_cross_entropy_with_logits(bag_logits, labels)
        loss_preserve = feature_preserve_loss(
            feat_raw=patch_feat_raw,
            feat_plugin=patch_feat_plugin,
            patch_mask=patch_mask,
            mode=preserve_mode,
        )
        total = loss_cls + feat_preserve_weight * loss_preserve

        if shared_role_proto.prototypes.requires_grad and proto_anchor_weight > 0:
            total = total + proto_anchor_weight * compute_role_proto_anchor_loss(
                current_proto=shared_role_proto.get_prototypes(),
                init_proto=shared_role_proto.get_init_prototypes(),
                normalize=False,
                mode="cosine",
            )

        losses.append(safe_float(total))
        logits_all.append(bag_logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
    labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)

    metric_dict = compute_metrics_from_logits(logits_all, labels_all) if len(labels_all) > 0 else {
        "acc": 0.0, "f1": 0.0, "auc": float("nan"), "sens": 0.0, "spec": 0.0
    }

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **metric_dict,
    }


@torch.no_grad()
def evaluate_no_plugin(
    aggregator: nn.Module,
    loader: DataLoader,
    device: str,
):
    aggregator.eval()

    losses = []
    logits_all = []
    labels_all = []

    for batch in tqdm(loader, desc="Eval(no-plugin)", leave=False):
        patch_feat_raw = batch["patch_feat_raw"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels = batch["labels"].to(device).float()

        bag_logits = run_mil_per_bag(
            aggregator=aggregator,
            patch_feat=patch_feat_raw,
            patch_mask=patch_mask,
        )

        loss = F.binary_cross_entropy_with_logits(bag_logits, labels)

        losses.append(safe_float(loss))
        logits_all.append(bag_logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
    labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)

    metric_dict = compute_metrics_from_logits(logits_all, labels_all) if len(labels_all) > 0 else {
        "acc": 0.0, "f1": 0.0, "auc": float("nan"), "sens": 0.0, "spec": 0.0
    }

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **metric_dict,
    }

@torch.no_grad()
def evaluate_plugin_with_predictions(
    plugin: nn.Module,
    aggregator: nn.Module,
    shared_role_proto: SharedRolePrototype,
    summary_builder: PatchRoleSummaryFromSharedProto,
    loader: DataLoader,
    device: str,
    use_role_logits: bool,
    feat_preserve_weight: float,
    proto_anchor_weight: float,
    preserve_mode: str,
):
    plugin.eval()
    aggregator.eval()
    if shared_role_proto.prototypes.requires_grad:
        shared_role_proto.eval()

    losses = []
    logits_all = []
    labels_all = []
    slide_ids_all = []

    for batch in tqdm(loader, desc="Eval(plugin,preds)", leave=False):
        patch_feat_raw = batch["patch_feat_raw"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels = batch["labels"].to(device).float()
        slide_ids = batch["slide_id"]

        role_summary = build_role_summary_from_batch(
            batch=batch,
            summary_builder=summary_builder,
            device=device,
            use_role_logits=use_role_logits,
        )

        plugin_out = plugin(
            patch_feat=patch_feat_raw,
            patch_role_probs=role_summary["patch_role_probs"],
            patch_role_gaps=role_summary["patch_role_gaps"],
            patch_role_logits=role_summary["patch_role_logits"],
            patch_top1_gap=role_summary["patch_top1_gap"],
            return_aux=True,
        )
        if isinstance(plugin_out, tuple):
            patch_feat_plugin, _ = plugin_out
        else:
            patch_feat_plugin = plugin_out

        bag_logits = run_mil_per_bag(
            aggregator=aggregator,
            patch_feat=patch_feat_plugin,
            patch_mask=patch_mask,
        )

        loss_cls = F.binary_cross_entropy_with_logits(bag_logits, labels)
        loss_preserve = feature_preserve_loss(
            feat_raw=patch_feat_raw,
            feat_plugin=patch_feat_plugin,
            patch_mask=patch_mask,
            mode=preserve_mode,
        )
        total = loss_cls + feat_preserve_weight * loss_preserve

        if shared_role_proto.prototypes.requires_grad and proto_anchor_weight > 0:
            total = total + proto_anchor_weight * compute_role_proto_anchor_loss(
                current_proto=shared_role_proto.get_prototypes(),
                init_proto=shared_role_proto.get_init_prototypes(),
                normalize=False,
                mode="cosine",
            )

        losses.append(safe_float(total))
        logits_all.append(bag_logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        slide_ids_all.extend(slide_ids)

    logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
    labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)

    metric_dict = compute_metrics_from_logits(logits_all, labels_all) if len(labels_all) > 0 else {
        "acc": 0.0, "f1": 0.0, "auc": float("nan"), "sens": 0.0, "spec": 0.0
    }

    probs_all = torch.sigmoid(normalize_binary_logits(logits_all)).numpy() if len(logits_all) > 0 else np.array([])
    pred_df = pd.DataFrame({
        "slide_id": slide_ids_all,
        "y_true": labels_all.numpy() if len(labels_all) > 0 else [],
        "y_prob": probs_all,
    })

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **metric_dict,
    }, pred_df


@torch.no_grad()
def evaluate_no_plugin_with_predictions(
    aggregator: nn.Module,
    loader: DataLoader,
    device: str,
):
    aggregator.eval()

    losses = []
    logits_all = []
    labels_all = []
    slide_ids_all = []

    for batch in tqdm(loader, desc="Eval(no-plugin,preds)", leave=False):
        patch_feat_raw = batch["patch_feat_raw"].to(device)
        patch_mask = batch["patch_mask"].to(device)
        labels = batch["labels"].to(device).float()
        slide_ids = batch["slide_id"]

        bag_logits = run_mil_per_bag(
            aggregator=aggregator,
            patch_feat=patch_feat_raw,
            patch_mask=patch_mask,
        )

        loss = F.binary_cross_entropy_with_logits(bag_logits, labels)

        losses.append(safe_float(loss))
        logits_all.append(bag_logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        slide_ids_all.extend(slide_ids)

    logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
    labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)

    metric_dict = compute_metrics_from_logits(logits_all, labels_all) if len(labels_all) > 0 else {
        "acc": 0.0, "f1": 0.0, "auc": float("nan"), "sens": 0.0, "spec": 0.0
    }

    probs_all = torch.sigmoid(normalize_binary_logits(logits_all)).numpy() if len(logits_all) > 0 else np.array([])
    pred_df = pd.DataFrame({
        "slide_id": slide_ids_all,
        "y_true": labels_all.numpy() if len(labels_all) > 0 else [],
        "y_prob": probs_all,
    })

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **metric_dict,
    }, pred_df

def train_no_plugin_baseline(
    args,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
):
    print("\n========== Train no-plugin baseline ==========")

    aggregator = build_aggregator(
        mil_model=args.mil_model,
        in_dim=args.plugin_feat_dim,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        aggregator.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_metric = -1e18
    best_ckpt = os.path.join(args.out_dir, "baseline_no_plugin_best.pt")

    for epoch in range(1, args.epochs + 1):
        aggregator.train()

        train_losses = []
        logits_all = []
        labels_all = []

        pbar = tqdm(train_loader, desc=f"Train no-plugin {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            patch_feat_raw = batch["patch_feat_raw"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            labels = batch["labels"].to(device).float()

            bag_logits = run_mil_per_bag(
                aggregator=aggregator,
                patch_feat=patch_feat_raw,
                patch_mask=patch_mask,
            )

            loss = F.binary_cross_entropy_with_logits(bag_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(safe_float(loss))
            logits_all.append(bag_logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

            pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
        labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)
        train_metric = compute_metrics_from_logits(logits_all, labels_all) if len(labels_all) > 0 else {
            "acc": 0.0, "f1": 0.0, "auc": float("nan"), "sens": 0.0, "spec": 0.0
        }

        val_stats, val_pred_df = evaluate_no_plugin_with_predictions(
            aggregator=aggregator,
            loader=val_loader,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "train_acc": train_metric["acc"],
            "train_f1": train_metric["f1"],
            "train_auc": train_metric["auc"],
            "train_sens": train_metric["sens"],
            "train_spec": train_metric["spec"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_f1": val_stats["f1"],
            "val_auc": val_stats["auc"],
            "val_sens": val_stats["sens"],
            "val_spec": val_stats["spec"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            os.path.join(args.out_dir, "baseline_no_plugin_train_history.csv"),
            index=False
        )

        print(
            f"[NoPlugin Epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_auc={row['train_auc']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_auc={row['val_auc']:.4f} "
            f"val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_f1']:.4f}"
        )

        cur_metric = row["val_auc"]
        if (math.isnan(cur_metric) and best_metric == -1e18) or (not math.isnan(cur_metric) and cur_metric > best_metric):
            best_metric = cur_metric
            torch.save(
                {
                    "epoch": epoch,
                    "aggregator_state_dict": aggregator.state_dict(),
                    "feat_dim": args.plugin_feat_dim,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"[NoPlugin Best] epoch={epoch}, val_auc={best_metric:.4f}")

    ckpt = torch.load(best_ckpt, map_location=device)
    aggregator.load_state_dict(ckpt["aggregator_state_dict"])

    
    test_stats, test_pred_df = evaluate_no_plugin_with_predictions(
        aggregator=aggregator,
        loader=test_loader,
        device=device,
    )
    test_pred_df.to_csv(
        os.path.join(args.out_dir, "baseline_no_plugin_test_predictions.csv"),
        index=False
    )
    with open(os.path.join(args.out_dir, "baseline_no_plugin_final_test_metrics.json"), "w") as f:
        json.dump({
            "best_val_auc": best_metric,
            "test_stats": test_stats,
        }, f, indent=2)
    val_pred_df.to_csv(
        os.path.join(args.out_dir, "baseline_no_plugin_best_val_predictions.csv"),
        index=False
    )
    print(
        f"[NoPlugin Test] "
        f"test_loss={test_stats['loss']:.4f} "
        f"test_auc={test_stats['auc']:.4f} "
        f"test_acc={test_stats['acc']:.4f} "
        f"test_f1={test_stats['f1']:.4f} "
        f"test_sens={test_stats['sens']:.4f} "
        f"test_spec={test_stats['spec']:.4f}"
    )

    return {
        "best_val_auc": best_metric,
        "test": test_stats,
    }


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Train plugin-v1 + official MIL on cached slide bags")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_cache_csv", type=str, required=None)
    parser.add_argument("--val_cache_csv", type=str, required=None)
    parser.add_argument("--test_cache_csv", type=str, required=None)
    parser.add_argument("--role_proto_dir", type=str, required=None)
    parser.add_argument("--out_dir", type=str, required=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--max_patches_per_slide", type=int, default=256)
    parser.add_argument("--random_sample_patches", action="store_true")

    parser.add_argument("--plugin_feat_dim", type=int, default=384)
    parser.add_argument("--plugin_hidden_dim", type=int, default=128)
    parser.add_argument("--plugin_dropout", type=float, default=0.0)
    parser.add_argument("--plugin_init_scale", type=float, default=0.05)

    parser.add_argument("--use_role_logits", action="store_true")
    parser.add_argument("--use_top1_gap", action="store_true")
    parser.add_argument("--use_beta", action="store_true")

    parser.add_argument("--shared_proto_learnable", action="store_true")
    parser.add_argument("--proto_anchor_weight", type=float, default=0.1)
    parser.add_argument("--role_tau", type=float, default=1.0)

    parser.add_argument("--mil_model", type=str, default="abmil", choices=["abmil", "dsmil"])
    parser.add_argument("--feat_preserve_weight", type=float, default=0.01)
    parser.add_argument("--preserve_mode", type=str, default="l2", choices=["l2", "cosine"])

    parser.add_argument("--run_no_plugin_baseline", action="store_true")

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    ensure_dir(args.out_dir)
    log_path = os.path.join(args.out_dir, "train.log")

    with open(log_path, "a", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with redirect_stdout(tee), redirect_stderr(tee):
            set_seed(args.seed)

            print("=" * 80)
            print("Start training")
            print(json.dumps(vars(args), indent=2, ensure_ascii=False))
            print("=" * 80)

            device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

            train_set = CachedSlideBagDataset(
                cache_index_csv=args.train_cache_csv,
                max_patches_per_slide=args.max_patches_per_slide,
                random_sample_patches=args.random_sample_patches,
                seed=args.seed,
            )
            val_set = CachedSlideBagDataset(
                cache_index_csv=args.val_cache_csv,
                max_patches_per_slide=args.max_patches_per_slide,
                random_sample_patches=False,
                seed=args.seed + 999,
            )
            test_set = CachedSlideBagDataset(
                cache_index_csv=args.test_cache_csv,
                max_patches_per_slide=args.max_patches_per_slide,
                random_sample_patches=False,
                seed=args.seed + 1999,
            )

            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=cached_bag_collate_fn,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=cached_bag_collate_fn,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=cached_bag_collate_fn,
            )

            shared_role_proto = SharedRolePrototype.from_files(
                role_proto_dir=args.role_proto_dir,
                normalize=True,
                learnable=args.shared_proto_learnable,
                device=device,
            )

            summary_builder = PatchRoleSummaryFromSharedProto(
                shared_role_proto=shared_role_proto,
                tau=args.role_tau,
                use_softmax=True,
            ).to(device)

            plugin = PluginClass(
                feat_dim=args.plugin_feat_dim,
                num_roles=shared_role_proto.num_roles,
                hidden_dim=args.plugin_hidden_dim,
                dropout=args.plugin_dropout,
                use_role_logits=args.use_role_logits,
                use_top1_gap=args.use_top1_gap,
                use_beta=args.use_beta,
                init_scale=args.plugin_init_scale,
            ).to(device)

            aggregator = build_aggregator(
                mil_model=args.mil_model,
                in_dim=args.plugin_feat_dim,
                device=str(device),
            )

            params = list(plugin.parameters()) + list(aggregator.parameters())
            if args.shared_proto_learnable:
                params += [p for p in shared_role_proto.parameters() if p.requires_grad]

            optimizer = torch.optim.AdamW(
                params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            history = []
            best_metric = -1e18
            best_ckpt = os.path.join(args.out_dir, "best_plugin_cached_bag.pt")

            # ------------------- plugin train -------------------
            for epoch in range(1, args.epochs + 1):
                plugin.train()
                aggregator.train()
                if args.shared_proto_learnable:
                    shared_role_proto.train()
                else:
                    shared_role_proto.eval()

                train_losses = []
                logits_all = []
                labels_all = []

                pbar = tqdm(train_loader, desc=f"Train plugin {epoch}/{args.epochs}", leave=False)
                for batch in pbar:
                    patch_feat_raw = batch["patch_feat_raw"].to(device)
                    patch_mask = batch["patch_mask"].to(device)
                    labels = batch["labels"].to(device).float()

                    role_summary = build_role_summary_from_batch(
                        batch=batch,
                        summary_builder=summary_builder,
                        device=device,
                        use_role_logits=args.use_role_logits,
                    )

                    plugin_out = plugin(
                        patch_feat=patch_feat_raw,
                        patch_role_probs=role_summary["patch_role_probs"],
                        patch_role_gaps=role_summary["patch_role_gaps"],
                        patch_role_logits=role_summary["patch_role_logits"],
                        patch_top1_gap=role_summary["patch_top1_gap"],
                        return_aux=True,
                    )
                    if isinstance(plugin_out, tuple):
                        patch_feat_plugin, _ = plugin_out
                    else:
                        patch_feat_plugin = plugin_out

                    bag_logits = run_mil_per_bag(
                        aggregator=aggregator,
                        patch_feat=patch_feat_plugin,
                        patch_mask=patch_mask,
                    )

                    loss_cls = F.binary_cross_entropy_with_logits(bag_logits, labels)
                    loss_preserve = feature_preserve_loss(
                        feat_raw=patch_feat_raw,
                        feat_plugin=patch_feat_plugin,
                        patch_mask=patch_mask,
                        mode=args.preserve_mode,
                    )
                    loss = loss_cls + args.feat_preserve_weight * loss_preserve

                    if args.shared_proto_learnable and args.proto_anchor_weight > 0:
                        loss = loss + args.proto_anchor_weight * compute_role_proto_anchor_loss(
                            current_proto=shared_role_proto.get_prototypes(),
                            init_proto=shared_role_proto.get_init_prototypes(),
                            normalize=False,
                            mode="cosine",
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_losses.append(safe_float(loss))
                    logits_all.append(bag_logits.detach().cpu())
                    labels_all.append(labels.detach().cpu())

                    pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

                logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
                labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)
                train_metric = compute_metrics_from_logits(logits_all, labels_all) if len(labels_all) > 0 else {
                    "acc": 0.0, "f1": 0.0, "auc": float("nan"), "sens": 0.0, "spec": 0.0
                }

                val_stats, val_pred_df = evaluate_plugin_with_predictions(
                    plugin=plugin,
                    aggregator=aggregator,
                    shared_role_proto=shared_role_proto,
                    summary_builder=summary_builder,
                    loader=val_loader,
                    device=device,
                    use_role_logits=args.use_role_logits,
                    feat_preserve_weight=args.feat_preserve_weight,
                    proto_anchor_weight=args.proto_anchor_weight,
                    preserve_mode=args.preserve_mode,
                )

                row = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                    "train_acc": train_metric["acc"],
                    "train_f1": train_metric["f1"],
                    "train_auc": train_metric["auc"],
                    "train_sens": train_metric["sens"],
                    "train_spec": train_metric["spec"],
                    "val_loss": val_stats["loss"],
                    "val_acc": val_stats["acc"],
                    "val_f1": val_stats["f1"],
                    "val_auc": val_stats["auc"],
                    "val_sens": val_stats["sens"],
                    "val_spec": val_stats["spec"],
                }
                history.append(row)
                pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "train_history.csv"), index=False)

                print(
                    f"[Plugin Epoch {epoch:03d}] "
                    f"train_loss={row['train_loss']:.4f} "
                    f"train_auc={row['train_auc']:.4f} "
                    f"val_loss={row['val_loss']:.4f} "
                    f"val_auc={row['val_auc']:.4f} "
                    f"val_acc={row['val_acc']:.4f} "
                    f"val_f1={row['val_f1']:.4f}"
                )

                cur_metric = row["val_auc"]
                if (math.isnan(cur_metric) and best_metric == -1e18) or (not math.isnan(cur_metric) and cur_metric > best_metric):
                    best_metric = cur_metric
                    torch.save(
                        {
                            "epoch": epoch,
                            "plugin_state_dict": plugin.state_dict(),
                            "aggregator_state_dict": aggregator.state_dict(),
                            "shared_role_proto_state_dict": shared_role_proto.state_dict(),
                            "role_names": shared_role_proto.role_names,
                            "feat_dim": args.plugin_feat_dim,
                            "args": vars(args),
                        },
                        best_ckpt,
                    )
                    print(f"[Plugin Best] epoch={epoch}, val_auc={best_metric:.4f}")

            # ------------------- plugin test -------------------
            ckpt = torch.load(best_ckpt, map_location=device)
            plugin.load_state_dict(ckpt["plugin_state_dict"])
            aggregator.load_state_dict(ckpt["aggregator_state_dict"])
            shared_role_proto.load_state_dict(ckpt["shared_role_proto_state_dict"])

            plugin_test_stats, plugin_test_pred_df = evaluate_plugin_with_predictions(
                plugin=plugin,
                aggregator=aggregator,
                shared_role_proto=shared_role_proto,
                summary_builder=summary_builder,
                loader=test_loader,
                device=device,
                use_role_logits=args.use_role_logits,
                feat_preserve_weight=args.feat_preserve_weight,
                proto_anchor_weight=args.proto_anchor_weight,
                preserve_mode=args.preserve_mode,
            )
            plugin_test_pred_df.to_csv(
                os.path.join(args.out_dir, "test_predictions.csv"),
                index=False
            )
            val_pred_df.to_csv(
                os.path.join(args.out_dir, "best_val_predictions.csv"),
                index=False
            )

            with open(os.path.join(args.out_dir, "plugin_final_test_metrics.json"), "w") as f:
                json.dump({
                    "best_val_auc": best_metric,
                    "test_stats": plugin_test_stats,
                }, f, indent=2)

            print(
                f"[Plugin Test] "
                f"test_loss={plugin_test_stats['loss']:.4f} "
                f"test_auc={plugin_test_stats['auc']:.4f} "
                f"test_acc={plugin_test_stats['acc']:.4f} "
                f"test_f1={plugin_test_stats['f1']:.4f} "
                f"test_sens={plugin_test_stats['sens']:.4f} "
                f"test_spec={plugin_test_stats['spec']:.4f}"
            )

            comparison = {
                "plugin": {
                    "best_val_auc": best_metric,
                    "test": plugin_test_stats,
                }
            }

            # ------------------- no-plugin baseline -------------------
            if args.run_no_plugin_baseline:
                baseline_result = train_no_plugin_baseline(
                    args=args,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                )
                comparison["no_plugin"] = baseline_result

                if not math.isnan(plugin_test_stats["auc"]) and not math.isnan(baseline_result["test"]["auc"]):
                    comparison["delta"] = {
                        "test_auc": plugin_test_stats["auc"] - baseline_result["test"]["auc"],
                        "test_acc": plugin_test_stats["acc"] - baseline_result["test"]["acc"],
                        "test_f1": plugin_test_stats["f1"] - baseline_result["test"]["f1"],
                        "test_sens": plugin_test_stats["sens"] - baseline_result["test"]["sens"],
                        "test_spec": plugin_test_stats["spec"] - baseline_result["test"]["spec"],
                    }

            with open(os.path.join(args.out_dir, "comparison_summary.json"), "w") as f:
                json.dump(comparison, f, indent=2)

            print(f"[Done] best ckpt saved to: {best_ckpt}")
            print(f"[Done] log saved to: {log_path}")


if __name__ == "__main__":
    main()