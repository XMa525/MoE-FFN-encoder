#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import random
import argparse
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import openslide
from PIL import ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import torchvision.transforms.v2 as T

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.encoders.moe_encoder import MoEEncoder
from models.plugins.shared_role_prototype import (
    SharedRolePrototype,
    PatchRoleSummaryFromSharedProto,
)
from downstream.role_transfer_losses import (
    compute_asymmetric_role_proto_loss,
    compute_slide_proxy_loss,
    compute_preserve_loss,
    compute_pairwise_ranking_loss,
    compute_online_role_scores,
    select_online_positive_proposals,
    select_online_negative_proposals,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def build_transform(img_size: int = 224, is_train: bool = False):
    if is_train:
        return T.Compose([
            T.ToImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def print_trainable_params(module: nn.Module, prefix: str):
    total = 0
    trainable = 0
    names = []
    for n, p in module.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            names.append(n)
    print(f"[{prefix}] total={total:,} trainable={trainable:,}")
    if len(names) > 0:
        print(f"[{prefix}] trainable parameter names:")
        for n in names:
            print("  ", n)


# =========================================================
# proj_l12
# =========================================================
def load_proj_l12_from_stage2(
    stage2_full_ckpt: str,
    device: str,
) -> nn.Module:
    ckpt = torch.load(stage2_full_ckpt, map_location="cpu")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in checkpoint")

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise KeyError("proj_l12 not found in distiller_state_dict")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    proj_l12 = nn.Linear(proj_in_dim, proj_out_dim)
    proj_l12.load_state_dict({
        "weight": distiller_sd["proj_l12.weight"],
        "bias": distiller_sd["proj_l12.bias"],
    })
    proj_l12 = proj_l12.to(device)
    proj_l12.eval()

    for p in proj_l12.parameters():
        p.requires_grad = False

    print(f"[proj_l12] loaded: {proj_in_dim} -> {proj_out_dim}")
    return proj_l12

def _norm_path(p: str) -> str:
    return os.path.normpath(os.path.expanduser(str(p)))


def find_svs_path_from_root(
    svs_root: str,
    slide_id: str,
    project: Optional[str] = None,
) -> str:
    svs_root = _norm_path(svs_root)
    search_roots = []

    if project is not None:
        pdir = os.path.join(svs_root, str(project))
        if os.path.isdir(pdir):
            search_roots.append(pdir)

    search_roots.append(svs_root)

    slide_id = str(slide_id)
    barcode_prefix = slide_id.split(".")[0]

    candidates = []
    valid_exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")

    for root in search_roots:
        if not os.path.isdir(root):
            continue

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith(valid_exts):
                    continue

                full = os.path.join(dirpath, fn)
                stem = os.path.splitext(fn)[0]

                score = 0
                if stem == slide_id:
                    score = 100
                elif fn == slide_id:
                    score = 95
                elif slide_id in fn:
                    score = 90
                elif barcode_prefix and barcode_prefix in fn:
                    score = 80
                elif stem in slide_id:
                    score = 60

                if score > 0:
                    candidates.append((score, full))

        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find WSI for slide_id={slide_id} under svs_root={svs_root}, project={project}"
        )

    candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def find_h5_path_from_root(
    h5_root: str,
    slide_id: str,
) -> str:
    h5_root = _norm_path(h5_root)
    slide_id = str(slide_id)
    barcode_prefix = slide_id.split(".")[0]

    direct_names = [
        f"{slide_id}.h5",
        f"{barcode_prefix}.h5",
    ]

    for name in direct_names:
        p = os.path.join(h5_root, name)
        if os.path.exists(p):
            return p

    candidates = []
    valid_exts = (".h5", ".hdf5")

    if not os.path.isdir(h5_root):
        raise FileNotFoundError(f"h5_root not found: {h5_root}")

    for dirpath, _, filenames in os.walk(h5_root):
        for fn in filenames:
            if not fn.lower().endswith(valid_exts):
                continue

            full = os.path.join(dirpath, fn)
            stem = os.path.splitext(fn)[0]

            score = 0
            if stem == slide_id:
                score = 100
            elif stem == barcode_prefix:
                score = 95
            elif slide_id in fn:
                score = 90
            elif barcode_prefix and barcode_prefix in fn:
                score = 80
            elif stem in slide_id:
                score = 60

            if score > 0:
                candidates.append((score, full))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find H5 for slide_id={slide_id} under h5_root={h5_root}"
        )

    candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def resolve_slide_paths_from_config(
    df: pd.DataFrame,
    svs_root: Optional[str] = None,
    h5_root: Optional[str] = None,
    project: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()

    if "svs_path" not in df.columns:
        df["svs_path"] = ""
    if "h5_path" not in df.columns:
        df["h5_path"] = ""

    svs_cache = {}
    h5_cache = {}
    resolved_svs = []
    resolved_h5 = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resolve train svs/h5", leave=False):
        slide_id = str(row["slide_id"])
        svs_path = str(row.get("svs_path", "") or "")
        h5_path = str(row.get("h5_path", "") or "")

        if svs_path and os.path.exists(svs_path):
            final_svs = svs_path
        else:
            if svs_root is None:
                raise ValueError(
                    f"svs_path missing/not found for slide_id={slide_id}, and svs_root is None"
                )
            if slide_id not in svs_cache:
                svs_cache[slide_id] = find_svs_path_from_root(
                    svs_root=svs_root,
                    slide_id=slide_id,
                    project=project,
                )
            final_svs = svs_cache[slide_id]

        if h5_path and os.path.exists(h5_path):
            final_h5 = h5_path
        else:
            if h5_root is None:
                raise ValueError(
                    f"h5_path missing/not found for slide_id={slide_id}, and h5_root is None"
                )
            if slide_id not in h5_cache:
                h5_cache[slide_id] = find_h5_path_from_root(
                    h5_root=h5_root,
                    slide_id=slide_id,
                )
            final_h5 = h5_cache[slide_id]

        resolved_svs.append(final_svs)
        resolved_h5.append(final_h5)

    df["svs_path"] = resolved_svs
    df["h5_path"] = resolved_h5
    return df

# =========================================================
# dataset
# =========================================================
def subsample_slide_ids(
    slide_df: pd.DataFrame,
    max_slides: Optional[int],
    seed: int = 42,
    balance_by_label: bool = True,
) -> pd.DataFrame:
    if max_slides is None or max_slides >= len(slide_df):
        return slide_df.reset_index(drop=True).copy()

    rng = np.random.default_rng(seed)

    if (not balance_by_label) or ("label" not in slide_df.columns):
        idx = rng.choice(len(slide_df), size=max_slides, replace=False)
        return slide_df.iloc[idx].reset_index(drop=True).copy()

    groups = list(slide_df.groupby("label"))
    if len(groups) == 0:
        return slide_df.reset_index(drop=True).copy()

    per_group = max_slides // len(groups)
    rem = max_slides % len(groups)

    parts = []
    used_ids = set()

    for gi, (_, sub) in enumerate(groups):
        take = per_group + (1 if gi < rem else 0)
        take = min(take, len(sub))
        if take <= 0:
            continue
        idx = rng.choice(len(sub), size=take, replace=False)
        chosen = sub.iloc[idx].copy()
        parts.append(chosen)
        used_ids.update(chosen["slide_id"].astype(str).tolist())

    cur_n = sum(len(x) for x in parts)
    if cur_n < max_slides:
        leftover = slide_df[~slide_df["slide_id"].astype(str).isin(used_ids)].copy()
        if len(leftover) > 0:
            take = min(max_slides - cur_n, len(leftover))
            idx = rng.choice(len(leftover), size=take, replace=False)
            parts.append(leftover.iloc[idx].copy())

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


class ProposalPoolSlideDataset(Dataset):
    """
    正负双源：
    - label=1 -> positive_pool_csv
    - label=0 -> negative_pool_csv
    """
    def __init__(
        self,
        split_csv: str,
        positive_pool_csv: str,
        negative_pool_csv: str,
        split: str,
        max_slides: Optional[int] = None,
        seed: int = 42,
        balance_by_label: bool = True,
        svs_root: Optional[str] = None,
        h5_root: Optional[str] = None,
        project: Optional[str] = None,
    ):
        split_df = pd.read_csv(split_csv)

        need = ["slide_id", "label", "split"]
        miss = [c for c in need if c not in split_df.columns]
        if miss:
            raise ValueError(f"split csv missing columns: {miss}")

        split_df = resolve_slide_paths_from_config(
            split_df,
            svs_root=svs_root,
            h5_root=h5_root,
            project=project,
        )

        split_df = split_df[split_df["split"] == split].copy().reset_index(drop=True)
        if len(split_df) == 0:
            raise ValueError(f"No rows found for split={split}")

        pos_df = pd.read_csv(positive_pool_csv)
        neg_df = pd.read_csv(negative_pool_csv)

        need_pool = ["slide_id", "label", "coord_x", "coord_y", "candidate_type"]
        miss_pos = [c for c in need_pool if c not in pos_df.columns]
        miss_neg = [c for c in need_pool if c not in neg_df.columns]
        if miss_pos:
            raise ValueError(f"positive pool csv missing columns: {miss_pos}")
        if miss_neg:
            raise ValueError(f"negative pool csv missing columns: {miss_neg}")

        if "split" in pos_df.columns:
            pos_df = pos_df[pos_df["split"] == split].copy()
        if "split" in neg_df.columns:
            neg_df = neg_df[neg_df["split"] == split].copy()

        pos_df["slide_id"] = pos_df["slide_id"].astype(str)
        neg_df["slide_id"] = neg_df["slide_id"].astype(str)
        split_df["slide_id"] = split_df["slide_id"].astype(str)

        pos_ids = set(pos_df["slide_id"].unique().tolist())
        neg_ids = set(neg_df["slide_id"].unique().tolist())

        def has_pool(row):
            sid = str(row["slide_id"])
            lab = int(row["label"])
            return sid in (pos_ids if lab == 1 else neg_ids)

        split_df = split_df[split_df.apply(has_pool, axis=1)].copy().reset_index(drop=True)
        if len(split_df) == 0:
            raise ValueError(f"No slide in split={split} has valid proposal/fixed pool.")

        split_df = subsample_slide_ids(
            slide_df=split_df,
            max_slides=max_slides,
            seed=seed,
            balance_by_label=balance_by_label,
        )

        selected_ids = set(split_df["slide_id"].astype(str).tolist())
        pos_df = pos_df[pos_df["slide_id"].isin(selected_ids)].copy()
        neg_df = neg_df[neg_df["slide_id"].isin(selected_ids)].copy()

        grouped = {}

        def build_group(sub: pd.DataFrame):
            sort_cols = ["rank_in_slide"] if "rank_in_slide" in sub.columns else ["coord_x", "coord_y"]
            sub = sub.sort_values(sort_cols).reset_index(drop=True)
            return {
                "coords": sub[["coord_x", "coord_y"]].values.astype(np.int64),
                "candidate_type": str(sub["candidate_type"].iloc[0]),
                "num_candidates": int(len(sub)),
                "mean_tumor_prob": float(sub["tumor_prob"].mean()) if "tumor_prob" in sub.columns else float("nan"),
                "mean_tumor_gap": float(sub["tumor_gap"].mean()) if "tumor_gap" in sub.columns else float("nan"),
                "pos_context_score": sub["pos_context_score"].values.astype(np.float32) if "pos_context_score" in sub.columns else np.zeros(len(sub), dtype=np.float32),
                "neg_context_score": sub["neg_context_score"].values.astype(np.float32) if "neg_context_score" in sub.columns else np.zeros(len(sub), dtype=np.float32),
                "neighbor_gap_mean": sub["neighbor_gap_mean"].values.astype(np.float32) if "neighbor_gap_mean" in sub.columns else np.zeros(len(sub), dtype=np.float32),
                "neighbor_gap_max": sub["neighbor_gap_max"].values.astype(np.float32) if "neighbor_gap_max" in sub.columns else np.zeros(len(sub), dtype=np.float32),
                "neighbor_prob_mean": sub["neighbor_prob_mean"].values.astype(np.float32) if "neighbor_prob_mean" in sub.columns else np.zeros(len(sub), dtype=np.float32),
                "neighbor_prob_max": sub["neighbor_prob_max"].values.astype(np.float32) if "neighbor_prob_max" in sub.columns else np.zeros(len(sub), dtype=np.float32),
            }

        for slide_id, sub in pos_df.groupby("slide_id"):
            grouped[str(slide_id)] = build_group(sub)
        for slide_id, sub in neg_df.groupby("slide_id"):
            grouped[str(slide_id)] = build_group(sub)

        self.df = split_df.reset_index(drop=True)
        self.grouped = grouped

        print(f"[{split}] slides={len(self.df)}")
        print(self.df["label"].value_counts().sort_index())

        cand_counts = []
        for sid in self.df["slide_id"].astype(str).tolist():
            cand_counts.append(self.grouped[sid]["num_candidates"])
        if len(cand_counts) > 0:
            print(
                f"[{split}] proposal pool count per slide: "
                f"mean={np.mean(cand_counts):.2f}, "
                f"min={np.min(cand_counts)}, max={np.max(cand_counts)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        slide_id = str(row["slide_id"])
        group = self.grouped[slide_id]
        return {
            "slide_id": slide_id,
            "label": int(row["label"]),
            "svs_path": str(row["svs_path"]),
            "h5_path": str(row["h5_path"]),
            "coords": group["coords"],
            "candidate_type": group["candidate_type"],
            "num_candidates": group["num_candidates"],
            "candidate_mean_tumor_prob": group["mean_tumor_prob"],
            "candidate_mean_tumor_gap": group["mean_tumor_gap"],
            "pos_context_score": group["pos_context_score"],
            "neg_context_score": group["neg_context_score"],
            "neighbor_gap_mean": group["neighbor_gap_mean"],
            "neighbor_gap_max": group["neighbor_gap_max"],
            "neighbor_prob_mean": group["neighbor_prob_mean"],
            "neighbor_prob_max": group["neighbor_prob_max"],
        }


def proposal_pool_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "slide_id": [x["slide_id"] for x in batch],
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "svs_path": [x["svs_path"] for x in batch],
        "h5_path": [x["h5_path"] for x in batch],
        "coords": [x["coords"] for x in batch],
        "candidate_type": [x["candidate_type"] for x in batch],
        "num_candidates": [x["num_candidates"] for x in batch],
        "candidate_mean_tumor_prob": [x["candidate_mean_tumor_prob"] for x in batch],
        "candidate_mean_tumor_gap": [x["candidate_mean_tumor_gap"] for x in batch],
        "pos_context_score": [x["pos_context_score"] for x in batch],
        "neg_context_score": [x["neg_context_score"] for x in batch],
        "neighbor_gap_mean": [x["neighbor_gap_mean"] for x in batch],
        "neighbor_gap_max": [x["neighbor_gap_max"] for x in batch],
        "neighbor_prob_mean": [x["neighbor_prob_mean"] for x in batch],
        "neighbor_prob_max": [x["neighbor_prob_max"] for x in batch],
    }


# =========================================================
# encoder loading / unfreeze
# =========================================================
def build_encoder_from_stage2(
    base_encoder_cfg,
    moe_encoder_cfg,
    stage2_full_ckpt: str,
    device: str,
):
    ckpt = torch.load(stage2_full_ckpt, map_location="cpu")
    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in checkpoint")

    encoder = MoEEncoder(base_encoder_cfg, moe_encoder_cfg)
    encoder.load_state_dict(ckpt["student_state_dict"], strict=True)
    encoder = encoder.to(device)
    print("[Encoder] loaded stage2 student_state_dict")
    return encoder


def resolve_last_moe_layer_idx(moe_layers, num_layers: int = 12) -> int:
    last_idx = moe_layers[-1]
    if last_idx < 0:
        last_idx = num_layers + last_idx
    return int(last_idx)


def unfreeze_last_moe_params_by_name(
    encoder: nn.Module,
    moe_layers: List[int],
    num_layers: int = 12,
    train_gate: bool = True,
    train_experts: bool = True,
):
    last_idx = resolve_last_moe_layer_idx(moe_layers, num_layers)
    gate_keys = ["router", "gate", "gating", "routing"]
    expert_keys = ["expert", "experts"]

    patterns = [
        f".layer.{last_idx}.",
        f".blocks.{last_idx}.",
        f"blocks.{last_idx}.",
        f"layer.{last_idx}.",
    ]

    trainable = []
    for name, p in encoder.named_parameters():
        if not any(pt in name for pt in patterns):
            continue
        lname = name.lower()
        ok = False
        if train_gate and any(k in lname for k in gate_keys):
            ok = True
        if train_experts and any(k in lname for k in expert_keys):
            ok = True
        if ok:
            p.requires_grad = True
            trainable.append(name)

    print("[Encoder] unfrozen params in last MoE:")
    for n in trainable:
        print("  ", n)


# =========================================================
# patch reading
# =========================================================
def read_patch_meta_from_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        attrs = dict(f["coords"].attrs.items())
    patch_size = int(attrs.get("patch_size", 256))
    patch_level = int(attrs.get("patch_level", 0))
    return patch_size, patch_level


def read_patch_batch(slide, coords, patch_size, patch_level, transform):
    imgs = []
    for xy in coords:
        x, y = int(xy[0]), int(xy[1])
        img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs, dim=0)


# =========================================================
# feature extraction
# =========================================================
def extract_patch_features_stage2_style(
    encoder: nn.Module,
    patch_imgs: torch.Tensor,
    use_last_moe_output: bool = True,
):
    out = encoder(
        patch_imgs,
        return_gates=True,
        mask=None,
        is_eval=True,
        return_features=True,
        offline_cluster_ids=None,
    )
    _, _, feature_dict, moe_feature_list = out

    if use_last_moe_output and len(moe_feature_list) > 0:
        feat_tokens = moe_feature_list[-1]
    else:
        feat_tokens = feature_dict["layer_12"]

    patch_tokens = feat_tokens[:, 1:, :]
    patch_feat = patch_tokens.mean(dim=1)
    return patch_feat


def extract_pool_patch_feats(
    encoder: nn.Module,
    svs_path: str,
    h5_path: str,
    coords: np.ndarray,
    transform,
    device: str,
    patch_batch_size: int,
    use_last_moe_output: bool = True,
    slide_id: Optional[str] = None,
    show_inner_progress: bool = False,
):
    patch_size, patch_level = read_patch_meta_from_h5(h5_path)

    slide = openslide.OpenSlide(svs_path)
    feats = []

    try:
        iterator = range(0, len(coords), patch_batch_size)
        if show_inner_progress:
            desc_name = slide_id if slide_id is not None else Path(str(svs_path)).stem[:30]
            iterator = tqdm(
                iterator,
                total=math.ceil(len(coords) / patch_batch_size),
                desc=f"  Patches[{desc_name}]",
                leave=False,
            )

        for i in iterator:
            coord_chunk = coords[i:i + patch_batch_size]
            imgs = read_patch_batch(
                slide=slide,
                coords=coord_chunk,
                patch_size=patch_size,
                patch_level=patch_level,
                transform=transform,
            ).to(device, non_blocking=True)

            feat = extract_patch_features_stage2_style(
                encoder=encoder,
                patch_imgs=imgs,
                use_last_moe_output=use_last_moe_output,
            )
            feats.append(feat)
    finally:
        slide.close()

    return torch.cat(feats, dim=0)


# =========================================================
# train / eval
# =========================================================
def run_one_epoch(
    encoder_adapt: nn.Module,
    encoder_frozen: nn.Module,
    proj_l12: nn.Module,
    summary_builder: PatchRoleSummaryFromSharedProto,
    loader: DataLoader,
    transform,
    device: str,
    args,
    optimizer: Optional[torch.optim.Optimizer],
    role_names: List[str],
    is_train: bool,
    epoch: int,
):
    encoder_adapt.train(is_train)
    encoder_frozen.eval()
    proj_l12.eval()
    summary_builder.eval()

    meter = {
        "total_loss": [],
        "role_proto_loss": [],
        "pos_proto_loss": [],
        "neg_proto_loss": [],
        "rank_loss": [],
        "slide_proxy_loss": [],
        "preserve_loss": [],
        "num_candidates": [],
        "num_selected": [],
        "mean_tumor_gap": [],
        "mean_tumor_prob": [],
        "mean_top1_gap": [],
        "mean_context_score": [],
        "mean_weight": [],
        "pos_mean_gap": [],
        "neg_mean_gap": [],
        "slide_score": [],
        "pos_support_num_before": [],
        "pos_support_num_after_strong": [],
        "pos_support_num_after_final": [],
        "pos_support_ratio_strong": [],
        "pos_support_ratio_final": [],
        "pos_support_used_fallback": [],
        "pos_selected_num": [],
        "pos_selected_ratio": [],
        "neg_selected_num": [],
        "neg_selected_ratio": [],
    }

    desc = f"{'Train' if is_train else 'Val'} {epoch}"
    pbar = tqdm(loader, desc=desc, leave=False)

    for _, batch in enumerate(pbar):
        if is_train:
            optimizer.zero_grad()

        batch_items = len(batch["slide_id"])

        batch_pos_gaps = []
        batch_neg_gaps = []

        batch_role_losses = []
        batch_slide_losses = []
        batch_preserve_losses = []
        batch_role_stats = []

        for i in range(batch_items):
            slide_id = batch["slide_id"][i]
            label = int(batch["labels"][i].item())
            svs_path = batch["svs_path"][i]
            h5_path = batch["h5_path"][i]
            coords = batch["coords"][i]
            candidate_type = batch["candidate_type"][i]

            pos_context_score = torch.as_tensor(batch["pos_context_score"][i], dtype=torch.float32, device=device)
            neg_context_score = torch.as_tensor(batch["neg_context_score"][i], dtype=torch.float32, device=device)
            neighbor_gap_mean = torch.as_tensor(batch["neighbor_gap_mean"][i], dtype=torch.float32, device=device)
            neighbor_gap_max = torch.as_tensor(batch["neighbor_gap_max"][i], dtype=torch.float32, device=device)
            neighbor_prob_mean = torch.as_tensor(batch["neighbor_prob_mean"][i], dtype=torch.float32, device=device)
            neighbor_prob_max = torch.as_tensor(batch["neighbor_prob_max"][i], dtype=torch.float32, device=device)

            with torch.no_grad():
                patch_feat_frozen = extract_pool_patch_feats(
                    encoder=encoder_frozen,
                    svs_path=svs_path,
                    h5_path=h5_path,
                    coords=coords,
                    transform=transform,
                    device=device,
                    patch_batch_size=args.patch_batch_size,
                    use_last_moe_output=args.use_last_moe_output,
                    slide_id=slide_id,
                    show_inner_progress=getattr(args, "show_inner_patch_progress", False),
                )

            patch_feat_adapt = extract_pool_patch_feats(
                encoder=encoder_adapt,
                svs_path=svs_path,
                h5_path=h5_path,
                coords=coords,
                transform=transform,
                device=device,
                patch_batch_size=args.patch_batch_size,
                use_last_moe_output=args.use_last_moe_output,
                slide_id=slide_id,
                show_inner_progress=getattr(args, "show_inner_patch_progress", False),
            )

            online_scores = compute_online_role_scores(
                patch_feat_adapt=patch_feat_adapt,
                proj_l12=proj_l12,
                summary_builder=summary_builder,
                role_names=role_names,
                tumor_name=args.proto_tumor_name,
                negative_role_names=args.proto_negative_role_names,
            )

            if label == 1:
                selected_idx, select_stats = select_online_positive_proposals(
                    tumor_gap=online_scores["tumor_gap"],
                    tumor_prob=online_scores["tumor_prob"],
                    top1_gap=online_scores["top1_gap"],
                    pos_context_score=pos_context_score,
                    pos_neighbor_gap_mean=neighbor_gap_mean,
                    pos_neighbor_gap_max=neighbor_gap_max,
                    pos_neighbor_prob_mean=neighbor_prob_mean,
                    pos_neighbor_prob_max=neighbor_prob_max,
                    use_strong_pos_support=getattr(args, "use_strong_pos_support", True),
                    allow_pos_support_fallback=getattr(args, "allow_pos_support_fallback", True),
                    min_pos_keep=getattr(args, "min_pos_keep", 4),
                    select_topk=getattr(args, "online_pos_topk", None),
                    pos_support_min_tumor_prob=getattr(args, "pos_support_min_tumor_prob", -1e6),
                    pos_support_min_center_gap=getattr(args, "pos_support_min_center_gap", -1e6),
                    pos_support_min_top1_gap=getattr(args, "pos_support_min_top1_gap", -1e6),
                    pos_support_min_context_score=getattr(args, "pos_support_min_context_score", -1e6),
                    pos_support_min_neighbor_gap_mean=getattr(args, "pos_support_min_neighbor_gap_mean", -1e6),
                    pos_support_min_neighbor_gap_max=getattr(args, "pos_support_min_neighbor_gap_max", -1e6),
                    pos_support_min_neighbor_prob_mean=getattr(args, "pos_support_min_neighbor_prob_mean", -1e6),
                    pos_support_min_neighbor_prob_max=getattr(args, "pos_support_min_neighbor_prob_max", -1e6),
                    pos_fallback_min_tumor_prob=getattr(args, "pos_fallback_min_tumor_prob", -1e6),
                    pos_fallback_min_center_gap=getattr(args, "pos_fallback_min_center_gap", -1e6),
                    pos_fallback_min_top1_gap=getattr(args, "pos_fallback_min_top1_gap", -1e6),
                    pos_fallback_min_context_score=getattr(args, "pos_fallback_min_context_score", -1e6),
                    pos_fallback_min_neighbor_gap_mean=getattr(args, "pos_fallback_min_neighbor_gap_mean", -1e6),
                    pos_fallback_min_neighbor_gap_max=getattr(args, "pos_fallback_min_neighbor_gap_max", -1e6),
                    pos_fallback_min_neighbor_prob_mean=getattr(args, "pos_fallback_min_neighbor_prob_mean", -1e6),
                    pos_fallback_min_neighbor_prob_max=getattr(args, "pos_fallback_min_neighbor_prob_max", -1e6),
                )
            else:
                selected_idx, select_stats = select_online_negative_proposals(
                    tumor_gap=online_scores["tumor_gap"],
                    tumor_prob=online_scores["tumor_prob"],
                    top1_gap=online_scores["top1_gap"],
                    neg_context_score=neg_context_score,
                    neighbor_gap_mean=neighbor_gap_mean,
                    neighbor_gap_max=neighbor_gap_max,
                    select_topk=getattr(args, "online_neg_topk", None),
                )

            role_loss, role_stats, tumor_gap_used, _ = compute_asymmetric_role_proto_loss(
                patch_feat_adapt=patch_feat_adapt,
                slide_label=label,
                candidate_type=candidate_type,
                proj_l12=proj_l12,
                summary_builder=summary_builder,
                role_names=role_names,
                tumor_name=args.proto_tumor_name,
                negative_role_names=args.proto_negative_role_names,
                margin_pos=args.margin_pos,
                margin_neg=args.margin_neg,
                pos_context_score=pos_context_score if label == 1 else None,
                neg_context_score=neg_context_score if label == 0 else None,
                context_weight_mode=getattr(args, "context_weight_mode", "none"),
                alpha_tumor_prob=getattr(args, "alpha_tumor_prob", 1.0),
                alpha_top1_gap=getattr(args, "alpha_top1_gap", 1.0),
                alpha_context=getattr(args, "alpha_context", 1.0),
                detach_weight=getattr(args, "detach_weight", True),
                selected_idx=selected_idx,
            )

            role_stats.update(select_stats)

            if label == 1:
                batch_pos_gaps.append(tumor_gap_used)
            else:
                batch_neg_gaps.append(tumor_gap_used)

            if getattr(args, "enable_slide_proxy_loss", False):
                slide_loss, slide_score = compute_slide_proxy_loss(
                    tumor_gap=tumor_gap_used,
                    slide_label=label,
                    topk=min(args.slide_proxy_topk, len(tumor_gap_used)),
                )
            else:
                slide_loss = torch.zeros((), device=patch_feat_adapt.device)
                slide_score = 0.0

            preserve_loss = compute_preserve_loss(
                patch_feat_adapt=patch_feat_adapt,
                patch_feat_frozen=patch_feat_frozen,
            )

            batch_role_losses.append(role_loss)
            batch_slide_losses.append(slide_loss)
            batch_preserve_losses.append(preserve_loss)
            batch_role_stats.append((role_stats, slide_score))

        if len(batch_pos_gaps) > 0 and len(batch_neg_gaps) > 0 and getattr(args, "rank_loss_weight", 0.0) > 0:
            all_pos_gap = torch.cat(batch_pos_gaps, dim=0)
            all_neg_gap = torch.cat(batch_neg_gaps, dim=0)
            rank_loss = compute_pairwise_ranking_loss(
                pos_gap=all_pos_gap,
                neg_gap=all_neg_gap,
                margin_rank=args.margin_rank,
                mode=args.rank_mode,
                topk=args.rank_topk,
            )
        else:
            rank_loss = torch.zeros((), device=device)

        mean_role_loss = torch.stack(batch_role_losses).mean() if len(batch_role_losses) > 0 else torch.zeros((), device=device)
        mean_slide_loss = torch.stack(batch_slide_losses).mean() if len(batch_slide_losses) > 0 else torch.zeros((), device=device)
        mean_preserve_loss = torch.stack(batch_preserve_losses).mean() if len(batch_preserve_losses) > 0 else torch.zeros((), device=device)

        total = (
            args.role_proto_loss_weight * mean_role_loss
            + args.slide_proxy_loss_weight * mean_slide_loss
            + args.preserve_loss_weight * mean_preserve_loss
            + args.rank_loss_weight * rank_loss
        )

        if is_train:
            total.backward()
            optimizer.step()

        meter["total_loss"].append(safe_float(total))
        meter["role_proto_loss"].append(safe_float(mean_role_loss))
        meter["slide_proxy_loss"].append(safe_float(mean_slide_loss))
        meter["preserve_loss"].append(safe_float(mean_preserve_loss))
        meter["rank_loss"].append(safe_float(rank_loss))

        for role_stats, slide_score in batch_role_stats:
            meter["pos_proto_loss"].append(role_stats["pos_proto_loss"])
            meter["neg_proto_loss"].append(role_stats["neg_proto_loss"])
            meter["num_candidates"].append(role_stats["num_candidates"])
            meter["num_selected"].append(role_stats["num_selected"])
            meter["mean_tumor_gap"].append(role_stats["mean_tumor_gap"])
            meter["mean_tumor_prob"].append(role_stats["mean_tumor_prob"])
            meter["mean_top1_gap"].append(role_stats["mean_top1_gap"])

            if not math.isnan(role_stats["mean_context_score"]):
                meter["mean_context_score"].append(role_stats["mean_context_score"])
            if not math.isnan(role_stats["mean_weight"]):
                meter["mean_weight"].append(role_stats["mean_weight"])
            if not math.isnan(role_stats["pos_mean_gap"]):
                meter["pos_mean_gap"].append(role_stats["pos_mean_gap"])
            if not math.isnan(role_stats["neg_mean_gap"]):
                meter["neg_mean_gap"].append(role_stats["neg_mean_gap"])

            for k in [
                "pos_support_num_before",
                "pos_support_num_after_strong",
                "pos_support_num_after_final",
                "pos_support_ratio_strong",
                "pos_support_ratio_final",
                "pos_support_used_fallback",
                "pos_selected_num",
                "pos_selected_ratio",
                "neg_selected_num",
                "neg_selected_ratio",
            ]:
                if k in role_stats and not math.isnan(role_stats[k]):
                    meter[k].append(role_stats[k])

            meter["slide_score"].append(slide_score)

        pbar.set_postfix(
            loss=f"{np.mean(meter['total_loss']):.4f}",
            rank=f"{np.mean(meter['rank_loss']):.4f}" if len(meter["rank_loss"]) > 0 else "0.0000",
            gap=f"{np.mean(meter['mean_tumor_gap']):.4f}" if len(meter["mean_tumor_gap"]) > 0 else "nan",
            sel=f"{np.mean(meter['num_selected']):.1f}" if len(meter["num_selected"]) > 0 else "0",
        )

    out = {
        "loss": float(np.mean(meter["total_loss"])) if meter["total_loss"] else 0.0,
        "role_proto_loss": float(np.mean(meter["role_proto_loss"])) if meter["role_proto_loss"] else 0.0,
        "pos_proto_loss": float(np.mean(meter["pos_proto_loss"])) if meter["pos_proto_loss"] else 0.0,
        "neg_proto_loss": float(np.mean(meter["neg_proto_loss"])) if meter["neg_proto_loss"] else 0.0,
        "rank_loss": float(np.mean(meter["rank_loss"])) if meter["rank_loss"] else 0.0,
        "slide_proxy_loss": float(np.mean(meter["slide_proxy_loss"])) if meter["slide_proxy_loss"] else 0.0,
        "preserve_loss": float(np.mean(meter["preserve_loss"])) if meter["preserve_loss"] else 0.0,
        "num_candidates": float(np.mean(meter["num_candidates"])) if meter["num_candidates"] else 0.0,
        "num_selected": float(np.mean(meter["num_selected"])) if meter["num_selected"] else 0.0,
        "mean_tumor_gap": float(np.mean(meter["mean_tumor_gap"])) if meter["mean_tumor_gap"] else 0.0,
        "mean_tumor_prob": float(np.mean(meter["mean_tumor_prob"])) if meter["mean_tumor_prob"] else 0.0,
        "mean_top1_gap": float(np.mean(meter["mean_top1_gap"])) if meter["mean_top1_gap"] else 0.0,
        "mean_context_score": float(np.mean(meter["mean_context_score"])) if meter["mean_context_score"] else float("nan"),
        "mean_weight": float(np.mean(meter["mean_weight"])) if meter["mean_weight"] else float("nan"),
        "pos_mean_gap": float(np.mean(meter["pos_mean_gap"])) if meter["pos_mean_gap"] else float("nan"),
        "neg_mean_gap": float(np.mean(meter["neg_mean_gap"])) if meter["neg_mean_gap"] else float("nan"),
        "slide_score": float(np.mean(meter["slide_score"])) if meter["slide_score"] else 0.0,
        "pos_support_num_before": float(np.mean(meter["pos_support_num_before"])) if meter["pos_support_num_before"] else float("nan"),
        "pos_support_num_after_strong": float(np.mean(meter["pos_support_num_after_strong"])) if meter["pos_support_num_after_strong"] else float("nan"),
        "pos_support_num_after_final": float(np.mean(meter["pos_support_num_after_final"])) if meter["pos_support_num_after_final"] else float("nan"),
        "pos_support_ratio_strong": float(np.mean(meter["pos_support_ratio_strong"])) if meter["pos_support_ratio_strong"] else float("nan"),
        "pos_support_ratio_final": float(np.mean(meter["pos_support_ratio_final"])) if meter["pos_support_ratio_final"] else float("nan"),
        "pos_support_used_fallback": float(np.mean(meter["pos_support_used_fallback"])) if meter["pos_support_used_fallback"] else float("nan"),
        "pos_selected_num": float(np.mean(meter["pos_selected_num"])) if meter["pos_selected_num"] else float("nan"),
        "pos_selected_ratio": float(np.mean(meter["pos_selected_ratio"])) if meter["pos_selected_ratio"] else float("nan"),
        "neg_selected_num": float(np.mean(meter["neg_selected_num"])) if meter["neg_selected_num"] else float("nan"),
        "neg_selected_ratio": float(np.mean(meter["neg_selected_ratio"])) if meter["neg_selected_ratio"] else float("nan"),
    }
    return out


# =========================================================
# save
# =========================================================
def save_encoder_bundle(
    out_path: str,
    encoder: nn.Module,
    cfg: dict,
    epoch: int,
    train_stats: dict,
    val_stats: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "student_state_dict": encoder.state_dict(),
            "cfg": cfg,
            "train_stats": train_stats,
            "val_stats": val_stats,
        },
        out_path,
    )


def save_stage2_style_student_only(
    out_path: str,
    encoder: nn.Module,
    cfg: dict,
    epoch: int,
    train_stats: dict,
    val_stats: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "student_state_dict": encoder.state_dict(),
            "cfg": cfg,
            "train_stats": train_stats,
            "val_stats": val_stats,
        },
        out_path,
    )


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Encoder-only transfer with proposal pool + online support filtering")
    parser.add_argument("--config", type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, "r") as f:
        cfg = yaml.safe_load(f)

    class Args:
        pass

    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    defaults = {
        "use_last_moe_output": True,
        "enable_slide_proxy_loss": True,
        "margin_pos": 0.08,
        "margin_neg": 0.10,
        "context_weight_mode": "none",
        "alpha_tumor_prob": 1.0,
        "alpha_top1_gap": 1.0,
        "alpha_context": 1.0,
        "detach_weight": True,
        "rank_loss_weight": 0.0,
        "margin_rank": 0.03,
        "rank_mode": "topk_mean",
        "rank_topk": 8,
        "show_inner_patch_progress": False,
        "use_strong_pos_support": True,
        "allow_pos_support_fallback": True,
        "min_pos_keep": 4,
        "online_pos_topk": None,
        "online_neg_topk": None,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    for name, default in [
        ("pos_support_min_tumor_prob", -1e6),
        ("pos_support_min_center_gap", -1e6),
        ("pos_support_min_top1_gap", -1e6),
        ("pos_support_min_context_score", -1e6),
        ("pos_support_min_neighbor_gap_mean", -1e6),
        ("pos_support_min_neighbor_gap_max", -1e6),
        ("pos_support_min_neighbor_prob_mean", -1e6),
        ("pos_support_min_neighbor_prob_max", -1e6),
        ("pos_fallback_min_tumor_prob", -1e6),
        ("pos_fallback_min_center_gap", -1e6),
        ("pos_fallback_min_top1_gap", -1e6),
        ("pos_fallback_min_context_score", -1e6),
        ("pos_fallback_min_neighbor_gap_mean", -1e6),
        ("pos_fallback_min_neighbor_gap_max", -1e6),
        ("pos_fallback_min_neighbor_prob_mean", -1e6),
        ("pos_fallback_min_neighbor_prob_max", -1e6),
    ]:
        if not hasattr(args, name):
            setattr(args, name, default)

    ensure_dir(args.out_dir)
    log_path = os.path.join(args.out_dir, "train.log")

    with open(log_path, "w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with redirect_stdout(tee), redirect_stderr(tee):
            set_seed(args.seed)
            device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

            print("=" * 80)
            print(json.dumps(cfg, indent=2, ensure_ascii=False))
            print("=" * 80)

            train_tf = build_transform(args.img_size, is_train=True)
            val_tf = build_transform(args.img_size, is_train=False)

            train_set = ProposalPoolSlideDataset(
                split_csv=args.split_csv,
                positive_pool_csv=args.train_positive_pool_csv,
                negative_pool_csv=args.train_negative_pool_csv,
                split="train",
                max_slides=getattr(args, "max_train_slides", None),
                seed=args.seed,
                balance_by_label=getattr(args, "balance_train_slides_by_label", True),
                svs_root=getattr(args, "svs_root", None),
                h5_root=getattr(args, "h5_root", None),
                project=getattr(args, "project", None),
            )
            val_set = ProposalPoolSlideDataset(
                split_csv=args.split_csv,
                positive_pool_csv=args.val_positive_pool_csv,
                negative_pool_csv=args.val_negative_pool_csv,
                split="val",
                max_slides=getattr(args, "max_val_slides", None),
                seed=args.seed + 1,
                balance_by_label=getattr(args, "balance_val_slides_by_label", True),
                svs_root=getattr(args, "svs_root", None),
                h5_root=getattr(args, "h5_root", None),
                project=getattr(args, "project", None),
            )

            train_loader = DataLoader(
                train_set,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=proposal_pool_collate_fn,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=proposal_pool_collate_fn,
            )

            encoder_frozen = build_encoder_from_stage2(
                args.base_encoder,
                args.moe_encoder,
                args.stage2_full_ckpt,
                device,
            )
            freeze_module(encoder_frozen)
            encoder_frozen.eval()

            encoder_adapt = build_encoder_from_stage2(
                args.base_encoder,
                args.moe_encoder,
                args.stage2_full_ckpt,
                device,
            )
            freeze_module(encoder_adapt)
            unfreeze_last_moe_params_by_name(
                encoder=encoder_adapt,
                moe_layers=args.moe_encoder["moe_layers"],
                num_layers=12,
                train_gate=args.train_last_gate,
                train_experts=args.train_last_experts,
            )

            proj_l12 = load_proj_l12_from_stage2(
                stage2_full_ckpt=args.stage2_full_ckpt,
                device=device,
            )

            shared_role_proto = SharedRolePrototype.from_files(
                role_proto_dir=args.role_proto_dir,
                normalize=True,
                learnable=False,
                device=device,
            )
            role_names = shared_role_proto.role_names
            print(f"[RoleProto] role_names = {role_names}")

            summary_builder = PatchRoleSummaryFromSharedProto(
                shared_role_proto=shared_role_proto,
                tau=args.role_tau,
                use_softmax=True,
            ).to(device)
            summary_builder.eval()

            print_trainable_params(encoder_adapt, "EncoderAdapt")

            params = [p for p in encoder_adapt.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            history = []
            best_metric = 1e18
            best_path = os.path.join(args.out_dir, "best_encoder_only_online_pool.pt")
            last_path = os.path.join(args.out_dir, "last_encoder_only_online_pool.pt")
            best_student_path = os.path.join(args.out_dir, "best_encoder_only_online_pool_student.pth")

            for epoch in range(1, args.epochs + 1):
                print("\n" + "=" * 80)
                print(f"[Epoch {epoch:03d}] start")
                print("=" * 80)

                train_stats = run_one_epoch(
                    encoder_adapt=encoder_adapt,
                    encoder_frozen=encoder_frozen,
                    proj_l12=proj_l12,
                    summary_builder=summary_builder,
                    loader=train_loader,
                    transform=train_tf,
                    device=device,
                    args=args,
                    optimizer=optimizer,
                    role_names=role_names,
                    is_train=True,
                    epoch=epoch,
                )

                val_stats = run_one_epoch(
                    encoder_adapt=encoder_adapt,
                    encoder_frozen=encoder_frozen,
                    proj_l12=proj_l12,
                    summary_builder=summary_builder,
                    loader=val_loader,
                    transform=val_tf,
                    device=device,
                    args=args,
                    optimizer=None,
                    role_names=role_names,
                    is_train=False,
                    epoch=epoch,
                )

                row = {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "train_role_proto_loss": train_stats["role_proto_loss"],
                    "train_pos_proto_loss": train_stats["pos_proto_loss"],
                    "train_neg_proto_loss": train_stats["neg_proto_loss"],
                    "train_rank_loss": train_stats["rank_loss"],
                    "train_slide_proxy_loss": train_stats["slide_proxy_loss"],
                    "train_preserve_loss": train_stats["preserve_loss"],
                    "train_num_candidates": train_stats["num_candidates"],
                    "train_num_selected": train_stats["num_selected"],
                    "train_mean_tumor_prob": train_stats["mean_tumor_prob"],
                    "train_mean_tumor_gap": train_stats["mean_tumor_gap"],
                    "train_mean_top1_gap": train_stats["mean_top1_gap"],
                    "train_mean_context_score": train_stats["mean_context_score"],
                    "train_mean_weight": train_stats["mean_weight"],
                    "train_pos_mean_gap": train_stats["pos_mean_gap"],
                    "train_neg_mean_gap": train_stats["neg_mean_gap"],
                    "train_pos_support_num_before": train_stats["pos_support_num_before"],
                    "train_pos_support_num_after_strong": train_stats["pos_support_num_after_strong"],
                    "train_pos_support_num_after_final": train_stats["pos_support_num_after_final"],
                    "train_pos_support_ratio_strong": train_stats["pos_support_ratio_strong"],
                    "train_pos_support_ratio_final": train_stats["pos_support_ratio_final"],
                    "train_pos_support_used_fallback": train_stats["pos_support_used_fallback"],
                    "train_pos_selected_num": train_stats["pos_selected_num"],
                    "train_pos_selected_ratio": train_stats["pos_selected_ratio"],
                    "train_neg_selected_num": train_stats["neg_selected_num"],
                    "train_neg_selected_ratio": train_stats["neg_selected_ratio"],

                    "val_loss": val_stats["loss"],
                    "val_role_proto_loss": val_stats["role_proto_loss"],
                    "val_pos_proto_loss": val_stats["pos_proto_loss"],
                    "val_neg_proto_loss": val_stats["neg_proto_loss"],
                    "val_rank_loss": val_stats["rank_loss"],
                    "val_slide_proxy_loss": val_stats["slide_proxy_loss"],
                    "val_preserve_loss": val_stats["preserve_loss"],
                    "val_num_candidates": val_stats["num_candidates"],
                    "val_num_selected": val_stats["num_selected"],
                    "val_mean_tumor_prob": val_stats["mean_tumor_prob"],
                    "val_mean_tumor_gap": val_stats["mean_tumor_gap"],
                    "val_mean_top1_gap": val_stats["mean_top1_gap"],
                    "val_mean_context_score": val_stats["mean_context_score"],
                    "val_mean_weight": val_stats["mean_weight"],
                    "val_pos_mean_gap": val_stats["pos_mean_gap"],
                    "val_neg_mean_gap": val_stats["neg_mean_gap"],
                    "val_slide_score": val_stats["slide_score"],
                    "val_pos_support_num_before": val_stats["pos_support_num_before"],
                    "val_pos_support_num_after_strong": val_stats["pos_support_num_after_strong"],
                    "val_pos_support_num_after_final": val_stats["pos_support_num_after_final"],
                    "val_pos_support_ratio_strong": val_stats["pos_support_ratio_strong"],
                    "val_pos_support_ratio_final": val_stats["pos_support_ratio_final"],
                    "val_pos_support_used_fallback": val_stats["pos_support_used_fallback"],
                    "val_pos_selected_num": val_stats["pos_selected_num"],
                    "val_pos_selected_ratio": val_stats["pos_selected_ratio"],
                    "val_neg_selected_num": val_stats["neg_selected_num"],
                    "val_neg_selected_ratio": val_stats["neg_selected_ratio"],
                }
                history.append(row)
                pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "train_history.csv"), index=False)

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={row['train_loss']:.4f} "
                    f"train_proto={row['train_role_proto_loss']:.4f} "
                    f"train_pos_proto={row['train_pos_proto_loss']:.4f} "
                    f"train_neg_proto={row['train_neg_proto_loss']:.4f} "
                    f"train_rank={row['train_rank_loss']:.4f} "
                    f"train_proxy={row['train_slide_proxy_loss']:.4f} "
                    f"train_preserve={row['train_preserve_loss']:.4f} "
                    f"train_gap={row['train_mean_tumor_gap']:.4f} "
                    f"train_pos_gap={row['train_pos_mean_gap']:.4f} "
                    f"train_neg_gap={row['train_neg_mean_gap']:.4f} "
                    f"train_sel={row['train_num_selected']:.2f} "
                    f"train_pos_keep={row['train_pos_selected_num']:.2f} "
                    f"train_neg_keep={row['train_neg_selected_num']:.2f}"
                )
                print(
                    f"[Epoch {epoch:03d}] "
                    f"val_loss={row['val_loss']:.4f} "
                    f"val_proto={row['val_role_proto_loss']:.4f} "
                    f"val_pos_proto={row['val_pos_proto_loss']:.4f} "
                    f"val_neg_proto={row['val_neg_proto_loss']:.4f} "
                    f"val_rank={row['val_rank_loss']:.4f} "
                    f"val_proxy={row['val_slide_proxy_loss']:.4f} "
                    f"val_preserve={row['val_preserve_loss']:.4f} "
                    f"val_gap={row['val_mean_tumor_gap']:.4f} "
                    f"val_pos_gap={row['val_pos_mean_gap']:.4f} "
                    f"val_neg_gap={row['val_neg_mean_gap']:.4f} "
                    f"val_sel={row['val_num_selected']:.2f} "
                    f"val_pos_keep={row['val_pos_selected_num']:.2f} "
                    f"val_neg_keep={row['val_neg_selected_num']:.2f} "
                    f"val_score={row['val_slide_score']:.4f}"
                )

                save_encoder_bundle(
                    out_path=last_path,
                    encoder=encoder_adapt,
                    cfg=cfg,
                    epoch=epoch,
                    train_stats=train_stats,
                    val_stats=val_stats,
                )

                cur_metric = row["val_loss"]
                if cur_metric < best_metric:
                    best_metric = cur_metric
                    save_encoder_bundle(
                        out_path=best_path,
                        encoder=encoder_adapt,
                        cfg=cfg,
                        epoch=epoch,
                        train_stats=train_stats,
                        val_stats=val_stats,
                    )
                    save_stage2_style_student_only(
                        out_path=best_student_path,
                        encoder=encoder_adapt,
                        cfg=cfg,
                        epoch=epoch,
                        train_stats=train_stats,
                        val_stats=val_stats,
                    )
                    print(f"[Best] epoch={epoch}, val_loss={best_metric:.4f}")

            with open(os.path.join(args.out_dir, "final_summary.json"), "w") as f:
                json.dump(
                    {
                        "best_val_loss": best_metric,
                        "best_ckpt": best_path,
                        "last_ckpt": last_path,
                        "best_student_only_ckpt": best_student_path,
                    },
                    f,
                    indent=2,
                )

            print(f"[Done] best: {best_path}")
            print(f"[Done] last: {last_path}")
            print(f"[Done] best student-only: {best_student_path}")
            print(f"[Done] log: {log_path}")


if __name__ == "__main__":
    main()