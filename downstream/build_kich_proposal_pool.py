#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import openslide
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
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

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def build_transform(img_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def robust_zscore_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mean = x.mean()
    std = x.std(unbiased=False)
    if float(std) < eps:
        return x - mean
    return (x - mean) / (std + eps)


def unique_preserve_order(indices: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in indices:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _norm_path(p: str) -> str:
    return os.path.normpath(os.path.expanduser(str(p)))


# =========================================================
# path resolving
# =========================================================
def find_svs_path_from_root(
    svs_root: str,
    slide_id: str,
    project: Optional[str] = None,
) -> str:
    """
    根据 slide_id 在 svs_root 下寻找对应 WSI。

    支持常见结构：
      svs_root/KICH/*.svs
      svs_root/**/*.svs

    匹配优先级：
      1. stem == slide_id
      2. filename == slide_id
      3. slide_id in filename
      4. slide_id 去掉 UUID 后的前缀 in filename
      5. stem in slide_id
    """
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
    """
    根据 slide_id 在 h5_root 下寻找对应 H5。

    常见命名：
      slide_id.h5
      slide_id_without_uuid.h5
      TCGA-XX-XXXX-01Z-00-DX1.h5
    """
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


def resolve_slide_paths_from_config(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    如果 CSV 没有 svs_path/h5_path，或者已有路径不存在，则根据 config 中的
    svs_root / h5_root / project 自动补全。
    """
    df = df.copy()

    project = getattr(args, "project", None)
    svs_root = getattr(args, "svs_root", None)
    h5_root = getattr(args, "h5_root", None)

    if "svs_path" not in df.columns:
        df["svs_path"] = ""
    if "h5_path" not in df.columns:
        df["h5_path"] = ""

    resolved_svs = []
    resolved_h5 = []

    svs_cache = {}
    h5_cache = {}

    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Resolve svs/h5 paths",
        leave=False,
    )

    for _, row in iterator:
        slide_id = str(row["slide_id"])

        svs_path = str(row.get("svs_path", "") or "")
        h5_path = str(row.get("h5_path", "") or "")

        if svs_path and os.path.exists(svs_path):
            final_svs = svs_path
        else:
            if svs_root is None:
                raise ValueError(
                    f"svs_path missing/not found for slide_id={slide_id}, "
                    f"and config has no svs_root"
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
                    f"h5_path missing/not found for slide_id={slide_id}, "
                    f"and config has no h5_root"
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
# loading
# =========================================================
def load_split_csv(
    split_csv: str,
    split: Optional[str] = None,
    args=None,
) -> pd.DataFrame:
    if not os.path.exists(split_csv):
        raise FileNotFoundError(split_csv)

    df = pd.read_csv(split_csv)

    need = ["slide_id", "label", "split"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"split csv missing columns: {miss}")

    if split is not None:
        df = df[df["split"] == split].copy()

    df = df.reset_index(drop=True)

    if args is not None:
        df = resolve_slide_paths_from_config(df, args)
    else:
        need_paths = ["svs_path", "h5_path"]
        miss = [c for c in need_paths if c not in df.columns]
        if miss:
            raise ValueError(f"split csv missing columns: {miss}")

    return df


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
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    print("[Encoder] loaded stage2 student_state_dict")
    return encoder


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


# =========================================================
# patch io
# =========================================================
def sample_coords_from_h5(
    h5_path: str,
    max_patches_per_slide: Optional[int],
    random_sample: bool,
    seed: int,
):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs.items())

    patch_size = int(attrs.get("patch_size", 256))
    patch_level = int(attrs.get("patch_level", 0))

    n = len(coords)
    patch_indices = np.arange(n, dtype=np.int64)

    if max_patches_per_slide is not None and n > max_patches_per_slide:
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=max_patches_per_slide, replace=False)
        else:
            idx = np.arange(max_patches_per_slide)

        coords = coords[idx]
        patch_indices = patch_indices[idx]

    return coords, patch_indices, patch_size, patch_level


def read_patch_batch(
    slide: openslide.OpenSlide,
    coords: np.ndarray,
    patch_size: int,
    patch_level: int,
    transform,
):
    imgs = []
    for xy in coords:
        x, y = int(xy[0]), int(xy[1])
        img = slide.read_region(
            (x, y),
            patch_level,
            (patch_size, patch_size),
        ).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs, dim=0)


# =========================================================
# feature extraction
# =========================================================
@torch.no_grad()
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

    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise RuntimeError(f"Unexpected encoder output type/length: type={type(out)}")

    _, _, feature_dict, moe_feature_list = out

    if use_last_moe_output and len(moe_feature_list) > 0:
        feat_tokens = moe_feature_list[-1]
    else:
        if "layer_12" not in feature_dict:
            raise KeyError(f"'layer_12' not found in feature_dict keys={list(feature_dict.keys())}")
        feat_tokens = feature_dict["layer_12"]

    patch_tokens = feat_tokens[:, 1:, :]
    if patch_tokens.shape[1] == 0:
        raise RuntimeError(f"No patch tokens found, got shape={tuple(patch_tokens.shape)}")

    patch_feat = patch_tokens.mean(dim=1)
    return patch_feat


@torch.no_grad()
def extract_one_slide_patch_feats(
    encoder: nn.Module,
    svs_path: str,
    h5_path: str,
    transform,
    device: str,
    patch_batch_size: int,
    use_last_moe_output: bool,
    max_patches_per_slide: Optional[int],
    random_sample_patches: bool,
    seed: int,
    show_inner_progress: bool = False,
):
    coords, patch_indices, patch_size, patch_level = sample_coords_from_h5(
        h5_path=h5_path,
        max_patches_per_slide=max_patches_per_slide,
        random_sample=random_sample_patches,
        seed=seed,
    )

    slide = openslide.OpenSlide(svs_path)
    feats = []

    try:
        iterator = range(0, len(coords), patch_batch_size)
        if show_inner_progress:
            iterator = tqdm(
                iterator,
                total=math.ceil(len(coords) / patch_batch_size),
                desc=f"  Patches[{os.path.basename(str(svs_path))[:30]}]",
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
            feats.append(feat.cpu())

    finally:
        slide.close()

    patch_feat = torch.cat(feats, dim=0)
    return patch_feat, coords, patch_indices


# =========================================================
# scoring
# =========================================================
@torch.no_grad()
def score_patches_with_role_proto(
    patch_feat_raw: torch.Tensor,
    proj_l12: nn.Module,
    summary_builder: PatchRoleSummaryFromSharedProto,
    role_names: List[str],
    tumor_name: str,
    negative_role_names: List[str],
    device: str,
):
    patch_feat_raw = patch_feat_raw.to(device, non_blocking=True)

    patch_feat_teacher = proj_l12(patch_feat_raw)
    patch_feat_teacher = F.normalize(patch_feat_teacher, dim=-1)

    role_dict = summary_builder(patch_feat_teacher.unsqueeze(0))
    role_logits = role_dict["patch_role_logits"][0].detach().cpu()
    role_probs = role_dict["patch_role_probs"][0].detach().cpu()
    top1_gap = role_dict["patch_top1_gap"][0].detach().cpu().squeeze(-1)

    role_to_idx = {n: i for i, n in enumerate(role_names)}
    if tumor_name not in role_to_idx:
        raise KeyError(f"tumor role '{tumor_name}' not found in role_names={role_names}")

    tumor_idx = role_to_idx[tumor_name]
    neg_ids = [role_to_idx[n] for n in negative_role_names if n in role_to_idx]
    if len(neg_ids) == 0:
        raise ValueError(
            f"No valid negative role names found. "
            f"got={negative_role_names}, role_names={role_names}"
        )

    tumor_logit = role_logits[:, tumor_idx]
    tumor_prob = role_probs[:, tumor_idx]
    neg_logit = role_logits[:, neg_ids].max(dim=-1).values
    tumor_gap = tumor_logit - neg_logit

    return {
        "role_logits": role_logits,
        "role_probs": role_probs,
        "top1_gap": top1_gap,
        "tumor_logit": tumor_logit,
        "tumor_prob": tumor_prob,
        "tumor_gap": tumor_gap,
    }


# =========================================================
# neighborhood context
# =========================================================
@torch.no_grad()
def build_neighbor_context_scores(
    coords: np.ndarray,
    scores: Dict[str, torch.Tensor],
    args,
):
    tumor_prob = scores["tumor_prob"].float()
    tumor_gap = scores["tumor_gap"].float()
    top1_gap = scores["top1_gap"].float()

    n = len(tumor_prob)
    if n == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return {
            "neighbor_gap_mean": empty,
            "neighbor_gap_max": empty,
            "neighbor_prob_mean": empty,
            "neighbor_prob_max": empty,
            "isolation_score": empty,
            "consistency_score": empty,
            "pos_context_score": empty,
            "neg_context_score": empty,
            "neighbor_num_used_mean": 0.0,
        }

    if n == 1:
        z = torch.zeros(1, dtype=torch.float32)
        return {
            "neighbor_gap_mean": z.clone(),
            "neighbor_gap_max": z.clone(),
            "neighbor_prob_mean": z.clone(),
            "neighbor_prob_max": z.clone(),
            "isolation_score": tumor_gap.clone(),
            "consistency_score": tumor_gap.clone(),
            "pos_context_score": tumor_gap.clone(),
            "neg_context_score": tumor_gap.clone(),
            "neighbor_num_used_mean": 0.0,
        }

    xy = torch.as_tensor(coords, dtype=torch.float32)
    dmat = torch.cdist(xy, xy)
    dmat.fill_diagonal_(float("inf"))

    radius = float(args.neighbor_radius)
    min_neighbors = int(args.neighbor_min_count)
    knn_fallback = int(args.neighbor_k_fallback)

    neighbor_gap_mean = []
    neighbor_gap_max = []
    neighbor_prob_mean = []
    neighbor_prob_max = []
    used_counts = []

    for i in range(n):
        dist_i = dmat[i]
        radius_mask = dist_i <= radius
        idx = torch.nonzero(radius_mask, as_tuple=False).squeeze(-1)

        if idx.numel() < min_neighbors:
            k = min(knn_fallback, n - 1)
            idx = torch.topk(dist_i, k=k, largest=False).indices

        gap_nb = tumor_gap[idx]
        prob_nb = tumor_prob[idx]

        neighbor_gap_mean.append(gap_nb.mean())
        neighbor_gap_max.append(gap_nb.max())
        neighbor_prob_mean.append(prob_nb.mean())
        neighbor_prob_max.append(prob_nb.max())
        used_counts.append(float(idx.numel()))

    neighbor_gap_mean = torch.stack(neighbor_gap_mean, dim=0)
    neighbor_gap_max = torch.stack(neighbor_gap_max, dim=0)
    neighbor_prob_mean = torch.stack(neighbor_prob_mean, dim=0)
    neighbor_prob_max = torch.stack(neighbor_prob_max, dim=0)

    isolation_score = tumor_gap - neighbor_gap_mean
    consistency_score = 0.5 * (tumor_gap + neighbor_gap_mean)

    z_tumor_gap = robust_zscore_torch(tumor_gap)
    z_tumor_prob = robust_zscore_torch(tumor_prob)
    z_top1_gap = robust_zscore_torch(top1_gap)
    z_nb_gap_mean = robust_zscore_torch(neighbor_gap_mean)
    z_nb_gap_max = robust_zscore_torch(neighbor_gap_max)
    z_iso = robust_zscore_torch(isolation_score)
    z_cons = robust_zscore_torch(consistency_score)

    pos_context_score = (
        args.pos_ctx_center_weight * z_tumor_gap
        + args.pos_ctx_neighbor_weight * z_nb_gap_mean
        + args.pos_ctx_neighbor_max_weight * z_nb_gap_max
        + args.pos_ctx_prob_weight * z_tumor_prob
        + args.pos_ctx_top1_weight * z_top1_gap
        + args.pos_ctx_consistency_weight * z_cons
    )

    neg_context_score = (
        args.neg_ctx_center_weight * z_tumor_gap
        - args.neg_ctx_neighbor_weight * z_nb_gap_mean
        + args.neg_ctx_isolation_weight * z_iso
        + args.neg_ctx_prob_weight * z_tumor_prob
        + args.neg_ctx_top1_weight * z_top1_gap
    )

    return {
        "neighbor_gap_mean": neighbor_gap_mean,
        "neighbor_gap_max": neighbor_gap_max,
        "neighbor_prob_mean": neighbor_prob_mean,
        "neighbor_prob_max": neighbor_prob_max,
        "isolation_score": isolation_score,
        "consistency_score": consistency_score,
        "pos_context_score": pos_context_score,
        "neg_context_score": neg_context_score,
        "neighbor_num_used_mean": float(np.mean(used_counts)),
    }


# =========================================================
# selection helpers
# =========================================================
def _topk_from_mask(score: torch.Tensor, valid_mask: torch.Tensor, k: int) -> List[int]:
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return []
    k = min(k, int(valid_idx.numel()))
    local = torch.topk(score[valid_idx], k=k, dim=0).indices
    return valid_idx[local].tolist()


def select_positive_proposal_pool(
    scores: Dict[str, torch.Tensor],
    ctx_scores: Dict[str, torch.Tensor],
    args,
):
    tumor_prob = scores["tumor_prob"]
    tumor_gap = scores["tumor_gap"]
    top1_gap = scores["top1_gap"]

    neighbor_gap_mean = ctx_scores["neighbor_gap_mean"]
    neighbor_gap_max = ctx_scores["neighbor_gap_max"]
    pos_context_score = ctx_scores["pos_context_score"]

    n = len(tumor_prob)
    if n == 0:
        return [], {}

    weak_mask = torch.ones(n, dtype=torch.bool)

    if getattr(args, "pos_pool_min_tumor_prob", -1e6) > -1e5:
        weak_mask &= (tumor_prob >= args.pos_pool_min_tumor_prob)
    if getattr(args, "pos_pool_min_center_gap", -1e6) > -1e5:
        weak_mask &= (tumor_gap >= args.pos_pool_min_center_gap)
    if getattr(args, "pos_pool_min_neighbor_gap_mean", -1e6) > -1e5:
        weak_mask &= (neighbor_gap_mean >= args.pos_pool_min_neighbor_gap_mean)

    if weak_mask.sum().item() == 0:
        weak_mask = torch.ones(n, dtype=torch.bool)

    tumor_pool_score = (
        getattr(args, "pos_pool_tumor_gap_weight", 1.0) * tumor_gap
        + getattr(args, "pos_pool_tumor_prob_weight", 0.5) * tumor_prob
        + getattr(args, "pos_pool_top1_gap_weight", 0.2) * top1_gap
    )

    tumor_branch_idx = _topk_from_mask(
        score=tumor_pool_score,
        valid_mask=weak_mask,
        k=int(args.pos_pool_topk_tumor),
    )

    support_pool_score = (
        getattr(args, "pos_pool_context_weight", 1.0) * pos_context_score
        + getattr(args, "pos_pool_neighbor_gap_mean_weight", 0.3) * neighbor_gap_mean
        + getattr(args, "pos_pool_neighbor_gap_max_weight", 0.2) * neighbor_gap_max
    )

    context_branch_idx = _topk_from_mask(
        score=support_pool_score,
        valid_mask=weak_mask,
        k=int(args.pos_pool_topk_context),
    )

    merged_idx = unique_preserve_order(tumor_branch_idx + context_branch_idx)

    max_merged = int(getattr(args, "pos_pool_max_merged", 0))
    if max_merged > 0 and len(merged_idx) > max_merged:
        merged_score = tumor_pool_score + 0.5 * support_pool_score
        merged_tensor = torch.as_tensor(merged_idx, dtype=torch.long)
        order = torch.argsort(merged_score[merged_tensor], descending=True)
        merged_idx = merged_tensor[order[:max_merged]].tolist()

    branch_tag = {}
    for idx in tumor_branch_idx:
        branch_tag[idx] = "tumor_branch"
    for idx in context_branch_idx:
        if idx in branch_tag:
            branch_tag[idx] = "merged"
        else:
            branch_tag[idx] = "context_branch"

    info = {
        "tumor_branch_idx": tumor_branch_idx,
        "context_branch_idx": context_branch_idx,
        "merged_idx": merged_idx,
        "branch_tag": branch_tag,
        "tumor_pool_score": tumor_pool_score,
        "support_pool_score": support_pool_score,
    }
    return merged_idx, info


def select_negative_fixed_candidates(
    scores: Dict[str, torch.Tensor],
    ctx_scores: Dict[str, torch.Tensor],
    args,
):
    tumor_prob = scores["tumor_prob"]
    tumor_gap = scores["tumor_gap"]
    top1_gap = scores["top1_gap"]

    neighbor_gap_mean = ctx_scores["neighbor_gap_mean"]
    neighbor_gap_max = ctx_scores["neighbor_gap_max"]
    neg_context_score = ctx_scores["neg_context_score"]

    n = len(tumor_prob)
    if n == 0:
        return [], {}

    valid = torch.ones(n, dtype=torch.bool)

    if getattr(args, "neg_min_tumor_prob", -1e6) > -1e5:
        valid &= (tumor_prob >= args.neg_min_tumor_prob)
    if getattr(args, "neg_min_center_gap", -1e6) > -1e5:
        valid &= (tumor_gap >= args.neg_min_center_gap)
    if getattr(args, "neg_max_neighbor_gap", 1e6) < 1e5:
        valid &= (neighbor_gap_mean <= args.neg_max_neighbor_gap)
    if getattr(args, "neg_max_neighbor_gap_max", 1e6) < 1e5:
        valid &= (neighbor_gap_max <= args.neg_max_neighbor_gap_max)
    if getattr(args, "neg_min_top1_gap", -1e6) > -1e5:
        valid &= (top1_gap >= args.neg_min_top1_gap)

    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    used_fallback = False

    if valid_idx.numel() == 0:
        fallback = torch.ones(n, dtype=torch.bool)
        if getattr(args, "neg_fallback_min_tumor_prob", -1e6) > -1e5:
            fallback &= (tumor_prob >= args.neg_fallback_min_tumor_prob)
        if getattr(args, "neg_fallback_min_center_gap", -1e6) > -1e5:
            fallback &= (tumor_gap >= args.neg_fallback_min_center_gap)

        valid_idx = torch.nonzero(fallback, as_tuple=False).squeeze(-1)
        used_fallback = True

        if valid_idx.numel() == 0:
            valid_idx = torch.arange(n)
            used_fallback = True

    k = min(int(args.neg_topk), int(valid_idx.numel()))
    local = torch.topk(neg_context_score[valid_idx], k=k, dim=0).indices
    selected_idx = valid_idx[local]

    order = torch.argsort(neg_context_score[selected_idx], descending=True)
    selected_idx = selected_idx[order]

    info = {
        "selected_idx": selected_idx.tolist(),
        "used_fallback": used_fallback,
        "neg_context_score": neg_context_score,
    }
    return selected_idx.tolist(), info


# =========================================================
# row builders
# =========================================================
def build_positive_proposal_rows(
    slide_id: str,
    split: str,
    label: int,
    svs_path: str,
    h5_path: str,
    coords: np.ndarray,
    patch_indices: np.ndarray,
    scores: Dict[str, torch.Tensor],
    ctx_scores: Dict[str, torch.Tensor],
    selected_idx: List[int],
    select_info: Dict,
):
    rows = []

    tumor_pool_score = select_info["tumor_pool_score"]
    support_pool_score = select_info["support_pool_score"]
    branch_tag = select_info["branch_tag"]

    merged_tensor = torch.as_tensor(selected_idx, dtype=torch.long)
    merged_score = tumor_pool_score[merged_tensor] + 0.5 * support_pool_score[merged_tensor]

    order = torch.argsort(merged_score, descending=True)
    ordered_idx = merged_tensor[order].tolist()

    for rank, idx_t in enumerate(ordered_idx):
        coord = coords[idx_t]
        patch_idx = int(patch_indices[idx_t])

        rows.append({
            "slide_id": slide_id,
            "split": split,
            "label": label,
            "candidate_type": "positive_proposal_pool",
            "pool_source": branch_tag.get(idx_t, "merged"),
            "rank_in_slide": rank,
            "patch_idx": patch_idx,
            "coord_x": int(coord[0]),
            "coord_y": int(coord[1]),

            "tumor_logit": safe_float(scores["tumor_logit"][idx_t]),
            "tumor_prob": safe_float(scores["tumor_prob"][idx_t]),
            "tumor_gap": safe_float(scores["tumor_gap"][idx_t]),
            "top1_gap": safe_float(scores["top1_gap"][idx_t]),

            "neighbor_gap_mean": safe_float(ctx_scores["neighbor_gap_mean"][idx_t]),
            "neighbor_gap_max": safe_float(ctx_scores["neighbor_gap_max"][idx_t]),
            "neighbor_prob_mean": safe_float(ctx_scores["neighbor_prob_mean"][idx_t]),
            "neighbor_prob_max": safe_float(ctx_scores["neighbor_prob_max"][idx_t]),
            "isolation_score": safe_float(ctx_scores["isolation_score"][idx_t]),
            "consistency_score": safe_float(ctx_scores["consistency_score"][idx_t]),
            "pos_context_score": safe_float(ctx_scores["pos_context_score"][idx_t]),
            "neg_context_score": safe_float(ctx_scores["neg_context_score"][idx_t]),

            "tumor_pool_score": safe_float(tumor_pool_score[idx_t]),
            "support_pool_score": safe_float(support_pool_score[idx_t]),
            "proposal_score": safe_float(tumor_pool_score[idx_t] + 0.5 * support_pool_score[idx_t]),

            "num_patches_scored": int(len(coords)),
            "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
            "svs_path": svs_path,
            "h5_path": h5_path,
        })

    return rows


def build_negative_candidate_rows(
    slide_id: str,
    split: str,
    label: int,
    svs_path: str,
    h5_path: str,
    coords: np.ndarray,
    patch_indices: np.ndarray,
    scores: Dict[str, torch.Tensor],
    ctx_scores: Dict[str, torch.Tensor],
    selected_idx: List[int],
    select_info: Dict,
):
    rows = []
    neg_context_score = select_info["neg_context_score"]

    for rank, idx_t in enumerate(selected_idx):
        coord = coords[idx_t]
        patch_idx = int(patch_indices[idx_t])

        rows.append({
            "slide_id": slide_id,
            "split": split,
            "label": label,
            "candidate_type": "negative_fixed_candidates",
            "pool_source": "negative_fixed",
            "rank_in_slide": rank,
            "patch_idx": patch_idx,
            "coord_x": int(coord[0]),
            "coord_y": int(coord[1]),

            "tumor_logit": safe_float(scores["tumor_logit"][idx_t]),
            "tumor_prob": safe_float(scores["tumor_prob"][idx_t]),
            "tumor_gap": safe_float(scores["tumor_gap"][idx_t]),
            "top1_gap": safe_float(scores["top1_gap"][idx_t]),

            "neighbor_gap_mean": safe_float(ctx_scores["neighbor_gap_mean"][idx_t]),
            "neighbor_gap_max": safe_float(ctx_scores["neighbor_gap_max"][idx_t]),
            "neighbor_prob_mean": safe_float(ctx_scores["neighbor_prob_mean"][idx_t]),
            "neighbor_prob_max": safe_float(ctx_scores["neighbor_prob_max"][idx_t]),
            "isolation_score": safe_float(ctx_scores["isolation_score"][idx_t]),
            "consistency_score": safe_float(ctx_scores["consistency_score"][idx_t]),
            "pos_context_score": safe_float(ctx_scores["pos_context_score"][idx_t]),
            "neg_context_score": safe_float(ctx_scores["neg_context_score"][idx_t]),

            "tumor_pool_score": float("nan"),
            "support_pool_score": float("nan"),
            "proposal_score": safe_float(neg_context_score[idx_t]),

            "num_patches_scored": int(len(coords)),
            "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
            "svs_path": svs_path,
            "h5_path": h5_path,
        })

    return rows


# =========================================================
# per-slide build
# =========================================================
@torch.no_grad()
def build_outputs_for_one_slide(
    row,
    encoder: nn.Module,
    proj_l12: nn.Module,
    summary_builder: PatchRoleSummaryFromSharedProto,
    transform,
    device: str,
    args,
    role_names: List[str],
):
    slide_id = str(row["slide_id"])
    label = int(row["label"])
    split = str(row["split"])
    svs_path = str(row["svs_path"])
    h5_path = str(row["h5_path"])

    if not os.path.exists(svs_path):
        raise FileNotFoundError(f"svs_path not found: {svs_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"h5_path not found: {h5_path}")

    patch_feat_raw, coords, patch_indices = extract_one_slide_patch_feats(
        encoder=encoder,
        svs_path=svs_path,
        h5_path=h5_path,
        transform=transform,
        device=device,
        patch_batch_size=args.patch_batch_size,
        use_last_moe_output=args.use_last_moe_output,
        max_patches_per_slide=args.max_patches_per_slide,
        random_sample_patches=args.random_sample_patches,
        seed=args.seed + (abs(hash(slide_id)) % 100000),
        show_inner_progress=getattr(args, "show_inner_patch_progress", False),
    )

    scores = score_patches_with_role_proto(
        patch_feat_raw=patch_feat_raw,
        proj_l12=proj_l12,
        summary_builder=summary_builder,
        role_names=role_names,
        tumor_name=args.proto_tumor_name,
        negative_role_names=args.proto_negative_role_names,
        device=device,
    )

    ctx_scores = build_neighbor_context_scores(
        coords=coords,
        scores=scores,
        args=args,
    )

    if label == 1:
        selected_idx, select_info = select_positive_proposal_pool(
            scores=scores,
            ctx_scores=ctx_scores,
            args=args,
        )
        rows = build_positive_proposal_rows(
            slide_id=slide_id,
            split=split,
            label=label,
            svs_path=svs_path,
            h5_path=h5_path,
            coords=coords,
            patch_indices=patch_indices,
            scores=scores,
            ctx_scores=ctx_scores,
            selected_idx=selected_idx,
            select_info=select_info,
        )
        summary = {
            "slide_id": slide_id,
            "split": split,
            "label": label,
            "output_type": "positive_proposal_pool",
            "num_patches_scored": int(len(coords)),
            "num_selected": int(len(selected_idx)),
            "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
            "mean_tumor_prob_all": safe_float(scores["tumor_prob"].mean()),
            "mean_tumor_gap_all": safe_float(scores["tumor_gap"].mean()),
            "mean_pos_context_all": safe_float(ctx_scores["pos_context_score"].mean()),
            "mean_selected_tumor_gap": safe_float(scores["tumor_gap"][torch.as_tensor(selected_idx)].mean()) if len(selected_idx) > 0 else 0.0,
            "mean_selected_pos_context": safe_float(ctx_scores["pos_context_score"][torch.as_tensor(selected_idx)].mean()) if len(selected_idx) > 0 else 0.0,
            "num_tumor_branch": int(len(select_info["tumor_branch_idx"])),
            "num_context_branch": int(len(select_info["context_branch_idx"])),
        }
        return "positive", rows, summary

    selected_idx, select_info = select_negative_fixed_candidates(
        scores=scores,
        ctx_scores=ctx_scores,
        args=args,
    )
    rows = build_negative_candidate_rows(
        slide_id=slide_id,
        split=split,
        label=label,
        svs_path=svs_path,
        h5_path=h5_path,
        coords=coords,
        patch_indices=patch_indices,
        scores=scores,
        ctx_scores=ctx_scores,
        selected_idx=selected_idx,
        select_info=select_info,
    )
    summary = {
        "slide_id": slide_id,
        "split": split,
        "label": label,
        "output_type": "negative_fixed_candidates",
        "num_patches_scored": int(len(coords)),
        "num_selected": int(len(selected_idx)),
        "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
        "mean_tumor_prob_all": safe_float(scores["tumor_prob"].mean()),
        "mean_tumor_gap_all": safe_float(scores["tumor_gap"].mean()),
        "mean_neg_context_all": safe_float(ctx_scores["neg_context_score"].mean()),
        "mean_selected_tumor_gap": safe_float(scores["tumor_gap"][torch.as_tensor(selected_idx)].mean()) if len(selected_idx) > 0 else 0.0,
        "mean_selected_neg_context": safe_float(ctx_scores["neg_context_score"][torch.as_tensor(selected_idx)].mean()) if len(selected_idx) > 0 else 0.0,
        "used_fallback": bool(select_info["used_fallback"]),
    }
    return "negative", rows, summary


# =========================================================
# defaults
# =========================================================
def set_default_args(args):
    if not hasattr(args, "show_inner_patch_progress"):
        args.show_inner_patch_progress = False

    if not hasattr(args, "random_sample_patches"):
        args.random_sample_patches = False

    if not hasattr(args, "neighbor_radius"):
        args.neighbor_radius = 1024.0
    if not hasattr(args, "neighbor_min_count"):
        args.neighbor_min_count = 4
    if not hasattr(args, "neighbor_k_fallback"):
        args.neighbor_k_fallback = 8

    if not hasattr(args, "pos_ctx_center_weight"):
        args.pos_ctx_center_weight = 1.0
    if not hasattr(args, "pos_ctx_neighbor_weight"):
        args.pos_ctx_neighbor_weight = 1.0
    if not hasattr(args, "pos_ctx_neighbor_max_weight"):
        args.pos_ctx_neighbor_max_weight = 0.25
    if not hasattr(args, "pos_ctx_prob_weight"):
        args.pos_ctx_prob_weight = 0.25
    if not hasattr(args, "pos_ctx_top1_weight"):
        args.pos_ctx_top1_weight = 0.10
    if not hasattr(args, "pos_ctx_consistency_weight"):
        args.pos_ctx_consistency_weight = 0.30

    if not hasattr(args, "neg_ctx_center_weight"):
        args.neg_ctx_center_weight = 1.0
    if not hasattr(args, "neg_ctx_neighbor_weight"):
        args.neg_ctx_neighbor_weight = 1.0
    if not hasattr(args, "neg_ctx_isolation_weight"):
        args.neg_ctx_isolation_weight = 0.60
    if not hasattr(args, "neg_ctx_prob_weight"):
        args.neg_ctx_prob_weight = 0.20
    if not hasattr(args, "neg_ctx_top1_weight"):
        args.neg_ctx_top1_weight = 0.10

    if not hasattr(args, "pos_pool_topk_tumor"):
        args.pos_pool_topk_tumor = 64
    if not hasattr(args, "pos_pool_topk_context"):
        args.pos_pool_topk_context = 64
    if not hasattr(args, "pos_pool_max_merged"):
        args.pos_pool_max_merged = 128

    if not hasattr(args, "pos_pool_min_tumor_prob"):
        args.pos_pool_min_tumor_prob = 0.28
    if not hasattr(args, "pos_pool_min_center_gap"):
        args.pos_pool_min_center_gap = -0.05
    if not hasattr(args, "pos_pool_min_neighbor_gap_mean"):
        args.pos_pool_min_neighbor_gap_mean = -1e6

    if not hasattr(args, "pos_pool_tumor_gap_weight"):
        args.pos_pool_tumor_gap_weight = 1.0
    if not hasattr(args, "pos_pool_tumor_prob_weight"):
        args.pos_pool_tumor_prob_weight = 0.5
    if not hasattr(args, "pos_pool_top1_gap_weight"):
        args.pos_pool_top1_gap_weight = 0.2

    if not hasattr(args, "pos_pool_context_weight"):
        args.pos_pool_context_weight = 1.0
    if not hasattr(args, "pos_pool_neighbor_gap_mean_weight"):
        args.pos_pool_neighbor_gap_mean_weight = 0.3
    if not hasattr(args, "pos_pool_neighbor_gap_max_weight"):
        args.pos_pool_neighbor_gap_max_weight = 0.2

    if not hasattr(args, "neg_topk"):
        args.neg_topk = 32
    if not hasattr(args, "neg_min_tumor_prob"):
        args.neg_min_tumor_prob = 0.33
    if not hasattr(args, "neg_min_center_gap"):
        args.neg_min_center_gap = 0.00
    if not hasattr(args, "neg_max_neighbor_gap"):
        args.neg_max_neighbor_gap = 0.00
    if not hasattr(args, "neg_max_neighbor_gap_max"):
        args.neg_max_neighbor_gap_max = 0.05
    if not hasattr(args, "neg_min_top1_gap"):
        args.neg_min_top1_gap = -1e6
    if not hasattr(args, "neg_fallback_min_tumor_prob"):
        args.neg_fallback_min_tumor_prob = 0.30
    if not hasattr(args, "neg_fallback_min_center_gap"):
        args.neg_fallback_min_center_gap = -0.03

    if not hasattr(args, "use_last_moe_output"):
        args.use_last_moe_output = True

    if not hasattr(args, "max_patches_per_slide"):
        args.max_patches_per_slide = None

    return args


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Build positive proposal pool + negative fixed candidates")
    parser.add_argument("--config", type=str, required=True, help="yaml config")
    args_cmd = parser.parse_args()

    with open(args_cmd.config, "r") as f:
        cfg = yaml.safe_load(f)

    class Args:
        pass

    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    args = set_default_args(args)

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    transform = build_transform(args.img_size)

    print("=" * 80)
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("=" * 80)
    print(
        f"[Neighbor] radius={args.neighbor_radius}, "
        f"min_count={args.neighbor_min_count}, "
        f"k_fallback={args.neighbor_k_fallback}"
    )
    print(
        f"[Sampling] max_patches_per_slide={getattr(args, 'max_patches_per_slide', None)}, "
        f"random_sample_patches={args.random_sample_patches}"
    )
    print(
        f"[PositivePool] topk_tumor={args.pos_pool_topk_tumor}, "
        f"topk_context={args.pos_pool_topk_context}, "
        f"max_merged={args.pos_pool_max_merged}"
    )
    print(f"[NegativeFixed] neg_topk={args.neg_topk}")

    encoder = build_encoder_from_stage2(
        base_encoder_cfg=args.base_encoder,
        moe_encoder_cfg=args.moe_encoder,
        stage2_full_ckpt=args.stage2_full_ckpt,
        device=device,
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
    role_names = list(shared_role_proto.role_names)

    print(f"[RoleProto] role_names={role_names}")
    print(f"[RoleProto] proto_tumor_name={args.proto_tumor_name}")
    print(f"[RoleProto] proto_negative_role_names={args.proto_negative_role_names}")

    summary_builder = PatchRoleSummaryFromSharedProto(
        shared_role_proto=shared_role_proto,
        tau=args.role_tau,
        use_softmax=True,
    ).to(device)
    summary_builder.eval()

    split_summaries = {}
    for split in args.build_splits:
        df = load_split_csv(args.split_csv, split=split, args=args)
        print(f"[{split}] num_slides = {len(df)}")

        resolved_split_csv = os.path.join(args.out_dir, f"{split}_resolved_slide_paths.csv")
        df.to_csv(resolved_split_csv, index=False)
        print(f"[Saved] resolved paths: {resolved_split_csv}")

        pos_rows = []
        neg_rows = []
        pos_summary_rows = []
        neg_summary_rows = []
        failed_rows = []

        pbar = tqdm(df.iterrows(), total=len(df), desc=f"Build {split} proposal/fixed")
        for _, row in pbar:
            slide_id = str(row["slide_id"])
            try:
                out_type, rows, summary = build_outputs_for_one_slide(
                    row=row,
                    encoder=encoder,
                    proj_l12=proj_l12,
                    summary_builder=summary_builder,
                    transform=transform,
                    device=device,
                    args=args,
                    role_names=role_names,
                )

                if out_type == "positive":
                    pos_rows.extend(rows)
                    pos_summary_rows.append(summary)
                else:
                    neg_rows.extend(rows)
                    neg_summary_rows.append(summary)

                pbar.set_postfix(
                    slide=slide_id[:20],
                    type=summary["output_type"],
                    selected=summary["num_selected"],
                    scored=summary["num_patches_scored"],
                )

            except Exception as e:
                print(f"[ERROR] split={split} slide_id={slide_id}: {e}")
                failed_rows.append({
                    "slide_id": slide_id,
                    "split": split,
                    "label": int(row["label"]),
                    "error": str(e),
                    "svs_path": str(row.get("svs_path", "")),
                    "h5_path": str(row.get("h5_path", "")),
                })

        pos_csv = os.path.join(args.out_dir, f"{split}_positive_proposal_pool.csv")
        neg_csv = os.path.join(args.out_dir, f"{split}_negative_fixed_candidates.csv")
        pos_summary_csv = os.path.join(args.out_dir, f"{split}_positive_proposal_summary.csv")
        neg_summary_csv = os.path.join(args.out_dir, f"{split}_negative_fixed_summary.csv")
        fail_csv = os.path.join(args.out_dir, f"{split}_build_failures.csv")

        pd.DataFrame(pos_rows).to_csv(pos_csv, index=False)
        pd.DataFrame(neg_rows).to_csv(neg_csv, index=False)
        pd.DataFrame(pos_summary_rows).to_csv(pos_summary_csv, index=False)
        pd.DataFrame(neg_summary_rows).to_csv(neg_summary_csv, index=False)

        if len(failed_rows) > 0:
            pd.DataFrame(failed_rows).to_csv(fail_csv, index=False)

        print(f"[Saved] {pos_csv}")
        print(f"[Saved] {neg_csv}")
        print(f"[Saved] {pos_summary_csv}")
        print(f"[Saved] {neg_summary_csv}")
        if len(failed_rows) > 0:
            print(f"[Saved] {fail_csv}")

        split_summaries[split] = {
            "num_slides": int(len(df)),
            "num_positive_pool_rows": int(len(pos_rows)),
            "num_negative_fixed_rows": int(len(neg_rows)),
            "num_failed_slides": int(len(failed_rows)),
            "positive_pool_csv": pos_csv,
            "negative_fixed_csv": neg_csv,
            "positive_summary_csv": pos_summary_csv,
            "negative_summary_csv": neg_summary_csv,
            "failure_csv": fail_csv if len(failed_rows) > 0 else None,
            "resolved_split_csv": resolved_split_csv,
        }

    summary_json = os.path.join(args.out_dir, "build_proposal_and_fixed_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg,
                "role_names": role_names,
                "splits": split_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[Saved] {summary_json}")
    print("[Done]")


if __name__ == "__main__":
    main()