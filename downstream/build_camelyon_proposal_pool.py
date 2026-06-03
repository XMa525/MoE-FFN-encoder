#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import random
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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


def safe_mean(x: torch.Tensor, default: float = 0.0) -> float:
    if x is None or x.numel() == 0:
        return float(default)
    return safe_float(x.mean())


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


def _get_arg(args, name: str, default=None):
    return getattr(args, name, default)


def stable_slide_seed(base_seed: int, slide_id: str) -> int:
    """
    Stable per-slide seed across different Python processes.

    Do not use Python's built-in hash(slide_id), because it is randomized
    between processes and will break cache reuse.
    """
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(base_seed) + (int(h[:8], 16) % 100000)


# =========================================================
# path resolving
# =========================================================
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


def resolve_slide_paths_from_config(df: pd.DataFrame, args) -> pd.DataFrame:
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
                    f"svs_path missing/not found for slide_id={slide_id}, and config has no svs_root"
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
                    f"h5_path missing/not found for slide_id={slide_id}, and config has no h5_root"
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
    return patch_feat, coords, patch_indices, patch_size, patch_level


# =========================================================
# scoring / cache
# =========================================================
def get_slide_cache_path(args, split: str, slide_id: str) -> Optional[str]:
    cache_dir = getattr(args, "score_cache_dir", None)
    if cache_dir is None or str(cache_dir).strip() == "":
        return None
    safe_sid = str(slide_id).replace("/", "_").replace("\\", "_")
    return os.path.join(str(cache_dir), split, f"{safe_sid}.pt")


def tensor_dict_to_cpu(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in d.items()}


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
            f"No valid negative role names found. got={negative_role_names}, role_names={role_names}"
        )

    tumor_logit = role_logits[:, tumor_idx]
    tumor_prob = role_probs[:, tumor_idx]
    neg_logit = role_logits[:, neg_ids].max(dim=-1).values
    tumor_gap = tumor_logit - neg_logit

    pred_role_idx = role_probs.argmax(dim=-1)
    pred_role_name = [role_names[int(i)] for i in pred_role_idx.tolist()]

    out = {
        "role_logits": role_logits,
        "role_probs": role_probs,
        "top1_gap": top1_gap,
        "tumor_logit": tumor_logit,
        "tumor_prob": tumor_prob,
        "neg_logit": neg_logit,
        "tumor_gap": tumor_gap,
        "pred_role_idx": pred_role_idx,
    }

    # Save per-role logit/prob, including lymphoid_tissue, stroma, etc.
    for ridx, rname in enumerate(role_names):
        out[f"role_logit__{rname}"] = role_logits[:, ridx]
        out[f"role_prob__{rname}"] = role_probs[:, ridx]

    # Role-vs-tumor margins are useful for exclusion/debugging.
    for rname in role_names:
        if rname == tumor_name:
            continue
        if rname in role_to_idx:
            ridx = role_to_idx[rname]
            out[f"margin_{rname}_over_tumor"] = role_logits[:, ridx] - tumor_logit
            out[f"margin_tumor_over_{rname}"] = tumor_logit - role_logits[:, ridx]

    out["pred_role_name"] = pred_role_name
    return out


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


def load_or_compute_slide_scores(
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
    split = str(row["split"])
    svs_path = str(row["svs_path"])
    h5_path = str(row["h5_path"])

    cache_path = get_slide_cache_path(args, split, slide_id)
    use_cache = bool(getattr(args, "reuse_score_cache", True))

    # Stable seed for newly generated cache files.
    patch_seed = stable_slide_seed(args.seed, slide_id)

    if cache_path is not None and use_cache and os.path.exists(cache_path):
        try:
            obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            meta = obj.get("meta", {})

            same_h5 = str(meta.get("h5_path", "")) == h5_path
            same_svs = str(meta.get("svs_path", "")) == svs_path
            same_max = meta.get("max_patches_per_slide", None) == getattr(args, "max_patches_per_slide", None)
            same_random = bool(meta.get("random_sample_patches", False)) == bool(getattr(args, "random_sample_patches", False))
            same_roles = list(meta.get("role_names", [])) == list(role_names)
            same_use_last = bool(meta.get("use_last_moe_output", True)) == bool(getattr(args, "use_last_moe_output", True))

            # Important:
            # Do not require same_seed here.
            # Old cache may have been generated with Python hash(slide_id), which is not stable across processes.
            # Since cache already stores coords/scores, it is safe to reuse when paths, sampling mode,
            # max_patches, roles, and feature source match.
            if same_h5 and same_svs and same_max and same_random and same_roles and same_use_last:
                required_keys = ["scores", "ctx_scores", "coords", "patch_indices", "patch_size", "patch_level"]
                if all(k in obj for k in required_keys):
                    return (
                        obj["scores"],
                        obj["ctx_scores"],
                        obj["coords"],
                        obj["patch_indices"],
                        obj["patch_size"],
                        obj["patch_level"],
                        True,
                    )
                missing = [k for k in required_keys if k not in obj]
                print(f"[Cache invalid] missing keys={missing}: {cache_path}")
            else:
                print(
                    f"[Cache miss] {slide_id}: "
                    f"same_h5={same_h5}, same_svs={same_svs}, same_max={same_max}, "
                    f"same_random={same_random}, same_roles={same_roles}, same_use_last={same_use_last}"
                )

        except Exception as e:
            print(f"[Cache load failed] {cache_path}: {e}")

    patch_feat_raw, coords, patch_indices, patch_size, patch_level = extract_one_slide_patch_feats(
        encoder=encoder,
        svs_path=svs_path,
        h5_path=h5_path,
        transform=transform,
        device=device,
        patch_batch_size=args.patch_batch_size,
        use_last_moe_output=args.use_last_moe_output,
        max_patches_per_slide=args.max_patches_per_slide,
        random_sample_patches=args.random_sample_patches,
        seed=patch_seed,
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

    if cache_path is not None:
        ensure_dir(os.path.dirname(cache_path))
        torch.save(
            {
                "meta": {
                    "slide_id": slide_id,
                    "split": split,
                    "svs_path": svs_path,
                    "h5_path": h5_path,
                    "max_patches_per_slide": getattr(args, "max_patches_per_slide", None),
                    "random_sample_patches": bool(getattr(args, "random_sample_patches", False)),
                    "seed": int(patch_seed),
                    "seed_mode": "stable_md5",
                    "role_names": list(role_names),
                    "use_last_moe_output": bool(getattr(args, "use_last_moe_output", True)),
                    "patch_size": int(patch_size),
                    "patch_level": int(patch_level),
                },
                "scores": tensor_dict_to_cpu(scores),
                "ctx_scores": tensor_dict_to_cpu(ctx_scores),
                "coords": coords,
                "patch_indices": patch_indices,
                "patch_size": int(patch_size),
                "patch_level": int(patch_level),
            },
            cache_path,
        )

    return scores, ctx_scores, coords, patch_indices, patch_size, patch_level, False


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


def get_role_tensor(scores: Dict[str, torch.Tensor], prefix: str, role_name: str, n: int, default: float = 0.0) -> torch.Tensor:
    key = f"{prefix}__{role_name}"
    if key in scores:
        return scores[key].float()
    return torch.full((n,), float(default), dtype=torch.float32)


def get_max_role_tensor(scores: Dict[str, torch.Tensor], prefix: str, role_names: List[str], n: int, default: float = 0.0) -> torch.Tensor:
    vals = []
    for r in role_names:
        key = f"{prefix}__{r}"
        if key in scores:
            vals.append(scores[key].float())
    if not vals:
        return torch.full((n,), float(default), dtype=torch.float32)
    return torch.stack(vals, dim=0).max(dim=0).values


def get_margin_role_over_tumor(scores: Dict[str, torch.Tensor], role_name: str, n: int, default: float = -1e6) -> torch.Tensor:
    key = f"margin_{role_name}_over_tumor"
    if key in scores:
        return scores[key].float()
    role_logit = get_role_tensor(scores, "role_logit", role_name, n, default=0.0)
    tumor_logit = scores["tumor_logit"].float()
    return role_logit - tumor_logit


def select_positive_proposal_pool(
    scores: Dict[str, torch.Tensor],
    ctx_scores: Dict[str, torch.Tensor],
    args,
):
    tumor_prob = scores["tumor_prob"].float()
    tumor_gap = scores["tumor_gap"].float()
    top1_gap = scores["top1_gap"].float()

    neighbor_gap_mean = ctx_scores["neighbor_gap_mean"].float()
    neighbor_gap_max = ctx_scores["neighbor_gap_max"].float()
    pos_context_score = ctx_scores["pos_context_score"].float()

    n = len(tumor_prob)
    if n == 0:
        return [], {}

    weak_mask = torch.ones(n, dtype=torch.bool)

    if getattr(args, "pos_pool_min_tumor_prob", -1e6) > -1e5:
        weak_mask &= (tumor_prob >= float(args.pos_pool_min_tumor_prob))
    if getattr(args, "pos_pool_min_center_gap", -1e6) > -1e5:
        weak_mask &= (tumor_gap >= float(args.pos_pool_min_center_gap))
    if getattr(args, "pos_pool_min_neighbor_gap_mean", -1e6) > -1e5:
        weak_mask &= (neighbor_gap_mean >= float(args.pos_pool_min_neighbor_gap_mean))

    # Optional: prevent positive proposal pool from being dominated by lymphoid-like patches.
    pos_exclude_roles = list(getattr(args, "pos_pool_exclude_role_names", []))
    if pos_exclude_roles:
        max_excl_prob = get_max_role_tensor(scores, "role_prob", pos_exclude_roles, n, default=0.0)
        max_excl_margin = torch.stack([
            get_margin_role_over_tumor(scores, r, n, default=-1e6)
            for r in pos_exclude_roles
        ], dim=0).max(dim=0).values
        if getattr(args, "pos_pool_max_exclude_role_prob", 1e6) < 1e5:
            weak_mask &= (max_excl_prob <= float(args.pos_pool_max_exclude_role_prob))
        if getattr(args, "pos_pool_max_exclude_role_margin_over_tumor", 1e6) < 1e5:
            weak_mask &= (max_excl_margin <= float(args.pos_pool_max_exclude_role_margin_over_tumor))

    if weak_mask.sum().item() == 0:
        weak_mask = torch.ones(n, dtype=torch.bool)

    tumor_pool_score = (
        float(getattr(args, "pos_pool_tumor_gap_weight", 1.0)) * tumor_gap
        + float(getattr(args, "pos_pool_tumor_prob_weight", 0.5)) * tumor_prob
        + float(getattr(args, "pos_pool_top1_gap_weight", 0.2)) * top1_gap
    )

    tumor_branch_idx = _topk_from_mask(
        score=tumor_pool_score,
        valid_mask=weak_mask,
        k=int(args.pos_pool_topk_tumor),
    )

    support_pool_score = (
        float(getattr(args, "pos_pool_context_weight", 1.0)) * pos_context_score
        + float(getattr(args, "pos_pool_neighbor_gap_mean_weight", 0.3)) * neighbor_gap_mean
        + float(getattr(args, "pos_pool_neighbor_gap_max_weight", 0.2)) * neighbor_gap_max
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
        "pos_pool_valid_before_topk": int(weak_mask.sum().item()),
    }
    return merged_idx, info


def build_negative_exclusion_mask(
    scores: Dict[str, torch.Tensor],
    args,
    n: int,
    prefix: str = "cond_neg",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Explicit role-aware exclusion.

    Main use case for CAMELYON17:
      exclude_roles = ["lymphoid_tissue"]
      A negative candidate may be tumor-like, but should not be strongly lymphoid-like,
      otherwise it becomes a dangerous dense-lymphoid hard negative.
    """
    valid = torch.ones(n, dtype=torch.bool)
    debug = {}

    exclude_roles = list(getattr(args, f"{prefix}_exclude_role_names", []))
    if len(exclude_roles) == 0:
        # Backward compatibility: allow a single lymphoid name without list.
        lymphoid_name = getattr(args, f"{prefix}_lymphoid_role_name", None)
        if lymphoid_name is not None:
            exclude_roles = [str(lymphoid_name)]

    if len(exclude_roles) == 0:
        debug["max_exclude_role_prob"] = torch.zeros(n, dtype=torch.float32)
        debug["max_exclude_role_logit"] = torch.zeros(n, dtype=torch.float32)
        debug["max_exclude_role_margin_over_tumor"] = torch.full((n,), -1e6, dtype=torch.float32)
        return valid, debug

    max_excl_prob = get_max_role_tensor(scores, "role_prob", exclude_roles, n, default=0.0)
    max_excl_logit = get_max_role_tensor(scores, "role_logit", exclude_roles, n, default=-1e6)
    max_excl_margin = torch.stack([
        get_margin_role_over_tumor(scores, r, n, default=-1e6)
        for r in exclude_roles
    ], dim=0).max(dim=0).values

    debug["max_exclude_role_prob"] = max_excl_prob
    debug["max_exclude_role_logit"] = max_excl_logit
    debug["max_exclude_role_margin_over_tumor"] = max_excl_margin

    max_prob_thr = getattr(args, f"{prefix}_max_exclude_role_prob", 1e6)
    max_margin_thr = getattr(args, f"{prefix}_max_exclude_role_margin_over_tumor", 1e6)
    max_logit_thr = getattr(args, f"{prefix}_max_exclude_role_logit", 1e6)

    if max_prob_thr < 1e5:
        valid &= (max_excl_prob <= float(max_prob_thr))
    if max_margin_thr < 1e5:
        valid &= (max_excl_margin <= float(max_margin_thr))
    if max_logit_thr < 1e5:
        valid &= (max_excl_logit <= float(max_logit_thr))

    return valid, debug


def select_negative_fixed_candidates(
    scores: Dict[str, torch.Tensor],
    ctx_scores: Dict[str, torch.Tensor],
    args,
):
    tumor_prob = scores["tumor_prob"].float()
    tumor_gap = scores["tumor_gap"].float()
    top1_gap = scores["top1_gap"].float()

    neighbor_gap_mean = ctx_scores["neighbor_gap_mean"].float()
    neighbor_gap_max = ctx_scores["neighbor_gap_max"].float()
    neighbor_prob_mean = ctx_scores["neighbor_prob_mean"].float()
    neighbor_prob_max = ctx_scores["neighbor_prob_max"].float()
    isolation_score = ctx_scores["isolation_score"].float()
    consistency_score = ctx_scores["consistency_score"].float()
    neg_context_score = ctx_scores["neg_context_score"].float()

    n = len(tumor_prob)
    if n == 0:
        return [], {}

    use_conditional = bool(getattr(args, "use_conditional_neg_selection", False))

    if not use_conditional:
        valid = torch.ones(n, dtype=torch.bool)

        if getattr(args, "neg_min_tumor_prob", -1e6) > -1e5:
            valid &= (tumor_prob >= float(args.neg_min_tumor_prob))
        if getattr(args, "neg_min_center_gap", -1e6) > -1e5:
            valid &= (tumor_gap >= float(args.neg_min_center_gap))
        if getattr(args, "neg_max_neighbor_gap", 1e6) < 1e5:
            valid &= (neighbor_gap_mean <= float(args.neg_max_neighbor_gap))
        if getattr(args, "neg_max_neighbor_gap_max", 1e6) < 1e5:
            valid &= (neighbor_gap_max <= float(args.neg_max_neighbor_gap_max))
        if getattr(args, "neg_min_top1_gap", -1e6) > -1e5:
            valid &= (top1_gap >= float(args.neg_min_top1_gap))

        excl_valid, excl_debug = build_negative_exclusion_mask(scores, args, n, prefix="neg")
        valid &= excl_valid

        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        used_fallback = False
        fallback_level = 0

        if valid_idx.numel() == 0:
            fallback = torch.ones(n, dtype=torch.bool)
            if getattr(args, "neg_fallback_min_tumor_prob", -1e6) > -1e5:
                fallback &= (tumor_prob >= float(args.neg_fallback_min_tumor_prob))
            if getattr(args, "neg_fallback_min_center_gap", -1e6) > -1e5:
                fallback &= (tumor_gap >= float(args.neg_fallback_min_center_gap))
            # Keep exclusion even in fallback unless explicitly disabled.
            if bool(getattr(args, "neg_keep_exclusion_in_fallback", True)):
                fallback &= excl_valid

            valid_idx = torch.nonzero(fallback, as_tuple=False).squeeze(-1)
            used_fallback = True
            fallback_level = 1

            if valid_idx.numel() == 0:
                # Last fallback: still prefer non-excluded if possible.
                if bool(getattr(args, "neg_keep_exclusion_in_last_fallback", True)) and excl_valid.sum().item() > 0:
                    valid_idx = torch.nonzero(excl_valid, as_tuple=False).squeeze(-1)
                else:
                    valid_idx = torch.arange(n)
                used_fallback = True
                fallback_level = 2

        k = min(int(args.neg_topk), int(valid_idx.numel()))
        if k <= 0:
            selected_idx = torch.empty(0, dtype=torch.long)
        else:
            local = torch.topk(neg_context_score[valid_idx], k=k, dim=0).indices
            selected_idx = valid_idx[local]
            order = torch.argsort(neg_context_score[selected_idx], descending=True)
            selected_idx = selected_idx[order]

        info = {
            "selected_idx": selected_idx.tolist(),
            "used_fallback": used_fallback,
            "fallback_level": int(fallback_level),
            "neg_context_score": neg_context_score,
            "cond_neg_score": neg_context_score,
            "neg_select_mode": "old_with_role_exclusion",
            "num_valid_before_topk": int(valid_idx.numel()),
            **excl_debug,
        }
        return selected_idx.tolist(), info

    # -----------------------------------------------------
    # Conditional / isolation-aware / role-exclusion hard-negative selection
    # -----------------------------------------------------
    valid = torch.ones(n, dtype=torch.bool)

    if getattr(args, "cond_neg_min_tumor_prob", -1e6) > -1e5:
        valid &= (tumor_prob >= float(args.cond_neg_min_tumor_prob))
    if getattr(args, "cond_neg_min_center_gap", -1e6) > -1e5:
        valid &= (tumor_gap >= float(args.cond_neg_min_center_gap))
    if getattr(args, "cond_neg_max_center_gap", 1e6) < 1e5:
        valid &= (tumor_gap <= float(args.cond_neg_max_center_gap))
    if getattr(args, "cond_neg_min_top1_gap", -1e6) > -1e5:
        valid &= (top1_gap >= float(args.cond_neg_min_top1_gap))

    if getattr(args, "cond_neg_max_neighbor_gap_mean", 1e6) < 1e5:
        valid &= (neighbor_gap_mean <= float(args.cond_neg_max_neighbor_gap_mean))
    if getattr(args, "cond_neg_max_neighbor_gap_max", 1e6) < 1e5:
        valid &= (neighbor_gap_max <= float(args.cond_neg_max_neighbor_gap_max))
    if getattr(args, "cond_neg_max_neighbor_prob_mean", 1e6) < 1e5:
        valid &= (neighbor_prob_mean <= float(args.cond_neg_max_neighbor_prob_mean))
    if getattr(args, "cond_neg_max_neighbor_prob_max", 1e6) < 1e5:
        valid &= (neighbor_prob_max <= float(args.cond_neg_max_neighbor_prob_max))

    if getattr(args, "cond_neg_min_isolation", -1e6) > -1e5:
        valid &= (isolation_score >= float(args.cond_neg_min_isolation))
    if getattr(args, "cond_neg_max_isolation", 1e6) < 1e5:
        valid &= (isolation_score <= float(args.cond_neg_max_isolation))

    if getattr(args, "cond_neg_max_consistency", 1e6) < 1e5:
        valid &= (consistency_score <= float(args.cond_neg_max_consistency))

    excl_valid, excl_debug = build_negative_exclusion_mask(scores, args, n, prefix="cond_neg")
    valid &= excl_valid

    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    used_fallback = False
    fallback_level = 0

    if valid_idx.numel() == 0:
        fallback_level = 1
        used_fallback = True
        valid = torch.ones(n, dtype=torch.bool)

        if getattr(args, "cond_neg_fb_min_tumor_prob", -1e6) > -1e5:
            valid &= (tumor_prob >= float(args.cond_neg_fb_min_tumor_prob))
        if getattr(args, "cond_neg_fb_min_center_gap", -1e6) > -1e5:
            valid &= (tumor_gap >= float(args.cond_neg_fb_min_center_gap))
        if getattr(args, "cond_neg_fb_max_center_gap", 1e6) < 1e5:
            valid &= (tumor_gap <= float(args.cond_neg_fb_max_center_gap))
        if getattr(args, "cond_neg_fb_max_neighbor_gap_mean", 1e6) < 1e5:
            valid &= (neighbor_gap_mean <= float(args.cond_neg_fb_max_neighbor_gap_mean))
        if getattr(args, "cond_neg_fb_min_isolation", -1e6) > -1e5:
            valid &= (isolation_score >= float(args.cond_neg_fb_min_isolation))
        if getattr(args, "cond_neg_fb_max_isolation", 1e6) < 1e5:
            valid &= (isolation_score <= float(args.cond_neg_fb_max_isolation))

        # Still exclude lymphoid-like patches in fallback by default.
        if bool(getattr(args, "cond_neg_keep_exclusion_in_fallback", True)):
            valid &= excl_valid

        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)

    if valid_idx.numel() == 0:
        fallback_level = 2
        used_fallback = True
        # Final safety guard:
        # For CAMELYON17 safe-negative mining, do not globally fall back to all patches,
        # otherwise the selector returns the most tumor-like dense lymphoid/cellular regions again.
        if bool(getattr(args, "disable_last_global_fallback", False)):
            valid_idx = torch.empty(0, dtype=torch.long)
        elif bool(getattr(args, "cond_neg_keep_exclusion_in_last_fallback", True)) and excl_valid.sum().item() > 0:
            valid_idx = torch.nonzero(excl_valid, as_tuple=False).squeeze(-1)
        else:
            valid_idx = torch.arange(n)

    max_exclude_role_prob = excl_debug.get("max_exclude_role_prob", torch.zeros(n, dtype=torch.float32))
    max_exclude_margin = excl_debug.get("max_exclude_role_margin_over_tumor", torch.full((n,), -1e6, dtype=torch.float32))

    cond_neg_score = (
        float(getattr(args, "cond_neg_center_weight", 1.0)) * robust_zscore_torch(tumor_gap)
        + float(getattr(args, "cond_neg_prob_weight", 0.25)) * robust_zscore_torch(tumor_prob)
        + float(getattr(args, "cond_neg_isolation_weight", 1.5)) * robust_zscore_torch(isolation_score)
        - float(getattr(args, "cond_neg_neighbor_mean_penalty", 1.0)) * robust_zscore_torch(neighbor_gap_mean)
        - float(getattr(args, "cond_neg_neighbor_max_penalty", 0.5)) * robust_zscore_torch(neighbor_gap_max)
        - float(getattr(args, "cond_neg_consistency_penalty", 0.5)) * robust_zscore_torch(consistency_score)
        - float(getattr(args, "cond_neg_exclude_role_prob_penalty", 1.5)) * robust_zscore_torch(max_exclude_role_prob)
        - float(getattr(args, "cond_neg_exclude_role_margin_penalty", 1.0)) * robust_zscore_torch(max_exclude_margin)
    )

    k = min(int(args.neg_topk), int(valid_idx.numel()))
    if k <= 0:
        selected_idx = torch.empty(0, dtype=torch.long)
    else:
        local = torch.topk(cond_neg_score[valid_idx], k=k, dim=0).indices
        selected_idx = valid_idx[local]
        order = torch.argsort(cond_neg_score[selected_idx], descending=True)
        selected_idx = selected_idx[order]

    info = {
        "selected_idx": selected_idx.tolist(),
        "used_fallback": used_fallback,
        "fallback_level": int(fallback_level),
        "neg_context_score": neg_context_score,
        "cond_neg_score": cond_neg_score,
        "neg_select_mode": "conditional_isolation_role_exclusion",
        "num_valid_before_topk": int(valid_idx.numel()),
        **excl_debug,
    }
    return selected_idx.tolist(), info


# =========================================================
# row builders
# =========================================================
def add_role_columns_to_row(row: Dict[str, Any], scores: Dict[str, torch.Tensor], idx_t: int, role_names: List[str]):
    for rname in role_names:
        pk = f"role_prob__{rname}"
        lk = f"role_logit__{rname}"
        if pk in scores:
            row[pk] = safe_float(scores[pk][idx_t])
        if lk in scores:
            row[lk] = safe_float(scores[lk][idx_t])
        mk = f"margin_{rname}_over_tumor"
        if mk in scores:
            row[mk] = safe_float(scores[mk][idx_t])
    if "pred_role_idx" in scores:
        row["pred_role_idx"] = int(scores["pred_role_idx"][idx_t])
    if "pred_role_name" in scores:
        row["pred_role_name"] = str(scores["pred_role_name"][idx_t])


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
    role_names: List[str],
):
    rows = []

    tumor_pool_score = select_info["tumor_pool_score"]
    support_pool_score = select_info["support_pool_score"]

    if len(selected_idx) == 0:
        return rows

    merged_tensor = torch.as_tensor(selected_idx, dtype=torch.long)
    merged_score = tumor_pool_score[merged_tensor] + 0.5 * support_pool_score[merged_tensor]

    order = torch.argsort(merged_score, descending=True)
    ordered_idx = merged_tensor[order].tolist()
    branch_tag = select_info["branch_tag"]

    for rank, idx_t in enumerate(ordered_idx):
        coord = coords[idx_t]
        patch_idx = int(patch_indices[idx_t])

        row = {
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
            "neg_logit": safe_float(scores["neg_logit"][idx_t]) if "neg_logit" in scores else float("nan"),

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
            "pos_pool_valid_before_topk": int(select_info.get("pos_pool_valid_before_topk", -1)),

            "num_patches_scored": int(len(coords)),
            "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
            "svs_path": svs_path,
            "h5_path": h5_path,
        }
        add_role_columns_to_row(row, scores, idx_t, role_names)
        rows.append(row)

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
    role_names: List[str],
):
    rows = []
    neg_context_score = select_info["neg_context_score"]
    cond_neg_score = select_info.get("cond_neg_score", neg_context_score)
    neg_select_mode = select_info.get("neg_select_mode", "old")
    fallback_level = int(select_info.get("fallback_level", 0))
    num_valid_before_topk = int(select_info.get("num_valid_before_topk", -1))
    used_fallback = bool(select_info.get("used_fallback", False))
    max_exclude_role_prob = select_info.get("max_exclude_role_prob", torch.zeros(len(coords), dtype=torch.float32))
    max_exclude_role_logit = select_info.get("max_exclude_role_logit", torch.zeros(len(coords), dtype=torch.float32))
    max_exclude_role_margin = select_info.get("max_exclude_role_margin_over_tumor", torch.full((len(coords),), -1e6, dtype=torch.float32))

    for rank, idx_t in enumerate(selected_idx):
        coord = coords[idx_t]
        patch_idx = int(patch_indices[idx_t])

        row = {
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
            "neg_logit": safe_float(scores["neg_logit"][idx_t]) if "neg_logit" in scores else float("nan"),

            "neighbor_gap_mean": safe_float(ctx_scores["neighbor_gap_mean"][idx_t]),
            "neighbor_gap_max": safe_float(ctx_scores["neighbor_gap_max"][idx_t]),
            "neighbor_prob_mean": safe_float(ctx_scores["neighbor_prob_mean"][idx_t]),
            "neighbor_prob_max": safe_float(ctx_scores["neighbor_prob_max"][idx_t]),
            "isolation_score": safe_float(ctx_scores["isolation_score"][idx_t]),
            "consistency_score": safe_float(ctx_scores["consistency_score"][idx_t]),
            "pos_context_score": safe_float(ctx_scores["pos_context_score"][idx_t]),
            "neg_context_score": safe_float(ctx_scores["neg_context_score"][idx_t]),
            "cond_neg_score": safe_float(cond_neg_score[idx_t]),
            "neg_select_mode": neg_select_mode,
            "neg_used_fallback": int(used_fallback),
            "neg_fallback_level": fallback_level,
            "neg_num_valid_before_topk": num_valid_before_topk,
            "max_exclude_role_prob": safe_float(max_exclude_role_prob[idx_t]),
            "max_exclude_role_logit": safe_float(max_exclude_role_logit[idx_t]),
            "max_exclude_role_margin_over_tumor": safe_float(max_exclude_role_margin[idx_t]),

            "tumor_pool_score": float("nan"),
            "support_pool_score": float("nan"),
            "proposal_score": safe_float(cond_neg_score[idx_t]),

            "num_patches_scored": int(len(coords)),
            "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
            "svs_path": svs_path,
            "h5_path": h5_path,
        }
        add_role_columns_to_row(row, scores, idx_t, role_names)
        rows.append(row)

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

    scores, ctx_scores, coords, patch_indices, patch_size, patch_level, used_cache = load_or_compute_slide_scores(
        row=row,
        encoder=encoder,
        proj_l12=proj_l12,
        summary_builder=summary_builder,
        transform=transform,
        device=device,
        args=args,
        role_names=role_names,
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
            role_names=role_names,
        )
        sel_t = torch.as_tensor(selected_idx, dtype=torch.long)
        summary = {
            "slide_id": slide_id,
            "split": split,
            "label": label,
            "output_type": "positive_proposal_pool",
            "used_score_cache": int(used_cache),
            "num_patches_scored": int(len(coords)),
            "num_selected": int(len(selected_idx)),
            "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
            "mean_tumor_prob_all": safe_mean(scores["tumor_prob"]),
            "mean_tumor_gap_all": safe_mean(scores["tumor_gap"]),
            "mean_pos_context_all": safe_mean(ctx_scores["pos_context_score"]),
            "mean_selected_tumor_gap": safe_mean(scores["tumor_gap"][sel_t]) if len(selected_idx) > 0 else 0.0,
            "mean_selected_pos_context": safe_mean(ctx_scores["pos_context_score"][sel_t]) if len(selected_idx) > 0 else 0.0,
            "num_tumor_branch": int(len(select_info["tumor_branch_idx"])),
            "num_context_branch": int(len(select_info["context_branch_idx"])),
            "pos_pool_valid_before_topk": int(select_info.get("pos_pool_valid_before_topk", -1)),
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
        role_names=role_names,
    )
    sel_t = torch.as_tensor(selected_idx, dtype=torch.long)
    cond_score = select_info.get("cond_neg_score", ctx_scores["neg_context_score"])
    max_excl_prob = select_info.get("max_exclude_role_prob", torch.zeros(len(coords), dtype=torch.float32))
    max_excl_margin = select_info.get("max_exclude_role_margin_over_tumor", torch.full((len(coords),), -1e6, dtype=torch.float32))
    summary = {
        "slide_id": slide_id,
        "split": split,
        "label": label,
        "output_type": "negative_fixed_candidates",
        "used_score_cache": int(used_cache),
        "num_patches_scored": int(len(coords)),
        "num_selected": int(len(selected_idx)),
        "neighbor_num_used_mean": float(ctx_scores["neighbor_num_used_mean"]),
        "mean_tumor_prob_all": safe_mean(scores["tumor_prob"]),
        "mean_tumor_gap_all": safe_mean(scores["tumor_gap"]),
        "mean_neg_context_all": safe_mean(ctx_scores["neg_context_score"]),
        "mean_selected_tumor_gap": safe_mean(scores["tumor_gap"][sel_t]) if len(selected_idx) > 0 else 0.0,
        "mean_selected_neg_context": safe_mean(ctx_scores["neg_context_score"][sel_t]) if len(selected_idx) > 0 else 0.0,
        "mean_selected_cond_neg_score": safe_mean(cond_score[sel_t]) if len(selected_idx) > 0 else 0.0,
        "mean_selected_max_exclude_role_prob": safe_mean(max_excl_prob[sel_t]) if len(selected_idx) > 0 else 0.0,
        "mean_selected_max_exclude_role_margin_over_tumor": safe_mean(max_excl_margin[sel_t]) if len(selected_idx) > 0 else 0.0,
        "used_fallback": bool(select_info["used_fallback"]),
        "fallback_level": int(select_info.get("fallback_level", 0)),
        "neg_select_mode": select_info.get("neg_select_mode", "old"),
        "num_valid_before_topk": int(select_info.get("num_valid_before_topk", -1)),
    }
    return "negative", rows, summary


# =========================================================
# defaults
# =========================================================
def set_default_args(args):
    defaults = {
        "show_inner_patch_progress": False,
        "random_sample_patches": False,
        "reuse_score_cache": True,
        "score_cache_dir": None,

        "neighbor_radius": 1024.0,
        "neighbor_min_count": 4,
        "neighbor_k_fallback": 8,

        "pos_ctx_center_weight": 1.0,
        "pos_ctx_neighbor_weight": 1.0,
        "pos_ctx_neighbor_max_weight": 0.25,
        "pos_ctx_prob_weight": 0.25,
        "pos_ctx_top1_weight": 0.10,
        "pos_ctx_consistency_weight": 0.30,

        "neg_ctx_center_weight": 1.0,
        "neg_ctx_neighbor_weight": 1.0,
        "neg_ctx_isolation_weight": 0.60,
        "neg_ctx_prob_weight": 0.20,
        "neg_ctx_top1_weight": 0.10,

        "pos_pool_topk_tumor": 64,
        "pos_pool_topk_context": 64,
        "pos_pool_max_merged": 128,
        "pos_pool_min_tumor_prob": 0.28,
        "pos_pool_min_center_gap": -0.05,
        "pos_pool_min_neighbor_gap_mean": -1e6,
        "pos_pool_tumor_gap_weight": 1.0,
        "pos_pool_tumor_prob_weight": 0.5,
        "pos_pool_top1_gap_weight": 0.2,
        "pos_pool_context_weight": 1.0,
        "pos_pool_neighbor_gap_mean_weight": 0.3,
        "pos_pool_neighbor_gap_max_weight": 0.2,
        "pos_pool_exclude_role_names": [],
        "pos_pool_max_exclude_role_prob": 1e6,
        "pos_pool_max_exclude_role_margin_over_tumor": 1e6,

        "neg_topk": 16,
        "neg_min_tumor_prob": 0.33,
        "neg_min_center_gap": 0.00,
        "neg_max_neighbor_gap": 0.00,
        "neg_max_neighbor_gap_max": 0.05,
        "neg_min_top1_gap": -1e6,
        "neg_fallback_min_tumor_prob": 0.30,
        "neg_fallback_min_center_gap": -0.03,
        "neg_exclude_role_names": [],
        "neg_max_exclude_role_prob": 1e6,
        "neg_max_exclude_role_margin_over_tumor": 1e6,
        "neg_max_exclude_role_logit": 1e6,
        "neg_keep_exclusion_in_fallback": True,
        "neg_keep_exclusion_in_last_fallback": True,

        "use_conditional_neg_selection": False,
        "cond_neg_min_tumor_prob": 0.33,
        "cond_neg_min_center_gap": 0.00,
        "cond_neg_max_center_gap": 1e6,
        "cond_neg_min_top1_gap": -1e6,
        "cond_neg_max_neighbor_gap_mean": -0.02,
        "cond_neg_max_neighbor_gap_max": 0.02,
        "cond_neg_max_neighbor_prob_mean": 0.34,
        "cond_neg_max_neighbor_prob_max": 0.37,
        "cond_neg_min_isolation": 0.03,
        "cond_neg_max_isolation": 1e6,
        "cond_neg_max_consistency": 0.03,
        "cond_neg_fb_min_tumor_prob": 0.30,
        "cond_neg_fb_min_center_gap": -0.03,
        "cond_neg_fb_max_center_gap": 1e6,
        "cond_neg_fb_max_neighbor_gap_mean": 0.02,
        "cond_neg_fb_min_isolation": 0.00,
        "cond_neg_fb_max_isolation": 1e6,
        "disable_last_global_fallback": False,
        "allow_empty_negative_selection": False,
        "cond_neg_center_weight": 1.0,
        "cond_neg_prob_weight": 0.25,
        "cond_neg_isolation_weight": 1.5,
        "cond_neg_neighbor_mean_penalty": 1.0,
        "cond_neg_neighbor_max_penalty": 0.5,
        "cond_neg_consistency_penalty": 0.5,
        "cond_neg_exclude_role_names": [],
        "cond_neg_max_exclude_role_prob": 1e6,
        "cond_neg_max_exclude_role_margin_over_tumor": 1e6,
        "cond_neg_max_exclude_role_logit": 1e6,
        "cond_neg_exclude_role_prob_penalty": 1.5,
        "cond_neg_exclude_role_margin_penalty": 1.0,
        "cond_neg_keep_exclusion_in_fallback": True,
        "cond_neg_keep_exclusion_in_last_fallback": True,

        "use_last_moe_output": True,
        "max_patches_per_slide": None,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        "Build positive proposal pool + negative fixed candidates with conditional/role-aware negative selection"
    )
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
    if args.score_cache_dir is not None and str(args.score_cache_dir).strip() != "":
        ensure_dir(args.score_cache_dir)

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
        f"[ScoreCache] dir={args.score_cache_dir}, reuse={args.reuse_score_cache}"
    )
    print(
        f"[PositivePool] topk_tumor={args.pos_pool_topk_tumor}, "
        f"topk_context={args.pos_pool_topk_context}, "
        f"max_merged={args.pos_pool_max_merged}"
    )
    print(
        f"[NegativeFixed] neg_topk={args.neg_topk}, "
        f"conditional={args.use_conditional_neg_selection}, "
        f"cond_exclude_roles={getattr(args, 'cond_neg_exclude_role_names', [])}"
    )

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
                    cache=summary.get("used_score_cache", 0),
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
