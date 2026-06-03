#!/usr/bin/env python3

from __future__ import annotations

import argparse

import hashlib

import json

import math

import os

import sys

from pathlib import Path

from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import openslide

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision.transforms.v2 as T

from PIL import ImageFile

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

from tqdm import tqdm

import yaml

try:

    import umap  # type: ignore

    HAS_UMAP = True

except Exception:

    HAS_UMAP = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

if PROJECT_ROOT not in sys.path:

    sys.path.insert(0, PROJECT_ROOT)

# Adjust this when running outside the repo root.

if os.path.exists(os.path.join(PROJECT_ROOT, "models")):

    REPO_ROOT = PROJECT_ROOT

else:

    REPO_ROOT = os.getcwd()

if REPO_ROOT not in sys.path:

    sys.path.insert(0, REPO_ROOT)

from models.encoders.moe_encoder import MoEEncoder

from models.plugins.shared_role_prototype import (

    SharedRolePrototype,

    PatchRoleSummaryFromSharedProto,

)

# =========================================================

# utils

# =========================================================

def ensure_dir(path: str | Path):

    Path(path).mkdir(parents=True, exist_ok=True)

def stable_slide_seed(base_seed: int, slide_id: str) -> int:

    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()

    return int(base_seed) + (int(h[:8], 16) % 100000)

def set_seed(seed: int = 42):

    import random

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

def _norm_path(p: str) -> str:

    return os.path.normpath(os.path.expanduser(str(p)))

def build_transform(img_size: int = 224):

    return T.Compose([

        T.ToImage(),

        T.Resize((img_size, img_size), antialias=True),

        T.ToDtype(torch.float32, scale=True),

    ])

def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:

    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps

    return x / denom

# =========================================================

# path / csv helpers

# =========================================================

def read_split_csv(csv_path: str, split_names: Sequence[str]) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    if "slide_id" not in df.columns and "image_id" in df.columns:

        df["slide_id"] = df["image_id"]

    if "label" not in df.columns:

        if "slide_binary_label" in df.columns:

            df["label"] = df["slide_binary_label"]

        else:

            raise ValueError("split csv missing label / slide_binary_label")

    required = {"slide_id", "label", "split"}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(f"split csv missing columns: {missing}")

    return df[df["split"].isin(split_names)].copy().reset_index(drop=True)

def stratified_sample(df: pd.DataFrame, per_split_total: int, seed: int) -> pd.DataFrame:

    if per_split_total is None or per_split_total <= 0:

        return df.copy().reset_index(drop=True)

    parts = []

    for split, sub in df.groupby("split"):

        if len(sub) <= per_split_total:

            parts.append(sub.copy())

            continue

        labels = sorted(sub["label"].unique().tolist())

        total = len(sub)

        alloc: Dict[int, int] = {}

        for y in labels:

            alloc[y] = max(1, round(per_split_total * int((sub["label"] == y).sum()) / total))

        cur = sum(alloc.values())

        while cur > per_split_total:

            y = max(alloc, key=alloc.get)

            if alloc[y] > 1:

                alloc[y] -= 1

                cur -= 1

            else:

                break

        while cur < per_split_total:

            y = max(labels, key=lambda yy: int((sub["label"] == yy).sum()) - alloc.get(yy, 0))

            alloc[y] += 1

            cur += 1

        split_parts = []

        for y in labels:

            ss = sub[sub["label"] == y]

            k = min(len(ss), alloc[y])

            split_parts.append(ss.sample(n=k, random_state=seed))

        parts.append(pd.concat(split_parts, axis=0))

    out = pd.concat(parts, axis=0).drop_duplicates("slide_id").reset_index(drop=True)

    return out

def find_svs_path_from_root(svs_root: str, slide_id: str, project: Optional[str] = None) -> str:

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

def find_h5_path_from_root(h5_root: str, slide_id: str) -> str:

    h5_root = _norm_path(h5_root)

    slide_id = str(slide_id)

    barcode_prefix = slide_id.split(".")[0]

    direct_names = [f"{slide_id}.h5", f"{barcode_prefix}.h5"]

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

        raise FileNotFoundError(f"Cannot find H5 for slide_id={slide_id} under h5_root={h5_root}")

    candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))

    return candidates[0][1]

def resolve_paths(df: pd.DataFrame, svs_root: Optional[str], h5_root: Optional[str], project: Optional[str]) -> pd.DataFrame:

    df = df.copy()

    if "svs_path" not in df.columns:

        df["svs_path"] = ""

    if "h5_path" not in df.columns:

        df["h5_path"] = ""

    svs_cache: Dict[str, str] = {}

    h5_cache: Dict[str, str] = {}

    resolved_svs, resolved_h5 = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resolve svs/h5 paths", leave=False):

        slide_id = str(row["slide_id"])

        svs_path = str(row.get("svs_path", "") or "")

        h5_path = str(row.get("h5_path", "") or "")

        if svs_path and os.path.exists(svs_path):

            final_svs = svs_path

        else:

            if svs_root is None:

                raise ValueError(f"Missing svs_path for {slide_id} and no svs_root provided")

            if slide_id not in svs_cache:

                svs_cache[slide_id] = find_svs_path_from_root(svs_root, slide_id, project)

            final_svs = svs_cache[slide_id]

        if h5_path and os.path.exists(h5_path):

            final_h5 = h5_path

        else:

            if h5_root is None:

                raise ValueError(f"Missing h5_path for {slide_id} and no h5_root provided")

            if slide_id not in h5_cache:

                h5_cache[slide_id] = find_h5_path_from_root(h5_root, slide_id)

            final_h5 = h5_cache[slide_id]

        resolved_svs.append(final_svs)

        resolved_h5.append(final_h5)

    df["svs_path"] = resolved_svs

    df["h5_path"] = resolved_h5

    return df

# =========================================================

# model loading / feature extraction

# =========================================================

def build_encoder_from_stage2(base_encoder_cfg, moe_encoder_cfg, stage2_full_ckpt: str, device: str):

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

def load_proj_l12_from_stage2(stage2_full_ckpt: str, device: str) -> nn.Module:

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

def sample_coords_from_h5(h5_path: str, max_patches_per_slide: Optional[int], random_sample: bool, seed: int):

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

def read_patch_batch(slide: openslide.OpenSlide, coords: np.ndarray, patch_size: int, patch_level: int, transform):

    imgs = []

    for xy in coords:

        x, y = int(xy[0]), int(xy[1])

        img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")

        imgs.append(transform(img))

    return torch.stack(imgs, dim=0)

@torch.no_grad()

def extract_patch_features_dual(

    encoder: nn.Module,

    patch_imgs: torch.Tensor,

) -> Tuple[torch.Tensor, torch.Tensor]:

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

    if "layer_12" not in feature_dict:

        raise KeyError(f"'layer_12' not found in feature_dict keys={list(feature_dict.keys())}")

    if len(moe_feature_list) == 0:

        raise RuntimeError("moe_feature_list is empty; cannot extract scoring-layer adapted feature")

    backbone_tokens = feature_dict["layer_12"]

    adapted_tokens = moe_feature_list[-1]

    backbone_patch_tokens = backbone_tokens[:, 1:, :]

    adapted_patch_tokens = adapted_tokens[:, 1:, :]

    if backbone_patch_tokens.shape[1] == 0 or adapted_patch_tokens.shape[1] == 0:

        raise RuntimeError(

            f"No patch tokens found. backbone={tuple(backbone_patch_tokens.shape)} adapted={tuple(adapted_patch_tokens.shape)}"

        )

    backbone_feat = backbone_patch_tokens.mean(dim=1)

    adapted_feat = adapted_patch_tokens.mean(dim=1)

    return backbone_feat, adapted_feat

@torch.no_grad()

def extract_one_slide_dual_features(

    encoder: nn.Module,

    svs_path: str,

    h5_path: str,

    transform,

    device: str,

    patch_batch_size: int,

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

    backbone_feats, adapted_feats = [], []

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

            imgs = read_patch_batch(slide, coord_chunk, patch_size, patch_level, transform).to(device, non_blocking=True)

            feat_backbone, feat_adapted = extract_patch_features_dual(encoder, imgs)

            backbone_feats.append(feat_backbone.cpu())

            adapted_feats.append(feat_adapted.cpu())

    finally:

        slide.close()

    backbone_feat = torch.cat(backbone_feats, dim=0)

    adapted_feat = torch.cat(adapted_feats, dim=0)

    return backbone_feat, adapted_feat, coords, patch_indices, patch_size, patch_level

# =========================================================

# role scoring on scoring layer

# =========================================================

@torch.no_grad()

def score_patch_features_with_role_proto(

    patch_feat_raw: torch.Tensor,

    proj_l12: nn.Module,

    summary_builder: PatchRoleSummaryFromSharedProto,

    role_names: List[str],

    tumor_name: str,

    negative_role_names: List[str],

    device: str,

) -> Dict[str, torch.Tensor]:

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

        raise ValueError(f"No valid negative role names found: {negative_role_names}")

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

        "pred_role_name": pred_role_name,

    }

    for ridx, rname in enumerate(role_names):

        out[f"role_prob__{rname}"] = role_probs[:, ridx]

        out[f"role_logit__{rname}"] = role_logits[:, ridx]

    return out

# =========================================================

# plotting / metrics

# =========================================================

def fit_reducer(X: np.ndarray, seed: int, n_neighbors: int = 15, min_dist: float = 0.1):

    if HAS_UMAP:

        return umap.UMAP(

            n_components=2,

            n_neighbors=n_neighbors,

            min_dist=min_dist,

            metric="cosine",

            random_state=seed,

        ).fit(X)

    return PCA(n_components=2, random_state=seed).fit(X)

def transform_reducer(reducer, X: np.ndarray) -> np.ndarray:

    return reducer.transform(X)

def centroid_distance(X: np.ndarray, y: np.ndarray) -> float:

    if len(np.unique(y)) < 2:

        return float("nan")

    c0 = X[y == 0].mean(axis=0)

    c1 = X[y == 1].mean(axis=0)

    return float(np.linalg.norm(c1 - c0))

def plot_umap_cluster(ax, emb: np.ndarray, cluster_ids: np.ndarray, title: str):

    uniq = sorted(np.unique(cluster_ids).tolist())

    cmap = plt.get_cmap("tab20")

    for i, cid in enumerate(uniq):

        m = cluster_ids == cid

        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.70, color=cmap(i % 20), label=f"c={cid}")

    ax.set_title(title)

    ax.set_xticks([])

    ax.set_yticks([])

def plot_umap_continuous(ax, emb: np.ndarray, values: np.ndarray, title: str):

    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=8, alpha=0.75, cmap="coolwarm")

    ax.set_title(title)

    ax.set_xticks([])

    ax.set_yticks([])

    return sc

# =========================================================

# main

# =========================================================

def main():

    parser = argparse.ArgumentParser("Scoring-layer cluster analysis from raw model forward")

    parser.add_argument("--config", type=str, required=True, help="yaml config with base_encoder / moe_encoder / paths")

    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--splits", nargs="+", default=["train", "val"])

    parser.add_argument("--sample-slides-per-split", type=int, default=20)

    parser.add_argument("--max-patches-per-slide", type=int, default=512)

    parser.add_argument("--patch-batch-size", type=int, default=128)

    parser.add_argument("--random-sample-patches", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch-clusters", type=int, default=10)

    parser.add_argument("--slide-clusters", type=int, default=6)

    parser.add_argument("--pca-dim-before-cluster", type=int, default=32)

    parser.add_argument("--show-inner-patch-progress", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:

        cfg = yaml.safe_load(f)

    ensure_dir(args.out_dir)

    set_seed(args.seed)

    split_csv = cfg.get("split_csv")

    if split_csv is None:

        raise ValueError("config missing split_csv")

    svs_root = cfg.get("svs_root", None)

    h5_root = cfg.get("h5_root", None)

    project = cfg.get("project", None)

    stage2_full_ckpt = cfg.get("stage2_full_ckpt")

    if stage2_full_ckpt is None:

        raise ValueError("config missing stage2_full_ckpt")

    role_proto_dir = cfg.get("role_proto_dir")

    if role_proto_dir is None:

        raise ValueError("config missing role_proto_dir")

    role_tau = float(cfg.get("role_tau", 1.0))

    proto_tumor_name = cfg.get("proto_tumor_name")

    proto_negative_role_names = cfg.get("proto_negative_role_names")

    if proto_tumor_name is None or proto_negative_role_names is None:

        raise ValueError("config missing proto_tumor_name / proto_negative_role_names")

    base_encoder_cfg = cfg.get("base_encoder")

    moe_encoder_cfg = cfg.get("moe_encoder")

    if base_encoder_cfg is None or moe_encoder_cfg is None:

        raise ValueError("config missing base_encoder / moe_encoder")

    img_size = int(cfg.get("img_size", 224))

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    # 1) sample slides

    df = read_split_csv(split_csv, args.splits)

    df = stratified_sample(df, args.sample_slides_per_split, args.seed)

    df = resolve_paths(df, svs_root, h5_root, project)

    df.to_csv(Path(args.out_dir) / "sampled_slides.csv", index=False)

    # 2) load model pieces

    transform = build_transform(img_size)

    encoder = build_encoder_from_stage2(base_encoder_cfg, moe_encoder_cfg, stage2_full_ckpt, device)

    proj_l12 = load_proj_l12_from_stage2(stage2_full_ckpt, device)

    shared_role_proto = SharedRolePrototype.from_files(

        role_proto_dir=role_proto_dir,

        normalize=True,

        learnable=False,

        device=device,

    )

    role_names = list(shared_role_proto.role_names)

    summary_builder = PatchRoleSummaryFromSharedProto(

        shared_role_proto=shared_role_proto,

        tau=role_tau,

        use_softmax=True,

    ).to(device)

    summary_builder.eval()

    print(f"[RoleProto] role_names={role_names}")

    print(f"[RoleProto] proto_tumor_name={proto_tumor_name}")

    print(f"[RoleProto] proto_negative_role_names={proto_negative_role_names}")

    # 3) forward selected slides, extract both scoring-layer input/output + scores

    pre_patch_feats, post_patch_feats = [], []

    patch_labels = []

    patch_rows = []

    pre_slide_feats, post_slide_feats = [], []

    slide_meta = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="Extract scoring-layer features")

    for _, row in pbar:

        slide_id = str(row["slide_id"])

        label = int(row["label"])

        split = str(row["split"])

        svs_path = str(row["svs_path"])

        h5_path = str(row["h5_path"])

        slide_seed = stable_slide_seed(args.seed, slide_id)

        feat_pre_t, feat_post_t, coords, patch_indices, patch_size, patch_level = extract_one_slide_dual_features(

            encoder=encoder,

            svs_path=svs_path,

            h5_path=h5_path,

            transform=transform,

            device=device,

            patch_batch_size=args.patch_batch_size,

            max_patches_per_slide=args.max_patches_per_slide,

            random_sample_patches=args.random_sample_patches,

            seed=slide_seed,

            show_inner_progress=args.show_inner_patch_progress,

        )

        score_pre = score_patch_features_with_role_proto(

            feat_pre_t, proj_l12, summary_builder, role_names,

            proto_tumor_name, proto_negative_role_names, device,

        )

        score_post = score_patch_features_with_role_proto(

            feat_post_t, proj_l12, summary_builder, role_names,

            proto_tumor_name, proto_negative_role_names, device,

        )

        feat_pre = feat_pre_t.numpy().astype(np.float32)

        feat_post = feat_post_t.numpy().astype(np.float32)

        pre_patch_feats.append(feat_pre)

        post_patch_feats.append(feat_post)

        patch_labels.extend([label] * len(feat_pre))

        for i in range(len(feat_pre)):

            rr = {

                "slide_id": slide_id,

                "split": split,

                "label": label,

                "coord_x": int(coords[i, 0]),

                "coord_y": int(coords[i, 1]),

                "patch_idx": int(patch_indices[i]),

                "pre_tumor_gap": float(score_pre["tumor_gap"][i]),

                "post_tumor_gap": float(score_post["tumor_gap"][i]),

                "pre_tumor_prob": float(score_pre["tumor_prob"][i]),

                "post_tumor_prob": float(score_post["tumor_prob"][i]),

                "pre_pred_role": str(score_pre["pred_role_name"][i]),

                "post_pred_role": str(score_post["pred_role_name"][i]),

            }

            for rn in role_names:

                rr[f"pre_role_prob__{rn}"] = float(score_pre[f"role_prob__{rn}"][i])

                rr[f"post_role_prob__{rn}"] = float(score_post[f"role_prob__{rn}"][i])

            patch_rows.append(rr)

        pre_slide_feats.append(feat_pre.mean(axis=0))

        post_slide_feats.append(feat_post.mean(axis=0))

        slide_meta.append({

            "slide_id": slide_id,

            "split": split,

            "label": label,

            "pre_mean_tumor_gap": float(score_pre["tumor_gap"].mean()),

            "post_mean_tumor_gap": float(score_post["tumor_gap"].mean()),

            "pre_mean_tumor_prob": float(score_pre["tumor_prob"].mean()),

            "post_mean_tumor_prob": float(score_post["tumor_prob"].mean()),

            "n_instances": int(len(feat_pre)),

        })

        pbar.set_postfix(

            slide=slide_id[:18],

            n=len(feat_pre),

            pre_gap=f"{float(score_pre['tumor_gap'].mean()):.3f}",

            post_gap=f"{float(score_post['tumor_gap'].mean()):.3f}",

        )

    pre_patch_feats = np.concatenate(pre_patch_feats, axis=0)

    post_patch_feats = np.concatenate(post_patch_feats, axis=0)

    patch_labels_np = np.asarray(patch_labels, dtype=np.int64)

    pre_slide_feats = np.stack(pre_slide_feats, axis=0)

    post_slide_feats = np.stack(post_slide_feats, axis=0)

    slide_labels_np = np.asarray([int(x["label"]) for x in slide_meta], dtype=np.int64)

    patch_df = pd.DataFrame(patch_rows)

    slide_df = pd.DataFrame(slide_meta)

    # 4) UMAP by cluster / by score

    patch_all = np.concatenate([pre_patch_feats, post_patch_feats], axis=0)

    patch_pca_dim = min(args.pca_dim_before_cluster, patch_all.shape[1], max(2, patch_all.shape[0] - 1))

    patch_pca = PCA(n_components=patch_pca_dim, random_state=args.seed)

    patch_all_pca = patch_pca.fit_transform(l2_normalize_np(patch_all))

    patch_kmeans = KMeans(n_clusters=args.patch_clusters, random_state=args.seed, n_init=10)

    patch_cluster_all = patch_kmeans.fit_predict(patch_all_pca)

    patch_cluster_pre = patch_cluster_all[:len(pre_patch_feats)]

    patch_cluster_post = patch_cluster_all[len(pre_patch_feats):]

    patch_df["pre_cluster"] = patch_cluster_pre

    patch_df["post_cluster"] = patch_cluster_post

    patch_reducer = fit_reducer(l2_normalize_np(patch_all), args.seed, 15, 0.1)

    pre_patch_emb = transform_reducer(patch_reducer, l2_normalize_np(pre_patch_feats))

    post_patch_emb = transform_reducer(patch_reducer, l2_normalize_np(post_patch_feats))

    patch_df["pre_umap_x"] = pre_patch_emb[:, 0]

    patch_df["pre_umap_y"] = pre_patch_emb[:, 1]

    patch_df["post_umap_x"] = post_patch_emb[:, 0]

    patch_df["post_umap_y"] = post_patch_emb[:, 1]

    patch_df.to_csv(Path(args.out_dir) / "patch_scoring_layer_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_umap_cluster(axes[0], pre_patch_emb, patch_cluster_pre, "Pre-MoE scoring-layer UMAP (cluster)")

    plot_umap_cluster(axes[1], post_patch_emb, patch_cluster_post, "Post-MoE scoring-layer UMAP (cluster)")

    fig.tight_layout()

    fig.savefig(Path(args.out_dir) / "patch_umap_by_cluster.png", dpi=220)

    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc0 = plot_umap_continuous(axes[0], pre_patch_emb, patch_df["pre_tumor_gap"].values, "Pre-MoE scoring-layer UMAP (tumor_gap)")

    sc1 = plot_umap_continuous(axes[1], post_patch_emb, patch_df["post_tumor_gap"].values, "Post-MoE scoring-layer UMAP (tumor_gap)")

    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()

    fig.savefig(Path(args.out_dir) / "patch_umap_by_tumor_gap.png", dpi=220)

    plt.close(fig)

    # slide-level cluster

    slide_all = np.concatenate([pre_slide_feats, post_slide_feats], axis=0)

    slide_pca_dim = min(args.pca_dim_before_cluster, slide_all.shape[1], max(2, slide_all.shape[0] - 1))

    slide_pca = PCA(n_components=slide_pca_dim, random_state=args.seed)

    slide_all_pca = slide_pca.fit_transform(l2_normalize_np(slide_all))

    slide_kmeans = KMeans(n_clusters=args.slide_clusters, random_state=args.seed, n_init=10)

    slide_cluster_all = slide_kmeans.fit_predict(slide_all_pca)

    slide_cluster_pre = slide_cluster_all[:len(pre_slide_feats)]

    slide_cluster_post = slide_cluster_all[len(pre_slide_feats):]

    slide_df["pre_cluster"] = slide_cluster_pre

    slide_df["post_cluster"] = slide_cluster_post

    slide_reducer = fit_reducer(l2_normalize_np(slide_all), args.seed, 10, 0.15)

    pre_slide_emb = transform_reducer(slide_reducer, l2_normalize_np(pre_slide_feats))

    post_slide_emb = transform_reducer(slide_reducer, l2_normalize_np(post_slide_feats))

    slide_df["pre_umap_x"] = pre_slide_emb[:, 0]

    slide_df["pre_umap_y"] = pre_slide_emb[:, 1]

    slide_df["post_umap_x"] = post_slide_emb[:, 0]

    slide_df["post_umap_y"] = post_slide_emb[:, 1]

    slide_df.to_csv(Path(args.out_dir) / "slide_scoring_layer_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_umap_cluster(axes[0], pre_slide_emb, slide_cluster_pre, "Pre-MoE slide UMAP (cluster)")

    plot_umap_cluster(axes[1], post_slide_emb, slide_cluster_post, "Post-MoE slide UMAP (cluster)")

    fig.tight_layout()

    fig.savefig(Path(args.out_dir) / "slide_umap_by_cluster.png", dpi=220)

    plt.close(fig)

    # 5) occupancy + semantics

    patch_occ_rows = []

    for version, cluster_ids in [("pre_moe", patch_cluster_pre), ("post_moe", patch_cluster_post)]:

        vc = pd.Series(cluster_ids).value_counts(normalize=False).sort_index()

        vc_ratio = pd.Series(cluster_ids).value_counts(normalize=True).sort_index()

        for cid in sorted(vc.index.tolist()):

            patch_occ_rows.append({

                "level": "patch",

                "version": version,

                "cluster_id": int(cid),

                "count": int(vc[cid]),

                "ratio": float(vc_ratio[cid]),

            })

    pd.DataFrame(patch_occ_rows).to_csv(Path(args.out_dir) / "patch_cluster_occupancy.csv", index=False)

    slide_occ_rows = []

    for version, cluster_ids in [("pre_moe", slide_cluster_pre), ("post_moe", slide_cluster_post)]:

        vc = pd.Series(cluster_ids).value_counts(normalize=False).sort_index()

        vc_ratio = pd.Series(cluster_ids).value_counts(normalize=True).sort_index()

        for cid in sorted(vc.index.tolist()):

            slide_occ_rows.append({

                "level": "slide",

                "version": version,

                "cluster_id": int(cid),

                "count": int(vc[cid]),

                "ratio": float(vc_ratio[cid]),

            })

    pd.DataFrame(slide_occ_rows).to_csv(Path(args.out_dir) / "slide_cluster_occupancy.csv", index=False)

    # cluster semantics

    sem_rows = []

    for version, cluster_col, gap_col, prob_col in [

        ("pre_moe", "pre_cluster", "pre_tumor_gap", "pre_tumor_prob"),

        ("post_moe", "post_cluster", "post_tumor_gap", "post_tumor_prob"),

    ]:

        for cid, sub in patch_df.groupby(cluster_col):

            row = {

                "version": version,

                "cluster_id": int(cid),

                "n": int(len(sub)),

                "label1_ratio": float((sub["label"] == 1).mean()),

                "mean_tumor_gap": float(sub[gap_col].mean()),

                "median_tumor_gap": float(sub[gap_col].median()),

                "mean_tumor_prob": float(sub[prob_col].mean()),

            }

            role_prefix = "pre_role_prob__" if version == "pre_moe" else "post_role_prob__"

            for rn in role_names:

                col = f"{role_prefix}{rn}"

                row[f"mean_role_prob__{rn}"] = float(sub[col].mean())

            sem_rows.append(row)

    pd.DataFrame(sem_rows).to_csv(Path(args.out_dir) / "patch_cluster_semantics.csv", index=False)

    # representation metrics

    metrics = []

    for level, X_pre, X_post, y in [

        ("patch", pre_patch_feats, post_patch_feats, patch_labels_np),

        ("slide", pre_slide_feats, post_slide_feats, slide_labels_np),

    ]:

        for version, X in [("pre_moe", X_pre), ("post_moe", X_post)]:

            rec = {

                "level": level,

                "version": version,

                "n": int(len(X)),

                "centroid_distance": centroid_distance(X, y),

            }

            if len(np.unique(y)) > 1 and len(X) >= 10:

                try:

                    rec["silhouette"] = float(silhouette_score(X, y, metric="cosine"))

                except Exception:

                    rec["silhouette"] = float("nan")

            else:

                rec["silhouette"] = float("nan")

            metrics.append(rec)

    pd.DataFrame(metrics).to_csv(Path(args.out_dir) / "representation_metrics.csv", index=False)

    summary = {

        "sampled_n_slides": int(len(df)),

        "patch_n_points": int(len(pre_patch_feats)),

        "slide_n_points": int(len(pre_slide_feats)),

        "role_names": role_names,

        "patch_clusters": int(args.patch_clusters),

        "slide_clusters": int(args.slide_clusters),

        "umap_backend": "umap" if HAS_UMAP else "pca",

        "note": "pre_moe = scoring-layer input path (feature_dict['layer_12']); post_moe = scoring-layer adapted path (moe_feature_list[-1]).",

    }

    with open(Path(args.out_dir) / "analysis_summary.json", "w", encoding="utf-8") as f:

        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved analysis to", args.out_dir)

if __name__ == "__main__":

    main()