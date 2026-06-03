#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import yaml
import numpy as np
import pandas as pd
import openslide

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.moe_encoder import MoEEncoder
from models.plugins.shared_role_prototype import SharedRolePrototype, PatchRoleSummaryFromSharedProto
from downstream.role_transfer_losses import (
    compute_online_role_scores,
    select_online_positive_proposals,
    select_online_negative_proposals,
)


# =========================================================
# utils
# =========================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def stable_hash(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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
    svs_root = _norm_path(svs_root)
    search_roots = []

    if project is not None:
        pdir = os.path.join(svs_root, str(project))
        if os.path.isdir(pdir):
            search_roots.append(pdir)

    search_roots.append(svs_root)

    slide_id = str(slide_id)
    barcode_prefix = slide_id.split(".")[0]
    valid_exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")

    candidates = []
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

    if not os.path.isdir(h5_root):
        raise FileNotFoundError(f"h5_root not found: {h5_root}")

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


def resolve_svs_h5_paths(
    df: pd.DataFrame,
    svs_root: Optional[str] = None,
    h5_root: Optional[str] = None,
    project: Optional[str] = None,
) -> pd.DataFrame:
    """
    If df lacks svs_path/h5_path, or paths do not exist, resolve them from svs_root/h5_root.
    CSV paths are preferred when valid.
    """
    df = df.copy()

    if "slide_id" not in df.columns:
        raise ValueError("CSV must contain slide_id column for path resolution.")

    if "svs_path" not in df.columns:
        df["svs_path"] = ""
    if "h5_path" not in df.columns:
        df["h5_path"] = ""

    svs_cache = {}
    h5_cache = {}
    resolved_svs = []
    resolved_h5 = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resolve svs/h5", leave=False):
        slide_id = str(row["slide_id"])
        svs_path = str(row.get("svs_path", "") or "")
        h5_path = str(row.get("h5_path", "") or "")

        if svs_path and os.path.exists(svs_path):
            final_svs = svs_path
        else:
            if svs_root is None:
                raise ValueError(
                    f"svs_path missing/not found for slide_id={slide_id}, and --svs_root is not provided"
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
                    f"h5_path missing/not found for slide_id={slide_id}, and --h5_root is not provided"
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
# dataframe helpers
# =========================================================
def subsample_df_balanced(
    df: pd.DataFrame,
    max_items: Optional[int],
    seed: int = 42,
    label_col: str = "label",
) -> pd.DataFrame:
    if max_items is None or max_items >= len(df):
        return df.reset_index(drop=True).copy()

    rng = np.random.default_rng(seed)

    if label_col not in df.columns:
        idx = rng.choice(len(df), size=max_items, replace=False)
        return df.iloc[idx].reset_index(drop=True).copy()

    parts = []
    groups = list(df.groupby(label_col))
    if len(groups) == 0:
        return df.reset_index(drop=True).copy()

    per_group = max_items // len(groups)
    rem = max_items % len(groups)

    used_ids = set()
    for gi, (_, sub) in enumerate(groups):
        take = min(len(sub), per_group + (1 if gi < rem else 0))
        if take <= 0:
            continue
        idx = rng.choice(len(sub), size=take, replace=False)
        chosen = sub.iloc[idx].copy()
        parts.append(chosen)
        used_ids.update(chosen.index.tolist())

    cur_n = sum(len(x) for x in parts)
    if cur_n < max_items:
        leftover = df.loc[~df.index.isin(list(used_ids))].copy()
        if len(leftover) > 0:
            take = min(max_items - cur_n, len(leftover))
            idx = rng.choice(len(leftover), size=take, replace=False)
            parts.append(leftover.iloc[idx].copy())

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def subsample_candidates_per_slide(
    sub_df: pd.DataFrame,
    max_candidates_per_slide: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    if max_candidates_per_slide is None or len(sub_df) <= max_candidates_per_slide:
        return sub_df.reset_index(drop=True).copy()

    sort_cols = ["rank_in_slide"] if "rank_in_slide" in sub_df.columns else ["patch_idx"]
    sub_df = sub_df.sort_values(sort_cols).reset_index(drop=True)
    return sub_df.iloc[:max_candidates_per_slide].reset_index(drop=True).copy()


def make_label_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("Need 'label' or 'slide_binary_label' in slides_csv.")
    return df


def load_slides_csv(
    slides_csv: str,
    split: Optional[str] = None,
    benchmark_split: Optional[str] = None,
    svs_root: Optional[str] = None,
    h5_root: Optional[str] = None,
    project: Optional[str] = None,
) -> pd.DataFrame:
    if not os.path.exists(slides_csv):
        raise FileNotFoundError(slides_csv)

    df = pd.read_csv(slides_csv)
    df = make_label_column_if_needed(df)

    required_cols = {"slide_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"slides_csv missing columns: {missing}")

    if benchmark_split is not None:
        if "benchmark_split" not in df.columns:
            raise ValueError("benchmark_split was specified but csv has no 'benchmark_split' column")
        df = df[df["benchmark_split"] == benchmark_split].copy()
    elif split is not None:
        if "split" not in df.columns:
            raise ValueError("split was specified but csv has no 'split' column")
        df = df[df["split"] == split].copy()

    df = df.reset_index(drop=True)
    df = resolve_svs_h5_paths(df, svs_root=svs_root, h5_root=h5_root, project=project)
    return df


def load_pool_df(
    positive_pool_csv: Optional[str] = None,
    negative_pool_csv: Optional[str] = None,
    pool_csv: Optional[str] = None,
    benchmark_csv: Optional[str] = None,
    svs_root: Optional[str] = None,
    h5_root: Optional[str] = None,
    project: Optional[str] = None,
) -> pd.DataFrame:
    """
    Supports:
      1. --pool_csv: merged proposal pool CSV.
      2. --positive_pool_csv + --negative_pool_csv: split positive/negative pool CSVs.

    Ensures svs_path/h5_path are present or resolved from roots.
    """
    if pool_csv is not None:
        pool_df = pd.read_csv(pool_csv)
    else:
        if positive_pool_csv is None or negative_pool_csv is None:
            raise ValueError("Either --pool_csv or both --positive_pool_csv/--negative_pool_csv must be provided.")
        pos_df = pd.read_csv(positive_pool_csv)
        neg_df = pd.read_csv(negative_pool_csv)
        pool_df = pd.concat([pos_df, neg_df], axis=0).reset_index(drop=True)

    if "slide_id" not in pool_df.columns:
        raise ValueError("pool csv must contain slide_id column")
    if "label" not in pool_df.columns:
        raise ValueError("pool csv must contain label column")

    pool_df["slide_id"] = pool_df["slide_id"].astype(str)

    if benchmark_csv is not None:
        bench_df = pd.read_csv(benchmark_csv)
        if "slide_id" not in bench_df.columns:
            raise ValueError("benchmark_csv must contain slide_id column")
        bench_ids = set(bench_df["slide_id"].astype(str).tolist())
        pool_df = pool_df[pool_df["slide_id"].isin(bench_ids)].copy().reset_index(drop=True)

    pool_df = resolve_svs_h5_paths(pool_df, svs_root=svs_root, h5_root=h5_root, project=project)
    return pool_df


# =========================================================
# patch IO
# =========================================================
def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def load_coords_attrs(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs.items())
    patch_size = int(attrs.get("patch_size", 256))
    patch_level = int(attrs.get("patch_level", 0))
    return coords, patch_size, patch_level


# =========================================================
# model loading
# =========================================================
def load_encoder_from_ckpt(
    config_path: str,
    student_ckpt_path: str,
    device: str = "cuda",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(student_ckpt_path, map_location="cpu")
    if "student_state_dict" not in ckpt:
        raise KeyError(f"student_state_dict not found in {student_ckpt_path}")

    encoder = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    encoder.load_state_dict(ckpt["student_state_dict"], strict=True)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, cfg


def load_proj_l12_from_stage2(
    stage2_full_ckpt: str,
    device: str,
):
    ckpt = torch.load(stage2_full_ckpt, map_location="cpu")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in stage2 full checkpoint")

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
# feature extraction / scoring
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
        raise RuntimeError(f"Unexpected encoder output type/len: {type(out)}")

    _, _, feature_dict, moe_feature_list = out

    if use_last_moe_output and len(moe_feature_list) > 0:
        feat_tokens = moe_feature_list[-1]
    else:
        feat_tokens = feature_dict["layer_12"]

    patch_tokens = feat_tokens[:, 1:, :]
    if patch_tokens.shape[1] == 0:
        raise RuntimeError(f"No patch tokens found, shape={tuple(patch_tokens.shape)}")

    patch_feat = patch_tokens.mean(dim=1)
    return patch_feat


@torch.no_grad()
def extract_selected_patch_features(
    encoder: nn.Module,
    svs_path: str,
    h5_path: str,
    selected_patch_indices: List[int],
    device: str,
    img_size: int = 224,
    batch_size: int = 64,
    use_last_moe_output: bool = True,
):
    coords, patch_size, patch_level = load_coords_attrs(h5_path)
    transform = build_transform(img_size)

    selected_patch_indices = list(selected_patch_indices)
    selected_coords = coords[selected_patch_indices]

    slide = openslide.OpenSlide(svs_path)
    feats = []
    try:
        for start in range(0, len(selected_coords), batch_size):
            batch_coords = selected_coords[start:start + batch_size]
            imgs = []
            for xy in batch_coords:
                img = read_patch_from_wsi(
                    slide=slide,
                    coord_xy=(int(xy[0]), int(xy[1])),
                    patch_size=patch_size,
                    read_level=patch_level,
                )
                imgs.append(transform(img))
            x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            feat = extract_patch_features_stage2_style(
                encoder=encoder,
                patch_imgs=x,
                use_last_moe_output=use_last_moe_output,
            )
            feats.append(feat.cpu())
    finally:
        slide.close()

    return torch.cat(feats, dim=0), selected_coords, patch_size, patch_level


def maybe_load_frozen_cache(
    cache_frozen_dir: Optional[str],
    frozen_student_ckpt: str,
    slide_id: str,
    patch_indices: List[int],
):
    if cache_frozen_dir is None:
        return None, None

    ckpt_tag = Path(frozen_student_ckpt).stem
    idx_tag = "_".join(map(str, patch_indices[:64]))
    idx_hash = hashlib.md5(idx_tag.encode("utf-8")).hexdigest()[:12]
    cache_path = os.path.join(cache_frozen_dir, f"{slide_id}__{ckpt_tag}__{len(patch_indices)}__{idx_hash}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu"), cache_path
    return None, cache_path


def maybe_save_frozen_cache(cache_obj: dict, cache_path: Optional[str]):
    if cache_path is None:
        return
    ensure_dir(os.path.dirname(cache_path))
    torch.save(cache_obj, cache_path)


@torch.no_grad()
def score_patch_features(
    patch_feat_raw: torch.Tensor,
    proj_l12: nn.Module,
    summary_builder: PatchRoleSummaryFromSharedProto,
    role_names: List[str],
    tumor_name: str,
    negative_role_names: List[str],
    device: str,
):
    x = patch_feat_raw.to(device, non_blocking=True)
    x_teacher = proj_l12(x)
    x_teacher = F.normalize(x_teacher, dim=-1)

    role_dict = summary_builder(x_teacher.unsqueeze(0))
    role_logits = role_dict["patch_role_logits"][0].detach().cpu()
    role_probs = role_dict["patch_role_probs"][0].detach().cpu()
    top1_gap = role_dict["patch_top1_gap"][0].detach().cpu().squeeze(-1)

    role_to_idx = {n: i for i, n in enumerate(role_names)}
    if tumor_name not in role_to_idx:
        raise KeyError(f"tumor role '{tumor_name}' not found in role_names={role_names}")

    tumor_idx = role_to_idx[tumor_name]
    neg_ids = [role_to_idx[n] for n in negative_role_names if n in role_to_idx]
    if len(neg_ids) == 0:
        raise ValueError(f"No valid negative role names found in role_names. got={negative_role_names}")

    tumor_logit = role_logits[:, tumor_idx]
    tumor_prob = role_probs[:, tumor_idx]
    neg_logit = role_logits[:, neg_ids].max(dim=-1).values
    tumor_gap = tumor_logit - neg_logit

    pred_role_idx = role_probs.argmax(dim=-1)
    pred_role_name = [role_names[int(i)] for i in pred_role_idx.tolist()]

    return {
        "role_logits": role_logits,
        "role_probs": role_probs,
        "top1_gap": top1_gap,
        "tumor_logit": tumor_logit,
        "tumor_prob": tumor_prob,
        "tumor_gap": tumor_gap,
        "pred_role_idx": pred_role_idx,
        "pred_role_name": pred_role_name,
    }


# =========================================================
# summary / plotting
# =========================================================
def summarize_patch_df(patch_df: pd.DataFrame, out_json: str):
    summary = {}
    for label_val, name in [(1, "positive"), (0, "negative")]:
        sub = patch_df[patch_df["label"] == label_val].copy()
        if len(sub) == 0:
            continue
        summary[name] = {
            "num_patches": int(len(sub)),
            "num_slides": int(sub["slide_id"].nunique()),
            "frozen_mean_tumor_gap": float(sub["frozen_tumor_gap"].mean()),
            "adapted_mean_tumor_gap": float(sub["adapted_tumor_gap"].mean()),
            "delta_mean_tumor_gap": float(sub["delta_tumor_gap"].mean()),
            "frozen_mean_tumor_prob": float(sub["frozen_tumor_prob"].mean()),
            "adapted_mean_tumor_prob": float(sub["adapted_tumor_prob"].mean()),
            "delta_mean_tumor_prob": float(sub["delta_tumor_prob"].mean()),
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def plot_candidate_distributions(patch_df: pd.DataFrame, out_dir: str, prefix: str = ""):
    for label_val, label_name in [(1, "positive"), (0, "negative")]:
        sub = patch_df[patch_df["label"] == label_val]
        if len(sub) == 0:
            continue

        plt.figure(figsize=(7, 5))
        plt.hist(sub["frozen_tumor_gap"], bins=20, alpha=0.6, label="frozen")
        plt.hist(sub["adapted_tumor_gap"], bins=20, alpha=0.6, label="adapted")
        plt.xlabel("tumor_gap")
        plt.ylabel("count")
        plt.title(f"{label_name} {prefix} tumor_gap distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}{label_name}_tumor_gap_hist.png"), dpi=180)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.boxplot(
            [sub["frozen_tumor_gap"].values, sub["adapted_tumor_gap"].values],
            labels=["frozen", "adapted"]
        )
        plt.ylabel("tumor_gap")
        plt.title(f"{label_name} {prefix} tumor_gap boxplot")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}{label_name}_tumor_gap_box.png"), dpi=180)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.boxplot(
            [sub["frozen_tumor_prob"].values, sub["adapted_tumor_prob"].values],
            labels=["frozen", "adapted"]
        )
        plt.ylabel("tumor_prob")
        plt.title(f"{label_name} {prefix} tumor_prob boxplot")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}{label_name}_tumor_prob_box.png"), dpi=180)
        plt.close()


def plot_slide_delta_lines(slide_df: pd.DataFrame, out_dir: str, prefix: str = ""):
    for label_val, label_name in [(1, "positive"), (0, "negative")]:
        sub = slide_df[slide_df["label"] == label_val].copy()
        if len(sub) == 0:
            continue

        sub = sub.sort_values("delta_mean_tumor_gap")
        plt.figure(figsize=(8, max(4, 0.28 * len(sub))))
        for row in sub.itertuples():
            plt.plot([0, 1], [row.frozen_mean_tumor_gap, row.adapted_mean_tumor_gap], marker="o")
        plt.xticks([0, 1], ["frozen", "adapted"])
        plt.ylabel("slide mean tumor_gap")
        plt.title(f"{label_name} {prefix} mean gap before/after")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}{label_name}_slide_gap_lines.png"), dpi=180)
        plt.close()


# =========================================================
# shared pool scoring core
# =========================================================
def _load_models_and_proto(
    config_path: str,
    frozen_student_ckpt: Optional[str],
    adapted_student_ckpt: str,
    stage2_full_ckpt: str,
    role_proto_dir: str,
    device: str,
):
    frozen_encoder = None
    if frozen_student_ckpt is not None:
        frozen_encoder, _ = load_encoder_from_ckpt(config_path, frozen_student_ckpt, device=device)
    adapted_encoder, _ = load_encoder_from_ckpt(config_path, adapted_student_ckpt, device=device)
    proj_l12 = load_proj_l12_from_stage2(stage2_full_ckpt=stage2_full_ckpt, device=device)

    shared_role_proto = SharedRolePrototype.from_files(
        role_proto_dir=role_proto_dir,
        normalize=True,
        learnable=False,
        device=device,
    )
    role_names = list(shared_role_proto.role_names)
    summary_builder = PatchRoleSummaryFromSharedProto(
        shared_role_proto=shared_role_proto,
        tau=1.0,
        use_softmax=True,
    ).to(device)
    summary_builder.eval()
    return frozen_encoder, adapted_encoder, proj_l12, summary_builder, role_names


def _prepare_pool_df(
    positive_pool_csv: Optional[str],
    negative_pool_csv: Optional[str],
    pool_csv: Optional[str],
    benchmark_csv: Optional[str],
    max_slides: Optional[int],
    svs_root: Optional[str],
    h5_root: Optional[str],
    project: Optional[str],
) -> pd.DataFrame:
    pool_df = load_pool_df(
        positive_pool_csv=positive_pool_csv,
        negative_pool_csv=negative_pool_csv,
        pool_csv=pool_csv,
        benchmark_csv=benchmark_csv,
        svs_root=svs_root,
        h5_root=h5_root,
        project=project,
    )

    need = {"slide_id", "label", "patch_idx", "coord_x", "coord_y", "svs_path", "h5_path"}
    miss = need - set(pool_df.columns)
    if miss:
        raise ValueError(f"pool csv missing columns after path resolution: {miss}")

    slide_meta = pool_df.groupby("slide_id").agg({"label": "first"}).reset_index()
    slide_meta = subsample_df_balanced(slide_meta, max_items=max_slides, seed=42, label_col="label")
    keep_ids = set(slide_meta["slide_id"].tolist())
    pool_df = pool_df[pool_df["slide_id"].isin(keep_ids)].copy().reset_index(drop=True)
    return pool_df


# =========================================================
# pool compare
# =========================================================
def analyze_proposal_pool(
    positive_pool_csv: Optional[str],
    negative_pool_csv: Optional[str],
    pool_csv: Optional[str],
    config_path: str,
    frozen_student_ckpt: str,
    adapted_student_ckpt: str,
    stage2_full_ckpt: str,
    role_proto_dir: str,
    out_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    use_last_moe_output: bool,
    tumor_name: str,
    negative_role_names: List[str],
    max_slides: Optional[int] = None,
    max_candidates_per_slide: Optional[int] = None,
    benchmark_csv: Optional[str] = None,
    cache_frozen_dir: Optional[str] = None,
    svs_root: Optional[str] = None,
    h5_root: Optional[str] = None,
    project: Optional[str] = None,
):
    ensure_dir(out_dir)

    pool_df = _prepare_pool_df(
        positive_pool_csv=positive_pool_csv,
        negative_pool_csv=negative_pool_csv,
        pool_csv=pool_csv,
        benchmark_csv=benchmark_csv,
        max_slides=max_slides,
        svs_root=svs_root,
        h5_root=h5_root,
        project=project,
    )

    frozen_encoder, adapted_encoder, proj_l12, summary_builder, role_names = _load_models_and_proto(
        config_path=config_path,
        frozen_student_ckpt=frozen_student_ckpt,
        adapted_student_ckpt=adapted_student_ckpt,
        stage2_full_ckpt=stage2_full_ckpt,
        role_proto_dir=role_proto_dir,
        device=device,
    )

    print("[Analyze Proposal Pool]")
    print(f"num pool rows after filter = {len(pool_df)}")
    print(f"num unique slides after filter = {pool_df['slide_id'].nunique()}")
    print(f"role_names = {role_names}")
    print(f"tumor_name = {tumor_name}")
    print(f"negative_role_names = {negative_role_names}")
    print(f"max_candidates_per_slide = {max_candidates_per_slide}")

    all_rows = []
    slide_summaries = []

    grouped = pool_df.groupby("slide_id")
    pbar = tqdm(grouped, total=pool_df["slide_id"].nunique(), desc="Scoring proposal pool slides")

    for slide_id, sub_df in pbar:
        sort_col = "rank_in_slide" if "rank_in_slide" in sub_df.columns else "patch_idx"
        sub_df = sub_df.sort_values(sort_col).reset_index(drop=True)
        sub_df = subsample_candidates_per_slide(
            sub_df,
            max_candidates_per_slide=max_candidates_per_slide,
            seed=42 + stable_hash(slide_id),
        )

        label = int(sub_df["label"].iloc[0])
        svs_path = str(sub_df["svs_path"].iloc[0])
        h5_path = str(sub_df["h5_path"].iloc[0])
        patch_indices = sub_df["patch_idx"].astype(int).tolist()

        frozen_cache_obj, frozen_cache_path = maybe_load_frozen_cache(
            cache_frozen_dir=cache_frozen_dir,
            frozen_student_ckpt=frozen_student_ckpt,
            slide_id=slide_id,
            patch_indices=patch_indices,
        )

        if frozen_cache_obj is not None:
            frozen_feat = frozen_cache_obj["frozen_feat"]
        else:
            frozen_feat, _, patch_size, patch_level = extract_selected_patch_features(
                encoder=frozen_encoder,
                svs_path=svs_path,
                h5_path=h5_path,
                selected_patch_indices=patch_indices,
                device=device,
                img_size=img_size,
                batch_size=batch_size,
                use_last_moe_output=use_last_moe_output,
            )
            maybe_save_frozen_cache(
                {"frozen_feat": frozen_feat, "patch_size": patch_size, "patch_level": patch_level},
                frozen_cache_path,
            )

        adapted_feat, _, _, _ = extract_selected_patch_features(
            encoder=adapted_encoder,
            svs_path=svs_path,
            h5_path=h5_path,
            selected_patch_indices=patch_indices,
            device=device,
            img_size=img_size,
            batch_size=batch_size,
            use_last_moe_output=use_last_moe_output,
        )

        frozen_score = score_patch_features(
            patch_feat_raw=frozen_feat,
            proj_l12=proj_l12,
            summary_builder=summary_builder,
            role_names=role_names,
            tumor_name=tumor_name,
            negative_role_names=negative_role_names,
            device=device,
        )
        adapted_score = score_patch_features(
            patch_feat_raw=adapted_feat,
            proj_l12=proj_l12,
            summary_builder=summary_builder,
            role_names=role_names,
            tumor_name=tumor_name,
            negative_role_names=negative_role_names,
            device=device,
        )

        for i in range(len(sub_df)):
            row_base = sub_df.iloc[i].to_dict()
            row = {
                **row_base,
                "frozen_tumor_prob": safe_float(frozen_score["tumor_prob"][i]),
                "adapted_tumor_prob": safe_float(adapted_score["tumor_prob"][i]),
                "delta_tumor_prob": safe_float(adapted_score["tumor_prob"][i] - frozen_score["tumor_prob"][i]),
                "frozen_tumor_gap": safe_float(frozen_score["tumor_gap"][i]),
                "adapted_tumor_gap": safe_float(adapted_score["tumor_gap"][i]),
                "delta_tumor_gap": safe_float(adapted_score["tumor_gap"][i] - frozen_score["tumor_gap"][i]),
                "frozen_top1_gap": safe_float(frozen_score["top1_gap"][i]),
                "adapted_top1_gap": safe_float(adapted_score["top1_gap"][i]),
                "delta_top1_gap": safe_float(adapted_score["top1_gap"][i] - frozen_score["top1_gap"][i]),
                "frozen_pred_role": frozen_score["pred_role_name"][i],
                "adapted_pred_role": adapted_score["pred_role_name"][i],
            }
            all_rows.append(row)

        slide_summaries.append({
            "slide_id": slide_id,
            "label": label,
            "num_candidates": len(sub_df),
            "frozen_mean_tumor_prob": safe_float(frozen_score["tumor_prob"].mean()),
            "adapted_mean_tumor_prob": safe_float(adapted_score["tumor_prob"].mean()),
            "delta_mean_tumor_prob": safe_float(adapted_score["tumor_prob"].mean() - frozen_score["tumor_prob"].mean()),
            "frozen_mean_tumor_gap": safe_float(frozen_score["tumor_gap"].mean()),
            "adapted_mean_tumor_gap": safe_float(adapted_score["tumor_gap"].mean()),
            "delta_mean_tumor_gap": safe_float(adapted_score["tumor_gap"].mean() - frozen_score["tumor_gap"].mean()),
            "frozen_frac_gap_gt0": float((frozen_score["tumor_gap"] > 0).float().mean().item()),
            "adapted_frac_gap_gt0": float((adapted_score["tumor_gap"] > 0).float().mean().item()),
        })

    patch_df = pd.DataFrame(all_rows)
    slide_df = pd.DataFrame(slide_summaries)

    patch_csv = os.path.join(out_dir, "pool_patch_before_after.csv")
    slide_csv = os.path.join(out_dir, "pool_slide_before_after.csv")
    patch_df.to_csv(patch_csv, index=False)
    slide_df.to_csv(slide_csv, index=False)

    print(f"[Saved] {patch_csv}")
    print(f"[Saved] {slide_csv}")

    summarize_patch_df(patch_df, os.path.join(out_dir, "pool_summary.json"))
    plot_candidate_distributions(patch_df, out_dir, prefix="pool_")
    plot_slide_delta_lines(slide_df, out_dir, prefix="pool_")

    return patch_df, slide_df, role_names


# =========================================================
# online selected compare
# =========================================================
def analyze_online_selected_proposals(
    positive_pool_csv: Optional[str],
    negative_pool_csv: Optional[str],
    pool_csv: Optional[str],
    config_path: str,
    frozen_student_ckpt: str,
    adapted_student_ckpt: str,
    stage2_full_ckpt: str,
    role_proto_dir: str,
    out_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    use_last_moe_output: bool,
    tumor_name: str,
    negative_role_names: List[str],
    max_slides: Optional[int] = None,
    max_candidates_per_slide: Optional[int] = None,
    benchmark_csv: Optional[str] = None,
    cache_frozen_dir: Optional[str] = None,
    svs_root: Optional[str] = None,
    h5_root: Optional[str] = None,
    project: Optional[str] = None,
    use_strong_pos_support: bool = True,
    allow_pos_support_fallback: bool = True,
    min_pos_keep: int = 4,
    online_pos_topk: Optional[int] = None,
    online_neg_topk: Optional[int] = None,
    pos_support_min_tumor_prob: float = -1e6,
    pos_support_min_center_gap: float = -1e6,
    pos_support_min_top1_gap: float = -1e6,
    pos_support_min_context_score: float = -1e6,
    pos_support_min_neighbor_gap_mean: float = -1e6,
    pos_support_min_neighbor_gap_max: float = -1e6,
    pos_support_min_neighbor_prob_mean: float = -1e6,
    pos_support_min_neighbor_prob_max: float = -1e6,
    pos_fallback_min_tumor_prob: float = -1e6,
    pos_fallback_min_center_gap: float = -1e6,
    pos_fallback_min_top1_gap: float = -1e6,
    pos_fallback_min_context_score: float = -1e6,
    pos_fallback_min_neighbor_gap_mean: float = -1e6,
    pos_fallback_min_neighbor_gap_max: float = -1e6,
    pos_fallback_min_neighbor_prob_mean: float = -1e6,
    pos_fallback_min_neighbor_prob_max: float = -1e6,
):
    ensure_dir(out_dir)

    pool_df = _prepare_pool_df(
        positive_pool_csv=positive_pool_csv,
        negative_pool_csv=negative_pool_csv,
        pool_csv=pool_csv,
        benchmark_csv=benchmark_csv,
        max_slides=max_slides,
        svs_root=svs_root,
        h5_root=h5_root,
        project=project,
    )

    frozen_encoder, adapted_encoder, proj_l12, summary_builder, role_names = _load_models_and_proto(
        config_path=config_path,
        frozen_student_ckpt=frozen_student_ckpt,
        adapted_student_ckpt=adapted_student_ckpt,
        stage2_full_ckpt=stage2_full_ckpt,
        role_proto_dir=role_proto_dir,
        device=device,
    )

    print("[Analyze Online Selected Proposals]")
    print(f"num pool rows after filter = {len(pool_df)}")
    print(f"num unique slides after filter = {pool_df['slide_id'].nunique()}")
    print(f"role_names = {role_names}")
    print(f"tumor_name = {tumor_name}")
    print(f"negative_role_names = {negative_role_names}")
    print(f"max_candidates_per_slide = {max_candidates_per_slide}")

    all_rows = []
    slide_summaries = []

    grouped = pool_df.groupby("slide_id")
    pbar = tqdm(grouped, total=pool_df["slide_id"].nunique(), desc="Scoring online-selected slides")

    for slide_id, sub_df in pbar:
        sort_col = "rank_in_slide" if "rank_in_slide" in sub_df.columns else "patch_idx"
        sub_df = sub_df.sort_values(sort_col).reset_index(drop=True)
        sub_df = subsample_candidates_per_slide(
            sub_df,
            max_candidates_per_slide=max_candidates_per_slide,
            seed=42 + stable_hash(slide_id),
        )

        label = int(sub_df["label"].iloc[0])
        svs_path = str(sub_df["svs_path"].iloc[0])
        h5_path = str(sub_df["h5_path"].iloc[0])
        patch_indices = sub_df["patch_idx"].astype(int).tolist()

        frozen_cache_obj, frozen_cache_path = maybe_load_frozen_cache(
            cache_frozen_dir=cache_frozen_dir,
            frozen_student_ckpt=frozen_student_ckpt,
            slide_id=slide_id,
            patch_indices=patch_indices,
        )

        if frozen_cache_obj is not None:
            frozen_feat = frozen_cache_obj["frozen_feat"]
        else:
            frozen_feat, _, patch_size, patch_level = extract_selected_patch_features(
                encoder=frozen_encoder,
                svs_path=svs_path,
                h5_path=h5_path,
                selected_patch_indices=patch_indices,
                device=device,
                img_size=img_size,
                batch_size=batch_size,
                use_last_moe_output=use_last_moe_output,
            )
            maybe_save_frozen_cache(
                {"frozen_feat": frozen_feat, "patch_size": patch_size, "patch_level": patch_level},
                frozen_cache_path,
            )

        adapted_feat, _, _, _ = extract_selected_patch_features(
            encoder=adapted_encoder,
            svs_path=svs_path,
            h5_path=h5_path,
            selected_patch_indices=patch_indices,
            device=device,
            img_size=img_size,
            batch_size=batch_size,
            use_last_moe_output=use_last_moe_output,
        )

        online_scores = compute_online_role_scores(
            patch_feat_adapt=adapted_feat.to(device),
            proj_l12=proj_l12,
            summary_builder=summary_builder,
            role_names=role_names,
            tumor_name=tumor_name,
            negative_role_names=negative_role_names,
        )

        pos_context_score = torch.as_tensor(
            sub_df["pos_context_score"].values.astype(np.float32), device=device
        ) if "pos_context_score" in sub_df.columns else None
        neg_context_score = torch.as_tensor(
            sub_df["neg_context_score"].values.astype(np.float32), device=device
        ) if "neg_context_score" in sub_df.columns else None
        neighbor_gap_mean = torch.as_tensor(
            sub_df["neighbor_gap_mean"].values.astype(np.float32), device=device
        ) if "neighbor_gap_mean" in sub_df.columns else None
        neighbor_gap_max = torch.as_tensor(
            sub_df["neighbor_gap_max"].values.astype(np.float32), device=device
        ) if "neighbor_gap_max" in sub_df.columns else None
        neighbor_prob_mean = torch.as_tensor(
            sub_df["neighbor_prob_mean"].values.astype(np.float32), device=device
        ) if "neighbor_prob_mean" in sub_df.columns else None
        neighbor_prob_max = torch.as_tensor(
            sub_df["neighbor_prob_max"].values.astype(np.float32), device=device
        ) if "neighbor_prob_max" in sub_df.columns else None

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
                use_strong_pos_support=use_strong_pos_support,
                allow_pos_support_fallback=allow_pos_support_fallback,
                min_pos_keep=min_pos_keep,
                select_topk=online_pos_topk,
                pos_support_min_tumor_prob=pos_support_min_tumor_prob,
                pos_support_min_center_gap=pos_support_min_center_gap,
                pos_support_min_top1_gap=pos_support_min_top1_gap,
                pos_support_min_context_score=pos_support_min_context_score,
                pos_support_min_neighbor_gap_mean=pos_support_min_neighbor_gap_mean,
                pos_support_min_neighbor_gap_max=pos_support_min_neighbor_gap_max,
                pos_support_min_neighbor_prob_mean=pos_support_min_neighbor_prob_mean,
                pos_support_min_neighbor_prob_max=pos_support_min_neighbor_prob_max,
                pos_fallback_min_tumor_prob=pos_fallback_min_tumor_prob,
                pos_fallback_min_center_gap=pos_fallback_min_center_gap,
                pos_fallback_min_top1_gap=pos_fallback_min_top1_gap,
                pos_fallback_min_context_score=pos_fallback_min_context_score,
                pos_fallback_min_neighbor_gap_mean=pos_fallback_min_neighbor_gap_mean,
                pos_fallback_min_neighbor_gap_max=pos_fallback_min_neighbor_gap_max,
                pos_fallback_min_neighbor_prob_mean=pos_fallback_min_neighbor_prob_mean,
                pos_fallback_min_neighbor_prob_max=pos_fallback_min_neighbor_prob_max,
            )
        else:
            selected_idx, select_stats = select_online_negative_proposals(
                tumor_gap=online_scores["tumor_gap"],
                tumor_prob=online_scores["tumor_prob"],
                top1_gap=online_scores["top1_gap"],
                neg_context_score=neg_context_score,
                neighbor_gap_mean=neighbor_gap_mean,
                neighbor_gap_max=neighbor_gap_max,
                select_topk=online_neg_topk,
            )

        selected_idx_cpu = selected_idx.detach().cpu().numpy().tolist()
        selected_sub = sub_df.iloc[selected_idx_cpu].reset_index(drop=True)

        frozen_score_all = score_patch_features(
            patch_feat_raw=frozen_feat,
            proj_l12=proj_l12,
            summary_builder=summary_builder,
            role_names=role_names,
            tumor_name=tumor_name,
            negative_role_names=negative_role_names,
            device=device,
        )
        adapted_score_all = score_patch_features(
            patch_feat_raw=adapted_feat,
            proj_l12=proj_l12,
            summary_builder=summary_builder,
            role_names=role_names,
            tumor_name=tumor_name,
            negative_role_names=negative_role_names,
            device=device,
        )

        frozen_score = {k: (v[selected_idx_cpu] if torch.is_tensor(v) else [v[j] for j in selected_idx_cpu]) for k, v in frozen_score_all.items()}
        adapted_score = {k: (v[selected_idx_cpu] if torch.is_tensor(v) else [v[j] for j in selected_idx_cpu]) for k, v in adapted_score_all.items()}

        for i in range(len(selected_sub)):
            row_base = selected_sub.iloc[i].to_dict()
            row = {
                **row_base,
                "selected_by_online": 1,
                "selected_num_this_slide": int(len(selected_sub)),
                "frozen_tumor_prob": safe_float(frozen_score["tumor_prob"][i]),
                "adapted_tumor_prob": safe_float(adapted_score["tumor_prob"][i]),
                "delta_tumor_prob": safe_float(adapted_score["tumor_prob"][i] - frozen_score["tumor_prob"][i]),
                "frozen_tumor_gap": safe_float(frozen_score["tumor_gap"][i]),
                "adapted_tumor_gap": safe_float(adapted_score["tumor_gap"][i]),
                "delta_tumor_gap": safe_float(adapted_score["tumor_gap"][i] - frozen_score["tumor_gap"][i]),
                "frozen_top1_gap": safe_float(frozen_score["top1_gap"][i]),
                "adapted_top1_gap": safe_float(adapted_score["top1_gap"][i]),
                "delta_top1_gap": safe_float(adapted_score["top1_gap"][i] - frozen_score["top1_gap"][i]),
                "frozen_pred_role": frozen_score["pred_role_name"][i],
                "adapted_pred_role": adapted_score["pred_role_name"][i],
            }
            for k, v in select_stats.items():
                row[k] = v
            all_rows.append(row)

        slide_summary = {
            "slide_id": slide_id,
            "label": label,
            "num_pool": int(len(sub_df)),
            "num_selected": int(len(selected_sub)),
            "selected_ratio": float(len(selected_sub) / max(len(sub_df), 1)),
            "frozen_mean_tumor_prob": safe_float(frozen_score["tumor_prob"].mean()) if len(selected_sub) > 0 else float("nan"),
            "adapted_mean_tumor_prob": safe_float(adapted_score["tumor_prob"].mean()) if len(selected_sub) > 0 else float("nan"),
            "delta_mean_tumor_prob": safe_float(adapted_score["tumor_prob"].mean() - frozen_score["tumor_prob"].mean()) if len(selected_sub) > 0 else float("nan"),
            "frozen_mean_tumor_gap": safe_float(frozen_score["tumor_gap"].mean()) if len(selected_sub) > 0 else float("nan"),
            "adapted_mean_tumor_gap": safe_float(adapted_score["tumor_gap"].mean()) if len(selected_sub) > 0 else float("nan"),
            "delta_mean_tumor_gap": safe_float(adapted_score["tumor_gap"].mean() - frozen_score["tumor_gap"].mean()) if len(selected_sub) > 0 else float("nan"),
        }
        for k, v in select_stats.items():
            slide_summary[k] = v
        slide_summaries.append(slide_summary)

    patch_df = pd.DataFrame(all_rows)
    slide_df = pd.DataFrame(slide_summaries)

    patch_csv = os.path.join(out_dir, "online_selected_patch_before_after.csv")
    slide_csv = os.path.join(out_dir, "online_selected_slide_before_after.csv")
    patch_df.to_csv(patch_csv, index=False)
    slide_df.to_csv(slide_csv, index=False)

    print(f"[Saved] {patch_csv}")
    print(f"[Saved] {slide_csv}")

    summarize_patch_df(patch_df, os.path.join(out_dir, "online_selected_summary.json"))
    plot_candidate_distributions(patch_df, out_dir, prefix="online_selected_")
    plot_slide_delta_lines(slide_df, out_dir, prefix="online_selected_")

    return patch_df, slide_df, role_names


# =========================================================
# role top-k visualization
# =========================================================
@torch.no_grad()
def collect_role_topk_patches(
    slides_csv: str,
    config_path: str,
    adapted_student_ckpt: str,
    stage2_full_ckpt: str,
    role_proto_dir: str,
    out_dir: str,
    device: str,
    img_size: int,
    extract_batch_size: int,
    max_patches_per_slide: int,
    use_last_moe_output: bool,
    split: Optional[str],
    benchmark_split: Optional[str],
    topk_per_role: int,
    max_slides: Optional[int] = None,
    svs_root: Optional[str] = None,
    h5_root: Optional[str] = None,
    project: Optional[str] = None,
):
    ensure_dir(out_dir)

    df = load_slides_csv(
        slides_csv,
        split=split,
        benchmark_split=benchmark_split,
        svs_root=svs_root,
        h5_root=h5_root,
        project=project,
    )
    df = subsample_df_balanced(df, max_items=max_slides, seed=42, label_col="label")

    adapted_encoder, _ = load_encoder_from_ckpt(config_path, adapted_student_ckpt, device=device)
    proj_l12 = load_proj_l12_from_stage2(stage2_full_ckpt=stage2_full_ckpt, device=device)
    shared_role_proto = SharedRolePrototype.from_files(
        role_proto_dir=role_proto_dir,
        normalize=True,
        learnable=False,
        device=device,
    )
    role_names = list(shared_role_proto.role_names)
    summary_builder = PatchRoleSummaryFromSharedProto(
        shared_role_proto=shared_role_proto,
        tau=1.0,
        use_softmax=True,
    ).to(device)
    summary_builder.eval()

    role_records = {r: [] for r in role_names}
    tf = build_transform(img_size)

    pbar = tqdm(df.iterrows(), total=len(df), desc="Collect role top-k patches")
    for row_idx, row in pbar:
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        svs_path = str(row["svs_path"])
        h5_path = str(row["h5_path"])

        coords_all, patch_size, patch_level = load_coords_attrs(h5_path)
        if len(coords_all) > max_patches_per_slide:
            rng = np.random.default_rng(42 + row_idx)
            keep_idx = rng.choice(len(coords_all), size=max_patches_per_slide, replace=False)
            coords = coords_all[keep_idx]
            patch_indices = keep_idx
        else:
            coords = coords_all
            patch_indices = np.arange(len(coords_all))

        slide = openslide.OpenSlide(svs_path)
        all_feats = []
        try:
            for st in range(0, len(coords), extract_batch_size):
                batch_coords = coords[st:st + extract_batch_size]
                imgs = []
                for xy in batch_coords:
                    img = read_patch_from_wsi(
                        slide=slide,
                        coord_xy=(int(xy[0]), int(xy[1])),
                        patch_size=patch_size,
                        read_level=patch_level,
                    )
                    imgs.append(tf(img))
                x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
                feat = extract_patch_features_stage2_style(
                    encoder=adapted_encoder,
                    patch_imgs=x,
                    use_last_moe_output=use_last_moe_output,
                )
                all_feats.append(feat.cpu())
        finally:
            slide.close()

        patch_feat = torch.cat(all_feats, dim=0)
        score_out = score_patch_features(
            patch_feat_raw=patch_feat,
            proj_l12=proj_l12,
            summary_builder=summary_builder,
            role_names=role_names,
            tumor_name=role_names[0],
            negative_role_names=role_names[1:] if len(role_names) > 1 else role_names,
            device=device,
        )

        role_probs = score_out["role_probs"]
        role_logits = score_out["role_logits"]
        top1_gap = score_out["top1_gap"]
        pred_role_name = score_out["pred_role_name"]

        for i in range(len(coords)):
            for ridx, rname in enumerate(role_names):
                role_records[rname].append({
                    "slide_id": slide_id,
                    "label": label,
                    "svs_path": svs_path,
                    "h5_path": h5_path,
                    "patch_idx": int(patch_indices[i]),
                    "coord_x": int(coords[i][0]),
                    "coord_y": int(coords[i][1]),
                    "patch_size": int(patch_size),
                    "patch_level": int(patch_level),
                    "role_name": rname,
                    "role_prob": safe_float(role_probs[i, ridx]),
                    "role_logit": safe_float(role_logits[i, ridx]),
                    "top1_gap": safe_float(top1_gap[i]),
                    "pred_role": pred_role_name[i],
                })

    saved_csvs = []
    for rname, recs in role_records.items():
        rdf = pd.DataFrame(recs)
        rdf = rdf.sort_values(["role_prob", "top1_gap"], ascending=[False, False]).reset_index(drop=True)
        topk_df = rdf.head(topk_per_role).copy()

        csv_path = os.path.join(out_dir, f"topk_{rname}.csv")
        topk_df.to_csv(csv_path, index=False)
        saved_csvs.append(csv_path)

        montage_path = os.path.join(out_dir, f"topk_{rname}_montage.png")
        save_patch_montage(topk_df, montage_path, tile_size=224, cols=4, title=f"Top-{topk_per_role} patches for role={rname}")

    with open(os.path.join(out_dir, "role_topk_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"role_names": role_names, "saved_csvs": saved_csvs}, f, indent=2, ensure_ascii=False)

    return role_names


def save_patch_montage(
    patch_df: pd.DataFrame,
    out_path: str,
    tile_size: int = 224,
    cols: int = 4,
    title: Optional[str] = None,
):
    if len(patch_df) == 0:
        return

    rows = math.ceil(len(patch_df) / cols)
    title_h = 40 if title else 0
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size + title_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((10, 10), title, fill=(0, 0, 0))

    slide_cache = {}

    for idx, row in enumerate(patch_df.itertuples()):
        svs_path = row.svs_path
        x = int(row.coord_x)
        y = int(row.coord_y)
        patch_size = int(row.patch_size)
        patch_level = int(row.patch_level)

        if svs_path not in slide_cache:
            slide_cache[svs_path] = openslide.OpenSlide(svs_path)
        slide = slide_cache[svs_path]
        patch = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        patch = patch.resize((tile_size, tile_size), resample=Image.BICUBIC)

        rr = idx // cols
        cc = idx % cols
        ox = cc * tile_size
        oy = rr * tile_size + title_h
        canvas.paste(patch, (ox, oy))

        overlay = ImageDraw.Draw(canvas)
        txt = f"{row.slide_id}\nprob={row.role_prob:.3f}\ngap={row.top1_gap:.3f}"
        overlay.rectangle([ox, oy, ox + tile_size, oy + 42], fill=(255, 255, 255))
        overlay.text((ox + 4, oy + 2), txt, fill=(0, 0, 0))

    for slide in slide_cache.values():
        slide.close()

    canvas.save(out_path)


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Analyze proposal pool / online-selected proposals / role top-k")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["pool_compare", "online_selected_compare", "role_topk", "all"])

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage2_full_ckpt", type=str, required=True)
    parser.add_argument("--frozen_student_ckpt", type=str, required=False, default=None)
    parser.add_argument("--adapted_student_ckpt", type=str, required=True)
    parser.add_argument("--role_proto_dir", type=str, required=True)

    parser.add_argument("--pool_csv", type=str, default=None,
                        help="Optional merged proposal pool CSV. If provided, positive/negative pool CSVs are not required.")
    parser.add_argument("--positive_pool_csv", type=str, default=None)
    parser.add_argument("--negative_pool_csv", type=str, default=None)
    parser.add_argument("--slides_csv", type=str, default=None)
    parser.add_argument("--benchmark_csv", type=str, default=None)

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--benchmark_split", type=str, default=None)

    parser.add_argument("--svs_root", type=str, default=None)
    parser.add_argument("--h5_root", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_patches_per_slide", type=int, default=512)
    parser.add_argument("--use_last_moe_output", action="store_true")

    parser.add_argument("--tumor_name", type=str, default="tumor")
    parser.add_argument("--negative_role_names", type=str, nargs="+", default=["stroma", "normal_kidney_parenchyma"])

    parser.add_argument("--topk_per_role", type=int, default=12)

    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--max_candidates_per_slide", type=int, default=None)
    parser.add_argument("--role_topk_max_slides", type=int, default=None)
    parser.add_argument("--cache_frozen_dir", type=str, default=None)

    # online selection params
    parser.add_argument("--use_strong_pos_support", action="store_true")
    parser.add_argument("--allow_pos_support_fallback", action="store_true")
    parser.add_argument("--min_pos_keep", type=int, default=4)
    parser.add_argument("--online_pos_topk", type=int, default=None)
    parser.add_argument("--online_neg_topk", type=int, default=None)

    parser.add_argument("--pos_support_min_tumor_prob", type=float, default=-1e6)
    parser.add_argument("--pos_support_min_center_gap", type=float, default=-1e6)
    parser.add_argument("--pos_support_min_top1_gap", type=float, default=-1e6)
    parser.add_argument("--pos_support_min_context_score", type=float, default=-1e6)
    parser.add_argument("--pos_support_min_neighbor_gap_mean", type=float, default=-1e6)
    parser.add_argument("--pos_support_min_neighbor_gap_max", type=float, default=-1e6)
    parser.add_argument("--pos_support_min_neighbor_prob_mean", type=float, default=-1e6)
    parser.add_argument("--pos_support_min_neighbor_prob_max", type=float, default=-1e6)

    parser.add_argument("--pos_fallback_min_tumor_prob", type=float, default=-1e6)
    parser.add_argument("--pos_fallback_min_center_gap", type=float, default=-1e6)
    parser.add_argument("--pos_fallback_min_top1_gap", type=float, default=-1e6)
    parser.add_argument("--pos_fallback_min_context_score", type=float, default=-1e6)
    parser.add_argument("--pos_fallback_min_neighbor_gap_mean", type=float, default=-1e6)
    parser.add_argument("--pos_fallback_min_neighbor_gap_max", type=float, default=-1e6)
    parser.add_argument("--pos_fallback_min_neighbor_prob_mean", type=float, default=-1e6)
    parser.add_argument("--pos_fallback_min_neighbor_prob_max", type=float, default=-1e6)

    args = parser.parse_args()
    ensure_dir(args.out_dir)
    set_seed(42)

    if args.mode in ["pool_compare", "online_selected_compare", "all"]:
        if args.pool_csv is None and (args.positive_pool_csv is None or args.negative_pool_csv is None):
            raise ValueError(
                "For pool/online_selected/all, provide either --pool_csv or both --positive_pool_csv and --negative_pool_csv."
            )
        if args.frozen_student_ckpt is None:
            raise ValueError("--frozen_student_ckpt is required for pool/online_selected/all")

    if args.mode in ["pool_compare", "all"]:
        pool_out = os.path.join(args.out_dir, "pool_compare")
        analyze_proposal_pool(
            positive_pool_csv=args.positive_pool_csv,
            negative_pool_csv=args.negative_pool_csv,
            pool_csv=args.pool_csv,
            config_path=args.config,
            frozen_student_ckpt=args.frozen_student_ckpt,
            adapted_student_ckpt=args.adapted_student_ckpt,
            stage2_full_ckpt=args.stage2_full_ckpt,
            role_proto_dir=args.role_proto_dir,
            out_dir=pool_out,
            device=args.device,
            img_size=args.img_size,
            batch_size=args.batch_size,
            use_last_moe_output=args.use_last_moe_output,
            tumor_name=args.tumor_name,
            negative_role_names=args.negative_role_names,
            max_slides=args.max_slides,
            max_candidates_per_slide=args.max_candidates_per_slide,
            benchmark_csv=args.benchmark_csv,
            cache_frozen_dir=args.cache_frozen_dir,
            svs_root=args.svs_root,
            h5_root=args.h5_root,
            project=args.project,
        )

    if args.mode in ["online_selected_compare", "all"]:
        online_out = os.path.join(args.out_dir, "online_selected_compare")
        analyze_online_selected_proposals(
            positive_pool_csv=args.positive_pool_csv,
            negative_pool_csv=args.negative_pool_csv,
            pool_csv=args.pool_csv,
            config_path=args.config,
            frozen_student_ckpt=args.frozen_student_ckpt,
            adapted_student_ckpt=args.adapted_student_ckpt,
            stage2_full_ckpt=args.stage2_full_ckpt,
            role_proto_dir=args.role_proto_dir,
            out_dir=online_out,
            device=args.device,
            img_size=args.img_size,
            batch_size=args.batch_size,
            use_last_moe_output=args.use_last_moe_output,
            tumor_name=args.tumor_name,
            negative_role_names=args.negative_role_names,
            max_slides=args.max_slides,
            max_candidates_per_slide=args.max_candidates_per_slide,
            benchmark_csv=args.benchmark_csv,
            cache_frozen_dir=args.cache_frozen_dir,
            svs_root=args.svs_root,
            h5_root=args.h5_root,
            project=args.project,
            use_strong_pos_support=args.use_strong_pos_support,
            allow_pos_support_fallback=args.allow_pos_support_fallback,
            min_pos_keep=args.min_pos_keep,
            online_pos_topk=args.online_pos_topk,
            online_neg_topk=args.online_neg_topk,
            pos_support_min_tumor_prob=args.pos_support_min_tumor_prob,
            pos_support_min_center_gap=args.pos_support_min_center_gap,
            pos_support_min_top1_gap=args.pos_support_min_top1_gap,
            pos_support_min_context_score=args.pos_support_min_context_score,
            pos_support_min_neighbor_gap_mean=args.pos_support_min_neighbor_gap_mean,
            pos_support_min_neighbor_gap_max=args.pos_support_min_neighbor_gap_max,
            pos_support_min_neighbor_prob_mean=args.pos_support_min_neighbor_prob_mean,
            pos_support_min_neighbor_prob_max=args.pos_support_min_neighbor_prob_max,
            pos_fallback_min_tumor_prob=args.pos_fallback_min_tumor_prob,
            pos_fallback_min_center_gap=args.pos_fallback_min_center_gap,
            pos_fallback_min_top1_gap=args.pos_fallback_min_top1_gap,
            pos_fallback_min_context_score=args.pos_fallback_min_context_score,
            pos_fallback_min_neighbor_gap_mean=args.pos_fallback_min_neighbor_gap_mean,
            pos_fallback_min_neighbor_gap_max=args.pos_fallback_min_neighbor_gap_max,
            pos_fallback_min_neighbor_prob_mean=args.pos_fallback_min_neighbor_prob_mean,
            pos_fallback_min_neighbor_prob_max=args.pos_fallback_min_neighbor_prob_max,
        )

    if args.mode in ["role_topk", "all"]:
        if args.slides_csv is None:
            raise ValueError("--slides_csv is required for role_topk/all")

        topk_out = os.path.join(args.out_dir, "role_topk")
        collect_role_topk_patches(
            slides_csv=args.slides_csv,
            config_path=args.config,
            adapted_student_ckpt=args.adapted_student_ckpt,
            stage2_full_ckpt=args.stage2_full_ckpt,
            role_proto_dir=args.role_proto_dir,
            out_dir=topk_out,
            device=args.device,
            img_size=args.img_size,
            extract_batch_size=args.batch_size,
            max_patches_per_slide=args.max_patches_per_slide,
            use_last_moe_output=args.use_last_moe_output,
            split=args.split,
            benchmark_split=args.benchmark_split,
            topk_per_role=args.topk_per_role,
            max_slides=args.role_topk_max_slides,
            svs_root=args.svs_root,
            h5_root=args.h5_root,
            project=args.project,
        )

    print("[Done]")


if __name__ == "__main__":
    main()
