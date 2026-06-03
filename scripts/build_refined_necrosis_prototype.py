#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageFile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from tqdm import tqdm

from models.distill_teacher.virchow2 import Virchow2FeatureExtractor

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# utils
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def cosine_to_proto_np(x: np.ndarray, proto: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    p = proto / (np.linalg.norm(proto) + eps)
    return x @ p


def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


# =========================================================
# exact Virchow2 layer token extraction
# =========================================================
@torch.no_grad()
def extract_teacher_layer_tokens(teacher_model, images: torch.Tensor, target_layer: int) -> torch.Tensor:
    """
    return: [B, num_patches, D]
    """
    model = teacher_model
    B = images.shape[0]

    x = model.patch_embed(images)

    if hasattr(model, "_pos_embed"):
        x = model._pos_embed(x)
    else:
        cls_token = model.cls_token.expand(B, -1, -1)
        num_reg = getattr(model, "reg_tokens", 0)

        if num_reg > 0 and hasattr(model, "reg_token") and model.reg_token is not None:
            reg_tokens = model.reg_token.expand(B, -1, -1)
            x = torch.cat([cls_token, reg_tokens, x], dim=1)
        else:
            x = torch.cat([cls_token, x], dim=1)

        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            x = x + model.pos_embed[:, :x.shape[1], :]
        if hasattr(model, "pos_drop"):
            x = model.pos_drop(x)

    if hasattr(model, "patch_drop"):
        x = model.patch_drop(x)
    if hasattr(model, "norm_pre"):
        x = model.norm_pre(x)

    assert hasattr(model, "blocks"), "teacher_model has no blocks"
    assert len(model.blocks) >= target_layer, (
        f"model.blocks has only {len(model.blocks)} layers, < {target_layer}"
    )

    for i, blk in enumerate(model.blocks, start=1):
        x = blk(x)
        if i == target_layer:
            break

    num_patches = model.patch_embed.num_patches
    total_tokens = x.shape[1]
    num_prefix = total_tokens - num_patches
    assert num_prefix >= 0, f"illegal prefix token count: total={total_tokens}, num_patches={num_patches}"

    patch_tokens = x[:, num_prefix:, :]
    assert patch_tokens.shape[1] == num_patches, (
        f"patch token count mismatch: got {patch_tokens.shape[1]}, expected {num_patches}"
    )
    return patch_tokens


def pool_patch_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "max":
        return tokens.max(dim=1).values
    raise ValueError(f"Unsupported pooling mode: {mode}")


# =========================================================
# dataset
# =========================================================
class TCGACandidateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size", "role"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df.reset_index(drop=True).copy()
        self.df["svs_path"] = self.df["svs_path"].map(canonicalize_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        svs_path = row["svs_path"]
        x = int(row["coord_x"])
        y = int(row["coord_y"])
        patch_level = int(row["patch_level"])
        patch_size = int(row["patch_size"])

        slide = openslide.OpenSlide(svs_path)
        try:
            img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()

        img = self.transform(img)

        meta = {
            "role": row["role"],
            "project": row["project"] if "project" in row else "",
            "slide_id": row["slide_id"] if "slide_id" in row else "",
            "svs_path": svs_path,
            "coord_x": x,
            "coord_y": y,
            "coord_idx": int(row["coord_idx"]) if "coord_idx" in row else -1,
            "patch_level": patch_level,
            "patch_size": patch_size,
        }
        return img, meta


def collate_with_meta(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    metas = [x[1] for x in batch]
    return images, metas


# =========================================================
# csv loading
# =========================================================
def load_role_csv(path: str, role_name: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df["role"] = role_name

    if "organ_name" not in df.columns:
        if "project" in df.columns:
            df["organ_name"] = df["project"].astype(str)
        else:
            df["organ_name"] = "all"

    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    return df.reset_index(drop=True)


# =========================================================
# feature extraction
# =========================================================
def extract_feature_df(
    df_all: pd.DataFrame,
    device: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    target_layer: int,
    pooling: str,
) -> pd.DataFrame:
    transform = build_transform(image_size=image_size)
    dataset = TCGACandidateDataset(df_all, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_with_meta,
    )

    teacher_wrapper = Virchow2FeatureExtractor(device=device)
    teacher_model = teacher_wrapper.model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    rows = []
    for images, metas in tqdm(loader, desc=f"Extract Virchow2 layer{target_layer}"):
        images = images.to(device, non_blocking=True)
        tokens = extract_teacher_layer_tokens(
            teacher_model=teacher_model,
            images=images,
            target_layer=target_layer,
        )
        feats = pool_patch_tokens(tokens, mode=pooling)
        feats = F.normalize(feats, dim=-1).float().cpu().numpy()

        for i, meta in enumerate(metas):
            row = dict(meta)
            row["feature"] = feats[i]
            rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# refinement
# =========================================================
def compute_rough_prototypes(feat_df: pd.DataFrame, role_order: List[str]) -> Dict[str, np.ndarray]:
    protos = {}
    for role in role_order:
        sub = feat_df[feat_df["role"] == role]
        if len(sub) == 0:
            raise RuntimeError(f"No samples for role: {role}")
        X = np.stack(sub["feature"].values, axis=0).astype(np.float32)
        proto = X.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        protos[role] = proto
    return protos


def add_similarity_columns(
    nec_df: pd.DataFrame,
    proto_nec: np.ndarray,
    proto_tumor: np.ndarray,
    proto_stroma: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    out = nec_df.copy()
    X = np.stack(out["feature"].values, axis=0).astype(np.float32)

    sim_nec = cosine_to_proto_np(X, proto_nec)
    sim_tumor = cosine_to_proto_np(X, proto_tumor)
    sim_stroma = cosine_to_proto_np(X, proto_stroma)
    sim_bg = np.maximum(sim_tumor, sim_stroma)

    purity_score = sim_nec - alpha * sim_bg
    margin_score = sim_nec - sim_bg

    out["sim_nec_rough"] = sim_nec
    out["sim_tumor"] = sim_tumor
    out["sim_stroma"] = sim_stroma
    out["sim_bg_max"] = sim_bg
    out["purity_score"] = purity_score
    out["margin_score"] = margin_score
    return out


def select_refined_necrosis_purity_topk(
    nec_df: pd.DataFrame,
    top_frac: float,
    min_keep: int,
    max_keep: int | None,
) -> pd.DataFrame:
    df = nec_df.sort_values(
        ["purity_score", "sim_nec_rough", "margin_score"],
        ascending=[False, False, False]
    ).copy()

    k = max(min_keep, int(round(len(df) * top_frac)))
    if max_keep is not None:
        k = min(k, max_keep)
    k = min(k, len(df))

    out = df.head(k).copy()
    out["refine_rank"] = np.arange(1, len(out) + 1)
    out["refine_method"] = "purity_topk"
    return out


def select_refined_necrosis_cluster_then_purity(
    nec_df: pd.DataFrame,
    n_clusters: int,
    cluster_top_frac: float,
    min_keep: int,
    max_keep: int | None,
    cluster_score_beta: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) cluster necrosis candidates in feature space
    2) score each cluster by:
         mean(purity_score) + beta * mean(sim_nec_rough) - compactness_penalty
       where compactness_penalty = std(sim_nec_rough)
    3) choose best cluster
    4) inside best cluster, do purity top-k
    """
    df = nec_df.copy()
    X = np.stack(df["feature"].values, axis=0).astype(np.float32)

    n_clusters = max(2, min(n_clusters, len(df)))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(X)
    df["cluster_id"] = cluster_ids

    cluster_rows = []
    for cid, sub in df.groupby("cluster_id"):
        cluster_rows.append({
            "cluster_id": int(cid),
            "count": int(len(sub)),
            "mean_purity_score": float(sub["purity_score"].mean()),
            "mean_sim_nec_rough": float(sub["sim_nec_rough"].mean()),
            "std_sim_nec_rough": float(sub["sim_nec_rough"].std()),
            "mean_margin_score": float(sub["margin_score"].mean()),
        })

    cluster_stats = pd.DataFrame(cluster_rows)
    cluster_stats["cluster_rank_score"] = (
        cluster_stats["mean_purity_score"]
        + cluster_score_beta * cluster_stats["mean_sim_nec_rough"]
        - cluster_stats["std_sim_nec_rough"].fillna(0.0)
    )
    cluster_stats = cluster_stats.sort_values(
        ["cluster_rank_score", "mean_purity_score", "mean_sim_nec_rough", "count"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    best_cid = int(cluster_stats.iloc[0]["cluster_id"])
    best_df = df[df["cluster_id"] == best_cid].copy()

    k = max(min_keep, int(round(len(best_df) * cluster_top_frac)))
    if max_keep is not None:
        k = min(k, max_keep)
    k = min(k, len(best_df))

    best_df = best_df.sort_values(
        ["purity_score", "sim_nec_rough", "margin_score"],
        ascending=[False, False, False]
    ).head(k).copy()
    best_df["refine_rank"] = np.arange(1, len(best_df) + 1)
    best_df["refine_method"] = "cluster_then_purity"
    best_df["selected_cluster_id"] = best_cid

    return best_df, cluster_stats


# =========================================================
# visualization
# =========================================================
def save_pca_three_roles(
    tumor_df: pd.DataFrame,
    stroma_df: pd.DataFrame,
    nec_df: pd.DataFrame,
    out_path: str,
    title: str,
):
    parts = [tumor_df.copy(), stroma_df.copy(), nec_df.copy()]
    all_df = pd.concat(parts, axis=0).reset_index(drop=True)
    X = np.stack(all_df["feature"].values, axis=0).astype(np.float32)
    y = all_df["role"].values.astype(object)

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)

    plt.figure(figsize=(8, 7))
    for role in ["tumor", "stroma", "necrosis"]:
        mask = (y == role)
        plt.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=0.60, label=role)

    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Build refined necrosis prototype from TCGA candidate CSVs")
    parser.add_argument("--tumor-csv", type=str, required=True)
    parser.add_argument("--stroma-csv", type=str, required=True)
    parser.add_argument("--necrosis-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--target-layer", type=int, default=26)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])

    parser.add_argument("--max-per-role", type=int, default=None)

    parser.add_argument("--refine-method", type=str, default="cluster_then_purity",
                        choices=["purity_topk", "cluster_then_purity"])
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="purity_score = sim_nec - alpha * max(sim_tumor, sim_stroma)")
    parser.add_argument("--top-frac", type=float, default=0.35)
    parser.add_argument("--min-keep", type=int, default=50)
    parser.add_argument("--max-keep", type=int, default=None)

    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--cluster-score-beta", type=float, default=0.5)

    parser.add_argument("--save-feature-bank", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    tumor_df = load_role_csv(args.tumor_csv, "tumor", args.max_per_role)
    stroma_df = load_role_csv(args.stroma_csv, "stroma", args.max_per_role)
    necrosis_df = load_role_csv(args.necrosis_csv, "necrosis", args.max_per_role)

    df_all = pd.concat([tumor_df, stroma_df, necrosis_df], axis=0).reset_index(drop=True)

    feat_df = extract_feature_df(
        df_all=df_all,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        target_layer=args.target_layer,
        pooling=args.pooling,
    )

    proto_map = compute_rough_prototypes(feat_df, ["tumor", "stroma", "necrosis"])

    tumor_feat_df = feat_df[feat_df["role"] == "tumor"].copy()
    stroma_feat_df = feat_df[feat_df["role"] == "stroma"].copy()
    nec_feat_df = feat_df[feat_df["role"] == "necrosis"].copy()

    nec_scored = add_similarity_columns(
        nec_df=nec_feat_df,
        proto_nec=proto_map["necrosis"],
        proto_tumor=proto_map["tumor"],
        proto_stroma=proto_map["stroma"],
        alpha=args.alpha,
    )

    cluster_stats = None
    if args.refine_method == "purity_topk":
        nec_refined = select_refined_necrosis_purity_topk(
            nec_df=nec_scored,
            top_frac=args.top_frac,
            min_keep=args.min_keep,
            max_keep=args.max_keep,
        )
    else:
        nec_refined, cluster_stats = select_refined_necrosis_cluster_then_purity(
            nec_df=nec_scored,
            n_clusters=args.n_clusters,
            cluster_top_frac=args.top_frac,
            min_keep=args.min_keep,
            max_keep=args.max_keep,
            cluster_score_beta=args.cluster_score_beta,
        )

    X_refined = np.stack(nec_refined["feature"].values, axis=0).astype(np.float32)
    refined_proto = X_refined.mean(axis=0)
    refined_proto = refined_proto / (np.linalg.norm(refined_proto) + 1e-8)

    pairwise = {
        "tumor__vs__stroma": float(np.dot(proto_map["tumor"], proto_map["stroma"])),
        "tumor__vs__necrosis_rough": float(np.dot(proto_map["tumor"], proto_map["necrosis"])),
        "stroma__vs__necrosis_rough": float(np.dot(proto_map["stroma"], proto_map["necrosis"])),
        "tumor__vs__necrosis_refined": float(np.dot(proto_map["tumor"], refined_proto)),
        "stroma__vs__necrosis_refined": float(np.dot(proto_map["stroma"], refined_proto)),
        "necrosis_rough__vs__necrosis_refined": float(np.dot(proto_map["necrosis"], refined_proto)),
    }

    refined_meta = {
        "target_layer": args.target_layer,
        "pooling": args.pooling,
        "refine_method": args.refine_method,
        "alpha": args.alpha,
        "top_frac": args.top_frac,
        "min_keep": args.min_keep,
        "max_keep": args.max_keep,
        "n_clusters": args.n_clusters,
        "cluster_score_beta": args.cluster_score_beta,
        "counts": {
            "tumor": int(len(tumor_feat_df)),
            "stroma": int(len(stroma_feat_df)),
            "necrosis_raw": int(len(nec_feat_df)),
            "necrosis_refined": int(len(nec_refined)),
        },
        "pairwise_cosine": pairwise,
        "refined_stats": {
            "mean_sim_nec_rough": float(nec_refined["sim_nec_rough"].mean()),
            "mean_sim_tumor": float(nec_refined["sim_tumor"].mean()),
            "mean_sim_stroma": float(nec_refined["sim_stroma"].mean()),
            "mean_purity_score": float(nec_refined["purity_score"].mean()),
            "mean_margin_score": float(nec_refined["margin_score"].mean()),
        }
    }

    # save prototypes
    np.save(os.path.join(args.output_dir, "necrosis_rough_prototype.npy"), proto_map["necrosis"].astype(np.float32))
    np.save(os.path.join(args.output_dir, "necrosis_refined_prototype.npy"), refined_proto.astype(np.float32))

    # save refined csv
    save_cols = [c for c in nec_refined.columns if c != "feature"]
    nec_refined[save_cols].to_csv(
        os.path.join(args.output_dir, "refined_necrosis_candidates.csv"),
        index=False
    )

    # save cluster stats if any
    if cluster_stats is not None:
        cluster_stats.to_csv(
            os.path.join(args.output_dir, "necrosis_cluster_stats.csv"),
            index=False
        )

    # save feature banks
    if args.save_feature_bank:
        np.savez_compressed(
            os.path.join(args.output_dir, "necrosis_refined_feature_bank.npz"),
            features=X_refined,
            svs_paths=nec_refined["svs_path"].values.astype(object),
            projects=nec_refined["project"].values.astype(object),
            slide_ids=nec_refined["slide_id"].values.astype(object),
            coord_x=nec_refined["coord_x"].values,
            coord_y=nec_refined["coord_y"].values,
            coord_idx=nec_refined["coord_idx"].values,
            patch_level=nec_refined["patch_level"].values,
            patch_size=nec_refined["patch_size"].values,
            sim_nec_rough=nec_refined["sim_nec_rough"].values,
            sim_tumor=nec_refined["sim_tumor"].values,
            sim_stroma=nec_refined["sim_stroma"].values,
            purity_score=nec_refined["purity_score"].values,
            margin_score=nec_refined["margin_score"].values,
        )

    # PCA before / after
    save_pca_three_roles(
        tumor_feat_df,
        stroma_feat_df,
        nec_feat_df,
        os.path.join(args.output_dir, "pca_tumor_stroma_necrosis_raw.png"),
        title=f"Layer {args.target_layer} | tumor/stroma/necrosis(raw)"
    )
    save_pca_three_roles(
        tumor_feat_df,
        stroma_feat_df,
        nec_refined.assign(role="necrosis"),
        os.path.join(args.output_dir, "pca_tumor_stroma_necrosis_refined.png"),
        title=f"Layer {args.target_layer} | tumor/stroma/necrosis(refined)"
    )

    with open(os.path.join(args.output_dir, "refined_necrosis_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(refined_meta, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(refined_meta, indent=2))


if __name__ == "__main__":
    main()