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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import ImageFile

import openslide
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

from sklearn.cluster import KMeans

from models.distill_teacher.virchow2 import Virchow2FeatureExtractor

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# utils
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / denom


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = l2_normalize(x)
    return x @ x.T


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform():
    return T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


# =========================================================
# args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Virchow2 role prototypes from candidate CSVs"
    )
    parser.add_argument("--role-csv", action="append", required=True,
                        help="Repeated arg in form role=/path/to/file.csv")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--target-layer", type=int, default=24)
    parser.add_argument("--token-pool", type=str, default="mean", choices=["mean", "cls"])

    parser.add_argument("--max-per-role", type=int, default=None,
                        help="Optional cap per role after CSV loading")
    parser.add_argument("--max-per-slide", type=int, default=None,
                        help="Optional cap per slide before prototype aggregation")

    parser.add_argument("--normalize-patch-feature", action="store_true")
    parser.add_argument("--normalize-prototype", action="store_true")

    parser.add_argument("--proto-agg", type=str, default="cluster_mean",
                        choices=["mean", "medoid", "cluster_mean"],
                        help="How to aggregate features into prototype")
    parser.add_argument("--cluster-k", type=int, default=3,
                        help="K for feature clustering when proto-agg=cluster_mean")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="Minimum cluster size to be considered valid")

    parser.add_argument(
        "--proto-source-level",
        type=str,
        default="patch",
        choices=["patch", "slide"],
        help="Prototype aggregation source level: "
             "'patch' = directly aggregate all matched patch features per role; "
             "'slide' = first aggregate features within each slide, then aggregate across slides."
    )

    parser.add_argument("--save-matched-features", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_role_csv_args(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --role-csv item: {item}, expected role=/path/to/file.csv")
        role, path = item.split("=", 1)
        role = role.strip()
        path = path.strip()
        out[role] = path
    return out


# =========================================================
# precise layer token extraction
# =========================================================
@torch.no_grad()
def extract_teacher_layer_tokens(teacher_model, images: torch.Tensor, target_layer: int) -> torch.Tensor:
    """
    精确提取 teacher 第 target_layer 层 patch tokens
    return:
        [B, num_patches, D]
    """
    model = teacher_model
    B = images.shape[0]

    x = model.patch_embed(images)   # [B, num_patches, D]

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
    assert len(model.blocks) >= target_layer, \
        f"model.blocks layers insufficient: {len(model.blocks)} < {target_layer}"

    for i, blk in enumerate(model.blocks, start=1):
        x = blk(x)
        if i == target_layer:
            break

    num_patches = model.patch_embed.num_patches
    total_tokens = x.shape[1]
    num_prefix = total_tokens - num_patches
    assert num_prefix >= 0, f"Illegal num_prefix: total={total_tokens}, num_patches={num_patches}"

    patch_tokens = x[:, num_prefix:, :]
    assert patch_tokens.shape[1] == num_patches, \
        f"patch token count mismatch: got {patch_tokens.shape[1]}, expected {num_patches}"

    return patch_tokens


# =========================================================
# dataset from candidate csv + svs locator
# =========================================================
class CandidateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

        required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size", "role"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Candidate CSV missing required columns: {missing}")

        self.df["svs_path"] = self.df["svs_path"].map(canonicalize_path)

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
            "h5_path": row["h5_path"] if "h5_path" in row else "",
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
# role csv loading / capping
# =========================================================
def load_role_csv(path: str, role_name: str, max_per_role: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["role"] = role_name

    required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    if max_per_role is not None and len(df) > max_per_role:
        df = df.iloc[:max_per_role].copy()

    return df.reset_index(drop=True)


def cap_per_slide(df: pd.DataFrame, max_per_slide: int | None) -> pd.DataFrame:
    if max_per_slide is None:
        return df.copy().reset_index(drop=True)

    if "slide_id" not in df.columns:
        return df.iloc[:].copy().reset_index(drop=True)

    parts = []
    for _, sub in df.groupby("slide_id", dropna=False):
        parts.append(sub.iloc[:max_per_slide].copy())

    if len(parts) == 0:
        return df.iloc[:0].copy()

    return pd.concat(parts, axis=0).reset_index(drop=True)


# =========================================================
# prototype aggregation
# =========================================================
def compute_medoid_feature(X: np.ndarray) -> np.ndarray:
    """
    Return one existing feature vector that is most central in cosine space.
    """
    Xn = l2_normalize(X)
    sim = Xn @ Xn.T
    mean_sim = sim.mean(axis=1)
    idx = int(np.argmax(mean_sim))
    return X[idx]


def compute_cluster_mean_feature(
    X: np.ndarray,
    cluster_k: int = 3,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """
    Cluster features in normalized cosine-like space using KMeans on L2-normalized features.
    Return mean feature of the largest valid cluster.
    """
    if len(X) <= cluster_k or len(X) < min_cluster_size:
        return X.mean(axis=0)

    Xn = l2_normalize(X)

    k = min(cluster_k, len(X))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xn)

    best_cluster = None
    best_size = -1
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) >= min_cluster_size and len(idx) > best_size:
            best_cluster = idx
            best_size = len(idx)

    if best_cluster is None:
        return X.mean(axis=0)

    return X[best_cluster].mean(axis=0)


def aggregate_role_feature(
    X: np.ndarray,
    proto_agg: str = "mean",
    cluster_k: int = 3,
    min_cluster_size: int = 3,
) -> np.ndarray:
    if proto_agg == "mean":
        return X.mean(axis=0)
    elif proto_agg == "medoid":
        return compute_medoid_feature(X)
    elif proto_agg == "cluster_mean":
        return compute_cluster_mean_feature(
            X,
            cluster_k=cluster_k,
            min_cluster_size=min_cluster_size,
        )
    else:
        raise ValueError(f"Unknown proto_agg: {proto_agg}")


def aggregate_role_feature_by_slide(
    sub_df: pd.DataFrame,
    normalize_patch_feature: bool,
    normalize_prototype: bool,
    proto_agg: str,
    cluster_k: int,
    min_cluster_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    First aggregate patch features within each slide, then aggregate slide-level features
    into the final role prototype.
    """
    if "slide_id" not in sub_df.columns:
        raise ValueError("slide-level prototype aggregation requires 'slide_id' column")

    slide_features = []
    slide_patch_counts = {}

    for slide_id, sub_slide in sub_df.groupby("slide_id", dropna=False):
        X_slide = np.stack(sub_slide["feature"].values, axis=0).astype(np.float32)

        if normalize_patch_feature:
            X_slide = l2_normalize(X_slide)

        slide_feat = aggregate_role_feature(
            X_slide,
            proto_agg=proto_agg,
            cluster_k=cluster_k,
            min_cluster_size=min_cluster_size,
        )
        slide_features.append(slide_feat)
        slide_patch_counts[str(slide_id)] = int(len(sub_slide))

    X_role = np.stack(slide_features, axis=0).astype(np.float32)

    proto = aggregate_role_feature(
        X_role,
        proto_agg=proto_agg,
        cluster_k=cluster_k,
        min_cluster_size=min_cluster_size,
    )

    if normalize_prototype:
        proto = l2_normalize(proto[None, :])[0]

    stats = {
        "num_matched_patches": int(len(sub_df)),
        "num_slides_used": int(len(slide_features)),
        "mean_patches_per_slide": float(np.mean(list(slide_patch_counts.values()))),
        "slide_patch_counts": slide_patch_counts,
    }
    return proto, stats


def build_role_prototypes(
    matched_df: pd.DataFrame,
    normalize_patch_feature: bool,
    normalize_prototype: bool,
    proto_agg: str,
    cluster_k: int,
    min_cluster_size: int,
    proto_source_level: str = "patch",
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    role_names = list(dict.fromkeys(matched_df["role"].tolist()))
    prototypes = []
    role_stats: Dict[str, object] = {}

    for role in role_names:
        sub = matched_df[matched_df["role"] == role]
        if len(sub) == 0:
            raise RuntimeError(f"No matched features found for role '{role}'")

        X = np.stack(sub["feature"].values, axis=0).astype(np.float32)

        if proto_source_level == "patch":
            if normalize_patch_feature:
                X = l2_normalize(X)

            proto = aggregate_role_feature(
                X,
                proto_agg=proto_agg,
                cluster_k=cluster_k,
                min_cluster_size=min_cluster_size,
            )

            if normalize_prototype:
                proto = l2_normalize(proto[None, :])[0]

            prototypes.append(proto)

            role_stats[role] = {
                "num_matched_patches": int(len(sub)),
                "feature_dim": int(X.shape[1]),
                "mean_feature_norm": float(np.linalg.norm(X, axis=1).mean()),
                "proto_agg": proto_agg,
                "proto_source_level": proto_source_level,
            }

        elif proto_source_level == "slide":
            proto, slide_stats = aggregate_role_feature_by_slide(
                sub_df=sub,
                normalize_patch_feature=normalize_patch_feature,
                normalize_prototype=normalize_prototype,
                proto_agg=proto_agg,
                cluster_k=cluster_k,
                min_cluster_size=min_cluster_size,
            )

            prototypes.append(proto)

            role_stats[role] = {
                "num_matched_patches": int(len(sub)),
                "feature_dim": int(X.shape[1]),
                "mean_feature_norm": float(np.linalg.norm(X, axis=1).mean()),
                "proto_agg": proto_agg,
                "proto_source_level": proto_source_level,
                **slide_stats,
            }

        else:
            raise ValueError(f"Unknown proto_source_level: {proto_source_level}")

        if "project" in sub.columns:
            role_stats[role]["project_counts"] = sub["project"].value_counts().to_dict()
        if "slide_id" in sub.columns:
            role_stats[role]["slide_counts"] = sub["slide_id"].value_counts().to_dict()

    prototypes = np.stack(prototypes, axis=0).astype(np.float32)

    meta = {
        "role_names": role_names,
        "num_total_target_rows": int(len(matched_df)),
        "normalize_patch_feature": bool(normalize_patch_feature),
        "normalize_prototype": bool(normalize_prototype),
        "proto_agg": proto_agg,
        "proto_source_level": proto_source_level,
        "cluster_k": int(cluster_k),
        "min_cluster_size": int(min_cluster_size),
        "role_stats": role_stats,
    }
    return prototypes, role_names, meta


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    seed_everything(args.seed)

    role_csv_map = parse_role_csv_args(args.role_csv)

    role_dfs = []
    for role_name, csv_path in role_csv_map.items():
        df = load_role_csv(csv_path, role_name, args.max_per_role)
        role_dfs.append(df)

    df_all = pd.concat(role_dfs, axis=0).reset_index(drop=True)
    print(f"[Info] total candidate rows before per-slide cap = {len(df_all)}")
    print(df_all["role"].value_counts())

    df_all = cap_per_slide(df_all, args.max_per_slide)
    print(f"[Info] total candidate rows after per-slide cap = {len(df_all)}")
    print(df_all["role"].value_counts())

    transform = build_transform()
    dataset = CandidateDataset(df_all, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_with_meta,
    )

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    teacher_wrapper = Virchow2FeatureExtractor(device=device)
    teacher_model = teacher_wrapper.model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    matched_rows = []

    for images, metas in tqdm(loader, desc=f"Extract Virchow2 layer{args.target_layer} features"):
        images = images.to(device, non_blocking=True)

        tokens = extract_teacher_layer_tokens(
            teacher_model=teacher_model,
            images=images,
            target_layer=args.target_layer,
        )  # [B, N, D]

        if args.token_pool == "mean":
            feats = tokens.mean(dim=1)
        elif args.token_pool == "cls":
            raise NotImplementedError(
                "CLS pooling is not supported here because patch tokens exclude prefix tokens."
            )
        else:
            raise ValueError(f"Unknown token_pool: {args.token_pool}")

        feats = feats.float().cpu().numpy()

        for i, meta in enumerate(metas):
            row = dict(meta)
            row["feature"] = feats[i]
            matched_rows.append(row)

    matched_df = pd.DataFrame(matched_rows)
    if len(matched_df) == 0:
        raise RuntimeError("No matched features extracted.")

    prototypes, role_names, meta = build_role_prototypes(
        matched_df=matched_df,
        normalize_patch_feature=args.normalize_patch_feature,
        normalize_prototype=args.normalize_prototype,
        proto_agg=args.proto_agg,
        cluster_k=args.cluster_k,
        min_cluster_size=args.min_cluster_size,
        proto_source_level=args.proto_source_level,
    )

    pairwise = cosine_matrix(prototypes)
    pairwise_dict = {}
    for i in range(len(role_names)):
        for j in range(i + 1, len(role_names)):
            pairwise_dict[f"{role_names[i]}__vs__{role_names[j]}"] = float(pairwise[i, j])

    np.save(os.path.join(args.output_dir, "role_prototypes_init.npy"), prototypes)

    with open(os.path.join(args.output_dir, "role_names.json"), "w", encoding="utf-8") as f:
        json.dump(role_names, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "prototype_pairwise_cosine.json"), "w", encoding="utf-8") as f:
        json.dump(pairwise_dict, f, ensure_ascii=False, indent=2)

    meta["pairwise_cosine"] = pairwise_dict
    meta["target_layer"] = args.target_layer
    meta["token_pool"] = args.token_pool
    meta["max_per_slide"] = args.max_per_slide
    meta["max_per_role"] = args.max_per_role

    with open(os.path.join(args.output_dir, "role_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_cols = [c for c in matched_df.columns if c != "feature"]
    matched_df[save_cols].to_csv(
        os.path.join(args.output_dir, "matched_patch_features.csv"),
        index=False
    )

    if args.save_matched_features:
        X = np.stack(matched_df["feature"].values, axis=0).astype(np.float32)
        out_npz = {
            "features": X,
            "roles": matched_df["role"].values.astype(object),
            "svs_paths": matched_df["svs_path"].values.astype(object),
            "coord_x": matched_df["coord_x"].values,
            "coord_y": matched_df["coord_y"].values,
            "patch_level": matched_df["patch_level"].values,
            "patch_size": matched_df["patch_size"].values,
        }
        if "project" in matched_df.columns:
            out_npz["projects"] = matched_df["project"].values.astype(object)
        if "slide_id" in matched_df.columns:
            out_npz["slide_ids"] = matched_df["slide_id"].values.astype(object)

        np.savez_compressed(
            os.path.join(args.output_dir, "matched_patch_features.npz"),
            **out_npz
        )

    print("[Done] prototypes saved.")
    print("Role counts:")
    for role, stats in meta["role_stats"].items():
        print(f"  {role}: {stats['num_matched_patches']}")
        if "num_slides_used" in stats:
            print(f"    slides_used: {stats['num_slides_used']}")
            print(f"    mean_patches_per_slide: {stats['mean_patches_per_slide']:.4f}")

    print("Pairwise cosine:")
    for k, v in pairwise_dict.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()