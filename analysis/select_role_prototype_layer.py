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
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from tqdm import tqdm

from models.distill_teacher.virchow2 import Virchow2FeatureExtractor

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# basic utils
# =========================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T


def parse_role_csv_args(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --role-csv item: {item}, expected role=/path/to/file.csv")
        role, path = item.split("=", 1)
        role = role.strip()
        path = path.strip()
        if not role or not path:
            raise ValueError(f"Invalid --role-csv item: {item}")
        out[role] = path
    return out


# =========================================================
# dataset from tcga candidate csv
# =========================================================
class TCGARoleCandidateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size", "role"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe missing required columns: {missing}")

        self.df = df.reset_index(drop=True).copy()
        self.df["svs_path"] = self.df["svs_path"].map(canonicalize_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        svs_path = str(row["svs_path"])
        x = int(row["coord_x"])
        y = int(row["coord_y"])
        patch_level = int(row["patch_level"])
        patch_size = int(row["patch_size"])

        slide = openslide.OpenSlide(svs_path)
        try:
            img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()

        if self.transform is not None:
            img = self.transform(img)

        item = {
            "image": img,
            "svs_path": svs_path,
            "role": str(row["role"]),
            "project": str(row["project"]) if "project" in row and pd.notna(row["project"]) else "",
            "slide_id": str(row["slide_id"]) if "slide_id" in row and pd.notna(row["slide_id"]) else "",
            "organ_name": str(row["organ_name"]) if "organ_name" in row and pd.notna(row["organ_name"]) else (
                str(row["project"]) if "project" in row and pd.notna(row["project"]) else ""
            ),
            "coord_x": x,
            "coord_y": y,
            "coord_idx": int(row["coord_idx"]) if "coord_idx" in row and pd.notna(row["coord_idx"]) else -1,
            "patch_level": patch_level,
            "patch_size": patch_size,
        }
        return item


def collate_dict(batch: List[Dict]) -> Dict[str, object]:
    images = torch.stack([x["image"] for x in batch], dim=0)
    return {
        "image": images,
        "svs_path": [x["svs_path"] for x in batch],
        "role": [x["role"] for x in batch],
        "project": [x["project"] for x in batch],
        "slide_id": [x["slide_id"] for x in batch],
        "organ_name": [x["organ_name"] for x in batch],
        "coord_x": [x["coord_x"] for x in batch],
        "coord_y": [x["coord_y"] for x in batch],
        "coord_idx": [x["coord_idx"] for x in batch],
        "patch_level": [x["patch_level"] for x in batch],
        "patch_size": [x["patch_size"] for x in batch],
    }


# =========================================================
# virchow2 feature extraction
# =========================================================
@torch.no_grad()
def extract_teacher_layer_tokens(teacher_model, images: torch.Tensor, target_layer: int) -> torch.Tensor:
    """
    Returns [B, num_patches, D]
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
            x = x + model.pos_embed[:, : x.shape[1], :]
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
    assert num_prefix >= 0, f"illegal prefix count: total={total_tokens}, num_patches={num_patches}"

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
# load role csvs
# =========================================================
def load_role_csv(path: str, role_name: str, max_rows: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing}: {path}")

    df = df.copy()
    df["role"] = role_name

    if "organ_name" not in df.columns:
        if "project" in df.columns:
            df["organ_name"] = df["project"].astype(str)
        else:
            df["organ_name"] = "all"

    if max_rows is not None and len(df) > max_rows:
        if "organ_name" in df.columns:
            parts = []
            groups = list(df.groupby("organ_name"))
            per_group = max(1, max_rows // max(1, len(groups)))
            for _, sub in groups:
                if len(sub) > per_group:
                    sub = sub.sample(n=per_group, random_state=seed)
                parts.append(sub)
            df = pd.concat(parts, axis=0)
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=seed)
        else:
            df = df.sample(n=max_rows, random_state=seed)

    return df.reset_index(drop=True)


# =========================================================
# metrics
# =========================================================
def compute_prototypes(features: np.ndarray, labels: np.ndarray, role_names: List[str]) -> np.ndarray:
    protos = []
    for role in role_names:
        proto = features[labels == role].mean(axis=0)
        protos.append(proto)
    return np.stack(protos, axis=0)


def prototype_pairwise_cosine(prototypes: np.ndarray, role_names: List[str]) -> Dict[str, float]:
    sim = cosine_np(prototypes, prototypes)
    out = {}
    for i in range(len(role_names)):
        for j in range(i + 1, len(role_names)):
            out[f"{role_names[i]}__vs__{role_names[j]}"] = float(sim[i, j])
    return out


def compute_intra_compactness(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    role_names: List[str],
) -> Dict[str, float]:
    out = {}
    prot_map = {role: prototypes[i] for i, role in enumerate(role_names)}
    for role in role_names:
        sub = features[labels == role]
        proto = prot_map[role][None, :]
        sims = cosine_np(sub, proto).reshape(-1)
        out[f"{role}_mean_cos_to_proto"] = float(sims.mean())
        out[f"{role}_std_cos_to_proto"] = float(sims.std())
    return out


def nearest_prototype_eval(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    role_names: List[str],
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    sim = cosine_np(features, prototypes)
    pred_idx = sim.argmax(axis=1)
    pred_labels = np.array([role_names[i] for i in pred_idx], dtype=object)

    acc = float((pred_labels == labels).mean())
    cm = confusion_matrix(labels, pred_labels, labels=role_names)
    cm_df = pd.DataFrame(cm, index=role_names, columns=role_names)

    metrics = {"nearest_prototype_acc": acc}
    for role in role_names:
        mask = labels == role
        role_acc = float((pred_labels[mask] == labels[mask]).mean()) if mask.any() else float("nan")
        metrics[f"nearest_proto_acc_{role}"] = role_acc

    assign_df = pd.DataFrame({
        "true_role": labels,
        "pred_role": pred_labels,
        "max_sim": sim.max(axis=1),
    })
    return metrics, cm_df, assign_df


# =========================================================
# visualization
# =========================================================
def save_pca_plot(features: np.ndarray, labels: np.ndarray, role_names: List[str], outpath: str, title: str) -> None:
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(features)

    plt.figure(figsize=(7, 6))
    for role in role_names:
        mask = labels == role
        plt.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.55, label=role)
    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# =========================================================
# extraction loop
# =========================================================
def extract_feature_bank(df_all: pd.DataFrame, args, target_layer: int, device: str) -> pd.DataFrame:
    transform = build_transform(args.image_size)
    dataset = TCGARoleCandidateDataset(df_all, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_dict,
    )

    teacher_wrapper = Virchow2FeatureExtractor(device=device)
    teacher_model = teacher_wrapper.model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    rows = []
    for batch in tqdm(loader, desc=f"Extract layer{target_layer}"):
        images = batch["image"].to(device, non_blocking=True)
        tokens = extract_teacher_layer_tokens(teacher_model, images, target_layer=target_layer)
        pooled = pool_patch_tokens(tokens, mode=args.pooling)
        pooled = F.normalize(pooled, dim=-1)
        pooled_np = pooled.float().cpu().numpy()

        for i in range(len(batch["svs_path"])):
            row = {
                "svs_path": batch["svs_path"][i],
                "role": batch["role"][i],
                "project": batch["project"][i],
                "slide_id": batch["slide_id"][i],
                "organ_name": batch["organ_name"][i],
                "coord_x": int(batch["coord_x"][i]),
                "coord_y": int(batch["coord_y"][i]),
                "coord_idx": int(batch["coord_idx"][i]),
                "patch_level": int(batch["patch_level"][i]),
                "patch_size": int(batch["patch_size"][i]),
                "feature": pooled_np[i],
            }
            rows.append(row)

    feat_df = pd.DataFrame(rows)
    return feat_df


# =========================================================
# cli
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Compare Virchow2 layers for TCGA role prototype selection")
    parser.add_argument("--role-csv", action="append", required=True,
                        help="Repeated arg in form role=/path/to/candidate.csv")
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-per-role", type=int, default=12000)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-feature-bank", action="store_true")
    return parser.parse_args()


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.outdir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    role_csv_map = parse_role_csv_args(args.role_csv)
    role_names = list(role_csv_map.keys())

    role_dfs = []
    for role_name, csv_path in role_csv_map.items():
        role_dfs.append(load_role_csv(csv_path, role_name, args.max_per_role, args.seed))
    df_all = pd.concat(role_dfs, axis=0).reset_index(drop=True)

    all_layer_rows = []
    best_layer = None
    best_score = -1e9

    for layer in args.layers:
        layer_dir = os.path.join(args.outdir, f"layer_{layer}")
        ensure_dir(layer_dir)

        feat_df = extract_feature_bank(df_all, args, target_layer=layer, device=device)
        features = np.stack(feat_df["feature"].values, axis=0)
        labels = feat_df["role"].values.astype(object)

        prototypes = compute_prototypes(features, labels, role_names)
        pairwise = prototype_pairwise_cosine(prototypes, role_names)
        compactness = compute_intra_compactness(features, labels, prototypes, role_names)
        nearest_metrics, cm_df, assign_df = nearest_prototype_eval(features, labels, prototypes, role_names)

        mean_pairwise = float(np.mean(list(pairwise.values()))) if len(pairwise) > 0 else 0.0
        mean_compact = float(np.mean([compactness[f"{r}_mean_cos_to_proto"] for r in role_names]))
        score = nearest_metrics["nearest_prototype_acc"] + mean_compact - mean_pairwise

        metrics = {
            "layer": int(layer),
            "num_samples": int(len(feat_df)),
            "role_names": role_names,
            "pooling": args.pooling,
            "pairwise_cosine": pairwise,
            "compactness": compactness,
            "nearest_prototype": nearest_metrics,
            "mean_pairwise_cosine": mean_pairwise,
            "mean_compactness": mean_compact,
            "ranking_score": score,
        }

        with open(os.path.join(layer_dir, f"layer_{layer}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        np.save(os.path.join(layer_dir, f"layer_{layer}_prototypes.npy"), prototypes.astype(np.float32))

        assign_out = feat_df[[
            "svs_path", "project", "slide_id", "organ_name",
            "coord_x", "coord_y", "coord_idx", "patch_level", "patch_size", "role"
        ]].copy()
        assign_out = pd.concat([assign_out, assign_df.reset_index(drop=True)], axis=1)
        assign_out.to_csv(os.path.join(layer_dir, f"layer_{layer}_assignments.csv"), index=False)
        cm_df.to_csv(os.path.join(layer_dir, f"layer_{layer}_confusion.csv"))

        save_pca_plot(
            features=features,
            labels=labels,
            role_names=role_names,
            outpath=os.path.join(layer_dir, f"layer_{layer}_pca.png"),
            title=f"Virchow2 layer {layer} | PCA | roles",
        )

        if args.save_feature_bank:
            np.savez_compressed(
                os.path.join(layer_dir, f"layer_{layer}_feature_bank.npz"),
                features=features.astype(np.float32),
                labels=labels,
                svs_paths=feat_df["svs_path"].values.astype(object),
                projects=feat_df["project"].values.astype(object),
                slide_ids=feat_df["slide_id"].values.astype(object),
                organ_names=feat_df["organ_name"].values.astype(object),
                coord_x=feat_df["coord_x"].values,
                coord_y=feat_df["coord_y"].values,
                coord_idx=feat_df["coord_idx"].values,
                patch_level=feat_df["patch_level"].values,
                patch_size=feat_df["patch_size"].values,
            )

        row = {
            "layer": int(layer),
            "nearest_prototype_acc": nearest_metrics["nearest_prototype_acc"],
            "mean_pairwise_cosine": mean_pairwise,
            "mean_compactness": mean_compact,
            "ranking_score": score,
        }
        for role in role_names:
            row[f"nearest_proto_acc_{role}"] = nearest_metrics[f"nearest_proto_acc_{role}"]
            row[f"compact_{role}_mean_cos_to_proto"] = compactness[f"{role}_mean_cos_to_proto"]
        for k, v in pairwise.items():
            row[f"pairwise_{k}"] = v
        all_layer_rows.append(row)

        if score > best_score:
            best_score = score
            best_layer = int(layer)

    comp_df = pd.DataFrame(all_layer_rows).sort_values("ranking_score", ascending=False)
    comp_df.to_csv(os.path.join(args.outdir, "layer_comparison.csv"), index=False)

    best = {
        "best_layer": best_layer,
        "ranking_metric": "nearest_prototype_acc + mean_compactness - mean_pairwise_cosine",
        "notes": "Use as heuristic only; inspect PCA/confusion/visual purity jointly.",
    }
    with open(os.path.join(args.outdir, "best_layer.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved layer selection results to: {args.outdir}")
    print(f"Best layer by ranking score: {best_layer}")


if __name__ == "__main__":
    main()