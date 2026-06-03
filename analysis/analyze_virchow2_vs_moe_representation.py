#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import h5py
import numpy as np
import pandas as pd
import openslide
from PIL import Image
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from models.encoders.virchow2_moe_encoder import Virchow2MoEEncoder


# =========================================================
# 基础工具
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir = Path(raw_dir)
    exts = [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"]

    exact_matches = []
    for ext in exts:
        exact_matches.extend(raw_dir.rglob(f"{slide_id}{ext}"))

    if len(exact_matches) == 1:
        return str(exact_matches[0])
    elif len(exact_matches) > 1:
        raise RuntimeError(
            f"Found multiple exact WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact_matches[:10])
        )

    fuzzy_matches = []
    for ext in exts:
        fuzzy_matches.extend(raw_dir.rglob(f"{slide_id}*{ext}"))

    if len(fuzzy_matches) == 1:
        return str(fuzzy_matches[0])
    elif len(fuzzy_matches) > 1:
        exact_name = [p for p in fuzzy_matches if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])

        raise RuntimeError(
            f"Found multiple WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy_matches[:10])
        )

    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def resolve_wsi_path(row: pd.Series, raw_dir: str | None = None) -> str:
    if "source_path" in row and pd.notna(row["source_path"]):
        source_path = str(row["source_path"])
        if os.path.exists(source_path):
            return source_path
        else:
            print(f"[WARN] source_path not found, fallback search: {source_path}")

    if raw_dir is not None:
        return find_wsi_path(raw_dir, str(row["slide_id"]))

    raise FileNotFoundError(
        f"Cannot resolve WSI path for slide_id={row['slide_id']}"
    )


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir = Path(h5_dir)

    exact = list(h5_dir.rglob(f"{slide_id}.h5"))
    if len(exact) == 1:
        return str(exact[0])
    elif len(exact) > 1:
        raise RuntimeError(
            f"Found multiple exact h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact[:10])
        )

    fuzzy = list(h5_dir.rglob(f"{slide_id}*.h5"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    elif len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])

        raise RuntimeError(
            f"Found multiple fuzzy h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy[:10])
        )

    raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")


def read_coords_from_h5(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return coords


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def load_slides_csv(slides_csv: str, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(slides_csv)

    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("slides_csv 需要包含 'label' 或 'slide_binary_label' 列")

    if split is not None:
        df = df[df["split"] == split].copy()

    df = df.reset_index(drop=True)
    return df


# =========================================================
# Frozen Virchow2 wrapper
# =========================================================
import timm
from timm.layers import SwiGLUPacked
from timm.data.transforms_factory import create_transform


class FrozenVirchow2Extractor(nn.Module):
    def __init__(self, weight_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=False,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            init_values=1e-5
        )

        state_dict = torch.load(weight_path, map_location="cpu")
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "encoder" in state_dict:
                state_dict = state_dict["encoder"]

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("model.", "")
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print(f"[FrozenVirchow2] strict load success")
        except Exception as e:
            print(f"[FrozenVirchow2] strict load failed, fallback strict=False: {e}")
            self.model.load_state_dict(new_state_dict, strict=False)

        if not hasattr(self.model, "pos_embed") or self.model.pos_embed is None:
            num_patches = self.model.patch_embed.num_patches + 1
            embed_dim = self.model.embed_dim
            self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.model = self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.transforms = create_transform(
            input_size=(3, 224, 224),
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0
        )

        self.embed_dim = self.model.embed_dim
        self.reg_tokens = getattr(self.model, "reg_tokens", 0)

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.transforms(img) for img in images]).to(self.device)
        tokens = self.model.forward_features(x)

        if isinstance(tokens, dict):
            if "x" in tokens:
                tokens = tokens["x"]
            elif "tokens" in tokens:
                tokens = tokens["tokens"]
            elif "features" in tokens:
                tokens = tokens["features"]
            else:
                raise TypeError(f"Unsupported Virchow2 forward_features dict keys: {tokens.keys()}")

        return tokens

    @torch.no_grad()
    def extract_patch_features(self, images: List[Image.Image]) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens
        patch_mean = tokens[:, patch_start:, :].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=-1)
        return feat


# =========================================================
# Virchow2 + MoE adapter wrapper
# =========================================================
class Virchow2MoEFeatureExtractor(nn.Module):
    def __init__(
        self,
        virchow2_weight: str,
        stage2_ckpt: str,
        device: str = "cuda",
        target_block_1: int = 29,
        target_block_2: int = 30,
        source_stage2_layer_1: int = 9,
        source_stage2_layer_2: int = 10,
        adapter_dim: int = 384,
        adapter_hidden_dim: int = 1536,
        num_experts: int = 4,
        shared_expert: bool = True,
        routing_strategy: str = "proto_topany",
        top_k: int = 2,
        init_threshold: float = 0.0,
        min_experts: int = 1,
        max_experts: int = 2,
        gate_init_scale: float = 2.0,
        gate_noise_std: float = 0.02,
        shared_alpha: float = 0.05,
        use_routing_proj: bool = True,
        routing_metric: str = "cosine",
        bridge_mode: str = "direct",
        residual_alpha_init: float = 0.05,
        learnable_residual_alpha: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        virchow2_cfg = {
            "weight_path": virchow2_weight,
            "device": str(self.device),
        }
        moe_cfg = {
            "moe_layers": [target_block_1, target_block_2],
            "adapter_dim": adapter_dim,
            "adapter_hidden_dim": adapter_hidden_dim,
            "num_experts": num_experts,
            "shared_expert": shared_expert,
            "routing_strategy": routing_strategy,
            "top_k": top_k,
            "init_threshold": init_threshold,
            "min_experts": min_experts,
            "max_experts": max_experts,
            "gate_init_scale": gate_init_scale,
            "gate_noise_std": gate_noise_std,
            "shared_alpha": shared_alpha,
            "use_routing_proj": use_routing_proj,
            "routing_metric": routing_metric,
            "bridge_mode": bridge_mode,
            "residual_alpha_init": residual_alpha_init,
            "learnable_residual_alpha": learnable_residual_alpha,
        }

        self.model = Virchow2MoEEncoder(virchow2_cfg, moe_cfg).to(self.device)
        self.model.load_stage2_moe_from_ckpt(
            stage2_ckpt_path=stage2_ckpt,
            target_to_source_layer_map={
                target_block_1: source_stage2_layer_1,
                target_block_2: source_stage2_layer_2,
            },
            strict=False,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.embed_dim = self.model.embed_dim
        self.reg_tokens = getattr(self.model, "reg_tokens", 0)
        self.bridge_mode = bridge_mode

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        tokens = self.model(images, return_gates=False, is_eval=True)
        return tokens

    @torch.no_grad()
    def extract_patch_features(self, images: List[Image.Image]) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens
        patch_mean = tokens[:, patch_start:, :].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=-1)
        return feat


# =========================================================
# 在线采样 + 特征提取
# =========================================================
@torch.no_grad()
def extract_slide_feature_online(
    extractor,
    slide_path: str,
    h5_path: str,
    patch_size: int = 256,
    num_sampled_patches: int = 128,
    batch_size: int = 32,
    sequential_sample: bool = False,
) -> np.ndarray:
    coords = read_coords_from_h5(h5_path)
    n_total = len(coords)
    if n_total == 0:
        raise RuntimeError(f"No coords in {h5_path}")

    if sequential_sample:
        coords = coords[: min(num_sampled_patches, n_total)]
    else:
        idx = np.random.permutation(n_total)[: min(num_sampled_patches, n_total)]
        coords = coords[idx]

    slide = openslide.OpenSlide(slide_path)

    patch_feats = []
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]

        batch_imgs = []
        for xy in batch_coords:
            img = read_patch_from_wsi(
                slide=slide,
                coord_xy=xy,
                patch_size=patch_size,
                read_level=0,
            )
            img = img.resize((224, 224), resample=Image.BICUBIC)
            batch_imgs.append(img)

        feats = extractor.extract_patch_features(batch_imgs).cpu().numpy()
        patch_feats.append(feats)

    slide.close()

    patch_feats = np.concatenate(patch_feats, axis=0)

    mean_feat = patch_feats.mean(axis=0)
    std_feat = patch_feats.std(axis=0)
    slide_feat = np.concatenate([mean_feat, std_feat], axis=0)

    return slide_feat


def build_subset(df: pd.DataFrame, max_slides_per_class: int) -> pd.DataFrame:
    dfs = []
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label].copy()
        if len(sub) > max_slides_per_class:
            sub = sub.sample(max_slides_per_class, random_state=42)
        dfs.append(sub)
    out = pd.concat(dfs, axis=0).reset_index(drop=True)
    return out


# =========================================================
# 分析指标
# =========================================================
def compute_distance_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    feats0 = features[labels == 0]
    feats1 = features[labels == 1]

    c0 = feats0.mean(axis=0)
    c1 = feats1.mean(axis=0)

    center_dist = float(np.linalg.norm(c0 - c1))
    intra0 = float(np.mean(np.linalg.norm(feats0 - c0[None, :], axis=1)))
    intra1 = float(np.mean(np.linalg.norm(feats1 - c1[None, :], axis=1)))
    ratio = center_dist / (0.5 * (intra0 + intra1) + 1e-8)

    return {
        "center_distance": center_dist,
        "intra_class_0": intra0,
        "intra_class_1": intra1,
        "distance_over_intra_mean": ratio,
    }


def compute_knn_label_consistency(
    features: np.ndarray,
    labels: np.ndarray,
    k_list: List[int] = [3, 5, 10],
) -> Dict[str, float]:
    max_k = max(k_list)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nbrs.fit(features)
    indices = nbrs.kneighbors(features, return_distance=False)

    results = {}
    for k in k_list:
        same_ratios = []
        for i in range(len(features)):
            neigh = indices[i, 1:k+1]
            ratio = float((labels[neigh] == labels[i]).mean())
            same_ratios.append(ratio)
        results[f"knn_same_label_ratio@{k}"] = float(np.mean(same_ratios))
    return results


def reduce_2d(features: np.ndarray, seed: int = 42) -> np.ndarray:
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(10, max(2, len(features) - 1)),
            min_dist=0.2,
            metric="euclidean",
            random_state=seed,
        )
        return reducer.fit_transform(features)
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=min(10, max(2, len(features) // 3)),
            random_state=seed,
            init="pca",
        )
        return reducer.fit_transform(features)


def plot_embedding_2d(
    emb2d: np.ndarray,
    labels: np.ndarray,
    slide_ids: List[str],
    title: str,
    save_path: str,
):
    plt.figure(figsize=(6, 5))
    for label in sorted(np.unique(labels)):
        mask = labels == label
        plt.scatter(
            emb2d[mask, 0],
            emb2d[mask, 1],
            label=f"class={label}",
            alpha=0.8,
            s=40,
        )

    for i in range(len(slide_ids)):
        plt.text(
            emb2d[i, 0],
            emb2d[i, 1],
            str(slide_ids[i])[:10],
            fontsize=7,
            alpha=0.7,
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--virchow2_weight", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--max_slides_per_class", type=int, default=15)
    parser.add_argument("--num_sampled_patches", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sequential_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target_block_1", type=int, default=29)
    parser.add_argument("--target_block_2", type=int, default=30)
    parser.add_argument("--source_stage2_layer_1", type=int, default=9)
    parser.add_argument("--source_stage2_layer_2", type=int, default=10)

    parser.add_argument("--residual_alpha_init", type=float, default=0.05)
    parser.add_argument("--learnable_residual_alpha", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    df = load_slides_csv(args.slides_csv, split=args.split)
    df = build_subset(df, args.max_slides_per_class)
    print(f"[Subset] num slides = {len(df)}")
    print(df["label"].value_counts())

    print("[Build] frozen Virchow2...")
    frozen_extractor = FrozenVirchow2Extractor(
        weight_path=args.virchow2_weight,
        device=args.device,
    )

    print("[Build] Virchow2 + direct bridge MoE...")
    direct_extractor = Virchow2MoEFeatureExtractor(
        virchow2_weight=args.virchow2_weight,
        stage2_ckpt=args.stage2_ckpt,
        device=args.device,
        target_block_1=args.target_block_1,
        target_block_2=args.target_block_2,
        source_stage2_layer_1=args.source_stage2_layer_1,
        source_stage2_layer_2=args.source_stage2_layer_2,
        bridge_mode="direct",
    )

    print("[Build] Virchow2 + residual bridge MoE...")
    residual_extractor = Virchow2MoEFeatureExtractor(
        virchow2_weight=args.virchow2_weight,
        stage2_ckpt=args.stage2_ckpt,
        device=args.device,
        target_block_1=args.target_block_1,
        target_block_2=args.target_block_2,
        source_stage2_layer_1=args.source_stage2_layer_1,
        source_stage2_layer_2=args.source_stage2_layer_2,
        bridge_mode="residual",
        residual_alpha_init=args.residual_alpha_init,
        learnable_residual_alpha=args.learnable_residual_alpha,
    )

    frozen_feats = []
    direct_feats = []
    residual_feats = []
    labels = []
    slide_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyze slides"):
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        slide_path = resolve_wsi_path(row, raw_dir=args.raw_dir)
        h5_path = find_h5_path(args.h5_dir, slide_id)

        try:
            feat_frozen = extract_slide_feature_online(
                extractor=frozen_extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                patch_size=args.patch_size,
                num_sampled_patches=args.num_sampled_patches,
                batch_size=args.batch_size,
                sequential_sample=args.sequential_sample,
            )
            feat_direct = extract_slide_feature_online(
                extractor=direct_extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                patch_size=args.patch_size,
                num_sampled_patches=args.num_sampled_patches,
                batch_size=args.batch_size,
                sequential_sample=args.sequential_sample,
            )
            feat_residual = extract_slide_feature_online(
                extractor=residual_extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                patch_size=args.patch_size,
                num_sampled_patches=args.num_sampled_patches,
                batch_size=args.batch_size,
                sequential_sample=args.sequential_sample,
            )
        except Exception as e:
            print(f"[ERROR] slide_id={slide_id}: {e}")
            continue

        frozen_feats.append(feat_frozen)
        direct_feats.append(feat_direct)
        residual_feats.append(feat_residual)
        labels.append(label)
        slide_ids.append(slide_id)

    frozen_feats = np.stack(frozen_feats, axis=0)
    direct_feats = np.stack(direct_feats, axis=0)
    residual_feats = np.stack(residual_feats, axis=0)
    labels = np.array(labels)

    np.save(os.path.join(args.out_dir, "frozen_slide_features.npy"), frozen_feats)
    np.save(os.path.join(args.out_dir, "direct_slide_features.npy"), direct_feats)
    np.save(os.path.join(args.out_dir, "residual_slide_features.npy"), residual_feats)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)

    frozen_dist = compute_distance_metrics(frozen_feats, labels)
    direct_dist = compute_distance_metrics(direct_feats, labels)
    residual_dist = compute_distance_metrics(residual_feats, labels)

    frozen_knn = compute_knn_label_consistency(frozen_feats, labels, k_list=[3, 5, 10])
    direct_knn = compute_knn_label_consistency(direct_feats, labels, k_list=[3, 5, 10])
    residual_knn = compute_knn_label_consistency(residual_feats, labels, k_list=[3, 5, 10])

    frozen_2d = reduce_2d(frozen_feats, seed=args.seed)
    direct_2d = reduce_2d(direct_feats, seed=args.seed)
    residual_2d = reduce_2d(residual_feats, seed=args.seed)

    plot_embedding_2d(
        frozen_2d, labels, slide_ids,
        title="Frozen Virchow2",
        save_path=os.path.join(args.out_dir, "umap_frozen.png"),
    )
    plot_embedding_2d(
        direct_2d, labels, slide_ids,
        title="Virchow2 + Direct Bridge MoE",
        save_path=os.path.join(args.out_dir, "umap_direct.png"),
    )
    plot_embedding_2d(
        residual_2d, labels, slide_ids,
        title="Virchow2 + Residual Bridge MoE",
        save_path=os.path.join(args.out_dir, "umap_residual.png"),
    )

    rows = [
        {"setting": "frozen", **frozen_dist, **frozen_knn},
        {"setting": "direct_bridge", **direct_dist, **direct_knn},
        {"setting": "residual_bridge", **residual_dist, **residual_knn},
    ]
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "representation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n[Summary]")
    print(summary_df)
    print(f"\nSaved to: {summary_path}")
    print(f"Saved plots:")
    print(f"  - {os.path.join(args.out_dir, 'umap_frozen.png')}")
    print(f"  - {os.path.join(args.out_dir, 'umap_direct.png')}")
    print(f"  - {os.path.join(args.out_dir, 'umap_residual.png')}")


if __name__ == "__main__":
    main()