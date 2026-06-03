#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.backbone_moe_factory import build_feature_extractor


# =========================================================
# 固定 case groups
# =========================================================
# CASE_GROUPS: Dict[str, List[str]] = {
#     "positive_improved": [
#         "F24-05247H04",
#         "F23-00599A01_H01",
#         "F22-00471A01_重切1",
#     ],
#     "positive_partially_fixed": [
#         "F24-06703H01_H02",
#         "F22-05516A02",
#     ],
#     "negative_hard_or_worse": [
#         "F23-02782H02",
#         "F24-00820H02",
#     ],
#     "stable_control": [
#         "F23-07239H01_H02",
#         "F23-03278H02_H03",
#     ],
# }

CASE_GROUPS: Dict[str, List[str]] = {
    "positive_improved": [
        "BRACS_1589",
        "BRACS_1936",
    ],
    "positive_partially_fixed": [
        "BRACS_1814",
        "BRACS_1938",
    ],
    "negative_improved": [
        "BRACS_1334",
        "BRACS_264",
    ],
    "negative_hard_or_worse": [
        "BRACS_1952",
        "BRACS_1843",
    ],
    "stable_control": [
        "BRACS_1003694",
        "BRACS_1940",
    ],
}


# =========================================================
# utils
# =========================================================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_slide_seed(base_seed: int, slide_id: str) -> int:
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(base_seed) + (int(h[:8], 16) % 100000)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / denom


def cosine_distance_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    sim = np.sum(a * b, axis=1)
    return 1.0 - sim


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


# =========================================================
# paths / csv
# =========================================================
def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir = Path(raw_dir)
    exts = [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"]

    exact_matches = []
    for ext in exts:
        exact_matches.extend(raw_dir.rglob(f"{slide_id}{ext}"))
    if len(exact_matches) == 1:
        return str(exact_matches[0])
    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Found multiple exact WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact_matches[:10])
        )

    fuzzy_matches = []
    for ext in exts:
        fuzzy_matches.extend(raw_dir.rglob(f"{slide_id}*{ext}"))
    if len(fuzzy_matches) == 1:
        return str(fuzzy_matches[0])
    if len(fuzzy_matches) > 1:
        exact_name = [p for p in fuzzy_matches if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(
            f"Found multiple WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy_matches[:10])
        )

    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir = Path(h5_dir)

    exact = list(h5_dir.rglob(f"{slide_id}.h5"))
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise RuntimeError(
            f"Found multiple exact h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact[:10])
        )

    fuzzy = list(h5_dir.rglob(f"{slide_id}*.h5"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
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
    return coords.astype(np.int64)


def load_pred_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip() == "":
        return None
    df = pd.read_csv(path)
    need = {"slide_id", "y_true", "y_prob"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"prediction csv missing columns: {missing}")
    return df.copy()


# =========================================================
# image / batch
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


def make_montage(
    pil_images: List[Image.Image],
    captions: List[str],
    tile_size: int = 224,
    n_cols: int = 4,
    caption_h: int = 24,
) -> Image.Image:
    assert len(pil_images) == len(captions)
    n = len(pil_images)
    n_rows = math.ceil(n / n_cols)
    canvas = Image.new("RGB", (n_cols * tile_size, n_rows * (tile_size + caption_h)), color=(255, 255, 255))

    for i, (img, cap) in enumerate(zip(pil_images, captions)):
        r = i // n_cols
        c = i % n_cols
        x0 = c * tile_size
        y0 = r * (tile_size + caption_h)
        img = ImageOps.fit(img, (tile_size, tile_size), method=Image.BICUBIC)
        canvas.paste(img, (x0, y0))

        # 简单 caption 条
        cap_img = Image.new("RGB", (tile_size, caption_h), color=(245, 245, 245))
        canvas.paste(cap_img, (x0, y0 + tile_size))

    return canvas


# =========================================================
# extractor args builder
# =========================================================
def build_base_namespace(args, encoder_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_name=encoder_name,
        device=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        max_patches=None,
        seed=args.seed,
        overwrite=False,

        # pathology foundation models
        virchow2_weight=args.virchow2_weight,
        uni_weight=args.uni_weight,
        uni2_weight="",

        # required placeholders for factory compatibility
        openclip_model_name="ViT-B-16",
        openclip_weight="",
        openclip_precision="fp16",
        no_openclip_normalize=False,
        hopt_local_hf_hub_id="",
        hopt_manual_arch_name="",
        hopt_weight="",
        dinov2_model_name="facebook/dinov2-small",
        dinov2_weight="",
        moe_ckpt="",
        moe_config="",
        dino_feature_layer=-1,

        # MoE bridge defaults / placeholders
        stage2_ckpt="",
        target_block_1=29,
        target_block_2=30,
        source_stage2_layer_1=9,
        source_stage2_layer_2=10,
        adapter_dim=384,
        adapter_hidden_dim=1536,
        num_experts=4,
        shared_expert=False,
        routing_strategy="proto_topany",
        top_k=2,
        init_threshold=0.0,
        min_experts=1,
        max_experts=2,
        gate_init_scale=2.0,
        gate_noise_std=0.02,
        shared_alpha=0.05,
        use_routing_proj=False,
        routing_metric="cosine",
        freeze_backbone_except_moe=False,
    )


def build_frozen_args(args) -> SimpleNamespace:
    ns = build_base_namespace(args, args.frozen_encoder_name)
    # frozen 分支不加载 stage2 moe
    ns.stage2_ckpt = ""
    return ns


def build_adapted_args(args) -> SimpleNamespace:
    ns = build_base_namespace(args, args.adapted_encoder_name)

    ns.stage2_ckpt = args.stage2_ckpt
    ns.target_block_1 = args.target_block_1
    ns.target_block_2 = args.target_block_2
    ns.source_stage2_layer_1 = args.source_stage2_layer_1
    ns.source_stage2_layer_2 = args.source_stage2_layer_2
    ns.adapter_dim = args.adapter_dim
    ns.adapter_hidden_dim = args.adapter_hidden_dim
    ns.num_experts = args.num_experts
    ns.shared_expert = args.shared_expert
    ns.routing_strategy = args.routing_strategy
    ns.top_k = args.top_k
    ns.init_threshold = args.init_threshold
    ns.min_experts = args.min_experts
    ns.max_experts = args.max_experts
    ns.gate_init_scale = args.gate_init_scale
    ns.gate_noise_std = args.gate_noise_std
    ns.shared_alpha = args.shared_alpha
    ns.use_routing_proj = args.use_routing_proj
    ns.routing_metric = args.routing_metric
    ns.freeze_backbone_except_moe = args.freeze_backbone_except_moe

    return ns
# =========================================================
# feature extraction
# =========================================================
@torch.no_grad()
def extract_one_slide_pair(
    frozen_extractor,
    adapted_extractor,
    slide_path: str,
    h5_path: str,
    patch_size: int,
    batch_size: int,
    max_patches: int,
    seed: int,
):
    coords = read_coords_from_h5(h5_path)
    if max_patches is not None and len(coords) > max_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_patches, replace=False)
        coords = coords[idx]

    slide = openslide.OpenSlide(slide_path)
    all_frozen, all_adapted, all_coords = [], [], []

    try:
        for start in tqdm(
            range(0, len(coords), batch_size),
            total=math.ceil(len(coords) / batch_size),
            desc=f"  Patches[{Path(slide_path).stem[:24]}]",
            leave=False,
        ):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]

            batch_images = []
            for xy in batch_coords.tolist():
                img = read_patch_from_wsi(
                    slide=slide,
                    coord_xy=xy,
                    patch_size=patch_size,
                    read_level=0,
                )
                img = img.resize((224, 224), resample=Image.BICUBIC)
                batch_images.append(img)

            frozen_feats = frozen_extractor.extract_features(batch_images).cpu().numpy().astype(np.float32)
            adapted_feats = adapted_extractor.extract_features(batch_images).cpu().numpy().astype(np.float32)

            all_frozen.append(frozen_feats)
            all_adapted.append(adapted_feats)
            all_coords.append(batch_coords.astype(np.int64))

    finally:
        slide.close()

    frozen = np.concatenate(all_frozen, axis=0)
    adapted = np.concatenate(all_adapted, axis=0)
    coords = np.concatenate(all_coords, axis=0)
    return frozen, adapted, coords


# =========================================================
# plotting helpers
# =========================================================
def plot_spatial_delta(ax, coords: np.ndarray, delta: np.ndarray, title: str):
    x = coords[:, 0]
    y = coords[:, 1]
    sc = ax.scatter(x, -y, c=delta, s=10, cmap="inferno")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def plot_umap_clusters(ax, emb: np.ndarray, cluster_ids: np.ndarray, title: str):
    uniq = sorted(np.unique(cluster_ids).tolist())
    cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(uniq):
        m = cluster_ids == cid
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.70, color=cmap(i % 20), label=f"c={cid}")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_transition_heatmap(ax, trans_mat: np.ndarray, title: str):
    im = ax.imshow(trans_mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("adapted cluster")
    ax.set_ylabel("frozen cluster")
    return im


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Case-slide visual analysis for frozen backbone vs backbone+MoE")
    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--frozen_pred_csv", type=str, default="")
    parser.add_argument("--adapted_pred_csv", type=str, default="")

    parser.add_argument(
        "--frozen_encoder_name",
        type=str,
        default="virchow2",
        choices=["virchow2", "uni"],
        help="Frozen backbone encoder name",
    )
    parser.add_argument(
        "--adapted_encoder_name",
        type=str,
        default="virchow2_moe",
        choices=["virchow2_moe", "uni_moe"],
        help="Adapted MoE encoder name",
    )

    parser.add_argument("--virchow2_weight", type=str, default="")
    parser.add_argument("--uni_weight", type=str, default="")

    parser.add_argument("--stage2_ckpt", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_patches", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--gallery_topk", type=int, default=16)

    # adapted bridge args
    parser.add_argument("--target_block_1", type=int, default=29)
    parser.add_argument("--target_block_2", type=int, default=30)
    parser.add_argument("--source_stage2_layer_1", type=int, default=9)
    parser.add_argument("--source_stage2_layer_2", type=int, default=10)
    parser.add_argument("--adapter_dim", type=int, default=384)
    parser.add_argument("--adapter_hidden_dim", type=int, default=1536)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--shared_expert", action="store_true")
    parser.add_argument("--routing_strategy", type=str, default="proto_topany")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--init_threshold", type=float, default=0.0)
    parser.add_argument("--min_experts", type=int, default=1)
    parser.add_argument("--max_experts", type=int, default=2)
    parser.add_argument("--gate_init_scale", type=float, default=2.0)
    parser.add_argument("--gate_noise_std", type=float, default=0.02)
    parser.add_argument("--shared_alpha", type=float, default=0.05)
    parser.add_argument("--use_routing_proj", action="store_true")
    parser.add_argument("--routing_metric", type=str, default="cosine")
    parser.add_argument("--freeze_backbone_except_moe", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(Path(args.out_dir) / "per_slide")
    ensure_dir(Path(args.out_dir) / "galleries")
    set_seed(args.seed)

    slides_df = pd.read_csv(args.slides_csv)
    if "slide_id" not in slides_df.columns and "image_id" in slides_df.columns:
        slides_df["slide_id"] = slides_df["image_id"]
    if "label" not in slides_df.columns:
        if "slide_binary_label" in slides_df.columns:
            slides_df["label"] = slides_df["slide_binary_label"]
        else:
            raise ValueError("slides_csv missing label / slide_binary_label")

    pred_frozen = load_pred_csv(args.frozen_pred_csv)
    pred_adapted = load_pred_csv(args.adapted_pred_csv)

    # build extractors
    frozen_args = build_frozen_args(args)
    adapted_args = build_adapted_args(args)

    print("[Build] frozen extractor ...")
    frozen_extractor = build_feature_extractor(frozen_args)
    print("[Build] adapted extractor ...")
    adapted_extractor = build_feature_extractor(adapted_args)

    selected_rows = []
    summary_rows = []

    for group_name, slide_ids in CASE_GROUPS.items():
        for slide_id in slide_ids:
            row = slides_df[slides_df["slide_id"] == slide_id]
            if len(row) == 0:
                print(f"[WARN] slide_id not found in slides_csv: {slide_id}")
                continue
            row = row.iloc[0]

            label = int(row["label"])
            slide_path = find_wsi_path(args.raw_dir, slide_id)
            h5_path = find_h5_path(args.h5_dir, slide_id)

            frozen_prob = None
            adapted_prob = None
            if pred_frozen is not None:
                sub = pred_frozen[pred_frozen["slide_id"] == slide_id]
                if len(sub) > 0:
                    frozen_prob = float(sub.iloc[0]["y_prob"])
            if pred_adapted is not None:
                sub = pred_adapted[pred_adapted["slide_id"] == slide_id]
                if len(sub) > 0:
                    adapted_prob = float(sub.iloc[0]["y_prob"])

            print(f"[Process] {group_name} | {slide_id}")

            frozen_feat, adapted_feat, coords = extract_one_slide_pair(
                frozen_extractor=frozen_extractor,
                adapted_extractor=adapted_extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
                seed=stable_slide_seed(args.seed, slide_id),
            )

            delta = cosine_distance_rows(frozen_feat, adapted_feat)

            # unified reducer + unified clustering
            all_feat = np.concatenate([frozen_feat, adapted_feat], axis=0)
            all_feat_norm = l2_normalize_np(all_feat)

            reducer = fit_reducer(all_feat_norm, args.seed, n_neighbors=15, min_dist=0.1)
            frozen_emb = transform_reducer(reducer, l2_normalize_np(frozen_feat))
            adapted_emb = transform_reducer(reducer, l2_normalize_np(adapted_feat))

            pca_dim = min(32, all_feat_norm.shape[1], max(2, all_feat_norm.shape[0] - 1))
            pca = PCA(n_components=pca_dim, random_state=args.seed)
            all_feat_pca = pca.fit_transform(all_feat_norm)
            km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
            all_cluster = km.fit_predict(all_feat_pca)
            frozen_cluster = all_cluster[:len(frozen_feat)]
            adapted_cluster = all_cluster[len(frozen_feat):]

            # transition matrix
            trans_mat = np.zeros((args.n_clusters, args.n_clusters), dtype=np.int64)
            for fc, ac in zip(frozen_cluster, adapted_cluster):
                trans_mat[int(fc), int(ac)] += 1

            slide_patch_df = pd.DataFrame({
                "slide_id": slide_id,
                "group": group_name,
                "label": label,
                "coord_x": coords[:, 0],
                "coord_y": coords[:, 1],
                "frozen_cluster": frozen_cluster,
                "adapted_cluster": adapted_cluster,
                "feature_cosine_delta": delta,
                "frozen_umap_x": frozen_emb[:, 0],
                "frozen_umap_y": frozen_emb[:, 1],
                "adapted_umap_x": adapted_emb[:, 0],
                "adapted_umap_y": adapted_emb[:, 1],
            })
            slide_patch_df.to_csv(Path(args.out_dir) / "per_slide" / f"{slide_id}_patch_analysis.csv", index=False)

            # main panel
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            plot_umap_clusters(
                axes[0, 0], frozen_emb, frozen_cluster,
                f"Frozen ({args.frozen_encoder_name}) UMAP\n{slide_id}"
            )
            plot_umap_clusters(
                axes[0, 1], adapted_emb, adapted_cluster,
                f"Adapted ({args.adapted_encoder_name}) UMAP\n{slide_id}"
            )
            sc = plot_spatial_delta(
                axes[1, 0], coords, delta,
                f"Spatial cosine shift\n{slide_id}"
            )
            im = plot_transition_heatmap(
                axes[1, 1], trans_mat,
                f"Cluster transition\n{slide_id}"
            )

            title_bits = [f"group={group_name}", f"y={label}"]
            if frozen_prob is not None:
                title_bits.append(f"frozen={frozen_prob:.3f}")
            if adapted_prob is not None:
                title_bits.append(f"adapted={adapted_prob:.3f}")
            fig.suptitle(" | ".join(title_bits), fontsize=12)

            fig.colorbar(sc, ax=axes[1, 0], fraction=0.046, pad=0.04)
            fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(Path(args.out_dir) / "per_slide" / f"{slide_id}_main_panel.png", dpi=220)
            plt.close(fig)

            # gallery: top changed / top stable
            order_changed = np.argsort(delta)[::-1]
            order_stable = np.argsort(delta)

            slide = openslide.OpenSlide(slide_path)
            try:
                changed_imgs, changed_caps = [], []
                stable_imgs, stable_caps = [], []

                for idx in order_changed[:args.gallery_topk]:
                    img = read_patch_from_wsi(
                        slide=slide,
                        coord_xy=(int(coords[idx, 0]), int(coords[idx, 1])),
                        patch_size=args.patch_size,
                        read_level=0,
                    )
                    changed_imgs.append(img)
                    changed_caps.append(f"{delta[idx]:.3f}")

                for idx in order_stable[:args.gallery_topk]:
                    img = read_patch_from_wsi(
                        slide=slide,
                        coord_xy=(int(coords[idx, 0]), int(coords[idx, 1])),
                        patch_size=args.patch_size,
                        read_level=0,
                    )
                    stable_imgs.append(img)
                    stable_caps.append(f"{delta[idx]:.3f}")

            finally:
                slide.close()

            if len(changed_imgs) > 0:
                montage = make_montage(changed_imgs, changed_caps, tile_size=224, n_cols=4)
                montage.save(Path(args.out_dir) / "galleries" / f"{slide_id}_top_changed_patches.png")

            if len(stable_imgs) > 0:
                montage = make_montage(stable_imgs, stable_caps, tile_size=224, n_cols=4)
                montage.save(Path(args.out_dir) / "galleries" / f"{slide_id}_top_stable_patches.png")

            selected_rows.append({
                "group": group_name,
                "slide_id": slide_id,
                "label": label,
                "frozen_prob": frozen_prob,
                "adapted_prob": adapted_prob,
                "delta_prob": (None if frozen_prob is None or adapted_prob is None else adapted_prob - frozen_prob),
            })

            summary_rows.append({
                "group": group_name,
                "slide_id": slide_id,
                "label": label,
                "num_patches": int(len(coords)),
                "mean_feature_cosine_delta": float(delta.mean()),
                "median_feature_cosine_delta": float(np.median(delta)),
                "top10pct_mean_delta": float(delta[order_changed[:max(1, len(delta)//10)]].mean()),
                "frozen_prob": frozen_prob,
                "adapted_prob": adapted_prob,
            })

    pd.DataFrame(selected_rows).to_csv(Path(args.out_dir) / "selected_case_slides.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(Path(args.out_dir) / "case_slide_summary.csv", index=False)

    with open(Path(args.out_dir) / "case_groups.json", "w", encoding="utf-8") as f:
        json.dump(CASE_GROUPS, f, indent=2, ensure_ascii=False)

    print(f"[Done] saved to: {args.out_dir}")


if __name__ == "__main__":
    main()