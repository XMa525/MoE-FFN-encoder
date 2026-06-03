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
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.backbone_moe_factory import build_feature_extractor
from models.plugins.shared_role_prototype import (
    SharedRolePrototype,
    PatchRoleSummaryFromSharedProto,
)


CASE_GROUPS: Dict[str, List[str]] = {
    "positive_improved": [
        "BRACS_1589",
        "BRACS_1936",
    ],
    "negative_improved": [
        "BRACS_1334",
        "BRACS_264",
    ],
    "negative_hard_or_worse": [
        "BRACS_1843",
        "BRACS_1952",
    ],
}


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(seed: int, slide_id: str) -> int:
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(seed) + (int(h[:8], 16) % 100000)


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def robust_zscore_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean()
    sd = x.std()
    if sd < eps:
        return x - mu
    return (x - mu) / (sd + eps)


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


def resolve_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "slide_id" not in df.columns and "image_id" in df.columns:
        df["slide_id"] = df["image_id"]
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("slides_csv missing label / slide_binary_label")
    return df


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    return slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")


def make_montage(
    pil_images: List[Image.Image],
    tile_size: int = 224,
    n_cols: int = 4,
) -> Image.Image:
    n = len(pil_images)
    if n == 0:
        return Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
    n_rows = math.ceil(n / n_cols)
    canvas = Image.new("RGB", (n_cols * tile_size, n_rows * tile_size), color=(255, 255, 255))

    for i, img in enumerate(pil_images):
        r = i // n_cols
        c = i % n_cols
        x0 = c * tile_size
        y0 = r * tile_size
        img = ImageOps.fit(img, (tile_size, tile_size), method=Image.BICUBIC)
        canvas.paste(img, (x0, y0))
    return canvas


def build_base_namespace(args, encoder_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_name=encoder_name,
        device=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        max_patches=None,
        seed=args.seed,
        overwrite=False,

        virchow2_weight=args.virchow2_weight,
        uni_weight=args.uni_weight,
        uni2_weight="",

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

        stage2_ckpt="",
        target_block_1=args.target_block_1,
        target_block_2=args.target_block_2,
        source_stage2_layer_1=args.source_stage2_layer_1,
        source_stage2_layer_2=args.source_stage2_layer_2,
        adapter_dim=args.adapter_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        num_experts=args.num_experts,
        shared_expert=args.shared_expert,
        routing_strategy=args.routing_strategy,
        top_k=args.top_k,
        init_threshold=args.init_threshold,
        min_experts=args.min_experts,
        max_experts=args.max_experts,
        gate_init_scale=args.gate_init_scale,
        gate_noise_std=args.gate_noise_std,
        shared_alpha=args.shared_alpha,
        use_routing_proj=args.use_routing_proj,
        routing_metric=args.routing_metric,
        freeze_backbone_except_moe=args.freeze_backbone_except_moe,
    )

def build_frozen_args(args) -> SimpleNamespace:
    ns = build_base_namespace(args, args.frozen_encoder_name)
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


@torch.no_grad()
def extract_features_for_coords(
    extractor,
    slide_path: str,
    coords: np.ndarray,
    patch_size: int,
    batch_size: int,
    feature_mode: str = "final",
) -> np.ndarray:
    slide = openslide.OpenSlide(slide_path)
    feats_all = []

    try:
        for start in range(0, len(coords), batch_size):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]

            batch_images = []
            for xy in batch_coords.tolist():
                img = read_patch_from_wsi(
                    slide=slide,
                    coord_xy=(int(xy[0]), int(xy[1])),
                    patch_size=patch_size,
                    read_level=0,
                )
                img = img.resize((224, 224), resample=Image.BICUBIC)
                batch_images.append(img)

            feat = extractor.extract_features(batch_images, feature_mode=feature_mode)
            if torch.is_tensor(feat):
                feat = feat.detach().cpu().numpy()
            feat = np.asarray(feat, dtype=np.float32)
            feats_all.append(feat)

    finally:
        slide.close()

    return np.concatenate(feats_all, axis=0)


@torch.no_grad()
def compute_role_scores(
    feats_np: np.ndarray,
    proj_layer: Optional[torch.nn.Module],
    summary_builder: Optional[PatchRoleSummaryFromSharedProto],
    role_names: Optional[List[str]],
    tumor_name: Optional[str],
    negative_role_names: Optional[List[str]],
    device: str,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    if proj_layer is None or summary_builder is None or role_names is None or tumor_name is None or negative_role_names is None:
        return out

    feats = torch.from_numpy(feats_np).float().to(device)
    feats = proj_layer(feats)
    feats = F.normalize(feats, dim=-1)

    role_dict = summary_builder(feats.unsqueeze(0))
    role_logits = role_dict["patch_role_logits"][0].detach().cpu().numpy()
    role_probs = role_dict["patch_role_probs"][0].detach().cpu().numpy()

    role_to_idx = {n: i for i, n in enumerate(role_names)}
    tumor_idx = role_to_idx[tumor_name]
    neg_ids = [role_to_idx[n] for n in negative_role_names if n in role_to_idx]
    if len(neg_ids) == 0:
        raise ValueError(f"No valid negative roles found in {negative_role_names}")

    tumor_logit = role_logits[:, tumor_idx]
    tumor_prob = role_probs[:, tumor_idx]
    neg_logit = role_logits[:, neg_ids].max(axis=1)
    tumor_gap = tumor_logit - neg_logit

    out["tumor_logit"] = tumor_logit.astype(np.float32)
    out["tumor_prob"] = tumor_prob.astype(np.float32)
    out["neg_logit"] = neg_logit.astype(np.float32)
    out["tumor_gap"] = tumor_gap.astype(np.float32)

    for ridx, rname in enumerate(role_names):
        out[f"role_logit__{rname}"] = role_logits[:, ridx].astype(np.float32)
        out[f"role_prob__{rname}"] = role_probs[:, ridx].astype(np.float32)
        if rname != tumor_name:
            out[f"margin_{rname}_over_tumor"] = (role_logits[:, ridx] - tumor_logit).astype(np.float32)

    return out


def local_knn_purity(
    feats: np.ndarray,
    slide_labels: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    feats = l2_normalize_np(feats)
    dists = pairwise_distances(feats, metric="cosine")
    np.fill_diagonal(dists, np.inf)

    k = min(k, len(feats) - 1)
    if k <= 0:
        return np.ones(len(feats), dtype=np.float32)

    nn_idx = np.argsort(dists, axis=1)[:, :k]
    pur = []
    for i in range(len(feats)):
        same = (slide_labels[nn_idx[i]] == slide_labels[i]).mean()
        pur.append(float(same))
    return np.asarray(pur, dtype=np.float32)


def plot_spatial_map(ax, coords: np.ndarray, values: np.ndarray, title: str, cmap: str = "viridis"):
    x = coords[:, 0]
    y = coords[:, 1]
    sc = ax.scatter(x, -y, c=values, s=10, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def plot_transition_heatmap(ax, mat: np.ndarray, title: str):
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("adapted cluster")
    ax.set_ylabel("frozen cluster")
    return im


def main():
    parser = argparse.ArgumentParser("Online encoder-level evidence analysis for frozen backbone vs backbone+MoE")

    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--frozen_encoder_name", type=str, default="uni", choices=["virchow2", "uni"])
    parser.add_argument("--adapted_encoder_name", type=str, default="uni_moe", choices=["virchow2_moe", "uni_moe"])

    parser.add_argument("--virchow2_weight", type=str, default="")
    parser.add_argument("--uni_weight", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_patches", type=int, default=1024)

    parser.add_argument("--feature_mode", type=str, default="final", choices=["final", "moe_last", "layer_last", "layer_m4"])

    parser.add_argument("--gallery_topk", type=int, default=16)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--knn_k", type=int, default=15)

    parser.add_argument("--role_proto_dir", type=str, default="")
    parser.add_argument("--role_tau", type=float, default=1.0)
    parser.add_argument("--proto_tumor_name", type=str, default="tumor")
    parser.add_argument("--proto_negative_role_names", nargs="*", default=["stroma"])
    parser.add_argument("--feature_proj_ckpt", type=str, default="")

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
    ensure_dir(Path(args.out_dir) / "tables")
    set_seed(args.seed)

    slides_df = pd.read_csv(args.slides_csv)
    slides_df = resolve_label(slides_df)

    selected_ids = []
    for _, lst in CASE_GROUPS.items():
        selected_ids.extend(lst)
    selected_ids = list(dict.fromkeys(selected_ids))
    use_df = slides_df[slides_df["slide_id"].isin(selected_ids)].copy().reset_index(drop=True)

    if len(use_df) == 0:
        raise RuntimeError("No selected slides found in slides_csv.")

    print("[Build] frozen extractor ...")
    frozen_extractor = build_feature_extractor(build_frozen_args(args))
    print("[Build] adapted extractor ...")
    adapted_extractor = build_feature_extractor(build_adapted_args(args))

    proj_layer = None
    summary_builder = None
    role_names = None

    if args.role_proto_dir.strip():
        device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        shared_role_proto = SharedRolePrototype.from_files(
            role_proto_dir=args.role_proto_dir,
            normalize=True,
            learnable=False,
            device=device,
        )
        role_names = list(shared_role_proto.role_names)

        summary_builder = PatchRoleSummaryFromSharedProto(
            shared_role_proto=shared_role_proto,
            tau=args.role_tau,
            use_softmax=True,
        ).to(device)
        summary_builder.eval()

        if args.feature_proj_ckpt.strip():
            ckpt = torch.load(args.feature_proj_ckpt, map_location="cpu", weights_only=False)
            if "distiller_state_dict" in ckpt:
                sd = ckpt["distiller_state_dict"]
                if "proj_l12.weight" in sd and "proj_l12.bias" in sd:
                    in_dim = sd["proj_l12.weight"].shape[1]
                    out_dim = sd["proj_l12.weight"].shape[0]
                    proj_layer = torch.nn.Linear(in_dim, out_dim)
                    proj_layer.load_state_dict({
                        "weight": sd["proj_l12.weight"],
                        "bias": sd["proj_l12.bias"],
                    })
                    proj_layer = proj_layer.to(device)
                    proj_layer.eval()
                else:
                    print("[WARN] proj_l12 not found in feature_proj_ckpt; role proto scoring disabled.")
                    summary_builder = None
                    role_names = None
            else:
                print("[WARN] distiller_state_dict not found in feature_proj_ckpt; role proto scoring disabled.")
                summary_builder = None
                role_names = None
        else:
            print("[WARN] role_proto_dir given but no feature_proj_ckpt; role proto scoring disabled.")
            summary_builder = None
            role_names = None

    all_frozen_feats = []
    all_adapted_feats = []
    all_patch_slide_labels = []
    all_patch_slide_ids = []
    all_patch_groups = []

    per_slide_summary_rows = []
    per_slide_cache = {}

    for _, row in use_df.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        group_name = "ungrouped"
        for g, lst in CASE_GROUPS.items():
            if slide_id in lst:
                group_name = g
                break

        print(f"[Process] {group_name} | {slide_id}")

        slide_path = find_wsi_path(args.raw_dir, slide_id)
        h5_path = find_h5_path(args.h5_dir, slide_id)

        coords = read_coords_from_h5(h5_path)
        if args.max_patches is not None and len(coords) > args.max_patches:
            rng = np.random.default_rng(stable_seed(args.seed, slide_id))
            sel = rng.choice(len(coords), size=args.max_patches, replace=False)
            coords = coords[sel]

        frozen_feat = extract_features_for_coords(
            extractor=frozen_extractor,
            slide_path=slide_path,
            coords=coords,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            feature_mode=args.feature_mode,
        )
        adapted_feat = extract_features_for_coords(
            extractor=adapted_extractor,
            slide_path=slide_path,
            coords=coords,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            feature_mode=args.feature_mode,
        )

        if len(frozen_feat) != len(adapted_feat) or len(frozen_feat) != len(coords):
            raise RuntimeError(f"Feature/coord length mismatch for {slide_id}")

        frozen_norm = np.linalg.norm(frozen_feat, axis=1)
        adapted_norm = np.linalg.norm(adapted_feat, axis=1)
        norm_shift = adapted_norm - frozen_norm

        cosine_sim = np.sum(l2_normalize_np(frozen_feat) * l2_normalize_np(adapted_feat), axis=1)
        cosine_shift = 1.0 - cosine_sim

        frozen_scores = compute_role_scores(
            frozen_feat, proj_layer, summary_builder, role_names,
            args.proto_tumor_name, args.proto_negative_role_names, args.device
        )
        adapted_scores = compute_role_scores(
            adapted_feat, proj_layer, summary_builder, role_names,
            args.proto_tumor_name, args.proto_negative_role_names, args.device
        )

        tumor_gap_delta = None
        if "tumor_gap" in frozen_scores and "tumor_gap" in adapted_scores:
            tumor_gap_delta = adapted_scores["tumor_gap"] - frozen_scores["tumor_gap"]

        all_frozen_feats.append(frozen_feat)
        all_adapted_feats.append(adapted_feat)
        all_patch_slide_labels.extend([label] * len(coords))
        all_patch_slide_ids.extend([slide_id] * len(coords))
        all_patch_groups.extend([group_name] * len(coords))

        per_slide_cache[slide_id] = {
            "slide_id": slide_id,
            "label": label,
            "group": group_name,
            "coords": coords,
            "frozen_feat": frozen_feat,
            "adapted_feat": adapted_feat,
            "cosine_shift": cosine_shift,
            "norm_shift": norm_shift,
            "frozen_scores": frozen_scores,
            "adapted_scores": adapted_scores,
            "tumor_gap_delta": tumor_gap_delta,
        }

        slide_df = pd.DataFrame({
            "slide_id": slide_id,
            "label": label,
            "group": group_name,
            "coord_x": coords[:, 0],
            "coord_y": coords[:, 1],
            "frozen_norm": frozen_norm,
            "adapted_norm": adapted_norm,
            "norm_shift": norm_shift,
            "feature_cosine_shift": cosine_shift,
        })
        if tumor_gap_delta is not None:
            slide_df["frozen_tumor_gap"] = frozen_scores["tumor_gap"]
            slide_df["adapted_tumor_gap"] = adapted_scores["tumor_gap"]
            slide_df["tumor_gap_delta"] = tumor_gap_delta
            slide_df["frozen_tumor_prob"] = frozen_scores["tumor_prob"]
            slide_df["adapted_tumor_prob"] = adapted_scores["tumor_prob"]
        slide_df.to_csv(Path(args.out_dir) / "per_slide" / f"{slide_id}_paired_patch_analysis.csv", index=False)

        if tumor_gap_delta is not None:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        sc0 = plot_spatial_map(
            axes[0], coords, cosine_shift,
            f"Spatial feature cosine shift\n{slide_id}",
            cmap="inferno",
        )
        sc1 = plot_spatial_map(
            axes[1], coords, norm_shift,
            f"Norm shift\n{slide_id}",
            cmap="coolwarm",
        )

        fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

        if tumor_gap_delta is not None:
            sc2 = plot_spatial_map(
                axes[2], coords, tumor_gap_delta,
                f"Tumor-gap delta\n{slide_id}",
                cmap="coolwarm",
            )
            fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(Path(args.out_dir) / "per_slide" / f"{slide_id}_encoder_shift_maps.png", dpi=220)
        plt.close(fig)

        slide = openslide.OpenSlide(slide_path)
        try:
            idx_changed = np.argsort(cosine_shift)[::-1][:args.gallery_topk]
            imgs = [
                read_patch_from_wsi(slide, (int(coords[i, 0]), int(coords[i, 1])), args.patch_size, 0)
                for i in idx_changed
            ]
            make_montage(imgs, tile_size=224, n_cols=4).save(
                Path(args.out_dir) / "galleries" / f"{slide_id}_top_changed_patch_gallery.png"
            )

            if tumor_gap_delta is not None:
                idx_improved = np.argsort(tumor_gap_delta)[::-1][:args.gallery_topk]
                imgs = [
                    read_patch_from_wsi(slide, (int(coords[i, 0]), int(coords[i, 1])), args.patch_size, 0)
                    for i in idx_improved
                ]
                make_montage(imgs, tile_size=224, n_cols=4).save(
                    Path(args.out_dir) / "galleries" / f"{slide_id}_improved_patch_gallery.png"
                )

                idx_worsened = np.argsort(tumor_gap_delta)[:args.gallery_topk]
                imgs = [
                    read_patch_from_wsi(slide, (int(coords[i, 0]), int(coords[i, 1])), args.patch_size, 0)
                    for i in idx_worsened
                ]
                make_montage(imgs, tile_size=224, n_cols=4).save(
                    Path(args.out_dir) / "galleries" / f"{slide_id}_worsened_patch_gallery.png"
                )
        finally:
            slide.close()

        per_slide_summary_rows.append({
            "slide_id": slide_id,
            "label": label,
            "group": group_name,
            "num_patches": int(len(coords)),
            "mean_feature_cosine_shift": float(np.mean(cosine_shift)),
            "median_feature_cosine_shift": float(np.median(cosine_shift)),
            "top10pct_mean_feature_cosine_shift": float(np.mean(np.sort(cosine_shift)[-max(1, len(cosine_shift)//10):])),
            "mean_norm_shift": float(np.mean(norm_shift)),
            "median_norm_shift": float(np.median(norm_shift)),
            "mean_tumor_gap_delta": float(np.mean(tumor_gap_delta)) if tumor_gap_delta is not None else np.nan,
            "median_tumor_gap_delta": float(np.median(tumor_gap_delta)) if tumor_gap_delta is not None else np.nan,
        })

    all_frozen_feats = np.concatenate(all_frozen_feats, axis=0)
    all_adapted_feats = np.concatenate(all_adapted_feats, axis=0)
    all_labels = np.asarray(all_patch_slide_labels, dtype=np.int64)
    all_slide_ids = np.asarray(all_patch_slide_ids)
    all_groups = np.asarray(all_patch_groups)

    frozen_purity = local_knn_purity(all_frozen_feats, all_labels, k=args.knn_k)
    adapted_purity = local_knn_purity(all_adapted_feats, all_labels, k=args.knn_k)

    knn_df = pd.DataFrame({
        "slide_id": all_slide_ids,
        "label": all_labels,
        "group": all_groups,
        "frozen_knn_purity": frozen_purity,
        "adapted_knn_purity": adapted_purity,
        "delta_knn_purity": adapted_purity - frozen_purity,
    })
    knn_df.to_csv(Path(args.out_dir) / "tables" / "patch_knn_purity.csv", index=False)

    knn_summary = knn_df.groupby(["group", "label"], as_index=False).agg({
        "frozen_knn_purity": "mean",
        "adapted_knn_purity": "mean",
        "delta_knn_purity": "mean",
    })
    knn_summary.to_csv(Path(args.out_dir) / "tables" / "patch_knn_purity_summary.csv", index=False)

    pca_dim = min(args.pca_dim, all_frozen_feats.shape[1], max(2, all_frozen_feats.shape[0] - 1))
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    frozen_pca = pca.fit_transform(l2_normalize_np(all_frozen_feats))
    adapted_pca = pca.transform(l2_normalize_np(all_adapted_feats))

    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    frozen_cluster = km.fit_predict(frozen_pca)
    adapted_cluster = km.predict(adapted_pca)

    trans_mat = np.zeros((args.n_clusters, args.n_clusters), dtype=np.int64)
    for fc, ac in zip(frozen_cluster, adapted_cluster):
        trans_mat[int(fc), int(ac)] += 1

    pd.DataFrame(trans_mat).to_csv(Path(args.out_dir) / "tables" / "cluster_transition_matrix.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = plot_transition_heatmap(ax, trans_mat, "Cluster transition matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "tables" / "cluster_transition_matrix.png", dpi=220)
    plt.close(fig)

    usage_rows = []
    for version, cid_arr in [("frozen", frozen_cluster), ("adapted", adapted_cluster)]:
        tmp = pd.DataFrame({
            "version": version,
            "label": all_labels,
            "cluster_id": cid_arr,
        })
        grp = tmp.groupby(["version", "label", "cluster_id"]).size().reset_index(name="count")
        grp["ratio"] = grp.groupby(["version", "label"])["count"].transform(lambda x: x / x.sum())
        usage_rows.append(grp)
    usage_df = pd.concat(usage_rows, axis=0).reset_index(drop=True)
    usage_df.to_csv(Path(args.out_dir) / "tables" / "cluster_usage_by_label.csv", index=False)

    if summary_builder is not None and proj_layer is not None:
        frozen_scores_all = compute_role_scores(
            all_frozen_feats, proj_layer, summary_builder, role_names,
            args.proto_tumor_name, args.proto_negative_role_names, args.device
        )
        adapted_scores_all = compute_role_scores(
            all_adapted_feats, proj_layer, summary_builder, role_names,
            args.proto_tumor_name, args.proto_negative_role_names, args.device
        )

        rows = []
        for version, cid_arr, sc in [
            ("frozen", frozen_cluster, frozen_scores_all),
            ("adapted", adapted_cluster, adapted_scores_all),
        ]:
            for cid in range(args.n_clusters):
                m = (cid_arr == cid)
                if m.sum() == 0:
                    continue
                rows.append({
                    "version": version,
                    "cluster_id": cid,
                    "count": int(m.sum()),
                    "mean_tumor_gap": float(np.mean(sc["tumor_gap"][m])),
                    "mean_tumor_prob": float(np.mean(sc["tumor_prob"][m])),
                    "std_tumor_gap": float(np.std(sc["tumor_gap"][m])),
                })
        cluster_margin_df = pd.DataFrame(rows)
        cluster_margin_df.to_csv(Path(args.out_dir) / "tables" / "cluster_margin_summary.csv", index=False)

        align_rows = []
        for version, sc in [("frozen", frozen_scores_all), ("adapted", adapted_scores_all)]:
            align_rows.append({
                "version": version,
                "mean_tumor_gap": float(np.mean(sc["tumor_gap"])),
                "median_tumor_gap": float(np.median(sc["tumor_gap"])),
                "mean_tumor_prob": float(np.mean(sc["tumor_prob"])),
                "median_tumor_prob": float(np.median(sc["tumor_prob"])),
            })
        cluster_align_df = pd.DataFrame(align_rows)
        cluster_align_df.to_csv(Path(args.out_dir) / "tables" / "cluster_proto_alignment.csv", index=False)
    else:
        pd.DataFrame([]).to_csv(Path(args.out_dir) / "tables" / "cluster_margin_summary.csv", index=False)
        pd.DataFrame([]).to_csv(Path(args.out_dir) / "tables" / "cluster_proto_alignment.csv", index=False)

    pd.DataFrame(per_slide_summary_rows).to_csv(
        Path(args.out_dir) / "tables" / "per_slide_encoder_summary.csv", index=False
    )

    with open(Path(args.out_dir) / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_case_groups": CASE_GROUPS,
            "feature_mode": args.feature_mode,
            "n_selected_slides": int(len(use_df)),
            "n_total_patches": int(len(all_labels)),
            "note": (
                "Online encoder-level evidence analysis on paired coords. "
                "Features are re-extracted on the same patch coordinates for frozen and adapted encoders. "
                "Main outputs include per-slide feature-shift maps, tumor-gap delta maps, patch galleries, "
                "cluster transition, cluster usage by label, cluster margin summary, and proto alignment."
            ),
        }, f, indent=2, ensure_ascii=False)

    print(f"[Done] saved to: {args.out_dir}")


if __name__ == "__main__":
    main()