#!/usr/bin/env python3
"""
MoE adapter behavior analysis, token-first version.

Main changes vs the previous patch-level script:
1. Keeps patch-level summaries, but adds token-level routing/feature analysis.
2. Token-level expert x cluster heatmaps are computed on sampled ViT image tokens.
3. Patch-level per-slide spatial figures are removed by default.
4. Adds optional frozen UNI comparison at the corresponding target block and final layer.
5. Adds per-slide npz cache so feature extraction does not need to be repeated.

Recommended use for UNI-MoE:

OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
python analysis/analyze_moe_adapter_token_behavior.py \
  --slides_csv ../data/BRACS/bracs_split.csv \
  --raw_dir /path/to/BRACS/raw \
  --h5_dir /path/to/BRACS/h5 \
  --out_dir outputs/moe_token_behavior_uni \
  --adapted_encoder_name uni_moe \
  --uni_weight /path/to/uni_weight.pt \
  --stage2_ckpt /path/to/stage2.pt \
  --target_block_1 21 --target_block_2 22 \
  --source_stage2_layer_1 9 --source_stage2_layer_2 10 \
  --n_slides 40 \
  --max_patches_per_slide 1024 \
  --token_sample_per_slide 4096 \
  --reducer umap \
  --compare_frozen_uni
"""
from __future__ import annotations

import os
# Must be set before numpy / sklearn / umap import.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from sklearn.metrics import silhouette_score
from tqdm import tqdm

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.backbone_moe_factory import build_feature_extractor


# =========================================================
# Basic utilities
# =========================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_slide_seed(base_seed: int, slide_id: str) -> int:
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(base_seed) + (int(h[:8], 16) % 100000)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def cosine_distance_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return 1.0 - np.sum(a * b, axis=1)


def entropy_np(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def top1_top2_margin_np(p: np.ndarray) -> np.ndarray:
    part = np.partition(p, kth=-2, axis=1)
    return part[:, -1] - part[:, -2]


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


# =========================================================
# Paths / CSV
# =========================================================
def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir_p = Path(raw_dir)
    exts = [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"]
    exact = []
    for ext in exts:
        exact.extend(raw_dir_p.rglob(f"{slide_id}{ext}"))
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise RuntimeError(f"Multiple exact WSI files for {slide_id}: {exact[:10]}")
    fuzzy = []
    for ext in exts:
        fuzzy.extend(raw_dir_p.rglob(f"{slide_id}*{ext}"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(f"Multiple fuzzy WSI files for {slide_id}: {fuzzy[:10]}")
    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir_p = Path(h5_dir)
    exact = list(h5_dir_p.rglob(f"{slide_id}.h5"))
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise RuntimeError(f"Multiple exact H5 files for {slide_id}: {exact[:10]}")
    fuzzy = list(h5_dir_p.rglob(f"{slide_id}*.h5"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(f"Multiple fuzzy H5 files for {slide_id}: {fuzzy[:10]}")
    raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")


def read_coords_from_h5(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        return f["coords"][:].astype(np.int64)


def load_pred_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip() == "":
        return None
    df = pd.read_csv(path)
    required = {"slide_id", "y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prediction csv missing columns: {missing}")
    return df.copy()


def prepare_slides_df(slides_csv: str, pred_csv: Optional[str], select_mode: str, n_slides: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(slides_csv)
    if "slide_id" not in df.columns and "image_id" in df.columns:
        df["slide_id"] = df["image_id"]
    if "slide_id" not in df.columns:
        raise ValueError("slides_csv must contain slide_id or image_id")
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        elif "y_true" in df.columns:
            df["label"] = df["y_true"]
        else:
            raise ValueError("slides_csv missing label / slide_binary_label / y_true")

    pred = load_pred_csv(pred_csv) if pred_csv else None
    if pred is not None:
        df = df.merge(pred[["slide_id", "y_prob"]].rename(columns={"y_prob": "adapted_prob"}), on="slide_id", how="left")
        df["pred_conf"] = np.abs(df["adapted_prob"].fillna(0.5) - 0.5)
    else:
        df["adapted_prob"] = np.nan
        df["pred_conf"] = 0.0

    rng = np.random.default_rng(seed)
    df = df.drop_duplicates("slide_id").reset_index(drop=True)

    if select_mode == "csv_order":
        out = df.head(n_slides).copy()
    elif select_mode in ["random_balanced", "high_conf_balanced"]:
        selected = []
        labels = sorted(df["label"].dropna().unique().tolist())
        per_label = max(1, math.ceil(n_slides / max(1, len(labels))))
        for lab in labels:
            sub = df[df["label"] == lab]
            if len(sub) == 0:
                continue
            if select_mode == "high_conf_balanced":
                sub = sub.sort_values("pred_conf", ascending=False)
                selected.append(sub.head(min(per_label, len(sub))))
            else:
                selected.append(sub.sample(n=min(per_label, len(sub)), random_state=int(rng.integers(1, 1_000_000))))
        out = pd.concat(selected, axis=0).drop_duplicates("slide_id")
        if len(out) < n_slides:
            remain = df[~df["slide_id"].isin(out["slide_id"])]
            if len(remain) > 0:
                out = pd.concat([out, remain.sample(n=min(n_slides - len(out), len(remain)), random_state=seed)], axis=0)
        out = out.sample(frac=1.0, random_state=seed).head(n_slides).reset_index(drop=True)
    elif select_mode == "hard_cases":
        out = df.sort_values("pred_conf", ascending=True).head(n_slides).copy()
    else:
        raise ValueError(f"Unknown select_mode: {select_mode}")
    return out


# =========================================================
# Image helpers
# =========================================================
def read_patch_from_wsi(slide: openslide.OpenSlide, coord_xy: Tuple[int, int], patch_size: int = 256, read_level: int = 0) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    return slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")


def make_montage(pil_images: List[Image.Image], tile_size: int = 224, n_cols: int = 4, caption_h: int = 26) -> Image.Image:
    n = len(pil_images)
    n_rows = math.ceil(n / n_cols)
    canvas = Image.new("RGB", (n_cols * tile_size, n_rows * (tile_size + caption_h)), color=(255, 255, 255))
    for i, img in enumerate(pil_images):
        r = i // n_cols
        c = i % n_cols
        x0 = c * tile_size
        y0 = r * (tile_size + caption_h)
        img = ImageOps.fit(img, (tile_size, tile_size), method=Image.BICUBIC)
        canvas.paste(img, (x0, y0))
        canvas.paste(Image.new("RGB", (tile_size, caption_h), color=(245, 245, 245)), (x0, y0 + tile_size))
    return canvas


# =========================================================
# Plotting / clustering
# =========================================================
def fit_joint_reducer(X: np.ndarray, reducer_type: str, seed: int, n_neighbors: int = 30, min_dist: float = 0.1):
    Xn = l2_normalize_np(X.astype(np.float32))
    if reducer_type == "umap" and HAS_UMAP:
        return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed).fit(Xn)
    return PCA(n_components=2, random_state=seed).fit(Xn)


def transform_reducer(reducer: Any, X: np.ndarray) -> np.ndarray:
    return reducer.transform(l2_normalize_np(X.astype(np.float32)))


def cluster_features(X: np.ndarray, n_clusters: int, seed: int, pca_dim: int = 32) -> np.ndarray:
    Xn = l2_normalize_np(X.astype(np.float32))
    dim = min(pca_dim, Xn.shape[1], max(2, Xn.shape[0] - 1))
    Xp = PCA(n_components=dim, random_state=seed).fit_transform(Xn)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(Xp).astype(np.int64)


def plot_scatter_by_category(ax, emb: np.ndarray, values: np.ndarray, title: str, prefix: str = "") -> None:
    uniq = sorted(pd.Series(values).dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    for i, val in enumerate(uniq):
        m = values == val
        ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.65, color=cmap(i % 20), label=f"{prefix}{val}")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(uniq) <= 12:
        ax.legend(markerscale=2, fontsize=7, frameon=False, loc="best")


def plot_scatter_continuous(ax, emb: np.ndarray, values: np.ndarray, title: str, cmap: str = "viridis"):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=5, alpha=0.70, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def save_heatmap(mat: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str, xticklabels: Sequence[str], yticklabels: Sequence[str], cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(xticklabels) * 0.55), max(4, len(yticklabels) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def expert_cluster_matrices(expert_probs: np.ndarray, clusters: np.ndarray, n_clusters: int) -> Dict[str, np.ndarray]:
    n_experts = expert_probs.shape[1]
    mean_weight = np.zeros((n_experts, n_clusters), dtype=np.float32)
    mass = np.zeros((n_experts, n_clusters), dtype=np.float32)
    hard_counts = np.zeros((n_experts, n_clusters), dtype=np.float32)
    counts = np.zeros((n_clusters,), dtype=np.float32)
    hard_ids = np.argmax(expert_probs, axis=1)
    for c in range(n_clusters):
        m = clusters == c
        counts[c] = float(m.sum())
        if m.sum() > 0:
            mean_weight[:, c] = expert_probs[m].mean(axis=0)
            mass[:, c] = expert_probs[m].sum(axis=0)
            for e in range(n_experts):
                hard_counts[e, c] = float(np.sum(hard_ids[m] == e))
    return {
        "mean_weight": mean_weight,
        "mean_weight_minus_uniform": mean_weight - (1.0 / n_experts),
        "p_cluster_given_expert": mass / np.clip(mass.sum(axis=1, keepdims=True), 1e-8, None),
        "p_expert_given_cluster": mass / np.clip(mass.sum(axis=0, keepdims=True), 1e-8, None),
        # Hard-assignment versions, closer to your old DINO visualization script.
        "hard_p_cluster_given_expert": hard_counts / np.clip(hard_counts.sum(axis=1, keepdims=True), 1e-8, None),
        "hard_p_expert_given_cluster": hard_counts / np.clip(hard_counts.sum(axis=0, keepdims=True), 1e-8, None),
        "hard_counts": hard_counts,
        "cluster_counts": counts,
    }


def save_moe_scatter_panels(out_dir: Path, df: pd.DataFrame, layer_name: str, prefix: str = "global") -> None:
    emb = df[[f"{layer_name}_x", f"{layer_name}_y"]].values.astype(np.float32)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    plot_scatter_by_category(axes[0, 0], emb, df[f"{layer_name}_cluster"].values, f"{layer_name}: cluster", prefix="c")
    plot_scatter_by_category(axes[0, 1], emb, df["dominant_expert"].values, f"{layer_name}: dominant expert", prefix="E")
    plot_scatter_by_category(axes[1, 0], emb, df["label"].values, f"{layer_name}: label", prefix="y=")
    sc = plot_scatter_continuous(axes[1, 1], emb, df["gate_entropy"].values, f"{layer_name}: gate entropy", cmap="viridis")
    fig.colorbar(sc, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_{layer_name}_scatter_panels.png", dpi=240)
    plt.close(fig)


def save_frozen_scatter_panels(out_dir: Path, df: pd.DataFrame, layer_name: str) -> None:
    emb = df[[f"{layer_name}_x", f"{layer_name}_y"]].values.astype(np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    plot_scatter_by_category(axes[0], emb, df[f"{layer_name}_cluster"].values, f"frozen {layer_name}: cluster", prefix="c")
    plot_scatter_by_category(axes[1], emb, df["label"].values, f"frozen {layer_name}: label", prefix="y=")
    fig.tight_layout()
    fig.savefig(out_dir / f"frozen_{layer_name}_scatter_panels.png", dpi=240)
    plt.close(fig)


def save_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 60) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# =========================================================
# Build extractors
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
        virchow2_weight=args.virchow2_weight,
        uni_weight=args.uni_weight,
        uni2_weight=getattr(args, "uni2_weight", ""),
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
        stage2_ckpt=args.stage2_ckpt,
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


def build_adapted_args(args) -> SimpleNamespace:
    return build_base_namespace(args, args.adapted_encoder_name)


def infer_frozen_encoder_name(adapted_name: str) -> str:
    if adapted_name == "uni_moe":
        return "uni"
    if adapted_name == "virchow2_moe":
        return "virchow2"
    raise ValueError(f"Cannot infer frozen encoder for {adapted_name}")


# =========================================================
# MoE analysis extraction
# =========================================================
def _pool_tokens_to_feature(tokens: torch.Tensor, reg_tokens: int) -> torch.Tensor:
    cls = tokens[:, 0, :]
    patch_start = 1 + int(reg_tokens)
    patch_mean = tokens[:, patch_start:, :].mean(dim=1)
    return torch.cat([cls, patch_mean], dim=-1)


def _find_tensor_with_last_dim(obj: Any, n_experts: int, prefer_dispatch: bool = True) -> Optional[torch.Tensor]:
    """
    Recursively find expert-routing tensor.

    Important: the older DINO-MoE visualization used gate_info['dispatch_weight']
    for hard token expert assignment. Prefer that key here as well; otherwise we
    may accidentally pick soft gate probabilities/logits, which often look nearly
    uniform and make specialization appear weaker than the dispatch actually is.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if obj.ndim >= 2 and obj.shape[-1] == n_experts:
            return obj
        return None
    if isinstance(obj, dict):
        if prefer_dispatch:
            preferred = [
                "dispatch_weight", "dispatch_weights", "dispatch_mask",
                "combine_weights", "routing_weights", "gate_weights", "weights",
                "expert_probs", "probs", "gate_probs", "routing_probs",
                "logits", "gate_logits", "router_logits",
            ]
        else:
            preferred = [
                "expert_probs", "probs", "gate_probs", "routing_probs",
                "dispatch_weight", "dispatch_weights", "dispatch_mask",
                "combine_weights", "routing_weights", "gate_weights", "weights",
                "logits", "gate_logits", "router_logits",
            ]
        for k in preferred:
            if k in obj:
                found = _find_tensor_with_last_dim(obj[k], n_experts, prefer_dispatch=prefer_dispatch)
                if found is not None:
                    return found
        for v in obj.values():
            found = _find_tensor_with_last_dim(v, n_experts, prefer_dispatch=prefer_dispatch)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_tensor_with_last_dim(v, n_experts, prefer_dispatch=prefer_dispatch)
            if found is not None:
                return found
    return None


def _normalize_probs_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    row_sum = x.sum(dim=-1, keepdim=True)
    looks_like_prob = bool(torch.all(x >= -1e-6) and torch.all(x <= 1.0 + 1e-6) and torch.allclose(row_sum.mean(), torch.ones_like(row_sum.mean()), atol=0.15))
    if looks_like_prob:
        x = x.clamp_min(0.0)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.softmax(x, dim=-1)


def _gate_info_to_token_probs(gate_info: Any, n_experts: int, B: int, n_tokens: int) -> torch.Tensor:
    gate_tensor = _find_tensor_with_last_dim(gate_info, n_experts, prefer_dispatch=True)
    if gate_tensor is None:
        raise RuntimeError("Could not parse expert probabilities from gate_info. Need a tensor with last dim == num_experts.")
    gate_tensor = gate_tensor.detach()
    if gate_tensor.ndim == 3 and gate_tensor.shape[0] == B:
        tok = gate_tensor
    elif gate_tensor.ndim == 2 and gate_tensor.shape[0] == B * n_tokens:
        tok = gate_tensor.reshape(B, n_tokens, n_experts)
    elif gate_tensor.ndim == 2 and gate_tensor.shape[0] == B:
        tok = gate_tensor[:, None, :].expand(B, n_tokens, n_experts)
    elif gate_tensor.ndim > 3 and gate_tensor.shape[0] == B:
        # Example [B, N, top_k, E] -> average top_k/routing dimension.
        dims_to_mean = tuple(range(2, gate_tensor.ndim - 1))
        tok = gate_tensor.mean(dim=dims_to_mean)
    else:
        raise RuntimeError(f"Cannot map gate tensor shape={tuple(gate_tensor.shape)} to [B,N,E] with B={B}, N={n_tokens}, E={n_experts}")
    return _normalize_probs_torch(tok)


@torch.no_grad()
def call_moe_analysis(extractor: Any, images: List[Image.Image], analysis_method: str) -> Dict[str, np.ndarray]:
    if hasattr(extractor, analysis_method):
        out = getattr(extractor, analysis_method)(images)
        return {k: to_numpy(v) for k, v in out.items() if v is not None}

    model = extractor.model
    reg_tokens = int(getattr(extractor, "reg_tokens", getattr(model, "reg_tokens", 0)))
    n_experts = int(getattr(model, "num_experts", 0))
    if n_experts <= 0:
        moe_blocks = sorted(list(getattr(model, "moe_layer_map", {}).keys()))
        if not moe_blocks:
            raise RuntimeError("model.moe_layer_map is empty; cannot locate MoE blocks.")
        last_blk = model.blocks[moe_blocks[-1]]
        if hasattr(last_blk.mlp, "moe") and hasattr(last_blk.mlp.moe, "num_experts"):
            n_experts = int(last_blk.mlp.moe.num_experts)
        elif hasattr(last_blk.mlp, "num_experts"):
            n_experts = int(last_blk.mlp.num_experts)
        else:
            raise RuntimeError("Cannot infer num_experts.")

    model_out = model(images, return_gates=True, is_eval=True, return_features=True)
    if not (isinstance(model_out, (list, tuple)) and len(model_out) == 4):
        raise RuntimeError("Expected model(..., return_gates=True, return_features=True) -> (x_out, gate_info_list, feature_dict, moe_feature_list)")
    x_out, gate_info_list, feature_dict, moe_feature_list = model_out
    if len(moe_feature_list) == 0 or len(gate_info_list) == 0:
        raise RuntimeError("Empty moe_feature_list or gate_info_list.")

    last_moe_tokens = moe_feature_list[-1]
    final_tokens = x_out
    B, n_tokens, _ = last_moe_tokens.shape
    token_probs_all = _gate_info_to_token_probs(gate_info_list[-1], n_experts=n_experts, B=B, n_tokens=n_tokens)

    token_start = 1 + reg_tokens
    image_token_probs = token_probs_all[:, token_start:, :]
    patch_expert_probs = image_token_probs.mean(dim=1)
    patch_expert_ids = torch.argmax(patch_expert_probs, dim=-1)
    patch_gate_entropy = -(patch_expert_probs.clamp_min(1e-8) * torch.log(patch_expert_probs.clamp_min(1e-8))).sum(dim=-1)

    return {
        "last_moe_feat": to_numpy(_pool_tokens_to_feature(last_moe_tokens, reg_tokens)),
        "final_feat": to_numpy(_pool_tokens_to_feature(final_tokens, reg_tokens)),
        "expert_probs": to_numpy(patch_expert_probs),
        "expert_ids": to_numpy(patch_expert_ids),
        "gate_entropy": to_numpy(patch_gate_entropy),
        "last_moe_tokens": to_numpy(last_moe_tokens[:, token_start:, :]),
        "final_tokens": to_numpy(final_tokens[:, token_start:, :]),
        "token_expert_probs": to_numpy(image_token_probs),
    }


@torch.no_grad()
def extract_slide_analysis(
    extractor: Any,
    slide_path: str,
    h5_path: str,
    patch_size: int,
    batch_size: int,
    max_patches: Optional[int],
    seed: int,
    analysis_method: str,
    token_sample_per_slide: int,
) -> Dict[str, np.ndarray]:
    coords = read_coords_from_h5(h5_path)
    if max_patches is not None and len(coords) > max_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_patches, replace=False)
        coords = coords[idx]

    rng = np.random.default_rng(seed + 17)
    slide = openslide.OpenSlide(slide_path)
    patch_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
    token_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
    patch_buffers["coords"].append(coords.astype(np.int64))

    total_patches = len(coords)
    sample_per_patch = token_sample_per_slide / max(1, total_patches)

    try:
        for start in tqdm(range(0, len(coords), batch_size), total=math.ceil(len(coords) / batch_size), desc=f"  Patches[{Path(slide_path).stem[:24]}]", leave=False):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]
            images = []
            for xy in batch_coords.tolist():
                img = read_patch_from_wsi(slide, (int(xy[0]), int(xy[1])), patch_size=patch_size, read_level=0)
                images.append(img.resize((224, 224), resample=Image.BICUBIC))

            out = call_moe_analysis(extractor, images, analysis_method)
            for k in ["last_moe_feat", "final_feat", "expert_probs", "expert_ids", "gate_entropy"]:
                patch_buffers[k].append(out[k])

            # Token-level sampled analysis.
            last_tok = out["last_moe_tokens"]          # [B, T, D]
            final_tok = out["final_tokens"]            # [B, T, D]
            tok_probs = out["token_expert_probs"]      # [B, T, E]
            B, T, D = last_tok.shape
            n_flat = B * T
            want = int(math.ceil(B * sample_per_patch))
            want = max(1, min(want, n_flat))
            flat_idx = rng.choice(n_flat, size=want, replace=False)
            local_patch = flat_idx // T
            token_idx = flat_idx % T

            token_buffers["token_last_moe_feat"].append(last_tok.reshape(n_flat, D)[flat_idx].astype(np.float32))
            token_buffers["token_final_feat"].append(final_tok.reshape(n_flat, D)[flat_idx].astype(np.float32))
            token_buffers["token_expert_probs"].append(tok_probs.reshape(n_flat, tok_probs.shape[-1])[flat_idx].astype(np.float32))
            token_buffers["token_coord_x"].append(batch_coords[local_patch, 0].astype(np.int64))
            token_buffers["token_coord_y"].append(batch_coords[local_patch, 1].astype(np.int64))
            token_buffers["token_patch_local_index"].append((start + local_patch).astype(np.int64))
            token_buffers["token_index_in_patch"].append(token_idx.astype(np.int64))
    finally:
        slide.close()

    merged: Dict[str, np.ndarray] = {}
    for k, values in patch_buffers.items():
        merged[k] = np.concatenate(values, axis=0)
    for k, values in token_buffers.items():
        arr = np.concatenate(values, axis=0)
        if len(arr) > token_sample_per_slide:
            keep = rng.choice(len(arr), size=token_sample_per_slide, replace=False)
            arr = arr[keep]
        merged[k] = arr
    return merged


@torch.no_grad()
def extract_frozen_features_for_coords(
    frozen_extractor: Any,
    slide_path: str,
    coords: np.ndarray,
    patch_size: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    slide = openslide.OpenSlide(slide_path)
    last_list, final_list = [], []
    try:
        for start in tqdm(range(0, len(coords), batch_size), total=math.ceil(len(coords) / batch_size), desc=f"  Frozen[{Path(slide_path).stem[:24]}]", leave=False):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]
            images = []
            for xy in batch_coords.tolist():
                img = read_patch_from_wsi(slide, (int(xy[0]), int(xy[1])), patch_size=patch_size, read_level=0)
                images.append(img.resize((224, 224), resample=Image.BICUBIC))
            last_list.append(to_numpy(frozen_extractor.extract_features(images, feature_mode="moe_last")).astype(np.float32))
            final_list.append(to_numpy(frozen_extractor.extract_features(images, feature_mode="final")).astype(np.float32))
    finally:
        slide.close()
    return {
        "frozen_last_moe_feat": np.concatenate(last_list, axis=0),
        "frozen_final_feat": np.concatenate(final_list, axis=0),
    }


# =========================================================
# Summaries / galleries
# =========================================================
def summarize_per_slide(global_df: pd.DataFrame, n_experts: int, n_clusters: int) -> pd.DataFrame:
    rows = []
    for slide_id, sub in global_df.groupby("slide_id"):
        row: Dict[str, Any] = {
            "slide_id": slide_id,
            "label": int(sub["label"].iloc[0]),
            "num_patches": int(len(sub)),
            "mean_gate_entropy": float(sub["gate_entropy"].mean()),
            "mean_margin": float(sub["expert_margin"].mean()),
            "mean_last_to_final_delta": float(sub["last_to_final_delta"].mean()),
        }
        if "adapted_prob" in sub.columns:
            row["adapted_prob"] = safe_float(sub["adapted_prob"].iloc[0])
        for e in range(n_experts):
            row[f"expert_{e}_mean_prob"] = float(sub[f"expert_prob_{e}"].mean())
            row[f"expert_{e}_dominant_frac"] = float((sub["dominant_expert"] == e).mean())
        for c in range(n_clusters):
            row[f"last_cluster_{c}_frac"] = float((sub["last_moe_cluster"] == c).mean())
            row[f"final_cluster_{c}_frac"] = float((sub["final_cluster"] == c).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_patch_galleries(out_dir: Path, global_df: pd.DataFrame, raw_dir: str, patch_size: int, n_experts: int, topk: int) -> None:
    ensure_dir(out_dir)
    rows = []
    chosen: Dict[str, List[int]] = defaultdict(list)
    for e in range(n_experts):
        col = f"expert_prob_{e}"
        for idx in global_df.sort_values(col, ascending=False).head(topk).index.tolist():
            chosen[str(global_df.loc[idx, "slide_id"])].append(int(idx))
    cache: Dict[int, Image.Image] = {}
    for slide_id, indices in chosen.items():
        slide = openslide.OpenSlide(find_wsi_path(raw_dir, slide_id))
        try:
            for idx in indices:
                r = global_df.loc[idx]
                cache[idx] = read_patch_from_wsi(slide, (int(r["coord_x"]), int(r["coord_y"])), patch_size=patch_size)
        finally:
            slide.close()
    for e in range(n_experts):
        col = f"expert_prob_{e}"
        top = global_df.sort_values(col, ascending=False).head(topk)
        imgs = []
        for idx, r in top.iterrows():
            if idx in cache:
                imgs.append(cache[idx])
                rows.append({"gallery": f"expert_{e}", "slide_id": r["slide_id"], "coord_x": r["coord_x"], "coord_y": r["coord_y"], "expert": e, "expert_prob": r[col], "margin": r["expert_margin"], "gate_entropy": r["gate_entropy"]})
        if imgs:
            make_montage(imgs, tile_size=224, n_cols=4).save(out_dir / f"top_patches_expert_{e}.png")
    pd.DataFrame(rows).to_csv(out_dir / "gallery_patch_index.csv", index=False)


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser("Token-level MoE adapter behavior analysis")
    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--adapted_pred_csv", type=str, default="")

    parser.add_argument("--adapted_encoder_name", type=str, default="uni_moe", choices=["virchow2_moe", "uni_moe"])
    parser.add_argument("--virchow2_weight", type=str, default="")
    parser.add_argument("--uni_weight", type=str, default="")
    parser.add_argument("--uni2_weight", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_patches_per_slide", type=int, default=1024)
    parser.add_argument("--token_sample_per_slide", type=int, default=4096)
    parser.add_argument("--n_slides", type=int, default=40)
    parser.add_argument("--select_mode", type=str, default="random_balanced", choices=["csv_order", "random_balanced", "high_conf_balanced", "hard_cases"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--analysis_method", type=str, default="extract_moe_analysis")
    parser.add_argument("--reducer", type=str, default="umap", choices=["umap", "pca"])
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--token_n_clusters", type=int, default=12)
    parser.add_argument("--gallery_topk", type=int, default=24)
    parser.add_argument("--save_galleries", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--compare_frozen_uni", action="store_true", help="Also extract frozen UNI corresponding target block and final features. For virchow2_moe, this compares frozen Virchow2.")

    # MoE bridge args.
    parser.add_argument("--target_block_1", type=int, default=21)
    parser.add_argument("--target_block_2", type=int, default=22)
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

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "per_slide_npz")
    ensure_dir(out_dir / "heatmaps")
    ensure_dir(out_dir / "token_analysis")
    ensure_dir(out_dir / "frozen_compare")
    ensure_dir(out_dir / "galleries")

    selected_df = prepare_slides_df(args.slides_csv, args.adapted_pred_csv, args.select_mode, args.n_slides, args.seed)
    selected_df.to_csv(out_dir / "selected_slides.csv", index=False)
    print(f"[Select] {len(selected_df)} slides selected by mode={args.select_mode}")
    print("[Select] label counts:")
    print(selected_df["label"].value_counts().sort_index())

    print("[Build] adapted extractor ...")
    extractor = build_feature_extractor(build_adapted_args(args))

    frozen_extractor = None
    if args.compare_frozen_uni:
        frozen_name = infer_frozen_encoder_name(args.adapted_encoder_name)
        print(f"[Build] frozen extractor for comparison: {frozen_name}")
        frozen_extractor = build_feature_extractor(build_base_namespace(args, frozen_name))

    all_rows: List[pd.DataFrame] = []
    last_feats: List[np.ndarray] = []
    final_feats: List[np.ndarray] = []
    expert_probs_all: List[np.ndarray] = []
    token_rows: List[pd.DataFrame] = []
    token_last_feats: List[np.ndarray] = []
    token_final_feats: List[np.ndarray] = []
    token_probs_all: List[np.ndarray] = []
    frozen_rows: List[pd.DataFrame] = []
    frozen_last_feats: List[np.ndarray] = []
    frozen_final_feats: List[np.ndarray] = []

    for _, row in selected_df.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        adapted_prob = safe_float(row.get("adapted_prob", None))
        print(f"[Process] slide={slide_id} | label={label}")
        slide_path = find_wsi_path(args.raw_dir, slide_id)
        h5_path = find_h5_path(args.h5_dir, slide_id)
        cache_path = out_dir / "per_slide_npz" / f"{slide_id}_moe_token_cache.npz"
        use_cache = (not args.no_cache) and cache_path.exists() and (not args.overwrite_cache)

        if use_cache:
            print(f"[Cache] load {cache_path}")
            c = np.load(cache_path, allow_pickle=True)
            out = {k: c[k] for k in c.files}
        else:
            out = extract_slide_analysis(
                extractor=extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                max_patches=args.max_patches_per_slide,
                seed=stable_slide_seed(args.seed, slide_id),
                analysis_method=args.analysis_method,
                token_sample_per_slide=args.token_sample_per_slide,
            )

        coords = out["coords"]
        last_feat = out["last_moe_feat"].astype(np.float32)
        final_feat = out["final_feat"].astype(np.float32)
        expert_probs = out["expert_probs"].astype(np.float32)
        expert_probs = np.clip(expert_probs, 0.0, None)
        expert_probs = expert_probs / np.clip(expert_probs.sum(axis=1, keepdims=True), 1e-8, None)
        expert_ids = np.argmax(expert_probs, axis=1).astype(np.int64)
        gate_entropy = entropy_np(expert_probs).astype(np.float32)
        expert_margin = top1_top2_margin_np(expert_probs).astype(np.float32)
        last_to_final_delta = cosine_distance_rows(last_feat, final_feat).astype(np.float32)
        n_patch = len(coords)
        n_experts = expert_probs.shape[1]

        patch_df = pd.DataFrame({
            "slide_id": [slide_id] * n_patch,
            "label": [label] * n_patch,
            "adapted_prob": [adapted_prob] * n_patch,
            "coord_x": coords[:, 0],
            "coord_y": coords[:, 1],
            "dominant_expert": expert_ids,
            "gate_entropy": gate_entropy,
            "expert_margin": expert_margin,
            "last_to_final_delta": last_to_final_delta,
        })
        for e in range(n_experts):
            patch_df[f"expert_prob_{e}"] = expert_probs[:, e]
        all_rows.append(patch_df)
        last_feats.append(last_feat)
        final_feats.append(final_feat)
        expert_probs_all.append(expert_probs)

        tok_probs = out["token_expert_probs"].astype(np.float32)
        tok_probs = np.clip(tok_probs, 0.0, None)
        tok_probs = tok_probs / np.clip(tok_probs.sum(axis=1, keepdims=True), 1e-8, None)
        tok_entropy = entropy_np(tok_probs).astype(np.float32)
        tok_margin = top1_top2_margin_np(tok_probs).astype(np.float32)
        tok_ids = np.argmax(tok_probs, axis=1).astype(np.int64)
        n_tok = len(tok_probs)
        tok_df = pd.DataFrame({
            "slide_id": [slide_id] * n_tok,
            "label": [label] * n_tok,
            "coord_x": out["token_coord_x"].astype(np.int64),
            "coord_y": out["token_coord_y"].astype(np.int64),
            "token_index_in_patch": out["token_index_in_patch"].astype(np.int64),
            "token_patch_local_index": out["token_patch_local_index"].astype(np.int64),
            "dominant_expert": tok_ids,
            "gate_entropy": tok_entropy,
            "expert_margin": tok_margin,
        })
        for e in range(n_experts):
            tok_df[f"expert_prob_{e}"] = tok_probs[:, e]
        token_rows.append(tok_df)
        token_last_feats.append(out["token_last_moe_feat"].astype(np.float32))
        token_final_feats.append(out["token_final_feat"].astype(np.float32))
        token_probs_all.append(tok_probs)

        if frozen_extractor is not None:
            frozen_cache = out_dir / "per_slide_npz" / f"{slide_id}_frozen_cache.npz"
            if (not args.no_cache) and frozen_cache.exists() and (not args.overwrite_cache):
                fc = np.load(frozen_cache)
                f_last = fc["frozen_last_moe_feat"].astype(np.float32)
                f_final = fc["frozen_final_feat"].astype(np.float32)
            else:
                fout = extract_frozen_features_for_coords(frozen_extractor, slide_path, coords, args.patch_size, args.batch_size)
                f_last = fout["frozen_last_moe_feat"].astype(np.float32)
                f_final = fout["frozen_final_feat"].astype(np.float32)
                if not args.no_cache:
                    np.savez_compressed(frozen_cache, frozen_last_moe_feat=f_last, frozen_final_feat=f_final)
            frozen_rows.append(pd.DataFrame({"slide_id": [slide_id] * n_patch, "label": [label] * n_patch, "coord_x": coords[:, 0], "coord_y": coords[:, 1]}))
            frozen_last_feats.append(f_last)
            frozen_final_feats.append(f_final)

        if (not args.no_cache) and (not use_cache):
            np.savez_compressed(
                cache_path,
                coords=coords.astype(np.int64),
                last_moe_feat=last_feat,
                final_feat=final_feat,
                expert_probs=expert_probs,
                expert_ids=expert_ids,
                gate_entropy=gate_entropy,
                token_last_moe_feat=out["token_last_moe_feat"].astype(np.float32),
                token_final_feat=out["token_final_feat"].astype(np.float32),
                token_expert_probs=tok_probs,
                token_coord_x=out["token_coord_x"].astype(np.int64),
                token_coord_y=out["token_coord_y"].astype(np.int64),
                token_patch_local_index=out["token_patch_local_index"].astype(np.int64),
                token_index_in_patch=out["token_index_in_patch"].astype(np.int64),
            )
            print(f"[Cache] saved {cache_path}")

    # Patch-level global analysis.
    print("[Analyze] patch-level reducers/clusters ...")
    global_df = pd.concat(all_rows, axis=0).reset_index(drop=True)
    last_all = np.concatenate(last_feats, axis=0)
    final_all = np.concatenate(final_feats, axis=0)
    expert_probs = np.concatenate(expert_probs_all, axis=0)
    n_experts = expert_probs.shape[1]

    for layer_name, feat, seed_offset in [("last_moe", last_all, 0), ("final", final_all, 1)]:
        reducer = fit_joint_reducer(feat, args.reducer, args.seed + seed_offset, n_neighbors=30, min_dist=0.1)
        emb = transform_reducer(reducer, feat)
        cluster = cluster_features(feat, args.n_clusters, args.seed + seed_offset)
        global_df[f"{layer_name}_x"] = emb[:, 0]
        global_df[f"{layer_name}_y"] = emb[:, 1]
        global_df[f"{layer_name}_cluster"] = cluster
        save_moe_scatter_panels(out_dir, global_df, layer_name, prefix="global_patch")

        mats = expert_cluster_matrices(expert_probs, cluster, args.n_clusters)
        for mat_name, mat in mats.items():
            if mat_name in ["cluster_counts", "hard_counts"]:
                continue
            np.save(out_dir / "heatmaps" / f"patch_{layer_name}_{mat_name}.npy", mat)
            save_heatmap(
                mat,
                out_dir / "heatmaps" / f"patch_{layer_name}_{mat_name}.png",
                title=f"patch {layer_name}: {mat_name}",
                xlabel="cluster",
                ylabel="expert",
                xticklabels=[f"c{i}" for i in range(args.n_clusters)],
                yticklabels=[f"E{i}" for i in range(n_experts)],
                cmap="coolwarm" if mat_name == "mean_weight_minus_uniform" else ("magma" if mat_name == "mean_weight" else "Blues"),
            )
        pd.DataFrame({"cluster": [f"c{i}" for i in range(args.n_clusters)], "count": mats["cluster_counts"]}).to_csv(out_dir / "heatmaps" / f"patch_{layer_name}_cluster_counts.csv", index=False)
        pd.DataFrame(mats["hard_counts"], index=[f"E{i}" for i in range(n_experts)], columns=[f"c{i}" for i in range(args.n_clusters)]).to_csv(out_dir / "heatmaps" / f"patch_{layer_name}_hard_expert_cluster_counts.csv")
        pd.DataFrame(mats["hard_counts"], index=[f"E{i}" for i in range(n_experts)], columns=[f"c{i}" for i in range(args.n_clusters)]).to_csv(out_dir / "heatmaps" / f"patch_{layer_name}_hard_expert_cluster_counts.csv")

    global_df.to_csv(out_dir / "global_patch_moe_analysis.csv", index=False)
    summarize_per_slide(global_df, n_experts=n_experts, n_clusters=args.n_clusters).to_csv(out_dir / "slide_moe_behavior_summary.csv", index=False)
    save_hist(global_df["gate_entropy"].values, out_dir / "patch_gate_entropy_hist.png", "Patch-level gate entropy", "entropy")
    save_hist(global_df["expert_margin"].values, out_dir / "patch_expert_margin_hist.png", "Patch-level top1-top2 expert margin", "margin")

    # Token-level global analysis.
    print("[Analyze] token-level reducers/clusters ...")
    token_df = pd.concat(token_rows, axis=0).reset_index(drop=True)
    token_last_all = np.concatenate(token_last_feats, axis=0)
    token_final_all = np.concatenate(token_final_feats, axis=0)
    token_probs = np.concatenate(token_probs_all, axis=0)
    token_df.to_csv(out_dir / "token_analysis" / "global_token_moe_table.csv", index=False)

    for layer_name, feat, seed_offset in [("token_last_moe", token_last_all, 11), ("token_final", token_final_all, 12)]:
        reducer = fit_joint_reducer(feat, args.reducer, args.seed + seed_offset, n_neighbors=30, min_dist=0.1)
        emb = transform_reducer(reducer, feat)
        cluster = cluster_features(feat, args.token_n_clusters, args.seed + seed_offset)
        token_df[f"{layer_name}_x"] = emb[:, 0]
        token_df[f"{layer_name}_y"] = emb[:, 1]
        token_df[f"{layer_name}_cluster"] = cluster
        # Reuse the MoE panel function by passing the correct layer_name.
        save_moe_scatter_panels(out_dir / "token_analysis", token_df, layer_name, prefix="global")

        mats = expert_cluster_matrices(token_probs, cluster, args.token_n_clusters)
        for mat_name, mat in mats.items():
            if mat_name in ["cluster_counts", "hard_counts"]:
                continue
            np.save(out_dir / "token_analysis" / f"{layer_name}_{mat_name}.npy", mat)
            save_heatmap(
                mat,
                out_dir / "token_analysis" / f"{layer_name}_{mat_name}.png",
                title=f"{layer_name}: {mat_name}",
                xlabel="token cluster",
                ylabel="expert",
                xticklabels=[f"tc{i}" for i in range(args.token_n_clusters)],
                yticklabels=[f"E{i}" for i in range(n_experts)],
                cmap="coolwarm" if mat_name == "mean_weight_minus_uniform" else ("magma" if mat_name == "mean_weight" else "Blues"),
            )
        pd.DataFrame({"cluster": [f"tc{i}" for i in range(args.token_n_clusters)], "count": mats["cluster_counts"]}).to_csv(out_dir / "token_analysis" / f"{layer_name}_cluster_counts.csv", index=False)
        pd.DataFrame(mats["hard_counts"], index=[f"E{i}" for i in range(n_experts)], columns=[f"tc{i}" for i in range(args.token_n_clusters)]).to_csv(out_dir / "token_analysis" / f"{layer_name}_hard_expert_cluster_counts.csv")
        pd.DataFrame(mats["hard_counts"], index=[f"E{i}" for i in range(n_experts)], columns=[f"tc{i}" for i in range(args.token_n_clusters)]).to_csv(out_dir / "token_analysis" / f"{layer_name}_hard_expert_cluster_counts.csv")

    token_df.to_csv(out_dir / "token_analysis" / "global_token_moe_analysis.csv", index=False)
    save_hist(token_df["gate_entropy"].values, out_dir / "token_analysis" / "token_gate_entropy_hist.png", "Token-level gate entropy", "entropy")
    save_hist(token_df["expert_margin"].values, out_dir / "token_analysis" / "token_expert_margin_hist.png", "Token-level top1-top2 expert margin", "margin")

    # Frozen UNI / frozen Virchow comparison.
    if frozen_extractor is not None:
        print("[Analyze] frozen comparison reducers/clusters ...")
        fdf = pd.concat(frozen_rows, axis=0).reset_index(drop=True)
        f_last_all = np.concatenate(frozen_last_feats, axis=0)
        f_final_all = np.concatenate(frozen_final_feats, axis=0)
        for layer_name, feat, seed_offset in [("last_moe", f_last_all, 31), ("final", f_final_all, 32)]:
            reducer = fit_joint_reducer(feat, args.reducer, args.seed + seed_offset, n_neighbors=30, min_dist=0.1)
            emb = transform_reducer(reducer, feat)
            cluster = cluster_features(feat, args.n_clusters, args.seed + seed_offset)
            fdf[f"{layer_name}_x"] = emb[:, 0]
            fdf[f"{layer_name}_y"] = emb[:, 1]
            fdf[f"{layer_name}_cluster"] = cluster
            save_frozen_scatter_panels(out_dir / "frozen_compare", fdf, layer_name)
        fdf.to_csv(out_dir / "frozen_compare" / "global_frozen_feature_analysis.csv", index=False)

    if args.save_galleries:
        save_patch_galleries(out_dir / "galleries", global_df, args.raw_dir, args.patch_size, n_experts, args.gallery_topk)

    metrics: Dict[str, Any] = {
        "n_total_patches": int(len(global_df)),
        "n_total_sampled_tokens": int(len(token_df)),
        "n_experts_detected": int(n_experts),
        "patch_mean_gate_entropy": float(global_df["gate_entropy"].mean()),
        "token_mean_gate_entropy": float(token_df["gate_entropy"].mean()),
        "patch_mean_margin": float(global_df["expert_margin"].mean()),
        "token_mean_margin": float(token_df["expert_margin"].mean()),
        "has_umap": HAS_UMAP,
    }
    for name, feat, cluster_col in [
        ("patch_last_moe", last_all, "last_moe_cluster"),
        ("patch_final", final_all, "final_cluster"),
        ("token_last_moe", token_last_all, "token_last_moe_cluster"),
        ("token_final", token_final_all, "token_final_cluster"),
    ]:
        try:
            sample_n = min(5000, len(feat))
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(feat), size=sample_n, replace=False) if len(feat) > sample_n else np.arange(len(feat))
            src_df = token_df if name.startswith("token") else global_df
            metrics[f"{name}_silhouette_cosine"] = float(silhouette_score(l2_normalize_np(feat[idx]), src_df[cluster_col].values[idx], metric="cosine"))
        except Exception as e:
            metrics[f"{name}_silhouette_cosine_error"] = str(e)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    config = vars(args).copy()
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[Done] saved to: {out_dir}")


if __name__ == "__main__":
    main()
