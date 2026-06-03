#!/usr/bin/env python3
"""
Token-level visualization for DINOv2 / UNI / UNI2H / Virchow2 / OpenCLIP + MoE adapter,
with t-SNE comparison and patch-level expert-composition analysis.

This script keeps the original DINO-specific path intact, and adds a safer factory
backend for non-DINO MoE adapters built by models.encoders.backbone_moe_factory.

Backends
--------
1. --backend dino
   - Uses MoEEncoder + DINOv2Encoder directly.
   - Requires --config and --moe_ckpt.
   - Supports --compare_frozen_dino.

2. --backend factory
   - Uses build_feature_extractor(...) for UNI / UNI2H / Virchow2 / OpenCLIP MoE adapters.
   - Requires --adapted_encoder_name and --stage2_ckpt.
   - Frozen comparison is intentionally disabled for this backend to avoid layer-token
     alignment bugs across heterogeneous wrapper implementations.

Main outputs
------------
1. Token-level feature-space panels for MoE last-MoE and final token features.
2. Optional frozen DINO comparison for --backend dino.
3. Expert x token-cluster heatmaps using dispatch_weight first when available.
4. Compact t-SNE comparison figure.
5. Patch expert composition analysis:
   - token routing aggregated into patch-level expert composition vectors
   - patch composition t-SNE/UMAP/PCA embedding
   - patch composition galleries
   - patch composition WSI thumbnail overlays
6. Optional token/expert-cluster morphology galleries and per-slide expert maps.

Example: DINO-MoE
-----------------
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python visualization/analyze_moe_backbone_token_behavior.py \
  --backend dino \
  --config /path/to/stage2_config.yaml \
  --moe_ckpt /path/to/dino_moe_best_full.pt \
  --slides_csv /path/to/slides.csv \
  --raw_dir /path/to/raw_wsi \
  --h5_dir /path/to/h5_coords \
  --out_dir outputs/dino_moe_token_analysis \
  --n_slides 40 \
  --max_patches_per_slide 512 \
  --token_sample_per_slide 8192 \
  --compare_frozen_dino \
  --reducer pca \
  --save_tsne_comparison \
  --save_patch_composition \
  --save_expert_cluster_galleries

Example: UNI-MoE / Virchow2-MoE / OpenCLIP-MoE
----------------------------------------------
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python visualization/analyze_moe_backbone_token_behavior.py \
  --backend factory \
  --adapted_encoder_name uni_moe \
  --uni_weight /path/to/uni_weight.pt \
  --stage2_ckpt /path/to/uni_moe_stage2.pt \
  --slides_csv /path/to/slides.csv \
  --raw_dir /path/to/raw_wsi \
  --h5_dir /path/to/h5_coords \
  --out_dir outputs/uni_moe_token_analysis \
  --target_block_1 21 \
  --target_block_2 22 \
  --source_stage2_layer_1 9 \
  --source_stage2_layer_2 10 \
  --reducer pca \
  --save_tsne_comparison \
  --save_patch_composition \
  --save_expert_cluster_galleries
"""
from __future__ import annotations

import os
# Set before numpy / sklearn / torch heavy imports.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
import torchvision.transforms.v2 as T
import yaml
from PIL import Image, ImageDraw, ImageFile, ImageOps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from models.encoders.moe_encoder import MoEEncoder
from models.encoders.dinov2_encoder import DINOv2Encoder
from models.encoders.backbone_moe_factory import build_feature_extractor

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_slide_seed(base_seed: int, slide_id: str) -> int:
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(base_seed) + (int(h[:8], 16) % 100000)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def entropy_np(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def top1_top2_margin_np(p: np.ndarray) -> np.ndarray:
    if p.shape[1] < 2:
        return np.zeros((len(p),), dtype=np.float32)
    part = np.partition(p, kth=-2, axis=1)
    return (part[:, -1] - part[:, -2]).astype(np.float32)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def build_transform(image_size: int = 224):
    # Match the earlier DINO-MoE visualization script.
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


# =========================================================
# Input data helpers
# =========================================================
def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir_p = Path(raw_dir)
    exts = [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"]

    exact: List[Path] = []
    for ext in exts:
        exact.extend(raw_dir_p.rglob(f"{slide_id}{ext}"))
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise RuntimeError(f"Multiple exact WSI files for slide_id={slide_id}: {exact[:10]}")

    fuzzy: List[Path] = []
    for ext in exts:
        fuzzy.extend(raw_dir_p.rglob(f"{slide_id}*{ext}"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(f"Multiple fuzzy WSI files for slide_id={slide_id}: {fuzzy[:10]}")

    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir_p = Path(h5_dir)
    exact = list(h5_dir_p.rglob(f"{slide_id}.h5"))
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise RuntimeError(f"Multiple exact H5 files for slide_id={slide_id}: {exact[:10]}")

    fuzzy = list(h5_dir_p.rglob(f"{slide_id}*.h5"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(f"Multiple fuzzy H5 files for slide_id={slide_id}: {fuzzy[:10]}")

    raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")


def read_coords_from_h5(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        return f["coords"][:].astype(np.int64)


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    return slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")


def prepare_slides_df(slides_csv: str, n_slides: int, select_mode: str, seed: int) -> pd.DataFrame:
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
            df["label"] = 0
            print("[WARN] slides_csv has no label / slide_binary_label / y_true. Use label=0 for all slides.")

    df = df.drop_duplicates("slide_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)

    if select_mode == "csv_order":
        out = df.head(n_slides).copy()
    elif select_mode == "random":
        out = df.sample(n=min(n_slides, len(df)), random_state=seed).reset_index(drop=True)
    elif select_mode == "random_balanced":
        selected = []
        labels = sorted(df["label"].dropna().unique().tolist())
        per_label = max(1, math.ceil(n_slides / max(1, len(labels))))
        for lab in labels:
            sub = df[df["label"] == lab]
            if len(sub) == 0:
                continue
            selected.append(sub.sample(n=min(per_label, len(sub)), random_state=int(rng.integers(1, 1_000_000))))
        out = pd.concat(selected, axis=0).drop_duplicates("slide_id") if selected else df.head(0)
        if len(out) < n_slides:
            remain = df[~df["slide_id"].isin(out["slide_id"])]
            if len(remain) > 0:
                out = pd.concat([out, remain.sample(n=min(n_slides - len(out), len(remain)), random_state=seed)], axis=0)
        out = out.sample(frac=1.0, random_state=seed).head(n_slides).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown select_mode={select_mode}")

    return out


# =========================================================
# DINO model loading
# =========================================================
def unwrap_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be dict, got {type(ckpt)}")

    for key in ["student_state_dict", "model_state_dict", "state_dict", "encoder", "model", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            print(f"[Load] use ckpt['{key}'] as model state_dict")
            return ckpt[key]

    if all(torch.is_tensor(v) for v in ckpt.values()):
        print("[Load] use checkpoint itself as raw state_dict")
        return ckpt

    raise KeyError(f"Cannot find model state_dict. Top-level keys: {list(ckpt.keys())}")


def clean_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for p in ["module.", "model.", "student.", "student_model."]:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
        out[new_k] = v
    return out


def load_dino_moe_model(config_path: str, ckpt_path: str, device: str, strict: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = clean_state_dict_keys(unwrap_state_dict(ckpt))
    msg = model.load_state_dict(sd, strict=strict)
    print(f"[Load] DINO-MoE ckpt loaded with strict={strict}")
    print(msg)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    moe_layers_idx = cfg["moe_encoder"].get("moe_layers", getattr(model, "moe_layers_idx", []))
    depth = len(model.blocks)
    real_moe_blocks = [i if i >= 0 else depth + i for i in moe_layers_idx]
    print(f"[Model] moe_layers_idx={moe_layers_idx}, real_moe_blocks={real_moe_blocks}")
    return model, cfg, real_moe_blocks


def load_frozen_dino_from_config(cfg: Dict, device: str):
    frozen = DINOv2Encoder(**cfg["base_encoder"])
    frozen = frozen.to(device).eval()
    for p in frozen.parameters():
        p.requires_grad = False
    return frozen


# =========================================================
# Factory model loading for UNI / UNI2H / Virchow2 / OpenCLIP MoE
# =========================================================
def build_factory_namespace(args) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_name=args.adapted_encoder_name,
        device=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        max_patches=None,
        seed=args.seed,
        overwrite=False,

        virchow2_weight=args.virchow2_weight,
        uni_weight=args.uni_weight,
        uni2_weight=args.uni2_weight,

        openclip_model_name=args.openclip_model_name,
        openclip_weight=args.openclip_weight,
        openclip_precision=args.openclip_precision,
        no_openclip_normalize=args.no_openclip_normalize,

        hopt_local_hf_hub_id="",
        hopt_manual_arch_name="",
        hopt_weight="",

        dinov2_model_name="facebook/dinov2-small",
        dinov2_weight="",
        dino_feature_layer=-1,

        moe_ckpt="",
        moe_config="",
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


def load_factory_moe_extractor(args):
    if not args.stage2_ckpt:
        raise ValueError("--stage2_ckpt is required when --backend factory")

    print(f"[Build] factory MoE extractor: {args.adapted_encoder_name}")
    extractor = build_feature_extractor(build_factory_namespace(args))

    if not hasattr(extractor, "model"):
        raise RuntimeError(
            "Factory extractor does not expose `.model`; cannot run token-level MoE analysis. "
            "Please check build_feature_extractor output."
        )

    model = extractor.model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return extractor


# =========================================================
# Forward helpers: DINO
# =========================================================
def normalize_dispatch(dispatch: torch.Tensor) -> torch.Tensor:
    dispatch = dispatch.float().clamp_min(0.0)
    denom = dispatch.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return dispatch / denom


def parse_dispatch_weight(gate_info: Any, B: int, seq_len: int) -> torch.Tensor:
    if isinstance(gate_info, dict) and "dispatch_weight" in gate_info:
        dispatch = gate_info["dispatch_weight"]
    elif isinstance(gate_info, dict):
        keys = list(gate_info.keys())
        raise KeyError(f"gate_info does not contain 'dispatch_weight'. Existing keys={keys}")
    else:
        raise TypeError(f"Expected gate_info dict with dispatch_weight, got {type(gate_info)}")

    if not torch.is_tensor(dispatch):
        raise TypeError(f"dispatch_weight must be tensor, got {type(dispatch)}")

    if dispatch.ndim != 2:
        raise ValueError(f"dispatch_weight expected [B*seq_len, E], got shape={tuple(dispatch.shape)}")

    total_tokens, n_experts = dispatch.shape
    if total_tokens != B * seq_len:
        raise ValueError(f"dispatch_weight first dim mismatch: got {total_tokens}, expected B*seq_len={B*seq_len}")

    return normalize_dispatch(dispatch.reshape(B, seq_len, n_experts))


@torch.no_grad()
def run_dino_moe_batch(model: MoEEncoder, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    model_out = model(images, return_gates=True, return_features=True, is_eval=True)
    if not (isinstance(model_out, (tuple, list)) and len(model_out) == 4):
        raise RuntimeError("MoEEncoder must return (final_tokens, gate_info_list, feature_dict, moe_feature_list)")
    final_tokens, gate_info_list, feature_dict, moe_feature_list = model_out
    if len(gate_info_list) == 0 or len(moe_feature_list) == 0:
        raise RuntimeError("Empty gate_info_list or moe_feature_list")

    B, seq_len, _ = final_tokens.shape
    token_probs_all = parse_dispatch_weight(gate_info_list[-1], B=B, seq_len=seq_len)

    # DINO has CLS only; remove CLS token.
    token_start = 1
    last_moe_tokens = moe_feature_list[-1][:, token_start:, :]
    final_image_tokens = final_tokens[:, token_start:, :]
    token_probs = token_probs_all[:, token_start:, :]

    return {
        "last_moe_tokens": last_moe_tokens,
        "final_tokens": final_image_tokens,
        "token_expert_probs": token_probs,
    }


@torch.no_grad()
def run_frozen_dino_batch(frozen: DINOv2Encoder, images: torch.Tensor, target_block: int) -> Dict[str, torch.Tensor]:
    x = frozen.patch_embed_forward(images)
    last_block_tokens = None
    for i, blk in enumerate(frozen.blocks):
        x = blk(x)
        if i == target_block:
            last_block_tokens = x
    final_tokens = frozen.norm(x)
    if last_block_tokens is None:
        raise RuntimeError(f"target_block={target_block} was not reached in frozen DINO")
    return {
        "frozen_last_feat": last_block_tokens[:, 1:, :],
        "frozen_final_feat": final_tokens[:, 1:, :],
    }


# =========================================================
# Forward helpers: factory backend
# =========================================================
def _find_gate_tensor_with_last_dim(obj: Any, n_experts: int, prefer_dispatch: bool = True) -> Optional[torch.Tensor]:
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
                "routing_weights", "gate_weights", "weights",
                "dispatch_weight", "dispatch_weights", "dispatch_mask",
                "combine_weights", "logits", "gate_logits", "router_logits",
            ]

        for k in preferred:
            if k in obj:
                found = _find_gate_tensor_with_last_dim(obj[k], n_experts, prefer_dispatch=prefer_dispatch)
                if found is not None:
                    return found

        for v in obj.values():
            found = _find_gate_tensor_with_last_dim(v, n_experts, prefer_dispatch=prefer_dispatch)
            if found is not None:
                return found

    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_gate_tensor_with_last_dim(v, n_experts, prefer_dispatch=prefer_dispatch)
            if found is not None:
                return found

    return None


def _normalize_gate_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    row_sum = x.sum(dim=-1, keepdim=True)
    looks_like_prob = (
        bool(torch.all(x >= -1e-6))
        and bool(torch.all(x <= 1.0 + 1e-6))
        and bool(torch.allclose(row_sum.mean(), torch.ones_like(row_sum.mean()), atol=0.15))
    )
    if looks_like_prob:
        x = x.clamp_min(0.0)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.softmax(x, dim=-1)


def _infer_num_experts_from_factory_model(model: Any, fallback: int) -> int:
    if hasattr(model, "num_experts"):
        n = int(getattr(model, "num_experts"))
        if n > 0:
            return n

    moe_layer_map = getattr(model, "moe_layer_map", None)
    if isinstance(moe_layer_map, dict) and len(moe_layer_map) > 0:
        moe_blocks = sorted(list(moe_layer_map.keys()))
        blk = model.blocks[moe_blocks[-1]]
        mlp = blk.mlp
        if hasattr(mlp, "moe") and hasattr(mlp.moe, "num_experts"):
            return int(mlp.moe.num_experts)
        if hasattr(mlp, "num_experts"):
            return int(mlp.num_experts)

    return int(fallback)


def _gate_info_to_token_probs_generic(gate_info: Any, n_experts: int, B: int, seq_len: int) -> torch.Tensor:
    gate_tensor = _find_gate_tensor_with_last_dim(gate_info, n_experts, prefer_dispatch=True)
    if gate_tensor is None:
        raise RuntimeError(
            "Could not parse gate_info. Need a tensor whose last dimension equals num_experts. "
            "Please print gate_info keys/shapes."
        )

    gate_tensor = gate_tensor.detach()

    if gate_tensor.ndim == 3 and gate_tensor.shape[0] == B and gate_tensor.shape[1] == seq_len:
        tok = gate_tensor
    elif gate_tensor.ndim == 2 and gate_tensor.shape[0] == B * seq_len:
        tok = gate_tensor.reshape(B, seq_len, n_experts)
    elif gate_tensor.ndim == 2 and gate_tensor.shape[0] == B:
        tok = gate_tensor[:, None, :].expand(B, seq_len, n_experts)
    elif gate_tensor.ndim > 3 and gate_tensor.shape[0] == B and gate_tensor.shape[1] == seq_len:
        dims_to_mean = tuple(range(2, gate_tensor.ndim - 1))
        tok = gate_tensor.mean(dim=dims_to_mean)
    else:
        raise RuntimeError(
            f"Cannot map gate tensor shape={tuple(gate_tensor.shape)} to [B, seq_len, E]. "
            f"B={B}, seq_len={seq_len}, E={n_experts}"
        )

    return _normalize_gate_tensor(tok)


@torch.no_grad()
def run_factory_moe_batch(extractor: Any, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
    """
    Generic token-level forward for UNI / UNI2H / Virchow2 / OpenCLIP MoE adapters.

    Expected model output:
        final_tokens, gate_info_list, feature_dict, moe_feature_list
    """
    model = extractor.model
    reg_tokens = int(getattr(extractor, "reg_tokens", getattr(model, "reg_tokens", 0)))
    n_experts = _infer_num_experts_from_factory_model(model=model, fallback=getattr(extractor, "num_experts", 4))

    try:
        model_out = model(images, return_gates=True, return_features=True, is_eval=True)
    except TypeError as e:
        raise RuntimeError(
            "Factory MoE model does not support "
            "model(images, return_gates=True, return_features=True, is_eval=True). "
            "Need to expose token-level MoE outputs in this extractor."
        ) from e

    if not (isinstance(model_out, (tuple, list)) and len(model_out) == 4):
        raise RuntimeError(
            "Expected factory MoE model to return "
            "(final_tokens, gate_info_list, feature_dict, moe_feature_list). "
            f"Got type={type(model_out)}, len={len(model_out) if isinstance(model_out, (tuple, list)) else 'NA'}."
        )

    final_tokens, gate_info_list, feature_dict, moe_feature_list = model_out
    if len(gate_info_list) == 0:
        raise RuntimeError("gate_info_list is empty.")
    if len(moe_feature_list) == 0:
        raise RuntimeError("moe_feature_list is empty.")

    last_moe_tokens_all = moe_feature_list[-1]
    final_tokens_all = final_tokens
    B, seq_len, _ = final_tokens_all.shape
    token_probs_all = _gate_info_to_token_probs_generic(
        gate_info=gate_info_list[-1],
        n_experts=n_experts,
        B=B,
        seq_len=seq_len,
    )

    token_start = 1 + reg_tokens
    if seq_len <= token_start:
        raise RuntimeError(f"seq_len={seq_len} <= token_start={token_start}. Check reg_tokens / token layout.")

    return {
        "last_moe_tokens": last_moe_tokens_all[:, token_start:, :],
        "final_tokens": final_tokens_all[:, token_start:, :],
        "token_expert_probs": token_probs_all[:, token_start:, :],
    }


# =========================================================
# Extraction
# =========================================================
def sample_token_indices(rng: np.random.Generator, B: int, T: int, want: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_flat = B * T
    want = max(1, min(int(want), n_flat))
    flat_idx = rng.choice(n_flat, size=want, replace=False)
    local_patch_idx = flat_idx // T
    token_idx = flat_idx % T
    return flat_idx, local_patch_idx.astype(np.int64), token_idx.astype(np.int64)


@torch.no_grad()
def extract_one_slide_tokens(
    backend: str,
    model: Optional[MoEEncoder],
    extractor: Optional[Any],
    frozen_model: Optional[DINOv2Encoder],
    slide_path: str,
    h5_path: str,
    transform,
    device: str,
    patch_size: int,
    batch_size: int,
    max_patches: Optional[int],
    token_sample_per_slide: int,
    seed: int,
    frozen_target_block: Optional[int],
) -> Dict[str, np.ndarray]:
    coords = read_coords_from_h5(h5_path)
    if max_patches is not None and len(coords) > max_patches:
        rng0 = np.random.default_rng(seed)
        idx = rng0.choice(len(coords), size=max_patches, replace=False)
        coords = coords[idx]

    rng = np.random.default_rng(seed + 137)
    slide = openslide.OpenSlide(slide_path)
    token_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
    total_patches = len(coords)
    sample_per_patch = token_sample_per_slide / max(1, total_patches)

    try:
        for start in tqdm(
            range(0, len(coords), batch_size),
            total=math.ceil(len(coords) / batch_size),
            desc=f"  Tokens[{Path(slide_path).stem[:24]}]",
            leave=False,
        ):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]
            images_tensor_input = []
            raw_pil_images = []
            for xy in batch_coords.tolist():
                img = read_patch_from_wsi(slide, (int(xy[0]), int(xy[1])), patch_size=patch_size, read_level=0)
                raw_pil_images.append(img)
                images_tensor_input.append(transform(img))

            if backend == "dino":
                if model is None:
                    raise RuntimeError("DINO backend requires model")
                x = torch.stack(images_tensor_input, dim=0).to(device, non_blocking=True)
                out = run_dino_moe_batch(model, x)
            elif backend == "factory":
                if extractor is None:
                    raise RuntimeError("Factory backend requires extractor")
                out = run_factory_moe_batch(extractor, raw_pil_images)
                x = None
            else:
                raise ValueError(f"Unknown backend={backend}")

            last_tok = out["last_moe_tokens"]          # [B,T,D]
            final_tok = out["final_tokens"]            # [B,T,D]
            probs_tok = out["token_expert_probs"]      # [B,T,E]
            B, T, D = last_tok.shape
            want = int(math.ceil(B * sample_per_patch))
            flat_idx, local_patch, token_idx = sample_token_indices(rng, B, T, want)

            token_buffers["token_last_moe_feat"].append(to_numpy(last_tok.reshape(B * T, D)[flat_idx]).astype(np.float32))
            token_buffers["token_final_feat"].append(to_numpy(final_tok.reshape(B * T, D)[flat_idx]).astype(np.float32))
            token_buffers["token_expert_probs"].append(to_numpy(probs_tok.reshape(B * T, probs_tok.shape[-1])[flat_idx]).astype(np.float32))
            token_buffers["coord_x"].append(batch_coords[local_patch, 0].astype(np.int64))
            token_buffers["coord_y"].append(batch_coords[local_patch, 1].astype(np.int64))
            token_buffers["patch_index_in_slide"].append((start + local_patch).astype(np.int64))
            token_buffers["token_index_in_patch"].append(token_idx.astype(np.int64))

            if backend == "dino" and frozen_model is not None:
                assert frozen_target_block is not None
                assert x is not None
                fout = run_frozen_dino_batch(frozen_model, x, target_block=frozen_target_block)
                f_last = fout["frozen_last_feat"]
                f_final = fout["frozen_final_feat"]
                if f_last.shape[1] != T:
                    raise RuntimeError(f"Frozen token count {f_last.shape[1]} != MoE token count {T}")
                token_buffers["frozen_last_feat"].append(to_numpy(f_last.reshape(B * T, f_last.shape[-1])[flat_idx]).astype(np.float32))
                token_buffers["frozen_final_feat"].append(to_numpy(f_final.reshape(B * T, f_final.shape[-1])[flat_idx]).astype(np.float32))
    finally:
        slide.close()

    merged: Dict[str, np.ndarray] = {}
    for k, values in token_buffers.items():
        arr = np.concatenate(values, axis=0)
        if len(arr) > token_sample_per_slide:
            keep = rng.choice(len(arr), size=token_sample_per_slide, replace=False)
            arr = arr[keep]
        merged[k] = arr
    return merged


# =========================================================
# Plotting / embeddings
# =========================================================
def compute_embedding(
    X: np.ndarray,
    reducer_type: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    tsne_perplexity: float = 30.0,
    tsne_max_points: int = 30000,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return embedding and optional sampled index.

    PCA / UMAP: embedding corresponds to all rows, selected_idx=None.
    t-SNE: may sample rows if X is too large, selected_idx gives source rows.
    """
    Xn = l2_normalize_np(X.astype(np.float32))

    if reducer_type == "umap" and HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        )
        emb = reducer.fit_transform(Xn)
        return emb.astype(np.float32), None

    if reducer_type == "pca":
        emb = PCA(n_components=2, random_state=seed).fit_transform(Xn)
        return emb.astype(np.float32), None

    if reducer_type == "tsne":
        rng = np.random.default_rng(seed)
        if len(Xn) > tsne_max_points:
            idx = rng.choice(len(Xn), size=tsne_max_points, replace=False)
            X_use = Xn[idx]
        else:
            idx = np.arange(len(Xn))
            X_use = Xn

        pca_dim = min(50, X_use.shape[1], max(2, X_use.shape[0] - 1))
        X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_use)
        perplexity = min(float(tsne_perplexity), max(5.0, (len(X_pca) - 1) / 3.0))
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                metric="cosine",
                random_state=seed,
                max_iter=1000,
                verbose=1,
            )
        except TypeError:
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                metric="cosine",
                random_state=seed,
                n_iter=1000,
                verbose=1,
            )
        emb = tsne.fit_transform(X_pca)
        return emb.astype(np.float32), idx.astype(np.int64)

    emb = PCA(n_components=2, random_state=seed).fit_transform(Xn)
    return emb.astype(np.float32), None


def cluster_features(X: np.ndarray, n_clusters: int, seed: int, pca_dim: int = 32) -> np.ndarray:
    Xn = l2_normalize_np(X.astype(np.float32))
    dim = min(pca_dim, Xn.shape[1], max(2, Xn.shape[0] - 1))
    Xp = PCA(n_components=dim, random_state=seed).fit_transform(Xn)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(Xp).astype(np.int64)


def plot_category_scatter(ax, emb: np.ndarray, values: np.ndarray, title: str, prefix: str = "", size: float = 4.0):
    uniq = sorted(pd.Series(values).dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    for i, v in enumerate(uniq):
        m = values == v
        ax.scatter(emb[m, 0], emb[m, 1], s=size, alpha=0.65, color=cmap(i % 20), label=f"{prefix}{v}", linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(uniq) <= 12:
        ax.legend(markerscale=3, fontsize=7, frameon=False)


def plot_continuous_scatter(ax, emb: np.ndarray, values: np.ndarray, title: str, cmap: str = "viridis", size: float = 4.0):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, s=size, alpha=0.70, cmap=cmap, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def save_token_scatter_panel(out_dir: Path, df: pd.DataFrame, layer_prefix: str, title_prefix: str):
    ensure_dir(out_dir)
    emb = df[[f"{layer_prefix}_x", f"{layer_prefix}_y"]].values.astype(np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    plot_category_scatter(axes[0, 0], emb, df[f"{layer_prefix}_cluster"].values, f"{title_prefix}: cluster", prefix="c")
    plot_category_scatter(axes[0, 1], emb, df["expert_id"].values, f"{title_prefix}: expert", prefix="E")
    plot_category_scatter(axes[0, 2], emb, df["label"].values, f"{title_prefix}: slide label", prefix="y=")
    sc1 = plot_continuous_scatter(axes[1, 0], emb, df["expert_entropy"].values, f"{title_prefix}: dispatch entropy", cmap="viridis")
    sc2 = plot_continuous_scatter(axes[1, 1], emb, df["expert_margin"].values, f"{title_prefix}: top1-top2 margin", cmap="inferno")
    plot_category_scatter(axes[1, 2], emb, df["slide_id"].values, f"{title_prefix}: slide", prefix="", size=3.0)
    fig.colorbar(sc1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    fig.colorbar(sc2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_prefix}_token_scatter_panel.png", dpi=240)
    plt.close(fig)


def save_frozen_scatter_panel(out_dir: Path, df: pd.DataFrame, layer_prefix: str, title_prefix: str):
    ensure_dir(out_dir)
    emb = df[[f"{layer_prefix}_x", f"{layer_prefix}_y"]].values.astype(np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    plot_category_scatter(axes[0], emb, df[f"{layer_prefix}_cluster"].values, f"{title_prefix}: cluster", prefix="c")
    plot_category_scatter(axes[1], emb, df["label"].values, f"{title_prefix}: slide label", prefix="y=")
    plot_category_scatter(axes[2], emb, df["slide_id"].values, f"{title_prefix}: slide", prefix="", size=3.0)
    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_prefix}_frozen_token_scatter_panel.png", dpi=240)
    plt.close(fig)


def save_compact_embedding_comparison(out_path: Path, panels: List[Tuple[str, np.ndarray, np.ndarray, str]], point_size: float = 2.5) -> None:
    ensure_dir(out_path.parent)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("tab20")

    for ax, (title, emb, values, mode) in zip(axes, panels):
        uniq = sorted(pd.Series(values).dropna().unique().tolist())
        for i, v in enumerate(uniq):
            m = values == v
            if mode == "expert":
                label = f"E{v}"
            elif mode == "cluster":
                label = f"c{v}"
            elif mode == "label":
                label = f"y={v}"
            else:
                label = str(v)
            ax.scatter(emb[m, 0], emb[m, 1], s=point_size, alpha=0.70, color=cmap(i % 20), label=label, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if len(uniq) <= 12:
            ax.legend(markerscale=4, fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def compute_tsne_for_fixed_subset(X: np.ndarray, idx: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    X_use = l2_normalize_np(X[idx].astype(np.float32))
    pca_dim = min(50, X_use.shape[1], max(2, X_use.shape[0] - 1))
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X_use)
    pp = min(perplexity, max(5.0, (len(X_pca) - 1) / 3.0))
    try:
        tsne = TSNE(n_components=2, perplexity=pp, init="pca", learning_rate="auto", metric="cosine", random_state=seed, max_iter=1000, verbose=1)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=pp, init="pca", learning_rate="auto", metric="cosine", random_state=seed, n_iter=1000, verbose=1)
    return tsne.fit_transform(X_pca).astype(np.float32)


def save_heatmap(mat: np.ndarray, out_path: Path, title: str, xlabel: str, ylabel: str, xticklabels: Sequence[str], yticklabels: Sequence[str], cmap: str = "Blues", vmin=None, vmax=None):
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(xticklabels)), max(4, 0.5 * len(yticklabels))))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def expert_cluster_tables(expert_probs: np.ndarray, clusters: np.ndarray, n_clusters: int) -> Dict[str, np.ndarray]:
    n_experts = expert_probs.shape[1]
    expert_ids = np.argmax(expert_probs, axis=1)
    hard_counts = np.zeros((n_experts, n_clusters), dtype=np.float32)
    soft_mass = np.zeros((n_experts, n_clusters), dtype=np.float32)
    cluster_counts = np.zeros((n_clusters,), dtype=np.float32)
    for c in range(n_clusters):
        m = clusters == c
        cluster_counts[c] = float(m.sum())
        if m.sum() == 0:
            continue
        soft_mass[:, c] = expert_probs[m].sum(axis=0)
        for e in range(n_experts):
            hard_counts[e, c] = float(np.sum(expert_ids[m] == e))
    return {
        "soft_p_expert_given_cluster": soft_mass / np.clip(soft_mass.sum(axis=0, keepdims=True), 1e-8, None),
        "soft_p_cluster_given_expert": soft_mass / np.clip(soft_mass.sum(axis=1, keepdims=True), 1e-8, None),
        "hard_p_expert_given_cluster": hard_counts / np.clip(hard_counts.sum(axis=0, keepdims=True), 1e-8, None),
        "hard_p_cluster_given_expert": hard_counts / np.clip(hard_counts.sum(axis=1, keepdims=True), 1e-8, None),
        "hard_counts": hard_counts,
        "cluster_counts": cluster_counts,
    }

# =========================================================
# Expert-cluster specialization stats: entropy / JSD
# =========================================================
def normalize_prob_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    row_sum = x.sum(axis=1, keepdims=True)
    return x / np.clip(row_sum, eps, None)


def js_divergence_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """
    Jensen-Shannon divergence with log base 2.
    Range: [0, 1].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p = p / np.clip(p.sum(), eps, None)
    q = q / np.clip(q.sum(), eps, None)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * (np.log2(p) - np.log2(m)))
    kl_qm = np.sum(q * (np.log2(q) - np.log2(m)))

    return float(0.5 * (kl_pm + kl_qm))


def pairwise_jsd_matrix_np(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    P: [N, K], each row is a probability distribution.
    """
    P = normalize_prob_rows(P, eps=eps)
    n = P.shape[0]
    mat = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            mat[i, j] = js_divergence_np(P[i], P[j], eps=eps)

    return mat


def save_expert_cluster_specialization_stats(
    out_dir: Path,
    tables: Dict[str, np.ndarray],
    layer_name: str,
    n_experts: int,
    n_clusters: int,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Save expert-cluster entropy and pairwise JSD based on P(cluster | expert).

    Main interpretation:
    - P(cluster | expert): which token clusters each expert tends to process.
    - Entropy: whether one expert focuses on a few clusters or broadly covers many.
    - Effective clusters: soft number of clusters used by each expert.
    - Pairwise JSD: how different the experts' cluster preferences are.
    """
    ensure_dir(out_dir)

    expert_names = [f"E{i}" for i in range(n_experts)]
    cluster_names = [f"c{i}" for i in range(n_clusters)]

    p_cluster_given_expert = tables["soft_p_cluster_given_expert"].astype(np.float64)
    p_expert_given_cluster = tables["soft_p_expert_given_cluster"].astype(np.float64)
    hard_counts = tables["hard_counts"].astype(np.float64)
    cluster_counts = tables["cluster_counts"].astype(np.float64)

    # Re-normalize for numerical safety.
    p_cluster_given_expert = normalize_prob_rows(p_cluster_given_expert, eps=eps)

    # Expert-wise cluster entropy.
    expert_entropy = entropy_np(p_cluster_given_expert, eps=eps).astype(np.float32)
    expert_entropy_norm = expert_entropy / float(np.log(max(2, n_clusters)))

    # Effective number of clusters used by each expert.
    effective_clusters = (1.0 / np.sum(p_cluster_given_expert ** 2, axis=1)).astype(np.float32)

    # Dominant cluster for each expert.
    dominant_cluster = np.argmax(p_cluster_given_expert, axis=1).astype(np.int64)
    dominant_cluster_prob = np.max(p_cluster_given_expert, axis=1).astype(np.float32)

    # Pairwise JSD among experts.
    jsd_mat = pairwise_jsd_matrix_np(p_cluster_given_expert, eps=eps)
    upper = jsd_mat[np.triu_indices(n_experts, k=1)]

    mean_pairwise_jsd = float(np.mean(upper)) if len(upper) > 0 else 0.0
    median_pairwise_jsd = float(np.median(upper)) if len(upper) > 0 else 0.0
    min_pairwise_jsd = float(np.min(upper)) if len(upper) > 0 else 0.0
    max_pairwise_jsd = float(np.max(upper)) if len(upper) > 0 else 0.0

    # =========================
    # Save CSV tables
    # =========================
    pd.DataFrame(
        p_cluster_given_expert,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{layer_name}_p_cluster_given_expert.csv")

    pd.DataFrame(
        p_expert_given_cluster,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{layer_name}_p_expert_given_cluster.csv")

    pd.DataFrame(
        hard_counts,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{layer_name}_hard_counts.csv")

    pd.DataFrame({
        "cluster": cluster_names,
        "count": cluster_counts.astype(np.int64),
    }).to_csv(out_dir / f"{layer_name}_cluster_counts.csv", index=False)

    pd.DataFrame(
        jsd_mat,
        index=expert_names,
        columns=expert_names,
    ).to_csv(out_dir / f"{layer_name}_expert_cluster_jsd_matrix.csv")

    entropy_df = pd.DataFrame({
        "expert": expert_names,
        "cluster_entropy": expert_entropy,
        "cluster_entropy_norm": expert_entropy_norm,
        "effective_clusters": effective_clusters,
        "dominant_cluster": [f"c{i}" for i in dominant_cluster],
        "dominant_cluster_prob": dominant_cluster_prob,
    })
    entropy_df.to_csv(out_dir / f"{layer_name}_expert_cluster_entropy.csv", index=False)

    # =========================
    # Save figures
    # =========================
    save_heatmap(
        p_cluster_given_expert,
        out_dir / f"{layer_name}_p_cluster_given_expert.png",
        title=f"{layer_name}: P(cluster | expert)",
        xlabel="token cluster",
        ylabel="expert",
        xticklabels=cluster_names,
        yticklabels=expert_names,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    save_heatmap(
        p_expert_given_cluster,
        out_dir / f"{layer_name}_p_expert_given_cluster.png",
        title=f"{layer_name}: P(expert | cluster)",
        xlabel="token cluster",
        ylabel="expert",
        xticklabels=cluster_names,
        yticklabels=expert_names,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    save_heatmap(
        jsd_mat,
        out_dir / f"{layer_name}_expert_cluster_jsd_matrix.png",
        title=f"{layer_name}: pairwise JSD between experts",
        xlabel="expert",
        ylabel="expert",
        xticklabels=expert_names,
        yticklabels=expert_names,
        cmap="Oranges",
        vmin=0.0,
        vmax=max(1e-6, float(jsd_mat.max())),
    )

    # Entropy bar
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(n_experts)
    ax.bar(x, expert_entropy_norm)
    ax.set_xticks(x)
    ax.set_xticklabels(expert_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalized cluster entropy")
    ax.set_title(f"{layer_name}: expert cluster entropy")

    for i, v in enumerate(expert_entropy_norm):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_name}_expert_cluster_entropy_bar.png", dpi=220)
    plt.close(fig)

    # Effective clusters bar
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, effective_clusters)
    ax.set_xticks(x)
    ax.set_xticklabels(expert_names)
    ax.set_ylabel("Effective number of clusters")
    ax.set_title(f"{layer_name}: effective clusters per expert")
    ax.set_ylim(0.0, max(1.0, float(np.max(effective_clusters)) * 1.2))

    for i, v in enumerate(effective_clusters):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_name}_effective_clusters_bar.png", dpi=220)
    plt.close(fig)

    # Dominant cluster probability bar
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, dominant_cluster_prob)
    ax.set_xticks(x)
    ax.set_xticklabels(expert_names)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Dominant cluster probability")
    ax.set_title(f"{layer_name}: dominant cluster strength")

    for i, v in enumerate(dominant_cluster_prob):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_name}_dominant_cluster_prob_bar.png", dpi=220)
    plt.close(fig)

    summary: Dict[str, Any] = {
        "layer": layer_name,
        "n_experts": int(n_experts),
        "n_clusters": int(n_clusters),
        "mean_expert_cluster_entropy": float(np.mean(expert_entropy)),
        "mean_expert_cluster_entropy_norm": float(np.mean(expert_entropy_norm)),
        "mean_effective_clusters": float(np.mean(effective_clusters)),
        "mean_pairwise_expert_jsd": mean_pairwise_jsd,
        "median_pairwise_expert_jsd": median_pairwise_jsd,
        "min_pairwise_expert_jsd": min_pairwise_jsd,
        "max_pairwise_expert_jsd": max_pairwise_jsd,
        "expert_entropy_norm": {
            f"E{i}": float(expert_entropy_norm[i]) for i in range(n_experts)
        },
        "effective_clusters": {
            f"E{i}": float(effective_clusters[i]) for i in range(n_experts)
        },
        "dominant_cluster": {
            f"E{i}": f"c{int(dominant_cluster[i])}" for i in range(n_experts)
        },
        "dominant_cluster_prob": {
            f"E{i}": float(dominant_cluster_prob[i]) for i in range(n_experts)
        },
    }

    with open(out_dir / f"{layer_name}_expert_cluster_stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

def save_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 60):
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_expert_usage_bar(expert_ids: np.ndarray, n_experts: int, out_path: Path):
    ensure_dir(out_path.parent)
    counts = np.array([(expert_ids == e).sum() for e in range(n_experts)], dtype=np.float32)
    frac = counts / max(1.0, counts.sum())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([f"E{e}" for e in range(n_experts)], frac)
    ax.set_ylim(0, max(0.05, float(frac.max()) * 1.2))
    ax.set_ylabel("fraction of sampled tokens")
    ax.set_title("Hard expert usage from routing/dispatch")
    for i, v in enumerate(frac):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# =========================================================
# Spatial maps / overlays
# =========================================================
def token_global_xy(df: pd.DataFrame, patch_size: int, n_tokens_per_patch: int) -> Tuple[np.ndarray, np.ndarray]:
    grid = int(round(math.sqrt(n_tokens_per_patch)))
    if grid * grid != n_tokens_per_patch:
        gx = df["coord_x"].values.astype(np.float32) + patch_size / 2.0
        gy = df["coord_y"].values.astype(np.float32) + patch_size / 2.0
        return gx, gy
    token_idx = df["token_index_in_patch"].values.astype(np.int64)
    token_col = token_idx % grid
    token_row = token_idx // grid
    cell = float(patch_size) / float(grid)
    gx = df["coord_x"].values.astype(np.float32) + (token_col.astype(np.float32) + 0.5) * cell
    gy = df["coord_y"].values.astype(np.float32) + (token_row.astype(np.float32) + 0.5) * cell
    return gx, gy


def save_one_slide_expert_routing_maps(
    out_dir: Path,
    slide_df: pd.DataFrame,
    slide_id: str,
    patch_size: int,
    n_tokens_per_patch: int,
    n_experts: int,
    max_points: int = 30000,
) -> None:
    ensure_dir(out_dir)
    sub = slide_df.copy()
    if len(sub) > max_points:
        sub = sub.sample(n=max_points, random_state=0).copy()

    gx, gy = token_global_xy(sub, patch_size=patch_size, n_tokens_per_patch=n_tokens_per_patch)
    x = gx
    y = -gy
    pad_x = max(1.0, 0.02 * (float(x.max()) - float(x.min()) + 1.0))
    pad_y = max(1.0, 0.02 * (float(y.max()) - float(y.min()) + 1.0))
    xlim = (float(x.min()) - pad_x, float(x.max()) + pad_x)
    ylim = (float(y.min()) - pad_y, float(y.max()) + pad_y)

    n_cols = min(4, n_experts)
    n_rows = int(math.ceil(n_experts / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False)
    for e in range(n_experts):
        ax = axes[e // n_cols][e % n_cols]
        vals = sub[f"expert_prob_{e}"].values.astype(np.float32)
        sc = ax.scatter(x, y, c=vals, s=3, cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.85, linewidths=0)
        ax.set_title(f"{slide_id}: E{e} routing prob")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    for k in range(n_experts, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slide_id}_expert_prob_maps.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for e in range(n_experts):
        m = sub["expert_id"].values == e
        ax.scatter(x[m], y[m], s=4, alpha=0.75, color=cmap(e % 10), label=f"E{e} n={int(m.sum())}", linewidths=0)
    ax.set_title(f"{slide_id}: hard expert routing map")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{slide_id}_hard_expert_map.png", dpi=220)
    plt.close(fig)


def save_slide_expert_routing_maps(
    out_dir: Path,
    token_df: pd.DataFrame,
    patch_size: int,
    n_tokens_per_patch: int,
    n_experts: int,
    max_slides: int,
    max_points_per_slide: int,
) -> None:
    ensure_dir(out_dir)
    slide_ids = token_df["slide_id"].drop_duplicates().tolist()[:max_slides]
    for slide_id in slide_ids:
        sub = token_df[token_df["slide_id"] == slide_id]
        save_one_slide_expert_routing_maps(out_dir, sub, str(slide_id), patch_size, n_tokens_per_patch, n_experts, max_points_per_slide)


def _find_wsi_path_for_overlay(raw_dir: str, slide_id: str) -> Optional[str]:
    try:
        return find_wsi_path(raw_dir, slide_id)
    except Exception as e:
        print(f"[Overlay][WARN] cannot find WSI for {slide_id}: {e}")
        return None


def aggregate_tokens_for_overlay(slide_df: pd.DataFrame, n_experts: int, aggregate: str = "patch_mean") -> pd.DataFrame:
    if aggregate == "token":
        out = slide_df.copy()
        out["overlay_x"] = out["token_global_x"]
        out["overlay_y"] = out["token_global_y"]
        return out

    if aggregate == "patch_mean":
        agg_func = "mean"
    elif aggregate == "patch_max":
        agg_func = "max"
    else:
        raise ValueError(f"Unknown overlay aggregate: {aggregate}")

    agg = slide_df.groupby(["coord_x", "coord_y"], as_index=False)[[f"expert_prob_{e}" for e in range(n_experts)]].agg(agg_func)
    agg["overlay_x"] = agg["coord_x"].astype(np.float32) + 0.5 * float(slide_df.attrs.get("patch_size", 256))
    agg["overlay_y"] = agg["coord_y"].astype(np.float32) + 0.5 * float(slide_df.attrs.get("patch_size", 256))
    prob_mat = agg[[f"expert_prob_{e}" for e in range(n_experts)]].values.astype(np.float32)
    agg["expert_id"] = np.argmax(prob_mat, axis=1)
    agg["expert_prob"] = prob_mat[np.arange(len(agg)), agg["expert_id"].values]
    return agg


def save_one_slide_expert_thumbnail_overlay(
    out_dir: Path,
    slide_df: pd.DataFrame,
    slide_id: str,
    raw_dir: str,
    patch_size: int,
    n_tokens_per_patch: int,
    n_experts: int,
    aggregate: str = "patch_mean",
    thumb_width: int = 2200,
    alpha: float = 0.55,
    point_size: float = 8.0,
) -> None:
    ensure_dir(out_dir)
    slide_path = _find_wsi_path_for_overlay(raw_dir, slide_id)
    if slide_path is None:
        return
    slide = openslide.OpenSlide(slide_path)
    try:
        W, H = slide.dimensions
        thumb_height = max(1, int(round(float(thumb_width) * H / max(1, W))))
        thumb = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
    finally:
        slide.close()
    sx = thumb.size[0] / float(W)
    sy = thumb.size[1] / float(H)

    sub = slide_df.copy()
    gx, gy = token_global_xy(sub, patch_size=patch_size, n_tokens_per_patch=n_tokens_per_patch)
    sub["token_global_x"] = gx
    sub["token_global_y"] = gy
    sub.attrs["patch_size"] = patch_size
    overlay_df = aggregate_tokens_for_overlay(sub, n_experts=n_experts, aggregate=aggregate)
    ox = overlay_df["overlay_x"].values.astype(np.float32) * sx
    oy = overlay_df["overlay_y"].values.astype(np.float32) * sy

    n_cols = min(4, n_experts)
    n_rows = int(math.ceil(n_experts / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 5.0 * n_rows), squeeze=False)
    for e in range(n_experts):
        ax = axes[e // n_cols][e % n_cols]
        ax.imshow(thumb)
        vals = overlay_df[f"expert_prob_{e}"].values.astype(np.float32)
        sc = ax.scatter(ox, oy, c=vals, s=point_size, cmap="inferno", vmin=0.0, vmax=1.0, alpha=alpha, linewidths=0)
        ax.set_title(f"{slide_id}: E{e} routing overlay ({aggregate})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    for k in range(n_experts, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{slide_id}_expert_prob_overlay_{aggregate}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(thumb)
    cmap = plt.get_cmap("tab10")
    hard = overlay_df["expert_id"].values.astype(np.int64)
    for e in range(n_experts):
        m = hard == e
        ax.scatter(ox[m], oy[m], s=point_size * 1.2, alpha=0.80, color=cmap(e % 10), label=f"E{e} n={int(m.sum())}", linewidths=0)
    ax.set_title(f"{slide_id}: hard expert overlay ({aggregate})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{slide_id}_hard_expert_overlay_{aggregate}.png", dpi=220)
    plt.close(fig)


def save_slide_expert_thumbnail_overlays(
    out_dir: Path,
    token_df: pd.DataFrame,
    raw_dir: str,
    patch_size: int,
    n_tokens_per_patch: int,
    n_experts: int,
    max_slides: int,
    aggregate: str,
    thumb_width: int,
    alpha: float,
    point_size: float,
) -> None:
    ensure_dir(out_dir)
    slide_ids = token_df["slide_id"].drop_duplicates().tolist()[:max_slides]
    for slide_id in slide_ids:
        sub = token_df[token_df["slide_id"] == slide_id]
        save_one_slide_expert_thumbnail_overlay(out_dir, sub, str(slide_id), raw_dir, patch_size, n_tokens_per_patch, n_experts, aggregate, thumb_width, alpha, point_size)


# =========================================================
# Galleries
# =========================================================
def draw_token_box(img: Image.Image, token_idx: int, n_tokens: int, box_color=(255, 0, 0), width: int = 4) -> Image.Image:
    img = img.convert("RGB")
    grid = int(round(math.sqrt(n_tokens)))
    if grid * grid != n_tokens:
        return img
    w, h = img.size
    cell_w = w / grid
    cell_h = h / grid
    r = int(token_idx) // grid
    c = int(token_idx) % grid
    x0 = int(c * cell_w)
    y0 = int(r * cell_h)
    x1 = int((c + 1) * cell_w)
    y1 = int((r + 1) * cell_h)
    draw = ImageDraw.Draw(img)
    for k in range(width):
        draw.rectangle([x0 + k, y0 + k, x1 - k, y1 - k], outline=box_color)
    return img


def make_montage(images: List[Image.Image], tile_size: int = 224, n_cols: int = 4, caption_h: int = 24) -> Image.Image:
    n = len(images)
    n_rows = math.ceil(n / n_cols) if n > 0 else 1
    canvas = Image.new("RGB", (n_cols * tile_size, n_rows * (tile_size + caption_h)), color=(255, 255, 255))
    for i, img in enumerate(images):
        r = i // n_cols
        c = i % n_cols
        x0 = c * tile_size
        y0 = r * (tile_size + caption_h)
        img = ImageOps.fit(img, (tile_size, tile_size), method=Image.BICUBIC)
        canvas.paste(img, (x0, y0))
        canvas.paste(Image.new("RGB", (tile_size, caption_h), color=(245, 245, 245)), (x0, y0 + tile_size))
    return canvas


def _read_patch_cached(
    patch_cache: Dict[Tuple[str, int, int], Image.Image],
    raw_dir: str,
    slide_id: str,
    coord_x: int,
    coord_y: int,
    patch_size: int,
) -> Image.Image:
    key = (slide_id, int(coord_x), int(coord_y))
    if key in patch_cache:
        return patch_cache[key].copy()
    slide_path = find_wsi_path(raw_dir, slide_id)
    slide = openslide.OpenSlide(slide_path)
    try:
        img = read_patch_from_wsi(slide, (int(coord_x), int(coord_y)), patch_size=patch_size)
    finally:
        slide.close()
    patch_cache[key] = img.copy()
    return img


def save_token_galleries(out_dir: Path, token_df: pd.DataFrame, raw_dir: str, patch_size: int, n_tokens_per_patch: int, n_experts: int, topk: int) -> None:
    ensure_dir(out_dir)
    rows = []
    patch_cache: Dict[Tuple[str, int, int], Image.Image] = {}
    for e in range(n_experts):
        sub = token_df[token_df["expert_id"] == e].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["expert_prob", "expert_margin"], ascending=False).head(topk)
        imgs = []
        for _, r in sub.iterrows():
            img = _read_patch_cached(patch_cache, raw_dir, str(r["slide_id"]), int(r["coord_x"]), int(r["coord_y"]), patch_size)
            img = draw_token_box(img, int(r["token_index_in_patch"]), n_tokens=n_tokens_per_patch)
            imgs.append(img)
            rows.append(r.to_dict())
        if imgs:
            make_montage(imgs, tile_size=224, n_cols=4).save(out_dir / f"top_tokens_expert_{e}.png")
    pd.DataFrame(rows).to_csv(out_dir / "token_gallery_index.csv", index=False)


def select_expert_cluster_pairs(token_df: pd.DataFrame, cluster_col: str, n_experts: int, min_tokens: int, top_pairs: int) -> pd.DataFrame:
    rows = []
    if cluster_col not in token_df.columns:
        raise KeyError(f"cluster_col={cluster_col} not found. Existing columns include: {list(token_df.columns)[:40]}")
    for c, sub_c in token_df.groupby(cluster_col):
        c = int(c)
        cluster_n = int(len(sub_c))
        if cluster_n < min_tokens:
            continue
        for e in range(n_experts):
            sub = sub_c[sub_c["expert_id"] == e]
            n = int(len(sub))
            if n < min_tokens:
                continue
            p_e_given_c = n / max(1, cluster_n)
            mean_prob = float(sub[f"expert_prob_{e}"].mean()) if f"expert_prob_{e}" in sub.columns else float(sub["expert_prob"].mean())
            mean_margin = float(sub["expert_margin"].mean()) if "expert_margin" in sub.columns else 0.0
            score = p_e_given_c * (0.5 + 0.5 * mean_prob) * (0.5 + 0.5 * mean_margin) * math.log1p(n)
            rows.append({
                "expert": e,
                "cluster": c,
                "n_pair_tokens": n,
                "n_cluster_tokens": cluster_n,
                "p_expert_given_cluster": p_e_given_c,
                "mean_expert_prob": mean_prob,
                "mean_margin": mean_margin,
                "score": float(score),
            })
    pair_df = pd.DataFrame(rows)
    if len(pair_df) == 0:
        return pair_df
    return pair_df.sort_values(["score", "p_expert_given_cluster", "n_pair_tokens"], ascending=False).head(top_pairs).reset_index(drop=True)


def save_expert_cluster_morphology_galleries(
    out_dir: Path,
    token_df: pd.DataFrame,
    raw_dir: str,
    patch_size: int,
    n_tokens_per_patch: int,
    n_experts: int,
    cluster_col: str,
    top_pairs: int,
    topk_per_pair: int,
    min_tokens_per_pair: int,
    diverse_slides: bool = True,
) -> None:
    ensure_dir(out_dir)
    pair_df = select_expert_cluster_pairs(token_df, cluster_col, n_experts, min_tokens_per_pair, top_pairs)
    pair_df.to_csv(out_dir / "selected_expert_cluster_pairs.csv", index=False)
    if len(pair_df) == 0:
        print(f"[ExpertClusterGallery][WARN] no expert-cluster pair passed min_tokens={min_tokens_per_pair}")
        return

    patch_cache: Dict[Tuple[str, int, int], Image.Image] = {}
    gallery_rows = []
    overview_imgs: List[Image.Image] = []

    for _, pair in pair_df.iterrows():
        e = int(pair["expert"])
        c = int(pair["cluster"])
        sub = token_df[(token_df["expert_id"] == e) & (token_df[cluster_col] == c)].copy()
        if len(sub) == 0:
            continue
        sort_cols = [col for col in [f"expert_prob_{e}", "expert_margin", "expert_prob"] if col in sub.columns]
        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=False)

        if diverse_slides:
            selected_parts = []
            for _, sub_slide in sub.groupby("slide_id", sort=False):
                selected_parts.append(sub_slide.head(1))
                if sum(len(x) for x in selected_parts) >= topk_per_pair:
                    break
            selected = pd.concat(selected_parts, axis=0) if selected_parts else sub.head(0)
            if len(selected) < topk_per_pair:
                remain = sub.drop(index=selected.index, errors="ignore")
                selected = pd.concat([selected, remain.head(topk_per_pair - len(selected))], axis=0)
        else:
            selected = sub.head(topk_per_pair)

        imgs = []
        for _, r in selected.iterrows():
            img = _read_patch_cached(patch_cache, raw_dir, str(r["slide_id"]), int(r["coord_x"]), int(r["coord_y"]), patch_size)
            img = draw_token_box(img, int(r["token_index_in_patch"]), n_tokens=n_tokens_per_patch)
            imgs.append(img)
            row = r.to_dict()
            row.update({
                "gallery_expert": e,
                "gallery_cluster": c,
                "gallery_cluster_col": cluster_col,
                "pair_p_expert_given_cluster": float(pair["p_expert_given_cluster"]),
                "pair_score": float(pair["score"]),
            })
            gallery_rows.append(row)

        if imgs:
            fname = f"E{e}_c{c}_{cluster_col}_p{float(pair['p_expert_given_cluster']):.2f}_n{int(pair['n_pair_tokens'])}.png"
            montage = make_montage(imgs, tile_size=224, n_cols=4)
            montage.save(out_dir / fname)
            overview_imgs.append(imgs[0])

    if overview_imgs:
        make_montage(overview_imgs, tile_size=224, n_cols=min(4, len(overview_imgs))).save(out_dir / "expert_cluster_pair_overview.png")
    pd.DataFrame(gallery_rows).to_csv(out_dir / "expert_cluster_gallery_index.csv", index=False)


# =========================================================
# Patch expert composition
# =========================================================
def build_patch_expert_composition(token_df: pd.DataFrame, n_experts: int) -> pd.DataFrame:
    group_cols = ["slide_id", "label", "coord_x", "coord_y", "patch_index_in_slide"]
    rows = []
    for key, sub in token_df.groupby(group_cols, sort=False):
        slide_id, label, coord_x, coord_y, patch_idx = key
        prob_mat = sub[[f"expert_prob_{e}" for e in range(n_experts)]].values.astype(np.float32)
        soft_frac = prob_mat.mean(axis=0)
        soft_frac = soft_frac / np.clip(soft_frac.sum(), 1e-8, None)
        hard_ids = sub["expert_id"].values.astype(np.int64)
        hard_frac = np.array([(hard_ids == e).mean() for e in range(n_experts)], dtype=np.float32)

        dominant_expert = int(np.argmax(soft_frac))
        hard_dominant_expert = int(np.argmax(hard_frac))
        entropy_soft = float(entropy_np(soft_frac[None, :])[0])
        entropy_hard = float(entropy_np(hard_frac[None, :])[0])
        row = {
            "slide_id": slide_id,
            "label": int(label),
            "coord_x": int(coord_x),
            "coord_y": int(coord_y),
            "patch_index_in_slide": int(patch_idx),
            "n_sampled_tokens": int(len(sub)),
            "dominant_expert": dominant_expert,
            "hard_dominant_expert": hard_dominant_expert,
            "purity": float(soft_frac.max()),
            "hard_purity": float(hard_frac.max()),
            "composition_entropy": entropy_soft,
            "hard_composition_entropy": entropy_hard,
        }
        for e in range(n_experts):
            row[f"frac_E{e}"] = float(soft_frac[e])
            row[f"hard_frac_E{e}"] = float(hard_frac[e])
        rows.append(row)
    return pd.DataFrame(rows)


def save_patch_composition_embedding(
    out_dir: Path,
    patch_df: pd.DataFrame,
    n_experts: int,
    seed: int,
    reducer: str = "tsne",
    tsne_perplexity: float = 30.0,
) -> None:
    ensure_dir(out_dir)
    X = patch_df[[f"frac_E{e}" for e in range(n_experts)]].values.astype(np.float32)
    if len(X) < 3:
        print("[PatchComposition][WARN] too few patches for embedding")
        return

    if reducer == "tsne":
        perplexity = min(tsne_perplexity, max(2.0, (len(X) - 1) / 3.0))
        try:
            emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed, max_iter=1000).fit_transform(X)
        except TypeError:
            emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed, n_iter=1000).fit_transform(X)
    elif reducer == "umap" and HAS_UMAP:
        emb = umap.UMAP(n_components=2, n_neighbors=min(30, max(2, len(X) - 1)), min_dist=0.1, metric="euclidean", random_state=seed).fit_transform(X)
    else:
        emb = PCA(n_components=2, random_state=seed).fit_transform(X)

    plot_df = patch_df.copy()
    plot_df["composition_x"] = emb[:, 0]
    plot_df["composition_y"] = emb[:, 1]
    plot_df.to_csv(out_dir / "patch_composition_embedding.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    embv = emb.astype(np.float32)
    plot_category_scatter(axes[0], embv, plot_df["dominant_expert"].values, "Patch expert composition: dominant expert", prefix="E", size=8.0)
    sc = plot_continuous_scatter(axes[1], embv, plot_df["purity"].values, "Patch expert composition: purity", cmap="inferno", size=8.0)
    plot_category_scatter(axes[2], embv, plot_df["label"].values, "Patch expert composition: slide label", prefix="y=", size=8.0)
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "patch_composition_embedding_panel.png", dpi=240)
    plt.close(fig)


def save_patch_composition_galleries(
    out_dir: Path,
    patch_df: pd.DataFrame,
    raw_dir: str,
    patch_size: int,
    n_experts: int,
    topk: int,
    purity_thr: float,
) -> None:
    ensure_dir(out_dir)
    rows = []
    patch_cache: Dict[Tuple[str, int, int], Image.Image] = {}

    for e in range(n_experts):
        sub = patch_df[(patch_df["dominant_expert"] == e) & (patch_df["purity"] >= purity_thr)].copy()
        if len(sub) == 0:
            sub = patch_df[patch_df["dominant_expert"] == e].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(["purity", f"frac_E{e}", "n_sampled_tokens"], ascending=False).head(topk)
        imgs = []
        for _, r in sub.iterrows():
            img = _read_patch_cached(patch_cache, raw_dir, str(r["slide_id"]), int(r["coord_x"]), int(r["coord_y"]), patch_size)
            imgs.append(img)
            row = r.to_dict()
            row["gallery"] = f"E{e}_dominant_patches"
            rows.append(row)
        if imgs:
            make_montage(imgs, tile_size=224, n_cols=4).save(out_dir / f"E{e}_dominant_patches.png")

    mixed = patch_df.sort_values(["composition_entropy", "purity"], ascending=[False, True]).head(topk)
    imgs = []
    for _, r in mixed.iterrows():
        img = _read_patch_cached(patch_cache, raw_dir, str(r["slide_id"]), int(r["coord_x"]), int(r["coord_y"]), patch_size)
        imgs.append(img)
        row = r.to_dict()
        row["gallery"] = "mixed_high_entropy_patches"
        rows.append(row)
    if imgs:
        make_montage(imgs, tile_size=224, n_cols=4).save(out_dir / "mixed_high_entropy_patches.png")

    pd.DataFrame(rows).to_csv(out_dir / "patch_composition_gallery_index.csv", index=False)


def save_patch_composition_overlays(
    out_dir: Path,
    patch_df: pd.DataFrame,
    raw_dir: str,
    patch_size: int,
    n_experts: int,
    max_slides: int,
    thumb_width: int,
    alpha: float,
    point_size: float,
) -> None:
    ensure_dir(out_dir)
    slide_ids = patch_df["slide_id"].drop_duplicates().tolist()[:max_slides]
    cmap = plt.get_cmap("tab10")
    for slide_id in slide_ids:
        sub = patch_df[patch_df["slide_id"] == slide_id].copy()
        slide_path = _find_wsi_path_for_overlay(raw_dir, str(slide_id))
        if slide_path is None:
            continue
        slide = openslide.OpenSlide(slide_path)
        try:
            W, H = slide.dimensions
            thumb_height = max(1, int(round(float(thumb_width) * H / max(1, W))))
            thumb = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
        finally:
            slide.close()
        sx = thumb.size[0] / float(W)
        sy = thumb.size[1] / float(H)
        ox = (sub["coord_x"].values.astype(np.float32) + 0.5 * patch_size) * sx
        oy = (sub["coord_y"].values.astype(np.float32) + 0.5 * patch_size) * sy

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.imshow(thumb)
        for e in range(n_experts):
            m = sub["dominant_expert"].values == e
            if m.sum() == 0:
                continue
            ax.scatter(ox[m], oy[m], s=point_size * (0.4 + sub["purity"].values[m]), alpha=alpha, color=cmap(e % 10), label=f"E{e} n={int(m.sum())}", linewidths=0)
        ax.set_title(f"{slide_id}: patch expert composition dominant expert")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2.5, fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{slide_id}_patch_composition_dominant_overlay.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.imshow(thumb)
        sc = ax.scatter(ox, oy, c=sub["purity"].values.astype(np.float32), s=point_size, cmap="inferno", vmin=0.0, vmax=1.0, alpha=alpha, linewidths=0)
        ax.set_title(f"{slide_id}: patch expert composition purity")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{slide_id}_patch_composition_purity_overlay.png", dpi=220)
        plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("MoE backbone token-level behavior visualization")

    # Backend selection.
    parser.add_argument("--backend", type=str, default="dino", choices=["dino", "factory"], help="dino: original DINO-MoE path; factory: UNI/UNI2H/Virchow2/OpenCLIP MoE via build_feature_extractor.")

    # DINO-only args.
    parser.add_argument("--config", type=str, default="", help="Required only when --backend dino. Stage2 yaml config used to build DINO-MoE.")
    parser.add_argument("--moe_ckpt", type=str, default="", help="Required only when --backend dino. Saved DINO-MoE checkpoint.")

    # Shared data args.
    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_slides", type=int, default=40)
    parser.add_argument("--select_mode", type=str, default="random_balanced", choices=["csv_order", "random", "random_balanced"])
    parser.add_argument("--max_patches_per_slide", type=int, default=512)
    parser.add_argument("--token_sample_per_slide", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)

    # Factory backend args.
    parser.add_argument("--adapted_encoder_name", type=str, default="dino_moe", help="Only used when --backend factory. Examples: uni_moe, uni2_h_moe, virchow2_moe, openclip_moe.")
    parser.add_argument("--stage2_ckpt", type=str, default="", help="Required when --backend factory.")
    parser.add_argument("--virchow2_weight", type=str, default="")
    parser.add_argument("--uni_weight", type=str, default="")
    parser.add_argument("--uni2_weight", type=str, default="")
    parser.add_argument("--openclip_model_name", type=str, default="ViT-B-16")
    parser.add_argument("--openclip_weight", type=str, default="")
    parser.add_argument("--openclip_precision", type=str, default="fp16")
    parser.add_argument("--no_openclip_normalize", action="store_true")

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

    # Analysis args.
    parser.add_argument("--reducer", type=str, default="umap", choices=["umap", "pca", "tsne"])
    parser.add_argument("--n_clusters", type=int, default=12)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_max_points", type=int, default=30000)
    parser.add_argument("--save_tsne_comparison", action="store_true")

    parser.add_argument("--compare_frozen_dino", action="store_true")
    parser.add_argument("--save_token_galleries", action="store_true")
    parser.add_argument("--save_expert_cluster_galleries", action="store_true", help="Save expert-cluster morphology galleries. Recommended as Figure C candidate.")
    parser.add_argument(
        "--save_expert_cluster_stats",
        action="store_true",
        help="Save expert-cluster entropy, effective cluster number, and pairwise JSD statistics."
    )
    parser.add_argument("--ec_gallery_layer", type=str, default="last_moe", choices=["last_moe", "final"], help="Which token feature cluster to use for expert-cluster galleries.")
    parser.add_argument("--ec_gallery_top_pairs", type=int, default=8)
    parser.add_argument("--ec_gallery_topk", type=int, default=16)
    parser.add_argument("--ec_gallery_min_tokens", type=int, default=50)
    parser.add_argument("--ec_gallery_no_diverse_slides", action="store_true")

    parser.add_argument("--save_patch_composition", action="store_true", help="Save patch-level expert composition table, plots, overlays and galleries.")
    parser.add_argument("--patch_composition_purity_thr", type=float, default=0.70)
    parser.add_argument("--patch_composition_topk", type=int, default=24)
    parser.add_argument("--patch_composition_max_overlay_slides", type=int, default=12)
    parser.add_argument("--patch_composition_overlay_thumb_width", type=int, default=2200)
    parser.add_argument("--patch_composition_overlay_alpha", type=float, default=0.75)
    parser.add_argument("--patch_composition_overlay_point_size", type=float, default=12.0)
    parser.add_argument("--patch_composition_reducer", type=str, default="tsne", choices=["tsne", "umap", "pca"])

    parser.add_argument("--save_slide_expert_maps", action="store_true", help="Save per-slide spatial expert routing maps on blank coordinates.")
    parser.add_argument("--save_slide_expert_overlays", action="store_true", help="Save per-slide expert routing overlays on WSI thumbnails.")
    parser.add_argument("--overlay_aggregate", type=str, default="patch_mean", choices=["token", "patch_mean", "patch_max"])
    parser.add_argument("--overlay_thumb_width", type=int, default=2200)
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--overlay_point_size", type=float, default=8.0)
    parser.add_argument("--max_slide_maps", type=int, default=12)
    parser.add_argument("--max_points_per_slide_map", type=int, default=30000)
    parser.add_argument("--gallery_topk", type=int, default=24)

    parser.add_argument("--strict_load", action="store_true", help="Use strict=True for loading the DINO-MoE checkpoint. Default is strict=False.")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out_dir = Path(args.out_dir)
    for p in [
        out_dir,
        out_dir / "cache",
        out_dir / "figures",
        out_dir / "heatmaps",
        out_dir / "expert_cluster_stats",
        out_dir / "frozen_compare",
        out_dir / "token_galleries",
    ]:
        ensure_dir(p)

    selected_df = prepare_slides_df(args.slides_csv, args.n_slides, args.select_mode, args.seed)
    selected_df.to_csv(out_dir / "selected_slides.csv", index=False)
    print(f"[Select] {len(selected_df)} slides")
    print(selected_df["label"].value_counts().sort_index())

    model: Optional[MoEEncoder] = None
    extractor: Optional[Any] = None
    cfg: Optional[Dict] = None
    real_moe_blocks: List[int] = []
    frozen_model: Optional[DINOv2Encoder] = None
    frozen_target_block: Optional[int] = None

    if args.backend == "dino":
        if not args.config:
            raise ValueError("--config is required when --backend dino")
        if not args.moe_ckpt:
            raise ValueError("--moe_ckpt is required when --backend dino")
        model, cfg, real_moe_blocks = load_dino_moe_model(args.config, args.moe_ckpt, device=device, strict=args.strict_load)
        frozen_target_block = real_moe_blocks[-1] if real_moe_blocks else None
        if args.compare_frozen_dino:
            print("[Build] frozen DINO from config")
            frozen_model = load_frozen_dino_from_config(cfg, device=device)
            print(f"[Frozen] compare block={frozen_target_block} and final layer")
    elif args.backend == "factory":
        extractor = load_factory_moe_extractor(args)
        if args.compare_frozen_dino:
            print("[WARN] --compare_frozen_dino is ignored when --backend factory")
            args.compare_frozen_dino = False
    else:
        raise ValueError(f"Unknown backend={args.backend}")

    transform = build_transform(args.image_size)

    token_rows: List[pd.DataFrame] = []
    last_feat_list: List[np.ndarray] = []
    final_feat_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    frozen_last_list: List[np.ndarray] = []
    frozen_final_list: List[np.ndarray] = []

    for _, row in selected_df.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        print(f"[Process] {slide_id} | label={label}")

        cache_tag = args.backend
        if args.backend == "factory":
            cache_tag += f"_{args.adapted_encoder_name}"
        cache_path = out_dir / "cache" / f"{slide_id}_{cache_tag}_tokens.npz"

        need_recompute = False
        if (not args.no_cache) and cache_path.exists() and (not args.overwrite_cache):
            print(f"[Cache] load {cache_path}")
            c = np.load(cache_path, allow_pickle=True)
            out = {k: c[k] for k in c.files}
            if args.compare_frozen_dino and ("frozen_last_feat" not in out or "frozen_final_feat" not in out):
                print("[Cache] frozen comparison requested but frozen features missing; recomputing this slide cache")
                need_recompute = True
        else:
            need_recompute = True

        if need_recompute:
            out = extract_one_slide_tokens(
                backend=args.backend,
                model=model,
                extractor=extractor,
                frozen_model=frozen_model,
                slide_path=find_wsi_path(args.raw_dir, slide_id),
                h5_path=find_h5_path(args.h5_dir, slide_id),
                transform=transform,
                device=device,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                max_patches=args.max_patches_per_slide,
                token_sample_per_slide=args.token_sample_per_slide,
                seed=stable_slide_seed(args.seed, slide_id),
                frozen_target_block=frozen_target_block,
            )
            if not args.no_cache:
                np.savez_compressed(cache_path, **out)
                print(f"[Cache] saved {cache_path}")

        probs = out["token_expert_probs"].astype(np.float32)
        probs = np.clip(probs, 0.0, None)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
        expert_id = np.argmax(probs, axis=1).astype(np.int64)
        entropy = entropy_np(probs).astype(np.float32)
        margin = top1_top2_margin_np(probs).astype(np.float32)
        n_tok = len(probs)
        n_experts = probs.shape[1]

        df = pd.DataFrame({
            "slide_id": [slide_id] * n_tok,
            "label": [label] * n_tok,
            "coord_x": out["coord_x"].astype(np.int64),
            "coord_y": out["coord_y"].astype(np.int64),
            "patch_index_in_slide": out["patch_index_in_slide"].astype(np.int64),
            "token_index_in_patch": out["token_index_in_patch"].astype(np.int64),
            "expert_id": expert_id,
            "expert_entropy": entropy,
            "expert_margin": margin,
            "expert_prob": probs[np.arange(n_tok), expert_id],
        })
        for e in range(n_experts):
            df[f"expert_prob_{e}"] = probs[:, e]
        token_rows.append(df)
        last_feat_list.append(out["token_last_moe_feat"].astype(np.float32))
        final_feat_list.append(out["token_final_feat"].astype(np.float32))
        prob_list.append(probs)
        if args.compare_frozen_dino:
            frozen_last_list.append(out["frozen_last_feat"].astype(np.float32))
            frozen_final_list.append(out["frozen_final_feat"].astype(np.float32))

    token_df = pd.concat(token_rows, axis=0).reset_index(drop=True)
    last_feat = np.concatenate(last_feat_list, axis=0)
    final_feat = np.concatenate(final_feat_list, axis=0)
    probs_all = np.concatenate(prob_list, axis=0)
    n_experts = probs_all.shape[1]
    n_tokens_per_patch_est = int(token_df["token_index_in_patch"].max()) + 1

    print("[Analyze] MoE token-level feature spaces")
    for layer_name, feat, seed_offset, title in [
        ("last_moe", last_feat, 0, f"{args.backend.upper()} MoE last-MoE token features"),
        ("final", final_feat, 1, f"{args.backend.upper()} MoE final token features"),
    ]:
        emb, emb_idx = compute_embedding(feat, args.reducer, args.seed + seed_offset, args.umap_n_neighbors, args.umap_min_dist, args.tsne_perplexity, args.tsne_max_points)
        cluster = cluster_features(feat, args.n_clusters, args.seed + seed_offset)
        token_df[f"{layer_name}_cluster"] = cluster
        if emb_idx is None:
            token_df[f"{layer_name}_x"] = emb[:, 0]
            token_df[f"{layer_name}_y"] = emb[:, 1]
            save_token_scatter_panel(out_dir / "figures", token_df, layer_name, title)
        else:
            plot_df = token_df.iloc[emb_idx].copy().reset_index(drop=True)
            plot_df[f"{layer_name}_x"] = emb[:, 0]
            plot_df[f"{layer_name}_y"] = emb[:, 1]
            plot_df[f"{layer_name}_cluster"] = cluster[emb_idx]
            save_token_scatter_panel(out_dir / "figures", plot_df, layer_name, title)
            plot_df.to_csv(out_dir / "figures" / f"{layer_name}_tsne_plot_points.csv", index=False)

        tables = expert_cluster_tables(probs_all, cluster, args.n_clusters)

        for name, mat in tables.items():
            if name in ["hard_counts", "cluster_counts"]:
                continue

            save_heatmap(
                mat,
                out_dir / "heatmaps" / f"{layer_name}_{name}.png",
                title=f"{layer_name}: {name}",
                xlabel="token cluster",
                ylabel="expert",
                xticklabels=[f"c{i}" for i in range(args.n_clusters)],
                yticklabels=[f"E{i}" for i in range(n_experts)],
                cmap="Blues",
                vmin=0.0,
                vmax=1.0 if "given" in name else None,
            )
            np.save(out_dir / "heatmaps" / f"{layer_name}_{name}.npy", mat)

        pd.DataFrame(
            tables["hard_counts"],
            index=[f"E{i}" for i in range(n_experts)],
            columns=[f"c{i}" for i in range(args.n_clusters)],
        ).to_csv(out_dir / "heatmaps" / f"{layer_name}_hard_counts.csv")

        pd.DataFrame({
            "cluster": [f"c{i}" for i in range(args.n_clusters)],
            "count": tables["cluster_counts"],
        }).to_csv(out_dir / "heatmaps" / f"{layer_name}_cluster_counts.csv", index=False)

        if args.save_expert_cluster_stats:
            print(f"[Analyze] expert-cluster specialization stats for {layer_name}")

            ec_summary = save_expert_cluster_specialization_stats(
                out_dir=out_dir / "expert_cluster_stats",
                tables=tables,
                layer_name=layer_name,
                n_experts=n_experts,
                n_clusters=args.n_clusters,
            )

            print(
                f"[ExpertClusterStats] {layer_name}: "
                f"mean_JSD={ec_summary['mean_pairwise_expert_jsd']:.4f}, "
                f"mean_entropy_norm={ec_summary['mean_expert_cluster_entropy_norm']:.4f}, "
                f"mean_effective_clusters={ec_summary['mean_effective_clusters']:.2f}"
            )
    if args.compare_frozen_dino:
        print("[Analyze] frozen DINO token feature spaces")
        frozen_last = np.concatenate(frozen_last_list, axis=0)
        frozen_final = np.concatenate(frozen_final_list, axis=0)
        for layer_name, feat, seed_offset, title in [
            ("frozen_last", frozen_last, 11, f"Frozen DINO block {frozen_target_block} token features"),
            ("frozen_final", frozen_final, 12, "Frozen DINO final token features"),
        ]:
            emb, emb_idx = compute_embedding(feat, args.reducer, args.seed + seed_offset, args.umap_n_neighbors, args.umap_min_dist, args.tsne_perplexity, args.tsne_max_points)
            cluster = cluster_features(feat, args.n_clusters, args.seed + seed_offset)
            token_df[f"{layer_name}_cluster"] = cluster
            if emb_idx is None:
                token_df[f"{layer_name}_x"] = emb[:, 0]
                token_df[f"{layer_name}_y"] = emb[:, 1]
                save_frozen_scatter_panel(out_dir / "frozen_compare", token_df, layer_name, title)
                token_df.to_csv(
                    out_dir / "frozen_compare" / f"{layer_name}_plot_points.csv",
                    index=False  
                )
            else:
                plot_df = token_df.iloc[emb_idx].copy().reset_index(drop=True)
                plot_df[f"{layer_name}_x"] = emb[:, 0]
                plot_df[f"{layer_name}_y"] = emb[:, 1]
                plot_df[f"{layer_name}_cluster"] = cluster[emb_idx]
                save_frozen_scatter_panel(out_dir / "frozen_compare", plot_df, layer_name, title)

                plot_df.to_csv(
                    out_dir / "frozen_compare" / f"{layer_name}_tsne_plot_points.csv",
                    index=False
                )
        token_df.to_csv(out_dir / "frozen_compare" / "token_level_with_frozen_embeddings.csv", index=False)

    if args.save_tsne_comparison:
        print("[Analyze] compact t-SNE comparison figure")
        rng = np.random.default_rng(args.seed + 202)
        n_plot = min(args.tsne_max_points, len(token_df))
        plot_idx = rng.choice(len(token_df), size=n_plot, replace=False) if len(token_df) > n_plot else np.arange(len(token_df))
        panels: List[Tuple[str, np.ndarray, np.ndarray, str]] = []
        if args.compare_frozen_dino and len(frozen_last_list) > 0:
            frozen_last_for_tsne = np.concatenate(frozen_last_list, axis=0)
            emb = compute_tsne_for_fixed_subset(frozen_last_for_tsne, plot_idx, args.seed + 301, args.tsne_perplexity)
            panels.append((f"Frozen DINO block {frozen_target_block}", emb, token_df.iloc[plot_idx]["label"].values, "label"))
        emb = compute_tsne_for_fixed_subset(last_feat, plot_idx, args.seed + 302, args.tsne_perplexity)
        panels.append(("MoE last-MoE by expert", emb, token_df.iloc[plot_idx]["expert_id"].values, "expert"))
        panels.append(("MoE last-MoE by cluster", emb, token_df.iloc[plot_idx]["last_moe_cluster"].values, "cluster"))
        save_compact_embedding_comparison(out_dir / "figures" / "compact_tsne_comparison.png", panels=panels, point_size=2.0)

    token_df.to_csv(out_dir / "token_level_moe_analysis.csv", index=False)
    save_expert_usage_bar(token_df["expert_id"].values, n_experts, out_dir / "figures" / "expert_usage_bar.png")
    save_hist(token_df["expert_entropy"].values, out_dir / "figures" / "dispatch_entropy_hist.png", "Token dispatch entropy", "entropy")
    save_hist(token_df["expert_margin"].values, out_dir / "figures" / "dispatch_margin_hist.png", "Token top1-top2 dispatch margin", "margin")

    if args.save_patch_composition:
        print("[Analyze] patch-level expert composition from token routing")
        patch_comp_dir = out_dir / "patch_composition"
        ensure_dir(patch_comp_dir)
        patch_df = build_patch_expert_composition(token_df, n_experts=n_experts)
        patch_df.to_csv(patch_comp_dir / "patch_expert_composition.csv", index=False)
        save_patch_composition_embedding(patch_comp_dir, patch_df, n_experts, args.seed, reducer=args.patch_composition_reducer, tsne_perplexity=args.tsne_perplexity)
        save_patch_composition_galleries(patch_comp_dir / "galleries", patch_df, args.raw_dir, args.patch_size, n_experts, args.patch_composition_topk, args.patch_composition_purity_thr)
        save_patch_composition_overlays(patch_comp_dir / "overlays", patch_df, args.raw_dir, args.patch_size, n_experts, args.patch_composition_max_overlay_slides, args.patch_composition_overlay_thumb_width, args.patch_composition_overlay_alpha, args.patch_composition_overlay_point_size)

    if args.save_token_galleries:
        save_token_galleries(out_dir / "token_galleries", token_df, args.raw_dir, args.patch_size, n_tokens_per_patch_est, n_experts, args.gallery_topk)

    if args.save_expert_cluster_galleries:
        cluster_col = f"{args.ec_gallery_layer}_cluster"
        save_expert_cluster_morphology_galleries(
            out_dir=out_dir / "expert_cluster_galleries",
            token_df=token_df,
            raw_dir=args.raw_dir,
            patch_size=args.patch_size,
            n_tokens_per_patch=n_tokens_per_patch_est,
            n_experts=n_experts,
            cluster_col=cluster_col,
            top_pairs=args.ec_gallery_top_pairs,
            topk_per_pair=args.ec_gallery_topk,
            min_tokens_per_pair=args.ec_gallery_min_tokens,
            diverse_slides=not args.ec_gallery_no_diverse_slides,
        )

    if args.save_slide_expert_maps:
        save_slide_expert_routing_maps(out_dir / "per_slide_expert_maps", token_df, args.patch_size, n_tokens_per_patch_est, n_experts, args.max_slide_maps, args.max_points_per_slide_map)

    if args.save_slide_expert_overlays:
        save_slide_expert_thumbnail_overlays(out_dir / "per_slide_expert_overlays", token_df, args.raw_dir, args.patch_size, n_tokens_per_patch_est, n_experts, args.max_slide_maps, args.overlay_aggregate, args.overlay_thumb_width, args.overlay_alpha, args.overlay_point_size)

    metrics: Dict[str, Any] = {
        "backend": args.backend,
        "adapted_encoder_name": args.adapted_encoder_name if args.backend == "factory" else "dino_moe",
        "n_sampled_tokens": int(len(token_df)),
        "n_experts": int(n_experts),
        "mean_dispatch_entropy": float(token_df["expert_entropy"].mean()),
        "mean_dispatch_margin": float(token_df["expert_margin"].mean()),
        "expert_counts": {f"E{i}": int((token_df["expert_id"].values == i).sum()) for i in range(n_experts)},
        "has_umap": HAS_UMAP,
        "frozen_target_block": None if frozen_target_block is None else int(frozen_target_block),
    }
    for name, feat, cluster_col in [("last_moe", last_feat, "last_moe_cluster"), ("final", final_feat, "final_cluster")]:
        try:
            sample_n = min(5000, len(feat))
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(feat), size=sample_n, replace=False) if len(feat) > sample_n else np.arange(len(feat))
            metrics[f"{name}_silhouette_cosine"] = float(silhouette_score(l2_normalize_np(feat[idx]), token_df[cluster_col].values[idx], metric="cosine"))
        except Exception as e:
            metrics[f"{name}_silhouette_cosine_error"] = str(e)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[Done] saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
