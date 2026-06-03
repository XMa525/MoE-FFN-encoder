#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.moe_encoder import MoEEncoder

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# Basic utils
# =========================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_slide_seed(base_seed: int, slide_id: str) -> int:
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(base_seed) + (int(h[:8], 16) % 100000)


def safe_slide_name(slide_id: str) -> str:
    return (
        str(slide_id)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def entropy_np(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


# =========================================================
# Paths / WSI
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


def make_wsi_thumbnail(slide_path: str, thumb_width: int) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    slide = openslide.OpenSlide(slide_path)
    try:
        W, H = slide.dimensions
        thumb_height = max(1, int(round(float(thumb_width) * H / max(1, W))))
        thumb = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
    finally:
        slide.close()

    sx = thumb.size[0] / float(W)
    sy = thumb.size[1] / float(H)
    return thumb, (W, H), (sx, sy)


# =========================================================
# Slide selection
# =========================================================
def prepare_slides_df(
    slides_csv: str,
    slide_ids: Sequence[str],
    labels: Sequence[int],
    n_slides_per_label: int,
    select_mode: str,
    seed: int,
) -> pd.DataFrame:
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
            print("[WARN] slides_csv has no label / slide_binary_label / y_true. Use label=0.")

    df = df.drop_duplicates("slide_id").reset_index(drop=True)
    df["slide_id"] = df["slide_id"].astype(str)
    df["label"] = df["label"].astype(int)

    explicit = [str(x) for x in slide_ids]
    if explicit:
        out = df[df["slide_id"].isin(explicit)].copy()
        missing = [x for x in explicit if x not in set(out["slide_id"])]
        for m in missing:
            print(f"[WARN] explicit slide_id not found in slides_csv: {m}")
        return out.reset_index(drop=True)

    if labels:
        df = df[df["label"].isin(labels)].copy()

    if select_mode == "csv_order":
        parts = []
        for _, sub in df.groupby("label"):
            parts.append(sub.head(n_slides_per_label))
        if not parts:
            return df.head(0)
        return pd.concat(parts, axis=0).reset_index(drop=True)

    if select_mode == "random_balanced":
        rng = np.random.default_rng(seed)
        parts = []
        for _, sub in df.groupby("label"):
            if len(sub) == 0:
                continue
            n = min(n_slides_per_label, len(sub))
            parts.append(sub.sample(n=n, random_state=int(rng.integers(1, 1_000_000))))
        if not parts:
            return df.head(0)
        return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    raise ValueError(f"Unknown select_mode={select_mode}")


# =========================================================
# Model loading / routing
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


def load_dino_moe_model(config_path: str, ckpt_path: str, device: str, strict: bool = False) -> Tuple[MoEEncoder, Dict]:
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
    return model, cfg


def normalize_dispatch(dispatch: torch.Tensor) -> torch.Tensor:
    dispatch = dispatch.float().clamp_min(0.0)
    return dispatch / dispatch.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def parse_dispatch_weight(gate_info: Any, B: int, seq_len: int) -> torch.Tensor:
    if not isinstance(gate_info, dict):
        raise TypeError(f"Expected gate_info dict, got {type(gate_info)}")
    if "dispatch_weight" not in gate_info:
        raise KeyError(f"gate_info does not contain dispatch_weight. Existing keys={list(gate_info.keys())}")

    dispatch = gate_info["dispatch_weight"]
    if not torch.is_tensor(dispatch):
        raise TypeError(f"dispatch_weight must be tensor, got {type(dispatch)}")
    if dispatch.ndim != 2:
        raise ValueError(f"dispatch_weight expected [B*seq_len, E], got {tuple(dispatch.shape)}")

    total_tokens, _ = dispatch.shape
    if total_tokens != B * seq_len:
        raise ValueError(f"dispatch first dim mismatch: got {total_tokens}, expected {B * seq_len}")

    return normalize_dispatch(dispatch.reshape(B, seq_len, -1))


@torch.no_grad()
def run_dino_moe_batch(model: MoEEncoder, images: torch.Tensor) -> torch.Tensor:
    model_out = model(images, return_gates=True, return_features=True, is_eval=True)
    if not (isinstance(model_out, (tuple, list)) and len(model_out) == 4):
        raise RuntimeError("Expected model(images, return_gates=True, return_features=True) -> 4 outputs")

    final_tokens, gate_info_list, _, _ = model_out
    if len(gate_info_list) == 0:
        raise RuntimeError("gate_info_list is empty")

    B, seq_len, _ = final_tokens.shape
    probs_all = parse_dispatch_weight(gate_info_list[-1], B=B, seq_len=seq_len)

    # DINO: remove CLS token. [B, T, E]
    return probs_all[:, 1:, :]


# =========================================================
# Full patch composition extraction
# =========================================================
@torch.no_grad()
def extract_patch_composition_for_slide(
    model: MoEEncoder,
    slide_path: str,
    h5_path: str,
    transform,
    device: str,
    patch_size: int,
    batch_size: int,
    max_patches_per_slide: int,
    seed: int,
) -> pd.DataFrame:
    coords = read_coords_from_h5(h5_path)

    if max_patches_per_slide is not None and max_patches_per_slide > 0 and len(coords) > max_patches_per_slide:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_patches_per_slide, replace=False)
        coords = coords[idx]

    slide = openslide.OpenSlide(slide_path)
    rows: List[Dict[str, Any]] = []

    try:
        for start in tqdm(
            range(0, len(coords), batch_size),
            total=math.ceil(len(coords) / batch_size),
            desc=f"  Extract[{Path(slide_path).stem[:24]}]",
            leave=False,
        ):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]

            imgs = []
            for xy in batch_coords.tolist():
                img = read_patch_from_wsi(slide, (int(xy[0]), int(xy[1])), patch_size=patch_size, read_level=0)
                imgs.append(transform(img))

            x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            token_probs = run_dino_moe_batch(model, x)  # [B, T, E]

            patch_probs = token_probs.mean(dim=1)
            patch_probs = patch_probs / patch_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            probs_np = patch_probs.detach().cpu().numpy().astype(np.float32)
            n_experts = probs_np.shape[1]
            dominant = np.argmax(probs_np, axis=1).astype(np.int64)
            purity = probs_np.max(axis=1).astype(np.float32)
            comp_entropy = entropy_np(probs_np).astype(np.float32)

            for i in range(len(batch_coords)):
                row: Dict[str, Any] = {
                    "coord_x": int(batch_coords[i, 0]),
                    "coord_y": int(batch_coords[i, 1]),
                    "patch_index_in_slide": int(start + i),
                    "dominant_expert": int(dominant[i]),
                    "purity": float(purity[i]),
                    "composition_entropy": float(comp_entropy[i]),
                    "n_tokens": int(token_probs.shape[1]),
                }
                for e in range(n_experts):
                    row[f"frac_E{e}"] = float(probs_np[i, e])
                rows.append(row)

    finally:
        slide.close()

    return pd.DataFrame(rows)


def get_num_experts(patch_df: pd.DataFrame) -> int:
    ids = []
    for c in patch_df.columns:
        if c.startswith("frac_E"):
            try:
                ids.append(int(c.replace("frac_E", "")))
            except Exception:
                pass
    if not ids:
        raise ValueError("Cannot infer experts from frac_E* columns")
    return max(ids) + 1


# =========================================================
# PAMOE-style assignment map rendering
# =========================================================
def expert_colors_rgb_high_contrast(n_experts: int) -> List[Tuple[int, int, int]]:
    base = [
        (31, 119, 180),    # E0 blue
        (255, 127, 14),    # E1 orange
        (44, 160, 44),     # E2 green
        (214, 39, 160),    # E3 magenta
        (214, 39, 40),     # E4 red
        (23, 190, 207),    # E5 cyan
        (188, 189, 34),    # E6 olive
        (227, 119, 194),   # E7 pink
        (127, 127, 127),   # E8 gray
        (140, 86, 75),     # E9 brown
    ]
    return [base[i % len(base)] for i in range(n_experts)]


def compute_assignment_canvas_geometry(
    sub: pd.DataFrame,
    patch_size: int,
    max_width: int = 2200,
    pad: int = 20,
) -> Tuple[int, int, float, int, int]:
    min_x = int(sub["coord_x"].min())
    min_y = int(sub["coord_y"].min())
    max_x = int(sub["coord_x"].max() + patch_size)
    max_y = int(sub["coord_y"].max() + patch_size)

    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)

    scale = (max_width - 2 * pad) / float(w)
    canvas_w = int(round(w * scale)) + 2 * pad
    canvas_h = int(round(h * scale)) + 2 * pad
    return canvas_w, canvas_h, scale, min_x, min_y


def render_expert_assignment_grid(
    sub: pd.DataFrame,
    patch_size: int,
    n_experts: int,
    max_width: int = 2200,
    pad: int = 20,
    background: Tuple[int, int, int] = (255, 255, 255),
    draw_grid: bool = True,
    grid_color: Tuple[int, int, int] = (35, 35, 35),
    grid_width: int = 1,
    purity_alpha: bool = False,
    discard_low_purity: bool = False,
    purity_thr: float = 0.0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    canvas_w, canvas_h, scale, min_x, min_y = compute_assignment_canvas_geometry(
        sub=sub,
        patch_size=patch_size,
        max_width=max_width,
        pad=pad,
    )

    img = Image.new("RGB", (canvas_w, canvas_h), color=background)
    draw = ImageDraw.Draw(img, "RGBA")
    colors = expert_colors_rgb_high_contrast(n_experts)

    for _, r in sub.iterrows():
        x0 = pad + (float(r["coord_x"]) - min_x) * scale
        y0 = pad + (float(r["coord_y"]) - min_y) * scale
        x1 = pad + (float(r["coord_x"]) - min_x + patch_size) * scale
        y1 = pad + (float(r["coord_y"]) - min_y + patch_size) * scale

        e = int(r["dominant_expert"])
        purity = float(r["purity"])

        if discard_low_purity and purity < purity_thr:
            rgb = (95, 95, 95)
            alpha = 235
        else:
            rgb = colors[e]
            alpha = int(255 * (0.45 + 0.55 * purity)) if purity_alpha else 255

        outline = grid_color + (255,) if draw_grid else None
        draw.rectangle([x0, y0, x1, y1], fill=rgb + (alpha,), outline=outline, width=grid_width)

    meta = {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "scale": scale,
        "min_x": min_x,
        "min_y": min_y,
        "pad": pad,
    }
    return img, meta


def render_purity_assignment_grid(
    sub: pd.DataFrame,
    patch_size: int,
    max_width: int = 2200,
    pad: int = 20,
    draw_grid: bool = True,
    grid_color: Tuple[int, int, int] = (35, 35, 35),
    grid_width: int = 1,
) -> Tuple[Image.Image, Dict[str, Any]]:
    canvas_w, canvas_h, scale, min_x, min_y = compute_assignment_canvas_geometry(
        sub=sub,
        patch_size=patch_size,
        max_width=max_width,
        pad=pad,
    )

    img = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    cmap = plt.get_cmap("inferno")

    for _, r in sub.iterrows():
        x0 = pad + (float(r["coord_x"]) - min_x) * scale
        y0 = pad + (float(r["coord_y"]) - min_y) * scale
        x1 = pad + (float(r["coord_x"]) - min_x + patch_size) * scale
        y1 = pad + (float(r["coord_y"]) - min_y + patch_size) * scale

        purity = float(np.clip(r["purity"], 0.0, 1.0))
        rr, gg, bb, _ = cmap(purity)
        rgb = (int(rr * 255), int(gg * 255), int(bb * 255))
        outline = grid_color + (255,) if draw_grid else None
        draw.rectangle([x0, y0, x1, y1], fill=rgb + (255,), outline=outline, width=grid_width)

    meta = {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "scale": scale,
        "min_x": min_x,
        "min_y": min_y,
        "pad": pad,
    }
    return img, meta


def select_interesting_roi(
    sub: pd.DataFrame,
    patch_size: int,
    n_experts: int,
    roi_size_patches: int = 18,
    stride_patches: int = 6,
    min_patches_in_roi: int = 30,
) -> Tuple[int, int, int, int]:
    xs = sub["coord_x"].values.astype(np.int64)
    ys = sub["coord_y"].values.astype(np.int64)

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    roi_w = roi_size_patches * patch_size
    roi_h = roi_size_patches * patch_size
    stride = stride_patches * patch_size

    best_score = -1.0
    best_roi = (min_x, min_y, min_x + roi_w, min_y + roi_h)

    for x0 in range(min_x, max_x + 1, stride):
        for y0 in range(min_y, max_y + 1, stride):
            x1 = x0 + roi_w
            y1 = y0 + roi_h
            m = (
                (sub["coord_x"].values >= x0)
                & (sub["coord_x"].values < x1)
                & (sub["coord_y"].values >= y0)
                & (sub["coord_y"].values < y1)
            )
            roi = sub[m]
            if len(roi) < min_patches_in_roi:
                continue

            experts = roi["dominant_expert"].values.astype(np.int64)
            counts = np.array([(experts == e).sum() for e in range(n_experts)], dtype=np.float32)
            p = counts / max(1.0, counts.sum())
            p_safe = np.clip(p, 1e-8, 1.0)
            expert_entropy = float(-(p_safe * np.log(p_safe)).sum() / math.log(max(2, n_experts)))
            mean_purity = float(roi["purity"].mean())
            coverage = min(1.0, len(roi) / float(roi_size_patches * roi_size_patches))
            score = 0.55 * expert_entropy + 0.30 * mean_purity + 0.15 * coverage

            if score > best_score:
                best_score = score
                best_roi = (x0, y0, x1, y1)

    return best_roi


def _roi_to_assignment_pixels(geom: Dict[str, Any], roi_xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi_xyxy
    scale = float(geom["scale"])
    min_x = int(geom["min_x"])
    min_y = int(geom["min_y"])
    pad = int(geom["pad"])

    ix0 = int(round(pad + (x0 - min_x) * scale))
    iy0 = int(round(pad + (y0 - min_y) * scale))
    ix1 = int(round(pad + (x1 - min_x) * scale))
    iy1 = int(round(pad + (y1 - min_y) * scale))
    return ix0, iy0, ix1, iy1


def crop_assignment_roi(
    full_img: Image.Image,
    geom: Dict[str, Any],
    roi_xyxy: Tuple[int, int, int, int],
    out_size: int = 640,
) -> Image.Image:
    crop = full_img.crop(_roi_to_assignment_pixels(geom, roi_xyxy)).convert("RGB")
    return crop.resize((out_size, out_size), Image.Resampling.NEAREST)


def draw_roi_box_on_assignment(
    img: Image.Image,
    geom: Dict[str, Any],
    roi_xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 220, 70),
    width: int = 8,
) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    ix0, iy0, ix1, iy1 = _roi_to_assignment_pixels(geom, roi_xyxy)
    for k in range(width):
        draw.rectangle([ix0 + k, iy0 + k, ix1 - k, iy1 - k], outline=color)
    return out


def save_pil_panel(out_path: Path, img: Image.Image, title: str, legend_type: str = "none", n_experts: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    if legend_type == "expert":
        import matplotlib.patches as mpatches
        colors = expert_colors_rgb_high_contrast(n_experts)
        handles = [mpatches.Patch(color=np.array(colors[e]) / 255.0, label=f"E{e}") for e in range(n_experts)]
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)
    elif legend_type == "purity":
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="purity")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_original_thumbnail(out_path: Path, thumb: Image.Image, slide_id: str, label: int) -> None:
    save_pil_panel(out_path, thumb, title=f"{slide_id} | original WSI | y={label}")


# =========================================================
# Galleries
# =========================================================
def make_gallery_grid(
    images: List[Image.Image],
    tile_size: int = 160,
    n_cols: int = 4,
    title: Optional[str] = None,
) -> Image.Image:
    if len(images) == 0:
        images = [Image.new("RGB", (tile_size, tile_size), color=(245, 245, 245))]

    n_rows = math.ceil(len(images) / n_cols)
    title_h = 34 if title else 0
    canvas = Image.new("RGB", (n_cols * tile_size, title_h + n_rows * tile_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((8, 8), title, fill=(0, 0, 0))

    for i, img in enumerate(images):
        r = i // n_cols
        c = i % n_cols
        x0 = c * tile_size
        y0 = title_h + r * tile_size
        img = ImageOps.fit(img.convert("RGB"), (tile_size, tile_size), method=Image.BICUBIC)
        canvas.paste(img, (x0, y0))

    return canvas


def save_expert_preferred_galleries(
    out_dir: Path,
    sub: pd.DataFrame,
    raw_dir: str,
    slide_id: str,
    patch_size: int,
    n_experts: int,
    topk_per_expert: int,
    purity_thr: float,
    tile_size: int,
    n_cols: int,
) -> Dict[int, Path]:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "single_patches")

    slide_path = find_wsi_path(raw_dir, slide_id)
    slide = openslide.OpenSlide(slide_path)

    saved: Dict[int, Path] = {}
    gallery_index_rows = []

    try:
        for e in range(n_experts):
            e_col = f"frac_E{e}"
            cand = sub[sub[e_col] > 0].copy()
            if len(cand) == 0:
                continue

            primary = cand[(cand["dominant_expert"] == e) & (cand["purity"] >= purity_thr)].copy()
            if len(primary) < topk_per_expert:
                fallback = cand.drop(index=primary.index, errors="ignore").copy()
                fallback = fallback.sort_values([e_col, "purity"], ascending=False)
                primary = pd.concat([primary, fallback], axis=0)

            primary = primary.sort_values([e_col, "purity"], ascending=False).head(topk_per_expert)

            imgs: List[Image.Image] = []
            for rank, (_, r) in enumerate(primary.iterrows(), start=1):
                img = read_patch_from_wsi(slide, (int(r["coord_x"]), int(r["coord_y"])), patch_size=patch_size, read_level=0)
                imgs.append(img)

                single_name = (
                    f"{safe_slide_name(slide_id)}_E{e}_rank{rank:02d}"
                    f"_score{float(r[e_col]):.3f}_purity{float(r['purity']):.3f}.png"
                )
                single_path = out_dir / "single_patches" / single_name
                img.save(single_path)

                row = r.to_dict()
                row.update({
                    "gallery_expert": e,
                    "gallery_rank": rank,
                    "gallery_score": float(r[e_col]),
                    "single_patch_path": str(single_path),
                })
                gallery_index_rows.append(row)

            if imgs:
                out_path = out_dir / f"{safe_slide_name(slide_id)}_E{e}_preferred_patches.png"
                gallery = make_gallery_grid(imgs, tile_size=tile_size, n_cols=n_cols, title=f"Expert {e} preferred patches")
                gallery.save(out_path)
                saved[e] = out_path

    finally:
        slide.close()

    if gallery_index_rows:
        pd.DataFrame(gallery_index_rows).to_csv(
            out_dir / f"{safe_slide_name(slide_id)}_expert_gallery_index.csv",
            index=False,
        )

    return saved


def save_all_experts_gallery_panel(
    out_path: Path,
    expert_gallery_paths: Dict[int, Path],
    n_experts: int,
    tile_width: int = 320,
) -> None:
    panels: List[Image.Image] = []
    for e in range(n_experts):
        if e not in expert_gallery_paths:
            img = Image.new("RGB", (tile_width, tile_width), color=(245, 245, 245))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Expert {e}\nNo selected patches", fill=(0, 0, 0))
        else:
            img = Image.open(expert_gallery_paths[e]).convert("RGB")
            ratio = tile_width / float(img.size[0])
            new_h = max(1, int(round(img.size[1] * ratio)))
            img = img.resize((tile_width, new_h), Image.BICUBIC)
        panels.append(img)

    max_h = max(img.size[1] for img in panels)
    canvas = Image.new("RGB", (tile_width * n_experts, max_h), color=(255, 255, 255))
    for e, img in enumerate(panels):
        canvas.paste(img, (e * tile_width, 0))

    canvas.save(out_path)


# =========================================================
# Combined figure
# =========================================================
def draw_panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )


def plot_gallery_as_image(ax, img_path: Optional[Path], title: str) -> None:
    if img_path is None or not img_path.exists():
        img = Image.new("RGB", (400, 400), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "No image", fill=(0, 0, 0))
    else:
        img = Image.open(img_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def save_combined_pamoe_figure(
    out_path: Path,
    original_thumb: Image.Image,
    assignment_with_roi: Image.Image,
    roi_img: Image.Image,
    expert_gallery_paths: Dict[int, Path],
    slide_id: str,
    label: int,
    n_experts: int,
) -> None:
    """
    Publication-style layout:
    Top row:
        Whole Slide Image | Expert Assignment Map | ROI with Expert Preference | legend at right
    Bottom row:
        Expert-preferred patch galleries

    The legend is placed in a dedicated right-side column to avoid overlap with
    the second row.
    """
    import matplotlib.patches as mpatches

    n_gallery_cols = n_experts

    # Add one extra narrow column on the right for expert legend.
    n_cols = max(4, n_gallery_cols + 1)
    fig = plt.figure(figsize=(4.0 * n_cols, 9.0))

    gs = fig.add_gridspec(
        2,
        n_cols,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.0] * (n_cols - 1) + [0.48],
        hspace=0.30,
        wspace=0.08,
    )

    # -------------------------
    # Top row
    # -------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(original_thumb)
    ax0.set_title("Whole Slide Image", fontsize=12)
    ax0.set_xticks([])
    ax0.set_yticks([])
    draw_panel_label(ax0, "A")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(assignment_with_roi)
    ax1.set_title("Expert Assignment Map", fontsize=12)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(roi_img)
    ax2.set_title("ROI with Expert Preference", fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Dedicated legend axis on the right.
    ax_leg = fig.add_subplot(gs[0, -1])
    ax_leg.axis("off")

    colors = expert_colors_rgb_high_contrast(n_experts)
    handles = [
        mpatches.Patch(
            color=np.array(colors[e]) / 255.0,
            label=f"Expert {e}",
        )
        for e in range(n_experts)
    ]

    ax_leg.legend(
        handles=handles,
        loc="center left",
        fontsize=10,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.5,
        handleheight=1.2,
        labelspacing=0.8,
    )

    # Hide unused top-row axes between ROI and legend if n_cols > 4.
    for k in range(3, n_cols - 1):
        ax = fig.add_subplot(gs[0, k])
        ax.axis("off")

    # -------------------------
    # Bottom row: galleries
    # -------------------------
    for e in range(n_experts):
        ax = fig.add_subplot(gs[1, e])
        plot_gallery_as_image(
            ax,
            expert_gallery_paths.get(e, None),
            title=f"Expert {e} preferred patches",
        )
        if e == 0:
            draw_panel_label(ax, "B")

    # Hide unused bottom axes.
    for k in range(n_experts, n_cols):
        ax = fig.add_subplot(gs[1, k])
        ax.axis("off")

    fig.suptitle(
        f"Patch-composition expert interpretation | {slide_id} | y={label}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# =========================================================
# Slide routine
# =========================================================
def process_one_slide(
    slide_id: str,
    label: int,
    model: MoEEncoder,
    transform,
    device: str,
    raw_dir: str,
    h5_dir: str,
    out_dir: Path,
    patch_size: int,
    batch_size: int,
    max_patches_per_slide: int,
    thumb_width: int,
    draw_grid: bool,
    gallery_topk_per_expert: int,
    gallery_purity_thr: float,
    gallery_tile_size: int,
    gallery_n_cols: int,
    roi_size_patches: int,
    roi_stride_patches: int,
    roi_min_patches: int,
    use_cache: bool,
    overwrite_cache: bool,
    seed: int,
) -> None:
    slide_safe = safe_slide_name(slide_id)
    slide_out = out_dir / slide_safe
    ensure_dir(slide_out)
    ensure_dir(slide_out / "cache")
    ensure_dir(slide_out / "per_slide_csv")
    ensure_dir(slide_out / "individual_panels")
    ensure_dir(slide_out / "expert_galleries")

    cache_path = slide_out / "cache" / f"{slide_safe}_patch_composition.npz"
    csv_path = slide_out / "per_slide_csv" / f"{slide_safe}_patch_composition.csv"

    if use_cache and cache_path.exists() and not overwrite_cache:
        print(f"[Cache] load {cache_path}")
        c = np.load(cache_path, allow_pickle=True)
        patch_df = pd.DataFrame({k: c[k] for k in c.files})
    else:
        print(f"[Extract] {slide_id}")
        patch_df = extract_patch_composition_for_slide(
            model=model,
            slide_path=find_wsi_path(raw_dir, slide_id),
            h5_path=find_h5_path(h5_dir, slide_id),
            transform=transform,
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
            max_patches_per_slide=max_patches_per_slide,
            seed=stable_slide_seed(seed, slide_id),
        )
        patch_df.insert(0, "slide_id", slide_id)
        patch_df.insert(1, "label", int(label))

        if use_cache:
            np.savez_compressed(cache_path, **{c: patch_df[c].values for c in patch_df.columns})
            print(f"[Cache] saved {cache_path}")

    patch_df.to_csv(csv_path, index=False)

    n_experts = get_num_experts(patch_df)
    slide_path = find_wsi_path(raw_dir, slide_id)
    thumb, _, _ = make_wsi_thumbnail(slide_path, thumb_width=thumb_width)

    original_path = slide_out / "individual_panels" / f"{slide_safe}_original_wsi.png"
    assignment_path = slide_out / "individual_panels" / f"{slide_safe}_expert_assignment_grid.png"
    assignment_roi_path = slide_out / "individual_panels" / f"{slide_safe}_expert_assignment_grid_with_roi.png"
    roi_path = slide_out / "individual_panels" / f"{slide_safe}_roi_expert_preference_grid.png"
    purity_path = slide_out / "individual_panels" / f"{slide_safe}_patch_composition_purity_grid.png"

    save_original_thumbnail(original_path, thumb, slide_id, label)

    assignment_img, assignment_geom = render_expert_assignment_grid(
        sub=patch_df,
        patch_size=patch_size,
        n_experts=n_experts,
        max_width=thumb_width,
        pad=20,
        draw_grid=draw_grid,
        grid_color=(35, 35, 35),
        grid_width=1,
        purity_alpha=False,
        discard_low_purity=False,
        purity_thr=gallery_purity_thr,
    )
    purity_img, _ = render_purity_assignment_grid(
        sub=patch_df,
        patch_size=patch_size,
        max_width=thumb_width,
        pad=20,
        draw_grid=draw_grid,
        grid_color=(35, 35, 35),
        grid_width=1,
    )

    roi_xyxy = select_interesting_roi(
        sub=patch_df,
        patch_size=patch_size,
        n_experts=n_experts,
        roi_size_patches=roi_size_patches,
        stride_patches=roi_stride_patches,
        min_patches_in_roi=roi_min_patches,
    )
    assignment_with_roi = draw_roi_box_on_assignment(
        assignment_img,
        assignment_geom,
        roi_xyxy,
        color=(255, 220, 70),
        width=8,
    )
    roi_img = crop_assignment_roi(
        assignment_img,
        assignment_geom,
        roi_xyxy,
        out_size=900,
    )

    assignment_img.save(assignment_path)
    assignment_with_roi.save(assignment_roi_path)
    roi_img.save(roi_path)
    purity_img.save(purity_path)

    save_pil_panel(
        slide_out / "individual_panels" / f"{slide_safe}_expert_assignment_grid_with_legend.png",
        assignment_with_roi,
        title=f"{slide_id} | expert assignment grid",
        legend_type="expert",
        n_experts=n_experts,
    )
    save_pil_panel(
        slide_out / "individual_panels" / f"{slide_safe}_patch_composition_purity_grid_with_colorbar.png",
        purity_img,
        title=f"{slide_id} | patch-composition purity grid",
        legend_type="purity",
        n_experts=n_experts,
    )

    expert_gallery_paths = save_expert_preferred_galleries(
        out_dir=slide_out / "expert_galleries",
        sub=patch_df,
        raw_dir=raw_dir,
        slide_id=slide_id,
        patch_size=patch_size,
        n_experts=n_experts,
        topk_per_expert=gallery_topk_per_expert,
        purity_thr=gallery_purity_thr,
        tile_size=gallery_tile_size,
        n_cols=gallery_n_cols,
    )

    save_all_experts_gallery_panel(
        out_path=slide_out / "individual_panels" / f"{slide_safe}_all_expert_galleries.png",
        expert_gallery_paths=expert_gallery_paths,
        n_experts=n_experts,
        tile_width=320,
    )

    save_combined_pamoe_figure(
        out_path=slide_out / f"{slide_safe}_pamoe_style_summary.png",
        original_thumb=thumb,
        assignment_with_roi=assignment_with_roi,
        roi_img=roi_img,
        expert_gallery_paths=expert_gallery_paths,
        slide_id=slide_id,
        label=label,
        n_experts=n_experts,
    )

    expert_counts = np.array([(patch_df["dominant_expert"].values == e).sum() for e in range(n_experts)])
    summary = {
        "slide_id": slide_id,
        "label": int(label),
        "n_patches": int(len(patch_df)),
        "n_experts": int(n_experts),
        "mean_purity": float(patch_df["purity"].mean()),
        "mean_composition_entropy": float(patch_df["composition_entropy"].mean()),
        "roi_xyxy_level0": [int(x) for x in roi_xyxy],
        "expert_counts": {f"E{e}": int(expert_counts[e]) for e in range(n_experts)},
        "expert_fraction": {f"E{e}": float(expert_counts[e] / max(1, expert_counts.sum())) for e in range(n_experts)},
    }
    with open(slide_out / "slide_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {slide_out / f'{slide_safe}_pamoe_style_summary.png'}")
    print(f"[Saved] single patches: {slide_out / 'expert_galleries' / 'single_patches'}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser("PAMOE-style full-patch MoE patch-composition visualization")

    parser.add_argument("--config", type=str, required=True, help="DINO-MoE stage2 config yaml.")
    parser.add_argument("--moe_ckpt", type=str, required=True, help="DINO-MoE checkpoint.")
    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--slide_ids", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--n_slides_per_label", type=int, default=3)
    parser.add_argument("--select_mode", type=str, default="random_balanced", choices=["csv_order", "random_balanced"])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_patches_per_slide",
        type=int,
        default=-1,
        help="-1 means full slide coords. Use 4096 for quick screening.",
    )

    parser.add_argument("--thumb_width", type=int, default=2600)
    parser.add_argument("--no_grid_edge", action="store_true")
    parser.add_argument("--roi_size_patches", type=int, default=30)
    parser.add_argument("--roi_stride_patches", type=int, default=8)
    parser.add_argument("--roi_min_patches", type=int, default=100)

    parser.add_argument("--gallery_topk_per_expert", type=int, default=12)
    parser.add_argument("--gallery_purity_thr", type=float, default=0.65)
    parser.add_argument("--gallery_tile_size", type=int, default=180)
    parser.add_argument("--gallery_n_cols", type=int, default=4)

    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "candidate_tables")

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    selected_df = prepare_slides_df(
        slides_csv=args.slides_csv,
        slide_ids=args.slide_ids,
        labels=args.labels,
        n_slides_per_label=args.n_slides_per_label,
        select_mode=args.select_mode,
        seed=args.seed,
    )
    selected_df.to_csv(out_dir / "candidate_tables" / "selected_visualization_slides.csv", index=False)

    print(f"[Select] {len(selected_df)} slides")
    if len(selected_df) > 0:
        print(selected_df[["slide_id", "label"]])

    model, _ = load_dino_moe_model(
        config_path=args.config,
        ckpt_path=args.moe_ckpt,
        device=device,
        strict=args.strict_load,
    )
    transform = build_transform(args.image_size)

    for _, row in selected_df.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        print(f"\n[Process] {slide_id} | y={label}")
        try:
            process_one_slide(
                slide_id=slide_id,
                label=label,
                model=model,
                transform=transform,
                device=device,
                raw_dir=args.raw_dir,
                h5_dir=args.h5_dir,
                out_dir=out_dir,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                max_patches_per_slide=args.max_patches_per_slide,
                thumb_width=args.thumb_width,
                draw_grid=not args.no_grid_edge,
                gallery_topk_per_expert=args.gallery_topk_per_expert,
                gallery_purity_thr=args.gallery_purity_thr,
                gallery_tile_size=args.gallery_tile_size,
                gallery_n_cols=args.gallery_n_cols,
                roi_size_patches=args.roi_size_patches,
                roi_stride_patches=args.roi_stride_patches,
                roi_min_patches=args.roi_min_patches,
                use_cache=not args.no_cache,
                overwrite_cache=args.overwrite_cache,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[WARN] failed slide={slide_id}: {e}")

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"\n[Done] saved PAMOE-style visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
